"""Train and evaluate the frozen oracle-C count-constrained assignment Q1."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.assignment_scorer import (
    OrbitAwareAssignmentScorer,
    faithful_parent_action,
    parent_action_site_features,
    parent_carrier_graph_features,
)
from gaugeflow.production.composition_assignment import (
    CountConstrainedAssignmentLaw,
    composition_counts_from_tokens,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


@dataclass(frozen=True)
class AssignmentExample:
    """One oracle-composition carrier; identifiers remain audit metadata."""

    material_id: str
    split: str
    site_features: torch.Tensor
    graph_features: torch.Tensor
    counts: torch.Tensor
    assignment: torch.Tensor
    parent_permutations: torch.Tensor
    parent_space_group: int
    cell_index: int
    uniform_quotient_log_probability: float
    target_quotient_size: int
    action_order: int


@dataclass(frozen=True)
class PackedAssignmentBatch:
    site_features: torch.Tensor
    graph_features: torch.Tensor
    batch: torch.Tensor
    counts: torch.Tensor
    assignment: torch.Tensor
    parent_permutations: tuple[torch.Tensor, ...]
    parent_space_group: torch.Tensor
    cell_index: torch.Tensor


def _normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Q1 requires a clean committed implementation tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_protocol(protocol: dict[str, Any], repository: Path, o1_root: Path) -> None:
    if (
        protocol.get("protocol") != "h1a_oracle_c_assignment_q1_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen Q1 protocol")
    prerequisite_paths = {
        "absolute_likelihood_e1_result_sha256": (
            repository / "reports/h1a_e1_absolute_likelihood_v1/result.json"
        ),
        "assignment_carrier_audit_result_sha256": (
            repository / "reports/h1a_assignment_carrier_audit_v1/result.json"
        ),
    }
    for name, path in prerequisite_paths.items():
        if sha256_file(path) != protocol["prerequisites"][name]:
            raise ValueError(f"Q1 prerequisite identity changed: {path}")
        result = load_json_object(path)
        if result.get("qualified") is not True or not all(result["checks"].values()):
            raise ValueError(f"Q1 prerequisite is not qualified: {path}")
    source = protocol["source"]
    artifact_paths = {
        "manifest.json": o1_root / "manifest.json",
        "results.json.gz": o1_root / "results.json.gz",
        "independent_audit.json": o1_root / "independent_audit.json",
    }
    for name, path in artifact_paths.items():
        if sha256_file(path) != source["artifact_sha256"][name]:
            raise ValueError(f"Q1 source identity changed: {path}")
    for relative, expected in source["normalized_implementation_sha256"].items():
        if _normalized_source_sha256(repository / relative) != expected:
            raise ValueError(f"Q1 implementation identity changed: {relative}")
    training = protocol["training"]
    if (
        training.get("precision") != "fp32"
        or int(training.get("seed_count", 0)) != 1
        or training.get("continuous_hyperparameter_search") is not False
        or protocol["model"].get("from_scratch_required") is not True
    ):
        raise ValueError("Q1 training contract changed")


def _prepare_example(
    record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_sites: int,
    radial_channels: int,
) -> AssignmentExample:
    tokens = torch.tensor(candidate["child_atomic_numbers"], dtype=torch.long) - 1
    if bool(((tokens < 0) | (tokens >= CHEMICAL_ELEMENT_COUNT)).any()):
        raise ValueError("Q1 target contains a nonphysical element")
    nodes = tokens.numel()
    if nodes > maximum_sites or int(candidate["child_site_count"]) != nodes:
        raise ValueError("Q1 carrier violates the frozen site bound")
    if int(candidate["parent_site_count"]) * int(candidate["cell_index"]) != nodes:
        raise ValueError("parent carrier expansion does not close on the target site count")
    permutations = faithful_parent_action(
        torch.tensor(candidate["parent_action_permutations"], dtype=torch.long)
    )
    if permutations.shape[1] != nodes:
        raise ValueError("parent action does not cover the assignment sites")
    counts = torch.bincount(tokens, minlength=CHEMICAL_ELEMENT_COUNT)
    active_counts = counts[counts > 0]
    assignment_count = math.factorial(nodes)
    for count in active_counts.tolist():
        assignment_count //= math.factorial(int(count))
    target_orbit = torch.unique(tokens[permutations], dim=0)
    uniform_log_probability = math.log(target_orbit.shape[0]) - math.log(assignment_count)
    parent_fractional = torch.tensor(candidate["parent_fractional"], dtype=torch.float32)
    parent_lattice = torch.tensor(candidate["parent_lattice"], dtype=torch.float32)
    return AssignmentExample(
        material_id=str(record["material_id"]),
        split=str(record["gaugeflow_split"]),
        site_features=parent_action_site_features(
            permutations,
            maximum_sites=maximum_sites,
        ),
        graph_features=parent_carrier_graph_features(
            parent_fractional,
            parent_lattice,
            permutations,
            cell_index=int(candidate["cell_index"]),
            maximum_sites=maximum_sites,
            radial_channels=radial_channels,
        ),
        counts=counts,
        assignment=tokens,
        parent_permutations=permutations,
        parent_space_group=int(candidate["parent_space_group"]),
        cell_index=int(candidate["cell_index"]),
        uniform_quotient_log_probability=uniform_log_probability,
        target_quotient_size=int(target_orbit.shape[0]),
        action_order=int(permutations.shape[0]),
    )


def _load_examples(
    o1_root: Path,
    *,
    maximum_sites: int,
    radial_channels: int,
) -> list[AssignmentExample]:
    manifest = load_json_object(o1_root / "manifest.json")
    independent = load_json_object(o1_root / "independent_audit.json")
    if manifest.get("qualified") is not True or not all(manifest["checks"].values()):
        raise ValueError("Q1 source manifest is not qualified")
    if (
        independent.get("audit_passed") is not True
        or independent.get("gate_qualified") is not True
        or not all(independent["checks"].values())
    ):
        raise ValueError("Q1 source independent audit is not qualified")
    with gzip.open(o1_root / "results.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    examples = [
        _prepare_example(
            record,
            candidate,
            maximum_sites=maximum_sites,
            radial_channels=radial_channels,
        )
        for record in records
        for candidate in record["candidates"]
    ]
    if not examples:
        raise ValueError("Q1 source contains no carrier candidates")
    return examples


def _pack(examples: Sequence[AssignmentExample], device: torch.device) -> PackedAssignmentBatch:
    node_counts = torch.tensor([value.assignment.numel() for value in examples], dtype=torch.long)
    return PackedAssignmentBatch(
        site_features=torch.cat([value.site_features for value in examples]).to(device),
        graph_features=torch.stack([value.graph_features for value in examples]).to(device),
        batch=torch.repeat_interleave(torch.arange(len(examples)), node_counts).to(device),
        counts=torch.stack([value.counts for value in examples]).to(device),
        assignment=torch.cat([value.assignment for value in examples]).to(device),
        parent_permutations=tuple(value.parent_permutations.to(device) for value in examples),
        parent_space_group=torch.tensor(
            [value.parent_space_group for value in examples], dtype=torch.long, device=device
        ),
        cell_index=torch.tensor(
            [value.cell_index for value in examples], dtype=torch.long, device=device
        ),
    )


def _scores(model: OrbitAwareAssignmentScorer, packed: PackedAssignmentBatch) -> torch.Tensor:
    return model(
        packed.site_features,
        packed.graph_features,
        packed.batch,
        packed.counts,
        packed.parent_space_group,
        packed.cell_index,
    )


class MaterialBalancedSampler:
    """Uniformly sample materials, then one certified carrier per material."""

    def __init__(self, examples: Sequence[AssignmentExample], *, seed: int) -> None:
        grouped: dict[str, list[AssignmentExample]] = defaultdict(list)
        for example in examples:
            grouped[example.material_id].append(example)
        self.grouped = dict(grouped)
        self.materials = sorted(grouped)
        self.random = random.Random(seed)

    def sample(self, batch_size: int) -> list[AssignmentExample]:
        if batch_size > len(self.materials):
            raise ValueError("material batch exceeds the number of training materials")
        materials = self.random.sample(self.materials, batch_size)
        return [self.random.choice(self.grouped[material]) for material in materials]


def _mean_nll(
    model: OrbitAwareAssignmentScorer,
    law: CountConstrainedAssignmentLaw,
    examples: Sequence[AssignmentExample],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            packed = _pack(examples[start : start + batch_size], device)
            result = law.quotient_log_prob(
                _scores(model, packed),
                packed.batch,
                packed.counts,
                packed.assignment,
                packed.parent_permutations,
            )
            values.append(-result.log_probability.detach().cpu())
    return float(torch.cat(values).mean())


def _target_orbit(example: AssignmentExample, device: torch.device) -> torch.Tensor:
    assignment = example.assignment.to(device)
    return torch.unique(assignment[example.parent_permutations.to(device)], dim=0)


def _in_orbit(tokens: torch.Tensor, orbit: torch.Tensor) -> bool:
    return bool(torch.all(tokens.unsqueeze(0) == orbit, dim=1).any())


def _orbit_site_accuracy(tokens: torch.Tensor, orbit: torch.Tensor) -> float:
    return float((tokens.unsqueeze(0) == orbit).float().mean(dim=1).max())


def _orbit_constant_family_ceiling(example: AssignmentExample) -> float:
    """Maximum target-orbit mass of the scorer's symmetry-respecting unary family.

    Parent-related sites necessarily have equal unary scores.  Once the model
    concentrates on the correct species counts in every parent site orbit,
    assignments within each such orbit remain equiprobable.  This target-only
    evaluation quantity records that honest expressivity ceiling; it is never
    exposed to the scorer.
    """
    permutations = example.parent_permutations
    assignment = example.assignment
    unseen = set(range(assignment.numel()))
    compatible_labelings = 1
    while unseen:
        seed = min(unseen)
        orbit = sorted(set(map(int, permutations[:, seed].tolist())))
        unseen.difference_update(orbit)
        orbit_tokens = assignment[orbit]
        denominator = math.factorial(len(orbit))
        for count in torch.bincount(orbit_tokens).tolist():
            denominator //= math.factorial(int(count))
        compatible_labelings *= denominator
    return example.target_quotient_size / compatible_labelings


def _relabel_action(action: torch.Tensor, relabel: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(relabel)
    inverse[relabel] = torch.arange(relabel.numel(), device=relabel.device)
    return inverse[action[:, relabel]]


def _evaluate_example(
    model: OrbitAwareAssignmentScorer,
    law: CountConstrainedAssignmentLaw,
    example: AssignmentExample,
    *,
    index: int,
    sample_draws: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    packed = _pack([example], device)
    with torch.no_grad():
        score = _scores(model, packed)
        target = law.quotient_log_prob(
            score,
            packed.batch,
            packed.counts,
            packed.assignment,
            packed.parent_permutations,
        )
        entropy = law.entropy(score, packed.batch, packed.counts)[0]
        mode = law.sample(score, packed.batch, packed.counts, mode=True)
    orbit = _target_orbit(example, device)
    sample_generator = torch.Generator(device=device).manual_seed(seed + 10_000 + index)
    retrieved = 0
    sample_composition_exact = 0
    sample_site_accuracy = 0.0
    for _ in range(sample_draws):
        sampled = law.sample(
            score,
            packed.batch,
            packed.counts,
            generator=sample_generator,
        ).tokens
        retrieved += int(_in_orbit(sampled, orbit))
        sample_site_accuracy += _orbit_site_accuracy(sampled, orbit)
        observed = composition_counts_from_tokens(sampled, packed.batch, 1)
        sample_composition_exact += int(torch.equal(observed, packed.counts))

    relabel_generator = torch.Generator().manual_seed(seed + 100_000 + index)
    relabel_cpu = torch.randperm(example.assignment.numel(), generator=relabel_generator)
    relabel = relabel_cpu.to(device)
    relabeled_action_cpu = _relabel_action(example.parent_permutations, relabel_cpu)
    relabeled = PackedAssignmentBatch(
        site_features=parent_action_site_features(
            relabeled_action_cpu,
            maximum_sites=model.maximum_sites,
        ).to(device),
        graph_features=packed.graph_features,
        batch=packed.batch,
        counts=packed.counts,
        assignment=packed.assignment[relabel],
        parent_permutations=(relabeled_action_cpu.to(device),),
        parent_space_group=packed.parent_space_group,
        cell_index=packed.cell_index,
    )
    with torch.no_grad():
        relabeled_score = _scores(model, relabeled)
        relabeled_target = law.quotient_log_prob(
            relabeled_score,
            relabeled.batch,
            relabeled.counts,
            relabeled.assignment,
            relabeled.parent_permutations,
        )
        probability_probes = [packed.assignment, mode.tokens]
        probe_error = 0.0
        for probe in probability_probes:
            reference_logp = law.log_prob(
                score,
                packed.batch,
                packed.counts,
                probe,
            ).log_probability
            transformed_logp = law.log_prob(
                relabeled_score,
                relabeled.batch,
                relabeled.counts,
                probe[relabel],
            ).log_probability
            probe_error = max(
                probe_error,
                float(torch.max(torch.abs(reference_logp - transformed_logp))),
            )
    return {
        "material_id": example.material_id,
        "nodes": int(example.assignment.numel()),
        "species": int((example.counts > 0).sum()),
        "action_order": example.action_order,
        "target_quotient_size": example.target_quotient_size,
        "quotient_nll": float(-target.log_probability[0]),
        "uniform_quotient_nll": -example.uniform_quotient_log_probability,
        "target_quotient_probability": float(target.log_probability[0].exp()),
        "assignment_entropy": float(entropy),
        "model_family_target_probability_ceiling": _orbit_constant_family_ceiling(example),
        "labeling_map_target_orbit_retrieval_diagnostic": float(
            _in_orbit(mode.tokens, orbit)
        ),
        "sample_retrieval": retrieved / sample_draws,
        "sample_orbit_aligned_site_accuracy": sample_site_accuracy / sample_draws,
        "labeling_map_orbit_aligned_site_accuracy_diagnostic": _orbit_site_accuracy(
            mode.tokens, orbit
        ),
        "fixed_cif_site_accuracy": float((mode.tokens == packed.assignment).float().mean()),
        "exact_composition": float(
            torch.equal(
                composition_counts_from_tokens(mode.tokens, packed.batch, 1),
                packed.counts,
            )
        ),
        "sample_exact_composition": sample_composition_exact / sample_draws,
        "parent_action_score_max_abs": float(
            torch.max(torch.abs(score[example.parent_permutations.to(device)] - score))
        ),
        "relabel_score_max_abs": float(torch.max(torch.abs(relabeled_score - score[relabel]))),
        "relabel_quotient_logp_abs": float(
            torch.max(torch.abs(relabeled_target.log_probability - target.log_probability))
        ),
        "relabel_assignment_logp_abs": probe_error,
    }


def _material_mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["material_id"])].append(float(row[key]))
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def _bootstrap_ucb95(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["material_id"])].append(
            float(row["quotient_nll"]) - float(row["uniform_quotient_nll"])
        )
    material_difference = torch.tensor(
        [sum(values) / len(values) for _, values in sorted(grouped.items())],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        material_difference.numel(),
        (resamples, material_difference.numel()),
        generator=generator,
    )
    bootstrap = material_difference[indices].mean(dim=1)
    return float(torch.quantile(bootstrap, 0.95))


def _action_bin(order: int) -> str:
    if order <= 4:
        return "le4"
    if order <= 16:
        return "5_16"
    if order <= 64:
        return "17_64"
    return "65_plus"


def _stratified(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "nodes": lambda row: str(row["nodes"]),
        "species": lambda row: str(row["species"]),
        "action_order": lambda row: _action_bin(int(row["action_order"])),
        "target_quotient_size": lambda row: str(row["target_quotient_size"]),
    }
    output: dict[str, Any] = {}
    for name, key_function in definitions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_function(row)].append(row)
        output[name] = {
            key: {
                "carriers": len(values),
                "materials": len({str(value["material_id"]) for value in values}),
                "quotient_nll": _material_mean(values, "quotient_nll"),
                "uniform_quotient_nll": _material_mean(values, "uniform_quotient_nll"),
                "target_quotient_probability": _material_mean(
                    values, "target_quotient_probability"
                ),
                "sample_retrieval": _material_mean(values, "sample_retrieval"),
                "sample_orbit_aligned_site_accuracy": _material_mean(
                    values, "sample_orbit_aligned_site_accuracy"
                ),
            }
            for key, values in sorted(grouped.items())
        }
    return output


def _evaluate_split(
    model: OrbitAwareAssignmentScorer,
    law: CountConstrainedAssignmentLaw,
    examples: Sequence[AssignmentExample],
    *,
    sample_draws: int,
    bootstrap_resamples: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, example in enumerate(examples):
        try:
            rows.append(
                _evaluate_example(
                    model,
                    law,
                    example,
                    index=index,
                    sample_draws=sample_draws,
                    seed=seed,
                    device=device,
                )
            )
        except (RuntimeError, ValueError) as error:
            failures.append(f"{example.material_id}:{type(error).__name__}:{error}")
    if not rows:
        raise RuntimeError("Q1 evaluation produced no valid carrier rows")
    summary = {
        "carriers": len(examples),
        "materials": len({value.material_id for value in examples}),
        "evaluation_failures": len(failures),
        "failure_messages": failures,
        "quotient_nll": _material_mean(rows, "quotient_nll"),
        "uniform_quotient_nll": _material_mean(rows, "uniform_quotient_nll"),
        "model_minus_uniform_nll": _material_mean(rows, "quotient_nll")
        - _material_mean(rows, "uniform_quotient_nll"),
        "model_minus_uniform_nll_ucb95": _bootstrap_ucb95(
            rows,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
        "target_quotient_probability": _material_mean(rows, "target_quotient_probability"),
        "model_family_target_probability_ceiling": _material_mean(
            rows, "model_family_target_probability_ceiling"
        ),
        "model_family_ceiling_fraction": _material_mean(
            [
                {
                    **row,
                    "ceiling_fraction": float(row["target_quotient_probability"])
                    / float(row["model_family_target_probability_ceiling"]),
                }
                for row in rows
            ],
            "ceiling_fraction",
        ),
        "assignment_entropy": _material_mean(rows, "assignment_entropy"),
        "labeling_map_target_orbit_retrieval_diagnostic": _material_mean(
            rows, "labeling_map_target_orbit_retrieval_diagnostic"
        ),
        "sample_retrieval": _material_mean(rows, "sample_retrieval"),
        "sample_orbit_aligned_site_accuracy": _material_mean(
            rows, "sample_orbit_aligned_site_accuracy"
        ),
        "labeling_map_orbit_aligned_site_accuracy_diagnostic": _material_mean(
            rows, "labeling_map_orbit_aligned_site_accuracy_diagnostic"
        ),
        "fixed_cif_site_accuracy_diagnostic": _material_mean(
            rows, "fixed_cif_site_accuracy"
        ),
        "exact_composition": min(float(row["exact_composition"]) for row in rows),
        "sample_exact_composition": min(
            float(row["sample_exact_composition"]) for row in rows
        ),
        "relabel_score_max_abs": max(float(row["relabel_score_max_abs"]) for row in rows),
        "parent_action_score_max_abs": max(
            float(row["parent_action_score_max_abs"]) for row in rows
        ),
        "relabel_quotient_logp_abs": max(
            float(row["relabel_quotient_logp_abs"]) for row in rows
        ),
        "relabel_assignment_logp_abs": max(
            float(row["relabel_assignment_logp_abs"]) for row in rows
        ),
        "strata": _stratified(rows),
    }
    return summary, rows


def _write_history(path: Path, history: Iterable[dict[str, float]]) -> None:
    rows = list(history)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    validation = result["metrics"]["validation"]
    test = result["metrics"]["test"]
    metric_rows = [
        ("quotient NLL", "quotient_nll"),
        ("uniform quotient NLL", "uniform_quotient_nll"),
        ("model-uniform UCB95", "model_minus_uniform_nll_ucb95"),
        ("exact target quotient probability", "target_quotient_probability"),
        ("unary-family probability ceiling", "model_family_target_probability_ceiling"),
        ("categorical sample retrieval", "sample_retrieval"),
        ("sampled orbit-aligned site accuracy", "sample_orbit_aligned_site_accuracy"),
        (
            "labeling-MAP target retrieval (diagnostic)",
            "labeling_map_target_orbit_retrieval_diagnostic",
        ),
        ("fixed-CIF site accuracy (diagnostic)", "fixed_cif_site_accuracy_diagnostic"),
        ("exact composition", "exact_composition"),
    ]
    table = "\n".join(
        f"| {label} | {validation[key]:.6f} | {test[key]:.6f} |"
        for label, key in metric_rows
    )
    text = f"""# H1a oracle-C assignment Q1 v1

Status: **{'qualified' if result['qualified'] else 'failed'}**.

This Gate evaluates only exact count-constrained site assignment conditioned on
oracle-labelled composition and a species-free parent carrier. It does not
qualify generated-C exposure, `p(N)`, lattice L1, joint M1, tensor conditioning,
relaxation, DFT, or DFPT.

| metric | validation | test |
|---|---:|---:|
{table}
| failures | {validation['evaluation_failures']} | {test['evaluation_failures']} |

Decision: `{result['decision']}`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol, repository, args.o1_root)
    implementation_commit = _git_identity(repository)

    seed = int(protocol["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("formal Q1 requires CUDA")
    device = torch.device("cuda", int(protocol["training"]["cuda_device"]))
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    model_config = protocol["model"]
    examples = _load_examples(
        args.o1_root,
        maximum_sites=int(model_config["maximum_sites"]),
        radial_channels=int(model_config["radial_channels"]),
    )
    by_split = {
        split: [value for value in examples if value.split == split]
        for split in ("train", "val", "test")
    }
    expected_split = protocol["source"]["candidate_carriers_by_split"]
    if {key: len(value) for key, value in by_split.items()} != expected_split:
        raise ValueError("Q1 source split counts changed")

    model = OrbitAwareAssignmentScorer(
        maximum_sites=int(model_config["maximum_sites"]),
        maximum_cell_index=int(model_config["maximum_cell_index"]),
        hidden_dim=int(model_config["hidden_dim"]),
        radial_channels=int(model_config["radial_channels"]),
    ).to(device)
    law = CountConstrainedAssignmentLaw(
        maximum_active_species=int(model_config["maximum_active_species"]),
        maximum_states=int(model_config["maximum_dynamic_program_states"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    sampler = MaterialBalancedSampler(by_split["train"], seed=seed)
    steps = int(protocol["training"]["steps"])
    batch_size = int(protocol["training"]["material_batch_size"])
    evaluation_interval = int(protocol["training"]["evaluation_interval"])
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        model.train()
        packed = _pack(sampler.sample(batch_size), device)
        score = _scores(model, packed)
        result = law.quotient_log_prob(
            score,
            packed.batch,
            packed.counts,
            packed.assignment,
            packed.parent_permutations,
        )
        loss = -result.log_probability.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(protocol["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        if step == 1 or step % evaluation_interval == 0 or step == steps:
            validation_nll = _mean_nll(
                model,
                law,
                by_split["val"],
                batch_size=int(protocol["evaluation"]["batch_size"]),
                device=device,
            )
            history.append(
                {
                    "step": float(step),
                    "train_quotient_nll": float(loss.detach()),
                    "validation_quotient_nll": validation_nll,
                    "gradient_norm": float(gradient_norm),
                }
            )

    evaluation = protocol["evaluation"]
    split_metrics: dict[str, Any] = {}
    carrier_rows: dict[str, Any] = {}
    for offset, split in enumerate(("validation", "test")):
        source_split = "val" if split == "validation" else "test"
        summary, rows = _evaluate_split(
            model,
            law,
            by_split[source_split],
            sample_draws=int(evaluation["sample_draws_per_carrier"]),
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            seed=seed + offset,
            device=device,
        )
        split_metrics[split] = summary
        carrier_rows[split] = rows

    acceptance = protocol["acceptance"]
    checks: dict[str, bool] = {}
    for split in ("validation", "test"):
        metrics = split_metrics[split]
        checks[f"{split}_nll_noninferiority"] = (
            metrics["model_minus_uniform_nll_ucb95"]
            <= float(acceptance["model_minus_uniform_nll_ucb95_max"])
        )
        checks[f"{split}_target_quotient_probability"] = metrics[
            "target_quotient_probability"
        ] >= float(acceptance["target_quotient_probability_min"])
        checks[f"{split}_sample_retrieval"] = metrics["sample_retrieval"] >= float(
            acceptance["sample_retrieval_min"]
        )
        checks[f"{split}_sample_orbit_site_accuracy"] = metrics[
            "sample_orbit_aligned_site_accuracy"
        ] >= float(acceptance["sample_orbit_aligned_site_accuracy_min"])
        checks[f"{split}_exact_composition"] = (
            metrics["exact_composition"] == float(acceptance["exact_composition"])
            and metrics["sample_exact_composition"]
            == float(acceptance["sample_exact_composition"])
        )
        checks[f"{split}_zero_failures"] = metrics["evaluation_failures"] == int(
            acceptance["evaluation_failures"]
        )
        checks[f"{split}_relabel_consistency"] = max(
            metrics["parent_action_score_max_abs"],
            metrics["relabel_score_max_abs"],
            metrics["relabel_quotient_logp_abs"],
            metrics["relabel_assignment_logp_abs"],
        ) <= float(acceptance["relabel_probability_max_abs"])
    qualified = all(checks.values())
    elapsed = time.perf_counter() - started

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "protocol_sha256": canonical_json_hash(protocol),
            "implementation_commit": implementation_commit,
            "seed": seed,
        },
        args.checkpoint,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    result_payload = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "implementation_commit": implementation_commit,
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "metrics": split_metrics,
        "training": {
            "seed": seed,
            "steps": steps,
            "material_batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "final_history": history[-1],
        },
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
        },
        "leakage_contract": protocol["leakage_contract"],
        "carrier_rows": carrier_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_history(args.output_dir / "training_history.csv", history)
    _write_readme(args.output_dir / "README.md", result_payload)
    print(json.dumps(result_payload, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
