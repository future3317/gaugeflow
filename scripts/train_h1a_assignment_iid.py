"""Train and evaluate the frozen IID parent-conditioned assignment Gate."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.assignment_data import (
    AssignmentCarrierExample,
    pack_assignment_carriers,
    prepare_assignment_carrier_example,
)
from gaugeflow.production.assignment_training import orderless_assignment_objective
from gaugeflow.production.autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
)


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
        raise ValueError("assignment IID Gate requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_examples(
    carrier_root: Path,
    role_result_path: Path,
    *,
    maximum_sites: int,
    radial_channels: int,
) -> list[AssignmentCarrierExample]:
    role_result = load_json_object(role_result_path)
    if role_result.get("qualified") is not True or not all(role_result["checks"].values()):
        raise ValueError("assignment IID role split is not qualified")
    roles = {
        (str(row["material_id"]), int(row["candidate_index"]), str(row["embedding_key"])): str(
            row["role"]
        )
        for row in role_result["carrier_rows"]
    }
    with gzip.open(carrier_root / "records.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    examples: list[AssignmentCarrierExample] = []
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        material_id = str(record["material_id_audit_only"])
        for candidate_index, candidate in enumerate(record["candidates"]):
            embedding_key = str(candidate["embedding_key"])
            key = (material_id, candidate_index, embedding_key)
            if key not in roles or key in seen:
                raise ValueError(f"carrier role identity is missing or duplicated: {key}")
            seen.add(key)
            examples.append(
                prepare_assignment_carrier_example(
                    candidate,
                    embedding_key=embedding_key,
                    material_id_audit_only=material_id,
                    evidence_role_audit_only=roles[key],
                    maximum_sites=maximum_sites,
                    radial_channels=radial_channels,
                )
            )
    if seen != set(roles):
        raise ValueError("geometry-complete carriers and frozen IID roles differ")
    return examples


def _orbit_size(example: AssignmentCarrierExample) -> int:
    return int(
        torch.unique(example.target_assignment[example.parent_permutations], dim=0).shape[0]
    )


def _uniform_quotient_nll(example: AssignmentCarrierExample) -> float:
    counts = example.composition_counts[example.composition_counts > 0]
    assignments = math.factorial(example.target_assignment.numel())
    for count in counts.tolist():
        assignments //= math.factorial(int(count))
    return math.log(assignments / _orbit_size(example))


def _chunks(values: Sequence[AssignmentCarrierExample], size: int) -> list[list[AssignmentCarrierExample]]:
    if size < 1:
        raise ValueError("evaluation carrier batch size must be positive")
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


@torch.no_grad()
def _bound_rows(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    *,
    order_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for group in _chunks(examples, batch_size):
        packed = pack_assignment_carriers(group, device=device)
        values = torch.stack(
            [
                orderless_assignment_objective(model, packed, generator=generator)
                .graph_log_probability.detach()
                .to(torch.float64)
                for _ in range(order_samples)
            ]
        ).mean(dim=0)
        for example, path_log_probability in zip(group, values.tolist()):
            nll = -(path_log_probability + math.log(_orbit_size(example)))
            uniform = _uniform_quotient_nll(example)
            rows.append(
                {
                    "material_id": example.material_id_audit_only,
                    "embedding_key": example.embedding_key,
                    "role": example.evidence_role_audit_only,
                    "sites": int(example.target_assignment.numel()),
                    "species": int((example.composition_counts > 0).sum()),
                    "orbit_size": _orbit_size(example),
                    "quotient_lower_bound_nll": nll,
                    "uniform_quotient_nll": uniform,
                    "model_minus_uniform_nll": nll - uniform,
                    "quotient_probability_lower_bound": math.exp(-nll),
                    "uniform_quotient_probability": math.exp(-uniform),
                }
            )
    return rows


def _material_bootstrap_ucb95(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["material_id"])].append(float(row["model_minus_uniform_nll"]))
    values = torch.tensor(
        [sum(group) / len(group) for group in grouped.values()],
        dtype=torch.float64,
    )
    if values.numel() < 2 or resamples < 100:
        raise ValueError("material bootstrap requires at least two materials and 100 resamples")
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(values.numel(), (resamples, values.numel()), generator=generator)
    return float(torch.quantile(values[index].mean(dim=1), 0.95))


def _bound_summary(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    model = torch.tensor([float(row["quotient_lower_bound_nll"]) for row in rows])
    uniform = torch.tensor([float(row["uniform_quotient_nll"]) for row in rows])
    return {
        "carriers": len(rows),
        "materials": len({str(row["material_id"]) for row in rows}),
        "mean_quotient_lower_bound_nll": float(model.mean()),
        "mean_uniform_quotient_nll": float(uniform.mean()),
        "relative_nll_reduction_from_uniform": float(
            (uniform.mean() - model.mean()) / uniform.mean().clamp_min(1e-12)
        ),
        "model_minus_uniform_nll_ucb95": _material_bootstrap_ucb95(
            rows,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
        "mean_target_probability_lower_bound": sum(
            float(row["quotient_probability_lower_bound"]) for row in rows
        )
        / len(rows),
        "mean_uniform_target_probability": sum(
            float(row["uniform_quotient_probability"]) for row in rows
        )
        / len(rows),
    }


@torch.no_grad()
def _sample_example(
    model: GeometryAwareRemainingCountScorer,
    example: AssignmentCarrierExample,
    *,
    draws: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    packed = pack_assignment_carriers([example] * draws, device=device)
    nodes = example.target_assignment.numel()
    generator = torch.Generator(device=device).manual_seed(seed)
    reveal_order = torch.argsort(torch.rand(draws, nodes, generator=generator, device=device), dim=1)
    partial = torch.full((draws, nodes), -1, dtype=torch.long, device=device)
    remaining = packed.composition_counts.clone()
    law = RemainingCountAssignmentLaw()
    row = torch.arange(draws, device=device)
    for depth in range(nodes):
        logits = model(
            packed.site_features,
            packed.graph_features,
            packed.batch,
            packed.edge_source,
            packed.edge_target,
            packed.edge_rbf,
            partial.reshape(-1),
            packed.composition_counts,
            remaining,
            packed.parent_space_group,
            packed.cell_index,
        )
        site = reveal_order[:, depth]
        log_probability = law.batched_step_log_probabilities(
            logits[row * nodes + site],
            remaining,
        )
        token = torch.multinomial(log_probability.exp(), 1, generator=generator).squeeze(1)
        partial[row, site] = token
        remaining[row, token] -= 1
    if bool((remaining != 0).any()) or bool((partial < 0).any()):
        raise RuntimeError("assignment sampler failed exact-count closure")
    return partial


@torch.no_grad()
def _sampling_summary(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    *,
    draws: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    retrieval: list[torch.Tensor] = []
    accuracy: list[torch.Tensor] = []
    exact: list[torch.Tensor] = []
    failures = 0
    for index, example in enumerate(examples):
        try:
            samples = _sample_example(
                model,
                example,
                draws=draws,
                seed=seed + index,
                device=device,
            )
        except (RuntimeError, ValueError):
            failures += 1
            continue
        orbit = torch.unique(
            example.target_assignment.to(device)[example.parent_permutations.to(device)],
            dim=0,
        )
        equality = torch.all(samples[:, None, :] == orbit[None, :, :], dim=2)
        retrieval.append(equality.any(dim=1).to(torch.float32).mean())
        aligned = (samples[:, None, :] == orbit[None, :, :]).to(torch.float32).mean(dim=2)
        accuracy.append(aligned.max(dim=1).values.mean())
        observed = torch.stack(
            [torch.bincount(value, minlength=example.composition_counts.numel()) for value in samples]
        )
        exact.append(
            torch.all(observed == example.composition_counts.to(device), dim=1)
            .to(torch.float32)
            .mean()
        )
    if not retrieval:
        return {
            "sample_retrieval": float("nan"),
            "sample_orbit_aligned_site_accuracy": float("nan"),
            "sample_exact_composition": 0.0,
            "sampling_failures": failures,
        }
    uniform_retrieval = sum(math.exp(-_uniform_quotient_nll(value)) for value in examples) / len(
        examples
    )
    return {
        "sample_retrieval": float(torch.stack(retrieval).mean()),
        "uniform_expected_retrieval": uniform_retrieval,
        "sample_retrieval_lift_over_uniform": float(torch.stack(retrieval).mean())
        - uniform_retrieval,
        "sample_orbit_aligned_site_accuracy": float(torch.stack(accuracy).mean()),
        "sample_exact_composition": float(torch.stack(exact).min()),
        "sampling_failures": failures,
    }


def _exact_subset_rows(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    lower_bound_lookup: dict[tuple[str, str], float],
) -> list[dict[str, float | int | str]]:
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    law = RemainingCountAssignmentLaw()
    rows: list[dict[str, float | int | str]] = []
    for example in examples:
        packed = pack_assignment_carriers([example], device="cpu")

        def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
            return cpu_model(
                packed.site_features,
                packed.graph_features,
                packed.batch,
                packed.edge_source,
                packed.edge_target,
                packed.edge_rbf,
                partial,
                packed.composition_counts,
                remaining.unsqueeze(0),
                packed.parent_space_group,
                packed.cell_index,
            )

        probability = law.exact_quotient_probability(
            score,
            example.target_assignment,
            example.composition_counts,
            example.parent_permutations,
        )
        uniform = math.exp(-_uniform_quotient_nll(example))
        lower = lower_bound_lookup[(example.material_id_audit_only, example.embedding_key)]
        rows.append(
            {
                "material_id": example.material_id_audit_only,
                "embedding_key": example.embedding_key,
                "sites": int(example.target_assignment.numel()),
                "exact_quotient_probability": probability,
                "uniform_quotient_probability": uniform,
                "model_minus_uniform_probability": probability - uniform,
                "lower_bound_probability": lower,
                "lower_bound_probability_excess": lower - probability,
            }
        )
    return rows


def _relabel_example(example: AssignmentCarrierExample, order: torch.Tensor) -> AssignmentCarrierExample:
    inverse = torch.argsort(order)
    return replace(
        example,
        site_features=example.site_features[order],
        edge_source=inverse[example.edge_source],
        edge_target=inverse[example.edge_target],
        target_assignment=example.target_assignment[order],
        parent_permutations=inverse[example.parent_permutations[:, order]],
    )


@torch.no_grad()
def _relabel_logit_max_abs(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    residual = 0.0
    for example in examples[:count]:
        nodes = example.target_assignment.numel()
        order = torch.randperm(nodes, generator=generator)
        changed = _relabel_example(example, order)
        logits = []
        for value in (example, changed):
            packed = pack_assignment_carriers([value], device=device)
            logits.append(
                model(
                    packed.site_features,
                    packed.graph_features,
                    packed.batch,
                    packed.edge_source,
                    packed.edge_target,
                    packed.edge_rbf,
                    torch.full((nodes,), -1, dtype=torch.long, device=device),
                    packed.composition_counts,
                    packed.composition_counts,
                    packed.parent_space_group,
                    packed.cell_index,
                )
            )
        residual = max(residual, float(torch.max(torch.abs(logits[1] - logits[0][order.to(device)]))))
    return residual


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_assignment_iid_gate_v1" or protocol.get(
        "status_before_run"
    ) != "frozen_not_run":
        raise ValueError("unexpected or unfrozen assignment IID protocol")
    role_path = repository / protocol["source"]["iid_role_result"]
    if _normalized_source_sha256(role_path) != protocol["source"][
        "iid_role_result_normalized_sha256"
    ]:
        raise ValueError("assignment IID role result identity changed")
    manifest = load_json_object(args.carrier_root / "manifest.json")
    if manifest.get("records_sha256") != protocol["source"]["carrier_records_sha256"]:
        raise ValueError("assignment carrier identity changed")
    implementation_commit = _git_identity(repository)

    training = protocol["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("assignment IID Gate requires CUDA")
    device = torch.device("cuda", int(training["cuda_device"]))
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    model_config = protocol["model"]
    examples = _load_examples(
        args.carrier_root,
        role_path,
        maximum_sites=int(model_config["maximum_sites"]),
        radial_channels=int(model_config["radial_channels"]),
    )
    by_role = {
        role: [value for value in examples if value.evidence_role_audit_only == role]
        for role in {
            "iid_fit",
            "iid_fit_rare",
            "iid_calibration",
            "iid_test",
            "ood_validation",
            "ood_test",
        }
    }
    fit = by_role["iid_fit"] + by_role["iid_fit_rare"]
    fit_by_material: dict[str, list[AssignmentCarrierExample]] = defaultdict(list)
    for example in fit:
        fit_by_material[example.material_id_audit_only].append(example)
    material_ids = sorted(fit_by_material)
    material_batch_size = int(training["material_batch_size"])
    if material_batch_size < 1 or material_batch_size > len(material_ids):
        raise ValueError("invalid material batch size")
    path_samples = int(training["path_samples_per_carrier"])
    if path_samples < 1:
        raise ValueError("path samples per carrier must be positive")

    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=examples[0].site_features.shape[1],
        graph_feature_dim=examples[0].graph_features.shape[0],
        radial_channels=examples[0].edge_rbf.shape[1],
        hidden_dim=int(model_config["hidden_dim"]),
        message_blocks=int(model_config["message_blocks"]),
        maximum_sites=int(model_config["maximum_sites"]),
        maximum_cell_index=int(model_config["maximum_cell_index"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    python_rng = random.Random(seed)
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    history: list[dict[str, float]] = []
    finite_gradient_steps = 0
    steps = int(training["steps"])
    interval = int(training["history_interval"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        selected_materials = python_rng.sample(material_ids, material_batch_size)
        selected = [python_rng.choice(fit_by_material[key]) for key in selected_materials]
        packed = pack_assignment_carriers(
            [example for example in selected for _ in range(path_samples)],
            device=device,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = orderless_assignment_objective(model, packed, generator=torch_generator)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training["gradient_clip_norm"]),
        )
        finite_gradient_steps += int(bool(torch.isfinite(gradient_norm)))
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            history.append(
                {
                    "step": float(step),
                    "train_path_nll": float(objective.loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )

    evaluation = protocol["evaluation"]
    role_rows: dict[str, list[dict[str, Any]]] = {}
    role_summary: dict[str, dict[str, float | int]] = {}
    for role_index, role in enumerate(("iid_calibration", "iid_test", "ood_validation", "ood_test")):
        rows = _bound_rows(
            model,
            by_role[role],
            order_samples=int(evaluation["order_samples"]),
            batch_size=int(evaluation["carrier_batch_size"]),
            seed=seed + 10_000 + role_index,
            device=device,
        )
        role_rows[role] = rows
        role_summary[role] = _bound_summary(
            rows,
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            seed=seed + 20_000 + role_index,
        )

    sampling = {
        role: _sampling_summary(
            model,
            by_role[role],
            draws=int(evaluation["sample_draws_per_carrier"]),
            seed=seed + 30_000 + index * 1_000,
            device=device,
        )
        for index, role in enumerate(("iid_calibration", "iid_test"))
    }
    maximum_exact_sites = int(evaluation["exact_subset_maximum_sites"])
    exact_per_split = int(evaluation["exact_subset_carriers_per_iid_split"])
    exact_examples = []
    for role in ("iid_calibration", "iid_test"):
        eligible = sorted(
            (value for value in by_role[role] if value.target_assignment.numel() <= maximum_exact_sites),
            key=lambda value: (value.material_id_audit_only, value.embedding_key),
        )
        if len(eligible) < exact_per_split:
            raise ValueError("exact subset panel has insufficient frozen support")
        exact_examples.extend(eligible[:exact_per_split])
    lower_lookup = {
        (str(row["material_id"]), str(row["embedding_key"])): float(
            row["quotient_probability_lower_bound"]
        )
        for role in ("iid_calibration", "iid_test")
        for row in role_rows[role]
    }
    exact_rows = _exact_subset_rows(model, exact_examples, lower_lookup)
    exact_summary = {
        "carriers": len(exact_rows),
        "mean_model_minus_uniform_probability": sum(
            float(row["model_minus_uniform_probability"]) for row in exact_rows
        )
        / len(exact_rows),
        "maximum_lower_bound_probability_excess": max(
            float(row["lower_bound_probability_excess"]) for row in exact_rows
        ),
    }
    relabel_residual = _relabel_logit_max_abs(
        model,
        by_role["iid_test"],
        count=int(evaluation["relabel_carriers"]),
        seed=seed + 40_000,
        device=device,
    )

    finite_gradient_fraction = finite_gradient_steps / steps
    acceptance = protocol["acceptance"]
    checks = {
        "iid_calibration_nll_reduction": role_summary["iid_calibration"][
            "relative_nll_reduction_from_uniform"
        ]
        >= float(acceptance["iid_calibration_relative_nll_reduction_min"]),
        "iid_test_nll_reduction": role_summary["iid_test"]["relative_nll_reduction_from_uniform"]
        >= float(acceptance["iid_test_relative_nll_reduction_min"]),
        "iid_calibration_paired_bootstrap": role_summary["iid_calibration"][
            "model_minus_uniform_nll_ucb95"
        ]
        <= float(acceptance["iid_calibration_model_minus_uniform_nll_ucb95_max"]),
        "iid_test_paired_bootstrap": role_summary["iid_test"]["model_minus_uniform_nll_ucb95"]
        <= float(acceptance["iid_test_model_minus_uniform_nll_ucb95_max"]),
        "iid_test_retrieval_lift": sampling["iid_test"]["sample_retrieval_lift_over_uniform"]
        >= float(acceptance["iid_test_sample_retrieval_lift_over_uniform_min"]),
        "iid_test_site_accuracy": sampling["iid_test"]["sample_orbit_aligned_site_accuracy"]
        >= float(acceptance["iid_test_sample_orbit_aligned_site_accuracy_min"]),
        "exact_subset_probability_lift": exact_summary["mean_model_minus_uniform_probability"]
        >= float(acceptance["exact_subset_mean_model_minus_uniform_probability_min"]),
        "exact_subset_bound_consistency": exact_summary["maximum_lower_bound_probability_excess"]
        <= float(acceptance["exact_subset_lower_bound_probability_excess_max"]),
        "relabel_consistency": relabel_residual <= float(acceptance["relabel_logit_max_abs"]),
        "exact_composition": min(
            float(sampling[role]["sample_exact_composition"])
            for role in ("iid_calibration", "iid_test")
        )
        == float(acceptance["sample_exact_composition"]),
        "zero_sampling_failures": sum(
            int(sampling[role]["sampling_failures"]) for role in ("iid_calibration", "iid_test")
        )
        == int(acceptance["sampling_failures"]),
        "finite_gradients": finite_gradient_fraction
        == float(acceptance["finite_gradient_step_fraction"]),
    }
    qualified = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "implementation_commit": implementation_commit,
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "training": {
            "steps": steps,
            "fit_carriers": len(fit),
            "fit_materials": len(material_ids),
            "finite_gradient_step_fraction": finite_gradient_fraction,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "history": history,
        },
        "likelihood": role_summary,
        "sampling": sampling,
        "exact_subset": exact_summary,
        "exact_subset_rows": exact_rows,
        "relabel_logit_max_abs": relabel_residual,
        "carrier_rows": role_rows,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_history(args.output_dir / "training_history.csv", history)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
