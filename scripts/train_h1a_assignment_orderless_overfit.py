"""Run the frozen fixed-batch overfit screen for orderless assignment."""

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
from pathlib import Path
from typing import Sequence

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
        raise ValueError("assignment overfit screen requires a clean committed tree")
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
        (str(row["material_id"]), int(row["candidate_index"]), str(row["embedding_key"])): str(row["role"])
        for row in role_result["carrier_rows"]
    }
    with gzip.open(carrier_root / "records.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    examples: list[AssignmentCarrierExample] = []
    for record in records:
        material_id = str(record["material_id_audit_only"])
        for candidate_index, candidate in enumerate(record["candidates"]):
            embedding_key = str(candidate["embedding_key"])
            key = (material_id, candidate_index, embedding_key)
            if key not in roles:
                raise ValueError(f"assignment carrier is absent from the frozen role split: {key}")
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
    if len(examples) != len(roles):
        raise ValueError("geometry-complete carriers and frozen roles have different support")
    return examples


def _fixed_material_batch(
    examples: Sequence[AssignmentCarrierExample],
    *,
    size: int,
) -> list[AssignmentCarrierExample]:
    grouped: dict[str, list[AssignmentCarrierExample]] = defaultdict(list)
    for example in examples:
        if example.evidence_role_audit_only in {"iid_fit", "iid_fit_rare"}:
            grouped[example.material_id_audit_only].append(example)
    selected = [sorted(grouped[key], key=lambda value: value.embedding_key)[0] for key in sorted(grouped)]
    if len(selected) < size:
        raise ValueError("fixed overfit batch exceeds IID-fit material support")
    return selected[:size]


def _orbit_size(example: AssignmentCarrierExample) -> int:
    return int(
        torch.unique(
            example.target_assignment[example.parent_permutations],
            dim=0,
        ).shape[0]
    )


def _uniform_quotient_nll(example: AssignmentCarrierExample) -> float:
    counts = example.composition_counts[example.composition_counts > 0]
    assignments = math.factorial(example.target_assignment.numel())
    for count in counts.tolist():
        assignments //= math.factorial(int(count))
    return math.log(assignments / _orbit_size(example))


@torch.no_grad()
def _path_bound_metrics(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    *,
    order_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    packed = pack_assignment_carriers(examples, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    graph_log_probability = []
    for _ in range(order_samples):
        graph_log_probability.append(
            orderless_assignment_objective(model, packed, generator=generator)
            .graph_log_probability.detach()
            .to(torch.float64)
        )
    mean_path_log_probability = torch.stack(graph_log_probability).mean(dim=0)
    log_orbit = torch.tensor(
        [math.log(_orbit_size(value)) for value in examples],
        dtype=torch.float64,
        device=device,
    )
    quotient_lower_bound_nll = -(mean_path_log_probability + log_orbit)
    uniform = torch.tensor(
        [_uniform_quotient_nll(value) for value in examples],
        dtype=torch.float64,
        device=device,
    )
    return {
        "mean_quotient_lower_bound_nll": float(quotient_lower_bound_nll.mean()),
        "mean_uniform_quotient_nll": float(uniform.mean()),
        "mean_target_quotient_probability_lower_bound": float(
            torch.exp(-quotient_lower_bound_nll).mean()
        ),
        "relative_nll_reduction_from_uniform": float(
            ((uniform.mean() - quotient_lower_bound_nll.mean()) / uniform.mean().clamp_min(1e-12))
        ),
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
    random_key = torch.rand(draws, nodes, generator=generator, device=device)
    reveal_order = torch.argsort(random_key, dim=1)
    partial = torch.full((draws, nodes), -1, dtype=torch.long, device=device)
    remaining = packed.composition_counts.clone()
    law = RemainingCountAssignmentLaw()
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
        flat_site = torch.arange(draws, device=device) * nodes + site
        log_probability = law.batched_step_log_probabilities(logits[flat_site], remaining)
        token = torch.multinomial(log_probability.exp(), 1, generator=generator).squeeze(1)
        partial[torch.arange(draws, device=device), site] = token
        remaining[torch.arange(draws, device=device), token] -= 1
    if bool((remaining != 0).any()) or bool((partial < 0).any()):
        raise RuntimeError("orderless assignment sampling failed exact-count closure")
    return partial


@torch.no_grad()
def _sampling_metrics(
    model: GeometryAwareRemainingCountScorer,
    examples: Sequence[AssignmentCarrierExample],
    *,
    draws: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    retrieval = []
    accuracy = []
    exact_count = []
    for index, example in enumerate(examples):
        samples = _sample_example(
            model,
            example,
            draws=draws,
            seed=seed + index,
            device=device,
        )
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
        exact_count.append(
            torch.all(observed == example.composition_counts.to(device), dim=1).to(torch.float32).mean()
        )
    return {
        "sample_retrieval": float(torch.stack(retrieval).mean()),
        "sample_orbit_aligned_site_accuracy": float(torch.stack(accuracy).mean()),
        "sample_exact_composition": float(torch.stack(exact_count).min()),
    }


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
    if (
        protocol.get("protocol") != "h1a_assignment_orderless_overfit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen orderless-assignment overfit protocol")
    role_path = repository / protocol["source"]["iid_role_result"]
    if _normalized_source_sha256(role_path) != protocol["source"]["iid_role_result_normalized_sha256"]:
        raise ValueError("assignment IID role result identity changed")
    carrier_manifest = load_json_object(args.carrier_root / "manifest.json")
    if carrier_manifest.get("records_sha256") != protocol["source"]["carrier_records_sha256"]:
        raise ValueError("geometry-complete carrier identity changed")
    implementation_commit = _git_identity(repository)

    seed = int(protocol["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("assignment overfit screen requires CUDA")
    device = torch.device("cuda", int(protocol["training"]["cuda_device"]))
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    model_config = protocol["model"]
    all_examples = _load_examples(
        args.carrier_root,
        role_path,
        maximum_sites=int(model_config["maximum_sites"]),
        radial_channels=int(model_config["radial_channels"]),
    )
    examples = _fixed_material_batch(
        all_examples,
        size=int(protocol["training"]["fixed_materials"]),
    )
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=examples[0].site_features.shape[1],
        graph_feature_dim=examples[0].graph_features.shape[0],
        radial_channels=int(model_config["radial_channels"]),
        hidden_dim=int(model_config["hidden_dim"]),
        message_blocks=int(model_config["message_blocks"]),
        maximum_sites=int(model_config["maximum_sites"]),
        maximum_cell_index=int(model_config["maximum_cell_index"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    packed = pack_assignment_carriers(examples, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    history: list[dict[str, float]] = []
    steps = int(protocol["training"]["steps"])
    interval = int(protocol["training"]["history_interval"])
    finite_gradient_steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = orderless_assignment_objective(model, packed, generator=generator)
        objective.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(protocol["training"]["gradient_clip_norm"]),
        )
        finite_gradient_steps += int(bool(torch.isfinite(gradient_norm)))
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            history.append(
                {
                    "step": float(step),
                    "path_nll": float(objective.loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )

    evaluation = protocol["evaluation"]
    path_metrics = _path_bound_metrics(
        model,
        examples,
        order_samples=int(evaluation["order_samples"]),
        seed=seed + 1_000,
        device=device,
    )
    sampling_metrics = _sampling_metrics(
        model,
        examples,
        draws=int(evaluation["sample_draws_per_carrier"]),
        seed=seed + 2_000,
        device=device,
    )
    metrics = {
        **path_metrics,
        **sampling_metrics,
        "finite_gradient_step_fraction": finite_gradient_steps / steps,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
    }
    acceptance = protocol["acceptance"]
    checks = {
        "nll_reduction": metrics["relative_nll_reduction_from_uniform"]
        >= float(acceptance["relative_nll_reduction_from_uniform_min"]),
        "target_probability": metrics["mean_target_quotient_probability_lower_bound"]
        >= float(acceptance["mean_target_quotient_probability_lower_bound_min"]),
        "sample_retrieval": metrics["sample_retrieval"]
        >= float(acceptance["sample_retrieval_min"]),
        "sample_site_accuracy": metrics["sample_orbit_aligned_site_accuracy"]
        >= float(acceptance["sample_orbit_aligned_site_accuracy_min"]),
        "exact_composition": metrics["sample_exact_composition"]
        == float(acceptance["sample_exact_composition"]),
        "finite_gradients": metrics["finite_gradient_step_fraction"]
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
        "metrics": metrics,
        "fixed_examples": [value.embedding_key for value in examples],
        "training_history": history,
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
