"""Oracle-free supporting diagnostics for the pre-registered GaugeFlow Gate A.

This script deliberately cannot declare the full gate passed.  It verifies
that each conditioning path affects the learned flow, measures tensor-orbit
representative consistency, and tests whether targets induce distinguishable
generated state distributions.  Tensor fidelity still requires the external
evidence listed in the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from gaugeflow.conditioning import randomize_tensor_orbit_representative
from gaugeflow.data import PiezoCrystalDataset, collate_crystals
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import normalize_isotypic


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _seed(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _load_panel(protocol: dict[str, Any], root: Path, device: torch.device):
    data = protocol["data"]
    dataset = PiezoCrystalDataset(
        _resolve(root, data["train_csv"]),
        split_manifest=_resolve(root, data["split_manifest"]),
        split=data["split"],
        target_cache_dir=_resolve(root, data["target_cache_dir"]),
    )
    lookup = {str(value): index for index, value in enumerate(dataset.frame.material_id)}
    missing = [value for value in protocol["material_ids"] if value not in lookup]
    if missing:
        raise ValueError(f"Gate A panel IDs are absent from the frozen split: {missing}")
    records = [dataset[lookup[value]] for value in protocol["material_ids"]]
    return collate_crystals(records).to(device)


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    model = GaugeFlowVectorField(
        config["hidden_dim"], config["layers"], config["orbit_frames"],
        conditioning_mode=config["conditioning_mode"],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload["isotypic_scales"].to(device), config["conditioning_mode"]


def _clone_with_conditions(batch, conditions: torch.Tensor):
    clone = batch.clone()
    clone.piezo_irreps = conditions
    return clone


def _flow_gap(
    model: GaugeFlowVectorField, batch, repeats: int, seed: int
) -> dict[str, float]:
    matcher = RiemannianCrystalFlowMatcher()
    permutation = torch.roll(torch.arange(batch.num_graphs, device=batch.piezo_irreps.device), 1)
    shuffled = _clone_with_conditions(batch, batch.piezo_irreps[permutation])
    correct_losses, shuffled_losses = [], []
    with torch.no_grad():
        for repeat in range(repeats):
            _seed(seed + repeat)
            correct_losses.append(float(matcher.loss(model, batch)["loss"]))
            _seed(seed + repeat)
            shuffled_losses.append(float(matcher.loss(model, shuffled)["loss"]))
    correct = torch.tensor(correct_losses)
    wrong = torch.tensor(shuffled_losses)
    gap = (wrong - correct) / correct.clamp_min(1e-8)
    return {
        "correct_flow_loss_mean": float(correct.mean()),
        "shuffled_flow_loss_mean": float(wrong.mean()),
        "shuffled_flow_gap_fraction_mean": float(gap.mean()),
        "shuffled_flow_gap_fraction_median": float(gap.median()),
        "shuffled_worse_fraction": float((wrong > correct).float().mean()),
    }


def _fixed_flow_state(matcher: RiemannianCrystalFlowMatcher, batch, seed: int):
    _seed(seed)
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    time = torch.full((batch.num_graphs,), 0.5, device=batch.frac_coords.device)
    node_time = time[batch.batch].unsqueeze(-1)
    velocity_type = target.type_state - base.type_state
    velocity_coord = torus_logmap(base.frac_coords, target.frac_coords)
    velocity_lattice = target.lattice_log - base.lattice_log
    state = CrystalFlowState(
        type_state=base.type_state + node_time * velocity_type,
        frac_coords=wrap01(base.frac_coords + node_time * velocity_coord),
        lattice_log=base.lattice_log + time.unsqueeze(-1) * velocity_lattice,
    )
    return state, time


def _relative_output_error(reference, alternative) -> tuple[float, dict[str, float]]:
    names = ("type", "coordinate", "lattice")
    values: dict[str, float] = {}
    for name, base, changed in zip(names, reference[:3], alternative[:3]):
        floor = math.sqrt(base.numel()) * 1e-4
        values[name] = float(torch.linalg.vector_norm(changed - base) / (torch.linalg.vector_norm(base) + floor))
    return sum(values.values()) / len(values), values


def _representative_consistency(
    model: GaugeFlowVectorField, batch, repeats: int, seed: int
) -> dict[str, Any]:
    matcher = RiemannianCrystalFlowMatcher()
    state, time = _fixed_flow_state(matcher, batch, seed)
    with torch.no_grad():
        reference = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
            batch.piezo_irreps, batch.condition_present,
        )
        errors, by_head = [], []
        for repeat in range(repeats):
            generator = torch.Generator(device=batch.piezo_irreps.device).manual_seed(seed + 1000 + repeat)
            representative = randomize_tensor_orbit_representative(
                batch.piezo_irreps, generator=generator
            )
            changed = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                representative, batch.condition_present,
            )
            aggregate, heads = _relative_output_error(reference, changed)
            errors.append(aggregate)
            by_head.append(heads)
    return {
        "relative_velocity_error_mean": float(torch.tensor(errors).mean()),
        "relative_velocity_error_max": max(errors),
        "per_head_mean": {
            key: sum(value[key] for value in by_head) / len(by_head)
            for key in by_head[0]
        },
    }


def _dummy_sampling_batch(
    conditions: torch.Tensor, replicates: int, atoms: int, device: torch.device
) -> tuple[Batch, torch.Tensor]:
    records, labels = [], []
    for target, condition in enumerate(conditions):
        for _ in range(replicates):
            records.append(Data(
                atom_types=torch.ones(atoms, dtype=torch.long),
                frac_coords=torch.zeros((atoms, 3)),
                lattice=torch.eye(3).unsqueeze(0),
                piezo_irreps=condition.cpu().unsqueeze(0),
                condition_present=torch.ones((1, 1), dtype=torch.bool),
                num_nodes=atoms,
            ))
            labels.append(target)
    return Batch.from_data_list(records).to(device), torch.tensor(labels, device=device)


def _state_features(state: CrystalFlowState, batch: Batch) -> torch.Tensor:
    features = []
    probabilities = torch.softmax(state.type_state, dim=-1)
    for graph in range(batch.num_graphs):
        nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
        composition = probabilities[nodes].mean(dim=0)
        coords = state.frac_coords[nodes]
        delta = torus_logmap(coords.unsqueeze(1), coords.unsqueeze(0))
        distance = torch.linalg.vector_norm(delta, dim=-1)
        upper = distance[torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)]
        radial = torch.stack((upper.mean(), upper.std(unbiased=False), upper.amin(), upper.amax()))
        features.append(torch.cat((composition, state.lattice_log[graph], radial)))
    return torch.stack(features)


def _standardize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean(dim=0)) / value.std(dim=0, unbiased=False).clamp_min(1e-4)


def _distance_diagnostics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    normalized = _standardize(features)
    distance = torch.cdist(normalized, normalized)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diagonal = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    within = distance[same & ~diagonal]
    between = distance[~same]
    predictions = []
    for index in range(labels.numel()):
        centroids = []
        for label in range(int(labels.max()) + 1):
            keep = (labels == label) & (torch.arange(labels.numel(), device=labels.device) != index)
            centroids.append(normalized[keep].mean(dim=0))
        centroids = torch.stack(centroids)
        predictions.append(int(torch.linalg.vector_norm(centroids - normalized[index], dim=-1).argmin()))
    accuracy = torch.tensor(predictions, device=labels.device).eq(labels).float().mean()
    return {
        "within_target_distance_mean": float(within.mean()),
        "between_target_distance_mean": float(between.mean()),
        "between_within_distance_ratio": float(between.mean() / within.mean().clamp_min(1e-8)),
        "leave_one_out_nearest_centroid_accuracy": float(accuracy),
    }


def _generated_distribution(
    model: GaugeFlowVectorField,
    conditions: torch.Tensor,
    replicates: int,
    atoms: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    matcher = RiemannianCrystalFlowMatcher()
    batch, labels = _dummy_sampling_batch(conditions, replicates, atoms, device)
    permutation = torch.roll(torch.arange(conditions.shape[0], device=device), 1)
    shuffled_conditions = conditions[permutation]
    shuffled_batch, _ = _dummy_sampling_batch(shuffled_conditions, replicates, atoms, device)
    with torch.no_grad():
        _seed(seed)
        correct = matcher.sample(model, batch, steps=steps)
        _seed(seed)
        shuffled = matcher.sample(model, shuffled_batch, steps=steps)
    correct_features = _state_features(correct, batch)
    shuffled_features = _state_features(shuffled, shuffled_batch)
    scale = torch.cat((correct_features, shuffled_features)).std(dim=0, unbiased=False).clamp_min(1e-4)
    shift = torch.linalg.vector_norm((correct_features - shuffled_features) / scale, dim=-1)
    return {
        **_distance_diagnostics(correct_features, labels),
        "condition_permutation_feature_shift_mean": float(shift.mean()),
        "condition_permutation_feature_shift_median": float(shift.median()),
        "all_generated_features_finite": bool(torch.isfinite(correct_features).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a_v1.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/gate_a_v1/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gate_a_v1/report.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    device = torch.device(args.device)
    raw_batch = _load_panel(protocol, root, device)
    evaluation = protocol["evaluation"]
    results: dict[str, Any] = {}
    for method in protocol["methods"]:
        model, scales, checkpoint_mode = _load_model(args.checkpoint_dir / f"{method}.pt", device)
        if checkpoint_mode != method:
            raise ValueError(f"Expected {method} checkpoint, found {checkpoint_mode}")
        batch = raw_batch.clone()
        batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales)
        results[method] = {
            "flow_condition_test": _flow_gap(
                model, batch, evaluation["loss_repeats"], evaluation["seed"]
            ),
            "representative_consistency": _representative_consistency(
                model, batch, evaluation["representative_repeats"], evaluation["seed"]
            ),
            "generated_distribution": _generated_distribution(
                model, batch.piezo_irreps, evaluation["sample_replicates"],
                evaluation["sample_atoms"], evaluation["sample_steps"],
                evaluation["seed"], device,
            ),
        }

    criteria = protocol["supporting_pass_criteria"]
    checks = {
        "all_methods_condition_gap": all(
            results[method]["flow_condition_test"]["shuffled_flow_gap_fraction_median"]
            >= criteria["median_shuffled_flow_gap_fraction_min"]
            for method in protocol["methods"]
        ),
        "gaugeflow_representative_error": (
            results["orbit_alignment"]["representative_consistency"]["relative_velocity_error_mean"]
            <= criteria["representative_velocity_error_max_orbit_alignment"]
        ),
        "gaugeflow_beats_raw_representative_error": (
            results["orbit_alignment"]["representative_consistency"]["relative_velocity_error_mean"]
            / max(results["raw_tensor"]["representative_consistency"]["relative_velocity_error_mean"], 1e-8)
            <= criteria["orbit_alignment_error_ratio_vs_raw_max"]
        ),
        "gaugeflow_target_separation": (
            results["orbit_alignment"]["generated_distribution"]["between_within_distance_ratio"]
            >= criteria["generated_between_within_distance_ratio_min"]
        ),
        "gaugeflow_condition_permutation_shift": (
            results["orbit_alignment"]["generated_distribution"]["condition_permutation_feature_shift_mean"]
            >= criteria["condition_permutation_feature_shift_min"]
        ),
    }
    supporting_passed = all(checks.values())
    report = {
        "schema": 1,
        "gate": "A",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "supporting_checks": checks,
        "oracle_free_supporting_status": "passed" if supporting_passed else "failed",
        "full_gate_status": "incomplete_external_tensor_evidence",
        "full_gate_requires": protocol["full_gate_requires"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
