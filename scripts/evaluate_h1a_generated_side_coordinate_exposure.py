"""Evaluate full coordinate rollouts under clean and generated side states."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.geometry import closest_image_displacements_numpy
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.assignment_data import (
    AssignmentCarrierExample,
    load_assignment_carrier_examples,
    pack_assignment_carriers,
)
from gaugeflow.production.assignment_training import sample_orderless_assignment
from gaugeflow.production.autoregressive_assignment import GeometryAwareRemainingCountScorer
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.runtime import load_tensor_free_ema_runtime


def _wasserstein(left: torch.Tensor, right: torch.Tensor, points: int) -> float:
    probability = torch.linspace(0.0, 1.0, points, dtype=torch.float64)
    return float(
        (
            torch.quantile(left.double(), probability)
            - torch.quantile(right.double(), probability)
        )
        .abs()
        .mean()
    )


def _nearest_neighbours(
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact per-node and per-structure periodic nearest distances."""

    node_values: list[torch.Tensor] = []
    graph_values: list[torch.Tensor] = []
    for graph in range(lattice.shape[0]):
        fractional = coordinates[batch == graph].double().cpu().numpy()
        nodes = fractional.shape[0]
        if nodes < 2:
            raise ValueError("the generated-side panel requires at least two sites")
        target, source = np.nonzero(~np.eye(nodes, dtype=bool))
        displacement, _ = closest_image_displacements_numpy(
            fractional[target] - fractional[source],
            lattice[graph].double().cpu().numpy(),
        )
        distance = torch.from_numpy(np.linalg.norm(displacement, axis=1)).reshape(nodes, nodes - 1)
        nearest = distance.min(dim=1).values
        node_values.append(nearest)
        graph_values.append(nearest.min())
    return torch.cat(node_values), torch.stack(graph_values)


def _geometry_metrics(
    coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    node_counts: torch.Tensor,
    reference_node: torch.Tensor,
    reference_scaled: torch.Tensor,
    *,
    points: int,
    minimum_distance: float,
) -> dict[str, Any]:
    node, graph = _nearest_neighbours(coordinates, lattice, batch)
    scale = (torch.linalg.det(lattice).double().cpu() / node_counts.double().cpu()).pow(1.0 / 3.0)
    scaled = node / torch.repeat_interleave(scale, node_counts.cpu())
    physical_iqr = torch.quantile(reference_node, 0.75) - torch.quantile(reference_node, 0.25)
    scaled_iqr = torch.quantile(reference_scaled, 0.75) - torch.quantile(reference_scaled, 0.25)
    if float(physical_iqr) <= 0.0 or float(scaled_iqr) <= 0.0:
        raise ValueError("reference nearest-neighbour IQR must be positive")
    probabilities = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float64)
    return {
        "node_nearest_w1_angstrom": _wasserstein(node, reference_node, points),
        "node_nearest_w1_normalized": _wasserstein(node, reference_node, points) / float(physical_iqr),
        "cell_scaled_nearest_w1_normalized": _wasserstein(scaled, reference_scaled, points)
        / float(scaled_iqr),
        "valid_minimum_distance_fraction": float((graph >= minimum_distance).double().mean()),
        "node_nearest_quantiles_angstrom": torch.quantile(node, probabilities).tolist(),
        "graph_minimum_quantiles_angstrom": torch.quantile(graph, probabilities).tolist(),
        "finite": bool(torch.isfinite(coordinates).all() and torch.isfinite(lattice).all()),
    }


def _select_supported_test_panel(
    examples: list[AssignmentCarrierExample],
    role_result: dict[str, Any],
    maximum_materials: int,
) -> list[AssignmentCarrierExample]:
    action = {
        (str(row["material_id"]), str(row["embedding_key"])): str(row["action_signature"])
        for row in role_result["carrier_rows"]
    }
    fit_action = {
        str(row["action_signature"])
        for row in role_result["carrier_rows"]
        if str(row["role"]) in {"iid_fit", "iid_fit_rare"}
    }
    grouped: dict[str, list[AssignmentCarrierExample]] = defaultdict(list)
    for example in examples:
        key = (example.material_id_audit_only, example.embedding_key)
        if example.evidence_role_audit_only == "iid_test" and action[key] in fit_action:
            grouped[example.material_id_audit_only].append(example)
    selected = [
        sorted(values, key=lambda value: value.embedding_key)[0]
        for _, values in sorted(grouped.items())
    ][:maximum_materials]
    if len(selected) < 4:
        raise ValueError("supported-IID generated-side panel has fewer than four materials")
    return selected


def _candidate_by_identity(carrier_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with gzip.open(carrier_root / "records.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        material = str(record["material_id_audit_only"])
        for candidate in record["candidates"]:
            key = (material, str(candidate["embedding_key"]))
            if key in candidates:
                raise ValueError(f"duplicate assignment carrier identity: {key}")
            candidates[key] = candidate
    return candidates


def _load_assignment_model(
    checkpoint: Path,
    protocol: dict[str, Any],
    examples: list[AssignmentCarrierExample],
    device: torch.device,
) -> GeometryAwareRemainingCountScorer:
    model_spec = protocol["assignment_model"]
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=examples[0].site_features.shape[1],
        graph_feature_dim=examples[0].graph_features.shape[0],
        radial_channels=int(model_spec["radial_channels"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        message_blocks=int(model_spec["message_blocks"]),
        maximum_sites=int(model_spec["maximum_sites"]),
        maximum_cell_index=int(model_spec["maximum_cell_index"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "schema": 2,
        "task": "parent_conditioned_assignment_iid_v3",
        "protocol_sha256": str(protocol["sources"]["assignment_protocol_sha256"]),
        "seed": int(protocol["evaluation"]["seed"]),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("assignment checkpoint provenance does not match the frozen Gate")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--coordinate-checkpoint", type=Path, required=True)
    parser.add_argument("--lattice-checkpoint", type=Path, required=True)
    parser.add_argument("--assignment-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_generated_side_coordinate_exposure_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen generated-side protocol")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the generated-side Gate requires CUDA")
    sources = protocol["sources"]
    hash_contract = {
        args.cache_root / "manifest.json": sources["cache_manifest_sha256"],
        args.carrier_root / "manifest.json": sources["carrier_manifest_sha256"],
        Path("reports/h1a_assignment_iid_calibration_split_v1/result.json"): sources[
            "assignment_role_result_sha256"
        ],
        Path("reports/h1a_assignment_iid_gate_v3/result.json"): sources["assignment_result_sha256"],
        Path("reports/h1a_lattice_l1_v1/result.json"): sources["lattice_result_sha256"],
        Path("reports/h1a_coordinate_clean_side_current_v1/result.json"): sources[
            "coordinate_result_sha256"
        ],
        args.assignment_checkpoint: sources["assignment_checkpoint_sha256"],
        args.lattice_checkpoint: sources["lattice_checkpoint_sha256"],
        args.coordinate_checkpoint: sources["coordinate_checkpoint_sha256"],
    }
    for path, expected in hash_contract.items():
        if sha256_file(path) != str(expected):
            raise ValueError(f"frozen generated-side input hash mismatch: {path}")

    evaluation = protocol["evaluation"]
    assignment_protocol = load_json_object(Path("configs/protocols/h1a_assignment_iid_gate_v3.json"))
    examples = load_assignment_carrier_examples(
        args.carrier_root,
        Path("reports/h1a_assignment_iid_calibration_split_v1/result.json"),
        maximum_sites=int(protocol["assignment_model"]["maximum_sites"]),
        radial_channels=int(protocol["assignment_model"]["radial_channels"]),
    )
    role_result = load_json_object(Path("reports/h1a_assignment_iid_calibration_split_v1/result.json"))
    selected = _select_supported_test_panel(
        examples,
        role_result,
        int(evaluation["maximum_materials"]),
    )
    assignment_model = _load_assignment_model(
        args.assignment_checkpoint,
        protocol,
        selected,
        device,
    )
    packed_carriers = pack_assignment_carriers(selected, device=device)
    generated_carrier_tokens = sample_orderless_assignment(
        assignment_model,
        packed_carriers,
        generator=torch.Generator(device=device).manual_seed(int(evaluation["assignment_seed"])),
    )

    dataset = PackedAlexP1Dataset(args.cache_root, "train", include_material_id=True)
    source_index = {value: index for index, value in enumerate(dataset.material_ids_audit_only)}
    if len(source_index) != len(dataset):
        raise ValueError("P1 train material IDs are not unique")
    indices = torch.tensor(
        [source_index[value.material_id_audit_only] for value in selected],
        dtype=torch.long,
    )
    source = dataset.select_model_batch(indices, device=device)
    node_counts = torch.bincount(source.batch, minlength=len(selected))
    if not torch.equal(node_counts.cpu(), torch.tensor([x.target_assignment.numel() for x in selected])):
        raise ValueError("assignment carriers and P1 source node counts differ")

    candidate_lookup = _candidate_by_identity(args.carrier_root)
    generated_tokens = torch.empty_like(source.atom_types)
    offset = 0
    panel_rows: list[dict[str, Any]] = []
    for graph, example in enumerate(selected):
        nodes = int(node_counts[graph])
        candidate = candidate_lookup[(example.material_id_audit_only, example.embedding_key)]
        source_by_carrier = torch.tensor(
            candidate["alignment_audit"]["source_node_by_carrier_node"],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(torch.sort(source_by_carrier).values, torch.arange(nodes, device=device)):
            raise ValueError("carrier-to-source assignment map is not a permutation")
        clean_mapped = torch.empty(nodes, dtype=torch.long, device=device)
        generated_mapped = torch.empty_like(clean_mapped)
        clean_mapped[source_by_carrier] = example.target_assignment.to(device)
        generated_mapped[source_by_carrier] = generated_carrier_tokens[offset : offset + nodes]
        selected_source = source.atom_types[source.batch == graph]
        if not torch.equal(clean_mapped, selected_source):
            raise ValueError("carrier target does not reproduce the audited P1 source ordering")
        generated_tokens[source.batch == graph] = generated_mapped
        panel_rows.append(
            {
                "material_id_audit_only": example.material_id_audit_only,
                "embedding_key": example.embedding_key,
                "nodes": nodes,
                "assignment_site_accuracy": float((generated_mapped == selected_source).float().mean()),
            }
        )
        offset += nodes
    clean_counts = torch.zeros((len(selected), 118), dtype=torch.long, device=device)
    generated_counts = torch.zeros_like(clean_counts)
    clean_counts.index_put_((source.batch, source.atom_types), torch.ones_like(source.atom_types), accumulate=True)
    generated_counts.index_put_((source.batch, generated_tokens), torch.ones_like(generated_tokens), accumulate=True)
    exact_composition = bool(torch.equal(clean_counts, generated_counts))

    blueprint = ParentBlueprintBatch.from_node_counts(
        node_counts,
        dtype=source.lattice.dtype,
        device=device,
    )
    lattice_runtime = load_tensor_free_ema_runtime(
        args.lattice_checkpoint,
        device,
        protocol_name="h1a_lattice_l1_v1",
        protocol_sha256=str(sources["lattice_protocol_sha256"]),
    )
    lattice_sampler = TensorFreeReverseSampler(
        lattice_runtime.model,
        lattice_runtime.lattice_standardizer,
        maximum_time=float(lattice_runtime.training_config["maximum_time"]),
        categorical_path=str(lattice_runtime.training_config["categorical_path"]),
    )
    lattice_initial = lattice_sampler.initialize_lattice_state(
        blueprint,
        generator=torch.Generator(device=device).manual_seed(int(evaluation["lattice_initial_seed"])),
    )

    def generated_lattice(tokens: torch.Tensor) -> torch.Tensor:
        return lattice_sampler.sample_lattice(
            tokens,
            blueprint,
            steps=int(evaluation["lattice_steps"]),
            initial_state=lattice_initial,
            continuous_generator=torch.Generator(device=device).manual_seed(
                int(evaluation["lattice_brownian_seed"])
            ),
            continuous_mode="reverse_sde",
            time_grid="uniform_log_alpha",
        ).lattice

    clean_conditioned_lattice = generated_lattice(source.atom_types)
    assignment_conditioned_lattice = generated_lattice(generated_tokens)
    lattice_permutation_residual = float(
        (clean_conditioned_lattice - assignment_conditioned_lattice).abs().max()
    )
    generated_lattice_value = clean_conditioned_lattice

    coordinate_runtime = load_tensor_free_ema_runtime(
        args.coordinate_checkpoint,
        device,
        protocol_name="h1a_coordinate_clean_side_current_v1",
        protocol_sha256=str(sources["coordinate_protocol_sha256"]),
    )
    coordinate_sampler = TensorFreeReverseSampler(
        coordinate_runtime.model,
        coordinate_runtime.lattice_standardizer,
        coordinate_sigma_min=float(coordinate_runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(coordinate_runtime.training_config["coordinate_sigma_max"]),
        maximum_time=float(coordinate_runtime.training_config["maximum_time"]),
        categorical_path=str(coordinate_runtime.training_config["categorical_path"]),
    )
    coordinate_initial = coordinate_sampler.initialize_coordinate_state(
        blueprint,
        generator=torch.Generator(device=device).manual_seed(int(evaluation["coordinate_initial_seed"])),
    )
    arms = {
        "clean_assignment_clean_lattice": (source.atom_types, source.lattice),
        "generated_assignment_clean_lattice": (generated_tokens, source.lattice),
        "clean_assignment_generated_lattice": (source.atom_types, generated_lattice_value),
        "generated_assignment_generated_lattice": (generated_tokens, generated_lattice_value),
    }
    terminal: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    failures: dict[str, str] = {}
    for arm, (tokens, lattice) in arms.items():
        try:
            output = coordinate_sampler.sample_coordinates(
                tokens,
                lattice,
                blueprint,
                steps=int(evaluation["coordinate_steps"]),
                initial_state=coordinate_initial,
                continuous_generator=torch.Generator(device=device).manual_seed(
                    int(evaluation["coordinate_brownian_seed"])
                ),
                continuous_mode="reverse_sde",
                time_grid="uniform_log_alpha",
            )
            terminal[arm] = (output.fractional_coordinates, lattice)
        except (SamplingFailure, RuntimeError, ValueError) as error:
            failures[arm] = f"{type(error).__name__}: {error}"

    reference_node, _ = _nearest_neighbours(
        source.fractional_coordinates,
        source.lattice,
        source.batch,
    )
    reference_scale = (
        torch.linalg.det(source.lattice).double().cpu() / node_counts.double().cpu()
    ).pow(1.0 / 3.0)
    reference_scaled = reference_node / torch.repeat_interleave(reference_scale, node_counts.cpu())
    metrics = {
        arm: _geometry_metrics(
            coordinates,
            lattice,
            source.batch,
            node_counts,
            reference_node,
            reference_scaled,
            points=int(evaluation["wasserstein_points"]),
            minimum_distance=float(evaluation["minimum_distance_angstrom"]),
        )
        for arm, (coordinates, lattice) in terminal.items()
    }
    acceptance = protocol["acceptance"]
    baseline = metrics.get("clean_assignment_clean_lattice", {})

    def degradation(arm: str, metric: str) -> float:
        if arm not in metrics or metric not in baseline:
            return math.inf
        return float(metrics[arm][metric]) - float(baseline[metric])

    checks = {
        "panel_size": len(selected) >= int(acceptance["minimum_materials"]),
        "exact_assignment_composition": exact_composition,
        "lattice_composition_permutation_invariance": lattice_permutation_residual
        <= float(acceptance["lattice_permutation_max_abs"]),
        "finite_positive_generated_lattice": bool(
            torch.isfinite(generated_lattice_value).all()
            and (torch.linalg.det(generated_lattice_value) > 0.0).all()
        ),
        "zero_sampling_failures": not failures,
        "clean_clean_nearest_w1": float(baseline.get("node_nearest_w1_normalized", math.inf))
        <= float(acceptance["clean_clean_nearest_w1_normalized_max"]),
        "clean_clean_cell_scaled_w1": float(
            baseline.get("cell_scaled_nearest_w1_normalized", math.inf)
        )
        <= float(acceptance["clean_clean_cell_scaled_w1_normalized_max"]),
        "clean_clean_valid_distance": float(
            baseline.get("valid_minimum_distance_fraction", -math.inf)
        )
        >= float(acceptance["clean_clean_valid_minimum_distance_fraction_min"]),
        "assignment_exposure_w1": degradation(
            "generated_assignment_clean_lattice", "node_nearest_w1_normalized"
        )
        <= float(acceptance["assignment_w1_additive_degradation_max"]),
        "lattice_exposure_w1": degradation(
            "clean_assignment_generated_lattice", "node_nearest_w1_normalized"
        )
        <= float(acceptance["lattice_w1_additive_degradation_max"]),
        "joint_exposure_w1": degradation(
            "generated_assignment_generated_lattice", "node_nearest_w1_normalized"
        )
        <= float(acceptance["joint_w1_additive_degradation_max"]),
        "joint_valid_distance": float(
            metrics.get("generated_assignment_generated_lattice", {}).get(
                "valid_minimum_distance_fraction", -math.inf
            )
        )
        >= float(acceptance["joint_valid_minimum_distance_fraction_min"]),
    }
    qualified = all(checks.values())
    if not all(checks[key] for key in checks if key.startswith("clean_clean")):
        classification = "full_prior_coordinate_failure"
    elif not checks["assignment_exposure_w1"]:
        classification = "assignment_exposure_failure"
    elif not checks["lattice_exposure_w1"]:
        classification = "lattice_exposure_failure"
    elif not checks["joint_exposure_w1"] or not checks["joint_valid_distance"]:
        classification = "joint_side_state_interaction_failure"
    else:
        classification = "generated_side_coordinate_closure_pass"
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": qualified,
        "classification": classification,
        "checks": checks,
        "panel_materials": len(selected),
        "panel_identity_sha256": canonical_json_hash(
            [(row["material_id_audit_only"], row["embedding_key"]) for row in panel_rows]
        ),
        "panel_rows": panel_rows,
        "exact_assignment_composition": exact_composition,
        "lattice_permutation_residual_max_abs": lattice_permutation_residual,
        "sampling_failures": failures,
        "geometry": metrics,
        "additive_w1_degradation": {
            "assignment": degradation(
                "generated_assignment_clean_lattice", "node_nearest_w1_normalized"
            ),
            "lattice": degradation(
                "clean_assignment_generated_lattice", "node_nearest_w1_normalized"
            ),
            "joint": degradation(
                "generated_assignment_generated_lattice", "node_nearest_w1_normalized"
            ),
        },
        "assignment_protocol_identity": canonical_json_hash(assignment_protocol),
        "boundary": protocol["decision_rule"]["boundary"],
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
