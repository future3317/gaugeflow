"""Factorial Stage-E1a rollout to localize generated-side condition drift.

The four arms hold progressively fewer side states fixed:

``oracle_cal``
    oracle composition, assignment and lattice; generate coordinates only.
``oracle_ca``
    oracle composition and assignment; generate lattice and coordinates.
``oracle_c``
    oracle composition only; generate assignment, lattice and coordinates.
``free``
    generate the complete product state.

Every arm is paired across the Stage-C base, the E0 conditioned model, and a
one-step tensor-swap control.  This is a diagnostic protocol, not a Stage-E
qualification gate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from train_stage_d_response import _load_model as load_stage_d_model
from train_stage_e_orbit_mimic import _load_backbones

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.generation_metrics import minimum_periodic_distances, quantile_wasserstein, robust_scale
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.response_data import StageDResponseDataset, collate_response_records
from gaugeflow.production.response_normalization import load_response_normalizer
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.tensor import fixed_so3_frames, piezo_to_irreps, rotate_rank3

_NOBLE_GAS_TOKENS = frozenset({1, 9, 17, 35, 53, 85, 117})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "cache",
        "normalizer",
        "stage-c-checkpoint",
        "stage-d-checkpoint",
        "stage-e-checkpoint",
        "composition-checkpoint",
        "composition-protocol",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--lattice-adapter",
        type=Path,
        default=None,
        help="optional generated-side lattice residual adapter payload",
    )
    parser.add_argument(
        "--lattice-adapter-role",
        choices=("both", "conditioned"),
        default="both",
        help=(
            "which model receives the optional lattice adapter; use conditioned "
            "for the strict A/B/C comparison"
        ),
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="optional JSON lattice trajectory aggregate for oracle_ca/oracle_c/free",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _orbit_error(prediction: torch.Tensor, target: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    rotated = rotate_rank3(target[:, None], rotations[None].to(target))
    return (prediction[:, None] - rotated).flatten(2).square().mean(dim=-1).min(dim=1).values.sqrt()


def _sampler(model: Any, metadata: dict[str, Any], composition_model: Any) -> TensorFreeReverseSampler:
    training = metadata["stage_b_metadata"]["a1_training_config"]
    return TensorFreeReverseSampler(
        model,
        P1LatticeStandardizer.from_mapping(metadata["stage_b_metadata"]["lattice_standardization"]),
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        maximum_time=float(training["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )


def _fixed_counts(tokens: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    flat = batch * 118 + tokens
    return torch.bincount(flat, minlength=graphs * 118).reshape(graphs, 118)


def _seeded(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _trajectory_stats(values: list[torch.Tensor]) -> dict[str, list[float]]:
    """Summarize [time, graph] telemetry without changing sampler state."""

    if not values:
        return {}
    merged = torch.cat(values, dim=1).double()
    probabilities = torch.tensor([0.5, 0.95, 0.99, 1.0], dtype=torch.float64)
    quantiles = torch.quantile(merged, probabilities, dim=1)
    return {
        "median": quantiles[0].tolist(),
        "p95": quantiles[1].tolist(),
        "p99": quantiles[2].tolist(),
        "max": quantiles[3].tolist(),
    }


def _build_trajectory_report(
    traces: dict[str, dict[str, list[Any]]],
    *,
    protocol: dict[str, Any],
    samples: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "gaugeflow.stage_e_lattice_trajectory.v1",
        "protocol": protocol.get("protocol"),
        "protocol_sha256": sha256_file(Path(protocol["_protocol_path"])),
        "samples": samples,
        "arms": {},
    }
    for arm, roles in traces.items():
        report["arms"][arm] = {}
        for role, diagnostics in roles.items():
            if not diagnostics:
                continue
            first = diagnostics[0]
            trajectory_time = first.trajectory_time
            if trajectory_time is None:
                continue
            log_volume = [item.trajectory_log_volume for item in diagnostics]
            shape_norm = [item.trajectory_shape_norm for item in diagnostics]
            physical_volume = [item.trajectory_physical_volume for item in diagnostics]
            condition_number = [item.trajectory_condition_number for item in diagnostics]
            report["arms"][arm][role] = {
                "batches": len(diagnostics),
                "graphs": int(sum(value.shape[1] for value in log_volume if value is not None)),
                "time": trajectory_time.tolist(),
                "log_volume": _trajectory_stats([value for value in log_volume if value is not None]),
                "physical_volume": _trajectory_stats(
                    [value for value in physical_volume if value is not None]
                ),
                "shape_norm": _trajectory_stats([value for value in shape_norm if value is not None]),
                "condition_number": _trajectory_stats(
                    [value for value in condition_number if value is not None]
                ),
                "posterior_clean_coefficient": first.posterior_clean_coefficient.tolist()
                if first.posterior_clean_coefficient is not None
                else [],
                "posterior_state_coefficient": first.posterior_state_coefficient.tolist()
                if first.posterior_state_coefficient is not None
                else [],
                "posterior_variance": first.posterior_variance.tolist()
                if first.posterior_variance is not None
                else [],
            }
    return report


def _load_lattice_adapter(model: Any, path: Path, source_checkpoint: Path) -> None:
    """Attach a hash-bound generated-side adapter without changing the base field."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "gaugeflow.stage_e_lattice_generated_exposure.v1":
        raise ValueError("unexpected lattice adapter schema")
    if payload.get("source_checkpoint_sha256") != sha256_file(source_checkpoint):
        raise ValueError("lattice adapter source checkpoint mismatch")
    state = payload.get("adapter")
    if not isinstance(state, dict):
        raise ValueError("lattice adapter payload has no state dict")
    model.attach_lattice_residual_adapter()
    assert model.lattice_residual_adapter is not None
    model.lattice_residual_adapter.load_state_dict(state, strict=True)


def _run_arm(
    sampler: TensorFreeReverseSampler,
    arm: str,
    role_condition: torch.Tensor | None,
    target_elements: torch.Tensor,
    target_lattice: torch.Tensor,
    target_batch: torch.Tensor,
    node_counts: torch.Tensor,
    *,
    steps: int,
    seed: int,
    continuous_mode: Literal["reverse_sde", "probability_flow"],
    trace_lattice: bool = False,
    diagnostics_sink: list[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blueprint = ParentBlueprintBatch.from_node_counts(node_counts)
    graphs = int(node_counts.numel())
    # Explicit common-noise states are reused for the baseline/conditioned/
    # swap roles.  The categorical stream is still paired by seed in the arms
    # that expose a discrete reverse path.
    if arm == "oracle_cal":
        initial = sampler.initialize_coordinate_state(
            blueprint, generator=_seeded(blueprint.batch.device, seed)
        )
        coordinate_generated = sampler.sample_coordinates(
            target_elements,
            target_lattice,
            blueprint,
            tensor_condition=role_condition,
            steps=steps,
            initial_state=initial,
            continuous_generator=_seeded(blueprint.batch.device, seed + 2),
            continuous_mode=continuous_mode,
        )
        return (
            coordinate_generated.element_tokens,
            coordinate_generated.fractional_coordinates,
            coordinate_generated.lattice,
        )

    if arm == "oracle_ca":
        lattice_initial = sampler.initialize_lattice_state(
            blueprint, generator=_seeded(blueprint.batch.device, seed)
        )
        lattice_sample = sampler.sample_lattice(
            target_elements,
            blueprint,
            tensor_condition=role_condition,
            steps=steps,
            initial_state=lattice_initial,
            continuous_generator=_seeded(blueprint.batch.device, seed + 1),
            continuous_mode=continuous_mode,
            trace_lattice=trace_lattice,
        )
        if diagnostics_sink is not None:
            diagnostics_sink.append(lattice_sample.diagnostics)
        lattice = lattice_sample.lattice
        coordinate_initial = sampler.initialize_coordinate_state(
            blueprint, generator=_seeded(blueprint.batch.device, seed + 3)
        )
        coordinate_generated = sampler.sample_coordinates(
            target_elements,
            lattice,
            blueprint,
            tensor_condition=role_condition,
            steps=steps,
            initial_state=coordinate_initial,
            continuous_generator=_seeded(blueprint.batch.device, seed + 4),
            continuous_mode=continuous_mode,
        )
        return coordinate_generated.element_tokens, coordinate_generated.fractional_coordinates, lattice

    counts = _fixed_counts(target_elements, target_batch, graphs) if arm == "oracle_c" else None
    generated = sampler.sample(
        blueprint,
        tensor_condition=role_condition,
        composition_counts=counts,
        steps=steps,
        initialization_generator=_seeded(blueprint.batch.device, seed),
        categorical_generator=_seeded(blueprint.batch.device, seed + 1),
        continuous_generator=_seeded(blueprint.batch.device, seed + 2),
        continuous_mode=continuous_mode,
        trace_lattice=trace_lattice,
    )
    if diagnostics_sink is not None:
        diagnostics_sink.append(generated.diagnostics)
    return generated.element_tokens, generated.fractional_coordinates, generated.lattice


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    protocol_name = protocol.get("protocol")
    if protocol_name not in {
        "stage_e_e1a_factorial_rollout_v1",
        "stage_e_e1a_factorial_rollout_v2",
    }:
        raise ValueError("unexpected Stage-E1a protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Stage-E1a rollout requires CUDA")

    validation = StageDResponseDataset(args.cache, "val")
    candidates = torch.arange(len(validation), dtype=torch.long)
    data_quality = protocol.get("data_quality", {})
    excluded: list[int] = []
    if bool(data_quality.get("exclude_pure_noble_gas", False)):
        eligible: list[int] = []
        for index in range(len(validation)):
            tokens = validation[index].element_tokens
            unique = torch.unique(tokens)
            if unique.numel() == 1 and int(unique.item()) in _NOBLE_GAS_TOKENS:
                excluded.append(index)
            else:
                eligible.append(index)
        candidates = torch.tensor(eligible, dtype=torch.long)
    if int(protocol["samples"]) > candidates.numel():
        raise ValueError("factorial protocol requests more samples than its eligible panel")
    permutation = torch.randperm(
        candidates.numel(), generator=torch.Generator().manual_seed(int(protocol["seed"]))
    )
    selected = candidates[permutation[: int(protocol["samples"])]].tolist()
    records = [validation[int(index)] for index in selected]
    target_batch = collate_response_records(records).to(device)
    normalizer = load_response_normalizer(
        args.normalizer, expected_cache_sha256=str(validation.manifest["cache_sha256"])
    ).to(device)
    target_normalized = normalizer.normalize_piezoelectric(
        target_batch.targets.piezoelectric, target_batch.source_index
    )
    conditions = piezo_to_irreps(target_normalized)
    rotations = fixed_so3_frames(int(protocol["orbit_frames"]), seed=int(protocol["seed"]) + 1).to(device)

    base_model, unused_teacher, _, metadata = _load_backbones(args.stage_c_checkpoint, device)
    del unused_teacher
    conditioned_model = copy.deepcopy(base_model)
    stage_e = torch.load(args.stage_e_checkpoint, map_location=device, weights_only=False)
    if stage_e.get("schema") not in {
        "gaugeflow.stage_e_e0.v1",
        "gaugeflow.stage_e_e1.v1",
        "gaugeflow.stage_e_e2.v1",
        "gaugeflow.stage_e_e3.v1",
        "gaugeflow.stage_e_e3.v2",
    }:
        raise ValueError("Stage-E checkpoint schema is not a supported conditional arm")
    if stage_e.get("arm") not in {
        "orbit_mimic",
        "clean_side",
        "mixed_side",
        "centered_adapter",
        "adapter_trust_region",
    }:
        raise ValueError("Stage-E checkpoint is not a supported conditional arm")
    if stage_e.get("source_checkpoint_sha256") != sha256_file(args.stage_c_checkpoint):
        raise ValueError("Stage-E checkpoint source mismatch")
    if stage_e.get("schema") in {
        "gaugeflow.stage_e_e2.v1",
        "gaugeflow.stage_e_e3.v1",
        "gaugeflow.stage_e_e3.v2",
    }:
        conditioned_model.attach_tensor_residual_adapter()
    conditioned_model.load_state_dict(stage_e["model"], strict=True)
    if args.lattice_adapter is not None:
        # Attaching after the Stage-E load keeps the old checkpoint schema
        # strict-load compatible.  ``conditioned`` is the strict A/B/C mode:
        # A=Stage-C base, B=E3, C=E3+exposure.  ``both`` is retained only for
        # backwards-compatible historical diagnostics.
        if args.lattice_adapter_role == "both":
            _load_lattice_adapter(base_model, args.lattice_adapter, args.stage_c_checkpoint)
        _load_lattice_adapter(conditioned_model, args.lattice_adapter, args.stage_c_checkpoint)
    base_model.eval()
    conditioned_model.eval()
    composition_model = load_qualified_composition_model(
        args.composition_checkpoint, args.composition_protocol, device=device
    )
    samplers = {
        "base": _sampler(base_model, metadata, composition_model),
        "conditioned": _sampler(conditioned_model, metadata, composition_model),
    }
    source_count = int(validation.payload["source_index"].max()) + 1
    evaluator, _, _ = load_stage_d_model(
        args.stage_c_checkpoint,
        source_count=source_count,
        seed=int(protocol["seed"]),
        device=device,
    )
    stage_d = torch.load(args.stage_d_checkpoint, map_location=device, weights_only=False)
    evaluator.load_state_dict(stage_d["model"], strict=True)
    evaluator.eval()

    target_volume = (torch.linalg.det(target_batch.lattice) / target_batch.node_counts).cpu()
    target_distance = minimum_periodic_distances(
        target_batch.fractional_coordinates, target_batch.lattice, target_batch.batch
    ).cpu()
    arms = tuple(str(value) for value in protocol["arms"])
    roles = ("base", "conditioned", "swapped")
    trajectory_diagnostics: dict[str, dict[str, list[Any]]] = {
        arm: {role: [] for role in roles} for arm in arms
    }
    output: dict[str, Any] = {
        "schema": (
            "gaugeflow.stage_e_e1a_result.v2"
            if protocol_name == "stage_e_e1a_factorial_rollout_v2"
            else "gaugeflow.stage_e_e1a_result.v1"
        ),
        "protocol": protocol.get("protocol"),
        "protocol_sha256": sha256_file(args.protocol),
        "samples": len(records),
        "eligible_validation_samples": int(candidates.numel()),
        "lattice_adapter": (
            {
                "path": str(args.lattice_adapter),
                "sha256": sha256_file(args.lattice_adapter),
                "schema": "gaugeflow.stage_e_lattice_generated_exposure.v1",
                "role": args.lattice_adapter_role,
            }
            if args.lattice_adapter is not None
            else None
        ),
        "data_quality": {
            "exclude_pure_noble_gas": bool(data_quality.get("exclude_pure_noble_gas", False)),
            "excluded_validation_indices": excluded,
        },
        "arms": {},
    }
    for arm in arms:
        role_errors: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
        role_volumes: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
        role_distances: dict[str, list[torch.Tensor]] = {role: [] for role in roles}
        failures = {role: 0 for role in roles}
        errors: dict[str, list[str]] = {role: [] for role in roles}
        successes = {role: 0 for role in roles}
        for start in range(0, len(records), int(protocol["batch_size"])):
            stop = min(start + int(protocol["batch_size"]), len(records))
            counts = target_batch.node_counts[start:stop]
            source_index = target_batch.source_index[start:stop]
            target_for_role = target_normalized[start:stop]
            target_conditions = conditions[start:stop]
            swapped_conditions = target_conditions.roll(1, 0)
            node_counts_all = target_batch.batch.bincount()
            node_start = int(node_counts_all[:start].sum())
            node_stop = int(node_counts_all[:stop].sum())
            local_elements = target_batch.element_tokens[node_start:node_stop]
            local_batch = target_batch.batch[node_start:node_stop] - start
            for role in roles:
                condition = None if role == "base" else swapped_conditions if role == "swapped" else target_conditions
                sampler = samplers["base"] if role == "base" else samplers["conditioned"]
                seed = int(protocol["seed"]) + 10000 * (arms.index(arm) + 1) + 100 * start
                try:
                    tokens, coordinates, lattice = _run_arm(
                        sampler,
                        arm,
                        condition,
                        local_elements,
                        target_batch.lattice[start:stop],
                        local_batch,
                        counts,
                        steps=int(protocol["reverse_steps"]),
                        seed=seed,
                        continuous_mode=cast(
                            Literal["reverse_sde", "probability_flow"],
                            protocol["continuous_mode"],
                        ),
                        trace_lattice=(
                            args.trajectory_output is not None
                            and arm in {"oracle_ca", "oracle_c", "free"}
                        ),
                        diagnostics_sink=(
                            trajectory_diagnostics[arm][role]
                            if args.trajectory_output is not None
                            and arm in {"oracle_ca", "oracle_c", "free"}
                            else None
                        ),
                    )
                except (SamplingFailure, RuntimeError, ValueError, FloatingPointError) as error:
                    failures[role] += stop - start
                    role_errors[role].append(
                        torch.full((stop - start,), float("nan"), dtype=torch.float32)
                    )
                    if len(errors[role]) < 3:
                        errors[role].append(f"{type(error).__name__}: {error}")
                    continue
                generated_batch = ParentBlueprintBatch.from_node_counts(counts).batch
                prediction = evaluator(tokens, coordinates, lattice, generated_batch, source_index).piezoelectric
                role_errors[role].append(_orbit_error(prediction, target_for_role, rotations).cpu())
                successes[role] += stop - start
                determinant = torch.linalg.det(lattice)
                role_volumes[role].append((determinant / counts).cpu())
                role_distances[role].append(
                    minimum_periodic_distances(coordinates, lattice, generated_batch).cpu()
                )
        arm_result: dict[str, Any] = {"roles": {}}
        for role in roles:
            if successes[role] == 0:
                arm_result["roles"][role] = {
                    "sampling_failures": failures[role],
                    "errors": errors[role],
                }
                continue
            volume = torch.cat(role_volumes[role])
            distance = torch.cat(role_distances[role])
            arm_result["roles"][role] = {
                "sampling_failures": failures[role],
                "finite_positive_lattice_fraction": float(torch.isfinite(volume).float().mean()),
                "minimum_distance_fraction_at_0_5_angstrom": float((distance >= 0.5).float().mean()),
                "normalized_nearest_neighbor_wasserstein": float(
                    quantile_wasserstein(distance, target_distance, points=int(protocol["wasserstein_quantile_points"]))
                    / robust_scale(target_distance)
                ),
                "normalized_volume_wasserstein": float(
                    quantile_wasserstein(volume, target_volume, points=int(protocol["wasserstein_quantile_points"]))
                    / robust_scale(target_volume)
                ),
                "mean_tensor_orbit_error": float(torch.cat(role_errors[role]).nanmean()),
                "errors": errors[role],
            }
        if successes["base"] and successes["conditioned"]:
            base_error = torch.cat(role_errors["base"])
            conditioned_error = torch.cat(role_errors["conditioned"])
            paired = torch.isfinite(base_error) & torch.isfinite(conditioned_error)
            if bool(paired.any()):
                arm_result["paired_successes"] = int(paired.sum())
                arm_result["conditioned_minus_base_orbit_error_mean"] = float(
                    (conditioned_error[paired] - base_error[paired]).mean()
                )
        output["arms"][arm] = arm_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_schema = (
        "gaugeflow.stage_e_e1a_result.v2"
        if protocol_name == "stage_e_e1a_factorial_rollout_v2"
        else "gaugeflow.stage_e_e1a_result.v1"
    )
    if output["schema"] != expected_schema:
        raise AssertionError("factorial output schema does not match its protocol")
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.trajectory_output is not None:
        trajectory_protocol = dict(protocol)
        trajectory_protocol["_protocol_path"] = str(args.protocol)
        trajectory = _build_trajectory_report(
            trajectory_diagnostics,
            protocol=trajectory_protocol,
            samples=len(records),
        )
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        if args.trajectory_output.exists():
            raise FileExistsError(f"refusing to overwrite trajectory report: {args.trajectory_output}")
        args.trajectory_output.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
