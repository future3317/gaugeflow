"""Attribute a failed tangent checkpoint to its head or learned carrier span."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.runtime import load_tensor_free_ema_runtime
from gaugeflow.production.state_projection import fractional_tangent_to_cartesian


@dataclass(frozen=True)
class DesignBlock:
    design: torch.Tensor
    target: torch.Tensor
    graph_observations: int


@dataclass(frozen=True)
class DesignPanel:
    design: torch.Tensor
    target: torch.Tensor
    graph_observations: int
    blocks: dict[tuple[float, int], DesignBlock]
    reconstruction_max_abs: float
    tensor_candidates: int


def solve_minimum_norm(
    design: torch.Tensor,
    target: torch.Tensor,
    *,
    rcond: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve one global float64 readout and report its numerical span."""
    if design.ndim != 2 or target.shape != design.shape[:1]:
        raise ValueError("readout design and target have incompatible shapes")
    if design.dtype != torch.float64 or target.dtype != torch.float64:
        raise ValueError("readout solve requires float64 inputs")
    if not 0.0 < rcond < 1.0:
        raise ValueError("SVD rcond must lie in (0,1)")
    solution = torch.linalg.lstsq(
        design,
        target[:, None],
        rcond=rcond,
        driver="gelsd",
    )
    singular = solution.singular_values
    rank = int(solution.rank)
    retained = singular[:rank]
    condition = (
        float(retained[0] / retained[-1]) if retained.numel() else float("inf")
    )
    probabilities = retained.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1.0e-300)
    effective_rank = float(
        torch.exp(-(probabilities * probabilities.clamp_min(1.0e-300).log()).sum())
    )
    return solution.solution[:, 0], {
        "rank": rank,
        "condition_number": condition,
        "effective_rank": effective_rank,
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(retained[-1]),
    }


def evaluate_readout(
    panel: DesignBlock | DesignPanel,
    weight: torch.Tensor,
) -> dict[str, float]:
    """Evaluate the graph-equal production quadratic for one fixed readout."""
    residual = panel.design @ weight - panel.target
    denominator = float(panel.graph_observations)
    loss = float(residual.square().sum() / denominator)
    target_energy = float(panel.target.square().sum() / denominator)
    return {
        "loss": loss,
        "target_energy": target_energy,
        "explained_fraction": 1.0 - loss / max(target_energy, 1.0e-300),
    }


def classify_attribution(
    *,
    train_relative: float,
    validation_relative: float,
    validation_oracle_explained: float,
    thresholds: dict[str, Any],
) -> str:
    """Apply the preregistered causal decision order."""
    train_bound = float(thresholds["generalizable_head_train_relative_max"])
    validation_bound = float(
        thresholds["generalizable_head_validation_relative_max"]
    )
    oracle_bound = float(
        thresholds["validation_oracle_explained_fraction_min"]
    )
    if train_relative <= train_bound and validation_relative <= validation_bound:
        return "head_optimization_limited"
    if validation_oracle_explained < oracle_bound:
        return "backbone_span_limited"
    if train_relative <= train_bound and validation_relative > validation_bound:
        return "split_specific_readout"
    return "distributed_nonlinear_optimization"


def tensor_candidate_count(gauge_atlas: Any) -> int:
    """Use the production null-condition accounting contract."""
    return int(gauge_atlas.effective_frame_count.sum())


@torch.no_grad()
def collect_design_panel(
    runtime: Any,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    specification: dict[str, Any],
    *,
    split: str,
    device: torch.device,
) -> DesignPanel:
    """Capture the centered carrier and exact tangent target on a fixed panel."""
    model = runtime.model
    model.eval()
    diffusion = TensorFreeHybridDiffusion(
        model,
        runtime.lattice_standardizer,
        coordinate_sigma_min=float(runtime.training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(runtime.training_config["coordinate_sigma_max"]),
        minimum_time=float(runtime.training_config["minimum_time"]),
        maximum_time=float(runtime.training_config["maximum_time"]),
    )
    noise_seed = int(specification[f"{split}_noise_seed"])
    blocks: dict[tuple[float, int], DesignBlock] = {}
    maximum_reconstruction = 0.0
    tensor_candidates = 0
    captured: list[torch.Tensor] = []

    def capture_carrier(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output.detach())

    handle = model.coordinate_carrier.register_forward_hook(capture_carrier)
    try:
        for time_offset, time_value_raw in enumerate(specification["times"]):
            time_value = float(time_value_raw)
            for replicate in range(int(specification["noise_replicates"])):
                design_parts: list[torch.Tensor] = []
                target_parts: list[torch.Tensor] = []
                graph_observations = 0
                generator = torch.Generator(device=device).manual_seed(
                    noise_seed + 10_000 * time_offset + replicate
                )
                for start in range(0, indices.numel(), int(specification["batch_size"])):
                    selected = indices[start : start + int(specification["batch_size"])]
                    packed = Batch.from_data_list(
                        [dataset[int(index)] for index in selected]
                    ).to(device)
                    graphs = int(packed.num_graphs)
                    counts = torch.bincount(packed.batch, minlength=graphs)
                    blueprint = ParentBlueprintBatch.from_node_counts(
                        counts, dtype=packed.frac_coords.dtype, device=device
                    )
                    time = packed.lattice.new_full((graphs,), time_value)
                    noisy = diffusion.noise_clean_batch(
                        packed.atom_types,
                        packed.frac_coords,
                        packed.lattice,
                        packed.batch,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                        time=time,
                        generator=generator,
                    )
                    condition = time.new_zeros((graphs, 18))
                    present = torch.zeros(
                        (graphs, 1), dtype=torch.bool, device=device
                    )
                    captured.clear()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=runtime.training_config["precision"] == "bf16",
                    ):
                        prediction = model(
                            noisy.element_tokens,
                            noisy.fractional_coordinates,
                            noisy.log_volume,
                            noisy.log_shape,
                            packed.batch,
                            time,
                            condition,
                            present,
                            blueprint.shape_projector,
                            blueprint.fractional_to_cartesian,
                        )
                    if len(captured) != 1:
                        raise RuntimeError("coordinate carrier hook did not fire exactly once")
                    carrier = captured[0].float()
                    reconstructed = torch.einsum(
                        "nci,c->ni",
                        carrier,
                        model.coordinate_carrier_head.weight[0].float(),
                    )
                    maximum_reconstruction = max(
                        maximum_reconstruction,
                        float(
                            (
                                reconstructed
                                - prediction.coordinate_cartesian_scaled_score.float()
                            )
                            .abs()
                            .max()
                        ),
                    )
                    noisy_lattice = LatticeVolumeShape(
                        noisy.log_volume.float(), noisy.log_shape.float()
                    ).lattice(blueprint.fractional_to_cartesian.float())
                    target = fractional_tangent_to_cartesian(
                        noisy.coordinate_scaled_score_target.float(),
                        noisy_lattice,
                        packed.batch,
                    )
                    row_weight = (
                        3.0 * counts[packed.batch].to(carrier)
                    ).rsqrt()
                    design = carrier.permute(0, 2, 1).reshape(
                        -1, carrier.shape[1]
                    )
                    target_flat = target.reshape(-1)
                    component_weight = row_weight[:, None].expand(-1, 3).reshape(-1)
                    design_parts.append(
                        (design * component_weight[:, None]).double().cpu()
                    )
                    target_parts.append((target_flat * component_weight).double().cpu())
                    graph_observations += graphs
                    tensor_candidates += tensor_candidate_count(
                        prediction.gauge_atlas
                    )
                blocks[(time_value, replicate)] = DesignBlock(
                    design=torch.cat(design_parts),
                    target=torch.cat(target_parts),
                    graph_observations=graph_observations,
                )
    finally:
        handle.remove()
    return DesignPanel(
        design=torch.cat([block.design for block in blocks.values()]),
        target=torch.cat([block.target for block in blocks.values()]),
        graph_observations=sum(block.graph_observations for block in blocks.values()),
        blocks=blocks,
        reconstruction_max_abs=maximum_reconstruction,
        tensor_candidates=tensor_candidates,
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_cartesian_tangent_readout_span_audit_v1":
        raise ValueError("tangent readout-span protocol mismatch")
    source_protocol = load_json_object(args.source_protocol)
    prerequisites = protocol["prerequisites"]
    if (
        source_protocol.get("protocol") != prerequisites["source_protocol"]
        or canonical_json_hash(source_protocol)
        != prerequisites["source_protocol_sha256"]
        or sha256_file(Path(prerequisites["source_result"]))
        != prerequisites["source_result_sha256"]
        or sha256_file(args.cache_root / "manifest.json")
        != prerequisites["cache_manifest_sha256"]
    ):
        raise ValueError("tangent readout-span prerequisite mismatch")
    specification = protocol["audit"]
    if int(specification["optimizer_steps"]) != 0:
        raise ValueError("tangent readout-span audit forbids optimizer steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("tangent readout-span audit requires CUDA")
    datasets = {
        "train": PackedAlexP1Dataset(args.cache_root, "train"),
        "validation": PackedAlexP1Dataset(args.cache_root, "val"),
    }
    indices = {
        split: torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(
                int(specification[f"{split}_selection_seed"])
            ),
        )[: int(specification[f"{split}_graphs"])]
        for split, dataset in datasets.items()
    }
    checkpoint_results: dict[str, Any] = {}
    final_panels: dict[str, DesignPanel] | None = None
    final_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    maximum_reconstruction = 0.0
    total_tensor_candidates = 0
    parameters_unchanged = True
    rcond = float(specification["svd_rcond"])
    for step_raw in prerequisites["checkpoint_steps"]:
        step = int(step_raw)
        checkpoint = (
            args.run_root
            / f"seed_{int(prerequisites['source_seed'])}"
            / f"checkpoint_step_{step:08d}.pt"
        )
        runtime = load_tensor_free_ema_runtime(
            checkpoint,
            device,
            protocol_name=str(source_protocol["protocol"]),
            protocol_sha256=str(prerequisites["source_protocol_sha256"]),
        )
        before = {
            name: value.detach().clone()
            for name, value in runtime.model.state_dict().items()
        }
        panels = {
            split: collect_design_panel(
                runtime,
                datasets[split],
                indices[split],
                specification,
                split=split,
                device=device,
            )
            for split in ("train", "validation")
        }
        train_weight, train_span = solve_minimum_norm(
            panels["train"].design,
            panels["train"].target,
            rcond=rcond,
        )
        validation_weight, validation_span = solve_minimum_norm(
            panels["validation"].design,
            panels["validation"].target,
            rcond=rcond,
        )
        current_weight = (
            runtime.model.coordinate_carrier_head.weight[0].detach().double().cpu()
        )
        current_train = evaluate_readout(panels["train"], current_weight)
        current_validation = evaluate_readout(
            panels["validation"], current_weight
        )
        train_optimum_train = evaluate_readout(panels["train"], train_weight)
        train_optimum_validation = evaluate_readout(
            panels["validation"], train_weight
        )
        validation_oracle = evaluate_readout(
            panels["validation"], validation_weight
        )
        checkpoint_results[str(step)] = {
            "current_train": current_train,
            "current_validation": current_validation,
            "train_optimum_train": train_optimum_train,
            "train_optimum_validation": train_optimum_validation,
            "validation_oracle": validation_oracle,
            "train_optimum_train_relative_to_current": (
                train_optimum_train["loss"] / current_train["loss"]
            ),
            "train_optimum_validation_relative_to_current": (
                train_optimum_validation["loss"] / current_validation["loss"]
            ),
            "train_span": train_span,
            "validation_span": validation_span,
            "current_head_norm": float(torch.linalg.vector_norm(current_weight)),
            "train_optimum_head_norm": float(torch.linalg.vector_norm(train_weight)),
            "validation_oracle_head_norm": float(
                torch.linalg.vector_norm(validation_weight)
            ),
            "current_to_train_optimum_distance": float(
                torch.linalg.vector_norm(current_weight - train_weight)
            ),
        }
        maximum_reconstruction = max(
            maximum_reconstruction,
            *(panel.reconstruction_max_abs for panel in panels.values()),
        )
        total_tensor_candidates += sum(
            panel.tensor_candidates for panel in panels.values()
        )
        parameters_unchanged = parameters_unchanged and all(
            torch.equal(value, before[name])
            for name, value in runtime.model.state_dict().items()
        )
        if step == int(prerequisites["checkpoint_steps"][-1]):
            final_panels = panels
            final_weights = (train_weight, validation_weight)
    if final_panels is None or final_weights is None:
        raise RuntimeError("final tangent checkpoint was not audited")
    final = checkpoint_results[str(prerequisites["checkpoint_steps"][-1])]
    train_weight, validation_weight = final_weights
    current_runtime = load_tensor_free_ema_runtime(
        args.run_root
        / f"seed_{int(prerequisites['source_seed'])}"
        / f"checkpoint_step_{int(prerequisites['checkpoint_steps'][-1]):08d}.pt",
        device,
        protocol_name=str(source_protocol["protocol"]),
        protocol_sha256=str(prerequisites["source_protocol_sha256"]),
    )
    current_weight = (
        current_runtime.model.coordinate_carrier_head.weight[0].detach().double().cpu()
    )
    time_resolved: list[dict[str, Any]] = []
    for time_value in specification["times"]:
        replicate_rows = []
        for replicate in range(int(specification["noise_replicates"])):
            block = final_panels["validation"].blocks[(float(time_value), replicate)]
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "current": evaluate_readout(block, current_weight),
                    "train_optimum": evaluate_readout(block, train_weight),
                    "validation_oracle": evaluate_readout(
                        block, validation_weight
                    ),
                }
            )
        time_resolved.append(
            {"time": float(time_value), "replicates": replicate_rows}
        )
    initial_oracle = checkpoint_results["0"]["validation_oracle"][
        "explained_fraction"
    ]
    final_oracle = final["validation_oracle"]["explained_fraction"]
    classification = classify_attribution(
        train_relative=final["train_optimum_train_relative_to_current"],
        validation_relative=final[
            "train_optimum_validation_relative_to_current"
        ],
        validation_oracle_explained=final_oracle,
        thresholds=protocol["classification"],
    )
    result: dict[str, Any] = {
        "protocol": protocol["protocol"],
        "source_protocol": source_protocol["protocol"],
        "checkpoint_results": checkpoint_results,
        "final_time_resolved_validation": time_resolved,
        "validation_oracle_explained_gain_step0_to_final": (
            final_oracle - initial_oracle
        ),
        "classification": classification,
        "carrier_head_reconstruction_max_abs": maximum_reconstruction,
        "tensor_candidates": total_tensor_candidates,
        "parameters_unchanged": parameters_unchanged,
        "optimizer_steps": 0,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    qualification = protocol["qualification"]
    result["checks"] = {
        "carrier_head_reconstruction": maximum_reconstruction
        <= float(qualification["carrier_head_reconstruction_max_abs"]),
        "minimum_design_rank": min(
            int(row[key]["rank"])
            for row in checkpoint_results.values()
            for key in ("train_span", "validation_span")
        )
        >= int(qualification["minimum_design_rank"]),
        "tensor_candidates": total_tensor_candidates
        == int(qualification["tensor_candidates"]),
        "parameters_unchanged": parameters_unchanged,
        "optimizer_steps": int(qualification["optimizer_steps"]) == 0,
    }
    result["checks"]["finite_all_metrics"] = _finite_tree(result)
    result["qualified_diagnostic"] = all(result["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
