"""Qualify the frozen GaugeFlow-base product-space runtime before A1 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.reverse_sampler import TensorFreeReverseSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--lattice-standardization", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _model_from_protocol(spec: dict[str, object], device: torch.device) -> HybridCrystalDenoiser:
    model = HybridCrystalDenoiser(
        hidden_dim=int(spec["hidden_dim"]),
        vector_dim=int(spec["vector_dim"]),
        layers=int(spec["layers"]),
        radial_dim=int(spec["radial_dim"]),
        radial_cutoff=float(spec["radial_cutoff_angstrom"]),
        atlas_residual_circle_samples=8,
        edge_dim=int(spec["edge_dim"]),
        angular_channels=int(spec["angular_channels"]),
        edge_refresh_rank=int(spec["edge_refresh_rank"]),
        modality_time_conditioning=str(spec["modality_time_conditioning"]),
    ).to(device)
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != int(spec["parameter_count"]):
        raise ValueError("integration model parameter count changed")
    return model


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite an integration result: {args.output}")
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "a1_product_space_integration_v2"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen A1 integration protocol")
    prerequisites = protocol["prerequisites"]
    runtime = protocol["runtime"]
    if sha256_file(Path(__file__)) != prerequisites["runner_sha256"]:
        raise ValueError("A1 integration runner changed after protocol freeze")
    if sha256_file(args.cache_root / "manifest.json") != prerequisites["cache_manifest_sha256"]:
        raise ValueError("A1 integration cache manifest changed")
    if canonical_json_hash(load_json_object(args.lattice_standardization)) != prerequisites[
        "lattice_standardization_canonical_sha256"
    ]:
        raise ValueError("A1 integration lattice standardization changed")
    if canonical_json_hash(load_json_object(args.composition_protocol)) != prerequisites[
        "composition_protocol_canonical_sha256"
    ]:
        raise ValueError("A1 integration composition protocol changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the production-size A1 integration Gate requires CUDA")
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    torch.set_float32_matmul_precision("high")

    composition_model = load_qualified_composition_model(
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
        expected_checkpoint_sha256=str(runtime["composition_law"]["checkpoint_sha256"]),
    )
    model = _model_from_protocol(protocol["model"], device)
    standardizer = P1LatticeStandardizer.from_mapping(load_json_object(args.lattice_standardization))
    dataset = PackedAlexP1Dataset(args.cache_root, "train")
    selected_rows: list[int] = []
    for row in range(int(protocol["smoke"]["scan_prefix_rows"])):
        start = int(dataset.offsets[row])
        stop = int(dataset.offsets[row + 1])
        species = torch.unique(dataset.atom_tokens[start:stop]).numel()
        if stop - start >= int(protocol["smoke"]["minimum_nodes"]) and species >= int(
            protocol["smoke"]["minimum_species"]
        ):
            selected_rows.append(row)
        if len(selected_rows) == int(protocol["smoke"]["graph_count"]):
            break
    if len(selected_rows) != int(protocol["smoke"]["graph_count"]):
        raise ValueError("frozen smoke prefix does not contain enough multi-species structures")
    smoke_indices = torch.tensor(selected_rows, dtype=torch.long)
    clean = dataset.select_model_batch(smoke_indices, device=device)
    node_counts = torch.bincount(clean.batch, minlength=smoke_indices.numel())
    blueprint = ParentBlueprintBatch.from_node_counts(node_counts, device=device)
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(protocol["training_path"]["coordinate_sigma_min"]),
        coordinate_sigma_max=float(protocol["training_path"]["coordinate_sigma_max"]),
        minimum_time=float(protocol["training_path"]["minimum_time"]),
        maximum_time=float(protocol["training_path"]["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_conditioning=True,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    generator = torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 1)
    smoke_time = torch.full(
        (smoke_indices.numel(),),
        float(protocol["smoke"]["joint_time"]),
        dtype=clean.lattice.dtype,
        device=device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = diffusion(
            clean.atom_types,
            clean.fractional_coordinates,
            clean.lattice,
            clean.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=smoke_time,
            element_time=smoke_time,
            lattice_time=smoke_time,
            generator=generator,
        )
    output.loss.backward()
    gradient_squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradient_squares:
        raise RuntimeError("product field produced no gradients")
    gradient_norm = torch.stack(gradient_squares).sum().sqrt()
    element_gradient_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in model.named_parameters()
        if name.startswith("element_head.") and parameter.grad is not None
    ]
    if not element_gradient_squares:
        raise RuntimeError("product field produced no assignment-head gradients")
    element_gradient_norm = torch.stack(element_gradient_squares).sum().sqrt()
    occupation = output.noisy.orderless_occupation
    if occupation is None:
        raise RuntimeError("product smoke did not construct an orderless occupation state")
    remaining_species = (occupation.remaining_counts > 0).sum(dim=1)

    sample_counts = torch.tensor(protocol["smoke"]["sample_node_counts"], device=device)
    sample_blueprint = ParentBlueprintBatch.from_node_counts(sample_counts, device=device)
    sampler = TensorFreeReverseSampler(
        model,
        standardizer,
        coordinate_sigma_min=float(protocol["training_path"]["coordinate_sigma_min"]),
        coordinate_sigma_max=float(protocol["training_path"]["coordinate_sigma_max"]),
        maximum_time=float(protocol["training_path"]["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )
    model.eval()
    sampled = sampler.sample(
        sample_blueprint,
        steps=int(protocol["smoke"]["reverse_steps"]),
        initialization_generator=torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 2),
        categorical_generator=torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 3),
        continuous_mode="probability_flow",
    )
    observed_counts = torch.bincount(
        sample_blueprint.batch * 118 + sampled.element_tokens,
        minlength=sample_counts.numel() * 118,
    ).reshape(sample_counts.numel(), 118)
    values = (
        output.loss,
        output.element_loss,
        output.coordinate_loss,
        output.volume_loss,
        output.shape_loss,
        gradient_norm,
        sampled.fractional_coordinates,
        sampled.lattice,
    )
    finite = all(bool(torch.isfinite(value).all()) for value in values)
    checks = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        == int(protocol["model"]["parameter_count"]),
        "finite_forward_backward": finite,
        "positive_gradient_norm": bool(gradient_norm > 0),
        "multispecies_assignment_support": bool(
            (remaining_species >= int(protocol["smoke"]["minimum_species"])).all()
        ),
        "positive_assignment_loss": bool(
            output.element_loss.detach() > float(protocol["acceptance"]["assignment_loss_min"])
        ),
        "positive_assignment_head_gradient": bool(
            element_gradient_norm > float(protocol["acceptance"]["assignment_gradient_norm_min"])
        ),
        "sampled_composition_exact": torch.equal(observed_counts, sampled.composition_counts),
        "sampled_node_count_exact": torch.equal(sampled.composition_counts.sum(dim=1), sample_counts),
        "every_step_composition_closure": bool((sampled.diagnostics.composition_closure_error == 0).all()),
        "terminal_remaining_atoms_zero": bool((sampled.diagnostics.remaining_atom_count[-1] == 0).all()),
        "terminal_masks_zero": int(sampled.diagnostics.masked_count[-1]) == 0,
        "lattice_positive_volume": bool((torch.linalg.det(sampled.lattice) > 0).all()),
    }
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": all(checks.values()),
        "checks": checks,
        "parameter_count": int(protocol["model"]["parameter_count"]),
        "smoke_cache_rows": smoke_indices.tolist(),
        "sample_node_counts": sample_counts.cpu().tolist(),
        "losses": {
            "total": float(output.loss.detach()),
            "assignment": float(output.element_loss.detach()),
            "coordinate": float(output.coordinate_loss.detach()),
            "volume": float(output.volume_loss.detach()),
            "shape": float(output.shape_loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "assignment_head_gradient_norm": float(element_gradient_norm.detach()),
        },
        "terminal_mask_count": int(sampled.diagnostics.masked_count[-1]),
        "maximum_composition_closure_error": int(sampled.diagnostics.composition_closure_error.max()),
        "composition_checkpoint_sha256": sha256_file(args.composition_checkpoint),
        "boundary": protocol["boundary"],
    }
    if not result["qualified"]:
        raise RuntimeError(f"A1 integration Gate failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
