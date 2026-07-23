"""Train a zero-initialized lattice adapter on generated-side exposure states.

The protocol uses one detached reverse VP step from a matched noisy lattice
state.  It is deliberately not terminal volume clipping: invalid/non-finite
states fail the run, while the adapter is trained against the exact clean
standardized lattice target.  Composition counts, the P1 shape projector and
the quotient lattice chart are inherited from the frozen Stage-C contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from train_stage_e_orbit_mimic import _load_backbones

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.continued_pretraining import collate_structure_records
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import project_lattice_state
from gaugeflow.production.lemat_index import IndexedLeMatDataset
from gaugeflow.production.orderless_product_state import partial_occupation_from_reveal_rank
from gaugeflow.production.physical_checkpointing import read_physical_checkpoint_metadata
from gaugeflow.production.response_data import StageDResponseDataset, collate_response_records
from gaugeflow.production.reverse_sampler import sample_uniform_reveal_ranks, vp_reverse_step
from gaugeflow.tensor import piezo_to_irreps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lemat-index", type=Path, default=None)
    parser.add_argument("--response-cache", type=Path, default=None)
    parser.add_argument("--normalizer", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-e-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _next_indices(
    count: int,
    batch_size: int,
    permutation: torch.Tensor,
    cursor: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pieces: list[torch.Tensor] = []
    remaining = batch_size
    while remaining:
        take = min(remaining, count - cursor)
        pieces.append(permutation[cursor : cursor + take])
        cursor += take
        remaining -= take
        if cursor == count:
            permutation = torch.randperm(count, generator=generator)
            cursor = 0
    return torch.cat(pieces), permutation, cursor


def _adapter_parameters(model: Any) -> list[torch.nn.Parameter]:
    adapter = model.lattice_residual_adapter
    if adapter is None:
        raise RuntimeError("lattice residual adapter was not attached")
    return [parameter for parameter in adapter.parameters() if parameter.requires_grad]


def _exact_composition_counts(
    element_tokens: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    vocabulary_size: int,
) -> torch.Tensor:
    flat = batch * vocabulary_size + element_tokens
    return torch.bincount(
        flat,
        minlength=graph_count * vocabulary_size,
    ).reshape(graph_count, vocabulary_size)


def _validate_element_exposure(value: str) -> str:
    if value not in {"clean", "orderless_partial"}:
        raise ValueError("element_exposure must be 'clean' or 'orderless_partial'")
    return value


def _orderless_partial_tokens(
    clean_tokens: torch.Tensor,
    batch: torch.Tensor,
    composition_counts: torch.Tensor,
    node_counts: torch.Tensor,
    time: torch.Tensor,
    reveal_rank: torch.Tensor,
    *,
    vocabulary_size: int,
    mask_token: int,
) -> torch.Tensor:
    reveal_count = torch.floor((1.0 - time) * node_counts.to(time)).long()
    reveal_count = torch.minimum(reveal_count, node_counts.long() - 1)
    return partial_occupation_from_reveal_rank(
        clean_tokens,
        batch,
        composition_counts,
        reveal_rank,
        reveal_count,
        vocabulary_size=vocabulary_size,
        mask_token=mask_token,
    ).partial_tokens


def _generated_exposure_loss(
    diffusion: TensorFreeHybridDiffusion,
    clean: Any,
    *,
    exposure_time: float,
    exposure_delta: float,
    generator: torch.Generator,
    precision: str,
    element_exposure: str = "clean",
    tensor_condition: torch.Tensor | None = None,
    condition_present: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    blueprint = ParentBlueprintBatch.from_node_counts(
        clean.node_counts,
        dtype=clean.lattice.dtype,
        device=clean.lattice.device,
    )
    graphs = clean.node_counts.numel()
    time_from = torch.full(
        (graphs,), exposure_time, dtype=clean.lattice.dtype, device=clean.lattice.device
    )
    time_to = torch.full(
        (graphs,), max(0.0, exposure_time - exposure_delta),
        dtype=clean.lattice.dtype,
        device=clean.lattice.device,
    )
    composition_counts = _exact_composition_counts(
        clean.element_tokens,
        clean.batch,
        graphs,
        diffusion.categorical.element_count,
    )
    element_exposure = _validate_element_exposure(element_exposure)
    first_element_tokens = clean.element_tokens
    exposed_element_tokens = clean.element_tokens
    if element_exposure == "orderless_partial":
        reveal_rank = sample_uniform_reveal_ranks(clean.batch, generator=generator)
        first_element_tokens = _orderless_partial_tokens(
            clean.element_tokens,
            clean.batch,
            composition_counts,
            clean.node_counts,
            time_from,
            reveal_rank,
            vocabulary_size=diffusion.categorical.element_count,
            mask_token=diffusion.categorical.mask_index,
        )
        exposed_element_tokens = _orderless_partial_tokens(
            clean.element_tokens,
            clean.batch,
            composition_counts,
            clean.node_counts,
            time_to,
            reveal_rank,
            vocabulary_size=diffusion.categorical.element_count,
            mask_token=diffusion.categorical.mask_index,
        )
    # Generate the exposure carrier without backpropagating through its
    # stochastic reverse transition.  The adapter still receives gradients
    # from the denoising query evaluated on that carrier.
    with torch.no_grad():
        noisy = diffusion.noise_lattice_batch(
            clean.element_tokens,
            clean.lattice,
            clean.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            lattice_time=time_from,
            generator=generator,
        )
        noisy_volume = diffusion.lattice_standardizer.encode_volume(
            noisy.log_volume, clean.node_counts
        )
        noisy_shape = diffusion.lattice_standardizer.encode_shape(noisy.log_shape)
        first = diffusion.denoiser.forward_lattice(
            first_element_tokens,
            noisy.log_volume,
            noisy.log_shape,
            clean.batch,
            time_from,
            blueprint.shape_projector,
            composition_counts=composition_counts,
            tensor_condition=tensor_condition,
            condition_present=condition_present,
        )
        generated_volume = vp_reverse_step(
            diffusion.vp_schedule,
            noisy_volume,
            first.clean_volume_latent,
            time_from,
            time_to,
            generator=generator,
            mode="reverse_sde",
        )
        generated_shape = vp_reverse_step(
            diffusion.vp_schedule,
            noisy_shape,
            first.clean_shape_latent,
            time_from[:, None],
            time_to[:, None],
            generator=generator,
            mode="reverse_sde",
        )
        generated_log_volume = diffusion.lattice_standardizer.decode_volume(
            generated_volume, clean.node_counts
        )
        generated_log_shape = project_lattice_state(
            diffusion.lattice_standardizer.decode_shape(generated_shape),
            blueprint.shape_projector,
        )
        if not torch.isfinite(generated_log_volume).all() or not torch.isfinite(generated_log_shape).all():
            raise FloatingPointError("generated lattice exposure produced a non-finite state")

    with torch.autocast(
        device_type=clean.lattice.device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and clean.lattice.device.type == "cuda",
    ):
        exposed = diffusion.denoiser.forward_lattice(
            exposed_element_tokens,
            generated_log_volume,
            generated_log_shape,
            clean.batch,
            time_to,
            blueprint.shape_projector,
            composition_counts=composition_counts,
            tensor_condition=tensor_condition,
            condition_present=condition_present,
        )
        target_volume = noisy.clean_volume_latent_target
        target_shape = noisy.clean_shape_latent_target
        generated_volume_loss = F.smooth_l1_loss(
            exposed.clean_volume_latent.float(), target_volume.float(), beta=1.0
        )
        generated_shape_loss = F.smooth_l1_loss(
            exposed.clean_shape_latent.float(), target_shape.float(), beta=1.0
        )
        # Retain the ordinary denoising interface on the same batch.  This
        # protects the qualified clean-side field while the adapter learns
        # the detached generated-side carrier.
        clean_prediction = diffusion.denoiser.forward_lattice(
            first_element_tokens,
            noisy.log_volume,
            noisy.log_shape,
            clean.batch,
            time_from,
            blueprint.shape_projector,
            composition_counts=composition_counts,
            tensor_condition=tensor_condition,
            condition_present=condition_present,
        )
        clean_volume_loss = F.smooth_l1_loss(
            clean_prediction.clean_volume_latent.float(), target_volume.float(), beta=1.0
        )
        clean_shape_loss = F.smooth_l1_loss(
            clean_prediction.clean_shape_latent.float(), target_shape.float(), beta=1.0
        )
        loss = generated_volume_loss + generated_shape_loss + 0.25 * (
            clean_volume_loss + clean_shape_loss
        )
    metrics = {
        "generated_volume_loss": float(generated_volume_loss.detach()),
        "generated_shape_loss": float(generated_shape_loss.detach()),
        "clean_volume_loss": float(clean_volume_loss.detach()),
        "clean_shape_loss": float(clean_shape_loss.detach()),
        "generated_log_volume_abs_max": float(generated_log_volume.abs().max().detach()),
        "first_mask_fraction": float((first_element_tokens == diffusion.categorical.mask_index).float().mean()),
        "exposed_mask_fraction": float((exposed_element_tokens == diffusion.categorical.mask_index).float().mean()),
    }
    return loss, metrics


def main() -> None:
    args = parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "stage_e_lattice_generated_exposure_v1":
        raise ValueError("unexpected lattice generated-exposure protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("lattice generated exposure requires CUDA")
    if (args.lemat_index is None) == (args.response_cache is None):
        raise ValueError("choose exactly one of --lemat-index and --response-cache")
    if args.response_cache is not None and args.normalizer is None:
        raise ValueError("--normalizer is required with --response-cache")
    metadata = read_physical_checkpoint_metadata(args.checkpoint)
    model, _, _, stage_metadata = _load_backbones(args.checkpoint, device)
    if metadata != stage_metadata:
        raise AssertionError("checkpoint metadata changed while loading")
    if args.stage_e_checkpoint is not None:
        stage_e = torch.load(args.stage_e_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(stage_e, dict) or stage_e.get("schema") not in {
            "gaugeflow.stage_e_e2.v1",
            "gaugeflow.stage_e_e3.v1",
            "gaugeflow.stage_e_e3.v2",
        }:
            raise ValueError("generated-exposure Stage-E checkpoint has an unsupported schema")
        if stage_e.get("source_checkpoint_sha256") != sha256_file(args.checkpoint):
            raise ValueError("Stage-E checkpoint source mismatch")
        model.attach_tensor_residual_adapter()
        model.load_state_dict(stage_e["model"], strict=True)
    model.attach_lattice_residual_adapter()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.lattice_residual_adapter is not None
    for parameter in model.lattice_residual_adapter.parameters():
        parameter.requires_grad_(True)
    model.train()
    standardizer = P1LatticeStandardizer.from_mapping(
        stage_metadata["stage_b_metadata"]["lattice_standardization"]
    ).to(device)
    training = stage_metadata["stage_b_metadata"]["a1_training_config"]
    diffusion = TensorFreeHybridDiffusion(
        model,
        standardizer,
        coordinate_sigma_min=float(training["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
        categorical_path="absorbing_mask",
    )
    normalizer = None
    if args.response_cache is not None:
        dataset = StageDResponseDataset(args.response_cache, "train")
        from gaugeflow.production.response_normalization import load_response_normalizer

        normalizer = load_response_normalizer(
            args.normalizer,
            expected_cache_sha256=str(dataset.manifest["cache_sha256"]),
        ).to(device)

        def collate(selected: torch.Tensor) -> Any:
            return collate_response_records([dataset[int(index)] for index in selected])

    else:
        dataset = IndexedLeMatDataset(args.lemat_index, "train", verify_hashes=True)

        def collate(selected: torch.Tensor) -> Any:
            return collate_structure_records(dataset.select(selected))
    batch_size = int(protocol["batch_size"])
    steps = int(protocol["steps"])
    element_exposure = _validate_element_exposure(str(protocol.get("element_exposure", "clean")))
    if batch_size < 1 or steps < 1:
        raise ValueError("lattice exposure batch size and steps must be positive")
    optimizer = torch.optim.AdamW(
        _adapter_parameters(model),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol.get("weight_decay", 0.0)),
    )
    index_generator = torch.Generator().manual_seed(int(protocol["seed"]))
    noise_generator = torch.Generator(device=device).manual_seed(int(protocol["seed"]) + 1)
    permutation = torch.randperm(len(dataset), generator=index_generator)
    cursor = 0
    history: list[dict[str, float]] = []
    for step in range(steps):
        selected, permutation, cursor = _next_indices(
            len(dataset), batch_size, permutation, cursor, index_generator
        )
        clean = collate(selected).to(device)
        tensor_condition = None
        condition_present = None
        if normalizer is not None:
            normalized = normalizer.normalize_piezoelectric(
                clean.targets.piezoelectric,
                clean.source_index,
            )
            tensor_condition = piezo_to_irreps(normalized)
            condition_present = clean.targets.piezoelectric_mask[:, None]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _generated_exposure_loss(
            diffusion,
            clean,
            exposure_time=float(protocol["exposure_time"]),
            exposure_delta=float(protocol["exposure_delta"]),
            generator=noise_generator,
            precision=str(protocol.get("precision", "bf16")),
            element_exposure=element_exposure,
            tensor_condition=tensor_condition,
            condition_present=condition_present,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite lattice exposure loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(_adapter_parameters(model), 1.0)
        optimizer.step()
        metrics.update({"step": float(step + 1), "loss": float(loss.detach()), "gradient_norm": float(gradient_norm)})
        history.append(metrics)
    payload = {
        "schema": "gaugeflow.stage_e_lattice_generated_exposure.v1",
        "protocol": protocol,
        "checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_kind": "response_cache" if args.response_cache is not None else "lemat",
        "lemat_index": str(args.lemat_index) if args.lemat_index is not None else None,
        "lemat_manifest_sha256": (
            sha256_file(args.lemat_index / "manifest.json")
            if args.lemat_index is not None
            else None
        ),
        "response_cache": str(args.response_cache) if args.response_cache is not None else None,
        "response_cache_sha256": (
            str(dataset.manifest["cache_sha256"])
            if args.response_cache is not None
            else None
        ),
        "stage_e_checkpoint": str(args.stage_e_checkpoint) if args.stage_e_checkpoint is not None else None,
        "adapter": {
            name: value.detach().cpu().clone()
            for name, value in model.lattice_residual_adapter.state_dict().items()
        },
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "schema": payload["schema"],
        "steps": steps,
        "trainable_parameters": sum(p.numel() for p in _adapter_parameters(model)),
        "initial_exact_zero": True,
        "final": history[-1],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_kind": payload["source_kind"],
        "element_exposure": element_exposure,
        "lemat_manifest_sha256": payload["lemat_manifest_sha256"],
        "response_cache_sha256": payload["response_cache_sha256"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
