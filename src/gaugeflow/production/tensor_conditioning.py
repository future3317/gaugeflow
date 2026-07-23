"""Typed Stage-E losses for tensor-orbit conditioned generation.

The tensor condition is an SO(3) orbit. Two representatives must therefore be
compared after the Cartesian atlas has marginalized its candidate measure,
never by assuming that finite candidate indices agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from gaugeflow.tensor import (
    fixed_lossless_response_probes,
    piezo_from_irreps,
    piezo_to_irreps,
    response_field,
    rotate_rank3,
)

from .equivariant_denoiser import HybridCrystalDenoiser, HybridDenoiserOutput
from .hybrid_diffusion import (
    HybridLossOutput,
    TensorFreeHybridDiffusion,
    TensorFreeNoisyBatch,
)
from .state_projection import graph_mean


@dataclass(frozen=True)
class TypedFieldWeights:
    """Weights for heterogeneous reverse-field distances."""

    assignment: float = 1.0
    composition: float = 0.25
    coordinate: float = 1.0
    volume: float = 0.25
    shape: float = 0.25
    response: float = 0.25

    def validate(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(value < 0.0 for value in values) or not any(value > 0.0 for value in values):
            raise ValueError("typed field weights must be nonnegative and not all zero")


@dataclass(frozen=True)
class TypedFieldDistance:
    loss: torch.Tensor
    assignment: torch.Tensor
    composition: torch.Tensor
    coordinate: torch.Tensor
    volume: torch.Tensor
    shape: torch.Tensor
    response: torch.Tensor


@dataclass(frozen=True)
class TensorConditioningTrainingOutput:
    """One common-noise Stage-E objective evaluation."""

    loss: torch.Tensor
    fine: torch.Tensor
    first_fine: HybridLossOutput
    rotated_fine: HybridLossOutput
    orbit_mimic: TypedFieldDistance
    null_retention: TypedFieldDistance


def sample_proper_rotations(
    count: int,
    reference: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Haar SO(3) matrices by QR with an explicit proper orientation."""

    if count < 1 or not reference.dtype.is_floating_point:
        raise ValueError("SO(3) sampling requires a positive count and floating reference")
    raw = torch.randn(
        (count, 3, 3),
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    orthogonal, triangular = torch.linalg.qr(raw)
    diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    orthogonal = orthogonal * signs.unsqueeze(-2)
    determinant = torch.linalg.det(orthogonal)
    orthogonal[:, :, -1] *= determinant.unsqueeze(-1)
    return orthogonal


def rotate_orbit_representative(
    piezo_irreps: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply one proper Cartesian rotation per tensor-orbit representative."""

    if piezo_irreps.ndim != 2 or piezo_irreps.shape[-1] != 18:
        raise ValueError("piezoelectric condition must have shape [graphs,18]")
    if rotation.shape != (piezo_irreps.shape[0], 3, 3):
        raise ValueError("representative rotation must have shape [graphs,3,3]")
    return piezo_to_irreps(
        rotate_rank3(piezo_from_irreps(piezo_irreps), rotation).contiguous()
    )


def _categorical_js(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    if first_logits.shape != second_logits.shape or first_logits.ndim != 2:
        raise ValueError("categorical fields must be matching matrices")
    first_log = functional.log_softmax(first_logits.float(), dim=-1)
    second_log = functional.log_softmax(second_logits.float(), dim=-1)
    first = first_log.exp()
    second = second_log.exp()
    mixture_log = torch.logaddexp(first_log, second_log) - first_log.new_tensor(2.0).log()
    divergence = 0.5 * (
        (first * (first_log - mixture_log)).sum(dim=-1)
        + (second * (second_log - mixture_log)).sum(dim=-1)
    )
    return divergence.clamp_min(0.0)


def _mean_square(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.shape[0] < 1:
        raise ValueError("typed continuous fields must have matching nonempty leading axes")
    return (first.float() - second.float()).flatten(1).square().mean(dim=-1)


def typed_field_distance(
    first: HybridDenoiserOutput,
    second: HybridDenoiserOutput,
    batch: torch.Tensor,
    graph_count: int,
    *,
    weights: TypedFieldWeights = TypedFieldWeights(),
    include_response: bool = True,
    probes: torch.Tensor | None = None,
) -> TypedFieldDistance:
    """Compare marginalized heterogeneous outputs in their physical types.

    Candidate posterior arrays are deliberately absent: a change of orbit
    representative can permute or resize the finite atlas support.
    """

    weights.validate()
    if batch.ndim != 1 or batch.dtype != torch.long or graph_count < 1:
        raise ValueError("typed distance requires a packed integer graph batch")
    if batch.numel() != first.clean_element_logits.shape[0] or (
        batch.numel() and (int(batch.min()) != 0 or int(batch.max()) + 1 != graph_count)
    ):
        raise ValueError("typed distance batch does not cover its output graphs")
    assignment = graph_mean(
        _categorical_js(first.clean_element_logits, second.clean_element_logits).unsqueeze(-1),
        batch,
        graph_count,
    ).mean()
    composition = _categorical_js(
        first.clean_composition_logits,
        second.clean_composition_logits,
    ).mean()
    coordinate = graph_mean(
        _mean_square(
            first.coordinate_cartesian_scaled_score,
            second.coordinate_cartesian_scaled_score,
        ).unsqueeze(-1),
        batch,
        graph_count,
    ).mean()
    volume = _mean_square(
        first.clean_volume_latent.unsqueeze(-1),
        second.clean_volume_latent.unsqueeze(-1),
    ).mean()
    shape = _mean_square(first.clean_shape_latent, second.clean_shape_latent).mean()
    if include_response:
        directions = (
            fixed_lossless_response_probes(
                dtype=first.gauge_atlas.aligned_tensor.dtype
            ).to(first.gauge_atlas.aligned_tensor.device)
            if probes is None
            else probes.to(first.gauge_atlas.aligned_tensor)
        )
        first_response = response_field(
            first.gauge_atlas.aligned_tensor.unsqueeze(1),
            directions.unsqueeze(0),
        )
        second_response = response_field(
            second.gauge_atlas.aligned_tensor.unsqueeze(1),
            directions.unsqueeze(0),
        )
        response = _mean_square(first_response, second_response).mean()
    else:
        response = assignment.new_zeros(())
    loss = (
        weights.assignment * assignment
        + weights.composition * composition
        + weights.coordinate * coordinate
        + weights.volume * volume
        + weights.shape * shape
        + (weights.response * response if include_response else 0.0)
    )
    return TypedFieldDistance(
        loss=loss,
        assignment=assignment,
        composition=composition,
        coordinate=coordinate,
        volume=volume,
        shape=shape,
        response=response,
    )


def orbit_mimic_distance(
    first: HybridDenoiserOutput,
    rotated_representative: HybridDenoiserOutput,
    batch: torch.Tensor,
    graph_count: int,
    *,
    weights: TypedFieldWeights = TypedFieldWeights(),
) -> TypedFieldDistance:
    """Same-state, same-noise loss between two representatives of one orbit."""

    return typed_field_distance(
        first,
        rotated_representative,
        batch,
        graph_count,
        weights=weights,
        include_response=True,
    )


def null_condition_retention_distance(
    student_null: HybridDenoiserOutput,
    frozen_stage_c_null: HybridDenoiserOutput,
    batch: torch.Tensor,
    graph_count: int,
    *,
    weights: TypedFieldWeights = TypedFieldWeights(response=0.0),
) -> TypedFieldDistance:
    """Preserve Stage-C only on the missing-condition branch.

    The caller must obtain both outputs with ``condition_present=False``.
    Response alignment is absent on that branch and is excluded by contract.
    """

    return typed_field_distance(
        student_null,
        frozen_stage_c_null,
        batch,
        graph_count,
        weights=weights,
        include_response=False,
    )


def predict_common_noisy_state(
    denoiser: HybridCrystalDenoiser,
    noisy: TensorFreeNoisyBatch,
    batch: torch.Tensor,
    tensor_condition: torch.Tensor,
    condition_present: torch.Tensor,
    shape_projector: torch.Tensor,
    fractional_to_cartesian: torch.Tensor,
) -> HybridDenoiserOutput:
    # Keep the repeated forward arguments identical across every paired path.
    time_arguments = (
        {
            "element_time": noisy.element_time,
            "lattice_time": noisy.lattice_time,
        }
        if denoiser.uses_side_modality_times
        else {}
    )
    return denoiser(
        noisy.element_tokens,
        noisy.fractional_coordinates,
        noisy.log_volume,
        noisy.log_shape,
        batch,
        noisy.time,
        tensor_condition,
        condition_present,
        shape_projector,
        fractional_to_cartesian,
        composition_counts=noisy.composition_counts,
        **time_arguments,
    )


def tensor_conditioning_training_loss(
    diffusion: TensorFreeHybridDiffusion,
    frozen_stage_c: HybridCrystalDenoiser,
    clean_elements: torch.Tensor,
    clean_fractional_coordinates: torch.Tensor,
    clean_lattice: torch.Tensor,
    batch: torch.Tensor,
    shape_projector: torch.Tensor,
    fractional_to_cartesian: torch.Tensor,
    tensor_condition: torch.Tensor,
    condition_present: torch.Tensor,
    *,
    orbit_weight: float,
    retention_weight: float,
    orbit_weights: TypedFieldWeights = TypedFieldWeights(),
    retention_weights: TypedFieldWeights = TypedFieldWeights(response=0.0),
    generator: torch.Generator | None = None,
    clean_side_information: bool = False,
) -> TensorConditioningTrainingOutput:
    """Evaluate fine, orbit-mimic and null-retention losses on one noisy draw.

    The original representative receives endpoint supervision and anchors the
    paired mimic term against a common-but-wrong collapse. The rotated path is
    scored for diagnostics but is trained through orbit mimic, matching the
    declared ``L_fine + lambda_OM L_OM`` ablation. The Stage-C teacher is
    queried only with ``condition_present=False``.
    """

    if orbit_weight < 0.0 or retention_weight < 0.0:
        raise ValueError("Stage-E auxiliary weights must be nonnegative")
    graphs = int(clean_lattice.shape[0])
    if tensor_condition.shape != (graphs, 18):
        raise ValueError("Stage-E tensor condition must have shape [graphs,18]")
    if condition_present.shape != (graphs, 1) or condition_present.dtype != torch.bool:
        raise ValueError("Stage-E condition presence must be boolean [graphs,1]")
    with torch.autocast(device_type=clean_lattice.device.type, enabled=False):
        noisy = diffusion.noise_clean_batch(
            clean_elements,
            clean_fractional_coordinates,
            clean_lattice,
            batch,
            shape_projector,
            fractional_to_cartesian,
            generator=generator,
            clean_side_information=clean_side_information,
        )
    first = predict_common_noisy_state(
        diffusion.denoiser,
        noisy,
        batch,
        tensor_condition,
        condition_present,
        shape_projector,
        fractional_to_cartesian,
    )
    first_fine = diffusion.loss_from_prediction(
        clean_elements,
        clean_lattice,
        batch,
        fractional_to_cartesian,
        noisy,
        first,
    )
    if orbit_weight > 0.0:
        with torch.autocast(device_type=clean_lattice.device.type, enabled=False):
            rotation = sample_proper_rotations(
                graphs, tensor_condition, generator=generator
            )
            rotated_condition = rotate_orbit_representative(
                tensor_condition, rotation
            )
        rotated = predict_common_noisy_state(
            diffusion.denoiser,
            noisy,
            batch,
            rotated_condition,
            condition_present,
            shape_projector,
            fractional_to_cartesian,
        )
        rotated_fine = diffusion.loss_from_prediction(
            clean_elements,
            clean_lattice,
            batch,
            fractional_to_cartesian,
            noisy,
            rotated,
        )
        mimic = orbit_mimic_distance(
            first, rotated, batch, graphs, weights=orbit_weights
        )
    else:
        rotated_fine = first_fine
        mimic = orbit_mimic_distance(
            first, first, batch, graphs, weights=orbit_weights
        )
    if retention_weight > 0.0:
        null_condition = tensor_condition.new_zeros((graphs, 18))
        null_present = torch.zeros(
            (graphs, 1), dtype=torch.bool, device=tensor_condition.device
        )
        student_null = predict_common_noisy_state(
            diffusion.denoiser,
            noisy,
            batch,
            null_condition,
            null_present,
            shape_projector,
            fractional_to_cartesian,
        )
        frozen_stage_c.eval()
        with torch.no_grad():
            teacher_null = predict_common_noisy_state(
                frozen_stage_c,
                noisy,
                batch,
                null_condition,
                null_present,
                shape_projector,
                fractional_to_cartesian,
            )
        retention = null_condition_retention_distance(
            student_null,
            teacher_null,
            batch,
            graphs,
            weights=retention_weights,
        )
    else:
        retention = null_condition_retention_distance(
            first,
            first,
            batch,
            graphs,
            weights=retention_weights,
        )
    fine = first_fine.loss
    loss = fine + orbit_weight * mimic.loss + retention_weight * retention.loss
    return TensorConditioningTrainingOutput(
        loss=loss,
        fine=fine,
        first_fine=first_fine,
        rotated_fine=rotated_fine,
        orbit_mimic=mimic,
        null_retention=retention,
    )
