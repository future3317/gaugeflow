"""Joint tensor-free hybrid diffusion objective for production S1a."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter

from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser, HybridDenoiserOutput
from .lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from .schedules import CosineNoiseSchedule, LinearWrappedVarianceSchedule, standard_normal
from .state_projection import project_translation_state


@dataclass(frozen=True)
class TensorFreeNoisyBatch:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    time: torch.Tensor
    coordinate_score_target: torch.Tensor
    clean_log_volume_target: torch.Tensor
    clean_log_shape_target: torch.Tensor
    element_was_masked: torch.Tensor
    clean_metric: torch.Tensor


@dataclass(frozen=True)
class HybridLossOutput:
    loss: torch.Tensor
    element_loss: torch.Tensor
    coordinate_loss: torch.Tensor
    volume_loss: torch.Tensor
    shape_loss: torch.Tensor
    masked_fraction: torch.Tensor
    noisy: TensorFreeNoisyBatch
    prediction: HybridDenoiserOutput


class TensorFreeHybridDiffusion(nn.Module):
    """Matched noising and denoising-score objective for the hybrid state.

    Element types follow an absorbing categorical path. Coordinates follow a
    Cartesian-isotropic Brownian path on the periodic translation quotient.
    Log volume and trace-free log shape follow the same cosine VP path used by
    the categorical survival schedule.
    """

    def __init__(
        self,
        denoiser: HybridCrystalDenoiser,
        *,
        coordinate_sigma_max: float = 4.0,
        minimum_time: float = 1.0e-3,
        maximum_time: float = 0.999,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_time < maximum_time < 1.0:
            raise ValueError("training times must satisfy 0 < minimum < maximum < 1")
        self.denoiser = denoiser
        self.categorical = AbsorbingMaskDiffusion()
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = LinearWrappedVarianceSchedule(sigma_max=coordinate_sigma_max)
        self.minimum_time = float(minimum_time)
        self.maximum_time = float(maximum_time)

    def sample_time(
        self,
        graph_count: int,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        uniform = torch.rand(
            (graph_count,), dtype=reference.dtype, device=reference.device, generator=generator
        )
        return self.minimum_time + (self.maximum_time - self.minimum_time) * uniform

    def noise_clean_batch(
        self,
        clean_elements: torch.Tensor,
        clean_fractional_coordinates: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TensorFreeNoisyBatch:
        if clean_elements.ndim != 1 or clean_elements.dtype != torch.long:
            raise ValueError("clean elements must be rank-one int64 tokens")
        if clean_fractional_coordinates.shape != (clean_elements.numel(), 3):
            raise ValueError("clean fractional coordinates must have shape [nodes,3]")
        if batch.shape != clean_elements.shape or batch.dtype != torch.long:
            raise ValueError("batch must provide one graph index per node")
        graphs = int(batch.max()) + 1 if batch.numel() else 0
        if graphs < 1 or clean_lattice.shape != (graphs, 3, 3):
            raise ValueError("clean lattice must provide one [3,3] matrix per graph")
        if shape_projector.shape != (graphs, 6, 6) or fractional_to_cartesian.shape != (graphs, 3, 3):
            raise ValueError("blueprint chart tensors do not match graph count")
        selected_time = time
        if selected_time is None:
            selected_time = self.sample_time(graphs, clean_fractional_coordinates, generator=generator)
        if selected_time.shape != (graphs,):
            raise ValueError("time must provide one scalar per graph")

        clean_coordinates = project_translation_state(clean_fractional_coordinates, batch, graphs)
        lattice_state = LatticeVolumeShape.from_lattice(clean_lattice, fractional_to_cartesian)
        clean_shape = project_lattice_state(lattice_state.log_shape, shape_projector)

        uniform = torch.rand(
            clean_elements.shape,
            dtype=clean_fractional_coordinates.dtype,
            device=clean_elements.device,
            generator=generator,
        )
        categorical_state = self.categorical.corrupt(clean_elements, selected_time, batch, uniform=uniform)

        coordinate_sigma = self.coordinate_schedule.sigma(selected_time)[batch]
        cartesian_noise = standard_normal(
            clean_fractional_coordinates.shape, clean_fractional_coordinates, generator
        )
        inverse_lattice = torch.linalg.inv(clean_lattice)
        fractional_noise = torch.einsum("ni,nij->nj", cartesian_noise, inverse_lattice[batch])
        fractional_noise = project_translation_state(fractional_noise, batch, graphs)
        displacement = coordinate_sigma.unsqueeze(-1) * fractional_noise
        noisy_coordinates = clean_coordinates + displacement
        clean_metric = clean_lattice @ clean_lattice.transpose(-1, -2)
        coordinate_variance = self.coordinate_schedule.variance(selected_time)[batch].clamp_min(1.0e-12)
        coordinate_target = -torch.einsum("ni,nij->nj", displacement, clean_metric[batch])
        coordinate_target = coordinate_target / coordinate_variance.unsqueeze(-1)
        coordinate_target = project_translation_state(coordinate_target, batch, graphs)

        alpha = self.vp_schedule.alpha(selected_time)
        sigma = self.vp_schedule.sigma(selected_time).clamp_min(1.0e-8)
        volume_noise = standard_normal(lattice_state.log_volume.shape, lattice_state.log_volume, generator)
        raw_shape_noise = standard_normal(clean_shape.shape, clean_shape, generator)
        shape_noise = project_lattice_state(raw_shape_noise, shape_projector)
        noisy_volume = alpha * lattice_state.log_volume + sigma * volume_noise
        noisy_shape = alpha.unsqueeze(-1) * clean_shape + sigma.unsqueeze(-1) * shape_noise

        return TensorFreeNoisyBatch(
            element_tokens=categorical_state.tokens,
            fractional_coordinates=noisy_coordinates,
            log_volume=noisy_volume,
            log_shape=noisy_shape,
            time=selected_time,
            coordinate_score_target=coordinate_target,
            clean_log_volume_target=lattice_state.log_volume,
            clean_log_shape_target=clean_shape,
            element_was_masked=~categorical_state.clean_mask,
            clean_metric=clean_metric,
        )

    def forward(
        self,
        clean_elements: torch.Tensor,
        clean_fractional_coordinates: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
        *,
        time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> HybridLossOutput:
        noisy = self.noise_clean_batch(
            clean_elements,
            clean_fractional_coordinates,
            clean_lattice,
            batch,
            shape_projector,
            fractional_to_cartesian,
            time=time,
            generator=generator,
        )
        graphs = noisy.time.numel()
        condition = noisy.log_volume.new_zeros((graphs, 18))
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.log_volume.device)
        prediction = self.denoiser(
            noisy.element_tokens,
            noisy.fractional_coordinates,
            noisy.log_volume,
            noisy.log_shape,
            batch,
            noisy.time,
            condition,
            condition_present,
            shape_projector,
            fractional_to_cartesian,
        )

        node_cross_entropy = F.cross_entropy(
            prediction.clean_element_logits, clean_elements, reduction="none"
        )
        mask = noisy.element_was_masked.to(node_cross_entropy)
        element_loss = (node_cross_entropy * mask).sum() / mask.sum().clamp_min(1.0)

        coordinate_error = prediction.coordinate_fractional_score - noisy.coordinate_score_target
        inverse_metric = torch.linalg.inv(noisy.clean_metric)
        coordinate_quadratic = torch.einsum(
            "ni,nij,nj->n", coordinate_error, inverse_metric[batch], coordinate_error
        )
        coordinate_weight = self.coordinate_schedule.variance(noisy.time)[batch]
        graph_coordinate = scatter(
            coordinate_weight * coordinate_quadratic,
            batch,
            dim=0,
            dim_size=graphs,
            reduce="mean",
        )
        coordinate_loss = graph_coordinate.mean()

        volume_error = prediction.clean_log_volume - noisy.clean_log_volume_target
        volume_loss = volume_error.square().mean()
        shape_error = prediction.clean_log_shape - noisy.clean_log_shape_target
        shape_dimension = torch.diagonal(shape_projector, dim1=-2, dim2=-1).sum(-1).clamp_min(1.0)
        shape_loss = (
            shape_error.square().sum(-1) / shape_dimension
        ).mean()
        loss = element_loss + coordinate_loss + volume_loss + shape_loss
        return HybridLossOutput(
            loss=loss,
            element_loss=element_loss,
            coordinate_loss=coordinate_loss,
            volume_loss=volume_loss,
            shape_loss=shape_loss,
            masked_fraction=mask.mean(),
            noisy=noisy,
            prediction=prediction,
        )
