"""Joint tensor-free hybrid diffusion objective for production S1a."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter

from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser, HybridDenoiserOutput
from .lattice_standardization import P1LatticeStandardizer
from .lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from .quotient_score import factorized_translation_quotient_scaled_score
from .schedules import (
    CosineNoiseSchedule,
    ExponentialTorusNoiseSchedule,
    standard_normal,
)
from .state_projection import project_translation_state


@dataclass(frozen=True)
class TensorFreeNoisyBatch:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    time: torch.Tensor
    coordinate_scaled_score_target: torch.Tensor
    clean_volume_latent_target: torch.Tensor
    clean_shape_latent_target: torch.Tensor
    element_was_masked: torch.Tensor


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
    cell-independent Brownian path on the fractional translation quotient.
    Standardized volume residual and whitened trace-free log shape follow the
    same cosine VP path used by the categorical survival schedule.
    """

    def __init__(
        self,
        denoiser: HybridCrystalDenoiser,
        lattice_standardizer: P1LatticeStandardizer,
        *,
        coordinate_sigma_min: float = 0.005,
        coordinate_sigma_max: float = 0.5,
        minimum_time: float = 1.0e-3,
        maximum_time: float = 0.999,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_time < maximum_time < 1.0:
            raise ValueError("training times must satisfy 0 < minimum < maximum < 1")
        self.denoiser = denoiser
        self.lattice_standardizer = lattice_standardizer
        self.categorical = AbsorbingMaskDiffusion()
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = ExponentialTorusNoiseSchedule(
            sigma_min=coordinate_sigma_min,
            sigma_max=coordinate_sigma_max,
        )
        self.minimum_time = float(minimum_time)
        self.maximum_time = float(maximum_time)

    def sample_time(
        self,
        graph_count: int,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if graph_count < 1:
            raise ValueError("time sampling requires at least one graph")
        # Randomized stratification is an unbiased uniform-time estimator, but
        # unlike iid draws it covers every 1/G interval in each G-graph batch.
        # This matters for the torus score, whose useful signal is concentrated
        # near the clean end of the path.  The implementation is fully batched
        # and does not alter the target objective.
        jitter = torch.rand(
            (graph_count,), dtype=reference.dtype, device=reference.device, generator=generator
        )
        strata = (
            torch.arange(graph_count, dtype=reference.dtype, device=reference.device)
            + jitter
        ) / graph_count
        order = torch.randperm(graph_count, device=reference.device, generator=generator)
        uniform = strata[order]
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
        node_counts = torch.bincount(batch, minlength=graphs)
        clean_volume_latent = self.lattice_standardizer.encode_volume(
            lattice_state.log_volume, node_counts
        )
        clean_shape_latent = self.lattice_standardizer.encode_shape(clean_shape)

        uniform = torch.rand(
            clean_elements.shape,
            dtype=clean_fractional_coordinates.dtype,
            device=clean_elements.device,
            generator=generator,
        )
        categorical_state = self.categorical.corrupt(clean_elements, selected_time, batch, uniform=uniform)

        coordinate_sigma = self.coordinate_schedule.sigma(selected_time)[batch]
        fractional_noise = standard_normal(
            clean_fractional_coordinates.shape, clean_fractional_coordinates, generator
        )
        displacement = coordinate_sigma.unsqueeze(-1) * fractional_noise
        noisy_coordinates = clean_coordinates + displacement
        coordinate_target = factorized_translation_quotient_scaled_score(
            displacement,
            self.coordinate_schedule.sigma(selected_time),
            batch,
            graphs,
        )

        alpha = self.vp_schedule.alpha(selected_time)
        sigma = self.vp_schedule.sigma(selected_time).clamp_min(1.0e-8)
        volume_noise = standard_normal(
            clean_volume_latent.shape, clean_volume_latent, generator
        )
        shape_noise = standard_normal(clean_shape_latent.shape, clean_shape_latent, generator)
        noisy_volume_latent = alpha * clean_volume_latent + sigma * volume_noise
        noisy_shape_latent = (
            alpha.unsqueeze(-1) * clean_shape_latent
            + sigma.unsqueeze(-1) * shape_noise
        )
        noisy_volume = self.lattice_standardizer.decode_volume(
            noisy_volume_latent, node_counts
        )
        noisy_shape = self.lattice_standardizer.decode_shape(noisy_shape_latent)
        noisy_shape = project_lattice_state(noisy_shape, shape_projector)

        return TensorFreeNoisyBatch(
            element_tokens=categorical_state.tokens,
            fractional_coordinates=noisy_coordinates,
            log_volume=noisy_volume,
            log_shape=noisy_shape,
            time=selected_time,
            coordinate_scaled_score_target=coordinate_target,
            clean_volume_latent_target=clean_volume_latent,
            clean_shape_latent_target=clean_shape_latent,
            element_was_masked=~categorical_state.clean_mask,
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
        # Probability-path construction and target generation stay FP32 even
        # when the learned network uses BF16 autocast.
        with torch.autocast(device_type=clean_lattice.device.type, enabled=False):
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

        coordinate_error = (
            prediction.coordinate_fractional_scaled_score
            - noisy.coordinate_scaled_score_target
        )
        coordinate_quadratic = coordinate_error.square().sum(dim=-1)
        graph_coordinate = scatter(
            coordinate_quadratic,
            batch,
            dim=0,
            dim_size=graphs,
            reduce="mean",
        )
        coordinate_loss = graph_coordinate.mean() / 3.0

        volume_error = (
            prediction.clean_volume_latent - noisy.clean_volume_latent_target
        )
        volume_loss = volume_error.square().mean()
        shape_error = prediction.clean_shape_latent - noisy.clean_shape_latent_target
        shape_loss = shape_error.square().mean()
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
