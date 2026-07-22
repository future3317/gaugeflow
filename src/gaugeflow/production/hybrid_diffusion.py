"""Joint tensor-free hybrid diffusion objective for production S1a."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter

from .categorical_mask import AbsorbingMaskDiffusion, MaskedCategoricalState
from .categorical_uniform import UniformCategoricalDiffusion
from .equivariant_denoiser import (
    HybridCrystalDenoiser,
    HybridDenoiserOutput,
    LatticeDenoiserOutput,
)
from .lattice_standardization import P1LatticeStandardizer
from .lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from .modality_task_measure import FiveRegimeTaskMeasure, ModalityNoiseTimes
from .orderless_product_state import (
    OrderlessPartialOccupation,
    orderless_next_reveal_nll,
    sample_orderless_partial_occupation,
)
from .quotient_score import factorized_translation_quotient_scaled_score
from .schedules import (
    CosineNoiseSchedule,
    ExponentialTorusNoiseSchedule,
    standard_normal,
)
from .state_projection import fractional_tangent_to_cartesian, project_translation_state


@dataclass(frozen=True)
class TensorFreeNoisyBatch:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    # ``time`` is the coordinate time retained as the primary reverse-process
    # coordinate.  Side modalities have explicit clocks even on the shared
    # diagonal, so training cannot silently confuse observed and corrupted
    # chemistry/lattice states.
    time: torch.Tensor
    element_time: torch.Tensor
    lattice_time: torch.Tensor
    modality_regime: torch.Tensor | None
    coordinate_scaled_score_target: torch.Tensor
    clean_volume_latent_target: torch.Tensor
    clean_shape_latent_target: torch.Tensor
    element_was_masked: torch.Tensor
    composition_counts: torch.Tensor | None
    orderless_occupation: OrderlessPartialOccupation | None


@dataclass(frozen=True)
class HybridLossOutput:
    loss: torch.Tensor
    element_loss: torch.Tensor
    composition_loss: torch.Tensor
    coordinate_loss: torch.Tensor
    graph_coordinate_loss: torch.Tensor
    volume_loss: torch.Tensor
    shape_loss: torch.Tensor
    masked_fraction: torch.Tensor
    noisy: TensorFreeNoisyBatch
    prediction: HybridDenoiserOutput


@dataclass(frozen=True)
class LatticeNoisyBatch:
    element_tokens: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    lattice_time: torch.Tensor
    clean_volume_latent_target: torch.Tensor
    clean_shape_latent_target: torch.Tensor


@dataclass(frozen=True)
class LatticeLossOutput:
    loss: torch.Tensor
    volume_loss: torch.Tensor
    shape_loss: torch.Tensor
    noisy: LatticeNoisyBatch
    prediction: LatticeDenoiserOutput


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
        categorical_path: str = "absorbing_mask",
        composition_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_time < maximum_time < 1.0:
            raise ValueError("training times must satisfy 0 < minimum < maximum < 1")
        self.denoiser = denoiser
        self.lattice_standardizer = lattice_standardizer
        self.categorical: AbsorbingMaskDiffusion | UniformCategoricalDiffusion
        self.orderless_reveal = categorical_path == "orderless_reveal"
        if categorical_path in {"absorbing_mask", "orderless_reveal"}:
            self.categorical = AbsorbingMaskDiffusion()
        elif categorical_path == "uniform_replacement":
            self.categorical = UniformCategoricalDiffusion()
        else:
            raise ValueError("unknown categorical probability path")
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = ExponentialTorusNoiseSchedule(
            sigma_min=coordinate_sigma_min,
            sigma_max=coordinate_sigma_max,
        )
        self.task_measure = FiveRegimeTaskMeasure()
        self.minimum_time = float(minimum_time)
        self.maximum_time = float(maximum_time)
        self.composition_conditioning = bool(composition_conditioning)
        if self.orderless_reveal and not self.composition_conditioning:
            raise ValueError("orderless categorical reveal requires an observed sampled composition")

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
        jitter = torch.rand((graph_count,), dtype=reference.dtype, device=reference.device, generator=generator)
        strata = (torch.arange(graph_count, dtype=reference.dtype, device=reference.device) + jitter) / graph_count
        order = torch.randperm(graph_count, device=reference.device, generator=generator)
        uniform = strata[order]
        return self.minimum_time + (self.maximum_time - self.minimum_time) * uniform

    def sample_task_measure_times(
        self,
        graph_count: int,
        reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ModalityNoiseTimes:
        """Sample the active task measure without exposing regime IDs to the model."""

        return self.task_measure.sample(
            graph_count,
            lambda count: self.sample_time(count, reference, generator=generator),
            generator=generator,
        )

    def _encode_clean_lattice(
        self,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the clean volume/shape endpoint once for all objectives."""

        graphs = shape_projector.shape[0]
        if clean_lattice.shape != (graphs, 3, 3):
            raise ValueError("clean lattice must provide one [3,3] matrix per graph")
        if fractional_to_cartesian.shape != (graphs, 3, 3):
            raise ValueError("fractional-to-Cartesian chart does not match graph count")
        node_counts = torch.bincount(batch, minlength=graphs)
        lattice_state = LatticeVolumeShape.from_lattice(
            clean_lattice,
            fractional_to_cartesian,
        )
        clean_shape = project_lattice_state(lattice_state.log_shape, shape_projector)
        clean_volume_latent = self.lattice_standardizer.encode_volume(
            lattice_state.log_volume,
            node_counts,
        )
        clean_shape_latent = self.lattice_standardizer.encode_shape(clean_shape)
        return node_counts, clean_volume_latent, clean_shape_latent, clean_shape

    def noise_lattice_batch(
        self,
        clean_elements: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
        *,
        lattice_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LatticeNoisyBatch:
        """Construct only the coordinate-independent lattice probability path."""

        if clean_elements.ndim != 1 or clean_elements.dtype != torch.long:
            raise ValueError("clean elements must be rank-one int64 tokens")
        if batch.shape != clean_elements.shape or batch.dtype != torch.long:
            raise ValueError("batch must provide one graph index per node")
        graphs = int(batch.max()) + 1 if batch.numel() else 0
        if graphs < 1 or shape_projector.shape != (graphs, 6, 6):
            raise ValueError("shape projector does not match a nonempty lattice batch")
        selected_time = lattice_time
        if selected_time is None:
            selected_time = self.sample_time(graphs, clean_lattice, generator=generator)
        if selected_time.shape != (graphs,):
            raise ValueError("lattice time must provide one scalar per graph")
        node_counts, clean_volume_latent, clean_shape_latent, _ = self._encode_clean_lattice(
            clean_lattice,
            batch,
            shape_projector,
            fractional_to_cartesian,
        )
        alpha = self.vp_schedule.alpha(selected_time)
        sigma = self.vp_schedule.sigma(selected_time)
        volume_noise = standard_normal(
            clean_volume_latent.shape,
            clean_volume_latent,
            generator,
        )
        shape_noise = standard_normal(
            clean_shape_latent.shape,
            clean_shape_latent,
            generator,
        )
        noisy_volume_latent = alpha * clean_volume_latent + sigma * volume_noise
        noisy_shape_latent = alpha.unsqueeze(-1) * clean_shape_latent + sigma.unsqueeze(-1) * shape_noise
        noisy_volume = self.lattice_standardizer.decode_volume(
            noisy_volume_latent,
            node_counts,
        )
        noisy_shape = project_lattice_state(
            self.lattice_standardizer.decode_shape(noisy_shape_latent),
            shape_projector,
        )
        return LatticeNoisyBatch(
            element_tokens=clean_elements,
            log_volume=noisy_volume,
            log_shape=noisy_shape,
            lattice_time=selected_time,
            clean_volume_latent_target=clean_volume_latent,
            clean_shape_latent_target=clean_shape_latent,
        )

    def forward_lattice(
        self,
        clean_elements: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        shape_projector: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
        *,
        lattice_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LatticeLossOutput:
        """Evaluate the lattice-only denoising objective without a graph."""

        with torch.autocast(device_type=clean_lattice.device.type, enabled=False):
            noisy = self.noise_lattice_batch(
                clean_elements,
                clean_lattice,
                batch,
                shape_projector,
                fractional_to_cartesian,
                lattice_time=lattice_time,
                generator=generator,
            )
        prediction = self.denoiser.forward_lattice(
            noisy.element_tokens,
            noisy.log_volume,
            noisy.log_shape,
            batch,
            noisy.lattice_time,
            shape_projector,
        )
        volume_loss = (prediction.clean_volume_latent - noisy.clean_volume_latent_target).square().mean()
        shape_loss = (prediction.clean_shape_latent - noisy.clean_shape_latent_target).square().mean()
        return LatticeLossOutput(
            loss=volume_loss + shape_loss,
            volume_loss=volume_loss,
            shape_loss=shape_loss,
            noisy=noisy,
            prediction=prediction,
        )

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
        element_time: torch.Tensor | None = None,
        lattice_time: torch.Tensor | None = None,
        modality_regime: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        clean_side_information: bool = False,
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
        if clean_side_information and (element_time is not None or lattice_time is not None):
            raise ValueError("clean side information cannot also specify corrupted side times")
        if (element_time is None) != (lattice_time is None):
            raise ValueError("element and lattice times must be supplied together")
        selected_element_time = (
            torch.zeros_like(selected_time)
            if clean_side_information
            else selected_time
            if element_time is None
            else element_time
        )
        selected_lattice_time = (
            torch.zeros_like(selected_time)
            if clean_side_information
            else selected_time
            if lattice_time is None
            else lattice_time
        )
        if selected_element_time.shape != (graphs,) or selected_lattice_time.shape != (graphs,):
            raise ValueError("side-modality times must provide one scalar per graph")
        if modality_regime is not None and (modality_regime.shape != (graphs,) or modality_regime.dtype != torch.long):
            raise ValueError("modality regime must be an int64 graph vector")

        clean_coordinates = project_translation_state(clean_fractional_coordinates, batch, graphs)
        composition_counts: torch.Tensor | None = None
        if self.composition_conditioning:
            flat_target = batch * self.categorical.element_count + clean_elements
            composition_counts = torch.bincount(
                flat_target,
                minlength=graphs * self.categorical.element_count,
            ).reshape(graphs, self.categorical.element_count)
        node_counts, clean_volume_latent, clean_shape_latent, _ = self._encode_clean_lattice(
            clean_lattice,
            batch,
            shape_projector,
            fractional_to_cartesian,
        )

        orderless_occupation: OrderlessPartialOccupation | None = None
        if self.orderless_reveal:
            assert composition_counts is not None
            orderless_occupation = sample_orderless_partial_occupation(
                clean_elements,
                batch,
                composition_counts,
                selected_element_time,
                generator=generator,
                vocabulary_size=self.categorical.element_count,
                mask_token=self.categorical.mask_index,
            )
            categorical_state = MaskedCategoricalState(
                tokens=orderless_occupation.partial_tokens,
                clean_mask=orderless_occupation.partial_tokens != self.categorical.mask_index,
            )
        elif clean_side_information:
            # This is observed side information, not the t=0 endpoint of the
            # categorical process.  Construct it directly so its contract is
            # independent of the categorical schedule implementation.
            self.categorical.validate_clean(clean_elements)
            categorical_state = MaskedCategoricalState(
                tokens=clean_elements,
                clean_mask=torch.ones_like(clean_elements, dtype=torch.bool),
            )
        else:
            uniform = torch.rand(
                clean_elements.shape,
                dtype=clean_fractional_coordinates.dtype,
                device=clean_elements.device,
                generator=generator,
            )
            if isinstance(self.categorical, UniformCategoricalDiffusion):
                token_uniform = torch.rand(
                    clean_elements.shape,
                    dtype=clean_fractional_coordinates.dtype,
                    device=clean_elements.device,
                    generator=generator,
                )
                categorical_state = self.categorical.corrupt(
                    clean_elements,
                    selected_element_time,
                    batch,
                    uniform=uniform,
                    token_uniform=token_uniform,
                )
            else:
                categorical_state = self.categorical.corrupt(
                    clean_elements, selected_element_time, batch, uniform=uniform
                )

        coordinate_sigma_graph = self.coordinate_schedule.sigma(selected_time)
        coordinate_sigma = coordinate_sigma_graph[batch]
        fractional_noise = standard_normal(clean_fractional_coordinates.shape, clean_fractional_coordinates, generator)
        displacement = coordinate_sigma.unsqueeze(-1) * fractional_noise
        noisy_coordinates = clean_coordinates + displacement
        # An observed coordinate state is represented by t_F=0.  Its Dirac
        # endpoint has no finite score and is never an active coordinate
        # objective.  Evaluate the analytic score at a harmless positive scale
        # and mask it to zero, keeping the whole operation vectorized and
        # finite for element-only training.
        positive_coordinate_sigma = torch.where(
            selected_time > 0.0,
            coordinate_sigma_graph,
            torch.ones_like(coordinate_sigma_graph),
        )
        coordinate_target = factorized_translation_quotient_scaled_score(
            displacement,
            positive_coordinate_sigma,
            batch,
            graphs,
        )
        coordinate_target = torch.where(
            (selected_time > 0.0)[batch, None],
            coordinate_target,
            torch.zeros_like(coordinate_target),
        )

        if clean_side_information:
            noisy_volume_latent = clean_volume_latent
            noisy_shape_latent = clean_shape_latent
        else:
            alpha = self.vp_schedule.alpha(selected_lattice_time)
            sigma = self.vp_schedule.sigma(selected_lattice_time)
            sigma = torch.where(
                selected_lattice_time == 0,
                torch.zeros_like(sigma),
                sigma.clamp_min(1.0e-8),
            )
            volume_noise = standard_normal(clean_volume_latent.shape, clean_volume_latent, generator)
            shape_noise = standard_normal(clean_shape_latent.shape, clean_shape_latent, generator)
            noisy_volume_latent = alpha * clean_volume_latent + sigma * volume_noise
            noisy_shape_latent = alpha.unsqueeze(-1) * clean_shape_latent + sigma.unsqueeze(-1) * shape_noise
        noisy_volume = self.lattice_standardizer.decode_volume(noisy_volume_latent, node_counts)
        noisy_shape = self.lattice_standardizer.decode_shape(noisy_shape_latent)
        noisy_shape = project_lattice_state(noisy_shape, shape_projector)

        return TensorFreeNoisyBatch(
            element_tokens=categorical_state.tokens,
            fractional_coordinates=noisy_coordinates,
            log_volume=noisy_volume,
            log_shape=noisy_shape,
            time=selected_time,
            element_time=selected_element_time,
            lattice_time=selected_lattice_time,
            modality_regime=modality_regime,
            coordinate_scaled_score_target=coordinate_target,
            clean_volume_latent_target=clean_volume_latent,
            clean_shape_latent_target=clean_shape_latent,
            element_was_masked=~categorical_state.clean_mask,
            composition_counts=composition_counts,
            orderless_occupation=orderless_occupation,
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
        element_time: torch.Tensor | None = None,
        lattice_time: torch.Tensor | None = None,
        modality_regime: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        clean_side_information: bool = False,
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
                element_time=element_time,
                lattice_time=lattice_time,
                modality_regime=modality_regime,
                generator=generator,
                clean_side_information=clean_side_information,
            )
        graphs = noisy.time.numel()
        condition = noisy.log_volume.new_zeros((graphs, 18))
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=noisy.log_volume.device)
        denoiser_time_arguments = (
            {
                "element_time": noisy.element_time,
                "lattice_time": noisy.lattice_time,
            }
            if self.denoiser.uses_side_modality_times
            else {}
        )
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
            composition_counts=noisy.composition_counts,
            **denoiser_time_arguments,
        )

        return self.loss_from_prediction(
            clean_elements,
            clean_lattice,
            batch,
            fractional_to_cartesian,
            noisy,
            prediction,
        )

    def loss_from_prediction(
        self,
        clean_elements: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
        noisy: TensorFreeNoisyBatch,
        prediction: HybridDenoiserOutput,
    ) -> HybridLossOutput:
        """Score one prediction against an already sampled common noisy state.

        Stage-E uses this public decomposition to evaluate two tensor-orbit
        representatives against exactly the same probability-path draw.  It
        does not resample noise or duplicate target construction.
        """

        graphs = noisy.time.numel()
        if noisy.orderless_occupation is None:
            node_cross_entropy = F.cross_entropy(prediction.clean_element_logits, clean_elements, reduction="none")
            mask = noisy.element_was_masked.to(node_cross_entropy)
            element_loss = (node_cross_entropy * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            _, element_loss = orderless_next_reveal_nll(
                prediction.clean_element_logits,
                noisy.orderless_occupation,
                vocabulary_size=self.categorical.element_count,
            )
            mask = noisy.element_was_masked.to(element_loss)
        if noisy.composition_counts is None:
            predicted_composition = torch.softmax(prediction.clean_composition_logits.float(), dim=-1)
            flat_target = batch * self.categorical.element_count + clean_elements
            target_counts = torch.bincount(
                flat_target,
                minlength=graphs * self.categorical.element_count,
            ).reshape(graphs, self.categorical.element_count)
            target_composition = target_counts.to(predicted_composition)
            target_composition = target_composition / target_composition.sum(
                dim=-1,
                keepdim=True,
            )
            composition_loss = -(
                target_composition * predicted_composition.clamp_min(1.0e-8).log()
            ).sum(dim=-1).mean()
        else:
            # The sampled composition is supplied by the qualified upstream
            # law, so relearning it from the endpoint would be target leakage.
            composition_loss = prediction.clean_element_logits.new_zeros(())

        # The analytic score components become a reverse-drift tangent under
        # the fractional Brownian mobility. With r=fL, its Cartesian target is
        # v_r=v_f L. This is the same tangent-vector chart used by the Cartesian
        # carrier and is exactly inverted by the denoiser before sampling.
        with torch.autocast(device_type=clean_lattice.device.type, enabled=False):
            noisy_lattice = LatticeVolumeShape(noisy.log_volume.float(), noisy.log_shape.float()).lattice(
                fractional_to_cartesian.float()
            )
            coordinate_target_cartesian = fractional_tangent_to_cartesian(
                noisy.coordinate_scaled_score_target.float(),
                noisy_lattice,
                batch,
            )
        # Optimize the equivalent dimensionless Cartesian chart.  Returning
        # the physical tangent from the denoiser keeps the probability path
        # and reverse sampler unchanged, while V^(-1/3) removes a purely
        # representational cell-size heteroscedasticity from the objective.
        cell_scale = torch.exp(noisy.log_volume.float() / 3.0)
        coordinate_error = (
            prediction.coordinate_cartesian_scaled_score.float() - coordinate_target_cartesian
        ) / cell_scale[batch, None]
        coordinate_quadratic = coordinate_error.square().sum(dim=-1)
        graph_coordinate = scatter(
            coordinate_quadratic,
            batch,
            dim=0,
            dim_size=graphs,
            reduce="mean",
        )
        coordinate_loss = graph_coordinate.mean() / 3.0

        volume_error = prediction.clean_volume_latent - noisy.clean_volume_latent_target
        volume_loss = volume_error.square().mean()
        shape_error = prediction.clean_shape_latent - noisy.clean_shape_latent_target
        shape_loss = shape_error.square().mean()
        loss = element_loss + coordinate_loss + volume_loss + shape_loss
        return HybridLossOutput(
            loss=loss,
            element_loss=element_loss,
            composition_loss=composition_loss,
            coordinate_loss=coordinate_loss,
            graph_coordinate_loss=graph_coordinate,
            volume_loss=volume_loss,
            shape_loss=shape_loss,
            masked_fraction=mask.mean(),
            noisy=noisy,
            prediction=prediction,
        )
