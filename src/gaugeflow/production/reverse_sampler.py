"""Joint tensor-free reverse sampler for the production hybrid diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.manifold import wrap01

from .blueprint import ParentBlueprintBatch
from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser
from .lattice_standardization import P1LatticeStandardizer
from .lattice_volume_shape import LatticeGuardrails, LatticeVolumeShape
from .schedules import CosineNoiseSchedule, FractionalTorusVarianceSchedule, standard_normal
from .state_projection import project_hybrid_reverse_state, project_translation_state


class SamplingFailure(RuntimeError):
    """Fail-closed signal for an invalid joint reverse trajectory."""


def quotient_coordinate_reverse_step(
    coordinates: torch.Tensor,
    scaled_score: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    *,
    generator: torch.Generator | None,
    stochastic: bool,
) -> torch.Tensor:
    """Apply one exact Brownian reverse bridge step on the translation quotient."""
    if coordinates.shape != scaled_score.shape or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinate reverse state and score must share shape [nodes,3]")
    if variance_from.shape != (graph_count,) or variance_to.shape != (graph_count,):
        raise ValueError("coordinate reverse variances must provide one value per graph")
    if batch.shape != coordinates.shape[:1]:
        raise ValueError("coordinate reverse batch must provide one graph per node")
    variance_drop = variance_from - variance_to
    if bool((variance_drop < 0.0).any()) or bool((variance_to < 0.0).any()):
        raise ValueError("coordinate reverse step requires decreasing nonnegative variance")
    score = (
        scaled_score
        / variance_from[batch].sqrt().clamp_min(1.0e-8).unsqueeze(-1)
    )
    updated = coordinates + variance_drop[batch].unsqueeze(-1) * score
    if stochastic and bool((variance_to > 0.0).any()):
        bridge_variance = (
            variance_to * variance_drop / variance_from.clamp_min(1.0e-12)
        )
        noise = standard_normal(coordinates.shape, coordinates, generator)
        noise = project_translation_state(noise, batch, graph_count)
        updated = updated + bridge_variance[batch].sqrt().unsqueeze(-1) * noise
    return project_translation_state(updated, batch, graph_count)


@dataclass(frozen=True)
class ReverseTrajectoryDiagnostics:
    time: torch.Tensor
    masked_count: torch.Tensor
    coordinate_step_rms: torch.Tensor
    volume_step_rms: torch.Tensor
    shape_step_rms: torch.Tensor


@dataclass(frozen=True)
class GeneratedHybridBatch:
    element_tokens: torch.Tensor
    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    batch: torch.Tensor
    diagnostics: ReverseTrajectoryDiagnostics


class TensorFreeReverseSampler:
    """Ancestral reverse process over categorical, torus and lattice states."""

    def __init__(
        self,
        denoiser: HybridCrystalDenoiser,
        lattice_standardizer: P1LatticeStandardizer,
        *,
        coordinate_fractional_sigma_max: float = 1.0,
        maximum_time: float = 0.999,
        guardrails: LatticeGuardrails | None = None,
    ) -> None:
        if not 0.0 < maximum_time < 1.0:
            raise ValueError("maximum reverse time must lie in (0,1)")
        self.denoiser = denoiser
        self.lattice_standardizer = lattice_standardizer
        self.categorical = AbsorbingMaskDiffusion()
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = FractionalTorusVarianceSchedule(
            sigma_max=coordinate_fractional_sigma_max
        )
        self.maximum_time = float(maximum_time)
        self.guardrails = guardrails

    def _vp_reverse_step(
        self,
        state: torch.Tensor,
        clean_estimate: torch.Tensor,
        time_from: torch.Tensor,
        time_to: torch.Tensor,
        *,
        generator: torch.Generator | None,
        stochastic: bool,
    ) -> torch.Tensor:
        alpha_to = self.vp_schedule.alpha(time_to)
        survival_from = self.vp_schedule.alpha(time_from).square()
        survival_to = alpha_to.square()
        noise_from = (1.0 - survival_from).clamp_min(1.0e-12)
        step_noise = (1.0 - survival_from / survival_to.clamp_min(1.0e-12)).clamp(0.0, 1.0)
        clean_coefficient = alpha_to * step_noise / noise_from
        state_coefficient = (
            (survival_from / survival_to.clamp_min(1.0e-12)).sqrt()
            * (1.0 - survival_to)
            / noise_from
        )
        mean = clean_coefficient * clean_estimate + state_coefficient * state
        if not stochastic:
            return mean
        variance = self.vp_schedule.posterior_variance(time_from, time_to)
        return mean + variance.sqrt() * standard_normal(state.shape, state, generator)

    def sample(
        self,
        blueprint: ParentBlueprintBatch,
        *,
        steps: int = 100,
        generator: torch.Generator | None = None,
        stochastic: bool = True,
        time_grid: str = "uniform_log_alpha",
    ) -> GeneratedHybridBatch:
        if steps < 1:
            raise ValueError("reverse sampler requires at least one step")
        device = blueprint.batch.device
        dtype = blueprint.shape_projector.dtype
        graphs = blueprint.node_counts.numel()
        nodes = blueprint.batch.numel()
        tokens = torch.full(
            (nodes,), self.categorical.mask_index, dtype=torch.long, device=device
        )
        coordinates = torch.rand((nodes, 3), dtype=dtype, device=device, generator=generator)
        coordinates = project_translation_state(coordinates, blueprint.batch, graphs)
        volume_latent = torch.randn(
            (graphs,), dtype=dtype, device=device, generator=generator
        )
        shape_latent = torch.randn(
            (graphs, 5), dtype=dtype, device=device, generator=generator
        )
        log_volume = self.lattice_standardizer.decode_volume(
            volume_latent, blueprint.node_counts
        )
        log_shape = self.lattice_standardizer.decode_shape(shape_latent)
        log_shape = torch.einsum("bij,bj->bi", blueprint.shape_projector, log_shape)
        condition = torch.zeros((graphs, 18), dtype=dtype, device=device)
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
        if time_grid == "uniform_time":
            times = torch.linspace(self.maximum_time, 0.0, steps + 1, dtype=dtype, device=device)
        elif time_grid == "uniform_log_alpha":
            initial_alpha = self.vp_schedule.alpha(
                torch.tensor(self.maximum_time, dtype=dtype, device=device)
            )
            alpha = torch.exp(
                torch.linspace(initial_alpha.log(), 0.0, steps + 1, dtype=dtype, device=device)
            )
            times = (2.0 / torch.pi) * torch.arccos(alpha.clamp(-1.0, 1.0))
        else:
            raise ValueError("time_grid must be 'uniform_time' or 'uniform_log_alpha'")

        masked_counts: list[torch.Tensor] = []
        coordinate_steps: list[torch.Tensor] = []
        volume_steps: list[torch.Tensor] = []
        shape_steps: list[torch.Tensor] = []
        was_training = self.denoiser.training
        self.denoiser.eval()
        trajectory_error: RuntimeError | ValueError | None = None
        try:
            with torch.no_grad():
                for index in range(steps):
                    scalar_from = times[index]
                    scalar_to = times[index + 1]
                    time_from = scalar_from.expand(graphs)
                    time_to = scalar_to.expand(graphs)
                    prediction = self.denoiser(
                        tokens,
                        coordinates,
                        log_volume,
                        log_shape,
                        blueprint.batch,
                        time_from,
                        condition,
                        condition_present,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                    )

                    probabilities = self.categorical.reverse_probabilities(
                        tokens,
                        prediction.clean_element_logits,
                        time_from,
                        time_to,
                        blueprint.batch,
                    )
                    tokens = torch.multinomial(
                        probabilities, 1, replacement=True, generator=generator
                    ).squeeze(-1)

                    variance_from = self.coordinate_schedule.variance(time_from)
                    variance_to = self.coordinate_schedule.variance(time_to)
                    next_coordinates = quotient_coordinate_reverse_step(
                        coordinates,
                        prediction.coordinate_fractional_scaled_score,
                        variance_from,
                        variance_to,
                        blueprint.batch,
                        graphs,
                        generator=generator,
                        stochastic=stochastic and float(scalar_to) > 0.0,
                    )

                    next_volume_latent = self._vp_reverse_step(
                        volume_latent,
                        prediction.clean_volume_latent,
                        time_from,
                        time_to,
                        generator=generator,
                        stochastic=stochastic,
                    )
                    next_shape_latent = self._vp_reverse_step(
                        shape_latent,
                        prediction.clean_shape_latent,
                        time_from.unsqueeze(-1),
                        time_to.unsqueeze(-1),
                        generator=generator,
                        stochastic=stochastic,
                    )
                    next_volume = self.lattice_standardizer.decode_volume(
                        next_volume_latent, blueprint.node_counts
                    )
                    next_shape = self.lattice_standardizer.decode_shape(
                        next_shape_latent
                    )
                    projected = project_hybrid_reverse_state(
                        next_coordinates,
                        next_shape,
                        blueprint.batch,
                        blueprint.shape_projector,
                    )
                    coordinate_steps.append((projected.fractional_coordinates - coordinates).square().mean().sqrt())
                    volume_steps.append((next_volume - log_volume).square().mean().sqrt())
                    shape_steps.append((projected.log_shape - log_shape).square().mean().sqrt())
                    coordinates = projected.fractional_coordinates
                    log_volume = next_volume
                    log_shape = projected.log_shape
                    volume_latent = next_volume_latent
                    shape_latent = self.lattice_standardizer.encode_shape(log_shape)
                    masked_counts.append((tokens == self.categorical.mask_index).sum())
        except (RuntimeError, ValueError) as error:
            trajectory_error = error
        finally:
            self.denoiser.train(was_training)

        if trajectory_error is not None:
            raise SamplingFailure(f"joint reverse trajectory failed: {trajectory_error}") from trajectory_error

        if bool((tokens == self.categorical.mask_index).any()):
            raise SamplingFailure("terminal categorical state contains absorbing masks")
        if not all(torch.isfinite(value).all() for value in (coordinates, log_volume, log_shape)):
            raise SamplingFailure("reverse trajectory produced a non-finite continuous state")
        try:
            lattice = LatticeVolumeShape(log_volume, log_shape).lattice(
                blueprint.fractional_to_cartesian
            )
            if self.guardrails is not None:
                self.guardrails.validate(lattice @ lattice.transpose(-1, -2))
        except (RuntimeError, ValueError) as error:
            raise SamplingFailure(f"terminal lattice is invalid: {error}") from error
        return GeneratedHybridBatch(
            element_tokens=tokens,
            atomic_numbers=self.categorical.decode(tokens),
            fractional_coordinates=wrap01(coordinates),
            lattice=lattice,
            log_volume=log_volume,
            log_shape=log_shape,
            batch=blueprint.batch,
            diagnostics=ReverseTrajectoryDiagnostics(
                time=times[1:].detach().cpu(),
                masked_count=torch.stack(masked_counts).detach().cpu(),
                coordinate_step_rms=torch.stack(coordinate_steps).detach().cpu(),
                volume_step_rms=torch.stack(volume_steps).detach().cpu(),
                shape_step_rms=torch.stack(shape_steps).detach().cpu(),
            ),
        )
