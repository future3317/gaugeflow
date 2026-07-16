"""Joint tensor-free reverse sampler for the production hybrid diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.manifold import wrap01

from .blueprint import ParentBlueprintBatch
from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser
from .lattice_volume_shape import LatticeGuardrails, LatticeVolumeShape
from .schedules import CosineNoiseSchedule, LinearWrappedVarianceSchedule, standard_normal
from .state_projection import project_hybrid_reverse_state, project_translation_state


class SamplingFailure(RuntimeError):
    """Fail-closed signal for an invalid joint reverse trajectory."""


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
        *,
        coordinate_sigma_max: float = 4.0,
        maximum_time: float = 0.999,
        guardrails: LatticeGuardrails | None = None,
    ) -> None:
        if not 0.0 < maximum_time < 1.0:
            raise ValueError("maximum reverse time must lie in (0,1)")
        self.denoiser = denoiser
        self.categorical = AbsorbingMaskDiffusion()
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = LinearWrappedVarianceSchedule(sigma_max=coordinate_sigma_max)
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
        log_volume = torch.randn((graphs,), dtype=dtype, device=device, generator=generator)
        raw_shape = torch.randn((graphs, 6), dtype=dtype, device=device, generator=generator)
        log_shape = torch.einsum("bij,bj->bi", blueprint.shape_projector, raw_shape)
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

                    lattice_state = LatticeVolumeShape(log_volume, log_shape)
                    lattice = lattice_state.lattice(blueprint.fractional_to_cartesian)
                    metric = lattice @ lattice.transpose(-1, -2)
                    inverse_metric = torch.linalg.inv(metric)
                    variance_from = self.coordinate_schedule.variance(time_from)
                    variance_to = self.coordinate_schedule.variance(time_to)
                    variance_drop = variance_from - variance_to
                    coordinate_drift = variance_drop[blueprint.batch].unsqueeze(-1) * torch.einsum(
                        "ni,nij->nj",
                        prediction.coordinate_fractional_score,
                        inverse_metric[blueprint.batch],
                    )
                    next_coordinates = coordinates + coordinate_drift
                    if stochastic and float(scalar_to) > 0.0:
                        bridge_variance = (
                            variance_to * variance_drop / variance_from.clamp_min(1.0e-12)
                        )
                        cartesian_noise = standard_normal(coordinates.shape, coordinates, generator)
                        fractional_noise = torch.einsum(
                            "ni,nij->nj", cartesian_noise, torch.linalg.inv(lattice)[blueprint.batch]
                        )
                        bridge_scale = bridge_variance[blueprint.batch].sqrt().unsqueeze(-1)
                        next_coordinates = next_coordinates + bridge_scale * fractional_noise

                    next_volume = self._vp_reverse_step(
                        log_volume,
                        prediction.clean_log_volume,
                        time_from,
                        time_to,
                        generator=generator,
                        stochastic=stochastic,
                    )
                    next_shape = self._vp_reverse_step(
                        log_shape,
                        prediction.clean_log_shape,
                        time_from.unsqueeze(-1),
                        time_to.unsqueeze(-1),
                        generator=generator,
                        stochastic=stochastic,
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
