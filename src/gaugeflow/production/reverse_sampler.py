"""Joint tensor-free reverse sampler for the production hybrid diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch

from gaugeflow.manifold import wrap01

from .assignment_training import sample_uniform_reveal_ranks
from .autoregressive_assignment import RemainingCountAssignmentLaw
from .blueprint import ParentBlueprintBatch
from .categorical_mask import AbsorbingMaskDiffusion
from .categorical_uniform import UniformCategoricalDiffusion
from .composition_assignment import composition_counts_from_tokens, count_projected_assignment
from .composition_state import StoichiometryFirstCompositionModel
from .equivariant_denoiser import HybridCrystalDenoiser
from .lattice_standardization import P1LatticeStandardizer
from .lattice_volume_shape import LatticeGuardrails, LatticeVolumeShape, project_lattice_state
from .schedules import CosineNoiseSchedule, ExponentialTorusNoiseSchedule, standard_normal
from .state_projection import project_hybrid_reverse_state, project_translation_state


class SamplingFailure(RuntimeError):
    """Fail-closed signal for an invalid joint reverse trajectory."""


ContinuousReverseMode: TypeAlias = Literal["reverse_sde", "probability_flow"]


def _validate_continuous_mode(mode: str) -> ContinuousReverseMode:
    if mode not in {"reverse_sde", "probability_flow"}:
        raise ValueError("continuous reverse mode must be 'reverse_sde' or 'probability_flow'")
    return cast(ContinuousReverseMode, mode)


def reverse_time_grid(
    schedule: CosineNoiseSchedule,
    maximum_time: float,
    steps: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    spacing: str,
) -> torch.Tensor:
    """Build the exact production reverse grid shared by runtime audits."""
    if steps < 1:
        raise ValueError("reverse time grid requires at least one step")
    if not 0.0 < maximum_time < 1.0:
        raise ValueError("maximum reverse time must lie in (0,1)")
    if spacing == "uniform_time":
        return torch.linspace(maximum_time, 0.0, steps + 1, dtype=dtype, device=device)
    if spacing == "uniform_log_alpha":
        initial_alpha = schedule.alpha(torch.tensor(maximum_time, dtype=dtype, device=device))
        alpha = torch.exp(torch.linspace(initial_alpha.log(), 0.0, steps + 1, dtype=dtype, device=device))
        return (2.0 / torch.pi) * torch.arccos(alpha.clamp(-1.0, 1.0))
    raise ValueError("reverse time spacing must be 'uniform_time' or 'uniform_log_alpha'")


def quotient_coordinate_reverse_step(
    coordinates: torch.Tensor,
    scaled_score: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    *,
    generator: torch.Generator | None,
    mode: ContinuousReverseMode,
    standard_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one reverse-SDE or probability-flow step on the translation quotient.

    The network returns ``sigma_t * score_t``.  In Brownian variance time, the
    reverse SDE uses the full score drift, whereas the probability-flow ODE
    uses one half of that drift.  Both stay on the horizontal universal-cover
    lift; periodic wrapping is a terminal decoding operation.
    """
    mode = _validate_continuous_mode(mode)
    if coordinates.shape != scaled_score.shape or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinate reverse state and score must share shape [nodes,3]")
    if variance_from.shape != (graph_count,) or variance_to.shape != (graph_count,):
        raise ValueError("coordinate reverse variances must provide one value per graph")
    if batch.shape != coordinates.shape[:1]:
        raise ValueError("coordinate reverse batch must provide one graph per node")
    variance_drop = variance_from - variance_to
    if bool((variance_drop < 0.0).any()) or bool((variance_to < 0.0).any()):
        raise ValueError("coordinate reverse step requires decreasing nonnegative variance")
    score = scaled_score / variance_from[batch].sqrt().clamp_min(1.0e-8).unsqueeze(-1)
    drift_multiplier = 1.0 if mode == "reverse_sde" else 0.5
    updated = coordinates + drift_multiplier * variance_drop[batch].unsqueeze(-1) * score
    if mode == "reverse_sde" and bool((variance_to > 0.0).any()):
        bridge_variance = variance_to * variance_drop / variance_from.clamp_min(1.0e-12)
        if standard_noise is not None:
            if generator is not None:
                raise ValueError("provide either a generator or prescribed coordinate noise")
            if standard_noise.shape != coordinates.shape:
                raise ValueError("prescribed coordinate noise must match the coordinate state")
            noise = standard_noise.to(dtype=coordinates.dtype, device=coordinates.device)
        else:
            noise = standard_normal(coordinates.shape, coordinates, generator)
        noise = project_translation_state(noise, batch, graph_count)
        updated = updated + bridge_variance[batch].sqrt().unsqueeze(-1) * noise
    return project_translation_state(updated, batch, graph_count)


def vp_reverse_step(
    schedule: CosineNoiseSchedule,
    state: torch.Tensor,
    clean_estimate: torch.Tensor,
    time_from: torch.Tensor,
    time_to: torch.Tensor,
    *,
    generator: torch.Generator | None,
    mode: ContinuousReverseMode,
) -> torch.Tensor:
    """Advance one variance-preserving reverse transition.

    ``reverse_sde`` is the ancestral DDPM transition. ``probability_flow`` is
    the deterministic DDIM transport induced by the same clean-state estimate;
    returning the DDPM posterior mean with its noise disabled would define
    neither of these dynamics.
    """
    mode = _validate_continuous_mode(mode)
    if state.shape != clean_estimate.shape:
        raise ValueError("VP reverse state and clean estimate must share shape")
    if time_from.shape != time_to.shape or bool((time_to > time_from).any()):
        raise ValueError("VP reverse endpoints must have equal shape and time_to <= time_from")
    alpha_from = schedule.alpha(time_from)
    alpha_to = schedule.alpha(time_to)
    sigma_from = schedule.sigma(time_from)
    sigma_to = schedule.sigma(time_to)
    if mode == "probability_flow":
        predicted_noise = (state - alpha_from * clean_estimate) / sigma_from.clamp_min(schedule.minimum_sigma)
        return alpha_to * clean_estimate + sigma_to * predicted_noise

    survival_from = alpha_from.square()
    survival_to = alpha_to.square()
    noise_from = (1.0 - survival_from).clamp_min(1.0e-12)
    step_noise = (1.0 - survival_from / survival_to.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    clean_coefficient = alpha_to * step_noise / noise_from
    state_coefficient = (survival_from / survival_to.clamp_min(1.0e-12)).sqrt() * (1.0 - survival_to) / noise_from
    mean = clean_coefficient * clean_estimate + state_coefficient * state
    variance = schedule.posterior_variance(time_from, time_to)
    return mean + variance.sqrt() * standard_normal(state.shape, state, generator)


@dataclass(frozen=True)
class ReverseTrajectoryDiagnostics:
    time: torch.Tensor
    masked_count: torch.Tensor
    remaining_atom_count: torch.Tensor
    composition_closure_error: torch.Tensor
    coordinate_step_rms: torch.Tensor
    volume_step_rms: torch.Tensor
    shape_step_rms: torch.Tensor


@dataclass(frozen=True)
class ContinuousReverseInitialState:
    """Common continuous prior draw that can be reused across solver modes."""

    fractional_coordinates: torch.Tensor
    volume_latent: torch.Tensor
    shape_latent: torch.Tensor


@dataclass(frozen=True)
class GeneratedHybridBatch:
    element_tokens: torch.Tensor
    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    composition_counts: torch.Tensor
    batch: torch.Tensor
    diagnostics: ReverseTrajectoryDiagnostics


@dataclass(frozen=True)
class ElementReverseDiagnostics:
    """Discrete-only trajectory diagnostics with observed geometry fixed."""

    time: torch.Tensor
    masked_count: torch.Tensor


@dataclass(frozen=True)
class GeneratedElementBatch:
    """Element sample conditioned on an observed coordinate/lattice carrier."""

    element_tokens: torch.Tensor
    atomic_numbers: torch.Tensor
    batch: torch.Tensor
    predicted_composition_counts: torch.Tensor
    terminal_clean_element_logits: torch.Tensor
    terminal_clean_composition_logits: torch.Tensor
    diagnostics: ElementReverseDiagnostics


@dataclass(frozen=True)
class LatticeReverseInitialState:
    volume_latent: torch.Tensor
    shape_latent: torch.Tensor


@dataclass(frozen=True)
class LatticeReverseDiagnostics:
    time: torch.Tensor
    volume_step_rms: torch.Tensor
    shape_step_rms: torch.Tensor


@dataclass(frozen=True)
class GeneratedLatticeBatch:
    lattice: torch.Tensor
    log_volume: torch.Tensor
    log_shape: torch.Tensor
    diagnostics: LatticeReverseDiagnostics


@dataclass(frozen=True)
class CoordinateReverseInitialState:
    """Reusable translation-quotient coordinate prior draw."""

    fractional_coordinates: torch.Tensor


@dataclass(frozen=True)
class CoordinateReverseDiagnostics:
    """Coordinate-only reverse trajectory diagnostics."""

    time: torch.Tensor
    coordinate_step_rms: torch.Tensor


@dataclass(frozen=True)
class GeneratedCoordinateBatch:
    """Coordinate sample with observed element and lattice side states."""

    element_tokens: torch.Tensor
    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor
    diagnostics: CoordinateReverseDiagnostics


class TensorFreeReverseSampler:
    """Hybrid categorical and continuous reverse process for crystal states."""

    def __init__(
        self,
        denoiser: HybridCrystalDenoiser,
        lattice_standardizer: P1LatticeStandardizer,
        *,
        coordinate_sigma_min: float = 0.005,
        coordinate_sigma_max: float = 0.5,
        maximum_time: float = 0.999,
        guardrails: LatticeGuardrails | None = None,
        categorical_path: str = "absorbing_mask",
        composition_model: StoichiometryFirstCompositionModel | None = None,
    ) -> None:
        if not 0.0 < maximum_time < 1.0:
            raise ValueError("maximum reverse time must lie in (0,1)")
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
        if self.orderless_reveal != (composition_model is not None):
            raise ValueError(
                "orderless product sampling requires a composition model, and composition models "
                "must not be silently ignored by a legacy categorical sampler"
            )
        self.composition_model = composition_model
        self.assignment_law = RemainingCountAssignmentLaw()
        self.vp_schedule = CosineNoiseSchedule()
        self.coordinate_schedule = ExponentialTorusNoiseSchedule(
            sigma_min=coordinate_sigma_min,
            sigma_max=coordinate_sigma_max,
        )
        self.maximum_time = float(maximum_time)
        self.guardrails = guardrails

    def _clean_composition_counts(
        self,
        element_tokens: torch.Tensor,
        batch: torch.Tensor,
        graphs: int,
    ) -> torch.Tensor:
        """Materialize the exact composition state for a clean side path.

        The joint sampler already carries ``composition_counts`` as an
        explicit state.  Coordinate-only and lattice-only entry points receive
        clean element tokens instead, but the denoiser was trained with the
        corresponding exact-count context whenever composition conditioning is
        enabled.  Reconstructing it here is lossless and keeps all reverse
        paths on the same product-space interface; it is not a terminal count
        repair or a new stochastic draw.
        """

        if graphs < 1:
            raise ValueError("clean composition counts require at least one graph")
        flat = batch * self.categorical.element_count + element_tokens
        return torch.bincount(
            flat,
            minlength=graphs * self.categorical.element_count,
        ).reshape(graphs, self.categorical.element_count)

    def initialize_coordinate_state(
        self,
        blueprint: ParentBlueprintBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> CoordinateReverseInitialState:
        """Draw a reusable uniform torus prior on the translation quotient."""

        coordinates = torch.rand(
            (int(blueprint.batch.numel()), 3),
            dtype=blueprint.shape_projector.dtype,
            device=blueprint.shape_projector.device,
            generator=generator,
        )
        return CoordinateReverseInitialState(
            fractional_coordinates=project_translation_state(
                coordinates,
                blueprint.batch,
                int(blueprint.node_counts.numel()),
            )
        )

    def sample_coordinates(
        self,
        element_tokens: torch.Tensor,
        lattice: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        tensor_condition: torch.Tensor | None = None,
        steps: int = 100,
        initial_state: CoordinateReverseInitialState | None = None,
        initialization_generator: torch.Generator | None = None,
        continuous_generator: torch.Generator | None = None,
        continuous_mode: ContinuousReverseMode = "reverse_sde",
        time_grid: str = "uniform_log_alpha",
    ) -> GeneratedCoordinateBatch:
        """Reverse only ``F_t`` while holding observed ``(A_0,L_0)`` fixed."""

        if steps < 1:
            raise ValueError("coordinate reverse sampler requires at least one step")
        if self.denoiser.modality_time_conditioning != "separate":
            raise ValueError("coordinate reverse sampling requires the unified separate-clock backbone")
        continuous_mode = _validate_continuous_mode(continuous_mode)
        device = blueprint.batch.device
        dtype = blueprint.shape_projector.dtype
        graphs = int(blueprint.node_counts.numel())
        nodes = int(blueprint.batch.numel())
        if element_tokens.shape != (nodes,) or element_tokens.dtype != torch.long:
            raise ValueError("coordinate conditioning requires one int64 element token per node")
        if lattice.shape != (graphs, 3, 3):
            raise ValueError("coordinate conditioning requires one 3x3 lattice per graph")
        if element_tokens.device != device or lattice.device != device or lattice.dtype != dtype:
            raise ValueError("coordinate side states must match the blueprint dtype and device")
        self.categorical.validate_clean(element_tokens)
        composition_counts = self._clean_composition_counts(element_tokens, blueprint.batch, graphs)
        if not bool(torch.isfinite(lattice).all()) or bool((torch.linalg.det(lattice) <= 0.0).any()):
            raise ValueError("coordinate conditioning lattices must be finite and right handed")
        if self.guardrails is not None:
            self.guardrails.validate(lattice @ lattice.transpose(-1, -2))

        if initial_state is None:
            initial_state = self.initialize_coordinate_state(
                blueprint,
                generator=initialization_generator,
            )
        coordinates = initial_state.fractional_coordinates
        if (
            coordinates.shape != (nodes, 3)
            or coordinates.dtype != dtype
            or coordinates.device != device
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError("initial coordinates must match the blueprint")
        coordinates = project_translation_state(coordinates.clone(), blueprint.batch, graphs)

        lattice_state = LatticeVolumeShape.from_lattice(
            lattice,
            blueprint.fractional_to_cartesian,
        )
        log_volume = lattice_state.log_volume
        log_shape = project_lattice_state(
            lattice_state.log_shape,
            blueprint.shape_projector,
        )
        clean_time = torch.zeros((graphs,), dtype=dtype, device=device)
        if tensor_condition is None:
            condition = torch.zeros((graphs, 18), dtype=dtype, device=device)
            condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
        else:
            if (
                tensor_condition.shape != (graphs, 18)
                or tensor_condition.dtype != dtype
                or tensor_condition.device != device
                or not bool(torch.isfinite(tensor_condition).all())
            ):
                raise ValueError("tensor condition must be finite [graphs,18] in the blueprint chart")
            condition = tensor_condition
            condition_present = torch.ones((graphs, 1), dtype=torch.bool, device=device)
        times = reverse_time_grid(
            self.vp_schedule,
            self.maximum_time,
            steps,
            dtype=dtype,
            device=device,
            spacing=time_grid,
        )
        coordinate_steps: list[torch.Tensor] = []
        was_training = self.denoiser.training
        self.denoiser.eval()
        trajectory_error: RuntimeError | ValueError | None = None
        try:
            with torch.no_grad():
                for index in range(steps):
                    time_from = times[index].expand(graphs)
                    time_to = times[index + 1].expand(graphs)
                    prediction = self.denoiser(
                        element_tokens,
                        coordinates,
                        log_volume,
                        log_shape,
                        blueprint.batch,
                        time_from,
                        condition,
                        condition_present,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                        composition_counts=composition_counts,
                        element_time=clean_time,
                        lattice_time=clean_time,
                    )
                    next_coordinates = quotient_coordinate_reverse_step(
                        coordinates,
                        prediction.coordinate_fractional_scaled_score,
                        self.coordinate_schedule.variance(time_from),
                        self.coordinate_schedule.variance(time_to),
                        blueprint.batch,
                        graphs,
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )
                    coordinate_steps.append((next_coordinates - coordinates).square().mean().sqrt())
                    coordinates = next_coordinates
        except (RuntimeError, ValueError) as error:
            trajectory_error = error
        finally:
            self.denoiser.train(was_training)
        if trajectory_error is not None:
            raise SamplingFailure(
                f"coordinate reverse trajectory failed: {trajectory_error}"
            ) from trajectory_error
        if not bool(torch.isfinite(coordinates).all()):
            raise SamplingFailure("coordinate reverse trajectory produced a non-finite state")
        return GeneratedCoordinateBatch(
            element_tokens=element_tokens,
            atomic_numbers=self.categorical.decode(element_tokens),
            fractional_coordinates=wrap01(coordinates),
            lattice=lattice,
            batch=blueprint.batch,
            diagnostics=CoordinateReverseDiagnostics(
                time=times[1:].detach().cpu(),
                coordinate_step_rms=torch.stack(coordinate_steps).detach().cpu(),
            ),
        )

    def initialize_lattice_state(
        self,
        blueprint: ParentBlueprintBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> LatticeReverseInitialState:
        """Draw a reusable standardized lattice prior without coordinates."""

        graphs = int(blueprint.node_counts.numel())
        reference = blueprint.shape_projector
        return LatticeReverseInitialState(
            volume_latent=torch.randn(
                (graphs,),
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            ),
            shape_latent=torch.randn(
                (graphs, 5),
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            ),
        )

    def sample_lattice(
        self,
        element_tokens: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        tensor_condition: torch.Tensor | None = None,
        steps: int = 100,
        initial_state: LatticeReverseInitialState | None = None,
        initialization_generator: torch.Generator | None = None,
        continuous_generator: torch.Generator | None = None,
        continuous_mode: ContinuousReverseMode = "reverse_sde",
        time_grid: str = "uniform_log_alpha",
    ) -> GeneratedLatticeBatch:
        """Reverse only ``L_t`` conditional on an unordered clean composition."""

        if steps < 1:
            raise ValueError("lattice reverse sampler requires at least one step")
        if self.denoiser.modality_time_conditioning != "separate":
            raise ValueError("lattice reverse sampling requires the unified separate-clock backbone")
        continuous_mode = _validate_continuous_mode(continuous_mode)
        graphs = int(blueprint.node_counts.numel())
        if element_tokens.shape != blueprint.batch.shape or element_tokens.dtype != torch.long:
            raise ValueError("lattice conditioning requires one int64 element token per node")
        if element_tokens.device != blueprint.batch.device:
            raise ValueError("element tokens and blueprint must share a device")
        self.categorical.validate_clean(element_tokens)
        composition_counts = self._clean_composition_counts(element_tokens, blueprint.batch, graphs)
        condition: torch.Tensor | None
        condition_present: torch.Tensor | None
        if tensor_condition is None:
            # Preserve the qualified Stage-C lattice-only path exactly: it
            # historically had no condition token at all.  Passing an
            # explicit null token would silently change that baseline if the
            # atlas null parameters are later calibrated.
            condition = None
            condition_present = None
        else:
            if (
                tensor_condition.shape != (graphs, 18)
                or tensor_condition.dtype != blueprint.shape_projector.dtype
                or tensor_condition.device != blueprint.batch.device
                or not bool(torch.isfinite(tensor_condition).all())
            ):
                raise ValueError("tensor condition must be finite [graphs,18] in the blueprint chart")
            condition = tensor_condition
            condition_present = torch.ones((graphs, 1), dtype=torch.bool, device=blueprint.batch.device)
        if initial_state is None:
            initial_state = self.initialize_lattice_state(
                blueprint,
                generator=initialization_generator,
            )
        expected = (
            (initial_state.volume_latent, (graphs,), "volume latent"),
            (initial_state.shape_latent, (graphs, 5), "shape latent"),
        )
        for value, shape, name in expected:
            if (
                value.shape != shape
                or value.dtype != blueprint.shape_projector.dtype
                or value.device != blueprint.shape_projector.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"initial {name} must match the blueprint")
        volume_latent = initial_state.volume_latent.clone()
        shape_latent = initial_state.shape_latent.clone()
        log_volume = self.lattice_standardizer.decode_volume(
            volume_latent,
            blueprint.node_counts,
        )
        log_shape = project_lattice_state(
            self.lattice_standardizer.decode_shape(shape_latent),
            blueprint.shape_projector,
        )
        times = reverse_time_grid(
            self.vp_schedule,
            self.maximum_time,
            steps,
            dtype=log_volume.dtype,
            device=log_volume.device,
            spacing=time_grid,
        )
        volume_steps: list[torch.Tensor] = []
        shape_steps: list[torch.Tensor] = []
        was_training = self.denoiser.training
        self.denoiser.eval()
        trajectory_error: RuntimeError | ValueError | None = None
        try:
            with torch.no_grad():
                for index in range(steps):
                    time_from = times[index].expand(graphs)
                    time_to = times[index + 1].expand(graphs)
                    lattice_kwargs = (
                        {"tensor_condition": condition, "condition_present": condition_present}
                        if condition is not None and condition_present is not None
                        else {}
                    )
                    prediction = self.denoiser.forward_lattice(
                        element_tokens,
                        log_volume,
                        log_shape,
                        blueprint.batch,
                        time_from,
                        blueprint.shape_projector,
                        composition_counts=composition_counts,
                        **lattice_kwargs,
                    )
                    next_volume_latent = vp_reverse_step(
                        self.vp_schedule,
                        volume_latent,
                        prediction.clean_volume_latent,
                        time_from,
                        time_to,
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )
                    next_shape_latent = vp_reverse_step(
                        self.vp_schedule,
                        shape_latent,
                        prediction.clean_shape_latent,
                        time_from.unsqueeze(-1),
                        time_to.unsqueeze(-1),
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )
                    next_volume = self.lattice_standardizer.decode_volume(
                        next_volume_latent,
                        blueprint.node_counts,
                    )
                    next_shape = project_lattice_state(
                        self.lattice_standardizer.decode_shape(next_shape_latent),
                        blueprint.shape_projector,
                    )
                    volume_steps.append((next_volume - log_volume).square().mean().sqrt())
                    shape_steps.append((next_shape - log_shape).square().mean().sqrt())
                    volume_latent = next_volume_latent
                    shape_latent = self.lattice_standardizer.encode_shape(next_shape)
                    log_volume = next_volume
                    log_shape = next_shape
        except (RuntimeError, ValueError) as error:
            trajectory_error = error
        finally:
            self.denoiser.train(was_training)
        if trajectory_error is not None:
            raise SamplingFailure(f"lattice reverse trajectory failed: {trajectory_error}") from trajectory_error
        if not all(torch.isfinite(value).all() for value in (log_volume, log_shape)):
            raise SamplingFailure("lattice reverse trajectory produced a non-finite state")
        try:
            lattice = LatticeVolumeShape(log_volume, log_shape).lattice(blueprint.fractional_to_cartesian)
            if self.guardrails is not None:
                self.guardrails.validate(lattice @ lattice.transpose(-1, -2))
        except (RuntimeError, ValueError) as error:
            raise SamplingFailure(f"terminal lattice is invalid: {error}") from error
        return GeneratedLatticeBatch(
            lattice=lattice,
            log_volume=log_volume,
            log_shape=log_shape,
            diagnostics=LatticeReverseDiagnostics(
                time=times[1:].detach().cpu(),
                volume_step_rms=torch.stack(volume_steps).detach().cpu(),
                shape_step_rms=torch.stack(shape_steps).detach().cpu(),
            ),
        )

    def sample_elements(
        self,
        blueprint: ParentBlueprintBatch,
        fractional_coordinates: torch.Tensor,
        lattice: torch.Tensor,
        *,
        steps: int = 100,
        categorical_generator: torch.Generator | None = None,
        time_grid: str = "uniform_log_alpha",
    ) -> GeneratedElementBatch:
        """Reverse only ``A_t`` while holding the observed ``(F_0,L_0)`` fixed.

        This is the E1 qualification path, not a partially disabled joint
        sampler.  The denoiser receives the explicit clock triple
        ``(t_A,t_F,t_L)=(t,0,0)`` and no continuous state is integrated.
        """

        if steps < 1:
            raise ValueError("element reverse sampler requires at least one step")
        if self.denoiser.modality_time_conditioning != "separate":
            raise ValueError("element reverse sampling requires the unified separate-clock backbone")
        device = blueprint.batch.device
        dtype = blueprint.shape_projector.dtype
        graphs = int(blueprint.node_counts.numel())
        nodes = int(blueprint.batch.numel())
        if fractional_coordinates.shape != (nodes, 3):
            raise ValueError("observed fractional coordinates must have shape [nodes,3]")
        if lattice.shape != (graphs, 3, 3):
            raise ValueError("observed lattice must have shape [graphs,3,3]")
        if (
            fractional_coordinates.device != device
            or lattice.device != device
            or fractional_coordinates.dtype != dtype
            or lattice.dtype != dtype
        ):
            raise ValueError("observed geometry must match the blueprint dtype and device")
        if not all(torch.isfinite(value).all() for value in (fractional_coordinates, lattice)):
            raise ValueError("observed geometry must be finite")

        coordinates = project_translation_state(
            fractional_coordinates,
            blueprint.batch,
            graphs,
        )
        lattice_state = LatticeVolumeShape.from_lattice(
            lattice,
            blueprint.fractional_to_cartesian,
        )
        log_volume = lattice_state.log_volume
        log_shape = torch.einsum(
            "bij,bj->bi",
            blueprint.shape_projector,
            lattice_state.log_shape,
        )
        tokens = self.categorical.sample_prior(
            nodes,
            blueprint.shape_projector,
            generator=categorical_generator,
        )
        condition = torch.zeros((graphs, 18), dtype=dtype, device=device)
        condition_present = torch.zeros((graphs, 1), dtype=torch.bool, device=device)
        clean_time = torch.zeros((graphs,), dtype=dtype, device=device)
        times = reverse_time_grid(
            self.vp_schedule,
            self.maximum_time,
            steps,
            dtype=dtype,
            device=device,
            spacing=time_grid,
        )
        masked_counts: list[torch.Tensor] = []
        was_training = self.denoiser.training
        self.denoiser.eval()
        trajectory_error: RuntimeError | ValueError | None = None
        try:
            with torch.no_grad():
                for index in range(steps):
                    time_from = times[index].expand(graphs)
                    time_to = times[index + 1].expand(graphs)
                    prediction = self.denoiser(
                        tokens,
                        coordinates,
                        log_volume,
                        log_shape,
                        blueprint.batch,
                        clean_time,
                        condition,
                        condition_present,
                        blueprint.shape_projector,
                        blueprint.fractional_to_cartesian,
                        element_time=time_from,
                        lattice_time=clean_time,
                    )
                    probabilities = self.categorical.reverse_probabilities(
                        tokens,
                        prediction.clean_element_logits,
                        time_from,
                        time_to,
                        blueprint.batch,
                    )
                    tokens = torch.multinomial(
                        probabilities,
                        1,
                        replacement=True,
                        generator=categorical_generator,
                    ).squeeze(-1)
                    masked_counts.append((tokens == self.categorical.mask_index).sum())
        except (RuntimeError, ValueError) as error:
            trajectory_error = error
        finally:
            self.denoiser.train(was_training)
        if trajectory_error is not None:
            raise SamplingFailure(f"element reverse trajectory failed: {trajectory_error}") from trajectory_error
        if bool((tokens == self.categorical.mask_index).any()):
            raise SamplingFailure("terminal categorical state contains absorbing masks")
        if isinstance(self.categorical, UniformCategoricalDiffusion):
            tokens, composition_counts = count_projected_assignment(
                prediction.clean_element_logits,
                prediction.clean_composition_logits,
                blueprint.batch,
                blueprint.node_counts,
            )
        else:
            composition_counts = composition_counts_from_tokens(
                tokens,
                blueprint.batch,
                graphs,
            )
        return GeneratedElementBatch(
            element_tokens=tokens,
            atomic_numbers=self.categorical.decode(tokens),
            batch=blueprint.batch,
            predicted_composition_counts=composition_counts,
            terminal_clean_element_logits=prediction.clean_element_logits,
            terminal_clean_composition_logits=prediction.clean_composition_logits,
            diagnostics=ElementReverseDiagnostics(
                time=times[1:].detach().cpu(),
                masked_count=torch.stack(masked_counts).detach().cpu(),
            ),
        )

    def initialize_continuous_state(
        self,
        blueprint: ParentBlueprintBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ContinuousReverseInitialState:
        """Draw one reusable continuous prior state for common-noise audits."""
        device = blueprint.batch.device
        dtype = blueprint.shape_projector.dtype
        graphs = blueprint.node_counts.numel()
        nodes = blueprint.batch.numel()
        coordinates = torch.rand((nodes, 3), dtype=dtype, device=device, generator=generator)
        return ContinuousReverseInitialState(
            fractional_coordinates=project_translation_state(coordinates, blueprint.batch, graphs),
            volume_latent=torch.randn((graphs,), dtype=dtype, device=device, generator=generator),
            shape_latent=torch.randn((graphs, 5), dtype=dtype, device=device, generator=generator),
        )

    @staticmethod
    def _validate_initial_state(
        state: ContinuousReverseInitialState,
        blueprint: ParentBlueprintBatch,
    ) -> None:
        graphs = blueprint.node_counts.numel()
        nodes = blueprint.batch.numel()
        dtype = blueprint.shape_projector.dtype
        device = blueprint.batch.device
        expected = (
            (state.fractional_coordinates, (nodes, 3), "fractional coordinates"),
            (state.volume_latent, (graphs,), "volume latent"),
            (state.shape_latent, (graphs, 5), "shape latent"),
        )
        for value, shape, name in expected:
            if value.shape != shape or value.dtype != dtype or value.device != device:
                raise ValueError(f"initial {name} must match the blueprint shape, dtype and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"initial {name} must be finite")

    def sample(
        self,
        blueprint: ParentBlueprintBatch,
        *,
        tensor_condition: torch.Tensor | None = None,
        composition_counts: torch.Tensor | None = None,
        steps: int = 100,
        initial_state: ContinuousReverseInitialState | None = None,
        initialization_generator: torch.Generator | None = None,
        categorical_generator: torch.Generator | None = None,
        continuous_generator: torch.Generator | None = None,
        continuous_mode: ContinuousReverseMode = "reverse_sde",
        time_grid: str = "uniform_log_alpha",
    ) -> GeneratedHybridBatch:
        if steps < 1:
            raise ValueError("reverse sampler requires at least one step")
        continuous_mode = _validate_continuous_mode(continuous_mode)
        if self.denoiser.modality_time_conditioning != "separate":
            raise ValueError(
                "product-space joint sampling requires separate modality clocks for partial occupation"
            )
        device = blueprint.batch.device
        dtype = blueprint.shape_projector.dtype
        graphs = blueprint.node_counts.numel()
        nodes = blueprint.batch.numel()
        if not self.orderless_reveal:
            raise ValueError(
                "the joint sampler is defined only for the composition-conditioned "
                "orderless exact-count product path; use the component samplers for "
                "historical conditional audits"
            )
        assert self.composition_model is not None
        if int(blueprint.node_counts.max()) > self.composition_model.maximum_atoms:
            raise ValueError("blueprint node count exceeds the qualified composition-law support")
        if self.composition_model.vocabulary_size != self.categorical.element_count:
            raise ValueError("composition and categorical vocabularies disagree")
        # The qualified C law is unconditional conditional on N: its frozen
        # context is the constant unit token used during its likelihood gate.
        # This is an explicit model state, never a target-derived composition.
        if composition_counts is None:
            composition_context = torch.ones(
                (graphs, self.composition_model.context_dim), dtype=dtype, device=device
            )
            composition_sample = self.composition_model.sample(
                composition_context,
                blueprint.node_counts.long(),
                generator=categorical_generator,
            )
            composition_counts = composition_sample.state.to_dense(self.categorical.element_count)
        else:
            if (
                composition_counts.shape != (graphs, self.categorical.element_count)
                or composition_counts.dtype != torch.long
                or composition_counts.device != device
                or bool((composition_counts < 0).any())
            ):
                raise ValueError("fixed composition counts must be nonnegative [graphs,elements] int64")
            composition_counts = composition_counts.clone()
        if not torch.equal(composition_counts.sum(dim=1), blueprint.node_counts.long()):
            raise SamplingFailure("sampled composition does not close on the sampled node count")
        tokens = self.categorical.sample_prior(nodes, blueprint.shape_projector, generator=categorical_generator)
        reveal_rank = sample_uniform_reveal_ranks(blueprint.batch, generator=categorical_generator)
        reveal_count = torch.zeros((graphs,), dtype=torch.long, device=device)
        remaining_counts = composition_counts.clone()
        if initial_state is None:
            initial_state = self.initialize_continuous_state(blueprint, generator=initialization_generator)
        else:
            self._validate_initial_state(initial_state, blueprint)
        coordinates = project_translation_state(initial_state.fractional_coordinates.clone(), blueprint.batch, graphs)
        volume_latent = initial_state.volume_latent.clone()
        shape_latent = initial_state.shape_latent.clone()
        log_volume = self.lattice_standardizer.decode_volume(volume_latent, blueprint.node_counts)
        log_shape = self.lattice_standardizer.decode_shape(shape_latent)
        log_shape = torch.einsum("bij,bj->bi", blueprint.shape_projector, log_shape)
        if tensor_condition is None:
            condition = torch.zeros((graphs, 18), dtype=dtype, device=device)
            condition_present = torch.zeros(
                (graphs, 1), dtype=torch.bool, device=device
            )
        else:
            if (
                tensor_condition.shape != (graphs, 18)
                or tensor_condition.dtype != dtype
                or tensor_condition.device != device
                or not bool(torch.isfinite(tensor_condition).all())
            ):
                raise ValueError(
                    "tensor condition must be finite [graphs,18] in the blueprint chart"
                )
            # A supplied all-zero tensor is a physical zero condition, not the
            # missing/null branch. Absence is represented only by ``None``.
            condition = tensor_condition
            condition_present = torch.ones(
                (graphs, 1), dtype=torch.bool, device=device
            )
        times = reverse_time_grid(
            self.vp_schedule,
            self.maximum_time,
            steps,
            dtype=dtype,
            device=device,
            spacing=time_grid,
        )

        masked_counts: list[torch.Tensor] = []
        remaining_atom_counts: list[torch.Tensor] = []
        composition_closure_errors: list[torch.Tensor] = []
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
                    # This is a first-order product-space split: continuous
                    # drift is evaluated on the complete state at ``t_from``;
                    # discrete reveals then use the exact count kernel.  Do
                    # not feed a newly revealed A state into an old-time
                    # continuous drift, which would define neither component
                    # reverse transition.
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
                        element_time=time_from,
                        lattice_time=time_from,
                        composition_counts=composition_counts,
                    )
                    # Reveal according to the same time-to-prefix convention
                    # used by the product-field objective.  A fixed uniform
                    # auxiliary order is sampled once; every reveal is legal
                    # under the persisted remaining-count state.  There is no
                    # independent-site kernel and no terminal count repair.
                    target_reveal = torch.floor(
                        (1.0 - scalar_to) * blueprint.node_counts.to(dtype)
                    ).long().clamp_max(blueprint.node_counts)
                    categorical_prediction = prediction
                    while bool((reveal_count < target_reveal).any()):
                        active = reveal_count < target_reveal
                        next_site = torch.nonzero(
                            reveal_rank == reveal_count[blueprint.batch], as_tuple=False
                        ).flatten()
                        next_site = next_site[active[blueprint.batch[next_site]]]
                        active_graph = torch.nonzero(active, as_tuple=False).flatten()
                        if next_site.shape != active_graph.shape or not torch.equal(
                            blueprint.batch[next_site], active_graph
                        ):
                            raise RuntimeError("orderless reveal order lost one next site per active graph")
                        active_site = next_site
                        log_probability = self.assignment_law.batched_step_log_probabilities(
                            categorical_prediction.clean_element_logits[active_site],
                            remaining_counts[active],
                        )
                        selected = torch.multinomial(
                            log_probability.exp(), 1, replacement=True, generator=categorical_generator
                        ).squeeze(1)
                        tokens[active_site] = selected
                        remaining_counts[active, selected] -= 1
                        reveal_count[active] += 1
                        # Rare coarse grids can cross more than one reveal
                        # threshold.  Re-evaluate only the additional
                        # categorical substep at its destination clock; the
                        # continuous drift remains the original t_from field.
                        if bool((reveal_count < target_reveal).any()):
                            categorical_prediction = self.denoiser(
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
                                element_time=time_to,
                                lattice_time=time_from,
                                composition_counts=composition_counts,
                            )

                    variance_from = self.coordinate_schedule.variance(time_from)
                    variance_to = self.coordinate_schedule.variance(time_to)
                    next_coordinates = quotient_coordinate_reverse_step(
                        coordinates,
                        prediction.coordinate_fractional_scaled_score,
                        variance_from,
                        variance_to,
                        blueprint.batch,
                        graphs,
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )

                    next_volume_latent = vp_reverse_step(
                        self.vp_schedule,
                        volume_latent,
                        prediction.clean_volume_latent,
                        time_from,
                        time_to,
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )
                    next_shape_latent = vp_reverse_step(
                        self.vp_schedule,
                        shape_latent,
                        prediction.clean_shape_latent,
                        time_from.unsqueeze(-1),
                        time_to.unsqueeze(-1),
                        generator=continuous_generator,
                        mode=continuous_mode,
                    )
                    next_volume = self.lattice_standardizer.decode_volume(next_volume_latent, blueprint.node_counts)
                    next_shape = self.lattice_standardizer.decode_shape(next_shape_latent)
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
                    observed_partial = composition_counts_from_tokens(
                        tokens[tokens != self.categorical.mask_index],
                        blueprint.batch[tokens != self.categorical.mask_index],
                        graphs,
                    )
                    composition_closure_errors.append(
                        (observed_partial + remaining_counts - composition_counts).abs().sum(dim=1)
                    )
                    remaining_atom_counts.append(remaining_counts.sum(dim=1))
        except (RuntimeError, ValueError) as error:
            trajectory_error = error
        finally:
            self.denoiser.train(was_training)

        if trajectory_error is not None:
            raise SamplingFailure(f"joint reverse trajectory failed: {trajectory_error}") from trajectory_error

        if bool((tokens == self.categorical.mask_index).any()) or bool((reveal_count != blueprint.node_counts).any()):
            raise SamplingFailure("terminal categorical state contains absorbing masks")
        if bool(remaining_counts.any()):
            raise SamplingFailure("terminal orderless assignment did not consume its composition")
        observed_counts = composition_counts_from_tokens(tokens, blueprint.batch, graphs)
        if not torch.equal(observed_counts, composition_counts):
            raise SamplingFailure("terminal orderless assignment violates its sampled composition")
        if not all(torch.isfinite(value).all() for value in (coordinates, log_volume, log_shape)):
            raise SamplingFailure("reverse trajectory produced a non-finite continuous state")
        try:
            lattice = LatticeVolumeShape(log_volume, log_shape).lattice(blueprint.fractional_to_cartesian)
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
            composition_counts=composition_counts,
            batch=blueprint.batch,
            diagnostics=ReverseTrajectoryDiagnostics(
                time=times[1:].detach().cpu(),
                masked_count=torch.stack(masked_counts).detach().cpu(),
                remaining_atom_count=torch.stack(remaining_atom_counts).detach().cpu(),
                composition_closure_error=torch.stack(composition_closure_errors).detach().cpu(),
                coordinate_step_rms=torch.stack(coordinate_steps).detach().cpu(),
                volume_step_rms=torch.stack(volume_steps).detach().cpu(),
                shape_step_rms=torch.stack(shape_steps).detach().cpu(),
            ),
        )
