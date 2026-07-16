"""Adaptive wrapped Gaussian on periodic coordinates modulo translation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WrappedQuotientEvaluation:
    log_unnormalized_density: torch.Tensor
    fractional_score: torch.Tensor
    image_count: int
    radius: int
    omitted_weight_upper_bound: float


@dataclass(frozen=True)
class ScalableWrappedQuotientEvaluation:
    """QMC quotient evaluation with explicit kernel and refinement diagnostics."""

    log_unnormalized_density: torch.Tensor
    fractional_score: torch.Tensor
    qmc_samples: int
    kernel_representation: str
    kernel_terms: int
    kernel_tail_upper_bound: float
    qmc_log_increment: float
    qmc_relative_score_increment: float


def translation_horizontal(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("periodic coordinates must have shape [sites,3]")
    return value - value.mean(dim=0, keepdim=True)


class AdaptiveWrappedQuotient:
    """Evaluate the manuscript's quotient kernel with a proved tail stop.

    Integer images are quotiented by their common translation by fixing the
    final site's image to zero.  The remaining ``3(M-1)`` integer lattice is
    expanded around its continuous optimum.  Expansion stops only when a
    singular-value Gaussian bound on every omitted image is below the declared
    absolute or relative tolerance.  ``max_images`` is a fail-closed resource
    guard; it never converts into a fixed-shell approximation.
    """

    def __init__(
        self,
        *,
        absolute_tail_tolerance: float = 1e-10,
        relative_tail_tolerance: float = 1e-8,
        max_images: int = 2_000_000,
        max_radius: int = 64,
    ) -> None:
        if absolute_tail_tolerance <= 0 or relative_tail_tolerance <= 0:
            raise ValueError("tail tolerances must be positive")
        if max_images < 1 or max_radius < 0:
            raise ValueError("adaptive resource guards must be positive")
        self.absolute_tail_tolerance = float(absolute_tail_tolerance)
        self.relative_tail_tolerance = float(relative_tail_tolerance)
        self.max_images = int(max_images)
        self.max_radius = int(max_radius)

    @staticmethod
    def _one_dimensional_tail(a: float, radius: int) -> float:
        if a <= 0:
            return math.inf
        lower = radius + 0.5
        integral = 0.5 * math.sqrt(math.pi / a) * math.erfc(math.sqrt(a) * lower)
        return 2.0 * (math.exp(-a * lower * lower) + integral)

    @classmethod
    def _box_tail_bound(cls, a: float, dimension: int, radius: int) -> float:
        tail = cls._one_dimensional_tail(a, radius)
        total = 1.0 + cls._one_dimensional_tail(a, 0)
        return dimension * tail * (total ** max(dimension - 1, 0))

    @staticmethod
    def _continuous_image_center(delta: torch.Tensor) -> torch.Tensor:
        # With K_M=0, minimizing ||P_M(delta-K)|| gives K_i=delta_i-delta_M.
        return delta[:-1] - delta[-1]

    @staticmethod
    def _quadratic_minimum_eigenvalue(sites: int, metric: torch.Tensor) -> float:
        projector = torch.eye(sites, dtype=metric.dtype, device=metric.device)
        projector = projector - torch.ones_like(projector) / sites
        reduced = projector[:-1, :-1]
        value = torch.linalg.eigvalsh(reduced)[0] * torch.linalg.eigvalsh(metric)[0]
        return float(value.detach().cpu())

    def evaluate(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        lattice: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> WrappedQuotientEvaluation:
        if current.shape != clean.shape or current.ndim != 2 or current.shape[-1] != 3:
            raise ValueError("current and clean coordinates must share shape [sites,3]")
        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape [3,3]")
        if current.shape[0] < 1:
            raise ValueError("translation quotient requires at least one asymmetric site")
        if not torch.isfinite(current).all() or not torch.isfinite(clean).all() or not torch.isfinite(lattice).all():
            raise ValueError("wrapped-kernel inputs must be finite")
        sigma_tensor = torch.as_tensor(sigma, dtype=current.dtype, device=current.device)
        if sigma_tensor.ndim != 0 or float(sigma_tensor) <= 0:
            raise ValueError("sigma must be a positive scalar")
        if current.shape[0] == 1:
            # P_1 is exactly zero: a single asymmetric coordinate is pure
            # global translation and has no quotient degree of freedom.
            zero = (current - clean).sum() * 0.0
            return WrappedQuotientEvaluation(
                log_unnormalized_density=zero,
                fractional_score=torch.zeros_like(current),
                image_count=1,
                radius=0,
                omitted_weight_upper_bound=0.0,
            )
        metric = lattice @ lattice.transpose(-1, -2)
        minimum_eigenvalue = self._quadratic_minimum_eigenvalue(current.shape[0], metric)
        if minimum_eigenvalue <= 0:
            raise ValueError("wrapped kernel requires an invertible positive metric")
        dimension = 3 * (current.shape[0] - 1)
        a = minimum_eigenvalue / (2.0 * float(sigma_tensor.detach().cpu()) ** 2)
        delta = current - clean
        center = self._continuous_image_center(delta)
        nearest = torch.round(center).to(dtype=torch.long)

        selected_images: torch.Tensor | None = None
        selected_bound = math.inf
        selected_radius = -1
        for radius in range(self.max_radius + 1):
            count = (2 * radius + 1) ** dimension
            if count > self.max_images:
                raise RuntimeError(
                    "adaptive wrapped image sum exceeded max_images before its Gaussian-tail proof closed"
                )
            offsets = torch.tensor(
                list(itertools.product(range(-radius, radius + 1), repeat=dimension)),
                dtype=torch.long,
                device=current.device,
            ).reshape(count, current.shape[0] - 1, 3)
            images = nearest.unsqueeze(0) + offsets
            full_images = torch.cat(
                (images, torch.zeros((count, 1, 3), dtype=torch.long, device=current.device)),
                dim=1,
            ).to(current)
            # ``translation_horizontal`` is rank-two, so perform the batched
            # projection explicitly here.
            displacement = delta.unsqueeze(0) - full_images
            displacement = displacement - displacement.mean(dim=1, keepdim=True)
            cartesian = displacement @ lattice
            exponent = -cartesian.square().sum(dim=(-1, -2)) / (2.0 * sigma_tensor.square())
            partial_weight = float(torch.exp(exponent).sum().detach().cpu())
            bound = self._box_tail_bound(a, dimension, radius)
            if bound <= self.absolute_tail_tolerance or bound <= self.relative_tail_tolerance * partial_weight:
                selected_images = full_images
                selected_bound = bound
                selected_radius = radius
                break
        if selected_images is None:
            raise RuntimeError("adaptive wrapped image sum did not meet its tail tolerance")

        displacement = delta.unsqueeze(0) - selected_images
        displacement = displacement - displacement.mean(dim=1, keepdim=True)
        cartesian = displacement @ lattice
        logits = -cartesian.square().sum(dim=(-1, -2)) / (2.0 * sigma_tensor.square())
        weights = torch.softmax(logits, dim=0)
        mean_displacement = (weights[:, None, None] * displacement).sum(dim=0)
        score = -(mean_displacement @ metric) / sigma_tensor.square()
        score = translation_horizontal(score)
        return WrappedQuotientEvaluation(
            log_unnormalized_density=torch.logsumexp(logits, dim=0),
            fractional_score=score,
            image_count=selected_images.shape[0],
            radius=selected_radius,
            omitted_weight_upper_bound=selected_bound,
        )


class ScalableWrappedQuotient:
    """Translation-QMC evaluation with adaptive three-dimensional kernels.

    The exponential ``3(M-1)`` image lattice is replaced by a three-dimensional
    integral over the common translation.  Each one-site wrapped Gaussian uses
    either a direct image sum or its Poisson/Fourier dual, whichever has the
    smaller certified truncation.  A dyadically nested torus lattice rule
    refines the remaining integral and fails closed if successive estimates do
    not meet the declared numerical tolerances.

    The returned log density uses the same unnormalized convention as
    :class:`AdaptiveWrappedQuotient`; the analytic common-translation Gaussian
    factor is removed after QMC integration.
    """

    def __init__(
        self,
        *,
        kernel_tail_tolerance: float = 1e-12,
        qmc_log_tolerance: float = 2e-7,
        qmc_relative_score_tolerance: float = 5e-5,
        minimum_grid_power: int = 3,
        maximum_grid_power: int = 7,
        max_kernel_radius: int = 24,
        max_kernel_terms: int = 250_000,
        chunk_size: int = 4096,
    ) -> None:
        if kernel_tail_tolerance <= 0 or qmc_log_tolerance <= 0:
            raise ValueError("wrapped quotient tolerances must be positive")
        if qmc_relative_score_tolerance <= 0:
            raise ValueError("wrapped quotient score tolerance must be positive")
        if not 1 <= minimum_grid_power < maximum_grid_power:
            raise ValueError("torus-grid powers must satisfy 1 <= minimum < maximum")
        if max_kernel_radius < 0 or max_kernel_terms < 1 or chunk_size < 1:
            raise ValueError("wrapped quotient resource guards must be positive")
        self.kernel_tail_tolerance = float(kernel_tail_tolerance)
        self.qmc_log_tolerance = float(qmc_log_tolerance)
        self.qmc_relative_score_tolerance = float(qmc_relative_score_tolerance)
        self.minimum_grid_power = int(minimum_grid_power)
        self.maximum_grid_power = int(maximum_grid_power)
        self.max_kernel_radius = int(max_kernel_radius)
        self.max_kernel_terms = int(max_kernel_terms)
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _tail_bound(a: float, radius: int) -> float:
        return AdaptiveWrappedQuotient._box_tail_bound(a, 3, radius)

    def _radius_for_tail(self, a: float) -> tuple[int, float]:
        for radius in range(self.max_kernel_radius + 1):
            terms = (2 * radius + 1) ** 3
            if terms > self.max_kernel_terms:
                break
            bound = self._tail_bound(a, radius)
            if bound <= self.kernel_tail_tolerance:
                return radius, bound
        raise RuntimeError("single-site wrapped kernel did not meet its truncation bound")

    @staticmethod
    def _integer_cube(radius: int, *, device: torch.device) -> torch.Tensor:
        axis = torch.arange(-radius, radius + 1, dtype=torch.long, device=device)
        return torch.cartesian_prod(axis, axis, axis).reshape(-1, 3)

    def _choose_representation(
        self,
        metric: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[str, int, float]:
        sigma_value = float(sigma.detach().cpu())
        real_a = float(torch.linalg.eigvalsh(metric)[0].detach().cpu()) / (2.0 * sigma_value**2)
        inverse_metric = torch.linalg.inv(metric)
        fourier_a = (
            2.0
            * math.pi**2
            * sigma_value**2
            * float(torch.linalg.eigvalsh(inverse_metric)[0].detach().cpu())
        )
        try:
            real_radius, real_bound = self._radius_for_tail(real_a)
        except RuntimeError:
            real_radius, real_bound = self.max_kernel_radius + 1, math.inf
        try:
            fourier_radius, fourier_bound = self._radius_for_tail(fourier_a)
        except RuntimeError:
            fourier_radius, fourier_bound = self.max_kernel_radius + 1, math.inf
        if math.isinf(real_bound) and math.isinf(fourier_bound):
            raise RuntimeError("neither wrapped-kernel representation met its truncation bound")
        if (2 * real_radius + 1) ** 3 <= (2 * fourier_radius + 1) ** 3:
            return "image", real_radius, real_bound
        return "fourier", fourier_radius, fourier_bound

    def _image_kernel(
        self,
        delta: torch.Tensor,
        metric: torch.Tensor,
        sigma: torch.Tensor,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._integer_cube(radius, device=delta.device).to(delta)
        nearest = torch.round(delta).detach()
        displacement = delta.unsqueeze(-2) - (nearest.unsqueeze(-2) + images)
        quadratic = torch.einsum("...ki,ij,...kj->...k", displacement, metric, displacement)
        logits = -quadratic / (2.0 * sigma.square())
        weights = torch.softmax(logits, dim=-1)
        mean_displacement = (weights.unsqueeze(-1) * displacement).sum(dim=-2)
        score = -(mean_displacement @ metric) / sigma.square()
        return torch.logsumexp(logits, dim=-1), score

    def _fourier_kernel(
        self,
        delta: torch.Tensor,
        metric: torch.Tensor,
        sigma: torch.Tensor,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modes = self._integer_cube(radius, device=delta.device).to(delta)
        inverse_metric = torch.linalg.inv(metric)
        quadratic = torch.einsum("ki,ij,kj->k", modes, inverse_metric, modes)
        coefficients = torch.exp(-2.0 * math.pi**2 * sigma.square() * quadratic)
        phase = 2.0 * math.pi * torch.einsum("...i,ki->...k", delta, modes)
        terms = coefficients * phase.cos()
        series = terms.sum(dim=-1)
        roundoff_floor = 64.0 * torch.finfo(series.dtype).eps * coefficients.sum()
        if bool((series <= roundoff_floor).any()):
            raise RuntimeError(
                "Fourier wrapped kernel lost positivity; use a tighter dual representation"
            )
        derivative = -2.0 * math.pi * torch.einsum(
            "...k,ki->...i", coefficients * phase.sin(), modes
        )
        log_prefactor = (
            1.5 * torch.log(2.0 * math.pi * sigma.square())
            - 0.5 * torch.logdet(metric)
        )
        return log_prefactor + series.log(), derivative / series.unsqueeze(-1)

    def _kernel(
        self,
        delta: torch.Tensor,
        metric: torch.Tensor,
        sigma: torch.Tensor,
        representation: str,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if representation == "image":
            return self._image_kernel(delta, metric, sigma, radius)
        return self._fourier_kernel(delta, metric, sigma, radius)

    def _translation_mode(
        self,
        delta: torch.Tensor,
        metric: torch.Tensor,
        sigma: torch.Tensor,
        representation: str,
        radius: int,
    ) -> torch.Tensor:
        """Deterministically refine a common-translation mode for QMC shifting."""
        phase = 2.0 * math.pi * delta.detach()
        translation = torch.atan2(phase.sin().mean(dim=0), phase.cos().mean(dim=0)) / (
            2.0 * math.pi
        )
        inverse_metric = torch.linalg.inv(metric)
        for _ in range(12):
            _, site_score = self._kernel(
                delta.detach() - translation, metric, sigma, representation, radius
            )
            gradient = -site_score.sum(dim=0)
            # This is the exact Newton step within a fixed image branch and a
            # stable Fisher-preconditioned ascent step for a wrapped mixture.
            step = (gradient @ inverse_metric) * sigma.square() / delta.shape[0]
            translation = torch.remainder(translation + step, 1.0)
            if float(torch.linalg.vector_norm(step).detach().cpu()) < 1e-12:
                break
        return translation.detach()

    def evaluate(
        self,
        current: torch.Tensor,
        clean: torch.Tensor,
        lattice: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> ScalableWrappedQuotientEvaluation:
        if current.shape != clean.shape or current.ndim != 2 or current.shape[-1] != 3:
            raise ValueError("current and clean coordinates must share shape [sites,3]")
        if lattice.shape != (3, 3) or current.shape[0] < 1:
            raise ValueError("translation quotient requires a [3,3] lattice and at least one site")
        if not torch.isfinite(current).all() or not torch.isfinite(clean).all():
            raise ValueError("wrapped-kernel coordinates must be finite")
        if not torch.isfinite(lattice).all():
            raise ValueError("wrapped-kernel lattice must be finite")
        sigma_tensor = torch.as_tensor(sigma, dtype=current.dtype, device=current.device)
        if sigma_tensor.ndim != 0 or float(sigma_tensor) <= 0:
            raise ValueError("sigma must be a positive scalar")
        sites = current.shape[0]
        if sites == 1:
            zero = (current - clean).sum() * 0.0
            return ScalableWrappedQuotientEvaluation(
                log_unnormalized_density=zero,
                fractional_score=torch.zeros_like(current),
                qmc_samples=1,
                kernel_representation="quotient-zero",
                kernel_terms=1,
                kernel_tail_upper_bound=0.0,
                qmc_log_increment=0.0,
                qmc_relative_score_increment=0.0,
            )
        metric = lattice @ lattice.transpose(-1, -2)
        if float(torch.linalg.eigvalsh(metric)[0]) <= 0:
            raise ValueError("wrapped kernel requires an invertible positive metric")
        representation, radius, tail_bound = self._choose_representation(metric, sigma_tensor)
        terms = (2 * radius + 1) ** 3
        delta = current - clean
        # Anchor one lattice node at a deterministic circular estimate of the
        # common-translation mode.  The torus integral is shift invariant, but
        # resolving a sigma/sqrt(M) peak with an unshifted grid can otherwise
        # require a needlessly fine global mesh for large M.
        qmc_shift = self._translation_mode(
            delta, metric, sigma_tensor, representation, radius
        )
        previous_log: torch.Tensor | None = None
        previous_score: torch.Tensor | None = None
        final_log_increment = math.inf
        final_score_increment = math.inf
        converged = False
        # A dyadically nested rank-three torus lattice rule is used instead of
        # pseudo-random points.  The periodic trapezoidal/QMC rule has spectral
        # convergence for the analytic wrapped integrand, and each refinement
        # contains every node from the preceding level.
        for power in range(self.minimum_grid_power, self.maximum_grid_power + 1):
            side = 2**power
            axis = torch.arange(side, dtype=current.dtype, device=current.device) / side
            points = torch.remainder(
                torch.cartesian_prod(axis, axis, axis).reshape(-1, 3) + qmc_shift, 1.0
            )
            target_samples = points.shape[0]
            accumulated_log_weight = current.new_tensor(-torch.inf)
            accumulated_score = torch.zeros_like(current)
            consumed = 0
            while consumed < target_samples:
                stop = min(consumed + self.chunk_size, target_samples)
                translation = points[consumed:stop]
                shifted = delta.unsqueeze(0) - translation.unsqueeze(1)
                log_kernel, site_score = self._kernel(
                    shifted.reshape(-1, 3), metric, sigma_tensor, representation, radius
                )
                log_integrand = log_kernel.reshape(stop - consumed, sites).sum(dim=-1)
                site_score = site_score.reshape(stop - consumed, sites, 3)
                chunk_log_weight = torch.logsumexp(log_integrand, dim=0)
                chunk_score = (
                    torch.softmax(log_integrand, dim=0)[:, None, None] * site_score
                ).sum(dim=0)
                combined_log_weight = torch.logaddexp(accumulated_log_weight, chunk_log_weight)
                old_fraction = torch.exp(accumulated_log_weight - combined_log_weight)
                new_fraction = torch.exp(chunk_log_weight - combined_log_weight)
                accumulated_score = old_fraction * accumulated_score + new_fraction * chunk_score
                accumulated_log_weight = combined_log_weight
                consumed = stop
            log_integral = accumulated_log_weight - math.log(float(target_samples))
            score = translation_horizontal(accumulated_score)
            if previous_log is not None and previous_score is not None:
                final_log_increment = float((log_integral - previous_log).abs().detach().cpu())
                denominator = torch.linalg.vector_norm(previous_score).clamp_min(1e-8)
                final_score_increment = float(
                    (torch.linalg.vector_norm(score - previous_score) / denominator).detach().cpu()
                )
                if (
                    final_log_increment <= self.qmc_log_tolerance
                    and final_score_increment <= self.qmc_relative_score_tolerance
                ):
                    converged = True
                    break
            previous_log = log_integral
            previous_score = score
        if not converged:
            raise RuntimeError("nested torus QMC did not meet its refinement tolerance")
        translation_log_factor = (
            1.5 * torch.log(2.0 * math.pi * sigma_tensor.square() / sites)
            - 0.5 * torch.logdet(metric)
        )
        return ScalableWrappedQuotientEvaluation(
            log_unnormalized_density=log_integral - translation_log_factor,
            fractional_score=score,
            qmc_samples=target_samples,
            kernel_representation=representation,
            kernel_terms=terms,
            kernel_tail_upper_bound=tail_bound,
            qmc_log_increment=final_log_increment,
            qmc_relative_score_increment=final_score_increment,
        )
