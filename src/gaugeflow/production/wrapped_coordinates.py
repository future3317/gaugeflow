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
