"""Continuous-time schedules shared by the production hybrid diffusion."""

from __future__ import annotations

import math

import torch


def standard_normal(
    shape: torch.Size | tuple[int, ...],
    reference: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample standard normal noise matching a state tensor's dtype/device."""
    return torch.randn(shape, dtype=reference.dtype, device=reference.device, generator=generator)


class CosineNoiseSchedule:
    """Variance-preserving cosine schedule with clean time zero.

    ``alpha(0)=1`` and ``alpha(1)=0``.  The returned standard deviation obeys
    ``alpha(t)^2 + sigma(t)^2 = 1``.  The same convention is used by the
    absorbing categorical path and the harmonic condition gate.
    """

    def __init__(self, *, minimum_sigma: float = 1e-5) -> None:
        if not 0.0 < minimum_sigma < 1.0:
            raise ValueError("minimum_sigma must lie in (0, 1)")
        self.minimum_sigma = float(minimum_sigma)

    @staticmethod
    def _validate(time: torch.Tensor) -> None:
        if not time.dtype.is_floating_point or not torch.isfinite(time).all():
            raise ValueError("time must be a finite floating tensor")
        if bool(((time < 0.0) | (time > 1.0)).any()):
            raise ValueError("time must lie in [0, 1]")

    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        self._validate(time)
        return torch.cos(0.5 * math.pi * time)

    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        self._validate(time)
        return torch.sin(0.5 * math.pi * time)

    def snr(self, time: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha(time)
        sigma = self.sigma(time).clamp_min(self.minimum_sigma)
        return alpha.square() / sigma.square()

    def posterior_variance(
        self,
        time_from: torch.Tensor,
        time_to: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM posterior variance for one reverse VP transition.

        ``time_from`` is the noisier endpoint and ``time_to`` the cleaner one.
        The expression is evaluated from the cumulative cosine survival rather
        than a singular continuous-time beta near ``t=1``.
        """
        self._validate(time_from)
        self._validate(time_to)
        if time_from.shape != time_to.shape or bool((time_to > time_from).any()):
            raise ValueError("reverse VP endpoints must have equal shape and time_to <= time_from")
        survival_from = self.alpha(time_from).square()
        survival_to = self.alpha(time_to).square()
        step_noise = (1.0 - survival_from / survival_to.clamp_min(self.minimum_sigma**2)).clamp(0.0, 1.0)
        return step_noise * (1.0 - survival_to) / (1.0 - survival_from).clamp_min(self.minimum_sigma**2)


class FractionalTorusVarianceSchedule:
    """Cell-independent Brownian variance on the fractional torus.

    This path is a genuine product Markov process with the independently
    diffused lattice chart. ``sigma_max`` is dimensionless; at one, the first
    non-zero Fourier mode has residual ``exp(-2*pi^2)``, so the finite terminal
    wrapped Gaussian is numerically matched to the uniform torus prior.
    """

    def __init__(self, *, sigma_max: float = 1.0) -> None:
        if sigma_max <= 0.0 or not math.isfinite(sigma_max):
            raise ValueError("sigma_max must be finite and positive")
        self.sigma_max = float(sigma_max)

    def variance(self, time: torch.Tensor) -> torch.Tensor:
        CosineNoiseSchedule._validate(time)
        return self.sigma_max**2 * time

    def sigma(self, time: torch.Tensor) -> torch.Tensor:
        return self.variance(time).sqrt()

    def increment(self, time_from: torch.Tensor, time_to: torch.Tensor) -> torch.Tensor:
        if time_from.shape != time_to.shape or bool((time_to > time_from).any()):
            raise ValueError("reverse wrapped endpoints must have equal shape and time_to <= time_from")
        return self.variance(time_from) - self.variance(time_to)
