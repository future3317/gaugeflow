"""Continuous-time schedules shared by the production hybrid diffusion."""

from __future__ import annotations

import math

import torch


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
