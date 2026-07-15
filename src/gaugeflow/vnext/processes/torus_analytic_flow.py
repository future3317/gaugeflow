"""Smooth finite-time diffeomorphic flow on periodic relative coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def wrap_centered(value: torch.Tensor) -> torch.Tensor:
    """Wrap fractional values to ``[-1/2, 1/2)`` without a hard Log target."""
    return torch.remainder(value + 0.5, 1.0) - 0.5


@dataclass(frozen=True)
class SmoothTorusFlow:
    """Componentwise ``dy/dt = -rate sin(y)`` in a translation quotient."""

    target_relative: torch.Tensor
    rate: float = math.log(4.0)
    anchor: int = 0

    def __post_init__(self) -> None:
        if self.target_relative.ndim != 2 or self.target_relative.shape[-1] != 3:
            raise ValueError("target_relative must have shape [sites - 1, 3]")
        if self.target_relative.shape[0] < 1:
            raise ValueError("the torus quotient requires at least two sites")
        if self.anchor != 0:
            raise ValueError("vNext Q2 registers the unique-type anchor at index zero")
        if self.rate <= 0.0 or not math.isfinite(self.rate):
            raise ValueError("rate must be finite and positive")
        if not torch.isfinite(self.target_relative).all():
            raise ValueError("target relative coordinates must be finite")

    @property
    def sites(self) -> int:
        return self.target_relative.shape[0] + 1

    def angles(self, frac: torch.Tensor) -> torch.Tensor:
        self._validate_frac(frac)
        relative = frac[..., 1:, :] - frac[..., :1, :]
        target = self.target_relative.to(frac)
        return 2.0 * math.pi * wrap_centered(relative - target)

    def angle_velocity(self, angle: torch.Tensor) -> torch.Tensor:
        return -self.rate * torch.sin(angle)

    def evolved_angles(self, angle: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        delta_time = delta_time.to(angle)
        self._validate_delta(delta_time)
        decay = torch.exp(-self.rate * delta_time)
        while decay.ndim < angle.ndim:
            decay = decay.unsqueeze(-1)
        half = 0.5 * angle
        return 2.0 * torch.atan2(decay * torch.sin(half), torch.cos(half))

    def velocity(self, frac: torch.Tensor) -> torch.Tensor:
        relative_velocity = self.angle_velocity(self.angles(frac)) / (2.0 * math.pi)
        anchor_velocity = -relative_velocity.sum(dim=-2, keepdim=True) / self.sites
        return torch.cat((anchor_velocity, relative_velocity + anchor_velocity), dim=-2)

    def map(self, frac: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        self._validate_frac(frac)
        start, end = start.to(frac), end.to(frac)
        self._validate_order(start, end)
        initial_angles = self.angles(frac)
        final_angles = self.evolved_angles(initial_angles, end - start)
        relative_change = (final_angles - initial_angles) / (2.0 * math.pi)
        anchor_change = -relative_change.sum(dim=-2, keepdim=True) / self.sites
        change = torch.cat((anchor_change, relative_change + anchor_change), dim=-2)
        return torch.remainder(frac + change, 1.0)

    def analytic_relative_jacobian(self, angle: torch.Tensor, delta_time: torch.Tensor) -> torch.Tensor:
        """Diagonal derivative of the exact angle map with respect to angle."""
        delta_time = delta_time.to(angle)
        self._validate_delta(delta_time)
        decay = torch.exp(-self.rate * delta_time)
        while decay.ndim < angle.ndim:
            decay = decay.unsqueeze(-1)
        half = 0.5 * angle
        denominator = torch.cos(half).square() + decay.square() * torch.sin(half).square()
        return decay / denominator

    def _validate_frac(self, frac: torch.Tensor) -> None:
        if frac.shape[-2:] != (self.sites, 3):
            raise ValueError(f"fractional coordinates must have trailing shape [{self.sites}, 3]")
        if not torch.isfinite(frac).all():
            raise ValueError("fractional coordinates must be finite")

    @staticmethod
    def _validate_delta(delta: torch.Tensor) -> None:
        if not torch.isfinite(delta).all() or bool((delta < 0.0).any()):
            raise ValueError("time interval must be finite and nonnegative")

    @staticmethod
    def _validate_order(start: torch.Tensor, end: torch.Tensor) -> None:
        if not torch.isfinite(start).all() or not torch.isfinite(end).all():
            raise ValueError("times must be finite")
        if bool(((start < 0.0) | (end > 1.0) | (end < start)).any()):
            raise ValueError("map requires 0 <= start <= end <= 1")
