"""Regular affine contraction in a translation-horizontal Euclidean chart."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def translation_horizontal_basis(
    sites: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an orthonormal basis for coordinates modulo global translation.

    The returned matrix has shape ``[3 * sites, 3 * (sites - 1)]``. Its columns
    are orthonormal and orthogonal to the three graphwise translation modes.
    """
    if sites < 2:
        raise ValueError("translation quotient requires at least two sites")
    contrast = torch.zeros((sites, sites - 1), dtype=dtype, device=device)
    for column in range(sites - 1):
        count = column + 1
        normalization = (count * (count + 1)) ** 0.5
        contrast[:count, column] = 1.0 / normalization
        contrast[count, column] = -count / normalization
    return torch.kron(contrast, torch.eye(3, dtype=dtype, device=device))


@dataclass(frozen=True)
class RegularAffineFlow:
    """Finite-time diffeomorphic affine flow with nondegenerate endpoint."""

    mean: torch.Tensor
    terminal_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("mean must be a one-dimensional reduced-coordinate vector")
        if not 0.0 < self.terminal_scale <= 1.0:
            raise ValueError("terminal_scale must lie in (0, 1]")
        if not torch.isfinite(self.mean).all():
            raise ValueError("mean must be finite")

    def scale(self, time: torch.Tensor) -> torch.Tensor:
        self._validate_time(time)
        return 1.0 - (1.0 - self.terminal_scale) * time

    def state(self, source: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self._validate_source(source)
        scale = self._broadcast_time(self.scale(time.to(source)), source)
        return self.mean.to(source) + scale * (source - self.mean.to(source))

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        self._validate_source(state)
        scale = self._broadcast_time(self.scale(time.to(state)), state)
        return (self.terminal_scale - 1.0) * (state - self.mean.to(state)) / scale

    def map(self, state: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Exact flow map from ``start`` to ``end`` for a state on the path."""
        self._validate_source(state)
        start_scale = self._broadcast_time(self.scale(start.to(state)), state)
        end_scale = self._broadcast_time(self.scale(end.to(state)), state)
        return self.mean.to(state) + (end_scale / start_scale) * (state - self.mean.to(state))

    def vector_jacobian(self, time: torch.Tensor) -> torch.Tensor:
        scale = self.scale(time.to(self.mean))
        coefficient = (self.terminal_scale - 1.0) / scale
        identity = torch.eye(self.mean.numel(), dtype=self.mean.dtype, device=self.mean.device)
        return coefficient[..., None, None] * identity

    def flow_jacobian(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        ratio = self.scale(end.to(self.mean)) / self.scale(start.to(self.mean))
        identity = torch.eye(self.mean.numel(), dtype=self.mean.dtype, device=self.mean.device)
        return ratio[..., None, None] * identity

    def log_abs_det(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        ratio = self.scale(end.to(self.mean)) / self.scale(start.to(self.mean))
        return self.mean.numel() * torch.log(ratio)

    @staticmethod
    def _validate_time(time: torch.Tensor) -> None:
        if not torch.isfinite(time).all() or bool(((time < 0.0) | (time > 1.0)).any()):
            raise ValueError("time must be finite and lie in [0, 1]")

    def _validate_source(self, value: torch.Tensor) -> None:
        if value.shape[-1] != self.mean.numel():
            raise ValueError("state dimension does not match the registered mean")
        if not torch.isfinite(value).all():
            raise ValueError("state must be finite")

    @staticmethod
    def _broadcast_time(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        while time.ndim < state.ndim:
            time = time.unsqueeze(-1)
        return time
