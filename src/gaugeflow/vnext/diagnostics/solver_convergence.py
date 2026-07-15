"""Small deterministic ODE solvers used only by diagnostic audits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SolverResult:
    """Final state and accounting for a deterministic integration."""

    state: torch.Tensor
    accepted_steps: int
    rejected_steps: int
    evaluations: int


def _validate(state: torch.Tensor, start: float, end: float, steps: int | None = None) -> None:
    if not torch.isfinite(state).all() or not 0.0 <= start < end <= 1.0:
        raise ValueError("solver requires finite state and 0 <= start < end <= 1")
    if steps is not None and steps < 1:
        raise ValueError("solver steps must be positive")


def euler_integrate(field: VectorField, state: torch.Tensor, *, start: float, end: float, steps: int) -> SolverResult:
    _validate(state, start, end, steps)
    value = state.clone()
    step = (end - start) / steps
    for index in range(steps):
        time = state.new_tensor(start + index * step)
        value = value + step * field(value, time)
    return SolverResult(value, steps, 0, steps)


def _rk4_step(field: VectorField, value: torch.Tensor, time: float, step: float) -> torch.Tensor:
    t0 = value.new_tensor(time)
    th = value.new_tensor(time + 0.5 * step)
    t1 = value.new_tensor(time + step)
    k1 = field(value, t0)
    k2 = field(value + 0.5 * step * k1, th)
    k3 = field(value + 0.5 * step * k2, th)
    k4 = field(value + step * k3, t1)
    return value + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(field: VectorField, state: torch.Tensor, *, start: float, end: float, steps: int) -> SolverResult:
    _validate(state, start, end, steps)
    value = state.clone()
    step = (end - start) / steps
    for index in range(steps):
        value = _rk4_step(field, value, start + index * step, step)
    return SolverResult(value, steps, 0, 4 * steps)


def adaptive_rk4(
    field: VectorField,
    state: torch.Tensor,
    *,
    start: float,
    end: float,
    rtol: float,
    atol: float,
    initial_steps: int = 16,
    max_steps: int = 100_000,
) -> SolverResult:
    """Adaptive RK4 using one-step/two-half-step error control."""
    _validate(state, start, end, initial_steps)
    if rtol <= 0.0 or atol <= 0.0 or max_steps < 1:
        raise ValueError("adaptive tolerances and max_steps must be positive")
    value = state.clone()
    time = start
    step = (end - start) / initial_steps
    accepted = rejected = evaluations = 0
    while time < end:
        if accepted + rejected >= max_steps:
            raise RuntimeError("adaptive RK4 exceeded max_steps")
        step = min(step, end - time)
        full = _rk4_step(field, value, time, step)
        half = _rk4_step(field, value, time, 0.5 * step)
        half = _rk4_step(field, half, time + 0.5 * step, 0.5 * step)
        evaluations += 12
        scale = atol + rtol * torch.maximum(value.abs(), half.abs())
        error = ((half - full).abs() / scale).max()
        error_value = float(error)
        if error_value <= 1.0:
            value = half
            time += step
            accepted += 1
        else:
            rejected += 1
        factor = 2.0 if error_value == 0.0 else min(2.0, max(0.2, 0.9 * error_value ** (-0.2)))
        step *= factor
    return SolverResult(value, accepted, rejected, evaluations)
