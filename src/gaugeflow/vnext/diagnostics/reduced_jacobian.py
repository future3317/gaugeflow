"""Translation-reduced vector-field and flow-map Jacobian diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

ReducedField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
JacobianField = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AnalyticEndpointJacobian:
    """Jacobian of the singular deterministic collapsed-endpoint path."""

    time: torch.Tensor
    vector_jacobian: torch.Tensor | None
    flow_jacobian: torch.Tensor
    singular_values: torch.Tensor
    log_abs_det: torch.Tensor


def reduced_vector_jacobian(field: ReducedField, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Compute ``d field(state,time) / d state`` with ``torch.func.jacrev``."""
    if state.ndim != 1:
        raise ValueError("reduced state must be one-dimensional")
    output = field(state, time)
    if output.shape != state.shape:
        raise ValueError("reduced field must preserve the state shape")
    return torch.func.jacrev(lambda value: field(value, time))(state)


def variational_flow_jacobian(
    jacobian: JacobianField,
    *,
    dimension: int,
    end_time: float,
    steps: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Integrate ``J' = A(t) J`` by RK4 from zero to ``end_time``."""
    if dimension < 1 or steps < 1 or not 0.0 <= end_time <= 1.0:
        raise ValueError("invalid variational integration settings")
    value = torch.eye(dimension, dtype=dtype, device=device)
    step = end_time / steps
    for index in range(steps):
        time = torch.tensor(index * step, dtype=dtype, device=device)
        half = torch.tensor((index + 0.5) * step, dtype=dtype, device=device)
        finish = torch.tensor((index + 1.0) * step, dtype=dtype, device=device)
        k1 = jacobian(time) @ value
        k2 = jacobian(half) @ (value + 0.5 * step * k1)
        k3 = jacobian(half) @ (value + 0.5 * step * k2)
        k4 = jacobian(finish) @ (value + step * k3)
        value = value + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return value


def analytic_endpoint_jacobians(dimension: int, time: torch.Tensor) -> AnalyticEndpointJacobian:
    """Return exact Jacobians for ``x_t=(1-t)x_0+t x_1``.

    At ``t=1`` the flow Jacobian is singular and the Eulerian vector-field
    Jacobian does not exist as a finite matrix.  That endpoint is represented
    explicitly by ``vector_jacobian=None`` and ``log_abs_det=-inf``.
    """
    if dimension < 1 or time.numel() != 1 or not torch.isfinite(time):
        raise ValueError("dimension must be positive and time a finite scalar tensor")
    scalar = float(time)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError("time must lie in [0, 1]")
    identity = torch.eye(dimension, dtype=time.dtype, device=time.device)
    scale = 1.0 - time
    flow = scale * identity
    singular_values = torch.full((dimension,), float(scale), dtype=time.dtype, device=time.device)
    log_abs_det = time.new_tensor(float("-inf")) if scalar == 1.0 else dimension * torch.log(scale)
    vector = None if scalar == 1.0 else -identity / scale
    return AnalyticEndpointJacobian(time, vector, flow, singular_values, log_abs_det)
