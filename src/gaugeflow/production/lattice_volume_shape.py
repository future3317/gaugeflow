"""Symmetry-constrained log-volume/log-shape lattice coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.manifold import spd_exp, spd_log, symmetric_to_vector, vector_to_symmetric


@dataclass(frozen=True)
class LatticeVolumeShape:
    """Orientation-free lattice state ``(v, A)`` with ``tr(A)=0``."""

    log_volume: torch.Tensor
    log_shape: torch.Tensor

    @classmethod
    def from_lattice(cls, lattice: torch.Tensor) -> "LatticeVolumeShape":
        if lattice.ndim < 2 or lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must end in [3, 3]")
        if not torch.isfinite(lattice).all():
            raise ValueError("lattice must be finite")
        determinant = torch.linalg.det(lattice)
        if bool((determinant <= 0).any()):
            raise ValueError("row-vector lattice must have positive determinant")
        metric = lattice @ lattice.transpose(-1, -2)
        log_volume = determinant.log()
        shape = torch.exp((-2.0 / 3.0) * log_volume)[..., None, None] * metric
        log_shape_matrix = spd_log(shape)
        # Remove only floating-point trace drift.  The mathematical state is
        # trace-free because det(shape)=1.
        identity = torch.eye(3, dtype=lattice.dtype, device=lattice.device)
        log_shape_matrix = log_shape_matrix - (
            torch.diagonal(log_shape_matrix, dim1=-2, dim2=-1).sum(-1) / 3.0
        )[..., None, None] * identity
        return cls(log_volume=log_volume, log_shape=symmetric_to_vector(log_shape_matrix))

    def metric(self) -> torch.Tensor:
        if self.log_shape.shape[-1] != 6:
            raise ValueError("log shape must use six Kelvin coordinates")
        matrix = vector_to_symmetric(self.log_shape)
        trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)
        tolerance = 128.0 * torch.finfo(matrix.dtype).eps
        if bool((trace.abs() > tolerance).any()):
            raise ValueError("log shape is not trace-free")
        return torch.exp((2.0 / 3.0) * self.log_volume)[..., None, None] * spd_exp(matrix)

    def lattice(self) -> torch.Tensor:
        return torch.linalg.cholesky(self.metric())


@dataclass(frozen=True)
class SymmetryShapeBasis:
    """Orthonormal basis of point-group-invariant trace-free log metrics."""

    matrix: torch.Tensor

    @classmethod
    def from_operations(
        cls,
        operations: torch.Tensor,
        *,
        tolerance: float = 1e-9,
    ) -> "SymmetryShapeBasis":
        if operations.ndim != 3 or operations.shape[-2:] != (3, 3):
            raise ValueError("point-group operations must have shape [operations,3,3]")
        if not operations.dtype.is_floating_point or not torch.isfinite(operations).all():
            raise ValueError("point-group operations must be finite floating tensors")
        identity = torch.eye(3, dtype=operations.dtype, device=operations.device)
        gram_error = (operations @ operations.transpose(-1, -2) - identity).abs().amax()
        if float(gram_error) > 1e-6:
            raise ValueError("point-group operations must be Cartesian O(3) matrices")
        basis_vectors = torch.eye(6, dtype=operations.dtype, device=operations.device)
        basis_matrices = vector_to_symmetric(basis_vectors)
        transformed = torch.einsum(
            "oia,kab,ojb->okij", operations, basis_matrices, operations
        )
        action = symmetric_to_vector(transformed).transpose(-1, -2)
        constraints = (action - basis_vectors).reshape(-1, 6)
        trace_constraint = operations.new_tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        constraints = torch.cat((constraints, trace_constraint), dim=0)
        _, singular_values, right = torch.linalg.svd(constraints, full_matrices=True)
        scale = singular_values.max().clamp_min(torch.finfo(operations.dtype).tiny)
        rank = int((singular_values > tolerance * scale).sum())
        null_basis = right[rank:].transpose(0, 1).contiguous()
        return cls(matrix=null_basis)

    @property
    def dimension(self) -> int:
        return self.matrix.shape[-1]

    @property
    def projector(self) -> torch.Tensor:
        return self.matrix @ self.matrix.transpose(-1, -2)

    def coordinates(self, log_shape: torch.Tensor) -> torch.Tensor:
        if log_shape.shape[-1] != 6:
            raise ValueError("log shape must use six Kelvin coordinates")
        return log_shape @ self.matrix.to(log_shape)

    def reconstruct(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.dimension:
            raise ValueError("shape coordinates do not match the symmetry basis")
        return coordinates @ self.matrix.to(coordinates).transpose(-1, -2)

    def project(self, log_shape: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.coordinates(log_shape))

    def residual(self, log_shape: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(log_shape - self.project(log_shape), dim=-1)


@dataclass(frozen=True)
class LatticeGuardrails:
    """Declared sampling-domain checks; values are protocol inputs, not clipping."""

    minimum_volume: float
    maximum_volume: float
    maximum_condition_number: float

    def validate(self, metric: torch.Tensor) -> None:
        if metric.shape[-2:] != (3, 3):
            raise ValueError("metric must end in [3,3]")
        eigenvalues = torch.linalg.eigvalsh(metric)
        volume = torch.linalg.det(metric).sqrt()
        condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(
            torch.finfo(metric.dtype).tiny
        )
        valid = (
            (eigenvalues[..., 0] > 0)
            & (volume >= self.minimum_volume)
            & (volume <= self.maximum_volume)
            & (condition <= self.maximum_condition_number)
        )
        if not bool(valid.all()):
            raise ValueError("sampled lattice lies outside the pre-registered physical domain")
