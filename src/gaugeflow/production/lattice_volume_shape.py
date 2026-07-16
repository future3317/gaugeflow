"""Symmetry-constrained log-volume/log-shape lattice coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.manifold import spd_exp, spd_log, symmetric_to_vector, vector_to_symmetric


@dataclass(frozen=True)
class LatticeVolumeShape:
    """Lattice state ``(v, A)`` in an orthonormal point-group metric chart.

    ``A`` is the trace-free logarithm of the Cartesian chart metric ``H``.
    The crystallographic fractional Gram matrix is reconstructed by the
    congruence ``C.T @ exp(A) @ C`` followed by determinant normalization.
    Keeping ``A`` and the fractional Gram matrix in distinct spaces prevents
    the hexagonal/trigonal chart error caused by applying Cartesian point-group
    matrices directly to fractional lattice indices.
    """

    log_volume: torch.Tensor
    log_shape: torch.Tensor

    @classmethod
    def from_lattice(
        cls,
        lattice: torch.Tensor,
        fractional_to_cartesian: torch.Tensor,
    ) -> "LatticeVolumeShape":
        if lattice.ndim < 2 or lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must end in [3, 3]")
        if not torch.isfinite(lattice).all():
            raise ValueError("lattice must be finite")
        determinant = torch.linalg.det(lattice)
        if bool((determinant <= 0).any()):
            raise ValueError("row-vector lattice must have positive determinant")
        if fractional_to_cartesian.shape[-2:] != (3, 3):
            raise ValueError("fractional-to-Cartesian chart must end in [3,3]")
        metric = lattice @ lattice.transpose(-1, -2)
        log_volume = determinant.log()
        fractional_shape = torch.exp((-2.0 / 3.0) * log_volume)[..., None, None] * metric
        chart = fractional_to_cartesian.to(metric)
        inverse_chart = torch.linalg.inv(chart)
        raw_cartesian_shape = (
            inverse_chart.transpose(-1, -2) @ fractional_shape @ inverse_chart
        )
        cartesian_shape = raw_cartesian_shape / torch.linalg.det(raw_cartesian_shape).pow(
            1.0 / 3.0
        )[..., None, None]
        log_shape_matrix = spd_log(cartesian_shape)
        # Remove only floating-point trace drift.  The mathematical state is
        # trace-free because det(shape)=1.
        identity = torch.eye(3, dtype=lattice.dtype, device=lattice.device)
        log_shape_matrix = log_shape_matrix - (
            torch.diagonal(log_shape_matrix, dim1=-2, dim2=-1).sum(-1) / 3.0
        )[..., None, None] * identity
        return cls(log_volume=log_volume, log_shape=symmetric_to_vector(log_shape_matrix))

    def metric(self, fractional_to_cartesian: torch.Tensor) -> torch.Tensor:
        if self.log_shape.shape[-1] != 6:
            raise ValueError("log shape must use six Kelvin coordinates")
        matrix = vector_to_symmetric(self.log_shape)
        trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)
        tolerance = 128.0 * torch.finfo(matrix.dtype).eps
        if bool((trace.abs() > tolerance).any()):
            raise ValueError("log shape is not trace-free")
        if fractional_to_cartesian.shape[-2:] != (3, 3):
            raise ValueError("fractional-to-Cartesian chart must end in [3,3]")
        chart = fractional_to_cartesian.to(matrix)
        cartesian_shape = spd_exp(matrix)
        raw_fractional_shape = chart.transpose(-1, -2) @ cartesian_shape @ chart
        fractional_shape = raw_fractional_shape / torch.linalg.det(raw_fractional_shape).pow(
            1.0 / 3.0
        )[..., None, None]
        return torch.exp((2.0 / 3.0) * self.log_volume)[..., None, None] * fractional_shape

    def lattice(self, fractional_to_cartesian: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(self.metric(fractional_to_cartesian))


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
class PointGroupMetricChart:
    """Fractional/Cartesian point-group chart for symmetry-compatible metrics."""

    fractional_operations: torch.Tensor
    cartesian_operations: torch.Tensor
    fractional_to_cartesian: torch.Tensor
    invariant_log_shape_basis: torch.Tensor

    @classmethod
    def from_fractional_operations(
        cls,
        fractional_operations: torch.Tensor,
        *,
        tolerance: float = 1e-9,
    ) -> "PointGroupMetricChart":
        if fractional_operations.ndim != 3 or fractional_operations.shape[-2:] != (3, 3):
            raise ValueError("fractional operations must have shape [operations,3,3]")
        if not fractional_operations.dtype.is_floating_point:
            raise ValueError("fractional operations must use a floating dtype")
        if not torch.isfinite(fractional_operations).all():
            raise ValueError("fractional operations must be finite")
        # The finite-group Reynolds average defines a positive invariant metric
        # M=C^T C. Consequently C U C^-1 is Cartesian orthogonal without any
        # polar-factor post-processing that could spoil exact group closure.
        invariant_metric = torch.einsum(
            "oji,ojk->ik", fractional_operations, fractional_operations
        ) / fractional_operations.shape[0]
        invariant_metric = 0.5 * (invariant_metric + invariant_metric.transpose(-1, -2))
        chart = torch.linalg.cholesky(invariant_metric).transpose(-1, -2)
        inverse_chart = torch.linalg.inv(chart)
        cartesian = chart.unsqueeze(0) @ fractional_operations @ inverse_chart.unsqueeze(0)
        orthogonality_error = torch.linalg.matrix_norm(
            cartesian.transpose(-1, -2) @ cartesian
            - torch.eye(3, dtype=cartesian.dtype, device=cartesian.device),
            dim=(-2, -1),
        ).amax()
        if float(orthogonality_error) > 1e-8:
            raise RuntimeError("failed to construct an orthogonal Cartesian point-group chart")
        basis = SymmetryShapeBasis.from_operations(cartesian, tolerance=tolerance)
        return cls(
            fractional_operations=fractional_operations,
            cartesian_operations=cartesian,
            fractional_to_cartesian=chart,
            invariant_log_shape_basis=basis.matrix,
        )

    @property
    def shape_dimension(self) -> int:
        return self.invariant_log_shape_basis.shape[-1]

    @property
    def shape_projector(self) -> torch.Tensor:
        basis = self.invariant_log_shape_basis
        return basis @ basis.transpose(-1, -2)

    def project_log_shape(self, log_shape: torch.Tensor) -> torch.Tensor:
        if log_shape.shape[-1] != 6:
            raise ValueError("log shape must use six Kelvin coordinates")
        projector = self.shape_projector.to(log_shape)
        return log_shape @ projector.transpose(-1, -2)

    def metric(self, log_volume: torch.Tensor, log_shape: torch.Tensor) -> torch.Tensor:
        projected = self.project_log_shape(log_shape)
        return LatticeVolumeShape(log_volume, projected).metric(
            self.fractional_to_cartesian.to(projected)
        )

    def invariance_residual(self, metric: torch.Tensor) -> torch.Tensor:
        operations = self.fractional_operations.to(metric)
        transformed = operations.transpose(-1, -2) @ metric.unsqueeze(-3) @ operations
        numerator = torch.linalg.matrix_norm(transformed - metric.unsqueeze(-3), dim=(-2, -1))
        denominator = torch.linalg.matrix_norm(metric, dim=(-2, -1)).unsqueeze(-1)
        return numerator / denominator.clamp_min(torch.finfo(metric.dtype).tiny)


def project_lattice_state(log_shape: torch.Tensor, shape_projector: torch.Tensor) -> torch.Tensor:
    """Project a batched reverse-process state into its allowed shape chart."""
    if log_shape.ndim != 2 or log_shape.shape[-1] != 6:
        raise ValueError("log shape must have shape [graphs,6]")
    if shape_projector.shape != (log_shape.shape[0], 6, 6):
        raise ValueError("shape projector must have shape [graphs,6,6]")
    return torch.einsum("bij,bj->bi", shape_projector.to(log_shape), log_shape)


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
