"""Cartesian auxiliary heads for post-A1 physical representation training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .state_projection import graph_mean, sorted_segment_sum


def symmetric_cartesian_to_kelvin(tensor: torch.Tensor) -> torch.Tensor:
    """Encode symmetric 3x3 tensors as orthonormal Kelvin six-vectors."""

    if tensor.ndim != 3 or tensor.shape[-2:] != (3, 3):
        raise ValueError("symmetric Cartesian tensor must have shape [graphs,3,3]")
    root_two = tensor.new_tensor(2.0).sqrt()
    return torch.stack(
        (
            tensor[:, 0, 0],
            tensor[:, 1, 1],
            tensor[:, 2, 2],
            root_two * tensor[:, 1, 2],
            root_two * tensor[:, 0, 2],
            root_two * tensor[:, 0, 1],
        ),
        dim=-1,
    )


def kelvin_to_symmetric_cartesian(kelvin: torch.Tensor) -> torch.Tensor:
    """Decode Kelvin six-vectors to symmetric 3x3 tensors."""

    if kelvin.ndim != 2 or kelvin.shape[-1] != 6:
        raise ValueError("Kelvin tensor must have shape [graphs,6]")
    inverse_root_two = kelvin.new_tensor(0.5).sqrt()
    xx, yy, zz, yz, xz, xy = kelvin.unbind(dim=-1)
    rows = (
        torch.stack((xx, inverse_root_two * xy, inverse_root_two * xz), dim=-1),
        torch.stack((inverse_root_two * xy, yy, inverse_root_two * yz), dim=-1),
        torch.stack((inverse_root_two * xz, inverse_root_two * yz, zz), dim=-1),
    )
    return torch.stack(rows, dim=-2)


@dataclass(frozen=True)
class PhysicalPredictions:
    energy_per_atom: torch.Tensor
    forces: torch.Tensor
    stress_kelvin: torch.Tensor
    teacher_features: torch.Tensor


@dataclass(frozen=True)
class PhysicalTargets:
    """Pre-normalized targets and explicit availability masks."""

    energy_per_atom: torch.Tensor
    forces: torch.Tensor
    stress_kelvin: torch.Tensor
    teacher_features: torch.Tensor
    energy_mask: torch.Tensor
    force_mask: torch.Tensor
    stress_mask: torch.Tensor
    teacher_mask: torch.Tensor


@dataclass(frozen=True)
class PhysicalLossOutput:
    loss: torch.Tensor
    energy_loss: torch.Tensor
    force_loss: torch.Tensor
    stress_loss: torch.Tensor
    feature_loss: torch.Tensor


class CartesianPhysicalHeads(nn.Module):
    """Linear-complexity invariant/vector/symmetric-tensor readouts."""

    def __init__(self, scalar_dim: int, vector_dim: int, teacher_dim: int) -> None:
        super().__init__()
        if scalar_dim < 1 or vector_dim < 1 or teacher_dim < 1:
            raise ValueError("physical head dimensions must be positive")
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.teacher_dim = teacher_dim
        self.energy_head = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, 1),
        )
        self.force_coefficients = nn.Linear(scalar_dim, vector_dim, bias=False)
        self.stress_coefficients = nn.Linear(scalar_dim, vector_dim, bias=False)
        self.pressure_head = nn.Linear(scalar_dim, 1)
        self.teacher_projection = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, teacher_dim),
        )

    def forward(
        self,
        node_scalar: torch.Tensor,
        node_vectors: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
    ) -> PhysicalPredictions:
        if node_scalar.ndim != 2 or node_scalar.shape[1] != self.scalar_dim:
            raise ValueError("node scalar representation has the wrong shape")
        if node_vectors.shape != (node_scalar.shape[0], self.vector_dim, 3):
            raise ValueError("node Cartesian vectors have the wrong shape")
        if batch.shape != node_scalar.shape[:1] or batch.dtype != torch.long:
            raise ValueError("physical head batch must index every node")
        if graph_count < 1:
            raise ValueError("physical heads require a nonempty graph batch")
        graph_scalar = graph_mean(node_scalar, batch, graph_count)
        energy = self.energy_head(graph_scalar).squeeze(-1)
        force_coefficients = self.force_coefficients(node_scalar)
        forces = torch.einsum("nv,nvc->nc", force_coefficients, node_vectors)

        stress_coefficients = self.stress_coefficients(node_scalar)
        dyadic = torch.einsum(
            "nv,nvi,nvj->nij",
            stress_coefficients,
            node_vectors,
            node_vectors,
        )
        identity = torch.eye(3, dtype=dyadic.dtype, device=dyadic.device)
        trace = torch.einsum("nii->n", dyadic) / 3.0
        deviatoric = dyadic - trace[:, None, None] * identity
        stress = graph_mean(deviatoric, batch, graph_count)
        pressure = self.pressure_head(graph_scalar).reshape(graph_count, 1, 1)
        stress = stress + pressure * identity
        return PhysicalPredictions(
            energy_per_atom=energy,
            forces=forces,
            stress_kelvin=symmetric_cartesian_to_kelvin(stress),
            teacher_features=self.teacher_projection(graph_scalar),
        )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != value.shape[:1] or mask.dtype != torch.bool:
        raise ValueError("physical label mask has the wrong shape or dtype")
    if bool(mask.any()):
        return value[mask].mean()
    return value.sum() * 0.0


def physical_multitask_loss(
    prediction: PhysicalPredictions,
    target: PhysicalTargets,
    batch: torch.Tensor,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    feature_weight: float = 1.0,
) -> PhysicalLossOutput:
    """Masked graph-equal loss for calibrated E/F/stress/features."""

    graph_count = prediction.energy_per_atom.numel()
    if target.energy_per_atom.shape != (graph_count,):
        raise ValueError("energy targets do not match graph count")
    if prediction.forces.shape != target.forces.shape or prediction.forces.shape != (batch.numel(), 3):
        raise ValueError("force targets do not match node count")
    if prediction.stress_kelvin.shape != target.stress_kelvin.shape or prediction.stress_kelvin.shape != (
        graph_count,
        6,
    ):
        raise ValueError("stress targets do not match graph count")
    if prediction.teacher_features.shape != target.teacher_features.shape:
        raise ValueError("teacher feature targets do not match projection")
    weights = (energy_weight, force_weight, stress_weight, feature_weight)
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("physical loss weights must be nonnegative and nonzero")

    energy_loss = _masked_mean(
        (prediction.energy_per_atom - target.energy_per_atom).square(),
        target.energy_mask,
    )
    node_force_error = (prediction.forces - target.forces).square().mean(dim=-1)
    if target.force_mask.shape != (batch.numel(),) or target.force_mask.dtype != torch.bool:
        raise ValueError("force mask must provide one boolean per node")
    force_sum = sorted_segment_sum(
        node_force_error * target.force_mask.to(node_force_error),
        batch,
        graph_count,
    )
    force_count = torch.bincount(
        batch[target.force_mask],
        minlength=graph_count,
    )
    force_graph_mask = force_count > 0
    force_graph_mean = force_sum / force_count.clamp_min(1).to(force_sum)
    force_loss = _masked_mean(force_graph_mean, force_graph_mask)
    stress_loss = _masked_mean(
        (prediction.stress_kelvin - target.stress_kelvin).square().mean(dim=-1),
        target.stress_mask,
    )
    feature_loss = _masked_mean(
        (prediction.teacher_features - target.teacher_features).square().mean(dim=-1),
        target.teacher_mask,
    )
    loss = (
        energy_weight * energy_loss
        + force_weight * force_loss
        + stress_weight * stress_loss
        + feature_weight * feature_loss
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("physical pretraining loss is non-finite")
    return PhysicalLossOutput(
        loss=loss,
        energy_loss=energy_loss,
        force_loss=force_loss,
        stress_loss=stress_loss,
        feature_loss=feature_loss,
    )
