"""Cartesian multi-task response model for Stage-D physical pretraining.

The module predicts physical tensors from the shared GaugeFlow structure
encoder without entering the generative sampler.  Every tensor is assembled
from scalar coefficients, polar Cartesian vectors and Kronecker deltas, so its
O(3) transformation law is architectural rather than an augmentation target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from .equivariant_denoiser import HybridCrystalDenoiser
from .state_projection import graph_mean, sorted_segment_sum


@dataclass(frozen=True)
class ResponsePredictions:
    piezoelectric: torch.Tensor
    dielectric: torch.Tensor
    elastic: torch.Tensor
    born_effective_charge: torch.Tensor
    gamma_soft_logits: torch.Tensor
    gamma_log_magnitude: torch.Tensor
    internal_strain: torch.Tensor


@dataclass(frozen=True)
class ResponseTargets:
    """Canonical targets with masks that distinguish missing labels from zero."""

    piezoelectric: torch.Tensor
    dielectric: torch.Tensor
    elastic: torch.Tensor
    born_effective_charge: torch.Tensor
    gamma_soft: torch.Tensor
    gamma_log_magnitude: torch.Tensor
    internal_strain: torch.Tensor
    piezoelectric_mask: torch.Tensor
    dielectric_mask: torch.Tensor
    elastic_mask: torch.Tensor
    born_mask: torch.Tensor
    gamma_mask: torch.Tensor
    internal_strain_mask: torch.Tensor


@dataclass(frozen=True)
class ResponseTaskWeights:
    piezoelectric: float = 1.0
    dielectric: float = 1.0
    elastic: float = 1.0
    born_effective_charge: float = 1.0
    gamma_frequency: float = 1.0
    internal_strain: float = 1.0
    piezoelectric_probe: float = 0.0


@dataclass(frozen=True)
class ResponseLossOutput:
    loss: torch.Tensor
    piezoelectric_loss: torch.Tensor
    dielectric_loss: torch.Tensor
    elastic_loss: torch.Tensor
    born_loss: torch.Tensor
    gamma_loss: torch.Tensor
    internal_strain_loss: torch.Tensor
    piezoelectric_probe_loss: torch.Tensor
    active_tasks: int


class _AdaptiveVectorMix(nn.Module):
    """Form state-dependent vector channels with no preferred spatial frame."""

    def __init__(self, scalar_dim: int, vector_dim: int, channels: int) -> None:
        super().__init__()
        if min(scalar_dim, vector_dim, channels) < 1:
            raise ValueError("adaptive vector-mix dimensions must be positive")
        self.vector_dim = vector_dim
        self.channels = channels
        self.coefficients = nn.Linear(scalar_dim, channels * vector_dim, bias=False)

    def forward(self, scalar: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if scalar.ndim != 2 or vectors.shape != (scalar.shape[0], self.vector_dim, 3):
            raise ValueError("adaptive vector mix received incompatible node features")
        coefficients = self.coefficients(scalar).reshape(-1, self.channels, self.vector_dim)
        scale = scalar.new_tensor(float(self.vector_dim)).rsqrt()
        return torch.einsum("nrv,nvc->nrc", coefficients, vectors) * scale


def _symmetric_dyad(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        torch.einsum("...i,...j->...ij", left, right)
        + torch.einsum("...i,...j->...ij", right, left)
    )


def _rank_three_from_vectors(
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...i,...jk->...ijk", first, _symmetric_dyad(second, third))


class CartesianResponseHeads(nn.Module):
    """Low-rank, vectorized Cartesian heads for the Stage-D response tasks."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        *,
        covariant_rank: int = 8,
        maximum_atoms: int = 20,
    ) -> None:
        super().__init__()
        if min(scalar_dim, vector_dim, covariant_rank, maximum_atoms) < 1:
            raise ValueError("response-head dimensions must be positive")
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.rank = covariant_rank
        self.maximum_modes = 3 * maximum_atoms
        self.scalar_norm = nn.LayerNorm(scalar_dim)

        self.dielectric_vectors = _AdaptiveVectorMix(scalar_dim, vector_dim, 2 * covariant_rank)
        self.piezo_vectors = _AdaptiveVectorMix(scalar_dim, vector_dim, 3 * covariant_rank + 2)
        self.elastic_vectors = _AdaptiveVectorMix(scalar_dim, vector_dim, 4 * covariant_rank)
        self.born_vectors = _AdaptiveVectorMix(scalar_dim, vector_dim, 2 * covariant_rank)
        self.internal_vectors = _AdaptiveVectorMix(scalar_dim, vector_dim, 3 * covariant_rank + 2)

        self.dielectric_isotropic = nn.Linear(scalar_dim, 1)
        self.elastic_isotropic = nn.Linear(scalar_dim, 2)
        self.born_isotropic = nn.Linear(scalar_dim, 1)
        self.piezo_delta_coefficients = nn.Linear(scalar_dim, 2)
        self.internal_delta_coefficients = nn.Linear(scalar_dim, 2)
        self.gamma_head = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, 2 * self.maximum_modes),
        )

    @staticmethod
    def _rank_three_delta_terms(
        vectors: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        if vectors.shape[-2:] != (2, 3) or coefficients.shape != vectors.shape[:-2] + (2,):
            raise ValueError("rank-three delta terms received incompatible features")
        identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
        first = coefficients[..., 0, None, None, None] * torch.einsum(
            "...i,jk->...ijk", vectors[..., 0, :], identity
        )
        second = coefficients[..., 1, None, None, None] * 0.5 * (
            torch.einsum("...j,ik->...ijk", vectors[..., 1, :], identity)
            + torch.einsum("...k,ij->...ijk", vectors[..., 1, :], identity)
        )
        return first + second

    def forward(
        self,
        node_scalar: torch.Tensor,
        node_vectors: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
    ) -> ResponsePredictions:
        if node_scalar.ndim != 2 or node_scalar.shape[1] != self.scalar_dim:
            raise ValueError("response head scalar features have the wrong shape")
        if node_vectors.shape != (node_scalar.shape[0], self.vector_dim, 3):
            raise ValueError("response head vector features have the wrong shape")
        if batch.shape != node_scalar.shape[:1] or batch.dtype != torch.long or graph_count < 1:
            raise ValueError("response head batch is invalid")

        scalar = self.scalar_norm(node_scalar)
        graph_scalar = graph_mean(scalar, batch, graph_count)
        identity = torch.eye(3, dtype=scalar.dtype, device=scalar.device)
        rank_scale = scalar.new_tensor(float(self.rank)).rsqrt()

        dielectric_vectors = self.dielectric_vectors(scalar, node_vectors).reshape(
            -1, self.rank, 2, 3
        )
        dielectric_nodes = _symmetric_dyad(
            dielectric_vectors[:, :, 0], dielectric_vectors[:, :, 1]
        ).sum(dim=1) * rank_scale
        dielectric = graph_mean(dielectric_nodes, batch, graph_count)
        dielectric = dielectric + self.dielectric_isotropic(graph_scalar)[:, :, None] * identity

        piezo_vectors = self.piezo_vectors(scalar, node_vectors)
        piezo_low_rank = piezo_vectors[:, : 3 * self.rank].reshape(-1, self.rank, 3, 3)
        piezo_nodes = _rank_three_from_vectors(
            piezo_low_rank[:, :, 0], piezo_low_rank[:, :, 1], piezo_low_rank[:, :, 2]
        ).sum(dim=1) * rank_scale
        piezo_nodes = piezo_nodes + self._rank_three_delta_terms(
            piezo_vectors[:, 3 * self.rank :], self.piezo_delta_coefficients(scalar)
        )
        piezoelectric = graph_mean(piezo_nodes, batch, graph_count)

        elastic_vectors = self.elastic_vectors(scalar, node_vectors).reshape(
            -1, self.rank, 4, 3
        )
        strain_left = _symmetric_dyad(elastic_vectors[:, :, 0], elastic_vectors[:, :, 1])
        strain_right = _symmetric_dyad(elastic_vectors[:, :, 2], elastic_vectors[:, :, 3])
        elastic_nodes = 0.5 * (
            torch.einsum("nrij,nrkl->nrijkl", strain_left, strain_right)
            + torch.einsum("nrij,nrkl->nrijkl", strain_right, strain_left)
        ).sum(dim=1) * rank_scale
        elastic = graph_mean(elastic_nodes, batch, graph_count)
        lame = self.elastic_isotropic(graph_scalar)
        delta_ij_delta_kl = torch.einsum("ij,kl->ijkl", identity, identity)
        shear_identity = 0.5 * (
            torch.einsum("ik,jl->ijkl", identity, identity)
            + torch.einsum("il,jk->ijkl", identity, identity)
        )
        elastic = elastic + lame[:, 0, None, None, None, None] * delta_ij_delta_kl
        elastic = elastic + lame[:, 1, None, None, None, None] * shear_identity

        born_vectors = self.born_vectors(scalar, node_vectors).reshape(-1, self.rank, 2, 3)
        born = torch.einsum(
            "nri,nrj->nij", born_vectors[:, :, 0], born_vectors[:, :, 1]
        ) * rank_scale
        born = born + self.born_isotropic(scalar)[:, :, None] * identity

        internal_vectors = self.internal_vectors(scalar, node_vectors)
        internal_low_rank = internal_vectors[:, : 3 * self.rank].reshape(-1, self.rank, 3, 3)
        internal = _rank_three_from_vectors(
            internal_low_rank[:, :, 0],
            internal_low_rank[:, :, 1],
            internal_low_rank[:, :, 2],
        ).sum(dim=1) * rank_scale
        internal = internal + self._rank_three_delta_terms(
            internal_vectors[:, 3 * self.rank :], self.internal_delta_coefficients(scalar)
        )

        gamma = self.gamma_head(graph_scalar).reshape(graph_count, 2, self.maximum_modes)
        return ResponsePredictions(
            piezoelectric=piezoelectric,
            dielectric=dielectric,
            elastic=elastic,
            born_effective_charge=born,
            gamma_soft_logits=gamma[:, 0],
            gamma_log_magnitude=gamma[:, 1],
            internal_strain=internal,
        )


class ResponseMultiTaskModel(nn.Module):
    """Shared GaugeFlow encoder plus source-calibrated Cartesian response heads."""

    def __init__(
        self,
        backbone: HybridCrystalDenoiser,
        *,
        source_count: int,
        covariant_rank: int = 8,
        maximum_atoms: int = 20,
    ) -> None:
        super().__init__()
        if source_count < 1:
            raise ValueError("response model needs at least one registered source")
        self.backbone = backbone
        scalar_dim = backbone.element_embedding.embedding_dim
        self.source_embedding = nn.Embedding(source_count, scalar_dim)
        self.heads = CartesianResponseHeads(
            scalar_dim,
            backbone.coordinate_carrier.vector_channels,
            covariant_rank=covariant_rank,
            maximum_atoms=maximum_atoms,
        )

    def forward(
        self,
        element_tokens: torch.Tensor,
        fractional_coordinates: torch.Tensor,
        lattice: torch.Tensor,
        batch: torch.Tensor,
        source_index: torch.Tensor,
    ) -> ResponsePredictions:
        graph_count = lattice.shape[0]
        if source_index.shape != (graph_count,) or source_index.dtype != torch.long:
            raise ValueError("response source index must contain one integer per graph")
        if int(source_index.min()) < 0 or int(source_index.max()) >= self.source_embedding.num_embeddings:
            raise ValueError("response source index lies outside the registered vocabulary")
        features = self.backbone.forward_physical_features(
            element_tokens,
            fractional_coordinates,
            lattice,
            batch,
        )
        scalar = features.node_scalar + self.source_embedding(source_index)[batch]
        return self.heads(scalar, features.node_vectors, batch, graph_count)


def _masked_graph_loss(error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if error.ndim != 1 or mask.shape != error.shape or mask.dtype != torch.bool:
        raise ValueError("graph response mask is invalid")
    if bool(mask.any()):
        return error[mask].mean(), True
    return error.sum() * 0.0, False


def _radial_huber(error: torch.Tensor) -> torch.Tensor:
    if error.ndim < 2:
        raise ValueError("radial tensor loss requires a leading example dimension")
    epsilon = error.new_tensor(1e-12)
    mean_square = error.flatten(1).square().mean(dim=-1)
    radius = (mean_square + epsilon).sqrt() - epsilon.sqrt()
    return functional.huber_loss(radius, torch.zeros_like(radius), reduction="none")


def _masked_node_graph_loss(
    component_error: torch.Tensor,
    component_mask: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> tuple[torch.Tensor, bool]:
    if component_error.shape != component_mask.shape or component_mask.dtype != torch.bool:
        raise ValueError("node response component mask is invalid")
    if component_error.shape[:1] != batch.shape:
        raise ValueError("node response batch is invalid")
    flattened_error = component_error.reshape(component_error.shape[0], -1)
    flattened_mask = component_mask.reshape(component_mask.shape[0], -1)
    component_count = flattened_mask.sum(dim=-1)
    node_valid = component_count > 0
    node_mean_square = (
        (flattened_error.square() * flattened_mask).sum(dim=-1)
        / component_count.clamp_min(1)
    )
    epsilon = flattened_error.new_tensor(1e-12)
    node_radius = (node_mean_square + epsilon).sqrt() - epsilon.sqrt()
    node_error = functional.huber_loss(
        node_radius, torch.zeros_like(node_radius), reduction="none"
    )
    graph_error_sum = sorted_segment_sum(node_error * node_valid, batch, graph_count)
    graph_node_count = sorted_segment_sum(node_valid.to(node_error), batch, graph_count)
    graph_valid = graph_node_count > 0
    graph_error = graph_error_sum / graph_node_count.clamp_min(1)
    return _masked_graph_loss(graph_error, graph_valid)


def icosahedral_response_directions(
    *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Six unoriented icosahedral axes forming a degree-five spherical design."""

    golden_ratio = (torch.tensor(5.0, dtype=dtype, device=device).sqrt() + 1.0) / 2.0
    zero = golden_ratio.new_zeros(())
    one = golden_ratio.new_ones(())
    directions = torch.stack(
        (
            torch.stack((zero, one, golden_ratio)),
            torch.stack((zero, one, -golden_ratio)),
            torch.stack((one, golden_ratio, zero)),
            torch.stack((one, -golden_ratio, zero)),
            torch.stack((golden_ratio, zero, one)),
            torch.stack((golden_ratio, zero, -one)),
        )
    )
    return functional.normalize(directions, dim=-1)


def piezoelectric_response_probe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Rotation-invariant response-field error on an exact quartic design."""

    if prediction.shape != target.shape or prediction.shape[-3:] != (3, 3, 3):
        raise ValueError("piezoelectric response probes require matching rank-three tensors")
    directions = icosahedral_response_directions(
        dtype=prediction.dtype, device=prediction.device
    )
    error = prediction - target
    response = torch.einsum("gijk,qj,qk->gqi", error, directions, directions)
    return _masked_graph_loss(_radial_huber(response), mask)


def response_multitask_loss(
    prediction: ResponsePredictions,
    target: ResponseTargets,
    batch: torch.Tensor,
    graph_count: int,
    *,
    weights: ResponseTaskWeights = ResponseTaskWeights(),
) -> ResponseLossOutput:
    """Task-balanced masked loss; a valid physical zero remains supervised."""

    # Componentwise robust losses are not invariant under a Cartesian basis
    # rotation. Radial Huber on the Frobenius norm is both O(3)-invariant and
    # robust to the physically meaningful heavy tails of soft response modes.
    graph_specs = (
        (
            _radial_huber(prediction.piezoelectric - target.piezoelectric),
            target.piezoelectric_mask,
        ),
        (
            _radial_huber(prediction.dielectric - target.dielectric),
            target.dielectric_mask,
        ),
        (
            _radial_huber(prediction.elastic - target.elastic),
            target.elastic_mask,
        ),
    )
    piezo_loss, piezo_active = _masked_graph_loss(*graph_specs[0])
    dielectric_loss, dielectric_active = _masked_graph_loss(*graph_specs[1])
    elastic_loss, elastic_active = _masked_graph_loss(*graph_specs[2])

    born_error = prediction.born_effective_charge - target.born_effective_charge
    born_component_mask = target.born_mask.reshape(-1, 1, 1).expand_as(born_error)
    born_loss, born_active = _masked_node_graph_loss(
        born_error, born_component_mask, batch, graph_count
    )

    gamma_classification = functional.binary_cross_entropy_with_logits(
        prediction.gamma_soft_logits, target.gamma_soft, reduction="none"
    )
    gamma_magnitude = functional.huber_loss(
        prediction.gamma_log_magnitude, target.gamma_log_magnitude, reduction="none"
    )
    if gamma_classification.shape != target.gamma_mask.shape:
        raise ValueError("gamma-frequency mask is invalid")
    gamma_count = target.gamma_mask.sum(dim=-1)
    gamma_valid = gamma_count > 0
    gamma_graph = (
        (gamma_classification + gamma_magnitude) * target.gamma_mask
    ).sum(dim=-1) / gamma_count.clamp_min(1)
    gamma_loss, gamma_active = _masked_graph_loss(gamma_graph, gamma_valid)

    internal_error = prediction.internal_strain - target.internal_strain
    internal_loss, internal_active = _masked_node_graph_loss(
        internal_error, target.internal_strain_mask, batch, graph_count
    )
    probe_loss, _ = piezoelectric_response_probe_loss(
        prediction.piezoelectric,
        target.piezoelectric,
        target.piezoelectric_mask,
    )

    losses = (
        piezo_loss,
        dielectric_loss,
        elastic_loss,
        born_loss,
        gamma_loss,
        internal_loss,
    )
    active = (
        piezo_active,
        dielectric_active,
        elastic_active,
        born_active,
        gamma_active,
        internal_active,
    )
    task_weights = (
        weights.piezoelectric,
        weights.dielectric,
        weights.elastic,
        weights.born_effective_charge,
        weights.gamma_frequency,
        weights.internal_strain,
    )
    if any(weight < 0.0 for weight in task_weights) or weights.piezoelectric_probe < 0.0:
        raise ValueError("response task weights must be nonnegative")
    denominator = sum(weight for weight, enabled in zip(task_weights, active, strict=True) if enabled)
    if denominator <= 0.0:
        total = sum(losses, start=prediction.piezoelectric.sum() * 0.0)
    else:
        total = sum(
            weight * loss
            for weight, loss, enabled in zip(task_weights, losses, active, strict=True)
            if enabled
        ) / denominator
    if piezo_active:
        total = total + weights.piezoelectric_probe * probe_loss
    return ResponseLossOutput(
        loss=total,
        piezoelectric_loss=piezo_loss,
        dielectric_loss=dielectric_loss,
        elastic_loss=elastic_loss,
        born_loss=born_loss,
        gamma_loss=gamma_loss,
        internal_strain_loss=internal_loss,
        piezoelectric_probe_loss=probe_loss,
        active_tasks=sum(active),
    )
