"""Parity-aware Cartesian stratified gauge atlas for rank-three conditions.

This module is the production successor to the archived finite-Hopf harmonic
conditioner.  It never selects one canonical frame.  Instead it builds finite
proper-SO(3) moving-frame atlases from Cartesian covariants and marginalizes
all signed-permutation frames.  When an eigenspace is axially degenerate, the
corresponding descriptor-frame ambiguity is integrated by a fixed circle
rule.  Smooth compact-support weights blend generic, axial, and isotropic
charts.  A nonzero descriptor whose quadratic covariance is isotropic is sent
to a fixed Cartesian cubature instead of being mistaken for a physically zero
condition.  Full O(3) compatibility is intentionally outside this module.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.utils import scatter

from gaugeflow.tensor import fixed_lossless_response_probes, piezo_from_irreps, response_field

from .schedules import CosineNoiseSchedule
from .so3_quadrature import nested_hopf_so3_grid


def _identity_like(value: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=value.dtype, device=value.device)


def _levi_civita(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    epsilon = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
    return epsilon


def cartesian_stf_moments(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the Cartesian STF realizations of degrees one, two, and three.

    The dimensions of these tensors are 3, 5, and 7 after trace constraints,
    exactly matching the low-order SO(3) irreps, without evaluating spherical
    harmonics or a Clebsch--Gordan kernel.
    """
    if directions.ndim != 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [edges,3]")
    unit = torch.nn.functional.normalize(directions, dim=-1)
    identity = _identity_like(unit)
    second_raw = torch.einsum("ei,ej->eij", unit, unit)
    second = second_raw - identity.unsqueeze(0) / 3.0
    third_raw = torch.einsum("ei,ej,ek->eijk", unit, unit, unit)
    correction = (
        torch.einsum("ei,jk->eijk", unit, identity)
        + torch.einsum("ej,ik->eijk", unit, identity)
        + torch.einsum("ek,ij->eijk", unit, identity)
    ) / 5.0
    return unit, second, third_raw - correction


def _stf_rank2(value: torch.Tensor) -> torch.Tensor:
    identity = _identity_like(value)
    symmetric = 0.5 * (value + value.transpose(-1, -2))
    return symmetric - torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1)[..., None, None] * identity / 3.0


def _proper_signed_permutations() -> torch.Tensor:
    matrices: list[torch.Tensor] = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = torch.zeros((3, 3), dtype=torch.float64)
        permutation_matrix[torch.arange(3), torch.tensor(permutation)] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = permutation_matrix @ torch.diag(torch.tensor(signs, dtype=torch.float64))
            if round(float(torch.linalg.det(matrix))) == 1:
                matrices.append(matrix)
    result = torch.stack(matrices)
    if result.shape != (24, 3, 3):
        raise RuntimeError("proper signed-permutation atlas must contain 24 frames")
    return result


def _axial_rotations(samples: int, axis: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if samples < 1:
        raise ValueError("residual circle needs at least one node")
    angle = 2.0 * math.pi * torch.arange(samples, dtype=dtype, device=device) / samples
    cosine, sine = angle.cos(), angle.sin()
    result = torch.zeros((samples, 3, 3), dtype=dtype, device=device)
    result[:, axis, axis] = 1.0
    remaining = [index for index in range(3) if index != axis]
    first, second = remaining
    result[:, first, first] = cosine
    result[:, first, second] = -sine
    result[:, second, first] = sine
    result[:, second, second] = cosine
    return result


@dataclass(frozen=True)
class CartesianGeometryQueries:
    """Condition-free Cartesian covariants and a rank-three score query."""

    first: torch.Tensor
    second: torch.Tensor
    third: torch.Tensor
    rank_three: torch.Tensor
    frame_tensor: torch.Tensor


@dataclass(frozen=True)
class CartesianGaugeAtlasOutput:
    graph_condition: torch.Tensor
    edge_response: torch.Tensor
    posterior: torch.Tensor
    candidate_prior: torch.Tensor
    candidate_mask: torch.Tensor
    aligned_tensor: torch.Tensor
    gate: torch.Tensor
    entropy: torch.Tensor
    effective_frame_count: torch.Tensor
    raw_candidate_count: torch.Tensor
    residual_kind: torch.Tensor


class _CartesianScalarBlock(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int, query_channels: int) -> None:
        super().__init__()
        angular_dim = 3 * query_channels
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_dim + 2 * angular_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        radial: torch.Tensor,
        angular: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(nodes)
        if source.numel():
            message = self.message(
                torch.cat((nodes[source], nodes[target], radial, angular[source], angular[target]), dim=-1)
            )
            # AMP linear kernels may emit BF16 while residual states remain
            # FP32. Accumulate messages in the state dtype for a stable and
            # legal mixed-precision reduction.
            aggregate.index_add_(0, target, message.to(aggregate.dtype))
            degree = torch.bincount(target, minlength=nodes.shape[0]).clamp_min(1).to(nodes)
            aggregate = aggregate / degree.unsqueeze(-1)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class CartesianSTFGeometryQueryEncoder(nn.Module):
    """Build low-order geometry covariants using only Cartesian STF algebra."""

    def __init__(self, hidden_dim: int, radial_dim: int, *, query_channels: int = 2, layers: int = 3) -> None:
        super().__init__()
        if layers < 2 or query_channels < 1:
            raise ValueError("Cartesian geometry encoder needs at least two blocks and one channel")
        self.query_channels = int(query_channels)
        self.blocks = nn.ModuleList(
            [_CartesianScalarBlock(hidden_dim, radial_dim, query_channels) for _ in range(layers)]
        )
        self.moment_weight = nn.Sequential(
            nn.Linear(2 * hidden_dim + radial_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, query_channels)
        )
        # Four rank-three Cartesian embeddings per channel: l=3 STF, both l=1
        # embeddings, and an l=2 embedding constructed through epsilon.
        self.rank_three_mix = nn.Parameter(torch.ones((query_channels, 4)))

    def _local_moments(
        self,
        nodes: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        radial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels = self.query_channels
        if not source.numel():
            return (
                nodes.new_zeros((nodes.shape[0], channels, 3)),
                nodes.new_zeros((nodes.shape[0], channels, 3, 3)),
                nodes.new_zeros((nodes.shape[0], channels, 3, 3, 3)),
            )
        weights = self.moment_weight(torch.cat((nodes[source], nodes[target], radial), dim=-1))
        first, second, third = cartesian_stf_moments(directions)
        return (
            scatter(weights.unsqueeze(-1) * first[:, None], target, dim=0, dim_size=nodes.shape[0], reduce="mean"),
            scatter(weights[:, :, None, None] * second[:, None], target, dim=0, dim_size=nodes.shape[0], reduce="mean"),
            scatter(
                weights[:, :, None, None, None] * third[:, None], target, dim=0, dim_size=nodes.shape[0], reduce="mean"
            ),
        )

    def _rank_three_query(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        identity = _identity_like(first)
        epsilon = _levi_civita(dtype=first.dtype, device=first.device)
        first_delta = torch.einsum("...i,jk->...ijk", first, identity)
        alternate_first = 0.5 * (
            torch.einsum("...j,ik->...ijk", first, identity) + torch.einsum("...k,ij->...ijk", first, identity)
        )
        cross_quadrupole = torch.einsum("iab,...a,...bj->...ij", epsilon, first, second)
        polar_quadrupole = _stf_rank2(cross_quadrupole)
        l2_embedding = 0.5 * (
            torch.einsum("ija,...ak->...ijk", epsilon, polar_quadrupole)
            + torch.einsum("ika,...aj->...ijk", epsilon, polar_quadrupole)
        )
        pieces = torch.stack((third, first_delta, alternate_first, l2_embedding), dim=-4)
        return torch.einsum("cq,...cqijk->...cijk", self.rank_three_mix.to(pieces), pieces)

    @staticmethod
    def _frame_tensor(first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        first_covariance = torch.einsum("...ci,...cj->...cij", first, first)
        second_covariance = second @ second.transpose(-1, -2)
        third_covariance = torch.einsum("...cijk,...cdjk->...cid", third, third)
        return (first_covariance + second_covariance + third_covariance).mean(dim=-3)

    def forward(
        self,
        initial_nodes: torch.Tensor,
        node_time: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        directions: torch.Tensor,
        radial: torch.Tensor,
        batch: torch.Tensor,
        graph_count: int,
    ) -> CartesianGeometryQueries:
        nodes = initial_nodes + node_time
        angular = nodes.new_zeros((nodes.shape[0], 3 * self.query_channels))
        for block in self.blocks:
            nodes = block(nodes, source, target, radial, angular)
            first, second, third = self._local_moments(nodes, source, target, directions, radial)
            angular = torch.cat(
                (
                    torch.linalg.vector_norm(first, dim=-1),
                    torch.linalg.matrix_norm(second, dim=(-2, -1)),
                    torch.linalg.vector_norm(third.flatten(-3), dim=-1),
                ),
                dim=-1,
            )
        first, second, third = self._local_moments(nodes, source, target, directions, radial)
        graph_first = scatter(first, batch, dim=0, dim_size=graph_count, reduce="mean")
        graph_second = scatter(second, batch, dim=0, dim_size=graph_count, reduce="mean")
        graph_third = scatter(third, batch, dim=0, dim_size=graph_count, reduce="mean")
        return CartesianGeometryQueries(
            first=graph_first,
            second=graph_second,
            third=graph_third,
            rank_three=self._rank_three_query(graph_first, graph_second, graph_third),
            frame_tensor=self._frame_tensor(graph_first, graph_second, graph_third),
        )


class CartesianOrbitInvariantEncoder(nn.Module):
    """Smooth proper-SO(3)-invariant token from Cartesian rank-three tensors."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    @staticmethod
    def frame_covariants(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_index = torch.einsum("bijk,bljk->bil", tensor, tensor)
        second_index = torch.einsum("bijk,bilk->bjl", tensor, tensor)
        return first_index, second_index

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Determinants and eigensystem descriptors are not implemented (and
        # are numerically unsafe) in CUDA BF16. Keep the invariant algebra and
        # frame covariance in FP32 while allowing the small MLP to autocast.
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            descriptor_tensor = tensor.float() if tensor.dtype in {torch.float16, torch.bfloat16} else tensor
            first, second = self.frame_covariants(descriptor_tensor)
            first2, second2 = first @ first, second @ second
            scale = descriptor_tensor.square().sum(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
            features = torch.stack(
                (
                    scale.squeeze(-1).squeeze(-1).squeeze(-1),
                    torch.diagonal(first, dim1=-2, dim2=-1).sum(-1),
                    torch.diagonal(first2, dim1=-2, dim2=-1).sum(-1),
                    torch.linalg.det(first),
                    torch.diagonal(second, dim1=-2, dim2=-1).sum(-1),
                    torch.diagonal(second2, dim1=-2, dim2=-1).sum(-1),
                    torch.linalg.det(second),
                    (first * second).sum(dim=(-1, -2)),
                    (first2 * second).sum(dim=(-1, -2)),
                ),
                dim=-1,
            )
            normalized = features / (1.0 + scale.reshape(-1, 1).square())
        return self.network(normalized), first + 0.61803398875 * second


@dataclass(frozen=True)
class _FrameData:
    basis: torch.Tensor
    # Smooth partition of unity: generic, lower-doublet axial,
    # upper-doublet axial, and descriptor-isotropic/global cubature.
    weights: torch.Tensor
    directional: bool


@dataclass(frozen=True)
class _CandidateMeasure:
    rotations: torch.Tensor
    prior: torch.Tensor
    raw_count: int
    raw_prior: torch.Tensor | None = None


class StratifiedCartesianGaugeAtlas(nn.Module):
    """Cartesian moving-frame atlas with chartwise proper-SO(3) marginalization.

    ``proper_frames`` resolves the 24-fold eigenframe gauge ambiguity.  It is
    not itself a quadrature for the continuous relative rotation.  Every
    generic atlas chart is consequently refined by the same deterministic
    Cartesian rotation cubature before posterior marginalization.
    """

    GENERIC = 0
    AXIAL = 1
    ISOTROPIC = 2

    def __init__(
        self,
        hidden_dim: int,
        *,
        residual_circle_samples: int = 8,
        generic_chart_samples: int = 7,
        relative_eigen_gap: float = 1e-3,
        lambda_max: float = 1.0,
        schedule: CosineNoiseSchedule | None = None,
    ) -> None:
        super().__init__()
        if (
            residual_circle_samples < 2
            or generic_chart_samples < 1
            or relative_eigen_gap <= 0.0
            or not 0.0 < lambda_max <= 1.0
        ):
            raise ValueError("invalid Cartesian gauge-atlas hyperparameters")
        self.residual_circle_samples = int(residual_circle_samples)
        self.relative_eigen_gap = float(relative_eigen_gap)
        self.lambda_max = float(lambda_max)
        self.schedule = schedule or CosineNoiseSchedule()
        self.invariant = CartesianOrbitInvariantEncoder(hidden_dim)
        self.aligned_token = nn.Sequential(nn.Linear(21, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.score_channel = nn.Parameter(torch.ones(2))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.present_bias = nn.Parameter(torch.zeros(hidden_dim))
        proper_frames = _proper_signed_permutations()
        # Discard the identity node: under the left/right proper-frame action
        # it has only 24 distinct products but would otherwise be replicated
        # 24 times and bias the finite posterior.
        generic_chart_nodes = nested_hopf_so3_grid(generic_chart_samples + 1)[1:]
        self.register_buffer("proper_frames", proper_frames)
        self.register_buffer("generic_chart_nodes", generic_chart_nodes)
        self.register_buffer(
            "generic_base_cubature",
            (
                proper_frames.float()[:, None, None]
                @ generic_chart_nodes[None, :, None]
                @ proper_frames.float()[None, None]
            ).reshape(-1, 3, 3),
        )
        self.register_buffer(
            "generic_base_cubature_high_precision",
            (
                proper_frames[:, None, None]
                @ generic_chart_nodes.to(proper_frames)[None, :, None]
                @ proper_frames[None, None]
            ).reshape(-1, 3, 3),
        )
        self.register_buffer("probes", fixed_lossless_response_probes())

    @staticmethod
    def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        coordinate = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
        return coordinate.square() * (3.0 - 2.0 * coordinate)

    def _frame_data(self, covariance: torch.Tensor, *, directional: bool = True) -> _FrameData:
        """Return an eigenframe and a smooth descriptor-ambiguity partition.

        These weights describe ambiguity of the selected Cartesian descriptor;
        they are not asserted to be the physical crystal/tensor stabilizer.
        """
        with torch.autocast(device_type=covariance.device.type, enabled=False):
            frame_covariance = covariance.float() if covariance.dtype in {torch.float16, torch.bfloat16} else covariance
            symmetric = 0.5 * (frame_covariance + frame_covariance.transpose(-1, -2))
            eigenvalues, basis = torch.linalg.eigh(symmetric)
            if torch.linalg.det(basis) < 0:
                basis = basis.clone()
                basis[:, -1] = -basis[:, -1]
            scale = torch.linalg.vector_norm(eigenvalues).clamp_min(torch.finfo(eigenvalues.dtype).eps)
            lower_gap = (eigenvalues[1] - eigenvalues[0]).abs() / scale
            upper_gap = (eigenvalues[2] - eigenvalues[1]).abs() / scale
            lower_resolved = self._smoothstep(lower_gap, 0.5 * self.relative_eigen_gap, 2.0 * self.relative_eigen_gap)
            upper_resolved = self._smoothstep(upper_gap, 0.5 * self.relative_eigen_gap, 2.0 * self.relative_eigen_gap)
        weights = torch.stack(
            (
                lower_resolved * upper_resolved,
                (1.0 - lower_resolved) * upper_resolved,
                lower_resolved * (1.0 - upper_resolved),
                (1.0 - lower_resolved) * (1.0 - upper_resolved),
            )
        )
        if not directional:
            weights = torch.zeros_like(weights)
        return _FrameData(basis, weights, directional)

    def _residual_nodes(self, kind: int, axis: int, reference: torch.Tensor) -> torch.Tensor:
        if kind == self.GENERIC:
            return _identity_like(reference).unsqueeze(0)
        if kind == self.AXIAL:
            return _axial_rotations(self.residual_circle_samples, axis, dtype=reference.dtype, device=reference.device)
        raise ValueError("isotropic charts use the global Cartesian cubature")

    @staticmethod
    def _components(data: _FrameData) -> list[tuple[int, int, torch.Tensor]]:
        # lower eigenvalue doublet leaves the third axis identifiable; upper
        # doublet leaves the first axis identifiable.
        definitions = ((0, 2), (1, 2), (1, 0), (2, 2))
        return [
            (kind, axis, weight) for (kind, axis), weight in zip(definitions, data.weights) if bool(weight.detach() > 0)
        ]

    def _base_cubature(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.dtype == torch.float64:
            return self.generic_base_cubature_high_precision.to(reference)
        return self.generic_base_cubature.to(reference)

    def _component_rotations(
        self,
        geometry: _FrameData,
        condition: _FrameData,
        geometry_kind: int,
        geometry_axis: int,
        condition_kind: int,
        condition_axis: int,
    ) -> torch.Tensor:
        base = self._base_cubature(geometry.basis)
        if geometry_kind == self.ISOTROPIC and condition_kind == self.ISOTROPIC:
            return base
        if geometry_kind == self.ISOTROPIC:
            right = self._residual_nodes(condition_kind, condition_axis, condition.basis) @ condition.basis.T
            return (base[:, None] @ right[None]).reshape(-1, 3, 3)
        left = geometry.basis @ self._residual_nodes(geometry_kind, geometry_axis, geometry.basis)
        if condition_kind == self.ISOTROPIC:
            return (left[:, None] @ base[None]).reshape(-1, 3, 3)
        right = self._residual_nodes(condition_kind, condition_axis, condition.basis) @ condition.basis.T
        return (left[:, None, None] @ base[None, :, None] @ right[None, None]).reshape(-1, 3, 3)

    def _raw_candidate_measure(self, geometry: _FrameData, condition: _FrameData) -> tuple[torch.Tensor, torch.Tensor]:
        if not geometry.directional or not condition.directional:
            empty = geometry.basis.new_empty((0, 3, 3))
            return empty, geometry.basis.new_empty((0,))
        rotations: list[torch.Tensor] = []
        masses: list[torch.Tensor] = []
        for g_kind, g_axis, g_weight in self._components(geometry):
            for c_kind, c_axis, c_weight in self._components(condition):
                candidates = self._component_rotations(geometry, condition, g_kind, g_axis, c_kind, c_axis)
                component_mass = g_weight * c_weight
                rotations.append(candidates)
                masses.append(component_mass.expand(candidates.shape[0]) / candidates.shape[0])
        if not rotations:
            raise RuntimeError(
                "directional atlas frames must activate at least one partition component"
            )
        raw_rotations = torch.cat(rotations)
        raw_prior = torch.cat(masses)
        return raw_rotations, raw_prior / raw_prior.sum()

    @staticmethod
    def _deduplicate_measure(
        rotations: torch.Tensor,
        raw_prior: torch.Tensor | None = None,
        *,
        tolerance: float = 1e-7,
    ) -> _CandidateMeasure:
        """Aggregate numerically identical rotations into a discrete measure.

        Each chart component carries its partition-of-unity mass, distributed
        uniformly within that component. Duplicate rotations are represented
        once with the sum of their raw masses. Consequently the posterior is
        invariant to enumeration order and to measure-preserving duplicate
        expansion of the raw list.
        """
        raw_count = int(rotations.shape[0])
        if raw_count == 0:
            return _CandidateMeasure(rotations, rotations.new_empty((0,)), 0, rotations.new_empty((0,)))
        if raw_prior is None:
            raw_prior = rotations.new_full((raw_count,), 1.0 / raw_count)
        if raw_prior.shape != (raw_count,) or bool((raw_prior < 0).any()):
            raise ValueError("raw candidate prior must be non-negative and match rotations")
        raw_prior = raw_prior / raw_prior.sum().clamp_min(torch.finfo(raw_prior.dtype).tiny)
        keys = torch.round(rotations.detach().reshape(raw_count, 9) / tolerance).to(torch.int64)
        _, inverse, multiplicity = torch.unique(keys, dim=0, sorted=True, return_inverse=True, return_counts=True)
        raw_index = torch.arange(raw_count, device=rotations.device)
        first = torch.full((multiplicity.shape[0],), raw_count, dtype=torch.long, device=rotations.device)
        first.scatter_reduce_(0, inverse, raw_index, reduce="amin", include_self=True)
        unique_rotations = rotations[first]
        prior = rotations.new_zeros((multiplicity.shape[0],))
        prior.index_add_(0, inverse, raw_prior)
        return _CandidateMeasure(unique_rotations, prior, raw_count, raw_prior)

    def _candidate_measure(self, geometry: _FrameData, condition: _FrameData) -> _CandidateMeasure:
        # SO(3) nodes, multiplicity keys, and prior masses are numerical
        # geometry and remain FP32/FP64 under AMP. Only learned contractions
        # below are allowed to autocast.
        with torch.autocast(device_type=geometry.basis.device.type, enabled=False):
            geometry_components = self._components(geometry)
            condition_components = self._components(condition)
            # In the interior generic chart there is exactly one left frame,
            # one right frame, and the 4,032 base nodes are pre-qualified as
            # unique. A rigid two-sided rotation is bijective, so running
            # torch.unique over quantized 3x3 matrices on every forward cannot
            # change the measure; it only adds a large GPU synchronization and
            # sort. The descriptor-isotropic/interior pair has the same proof.
            # Mixed or axial charts retain the full multiplicity-corrected
            # deduplication because their residual products can collide.
            if len(geometry_components) == len(condition_components) == 1:
                geometry_kind, geometry_axis, _ = geometry_components[0]
                condition_kind, condition_axis, _ = condition_components[0]
                proven_unique = (
                    geometry_kind == condition_kind == self.GENERIC
                    or geometry_kind == condition_kind == self.ISOTROPIC
                )
                if proven_unique:
                    rotations = self._component_rotations(
                        geometry,
                        condition,
                        geometry_kind,
                        geometry_axis,
                        condition_kind,
                        condition_axis,
                    )
                    count = int(rotations.shape[0])
                    prior = rotations.new_full((count,), 1.0 / count)
                    return _CandidateMeasure(rotations, prior, count, prior)
            rotations, raw_prior = self._raw_candidate_measure(geometry, condition)
            return self._deduplicate_measure(rotations, raw_prior)

    @staticmethod
    def _rotate_rank_three(tensor: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kab,kcd,kef,bdf->kace", rotations, rotations, rotations, tensor)

    @staticmethod
    def _rotate_rank_three_batch(tensors: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        """Rotate a grouped batch of tensors by its grouped atlas candidates."""
        return torch.einsum("gfia,gfjb,gfkc,gabc->gfijk", rotations, rotations, rotations, tensors)

    def forward(
        self,
        piezo_irreps: torch.Tensor,
        condition_present: torch.Tensor,
        edge_directions: torch.Tensor,
        edge_graph: torch.Tensor,
        geometry_queries: CartesianGeometryQueries,
        time: torch.Tensor,
    ) -> CartesianGaugeAtlasOutput:
        graphs = piezo_irreps.shape[0]
        if piezo_irreps.shape != (graphs, 18) or time.shape != (graphs,):
            raise ValueError("piezo condition and time must provide one record per graph")
        if condition_present.shape not in {(graphs,), (graphs, 1)}:
            raise ValueError("condition-present flag must provide one value per graph")
        if geometry_queries.rank_three.shape[:2] != (graphs, 2):
            raise ValueError("Cartesian geometry query must provide two channels per graph")
        tensor = piezo_from_irreps(piezo_irreps)
        invariant, condition_covariance = self.invariant(tensor)
        geometry_covariance = geometry_queries.frame_tensor
        geometry_signal = torch.linalg.vector_norm(geometry_queries.rank_three.flatten(1), dim=-1)
        condition_signal = torch.linalg.vector_norm(tensor.flatten(1), dim=-1)
        signal_floor = 32.0 * torch.finfo(tensor.dtype).eps
        geometry_frames = [
            self._frame_data(
                geometry_covariance[index],
                directional=bool(geometry_signal[index].detach() > signal_floor),
            )
            for index in range(graphs)
        ]
        condition_frames = [
            self._frame_data(
                condition_covariance[index],
                directional=bool(condition_signal[index].detach() > signal_floor),
            )
            for index in range(graphs)
        ]
        measures = [self._candidate_measure(left, right) for left, right in zip(geometry_frames, condition_frames)]
        counts = torch.tensor(
            [value.rotations.shape[0] for value in measures],
            dtype=torch.long,
            device=tensor.device,
        )
        raw_counts = torch.tensor([value.raw_count for value in measures], dtype=torch.long, device=tensor.device)
        maximum = max(int(counts.max()), 1)
        probability_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
        posterior = torch.zeros((graphs, maximum), dtype=probability_dtype, device=tensor.device)
        candidate_prior = torch.zeros((graphs, maximum), dtype=probability_dtype, device=tensor.device)
        candidate_mask = torch.zeros((graphs, maximum), dtype=torch.bool, device=tensor.device)
        aligned = torch.zeros_like(tensor)
        entropy = torch.zeros((graphs,), dtype=probability_dtype, device=tensor.device)
        confidence = torch.zeros((graphs,), dtype=probability_dtype, device=tensor.device)
        temperature = self.log_temperature.exp().clamp_min(1e-4).to(tensor)
        for count in torch.unique(counts).tolist():
            if count == 0:
                continue
            indices = torch.where(counts == count)[0]
            rotations = torch.stack([measures[int(index)].rotations for index in indices])
            prior = torch.stack([measures[int(index)].prior for index in indices])
            rotated = self._rotate_rank_three_batch(tensor[indices], rotations)
            query = geometry_queries.rank_three[indices]
            score = torch.einsum("gfijk,gcijk,c->gf", rotated, query, self.score_channel.to(query))
            weights = torch.softmax(
                score / temperature + prior.clamp_min(torch.finfo(prior.dtype).tiny).log(),
                dim=-1,
            )
            posterior[indices, :count] = weights
            candidate_prior[indices, :count] = prior
            candidate_mask[indices, :count] = True
            aligned[indices] = torch.einsum("gf,gfijk->gijk", weights, rotated)
            local_entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
            entropy[indices] = local_entropy
            local_kl = (
                weights
                * (
                    weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
                    - prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
                )
            ).sum(-1)
            maximum_kl = -prior.clamp_min(torch.finfo(prior.dtype).tiny).log().amax(-1)
            confidence[indices] = (local_kl / maximum_kl.clamp_min(1e-12)).clamp(0.0, 1.0)
        available = counts > 0
        snr = self.schedule.snr(time)
        gate = self.lambda_max * (snr / (1.0 + snr)) * confidence * available.to(confidence)
        fixed_response = response_field(aligned.unsqueeze(1), self.probes.to(aligned).unsqueeze(0)).reshape(graphs, 18)
        aligned_features = torch.cat(
            (fixed_response, entropy.unsqueeze(-1), counts.to(tensor).log1p().unsqueeze(-1), gate.unsqueeze(-1)), dim=-1
        )
        graph_condition = invariant + gate.unsqueeze(-1) * self.aligned_token(aligned_features) + self.present_bias
        present = condition_present.reshape(graphs, 1).to(dtype=torch.bool)
        graph_condition = torch.where(
            present, graph_condition, self.null_condition.unsqueeze(0).expand_as(graph_condition)
        )
        if edge_directions.numel():
            edge_response = response_field(aligned[edge_graph], edge_directions) * gate[edge_graph].unsqueeze(-1)
            edge_response = torch.where(present[edge_graph], edge_response, torch.zeros_like(edge_response))
        else:
            edge_response = edge_directions.new_empty((0, 3))
        residual_kind = torch.tensor(
            [
                max(
                    int(torch.argmax(left.weights).item()),
                    int(torch.argmax(right.weights).item()),
                )
                for left, right in zip(geometry_frames, condition_frames)
            ],
            dtype=torch.long,
            device=tensor.device,
        )
        # Both axial components have the public AXIAL diagnostic code; the
        # fourth partition component denotes descriptor-isotropic cubature.
        residual_kind = torch.where(
            residual_kind == 3,
            torch.full_like(residual_kind, self.ISOTROPIC),
            torch.where(residual_kind > 0, torch.full_like(residual_kind, self.AXIAL), residual_kind),
        )
        return CartesianGaugeAtlasOutput(
            graph_condition=graph_condition,
            edge_response=edge_response,
            posterior=posterior,
            candidate_prior=candidate_prior,
            candidate_mask=candidate_mask,
            aligned_tensor=aligned,
            gate=gate,
            entropy=entropy,
            effective_frame_count=counts,
            raw_candidate_count=raw_counts,
            residual_kind=residual_kind,
        )

    def null_output(
        self,
        *,
        graph_count: int,
        edge_count: int,
        reference: torch.Tensor,
    ) -> CartesianGaugeAtlasOutput:
        """Return the learned null-condition token without enumerating frames.

        Missing conditioning is a separate model state from a physically zero
        tensor.  Tensor-free pretraining therefore bypasses the Cartesian
        candidate measure completely; a present zero tensor still goes through
        :meth:`forward` and retains its invariant/present semantics.
        """
        if graph_count < 1 or edge_count < 0:
            raise ValueError("null atlas output requires positive graphs and nonnegative edges")
        dtype = reference.dtype
        device = reference.device
        probability_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        zeros = torch.zeros((graph_count,), dtype=probability_dtype, device=device)
        return CartesianGaugeAtlasOutput(
            graph_condition=self.null_condition.to(reference).unsqueeze(0).expand(graph_count, -1),
            edge_response=reference.new_zeros((edge_count, 3)),
            posterior=torch.ones((graph_count, 1), dtype=probability_dtype, device=device),
            candidate_prior=torch.ones((graph_count, 1), dtype=probability_dtype, device=device),
            candidate_mask=torch.ones((graph_count, 1), dtype=torch.bool, device=device),
            aligned_tensor=reference.new_zeros((graph_count, 3, 3, 3)),
            gate=zeros,
            entropy=zeros,
            effective_frame_count=torch.zeros((graph_count,), dtype=torch.long, device=device),
            raw_candidate_count=torch.zeros((graph_count,), dtype=torch.long, device=device),
            residual_kind=torch.full(
                (graph_count,), self.ISOTROPIC, dtype=torch.long, device=device
            ),
        )
