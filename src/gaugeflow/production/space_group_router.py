"""Full-O(3) compatibility for terminal groups and reachable child paths.

Compatibility is never applied as a hard filter to a parent space group.  A
centrosymmetric parent may be valid when a sampled inversion-odd distortion
reaches a non-centrosymmetric child compatible with the requested response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from pymatgen.symmetry.groups import SpaceGroup
from torch import nn

from gaugeflow.conditioning import normalized_low_order_orbit_invariants
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3

from .lattice_volume_shape import PointGroupMetricChart
from .so3_quadrature import nested_hopf_so3_grid


@dataclass(frozen=True)
class SpaceGroupCompatibilityRecord:
    number: int
    symbol: str
    point_group: str
    fractional_operations: torch.Tensor
    operations: torch.Tensor
    fractional_to_cartesian: torch.Tensor
    metric_chart: PointGroupMetricChart
    reynolds_irrep: torch.Tensor
    compatible_rank: int


def reynolds_project(tensor: torch.Tensor, operations: torch.Tensor) -> torch.Tensor:
    """Project a polar rank-three tensor with every full-O(3) operation."""
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError("piezoelectric tensor must end in [3,3,3]")
    if operations.ndim != 3 or operations.shape[-2:] != (3, 3):
        raise ValueError("point-group operations must have shape [operations,3,3]")
    transformed = rotate_rank3(tensor.unsqueeze(-4), operations.to(tensor))
    return transformed.mean(dim=-4)


def reynolds_irrep_matrix(operations: torch.Tensor) -> torch.Tensor:
    """Return the 18-dimensional Reynolds operator in the e3nn piezo basis."""
    basis = torch.eye(18, dtype=operations.dtype, device=operations.device)
    tensors = piezo_from_irreps(basis)
    projected = reynolds_project(tensors, operations)
    # Rows are projected basis inputs, hence transpose for column-operator
    # diagnostics.  Applying row irreps remains ``x @ matrix.T``.
    numerical = piezo_to_irreps(projected).transpose(0, 1).contiguous()
    # The Cartesian/e3nn conversion contains float32 Clebsch--Gordan constants
    # in the installed e3nn build, leaving O(1e-8) idempotence drift even for
    # float64 inputs.  Reynolds averaging is mathematically an orthogonal
    # projector, so recover its range by SVD and return the exact numerical
    # projector rather than propagating that basis-conversion drift.
    left, singular_values, _ = torch.linalg.svd(numerical)
    retained = left[:, singular_values > 0.5]
    return retained @ retained.transpose(-1, -2)


@lru_cache(maxsize=230)
def cartesian_point_group_operations(
    space_group_number: int,
) -> tuple[str, str, torch.Tensor, PointGroupMetricChart]:
    """Build the conventional Cartesian O(3) point group for a space group."""
    if not 1 <= space_group_number <= 230:
        raise ValueError("space-group number must lie in 1..230")
    space_group = SpaceGroup.from_int_number(space_group_number)
    # Extract the point-group action from the actual space-group setting.
    # Constructing ``PointGroup(space_group.point_group)`` is not total over
    # pymatgen's orientation aliases (for example ``-4m2``), and silently
    # canonicalising those aliases would lose the fractional setting needed by
    # the lattice chart.
    fractional_unique: list[np.ndarray] = []
    for operation in space_group.symmetry_ops:
        rotation = np.asarray(operation.rotation_matrix, dtype=np.float64)
        if not any(np.array_equal(rotation, seen) for seen in fractional_unique):
            fractional_unique.append(rotation)
    fractional = torch.from_numpy(np.stack(fractional_unique)).to(dtype=torch.float64)
    metric_chart = PointGroupMetricChart.from_fractional_operations(fractional)
    operations = metric_chart.cartesian_operations
    determinant = torch.linalg.det(operations)
    if not torch.allclose(determinant.abs(), torch.ones_like(determinant), atol=1e-10, rtol=1e-10):
        raise RuntimeError("pymatgen point group did not produce O(3) operations")
    # Preserve improper operations: they are essential to physical
    # compatibility for an odd-rank polar tensor.
    unique: list[torch.Tensor] = []
    for operation in operations:
        if not any(torch.allclose(operation, seen, atol=1e-10, rtol=1e-10) for seen in unique):
            unique.append(operation)
    # PointGroup already provides unique fractional operations. Keep the
    # chart and operation arrays index-aligned for closure/invariance audits.
    if len(unique) != operations.shape[0]:
        raise RuntimeError("point-group operation catalogue contains numerical duplicates")
    return space_group.symbol, space_group.point_group, operations, metric_chart


@lru_cache(maxsize=230)
def compatibility_record(space_group_number: int) -> SpaceGroupCompatibilityRecord:
    symbol, point_group, operations, metric_chart = cartesian_point_group_operations(
        space_group_number
    )
    reynolds = reynolds_irrep_matrix(operations)
    singular_values = torch.linalg.svdvals(reynolds)
    rank = int((singular_values > 1e-8).sum())
    return SpaceGroupCompatibilityRecord(
        number=space_group_number,
        symbol=symbol,
        point_group=point_group,
        fractional_operations=metric_chart.fractional_operations,
        operations=operations,
        fractional_to_cartesian=metric_chart.fractional_to_cartesian,
        metric_chart=metric_chart,
        reynolds_irrep=reynolds,
        compatible_rank=rank,
    )


def orbit_compatibility_residual(
    piezo_irreps: torch.Tensor,
    operations: torch.Tensor,
    rotations: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Finite-rule approximation to ``min_R ||rho(R)e-Pi rho(R)e||/||e||``."""
    if piezo_irreps.ndim != 2 or piezo_irreps.shape[-1] != 18:
        raise ValueError("piezo irreps must have shape [batch,18]")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("proper rotations must have shape [frames,3,3]")
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=2e-5, rtol=2e-5):
        raise ValueError("compatibility minimization uses proper SO(3) frames")
    tensor = piezo_from_irreps(piezo_irreps)
    rotated = rotate_rank3(tensor.unsqueeze(1), rotations.to(tensor).unsqueeze(0))
    projected = reynolds_project(rotated, operations.to(rotated))
    numerator = torch.linalg.vector_norm((rotated - projected).flatten(-3), dim=-1)
    denominator = torch.linalg.vector_norm(tensor.flatten(-3), dim=-1).unsqueeze(-1)
    residual = numerator.amin(dim=-1) / (denominator.squeeze(-1) + epsilon)
    return torch.where(denominator.squeeze(-1) <= epsilon, torch.zeros_like(residual), residual)


class TerminalGroupCompatibilityRouter(nn.Module):
    """Invariant prior times the Reynolds factor for a *terminal* space group."""

    def __init__(
        self,
        space_groups: tuple[int, ...] | list[int],
        *,
        hidden_dim: int = 128,
        rotation_count: int = 240,
        hard_zero_rank: bool = True,
    ) -> None:
        super().__init__()
        selected = tuple(int(value) for value in space_groups)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("router requires unique represented space groups")
        if any(value < 1 or value > 230 for value in selected):
            raise ValueError("space-group number lies outside 1..230")
        self.records = tuple(compatibility_record(value) for value in selected)
        self.hard_zero_rank = hard_zero_rank
        self.prior = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(selected))
        )
        self.log_beta = nn.Parameter(torch.zeros(len(selected)))
        self.register_buffer("rotations", nested_hopf_so3_grid(rotation_count, dtype=torch.float64))
        self.register_buffer(
            "compatible_rank", torch.tensor([record.compatible_rank for record in self.records], dtype=torch.long)
        )

    def compatibility(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        values = [
            orbit_compatibility_residual(
                piezo_irreps,
                record.operations.to(piezo_irreps),
                self.rotations.to(piezo_irreps),
            )
            for record in self.records
        ]
        return torch.stack(values, dim=-1)

    def forward(self, piezo_irreps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        invariants = normalized_low_order_orbit_invariants(piezo_irreps)
        prior_logits = self.prior(invariants)
        residual = self.compatibility(piezo_irreps)
        beta = torch.nn.functional.softplus(self.log_beta).to(residual)
        logits = prior_logits - beta * residual.square()
        physical_zero = torch.linalg.vector_norm(piezo_irreps, dim=-1) <= 1e-12
        if self.hard_zero_rank:
            incompatible = (self.compatible_rank == 0).unsqueeze(0) & ~physical_zero.unsqueeze(-1)
            logits = logits.masked_fill(incompatible, -torch.inf)
        return logits, residual


@dataclass(frozen=True)
class ReachableChildPath:
    """One representation of a physical parent-to-child path class.

    ``equivalence_class`` identifies paths related by parent normalizers,
    k-star relabeling, OPD basis gauges, unimodular supercell basis changes, or
    domain relabeling.  Such representations are deduplicated before a prior is
    built.  ``class_prior_mass`` is the physical base-measure mass of the
    *class*, not a multiplicity attached to one catalogue tuple.
    """

    parent_index: int
    child_space_group: int
    equivalence_class: str
    label: str
    class_prior_mass: float
    exact_branch: bool = False

    def __post_init__(self) -> None:
        if (
            self.parent_index < 0
            or not 1 <= self.child_space_group <= 230
            or not self.equivalence_class
            or not self.label
            or not math.isfinite(self.class_prior_mass)
            or self.class_prior_mass <= 0.0
        ):
            raise ValueError("reachable child path metadata is invalid")


@dataclass(frozen=True)
class ReachableChildRouting:
    parent_log_probability: torch.Tensor
    path_given_parent_log_probability: torch.Tensor
    path_compatibility_residual: torch.Tensor
    path_joint_log_probability: torch.Tensor


def _canonicalize_reachable_paths(
    parent_space_groups: tuple[int, ...],
    paths: tuple[ReachableChildPath, ...],
) -> tuple[tuple[ReachableChildPath, ...], tuple[int, ...]]:
    """Deduplicate catalogue representations into physical path classes.

    Duplicate representations must agree on their terminal group, physical
    class mass, and exact-branch status.  The result is sorted by parent and
    equivalence key, so input enumeration order cannot affect model columns.
    """
    grouped: dict[tuple[int, str], list[ReachableChildPath]] = {}
    for path in paths:
        grouped.setdefault((path.parent_index, path.equivalence_class), []).append(path)

    canonical: list[ReachableChildPath] = []
    multiplicity: list[int] = []
    for key in sorted(grouped):
        representations = grouped[key]
        reference = representations[0]
        if any(
            path.child_space_group != reference.child_space_group
            or path.exact_branch != reference.exact_branch
            or not math.isclose(
                path.class_prior_mass,
                reference.class_prior_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for path in representations[1:]
        ):
            raise ValueError(
                "equivalent reachable-path representations disagree on physical metadata"
            )
        canonical.append(
            ReachableChildPath(
                parent_index=reference.parent_index,
                child_space_group=reference.child_space_group,
                equivalence_class=reference.equivalence_class,
                label=min(path.label for path in representations),
                class_prior_mass=reference.class_prior_mass,
                exact_branch=reference.exact_branch,
            )
        )
        multiplicity.append(len(representations))

    for parent_index, parent_space_group in enumerate(parent_space_groups):
        selected = [path for path in canonical if path.parent_index == parent_index]
        exact = [path for path in selected if path.exact_branch]
        if len(exact) != 1:
            raise ValueError("every parent requires exactly one explicit exact path class")
        if exact[0].child_space_group != parent_space_group:
            raise ValueError("an exact path must terminate in its parent space group")
    return tuple(canonical), tuple(multiplicity)


class ReachableChildCompatibilityRouter(nn.Module):
    """Marginalize tensor compatibility over reachable child branches.

    The module implements

    ``p(Gp|[e]) proportional p0(Gp|c_inv) sum_d p0(d|Gp)
    exp(-beta_d r_H(d)([e])^2)``.

    The path catalogue must include the exact branch explicitly when it is
    allowed.  Therefore a rank-zero parent is not rejected merely because its
    exact branch is incompatible; another reachable child can carry the mass.
    """

    def __init__(
        self,
        parent_space_groups: tuple[int, ...] | list[int],
        paths: tuple[ReachableChildPath, ...] | list[ReachableChildPath],
        *,
        hidden_dim: int = 128,
        rotation_count: int = 240,
        hard_zero_rank: bool = True,
    ) -> None:
        super().__init__()
        parents = tuple(int(value) for value in parent_space_groups)
        catalogue_paths = tuple(paths)
        if not parents or len(set(parents)) != len(parents) or any(value < 1 or value > 230 for value in parents):
            raise ValueError("reachable router requires unique parent groups in 1..230")
        if not catalogue_paths or any(path.parent_index >= len(parents) for path in catalogue_paths):
            raise ValueError("every reachable path must reference a represented parent")
        if set(path.parent_index for path in catalogue_paths) != set(range(len(parents))):
            raise ValueError("every represented parent needs at least one reachable child path")
        selected_paths, representation_multiplicity = _canonicalize_reachable_paths(
            parents, catalogue_paths
        )
        self.parent_space_groups = parents
        self._path_equivalence_classes = tuple(
            path.equivalence_class for path in selected_paths
        )
        self.child_records = tuple(compatibility_record(path.child_space_group) for path in selected_paths)
        self.hard_zero_rank = hard_zero_rank
        self.parent_prior = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(parents))
        )
        self.path_prior = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(selected_paths))
        )
        self.log_beta = nn.Parameter(torch.zeros(len(selected_paths)))
        self.register_buffer("rotations", nested_hopf_so3_grid(rotation_count, dtype=torch.float64))
        self.register_buffer(
            "path_parent_index",
            torch.tensor([path.parent_index for path in selected_paths], dtype=torch.long),
        )
        self.register_buffer(
            "path_log_base_measure",
            torch.tensor(
                [math.log(path.class_prior_mass) for path in selected_paths],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "catalogue_representation_multiplicity",
            torch.tensor(representation_multiplicity, dtype=torch.long),
        )
        self.register_buffer(
            "child_compatible_rank",
            torch.tensor([record.compatible_rank for record in self.child_records], dtype=torch.long),
        )

    @property
    def path_equivalence_classes(self) -> tuple[str, ...]:
        """Canonical physical class keys in model-column order."""
        return self._path_equivalence_classes

    def compatibility(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        values = [
            orbit_compatibility_residual(
                piezo_irreps,
                record.operations.to(piezo_irreps),
                self.rotations.to(piezo_irreps),
            )
            for record in self.child_records
        ]
        return torch.stack(values, dim=-1)

    @staticmethod
    def _normalize_finite_logits(logits: torch.Tensor, *, name: str) -> torch.Tensor:
        """Normalize logits while allowing explicit ``-inf`` catalogue masks."""
        if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
            raise ValueError(f"{name} contains NaN or positive infinity")
        normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        if bool(torch.isneginf(normalizer).any()):
            raise ValueError(f"{name} masks every available choice")
        return logits - normalizer

    def route_from_logits(
        self,
        piezo_irreps: torch.Tensor,
        parent_prior_logits: torch.Tensor,
        path_prior_logits: torch.Tensor,
    ) -> ReachableChildRouting:
        """Combine externally predicted structural priors with orbit compatibility.

        ``parent_prior_logits`` may come from invariant composition/size context,
        while ``path_prior_logits`` may additionally read the sampled parent
        geometry.  This is the production interface for
        ``p_0(G_p|c_inv)`` and ``p_0(d|x_p)``; tensor compatibility remains a
        separate, auditable multiplicative factor.
        """
        if piezo_irreps.ndim != 2 or piezo_irreps.shape[-1] != 18:
            raise ValueError("piezo irreps must have shape [batch,18]")
        batch = piezo_irreps.shape[0]
        parent_count = len(self.parent_space_groups)
        path_count = self.path_parent_index.numel()
        if parent_prior_logits.shape != (batch, parent_count):
            raise ValueError("parent prior logits do not match the represented parents")
        if path_prior_logits.shape != (batch, path_count):
            raise ValueError("path prior logits do not match the deduplicated physical path classes")
        parent_log_prior = self._normalize_finite_logits(
            parent_prior_logits, name="parent prior logits"
        )
        residual = self.compatibility(piezo_irreps)
        compatibility = -torch.nn.functional.softplus(self.log_beta).to(residual) * residual.square()
        physical_zero = torch.linalg.vector_norm(piezo_irreps, dim=-1) <= 1e-12
        if self.hard_zero_rank:
            incompatible = (self.child_compatible_rank == 0).unsqueeze(0) & ~physical_zero.unsqueeze(-1)
            compatibility = compatibility.masked_fill(incompatible, -torch.inf)

        if bool(torch.isnan(path_prior_logits).any()) or bool(torch.isposinf(path_prior_logits).any()):
            raise ValueError("path prior logits contain NaN or positive infinity")
        measured_path_logits = path_prior_logits + self.path_log_base_measure.to(path_prior_logits)
        conditional_path = torch.full_like(path_prior_logits, -torch.inf)
        compatible_path = torch.full_like(path_prior_logits, -torch.inf)
        parent_evidence = torch.full_like(parent_log_prior, -torch.inf)
        for parent in range(len(self.parent_space_groups)):
            selected = self.path_parent_index == parent
            selected_logits = measured_path_logits[:, selected]
            path_normalizer = torch.logsumexp(selected_logits, dim=-1, keepdim=True)
            available = torch.isfinite(path_normalizer.squeeze(-1))
            if bool(available.any()):
                normalized = selected_logits[available] - path_normalizer[available]
                conditional_path[available.unsqueeze(-1) & selected.unsqueeze(0)] = normalized.reshape(-1)
            compatible_path[:, selected] = conditional_path[:, selected] + compatibility[:, selected]
            parent_evidence[:, parent] = torch.logsumexp(compatible_path[:, selected], dim=-1)

        unnormalized_parent = parent_log_prior + parent_evidence
        parent_normalizer = torch.logsumexp(unnormalized_parent, dim=-1, keepdim=True)
        if bool(torch.isneginf(parent_normalizer).any()):
            failed = torch.nonzero(torch.isneginf(parent_normalizer.squeeze(-1))).flatten().tolist()
            raise ValueError(
                "no tensor-compatible reachable child path for batch rows "
                f"{failed}"
            )
        parent_log_probability = unnormalized_parent - parent_normalizer
        path_given_parent = torch.full_like(compatible_path, -torch.inf)
        for parent in range(len(self.parent_space_groups)):
            selected = self.path_parent_index == parent
            normalizer = torch.logsumexp(compatible_path[:, selected], dim=-1, keepdim=True)
            available = torch.isfinite(normalizer.squeeze(-1))
            if bool(available.any()):
                normalized = compatible_path[available][:, selected] - normalizer[available]
                path_given_parent[available.unsqueeze(-1) & selected.unsqueeze(0)] = normalized.reshape(-1)
        joint = parent_log_probability[:, self.path_parent_index] + path_given_parent
        return ReachableChildRouting(
            parent_log_probability=parent_log_probability,
            path_given_parent_log_probability=path_given_parent,
            path_compatibility_residual=residual,
            path_joint_log_probability=joint,
        )

    def forward(self, piezo_irreps: torch.Tensor) -> ReachableChildRouting:
        invariants = normalized_low_order_orbit_invariants(piezo_irreps)
        return self.route_from_logits(
            piezo_irreps,
            self.parent_prior(invariants),
            self.path_prior(invariants),
        )
