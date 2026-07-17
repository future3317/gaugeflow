"""Leakage-free parent and symmetry-breaking blueprint contracts.

The parent blueprint is deliberately not identified with the symmetry of the
generated child.  A child may lower the parent symmetry through a finite set of
commensurate, order-parameter-direction (OPD) branches.  The objects in this
module contain only generative variables or catalogue data; paired target CIF
metadata is never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def trace_free_projector(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Kelvin-coordinate projector onto symmetric trace-free matrices."""
    identity = torch.eye(6, dtype=dtype, device=device)
    trace = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    return identity - torch.outer(trace, trace) / 3.0


@dataclass(frozen=True)
class ParentBlueprint:
    """Discrete geometric parent-phase description.

    The parent declares only a space group and its geometric Wyckoff carrier.
    Terminal integer elements are sampled as a separate occupational field.
    """

    parent_space_group: int
    wyckoff_orbits: tuple[str, ...]
    multiplicities: tuple[int, ...]

    def __post_init__(self) -> None:
        sites = len(self.wyckoff_orbits)
        if not 1 <= self.parent_space_group <= 230:
            raise ValueError("parent space-group number must lie in 1..230")
        if sites < 1 or len(self.multiplicities) != sites:
            raise ValueError("parent Wyckoff and multiplicity tuples must align")
        if any(value < 1 for value in self.multiplicities):
            raise ValueError("parent Wyckoff multiplicities must be positive")

    @property
    def atom_count(self) -> int:
        return sum(self.multiplicities)


@dataclass(frozen=True)
class ParentBlueprintBatch:
    """Minimal P1 parent batch used by the tensor-free S1a substrate.

    P1 deliberately treats every generated site as asymmetric.  It qualifies
    the joint generator without reading a paired target space group or Wyckoff
    labelling; it is not the future 230-space-group parent sampler.
    """

    node_counts: torch.Tensor
    batch: torch.Tensor
    shape_projector: torch.Tensor
    fractional_to_cartesian: torch.Tensor

    @classmethod
    def from_node_counts(
        cls,
        node_counts: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "ParentBlueprintBatch":
        selected_device = torch.device(device) if device is not None else node_counts.device
        counts = node_counts.to(device=selected_device, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 1).any()):
            raise ValueError("parent node counts must be a nonempty positive vector")
        graphs = counts.numel()
        graph_ids = torch.arange(graphs, device=selected_device)
        batch = torch.repeat_interleave(graph_ids, counts)
        projector = trace_free_projector(dtype=dtype, device=selected_device)
        chart = torch.eye(3, dtype=dtype, device=selected_device)
        return cls(
            node_counts=counts,
            batch=batch,
            shape_projector=projector.expand(graphs, -1, -1).clone(),
            fractional_to_cartesian=chart.expand(graphs, -1, -1).clone(),
        )


@dataclass(frozen=True)
class OccupationalPattern:
    """Strictly ordered categorical coloring of one expanded geometry carrier.

    ``site_classes`` is the element-name-independent partition of sites and
    ``species_by_class`` assigns one physical zero-based element token to each
    class.  It is neither a partial occupancy nor a probability vector.
    """

    site_classes: torch.Tensor
    species_by_class: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.site_classes.ndim != 1
            or self.site_classes.dtype != torch.long
            or self.site_classes.numel() < 1
        ):
            raise ValueError("occupational site classes must be a nonempty int64 vector")
        if (
            self.species_by_class.ndim != 1
            or self.species_by_class.dtype != torch.long
            or self.species_by_class.numel() < 1
        ):
            raise ValueError("occupational class species must be a nonempty int64 vector")
        classes = torch.unique(self.site_classes, sorted=True)
        expected = torch.arange(
            self.species_by_class.numel(),
            dtype=torch.long,
            device=self.site_classes.device,
        )
        if not torch.equal(classes, expected):
            raise ValueError("occupational classes must be contiguous from zero")
        if bool(
            (
                (self.species_by_class < 0)
                | (self.species_by_class >= CHEMICAL_ELEMENT_COUNT)
            ).any()
        ):
            raise ValueError("occupational species must lie in the 118-element vocabulary")
        if torch.unique(self.species_by_class).numel() != self.species_by_class.numel():
            raise ValueError("distinct occupational classes require distinct element tokens")

    @classmethod
    def from_tokens(cls, tokens: torch.Tensor) -> "OccupationalPattern":
        if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.numel() < 1:
            raise ValueError("ordered element tokens must be a nonempty int64 vector")
        species, classes = torch.unique(tokens, sorted=True, return_inverse=True)
        return cls(classes, species)

    @property
    def tokens(self) -> torch.Tensor:
        return self.species_by_class.to(self.site_classes.device)[self.site_classes]

    @property
    def partition(self) -> torch.Tensor:
        """Element-name-independent site-equivalence matrix."""
        return self.site_classes[:, None] == self.site_classes[None, :]

    def stabilizer_indices(self, permutations: torch.Tensor) -> torch.Tensor:
        """Return exact parent-action elements preserving the ordered coloring."""
        nodes = self.site_classes.numel()
        if (
            permutations.ndim != 2
            or permutations.shape[1] != nodes
            or permutations.dtype != torch.long
        ):
            raise ValueError("occupational permutations must have shape [group,nodes]")
        ordered = torch.sort(permutations, dim=1).values
        expected = torch.arange(nodes, dtype=torch.long, device=permutations.device)
        if not torch.equal(ordered, expected.expand_as(ordered)):
            raise ValueError("occupational group action contains a non-permutation row")
        tokens = self.tokens.to(permutations.device)
        preserved = torch.all(tokens[permutations] == tokens.unsqueeze(0), dim=1)
        indices = torch.nonzero(preserved, as_tuple=False).flatten()
        if indices.numel() < 1:
            raise RuntimeError("occupational stabilizer lost the identity")
        return indices

    def stabilizer_is_subgroup(
        self,
        permutations: torch.Tensor,
        stabilizer_indices: torch.Tensor | None = None,
    ) -> bool:
        """Check identity and closure directly in the faithful node action."""
        selected = (
            self.stabilizer_indices(permutations)
            if stabilizer_indices is None
            else stabilizer_indices.to(device=permutations.device, dtype=torch.long)
        )
        nodes = permutations.shape[1]
        identity_rows = torch.all(
            permutations
            == torch.arange(nodes, dtype=torch.long, device=permutations.device),
            dim=1,
        )
        if not bool(identity_rows[selected].any()):
            return False
        lookup = {tuple(map(int, row)): index for index, row in enumerate(permutations.tolist())}
        selected_set = set(map(int, selected.tolist()))
        for left in selected_set:
            for right in selected_set:
                product = tuple(
                    map(int, permutations[left, permutations[right]].tolist())
                )
                if lookup.get(product) not in selected_set:
                    return False
        return True


def _integer_determinant(matrix: torch.Tensor) -> int:
    values = matrix.to(device="cpu", dtype=torch.int64).tolist()
    return int(
        values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )


def validate_supercell_hnf(matrix: torch.Tensor, *, maximum_index: int = 4) -> int:
    """Validate an upper- or lower-triangular 3D Hermite normal form."""
    if matrix.shape != (3, 3) or matrix.dtype not in (torch.int32, torch.int64):
        raise ValueError("supercell HNF must be a 3x3 integer tensor")
    upper = bool(torch.equal(matrix, torch.triu(matrix)))
    lower = bool(torch.equal(matrix, torch.tril(matrix)))
    if not (upper or lower):
        raise ValueError("supercell matrix must use an upper or lower HNF convention")
    diagonal = torch.diagonal(matrix)
    if bool((diagonal <= 0).any()):
        raise ValueError("supercell HNF diagonal must be positive")
    if upper:
        for column in range(1, 3):
            if any(not 0 <= int(matrix[row, column]) < int(matrix[column, column]) for row in range(column)):
                raise ValueError("upper-HNF off-diagonal entries are outside their canonical range")
    if lower:
        for row in range(1, 3):
            if any(not 0 <= int(matrix[row, column]) < int(matrix[row, row]) for column in range(row)):
                raise ValueError("lower-HNF off-diagonal entries are outside their canonical range")
    determinant = _integer_determinant(matrix)
    if not 1 <= determinant <= maximum_index:
        raise ValueError(f"supercell index must lie in 1..{maximum_index}")
    return determinant


@dataclass(frozen=True)
class OPDBranch:
    """Finite order-parameter-direction branch for one irrep."""

    label: str
    basis: torch.Tensor
    stabilizer_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("OPD branch label must be nonempty")
        if self.basis.ndim != 2 or self.basis.shape[1] < 1 or not torch.isfinite(self.basis).all():
            raise ValueError("OPD basis must be a finite [irrep_dim,opd_dim] matrix")
        gram = self.basis.transpose(0, 1) @ self.basis
        identity = torch.eye(self.basis.shape[1], dtype=self.basis.dtype, device=self.basis.device)
        if not torch.allclose(gram, identity, atol=2e-6, rtol=2e-6):
            raise ValueError("OPD basis columns must be orthonormal")
        if self.stabilizer_indices.ndim != 1 or self.stabilizer_indices.dtype != torch.long:
            raise ValueError("OPD stabilizer indices must be an int64 vector")
        if self.stabilizer_indices.numel() < 1 or bool((self.stabilizer_indices < 0).any()):
            raise ValueError("OPD stabilizer must contain at least the identity operation")
        if torch.unique(self.stabilizer_indices).numel() != self.stabilizer_indices.numel():
            raise ValueError("OPD stabilizer operation indices must be unique")


@dataclass(frozen=True)
class ModeCatalogEntry:
    """One commensurate parent-mode irrep with its finite OPD catalogue."""

    parent_space_group: int
    supercell_matrix: torch.Tensor
    wave_vector: torch.Tensor
    irrep_label: str
    mode_basis: torch.Tensor
    branches: tuple[OPDBranch, ...]

    def __post_init__(self) -> None:
        validate_supercell_hnf(self.supercell_matrix)
        if not 1 <= self.parent_space_group <= 230:
            raise ValueError("mode parent space group must lie in 1..230")
        if self.wave_vector.shape != (3,) or not torch.isfinite(self.wave_vector).all():
            raise ValueError("mode wave vector must be a finite three-vector")
        commensurate = self.supercell_matrix.to(self.wave_vector) @ self.wave_vector
        if not torch.allclose(commensurate, commensurate.round(), atol=1e-7, rtol=0.0):
            raise ValueError("mode wave vector is not commensurate with its supercell")
        if not self.irrep_label:
            raise ValueError("mode irrep label must be nonempty")
        if self.mode_basis.ndim != 2 or self.mode_basis.shape[1] < 1:
            raise ValueError("mode basis must have shape [3N,irrep_dim]")
        gram = self.mode_basis.transpose(0, 1) @ self.mode_basis
        identity = torch.eye(self.mode_basis.shape[1], dtype=self.mode_basis.dtype, device=self.mode_basis.device)
        if not torch.allclose(gram, identity, atol=2e-6, rtol=2e-6):
            raise ValueError("mass-weighted mode basis columns must be orthonormal")
        if not self.branches or any(branch.basis.shape[0] != self.mode_basis.shape[1] for branch in self.branches):
            raise ValueError("every OPD branch must act in the mode irrep space")
        labels = tuple(branch.label for branch in self.branches)
        if len(set(labels)) != len(labels):
            raise ValueError("OPD branch labels must be unique within a mode entry")

    def branch(self, label: str) -> OPDBranch:
        selected = tuple(branch for branch in self.branches if branch.label == label)
        if len(selected) != 1:
            raise KeyError(f"unknown OPD branch {label!r} for {self.irrep_label}")
        return selected[0]


@dataclass(frozen=True)
class ModeCatalog:
    """Immutable catalogue of group-theoretically qualified mode branches."""

    entries: tuple[ModeCatalogEntry, ...]
    source_version: str
    source_hash: str

    def __post_init__(self) -> None:
        if not self.entries or not self.source_version or not self.source_hash:
            raise ValueError("mode catalogue requires entries, source version and source hash")

    def candidates(self, parent_space_group: int, supercell_matrix: torch.Tensor) -> tuple[int, ...]:
        return tuple(
            index
            for index, entry in enumerate(self.entries)
            if entry.parent_space_group == parent_space_group
            and torch.equal(entry.supercell_matrix.cpu(), supercell_matrix.cpu())
        )


@dataclass(frozen=True)
class SelectedMode:
    """A sampled mode and OPD branch; only its reduced amplitude is diffused."""

    entry: ModeCatalogEntry
    opd_class: str
    active: bool

    @property
    def branch(self) -> OPDBranch:
        return self.entry.branch(self.opd_class)


def supercell_compatible_operation_indices(
    parent_fractional_rotations: torch.Tensor,
    supercell_matrix: torch.Tensor,
    *,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Return parent operations preserving the sampled superlattice.

    For row fractional coordinates the condition is
    ``B U^T B^{-1} in Z^{3x3}``.
    """
    if parent_fractional_rotations.ndim != 3 or parent_fractional_rotations.shape[-2:] != (3, 3):
        raise ValueError("parent fractional rotations must have shape [operations,3,3]")
    validate_supercell_hnf(supercell_matrix)
    matrix = supercell_matrix.to(parent_fractional_rotations)
    transformed = matrix.unsqueeze(0) @ parent_fractional_rotations.transpose(-1, -2) @ torch.linalg.inv(matrix)
    compatible = (transformed - transformed.round()).abs().amax(dim=(-2, -1)) <= tolerance
    indices = torch.nonzero(compatible, as_tuple=False).flatten()
    if indices.numel() < 1:
        raise ValueError("no parent operation preserves the sampled supercell")
    return indices


@dataclass(frozen=True)
class DistortionBlueprint:
    """Sampled low-index commensurate symmetry-breaking path."""

    supercell_matrix: torch.Tensor
    modes: tuple[SelectedMode, ...]

    def __post_init__(self) -> None:
        validate_supercell_hnf(self.supercell_matrix)
        if sum(mode.active for mode in self.modes) > 2:
            raise ValueError("v1 distortion blueprint permits at most two active modes")
        if any(not torch.equal(mode.entry.supercell_matrix.cpu(), self.supercell_matrix.cpu()) for mode in self.modes):
            raise ValueError("all selected modes must use the distortion supercell")
        if len({mode.entry.parent_space_group for mode in self.modes}) > 1:
            raise ValueError("all selected modes must belong to one parent space group")

    @classmethod
    def exact_parent(cls, *, device: torch.device | str = "cpu") -> "DistortionBlueprint":
        return cls(torch.eye(3, dtype=torch.long, device=device), ())

    def child_operation_indices(
        self,
        parent_fractional_rotations: torch.Tensor,
        *,
        occupational_stabilizer_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        retained = supercell_compatible_operation_indices(parent_fractional_rotations, self.supercell_matrix)
        for mode in self.modes:
            if not mode.active:
                continue
            branch_indices = mode.branch.stabilizer_indices.to(retained.device)
            if bool((branch_indices >= parent_fractional_rotations.shape[0]).any()):
                raise ValueError("OPD stabilizer index lies outside the parent operation catalogue")
            retained = retained[torch.isin(retained, branch_indices)]
        if occupational_stabilizer_indices is not None:
            occupational = occupational_stabilizer_indices.to(
                device=retained.device, dtype=torch.long
            )
            if occupational.ndim != 1 or occupational.numel() < 1:
                raise ValueError("occupational stabilizer must be a nonempty index vector")
            if bool(
                (
                    (occupational < 0)
                    | (occupational >= parent_fractional_rotations.shape[0])
                ).any()
            ):
                raise ValueError("occupational stabilizer index lies outside parent operations")
            retained = retained[torch.isin(retained, occupational)]
        if retained.numel() < 1:
            raise ValueError("distortion branch intersection removed the identity operation")
        return retained


@dataclass(frozen=True)
class ModeDiffusionState:
    """Continuous child variables after the discrete OPD branches are fixed."""

    mode_amplitudes: tuple[torch.Tensor, ...]
    child_strain: torch.Tensor
    residual_displacements: torch.Tensor

    def __post_init__(self) -> None:
        if self.child_strain.ndim != 1 or not torch.isfinite(self.child_strain).all():
            raise ValueError("child strain coordinates must be a finite vector")
        if self.residual_displacements.ndim != 2 or self.residual_displacements.shape[-1] != 3:
            raise ValueError("residual displacement state must have shape [supercell_nodes,3]")
        values = (*self.mode_amplitudes, self.residual_displacements)
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("mode diffusion state contains non-finite values")


class EmpiricalNodeCountPrior:
    """Categorical node-count prior fitted only to training split counts."""

    def __init__(self, support: torch.Tensor, probabilities: torch.Tensor) -> None:
        if support.ndim != 1 or probabilities.shape != support.shape or support.numel() < 1:
            raise ValueError("node-count support and probabilities must be equal nonempty vectors")
        if support.dtype != torch.long or bool((support < 1).any()):
            raise ValueError("node-count support must contain positive integers")
        if bool((probabilities < 0).any()) or not torch.isfinite(probabilities).all():
            raise ValueError("node-count probabilities must be finite and nonnegative")
        total = probabilities.sum()
        if float(total) <= 0.0:
            raise ValueError("node-count probabilities must have positive mass")
        self.support = support.detach().cpu()
        self.probabilities = (probabilities / total).detach().cpu()

    @classmethod
    def fit(cls, node_counts: torch.Tensor) -> "EmpiricalNodeCountPrior":
        counts = node_counts.detach().to(device="cpu", dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 1).any()):
            raise ValueError("training node counts must be a nonempty positive vector")
        support, frequency = torch.unique(counts, sorted=True, return_counts=True)
        return cls(support, frequency.to(torch.float64))

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("sample count must be positive")
        indices = torch.multinomial(self.probabilities, count, replacement=True, generator=generator)
        return self.support[indices].to(device=device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"support": self.support.clone(), "probabilities": self.probabilities.clone()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "EmpiricalNodeCountPrior":
        return cls(state["support"], state["probabilities"])
