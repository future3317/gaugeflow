import pytest
import torch

from gaugeflow.production.blueprint import (
    DistortionBlueprint,
    ModeCatalogEntry,
    ModeDiffusionState,
    OPDBranch,
    ParentBlueprint,
    SelectedMode,
    supercell_compatible_operation_indices,
)
from gaugeflow.production.child_reconstruction import (
    ChildReconstructor,
    ParentCrystal,
    expand_parent_supercell,
    supercell_coset_translations,
)
from gaugeflow.production.mode_supervision import (
    eigenspace_projector,
    generalized_mode_force,
    mode_effective_charge,
    phonon_targets,
    subspace_projector_loss,
)
from gaugeflow.production.space_group_router import (
    ReachableChildCompatibilityRouter,
    ReachableChildPath,
)
from gaugeflow.production.terminal_symmetry_audit import (
    audit_terminal_symmetry,
    classify_group_relation,
    detect_cartesian_point_group,
)


def _identity_representations(operations: int, nodes: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(3 * nodes, dtype=dtype).expand(operations, -1, -1).clone()


def _parent() -> ParentCrystal:
    return ParentCrystal(
        species=torch.tensor([5, 7], dtype=torch.long),
        fractional_coordinates=torch.tensor([[0.15, 0.2, 0.3], [0.65, 0.7, 0.8]], dtype=torch.float64),
        lattice=torch.diag(torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)),
        masses=torch.tensor([10.81, 14.01], dtype=torch.float64),
    )


def test_parent_blueprint_is_not_a_child_space_group_claim():
    blueprint = ParentBlueprint(221, ("1a", "1b"), (1, 1), (5, 7))
    assert blueprint.parent_space_group == 221
    assert blueprint.atom_count == 2


def test_exact_distortion_branch_reconstructs_the_parent_exactly():
    parent = _parent()
    blueprint = DistortionBlueprint.exact_parent()
    state = ModeDiffusionState((), torch.zeros(0, dtype=torch.float64), torch.zeros((2, 3), dtype=torch.float64))
    child = ChildReconstructor().reconstruct(
        parent,
        blueprint,
        state,
        parent_fractional_rotations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        parent_cartesian_operations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        displacement_representations=_identity_representations(1, 2, torch.float64),
        invariant_strain_basis=torch.empty((0, 3, 3), dtype=torch.float64),
    )
    assert torch.equal(child.species, parent.species)
    assert torch.allclose(child.fractional_coordinates, parent.fractional_coordinates)
    assert torch.allclose(child.lattice, parent.lattice)
    assert child.child_operation_indices.tolist() == [0]


def test_low_index_commensurate_mode_creates_a_nontrivial_child():
    parent = ParentCrystal(
        species=torch.tensor([8]),
        fractional_coordinates=torch.zeros((1, 3), dtype=torch.float64),
        lattice=3.0 * torch.eye(3, dtype=torch.float64),
        masses=torch.ones(1, dtype=torch.float64),
    )
    supercell = torch.diag(torch.tensor([2, 1, 1], dtype=torch.long))
    mode_basis = torch.tensor([[1.0], [0.0], [0.0], [-1.0], [0.0], [0.0]], dtype=torch.float64)
    mode_basis = mode_basis / torch.linalg.vector_norm(mode_basis)
    branch = OPDBranch("a", torch.ones((1, 1), dtype=torch.float64), torch.tensor([0]))
    entry = ModeCatalogEntry(
        parent_space_group=1,
        supercell_matrix=supercell,
        wave_vector=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
        irrep_label="X1",
        mode_basis=mode_basis,
        branches=(branch,),
    )
    blueprint = DistortionBlueprint(supercell, (SelectedMode(entry, "a", True),))
    state = ModeDiffusionState(
        (torch.tensor([0.1], dtype=torch.float64),),
        torch.zeros(0, dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    )
    child = ChildReconstructor().reconstruct(
        parent,
        blueprint,
        state,
        parent_fractional_rotations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        parent_cartesian_operations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        displacement_representations=_identity_representations(1, 2, torch.float64),
        invariant_strain_basis=torch.empty((0, 3, 3), dtype=torch.float64),
    )
    assert child.species.tolist() == [8, 8]
    assert child.mode_displacement[:, 0].tolist() == pytest.approx([2**-0.5 * 0.1, -(2**-0.5) * 0.1])
    assert not torch.allclose(
        child.fractional_coordinates,
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
    )


def test_supercell_cosets_and_compatible_parent_operations_are_exact():
    supercell = torch.tensor([[2, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=torch.long)
    translations = supercell_coset_translations(supercell)
    assert translations.shape == (4, 3)
    parent = _parent()
    species, _, coordinates, lattice, _ = expand_parent_supercell(parent, supercell)
    assert species.numel() == 8 and coordinates.shape == (8, 3)
    assert torch.allclose(torch.linalg.det(lattice), 4.0 * torch.linalg.det(parent.lattice))
    rotations = torch.stack((torch.eye(3), torch.diag(torch.tensor([-1.0, -1.0, 1.0]))))
    compatible = supercell_compatible_operation_indices(rotations, supercell)
    assert 0 in compatible.tolist()


def test_residual_is_projected_and_budget_is_fail_closed():
    parent = _parent()
    blueprint = DistortionBlueprint.exact_parent()
    residual = torch.tensor([[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]], dtype=torch.float64)
    state = ModeDiffusionState((), torch.zeros(0, dtype=torch.float64), residual)
    child = ChildReconstructor(residual_rms_limit_angstrom=0.05).reconstruct(
        parent,
        blueprint,
        state,
        parent_fractional_rotations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        parent_cartesian_operations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        displacement_representations=_identity_representations(1, 2, torch.float64),
        invariant_strain_basis=torch.empty((0, 3, 3), dtype=torch.float64),
    )
    assert torch.allclose(
        (parent.masses.unsqueeze(-1) * child.residual_displacement).sum(0),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-12,
    )
    too_large = ModeDiffusionState((), torch.zeros(0, dtype=torch.float64), 10.0 * residual)
    with pytest.raises(ValueError, match="residual exceeds"):
        ChildReconstructor(residual_rms_limit_angstrom=0.05).reconstruct(
            parent,
            blueprint,
            too_large,
            parent_fractional_rotations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
            parent_cartesian_operations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
            displacement_representations=_identity_representations(1, 2, torch.float64),
            invariant_strain_basis=torch.empty((0, 3, 3), dtype=torch.float64),
        )


def test_reachable_child_router_does_not_reject_a_centrosymmetric_parent():
    torch.manual_seed(901)
    router = ReachableChildCompatibilityRouter(
        [2],
        [
            ReachableChildPath(0, 2, "exact", "exact_parent", 0.25, True),
            ReachableChildPath(0, 1, "inversion_odd", "inversion_odd_child", 0.75),
        ],
        hidden_dim=8,
        rotation_count=12,
    )
    output = router(torch.randn((2, 18)))
    assert torch.equal(output.parent_log_probability, torch.zeros_like(output.parent_log_probability))
    assert torch.isneginf(output.path_given_parent_log_probability[:, 0]).all()
    assert torch.allclose(output.path_given_parent_log_probability[:, 1], torch.zeros(2))
    assert torch.isfinite(output.path_joint_log_probability[:, 1]).all()


def test_reachable_child_router_accepts_parent_geometry_dependent_path_priors():
    router = ReachableChildCompatibilityRouter(
        [1],
        [
            ReachableChildPath(0, 1, "exact", "branch_a", 0.25, True),
            ReachableChildPath(0, 1, "distorted", "branch_b", 0.75),
        ],
        hidden_dim=8,
        rotation_count=12,
    )
    condition = torch.zeros((1, 18))
    output = router.route_from_logits(
        condition,
        parent_prior_logits=torch.zeros((1, 1)),
        path_prior_logits=torch.tensor([[-4.0, 4.0]]),
    )
    assert output.path_given_parent_log_probability[0, 1] > -2e-3
    assert torch.allclose(torch.logsumexp(output.path_joint_log_probability, dim=-1), torch.zeros(1))


def test_reachable_child_router_handles_dead_parent_and_fails_closed_globally():
    router = ReachableChildCompatibilityRouter(
        [2, 1],
        [
            ReachableChildPath(0, 2, "exact", "centrosymmetric_exact", 1.0, True),
            ReachableChildPath(1, 1, "exact", "polar_child", 1.0, True),
        ],
        hidden_dim=8,
        rotation_count=12,
    )
    condition = torch.randn((1, 18))
    output = router.route_from_logits(
        condition,
        parent_prior_logits=torch.zeros((1, 2)),
        path_prior_logits=torch.zeros((1, 2)),
    )
    assert torch.isneginf(output.parent_log_probability[0, 0])
    assert torch.allclose(output.parent_log_probability[0, 1], torch.tensor(0.0))
    assert not torch.isnan(output.path_joint_log_probability).any()

    blocked = ReachableChildCompatibilityRouter(
        [2],
        [ReachableChildPath(0, 2, "exact", "centrosymmetric_exact", 1.0, True)],
        hidden_dim=8,
        rotation_count=12,
    )
    with pytest.raises(ValueError, match="no tensor-compatible reachable child"):
        blocked.route_from_logits(
            condition,
            parent_prior_logits=torch.zeros((1, 1)),
            path_prior_logits=torch.zeros((1, 1)),
        )


def _measure_router(paths: list[ReachableChildPath]) -> ReachableChildCompatibilityRouter:
    return ReachableChildCompatibilityRouter(
        [1, 2],
        paths,
        hidden_dim=8,
        rotation_count=12,
    )


def test_reachable_catalogue_deduplicates_equivalent_representations():
    unique = [
        ReachableChildPath(0, 1, "exact", "exact", 0.4, True),
        ReachableChildPath(0, 1, "polar", "polar_a", 0.6),
        ReachableChildPath(1, 2, "exact", "exact", 1.0, True),
    ]
    expanded = [
        unique[2],
        ReachableChildPath(0, 1, "polar", "polar_domain_relabel", 0.6),
        unique[0],
        unique[1],
    ]
    base = _measure_router(unique)
    duplicate = _measure_router(expanded)
    assert base.path_equivalence_classes == duplicate.path_equivalence_classes
    assert duplicate.catalogue_representation_multiplicity.tolist() == [1, 2, 1]
    condition = torch.zeros((2, 18))
    parent_logits = torch.tensor([[0.2, -0.4], [-0.3, 0.7]])
    path_logits = torch.tensor([[0.5, -0.2, 0.1], [-0.4, 0.3, 0.8]])
    base_output = base.route_from_logits(condition, parent_logits, path_logits)
    duplicate_output = duplicate.route_from_logits(condition, parent_logits, path_logits)
    assert torch.allclose(
        base_output.parent_log_probability,
        duplicate_output.parent_log_probability,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        base_output.path_joint_log_probability,
        duplicate_output.path_joint_log_probability,
        atol=1e-12,
        rtol=1e-12,
    )


def test_reachable_catalogue_reorder_and_parent_size_are_prior_invariant():
    paths = [
        ReachableChildPath(0, 1, "exact", "exact", 1.0, True),
        ReachableChildPath(1, 2, "exact", "exact", 0.25, True),
        ReachableChildPath(1, 1, "odd", "odd", 0.75),
    ]
    router = _measure_router(list(reversed(paths)))
    assert router.path_equivalence_classes == ("exact", "exact", "odd")
    output = router.route_from_logits(
        torch.zeros((1, 18)),
        parent_prior_logits=torch.zeros((1, 2)),
        path_prior_logits=torch.zeros((1, 3)),
    )
    assert torch.allclose(
        output.parent_log_probability,
        torch.full((1, 2), -torch.log(torch.tensor(2.0))),
        atol=1e-6,
    )


def test_reachable_catalogue_rejects_inconsistent_equivalence_and_missing_exact():
    with pytest.raises(ValueError, match="disagree on physical metadata"):
        _measure_router(
            [
                ReachableChildPath(0, 1, "exact", "exact", 1.0, True),
                ReachableChildPath(0, 1, "same", "a", 0.5),
                ReachableChildPath(0, 2, "same", "b", 0.5),
                ReachableChildPath(1, 2, "exact", "exact", 1.0, True),
            ]
        )
    with pytest.raises(ValueError, match="exactly one explicit exact"):
        ReachableChildCompatibilityRouter(
            [1],
            [ReachableChildPath(0, 1, "distorted", "distorted", 1.0)],
            hidden_dim=8,
            rotation_count=12,
        )


def test_phonon_and_mode_targets_are_basis_gauge_safe():
    omega_squared = torch.tensor([-4.0, 9.0])
    targets = phonon_targets(omega_squared, omega0_squared=4.0)
    assert targets.soft.tolist() == [1.0, 0.0]
    basis = torch.linalg.qr(torch.randn((6, 2), dtype=torch.float64)).Q
    gauge = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    transformed = basis @ gauge
    assert torch.allclose(eigenspace_projector(basis), eigenspace_projector(transformed), atol=1e-12)
    assert subspace_projector_loss(basis, transformed) < 1e-20

    complex_basis = torch.tensor(
        [[1.0, 1.0j], [1.0j, 1.0], [0.0, 0.0]], dtype=torch.complex128
    ) / 2**0.5
    unitary = torch.tensor(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=torch.complex128
    ) / 2**0.5
    rotated_complex_basis = complex_basis @ unitary
    projector = eigenspace_projector(complex_basis)
    assert torch.allclose(projector, projector.mH, atol=1e-12)
    assert torch.allclose(projector, eigenspace_projector(rotated_complex_basis), atol=1e-12)
    assert subspace_projector_loss(complex_basis, rotated_complex_basis) < 1e-20


def test_mode_effective_charge_and_generalized_force_match_definitions():
    born = torch.zeros((1, 3, 3), dtype=torch.float64)
    born[0] = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    modes = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float64)
    charge = mode_effective_charge(born, modes, torch.tensor([4.0], dtype=torch.float64))
    assert torch.allclose(charge, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    force = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    mode_basis = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    generalized = generalized_mode_force(force, torch.tensor([4.0]), mode_basis, torch.ones((1, 1)))
    assert torch.allclose(generalized, torch.tensor([1.0], dtype=torch.float64))


def test_terminal_symmetry_audit_detects_accidental_restoration():
    species = torch.tensor([14], dtype=torch.long)
    coordinates = torch.zeros((1, 3), dtype=torch.float64)
    lattice = 4.0 * torch.eye(3, dtype=torch.float64)
    detected = detect_cartesian_point_group(species, coordinates, lattice)
    assert detected.space_group_number == 221
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    assert classify_group_relation(identity, detected.cartesian_operations) == "symmetry_restoration"
    audit = audit_terminal_symmetry(
        piezo_irreps=torch.randn(18, dtype=torch.float64),
        declared_space_group=1,
        declared_cartesian_operations=identity,
        raw_species=species,
        raw_fractional_coordinates=coordinates,
        raw_lattice=lattice,
        relaxed_species=species,
        relaxed_fractional_coordinates=coordinates,
        relaxed_lattice=lattice,
        rotation_count=12,
    )
    assert audit.declared_to_raw_relation == "symmetry_restoration"
    assert audit.raw_to_relaxed_relation == "equal"
    assert audit.raw_compatibility_residual > 0.9
    assert audit.compatibility_retained_after_relaxation is False
