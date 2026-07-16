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
            ReachableChildPath(0, 2, "exact_parent"),
            ReachableChildPath(0, 1, "inversion_odd_child"),
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
            ReachableChildPath(0, 1, "branch_a"),
            ReachableChildPath(0, 1, "branch_b"),
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
    assert output.path_given_parent_log_probability[0, 1] > -1e-3
    assert torch.allclose(torch.logsumexp(output.path_joint_log_probability, dim=-1), torch.zeros(1))


def test_reachable_child_router_handles_dead_parent_and_fails_closed_globally():
    router = ReachableChildCompatibilityRouter(
        [2, 1],
        [
            ReachableChildPath(0, 2, "centrosymmetric_exact"),
            ReachableChildPath(1, 1, "polar_child"),
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
        [ReachableChildPath(0, 2, "centrosymmetric_exact")],
        hidden_dim=8,
        rotation_count=12,
    )
    with pytest.raises(ValueError, match="no tensor-compatible reachable child"):
        blocked.route_from_logits(
            condition,
            parent_prior_logits=torch.zeros((1, 1)),
            path_prior_logits=torch.zeros((1, 1)),
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
