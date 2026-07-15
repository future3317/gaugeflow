import itertools
import runpy
from pathlib import Path

import torch

from gaugeflow.coupling import _type_preserving_assignments, fixed_lift_coupling
from gaugeflow.geometry import periodic_closest_image_edges

def test_p5_c0_exact_lift_recovers_translation_and_unique_type_assignment():
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts" / "run_gate_p5_c0_branch_audit_v1.py"))
    source = torch.tensor([[[0.10, 0.20, 0.30], [0.35, 0.45, 0.55], [0.70, 0.15, 0.80], [0.90, 0.65, 0.05]]])
    target = torch.remainder(source + torch.tensor([0.31, -0.27, 0.22]), 1.0)
    lattice = torch.eye(3).unsqueeze(0)
    types = torch.tensor([5, 7, 14, 32])
    solved = runner["solve_coupling"](source, target, lattice, types, types)
    assert solved["permutation_count"] == 1
    assert torch.equal(solved["assignment"], torch.arange(4).unsqueeze(0))
    assert solved["cost"].item() < 1.0e-10
    assert torch.isinf(solved["second_permutation_cost"]).all()


def test_p5_c0_equal_species_enumerates_both_assignments_without_duplicates():
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts" / "run_gate_p5_c0_branch_audit_v1.py"))
    assignments = _type_preserving_assignments(torch.tensor([5, 5]), torch.tensor([5, 5]))
    assert len(assignments) == 2
    assert {tuple(value.tolist()) for value in assignments} == {(0, 1), (1, 0)}


def test_fixed_lift_coupling_has_zero_drift_and_matches_endpoint_once_wrapped():
    source = torch.tensor([[0.10, 0.20, 0.30], [0.35, 0.45, 0.55], [0.70, 0.15, 0.80], [0.90, 0.65, 0.05]])
    target = torch.remainder(source + torch.tensor([0.31, -0.27, 0.22]), 1.0)
    types = torch.tensor([5, 7, 14, 32])
    coupling = fixed_lift_coupling(source, target, torch.eye(3), source_types=types, target_types=types)
    assert coupling.cost < 1.0e-10
    assert torch.allclose(coupling.velocity.mean(dim=0), torch.zeros(3), atol=1.0e-7)
    terminal = torch.remainder(coupling.endpoint_lift - coupling.translation, 1.0)
    assert torch.allclose(terminal, target[coupling.assignment], atol=1.0e-7)


def test_fixed_lift_bound_matches_brute_force_for_skew_lattice():
    source = torch.tensor([[0.93, 0.08, 0.71], [0.04, 0.87, 0.19]])
    target = torch.tensor([[0.11, 0.92, 0.16], [0.78, 0.13, 0.84]])
    lattice = torch.tensor([[3.8, 0.7, 0.2], [0.1, 4.2, 0.8], [0.4, 0.2, 5.0]])
    types = torch.tensor([5, 7])
    coupling = fixed_lift_coupling(source, target, lattice, source_types=types, target_types=types)
    costs = []
    difference = source - target
    for values in itertools.product(range(-3, 4), repeat=3):
        lift = torch.tensor([[0, 0, 0], values])
        unaligned = difference - lift
        residual = unaligned - unaligned.mean(dim=0, keepdim=True)
        costs.append(float((residual @ lattice).square().sum()))
    expected = sorted(costs)
    assert torch.allclose(coupling.cost, torch.tensor(expected[0]), atol=1.0e-6)
    assert torch.allclose(coupling.second_cost, torch.tensor(expected[1]), atol=1.0e-6)


def test_periodic_geometry_is_invariant_to_universal_cover_integer_lifts():
    coords = torch.tensor([[0.05, 0.10, 0.15], [0.82, 0.76, 0.61], [0.35, 0.45, 0.55]])
    lifted = coords + torch.tensor([[2.0, -1.0, 3.0], [-2.0, 2.0, -1.0], [1.0, 3.0, -2.0]])
    batch = torch.zeros(3, dtype=torch.long)
    lattice = torch.tensor([[[3.9, 0.2, 0.1], [0.3, 4.3, 0.4], [0.1, 0.4, 5.1]]])
    base = periodic_closest_image_edges(coords, lattice, batch)
    universal = periodic_closest_image_edges(lifted, lattice, batch)
    assert torch.allclose(base.displacement, universal.displacement, atol=1.0e-6)
    assert torch.allclose(base.distance, universal.distance, atol=1.0e-6)
