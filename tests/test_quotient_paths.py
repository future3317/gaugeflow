import torch
from torch_geometric.data import Batch, Data

from gaugeflow.coupling import (
    periodic_assignment,
    periodic_assignment_cost,
    remove_graphwise_translation,
    translation_aligned_torus_rms,
)
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.manifold import project_simplex, simplex_tangent
from gaugeflow.model import GaugeFlowVectorField


def test_exact_simplex_projection_and_tangent_constraint():
    value = torch.tensor([[2.0, -1.0, 0.3], [-2.0, -3.0, -4.0]])
    projected = project_simplex(value)
    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(-1), torch.ones(2))
    tangent = simplex_tangent(torch.randn(4, 7))
    assert torch.allclose(tangent.sum(-1), torch.zeros(4), atol=1e-6)


def test_periodic_transport_coupling_is_low_cost_and_permutation_consistent():
    source = torch.tensor([[0.10, 0.1, 0.1], [0.48, 0.1, 0.1], [0.90, 0.1, 0.1]])
    target = torch.tensor([[0.91, 0.1, 0.1], [0.09, 0.1, 0.1], [0.49, 0.1, 0.1]])
    assignment = periodic_assignment(source, target)
    identity_cost = periodic_assignment_cost(source, target, torch.arange(3))
    optimal_cost = periodic_assignment_cost(source, target, assignment)
    assert optimal_cost <= identity_cost
    permutation = torch.tensor([2, 0, 1])
    permuted_assignment = periodic_assignment(source[permutation], target)
    assert torch.allclose(target[permuted_assignment], target[assignment][permutation])


def test_no_drift_quotient_removes_global_translation_but_preserves_relative_error():
    velocity = torch.tensor([[0.3, 0.1, -0.2], [0.1, -0.2, 0.4], [0.8, 0.1, 0.2], [0.2, 0.4, -0.1]])
    batch = torch.tensor([0, 0, 1, 1])
    projected = remove_graphwise_translation(velocity, batch, 2)
    assert torch.allclose(projected[:2].mean(0), torch.zeros(3), atol=1e-7)
    assert torch.allclose(projected[2:].mean(0), torch.zeros(3), atol=1e-7)
    target = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.3, 0.4]])
    translated = torch.remainder(target + torch.tensor([0.25, 0.5, 0.75]), 1.0)
    assert translation_aligned_torus_rms(translated, target) < 1e-7


def test_riemannian_simplex_endpoint_objective_and_quotient_coupling_are_finite():
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.2, 0.3]]),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[1.0, 0.0]]),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.tensor([[0.2, 0.3, 0.4], [0.8, 0.3, 0.4]]),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[0.0, 1.0]]),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="endpoint_id")
    matcher = RiemannianCrystalFlowMatcher(
        active_heads=("type", "coord", "lattice"), type_path="riemannian_simplex",
        target_coupling="optimal_transport",
        loss_normalization="target_velocity_rms", endpoint_type_nll_weight=1.0,
    )
    terms = matcher.loss(model, batch)
    assert torch.isfinite(terms["loss"])
    assert torch.isfinite(terms["endpoint_type_nll"])
    terms["loss"].backward()
    initial = matcher.random_state(batch)
    sampled = matcher.sample(model, batch, steps=2, initial_state=initial)
    assert torch.all(sampled.type_state >= 0)
    assert torch.allclose(sampled.type_state.sum(-1), torch.ones(sampled.type_state.shape[0]), atol=1e-6)
