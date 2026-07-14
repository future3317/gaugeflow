import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.uncertainty import cartesian_isotropic_gaussian_nll
from torch_geometric.data import Batch, Data


def _batch() -> Batch:
    return Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])


def test_cartesian_isotropic_uncertainty_is_rotation_invariant():
    residual = torch.randn(7, 3)
    log_std = torch.randn(7, 1).clamp(-1.0, 1.0)
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    original = cartesian_isotropic_gaussian_nll(residual, log_std)
    rotated = cartesian_isotropic_gaussian_nll(residual @ rotation.T, log_std)
    assert torch.allclose(original, rotated, atol=1e-6)


def test_flow_uncertainty_objective_and_propagation_are_finite():
    batch = _batch()
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3)
    matcher = RiemannianCrystalFlowMatcher(uncertainty_weight=0.1)
    terms = matcher.loss(model, batch)
    assert torch.isfinite(terms["loss"])
    assert torch.isfinite(terms["uncertainty"])
    terms["loss"].backward()
    state, uncertainty = matcher.sample(model, batch, steps=2, return_uncertainty=True)
    assert torch.isfinite(state.frac_coords).all()
    assert torch.isfinite(uncertainty.coordinate_cartesian_variance).all()
    assert (uncertainty.coordinate_cartesian_variance >= 0).all()
