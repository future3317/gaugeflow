import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import QuotientRolloutFlowMap


def test_d08_quotient_perturbation_has_registered_per_graph_size_and_contract_penalty_backpropagates():
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts" / "run_gate_p5_d0_8_contractive_quotient_flow_map_v1.py"))
    d06 = runner["D06"]
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = d06["build_repeated_endpoint"](4, device=torch.device("cpu"))
    source, residual = d06["fixed_sources"](matcher, batch, seed=95)
    state = d06["analytic_state"](source, residual, batch, torch.full((4,), 0.2))
    perturbation = runner["quotient_perturbation"](matcher, batch, rms=0.001, generator=torch.Generator().manual_seed(96))
    energies = runner["_per_graph_energy"](perturbation, batch).sqrt()
    assert torch.allclose(energies, torch.full_like(energies, 0.001), atol=1.0e-6)
    model = QuotientRolloutFlowMap(hidden_dim=32, layers=1, coordinate_rbf_dim=8)
    penalty, ratio = runner["contractive_penalty"](model, matcher, batch, state, torch.full((4,), 0.2), torch.full((4,), 0.5), perturbation, lipschitz_bound=1.0, epsilon=1.0e-12)
    penalty.backward()
    assert torch.isfinite(penalty)
    assert torch.isfinite(ratio).all()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
