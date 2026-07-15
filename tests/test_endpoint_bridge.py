import torch
from torch_geometric.data import Batch, Data
import runpy
from pathlib import Path

from gaugeflow.coupling import translation_aligned_torus_rms
from gaugeflow.flow import CrystalFlowState, EndpointBridgeCoordinateMatcher
from gaugeflow.manifold import torus_logmap, wrap01


def test_endpoint_bridge_exact_residual_sampler_closes_without_terminal_velocity_conflict():
    batch = Batch.from_data_list([
        Data(
            atom_types=torch.tensor([5, 7, 14]),
            frac_coords=torch.tensor([[0.07, 0.11, 0.19], [0.34, 0.22, 0.31], [0.72, 0.48, 0.41]]),
            lattice=torch.eye(3).unsqueeze(0), num_nodes=3,
        )
    ])
    matcher = EndpointBridgeCoordinateMatcher()
    target = matcher.target_state(batch)

    class ExactResidual(torch.nn.Module):
        conditioning_mode = "unconditional"

        def forward(self, type_state, frac_coords, lattice_log, graph_batch, time, **kwargs):
            del lattice_log, time, kwargs
            residual = torus_logmap(frac_coords, target.frac_coords)
            residual = residual - residual.mean(dim=0, keepdim=True)
            return torch.zeros_like(type_state), residual, torch.zeros((1, 6)), torch.ones((1, 1))

    source = CrystalFlowState(
        target.type_state,
        wrap01(target.frac_coords + torch.tensor([[0.31, -0.22, 0.17], [-0.14, 0.08, 0.26], [0.21, 0.19, -0.35]])),
        target.lattice_log,
    )
    sampled = matcher.sample(ExactResidual(), batch, steps=16, initial_state=source)
    assert translation_aligned_torus_rms(sampled.frac_coords, target.frac_coords) < 2e-6


def test_d0_5_endpoint_bridge_runner_uses_a_finite_residual_target():
    runner = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_5_endpoint_bridge_metric_v1.py")
    )
    matcher = EndpointBridgeCoordinateMatcher()
    batch = runner["build_repeated_endpoint"](2, device=torch.device("cpu"))
    source, source_residual = runner["fixed_sources"](matcher, batch, seed=71)
    terminal = runner["interpolant"](source, source_residual, batch, torch.ones(2))
    terminal_residual = matcher.endpoint_residual(terminal, batch)
    assert torch.allclose(terminal_residual, torch.zeros_like(terminal_residual), atol=2e-6)
