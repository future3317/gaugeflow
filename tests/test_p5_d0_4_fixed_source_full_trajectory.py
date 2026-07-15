import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_4_fixed_source_full_trajectory_v1.py")
    )


def test_d0_4_uses_fixed_sources_with_resampled_time_and_finite_metric_gradients():
    runner = _runner()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = runner["build_repeated_endpoint"](3, device=torch.device("cpu"))
    source, velocity = runner["fixed_sources"](matcher, batch, seed=51)
    generator = torch.Generator().manual_seed(52)
    first_time, second_time = torch.rand(3, generator=generator), torch.rand(3, generator=generator)
    assert not torch.equal(first_time, second_time)
    assert torch.equal(source.frac_coords, runner["fixed_sources"](matcher, batch, seed=51)[0].frac_coords)
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="unconditional", coordinate_rbf_dim=8)
    velocity_mse, aligned_rms = runner["metrics_at_time"](
        model, matcher, batch, source, velocity, first_time
    )
    velocity_mse.backward()
    assert torch.isfinite(velocity_mse) and torch.isfinite(aligned_rms)
    assert model.layers[0].vector_gates[0].weight.grad[:, -8:].abs().sum() > 0
