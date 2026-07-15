import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_3_translation_quotient_metric_v1.py")
    )


def test_d0_3_fixed_examples_are_no_drift_and_metric_backbone_is_differentiable():
    runner = _runner()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = runner["build_repeated_endpoint"](3, device=torch.device("cpu"))
    state, target_velocity, time = runner["fixed_examples"](
        matcher, batch, source_noise_seed=41, time_seed=42
    )
    assert torch.allclose(target_velocity.reshape(3, 4, 3).mean(dim=1), torch.zeros(3, 3), atol=1e-7)
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="unconditional", coordinate_rbf_dim=8)
    mse, aligned_rms, absolute_rms = runner["fixed_batch_metrics"](
        model, matcher, batch, state, target_velocity, time
    )
    mse.backward()
    assert torch.isfinite(mse) and torch.isfinite(aligned_rms) and torch.isfinite(absolute_rms)
    assert model.layers[0].vector_gates[0].weight.grad[:, -8:].abs().sum() > 0
