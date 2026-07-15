import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_1_fixed_batch_overfit_v1.py")
    )


def test_fixed_examples_are_reproducible_and_do_not_require_a_condition():
    runner = _runner()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    first = runner["build_repeated_endpoint"](64, device=torch.device("cpu"))
    second = runner["build_repeated_endpoint"](64, device=torch.device("cpu"))
    first_values = runner["fixed_examples"](matcher, first, source_noise_seed=31, time_seed=32)
    second_values = runner["fixed_examples"](matcher, second, source_noise_seed=31, time_seed=32)
    for left, right in zip(first_values, second_values):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            assert torch.equal(left.frac_coords, right.frac_coords)
    assert first.num_graphs == 64
    for forbidden in ("piezo_irreps", "condition_present", "endpoint_id"):
        assert not hasattr(first, forbidden)


def test_fixed_batch_metrics_are_differentiable_and_free_of_resampling():
    runner = _runner()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = runner["build_repeated_endpoint"](4, device=torch.device("cpu"))
    state, target_velocity, time = runner["fixed_examples"](matcher, batch, source_noise_seed=8, time_seed=9)
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="unconditional")
    mse, rms = runner["fixed_batch_metrics"](model, batch, state, target_velocity, time)
    mse.backward()
    assert torch.isfinite(mse) and torch.isfinite(rms)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert runner["_classify"](False, 0.0, 0.0, {"unseen_teacher_forced_endpoint_rms_max": 0.02, "free_running_endpoint_rms_max": 0.02}) == "model_or_loss_cannot_memorize"
