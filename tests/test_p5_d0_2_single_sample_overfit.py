import runpy
from pathlib import Path

import torch

from gaugeflow.model import GaugeFlowVectorField


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_2_single_sample_overfit_v1.py")
    )


def test_d0_2_selects_exactly_the_first_d0_1_fixed_example():
    runner = _runner()
    batch, state, velocity, time, audit = runner["selected_example"](device=torch.device("cpu"))
    parent = runner["_d0_1_runner"]()
    matcher = parent["RiemannianCrystalFlowMatcher"](active_heads=("coord",))
    batch64 = parent["build_repeated_endpoint"](64, device=torch.device("cpu"))
    state64, velocity64, time64 = parent["fixed_examples"](
        matcher, batch64, source_noise_seed=520101, time_seed=520102
    )
    assert batch.num_graphs == 1
    assert torch.equal(state.frac_coords, state64.frac_coords[:4])
    assert torch.equal(velocity, velocity64[:4])
    assert torch.equal(time, time64[:1])
    assert audit["selected_index"] == 0


def test_d0_2_coordinate_head_has_a_finite_gradient_on_the_selected_example():
    runner = _runner()
    batch, state, velocity, time, _ = runner["selected_example"](device=torch.device("cpu"))
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="unconditional")
    mse, rms, prediction = runner["_metrics"](model, batch, state, velocity, time)
    mse.backward()
    assert torch.isfinite(mse) and torch.isfinite(rms) and torch.isfinite(prediction).all()
    assert model.coord_out.weight.grad is not None
    assert torch.isfinite(model.coord_out.weight.grad).all()
