import runpy
from pathlib import Path

import pytest
import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_d0_unconditional_coordinate_v1.py")
    )


def test_p5_d0_panel_reuses_p5_endpoint_without_any_condition_fields():
    runner = _runner()
    batch, target, species_scalar = runner["build_panel"](device=torch.device("cpu"))
    assert batch.num_graphs == 1
    assert batch.atom_types.tolist() == [5, 7, 14, 32]
    assert batch.frac_coords.shape == (4, 3)
    assert target.shape == (3, 3, 3)
    assert species_scalar.shape == (4,)
    for forbidden in ("piezo_irreps", "condition_present", "condition_orbit", "endpoint_id"):
        assert not hasattr(batch, forbidden)


def test_unconditional_coordinate_backbone_receives_no_condition_inputs_and_samples():
    runner = _runner()
    batch, _, _ = runner["build_panel"](device=torch.device("cpu"))
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, conditioning_mode="unconditional")
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    terms = matcher.loss(model, batch)
    assert torch.isfinite(terms["loss"])
    state = matcher.random_state(batch)
    output = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, torch.tensor([0.5]))
    assert len(output) == 4
    assert all(torch.isfinite(value).all() for value in output[:3])
    with pytest.raises(ValueError, match="must not receive"):
        model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, torch.tensor([0.5]),
            torch.zeros(1, 18), torch.ones(1, 1, dtype=torch.bool),
        )
    sampled = matcher.sample(model, batch, steps=2, guidance_scale=0.0)
    assert torch.isfinite(sampled.frac_coords).all()


def test_p5_d0_endpoint_and_teacher_match_the_frozen_p5_definition():
    d0 = _runner()
    p5 = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_exact_synthetic_tensor_control.py")
    )
    batch, target, species_scalar = d0["build_panel"](device=torch.device("cpu"))
    p5_specs, p5_scalar = p5["_endpoint_specifications"](torch.float32)
    assert torch.equal(batch.frac_coords, p5_specs[0]["frac"])
    assert torch.equal(batch.lattice[0], p5_specs[0]["lattice"])
    assert torch.equal(batch.atom_types, p5_specs[0]["atom_types"])
    assert torch.equal(species_scalar, p5_scalar)
    assert torch.allclose(target, p5["_teacher_tensor"](p5_specs[0]["frac"], p5_specs[0]["lattice"], p5_scalar))
