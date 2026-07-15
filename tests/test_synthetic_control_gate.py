import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.harmonic import deterministic_so3_grid


def _runner():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_gate_p5_exact_synthetic_tensor_control.py")
    )


def test_p5_synthetic_panel_has_distinct_exact_orbits_and_common_noise_contract():
    runner = _runner()
    batch, targets, frames = runner["build_panel"](4, device=torch.device("cpu"))
    assert batch.num_graphs == 8
    assert batch.atom_types.numel() == 32
    distances = runner["_orbit_distances"](
        targets, targets, deterministic_so3_grid(60, dtype=targets.dtype)
    )
    assert float(distances[0, 1]) > 0.05
    equivariance = runner["_teacher_equivariance"](
        targets, frames, batch.teacher_scalar[batch.batch == 0]
    )
    assert equivariance < 5e-5
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    torch.manual_seed(5)
    initial = runner["_common_initial_state"](matcher, batch)
    target_ids = batch.target_id.reshape(-1)
    for target_id in (0, 1):
        graph_ids = torch.nonzero(target_ids == target_id, as_tuple=False).flatten()
        reference = initial.frac_coords[batch.batch == graph_ids[0]]
        for graph_id in graph_ids[1:].tolist():
            assert torch.equal(reference, initial.frac_coords[batch.batch == graph_id])
