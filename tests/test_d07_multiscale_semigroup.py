import json
import runpy
from pathlib import Path

import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import QuotientRolloutFlowMap


def test_d07_time_sampler_reserves_endpoint_maps_and_regular_semigroup_triples():
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts" / "run_gate_p5_d0_7_multiscale_semigroup_flow_map_v1.py"))
    protocol = json.loads((root / "configs" / "gate_p5_d0_7_multiscale_semigroup_flow_map_v1.json").read_text())
    start, middle, end, regular = runner["_stratified_batch_times"](
        protocol, 64, torch.device("cpu"), torch.Generator().manual_seed(81)
    )
    assert int((~regular).sum()) >= 16
    assert torch.allclose(middle[~regular], torch.ones_like(middle[~regular]))
    assert torch.allclose(end[~regular], torch.ones_like(end[~regular]))
    assert torch.all(start[regular] < middle[regular])
    assert torch.all(middle[regular] < end[regular])


def test_d07_endpoint_rows_do_not_enter_invalid_second_rollout_interval():
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "scripts" / "run_gate_p5_d0_7_multiscale_semigroup_flow_map_v1.py"))
    d06 = runner["D06"]
    protocol = json.loads((root / "configs" / "gate_p5_d0_7_multiscale_semigroup_flow_map_v1.json").read_text())
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = d06["build_repeated_endpoint"](4, device=torch.device("cpu"))
    source, residual = d06["fixed_sources"](matcher, batch, seed=93)
    start, middle, end, regular = runner["_stratified_batch_times"](protocol, 4, torch.device("cpu"), torch.Generator().manual_seed(94))
    model = QuotientRolloutFlowMap(hidden_dim=32, layers=1, coordinate_rbf_dim=8)
    direct, rollout, semigroup = runner["d07_losses"](model, matcher, batch, source, residual, start, middle, end, regular)
    (direct + rollout + semigroup).backward()
    assert torch.isfinite(direct + rollout + semigroup)
