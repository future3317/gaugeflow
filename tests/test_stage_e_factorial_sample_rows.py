from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from scripts.evaluate_stage_e1a_factorial_rollout import (
    _first_abnormal_lattice_step,
    _lattice_sample_metrics,
)


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1] / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def test_lattice_sample_metrics_include_shape_density_and_nn_quantiles() -> None:
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2]))
    lattice = torch.stack((3.0 * torch.eye(3),))
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    rows = _lattice_sample_metrics(
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.node_counts,
        blueprint,
        _standardizer(),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["log_volume"] == pytest.approx(torch.tensor(27.0).log().item())
    assert row["total_volume"] == pytest.approx(27.0)
    assert row["volume_per_atom"] == pytest.approx(13.5)
    assert row["density"] == pytest.approx(2.0 / 27.0)
    assert row["condition_number"] == pytest.approx(1.0)
    assert row["lattice_singular_values"] == pytest.approx([3.0, 3.0, 3.0])
    assert len(row["shape_chart"]) == 5
    assert len(row["log_shape"]) == 6
    assert row["nearest_neighbor"]["q05"] == pytest.approx(1.5)
    assert row["nearest_neighbor"]["q10"] == pytest.approx(1.5)
    assert row["nearest_neighbor"]["median"] == pytest.approx(1.5)


def test_first_abnormal_lattice_step_reports_threshold_crossing() -> None:
    diagnostics = SimpleNamespace(
        trajectory_time=torch.tensor([0.8, 0.4, 0.0]),
        trajectory_log_volume=torch.tensor([[1.0], [1.1]]),
        trajectory_physical_volume=torch.tensor([[3.0], [3.1]]),
        trajectory_shape_norm=torch.tensor([[1.0], [4.2]]),
        trajectory_condition_number=torch.tensor([[2.0], [2.1]]),
    )

    result = _first_abnormal_lattice_step(diagnostics, 0)

    assert result == {"step": 2, "time": 0.0, "reason": "shape_norm_gt_4"}