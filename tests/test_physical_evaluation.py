import torch

from gaugeflow.production.physical_evaluation import (
    finalize_physical_metrics,
    physical_metric_sums,
)
from gaugeflow.production.physical_pretraining import PhysicalPredictions, PhysicalTargets


def test_physical_evaluation_is_graph_equal_and_functional_separated() -> None:
    batch = torch.tensor([0, 1, 1, 1])
    prediction = PhysicalPredictions(
        energy_per_atom=torch.tensor([1.0, 3.0]),
        forces=torch.tensor([[1.0, 0.0, 0.0]] * 4),
        stress_kelvin=torch.stack((torch.ones(6), torch.full((6,), 3.0))),
        teacher_features=torch.tensor([[1.0, 0.0]] * 4),
    )
    target = PhysicalTargets(
        energy_per_atom=torch.zeros(2),
        forces=torch.tensor([[1.0, 0.0, 0.0]] * 4),
        stress_kelvin=torch.zeros(2, 6),
        teacher_features=torch.tensor([[1.0, 0.0]] * 4),
        energy_mask=torch.ones(2, dtype=torch.bool),
        force_mask=torch.ones(4, dtype=torch.bool),
        stress_mask=torch.ones(2, dtype=torch.bool),
        teacher_mask=torch.tensor([True, False, False, False]),
    )
    sums = physical_metric_sums(
        prediction,
        target,
        batch,
        torch.tensor([0, 1]),
        2,
    )
    result = finalize_physical_metrics(sums, {"PBE": 0, "r2SCAN": 1})
    assert result["aggregate"]["normalized_energy_rmse"] == 5.0**0.5
    assert result["aggregate"]["normalized_force_rmse"] == 0.0
    assert result["aggregate"]["force_cosine"] == 1.0
    assert result["aggregate"]["normalized_kelvin_stress_rmse"] == 5.0**0.5
    assert result["per_functional"]["PBE"]["teacher_feature_cosine"] == 1.0
    assert result["per_functional"]["r2SCAN"]["teacher_feature_cosine"] is None
