import torch

from scripts.audit_h1a_coordinate_tangent import tangent_spectrum_metrics


def test_tangent_spectrum_reports_rank_condition_and_target_projection():
    gram = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    reachable = tangent_spectrum_metrics(
        gram, torch.tensor([1.0, 2.0, 0.0]), relative_threshold=1e-7
    )
    assert reachable["tangent_rank"] == 2
    assert reachable["condition_number"] == 4.0
    assert reachable["target_projection_relative_residual"] < 1e-12
    missing = tangent_spectrum_metrics(
        gram, torch.tensor([0.0, 0.0, 1.0]), relative_threshold=1e-7
    )
    assert missing["target_projection_relative_residual"] == 1.0
