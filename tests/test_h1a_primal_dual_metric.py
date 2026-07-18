import torch

from scripts.audit_h1a_primal_dual_metric import _metric_values


def test_primal_and_dual_forms_match_only_for_isotropic_metric() -> None:
    error = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    batch = torch.zeros(1, dtype=torch.long)
    isotropic = 3.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)
    dual, primal, dual_gradient, primal_gradient = _metric_values(
        error, isotropic, batch, 1
    )
    assert torch.allclose(primal, dual * 3.0**4)
    assert torch.allclose(primal_gradient, dual_gradient * 3.0**4)

    anisotropic = torch.diag(
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    ).unsqueeze(0)
    _, _, dual_gradient, primal_gradient = _metric_values(
        error, anisotropic, batch, 1
    )
    cosine = torch.nn.functional.cosine_similarity(
        dual_gradient.reshape(1, -1), primal_gradient.reshape(1, -1)
    )
    assert float(cosine) < 0.8
