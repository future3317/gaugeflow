from __future__ import annotations

import torch

from scripts.audit_h1a_coordinate_orthogonal_residual_basis import (
    block_orthogonal_residual_chart,
)


def _problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83)
    vector = torch.randn((48, 5), generator=generator, dtype=torch.float64)
    edge = vector @ torch.randn((5, 7), generator=generator, dtype=torch.float64)
    edge = edge + 0.2 * torch.randn((48, 7), generator=generator, dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, 48, dtype=torch.float64)
    return vector, edge, weights


def test_block_chart_is_orthonormal_and_span_exact() -> None:
    vector, edge, weights = _problem()
    chart = block_orthogonal_residual_chart(vector, edge, weights)
    design = torch.cat((vector, edge), dim=1)
    basis = design @ chart
    gram = (basis * weights[:, None]).T @ (basis * weights[:, None])
    torch.testing.assert_close(gram, torch.eye(12, dtype=gram.dtype), atol=1e-11, rtol=1e-11)
    raw = torch.randn(12, generator=torch.Generator().manual_seed(89), dtype=torch.float64)
    orthogonal = torch.linalg.solve(chart, raw)
    torch.testing.assert_close(design @ raw, basis @ orthogonal, atol=1e-11, rtol=1e-11)


def test_block_chart_is_row_order_and_orthogonal_action_invariant() -> None:
    vector, edge, weights = _problem()
    reference = block_orthogonal_residual_chart(vector, edge, weights)
    order = torch.randperm(48, generator=torch.Generator().manual_seed(97))
    permuted = block_orthogonal_residual_chart(vector[order], edge[order], weights[order])
    torch.testing.assert_close(permuted, reference, atol=1e-11, rtol=1e-11)

    rotation, _ = torch.linalg.qr(
        torch.randn((48, 48), generator=torch.Generator().manual_seed(101), dtype=torch.float64)
    )
    rotated = block_orthogonal_residual_chart(
        rotation @ (vector * weights[:, None]) / weights[:, None],
        rotation @ (edge * weights[:, None]) / weights[:, None],
        weights,
    )
    torch.testing.assert_close(rotated, reference, atol=2e-10, rtol=2e-10)


def test_chart_calibration_has_no_target_argument() -> None:
    names = block_orthogonal_residual_chart.__annotations__
    assert set(names) == {"vector_design", "edge_design", "row_weights", "return"}
