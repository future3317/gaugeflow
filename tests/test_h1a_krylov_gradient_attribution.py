import torch

from scripts.audit_h1a_krylov_gradient_attribution import (
    _parameter_group,
    _rms_normalize,
    classify_attribution,
)

THRESHOLDS = {
    "rms_derivative_amplification_min": 2.0,
    "fractional_over_cartesian_gradient_min": 3.0,
    "q2m_over_full_gradient_min": 1.0,
    "parameter_group_squared_fraction_min": 0.5,
}


def test_attribution_priority_is_frozen() -> None:
    assert classify_attribution(2.1, 8.0, 4.0, 0.9, THRESHOLDS) == "rms_derivative_dominant"
    assert classify_attribution(1.0, 3.1, 4.0, 0.9, THRESHOLDS) == "fractional_chart_dominant"
    assert classify_attribution(1.0, 2.0, 1.1, 0.9, THRESHOLDS) == "q2m_order_dominant"
    assert classify_attribution(1.0, 2.0, 0.5, 0.6, THRESHOLDS) == "parameter_group_dominant"
    assert classify_attribution(1.0, 2.0, 0.5, 0.4, THRESHOLDS) == "distributed_jacobian_scale"


def test_detached_rms_preserves_forward_but_removes_scale_derivative() -> None:
    generator = torch.Generator().manual_seed(5927)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    live_input = torch.randn((4, 2, 3), generator=generator, requires_grad=True)
    detached_input = live_input.detach().clone().requires_grad_(True)

    live, live_scale = _rms_normalize(
        live_input, batch, 2, 3.0, 1.0e-4, detach_scale=False
    )
    detached, detached_scale = _rms_normalize(
        detached_input, batch, 2, 3.0, 1.0e-4, detach_scale=True
    )

    assert torch.equal(live, detached)
    assert torch.equal(live_scale, detached_scale)
    live_gradient = torch.autograd.grad(live.square().mean(), live_input)[0]
    detached_gradient = torch.autograd.grad(
        detached.square().mean(), detached_input
    )[0]
    assert torch.isfinite(live_gradient).all()
    assert torch.isfinite(detached_gradient).all()
    assert not torch.allclose(live_gradient, detached_gradient)


def test_parameter_groups_cover_candidate_modules_without_fallback_aliases() -> None:
    assert _parameter_group("coordinate_carrier_head.weight") == "carrier_head"
    assert (
        _parameter_group("coordinate_carrier.moment_projection.weight")
        == "moment_projection"
    )
    assert _parameter_group("coordinate_edge_encoder.0.weight") == "edge_encoder"
    assert _parameter_group("coordinate_control_gate.weight") == "control_gate"
    assert _parameter_group("blocks.2.scalar_update.0.weight") == "message_blocks"
    assert _parameter_group("state_embedding.0.weight") == "shared"
