import math

import torch

from gaugeflow.vnext.processes import RegularAffineFlow, SmoothTorusFlow, translation_horizontal_basis


def test_translation_horizontal_basis_is_orthonormal_and_removes_translation():
    basis = translation_horizontal_basis(4)
    assert basis.shape == (12, 9)
    assert torch.allclose(basis.mT @ basis, torch.eye(9, dtype=basis.dtype), atol=1.0e-12)
    translations = torch.kron(torch.ones((4, 1), dtype=basis.dtype), torch.eye(3, dtype=basis.dtype))
    assert torch.allclose(basis.mT @ translations, torch.zeros((9, 3), dtype=basis.dtype), atol=1.0e-12)


def test_regular_affine_velocity_jacobian_logdet_and_semigroup_are_exact():
    generator = torch.Generator().manual_seed(6101)
    mean = torch.randn(9, dtype=torch.float64, generator=generator)
    source = torch.randn(7, 9, dtype=torch.float64, generator=generator)
    flow = RegularAffineFlow(mean)
    time = torch.full((7,), 0.37, dtype=torch.float64)
    epsilon = 1.0e-6
    finite_difference = (flow.state(source, time + epsilon) - flow.state(source, time - epsilon)) / (2 * epsilon)
    assert torch.allclose(finite_difference, flow.velocity(flow.state(source, time), time), atol=1.0e-9)
    jacobian = torch.func.jacrev(lambda value: flow.velocity(value, torch.tensor(0.37)))(source[0])
    assert torch.allclose(jacobian, flow.vector_jacobian(torch.tensor(0.37)), atol=1.0e-12)
    start, middle, end = (torch.tensor(value, dtype=torch.float64) for value in (0.13, 0.51, 0.89))
    direct = flow.map(flow.state(source, start), start, end)
    composed = flow.map(flow.map(flow.state(source, start), start, middle), middle, end)
    assert torch.allclose(direct, composed, atol=1.0e-12)
    expected_logdet = torch.linalg.slogdet(flow.flow_jacobian(start, end)).logabsdet
    assert torch.allclose(flow.log_abs_det(start, end), expected_logdet, atol=1.0e-12)


def _torus_flow() -> SmoothTorusFlow:
    target = torch.tensor([[0.21, 0.13, 0.37], [0.46, 0.72, 0.18], [0.79, 0.34, 0.63]], dtype=torch.float64)
    return SmoothTorusFlow(target)


def test_smooth_torus_flow_derivative_horizontal_lift_and_semigroup():
    generator = torch.Generator().manual_seed(6102)
    frac = torch.rand((5, 4, 3), dtype=torch.float64, generator=generator)
    flow = _torus_flow()
    velocity = flow.velocity(frac)
    assert torch.allclose(velocity.mean(dim=-2), torch.zeros((5, 3), dtype=torch.float64), atol=1.0e-12)
    epsilon = torch.tensor(1.0e-6, dtype=torch.float64)
    finite_difference = (flow.map(frac, torch.tensor(0.0), epsilon) - frac + 0.5).remainder(1.0) - 0.5
    assert torch.allclose(finite_difference / epsilon, velocity, atol=2.0e-6)
    direct = flow.map(frac, torch.tensor(0.1), torch.tensor(0.9))
    composed = flow.map(flow.map(frac, torch.tensor(0.1), torch.tensor(0.4)), torch.tensor(0.4), torch.tensor(0.9))
    periodic_difference = torch.remainder(direct - composed + 0.5, 1.0) - 0.5
    assert periodic_difference.abs().max() <= 1.0e-12


def test_smooth_torus_flow_is_translation_and_integer_lift_invariant():
    generator = torch.Generator().manual_seed(6103)
    frac = torch.rand((3, 4, 3), dtype=torch.float64, generator=generator)
    translation = torch.tensor([0.17, -0.29, 0.41], dtype=torch.float64)
    integer_lift = torch.tensor([[2, -1, 3], [-2, 2, -1], [1, 3, -2], [-3, 0, 2]], dtype=torch.float64)
    flow = _torus_flow()
    assert torch.allclose(flow.velocity(frac), flow.velocity(frac + translation + integer_lift), atol=1.0e-12)
    mapped = flow.map(frac, torch.tensor(0.0), torch.tensor(1.0))
    transformed = flow.map(frac + translation + integer_lift, torch.tensor(0.0), torch.tensor(1.0))
    difference = torch.remainder(transformed - mapped - translation + 0.5, 1.0) - 0.5
    assert difference.abs().max() <= 1.0e-12


def test_smooth_torus_exact_solution_has_registered_quarter_tangent_contraction():
    flow = _torus_flow()
    angle = torch.linspace(-0.9 * math.pi, 0.9 * math.pi, 31, dtype=torch.float64)
    evolved = flow.evolved_angles(angle, torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(torch.tan(evolved / 2), 0.25 * torch.tan(angle / 2), atol=1.0e-12)
    jacobian = flow.analytic_relative_jacobian(angle, torch.tensor(1.0, dtype=torch.float64))
    automatic = torch.func.jacrev(lambda value: flow.evolved_angles(value, torch.tensor(1.0)))(angle)
    assert torch.allclose(torch.diagonal(automatic), jacobian, atol=1.0e-12)
