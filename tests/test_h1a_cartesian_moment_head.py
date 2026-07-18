import torch

from scripts.audit_h1a_cartesian_moment_head import (
    _cartesian_moment_reference,
    tangent_spectrum,
)


def test_cartesian_moment_reference_is_covariant_under_improper_o3():
    generator = torch.Generator().manual_seed(701)
    directions = torch.randn((19, 3), generator=generator, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    target = torch.randint(5, (19,), generator=generator)
    vector = torch.randn((19, 4), generator=generator, dtype=torch.float64)
    tensor = torch.randn((19, 4), generator=generator, dtype=torch.float64)
    readout = torch.randn((4,), generator=generator, dtype=torch.float64)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    original = _cartesian_moment_reference(
        directions, target, vector, tensor, readout, 5
    )
    transformed = _cartesian_moment_reference(
        directions @ reflection, target, vector, tensor, readout, 5
    )
    assert torch.allclose(transformed, original @ reflection, atol=1e-12)


def test_tangent_spectrum_uses_relative_rank_and_entropy_rank():
    gram = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    result = tangent_spectrum(gram, relative_threshold=1e-8)
    assert result["tangent_rank"] == 2
    assert result["nullity"] == 1
    assert result["condition_number"] == 4.0
    assert 1.0 < result["effective_rank"] <= 2.0
