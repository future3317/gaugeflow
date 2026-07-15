import pytest
import torch

from gaugeflow.provenance import canonicalize_engineering_piezo_voigt, reynolds_project_proper_rank3
from gaugeflow.tensor import piezo_cartesian_to_voigt, rotate_rank3


def test_explicit_voigt_order_conversion_and_engineering_shear_rejection():
    canonical = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    source_order = ["xy", "xz", "yz", "zz", "yy", "xx"]
    source = canonical[:, [5, 4, 3, 2, 1, 0]]
    converted = canonicalize_engineering_piezo_voigt(source, source_order, engineering_shear=True)
    assert torch.equal(converted, canonical)
    with pytest.raises(ValueError):
        canonicalize_engineering_piezo_voigt(source, source_order, engineering_shear=False)


def test_proper_reynolds_projection_is_invariant_and_preserves_voigt_action():
    source = torch.tensor(
        [[1.0, 2.0, 1.0, 0.0, 0.0, 0.3], [0.2, 0.5, 0.2, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.7, 0.1, 0.0]]
    )
    rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    target, residual = reynolds_project_proper_rank3(source, torch.stack((torch.eye(3), rotation)))
    assert residual < 1e-6
    assert piezo_cartesian_to_voigt(target).shape == (3, 6)
    assert torch.allclose(rotate_rank3(target, rotation), target, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        reynolds_project_proper_rank3(source, torch.diag(torch.tensor([-1.0, 1.0, 1.0])).unsqueeze(0))
