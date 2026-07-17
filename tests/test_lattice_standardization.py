from pathlib import Path

import torch

from gaugeflow.production.lattice_standardization import P1LatticeStandardizer


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1]
        / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def test_volume_residual_standardization_is_exactly_invertible():
    standardizer = _standardizer()
    counts = torch.tensor([1, 4, 20])
    latent = torch.tensor([-1.2, 0.0, 2.3])
    decoded = standardizer.decode_volume(latent, counts)
    assert torch.allclose(standardizer.encode_volume(decoded, counts), latent)


def test_shape_whitening_is_exact_on_the_trace_free_p1_chart():
    standardizer = _standardizer()
    latent = torch.tensor(
        [[-1.0, 0.2, 0.7, -0.4, 1.1], [0.3, -0.8, 0.0, 0.5, -0.2]]
    )
    decoded = standardizer.decode_shape(latent)
    assert torch.allclose(standardizer.encode_shape(decoded), latent, atol=2e-6)
    trace = decoded[:, :3].sum(dim=-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=2e-6)
