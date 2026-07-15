import torch

from gaugeflow.parity import ParityAwareResponseBlock, parity_edge_features, transform_axial


def test_parity_edge_features_distinguish_even_odd_and_axial_quantities():
    fields = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64
    )
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    even, odd, axial = parity_edge_features(fields)
    reflected_even, reflected_odd, reflected_axial = parity_edge_features(fields @ reflection.T)
    assert torch.allclose(even, reflected_even)
    assert torch.allclose(reflected_odd, -odd)
    assert torch.allclose(reflected_axial, transform_axial(axial, reflection))


def test_parity_aware_message_block_is_o3_equivariant_including_reflection():
    torch.manual_seed(43)
    block = ParityAwareResponseBlock(scalar_dim=6, vector_dim=3, polar_edge_fields=3).double().eval()
    scalar_even = torch.randn(4, 6, dtype=torch.float64)
    scalar_odd = torch.randn(4, 6, dtype=torch.float64)
    polar = torch.randn(4, 3, 3, dtype=torch.float64)
    axial = torch.randn(4, 3, 3, dtype=torch.float64)
    source = torch.tensor([0, 1, 2, 3, 0, 2])
    target = torch.tensor([1, 2, 3, 0, 2, 1])
    edge_fields = torch.randn(6, 3, 3, dtype=torch.float64)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    with torch.no_grad():
        original = block(scalar_even, scalar_odd, polar, axial, source, target, edge_fields)
        reflected = block(
            scalar_even,
            -scalar_odd,
            polar @ reflection.T,
            transform_axial(axial, reflection),
            source,
            target,
            edge_fields @ reflection.T,
        )
    assert torch.allclose(original[0], reflected[0], atol=1e-10, rtol=1e-10)
    assert torch.allclose(-original[1], reflected[1], atol=1e-10, rtol=1e-10)
    assert torch.allclose(original[2] @ reflection.T, reflected[2], atol=1e-10, rtol=1e-10)
    assert torch.allclose(transform_axial(original[3], reflection), reflected[3], atol=1e-10, rtol=1e-10)
