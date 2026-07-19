import torch

from gaugeflow.geometry import GaussianRadialBasis, periodic_radius_multigraph
from scripts.audit_h1a_latent_clean_topology import (
    ScalarRidgeAccumulator,
    fit_standardized_ridge,
    smooth_first_shell_probability,
    topology_carrier,
    topology_fields,
)


def test_smooth_first_shell_is_monotone_and_cutoff_compact() -> None:
    distance = torch.tensor([1.0, 1.5, 2.0, 8.0, 8.1])
    target = torch.zeros(5, dtype=torch.long)
    nearest = torch.tensor([1.0])
    value = smooth_first_shell_probability(
        distance,
        target,
        nearest,
        multiplier=1.25,
        relative_width=0.08,
        cutoff=8.0,
    )
    assert bool((value[:-1][1:] <= value[:-1][:-1]).all())
    assert value[0] > 0.9
    assert value[-2] == 0.0
    assert value[-1] == 0.0


def test_topology_field_is_translation_and_rotation_invariant() -> None:
    clean = torch.tensor([[0.08, 0.12, 0.18], [0.36, 0.12, 0.18], [0.62, 0.52, 0.48]])
    noisy = torch.remainder(
        clean + torch.tensor([[0.02, -0.01, 0.01], [-0.01, 0.02, 0.0], [0.01, 0.0, -0.02]]),
        1.0,
    )
    lattice = torch.tensor(
        [[[4.1, 0.2, 0.0], [0.1, 4.4, 0.2], [0.0, 0.1, 4.8]]]
    )
    batch = torch.zeros(3, dtype=torch.long)
    edges = periodic_radius_multigraph(noisy, lattice, batch, cutoff=8.0)
    clean_field, noisy_field, coverage = topology_fields(
        clean,
        lattice,
        edges,
        batch,
        cutoff=8.0,
        multiplier=1.25,
        relative_width=0.08,
    )

    translated_clean = torch.remainder(clean + torch.tensor([0.31, -0.27, 0.19]), 1.0)
    translated_noisy = torch.remainder(noisy + torch.tensor([0.31, -0.27, 0.19]), 1.0)
    translated_edges = periodic_radius_multigraph(
        translated_noisy, lattice, batch, cutoff=8.0
    )
    translated_clean_field, translated_noisy_field, translated_coverage = topology_fields(
        translated_clean,
        lattice,
        translated_edges,
        batch,
        cutoff=8.0,
        multiplier=1.25,
        relative_width=0.08,
    )
    assert torch.allclose(clean_field, translated_clean_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(noisy_field, translated_noisy_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(coverage, translated_coverage, atol=2e-6, rtol=2e-6)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_lattice = lattice @ rotation.transpose(-1, -2)
    rotated_edges = periodic_radius_multigraph(noisy, rotated_lattice, batch, cutoff=8.0)
    rotated_clean_field, rotated_noisy_field, rotated_coverage = topology_fields(
        clean,
        rotated_lattice,
        rotated_edges,
        batch,
        cutoff=8.0,
        multiplier=1.25,
        relative_width=0.08,
    )
    assert torch.allclose(clean_field, rotated_clean_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(noisy_field, rotated_noisy_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(coverage, rotated_coverage, atol=2e-6, rtol=2e-6)


def test_topology_carrier_is_rotation_covariant_and_translation_horizontal() -> None:
    coordinates = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [0.7, 0.6, 0.5]])
    lattice = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.0]]])
    batch = torch.zeros(3, dtype=torch.long)
    edges = periodic_radius_multigraph(coordinates, lattice, batch, cutoff=6.0)
    radial = GaussianRadialBasis(6, 6.0)(edges.distance)
    weights = torch.exp(-0.4 * edges.distance)
    carrier = topology_carrier(
        weights,
        radial,
        edges.direction,
        edges.target,
        batch,
        coordinates.shape[0],
        1,
    )
    assert torch.allclose(carrier.mean(dim=0), torch.zeros_like(carrier.mean(dim=0)), atol=2e-6)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_lattice = lattice @ rotation.transpose(-1, -2)
    rotated_edges = periodic_radius_multigraph(
        coordinates, rotated_lattice, batch, cutoff=6.0
    )
    rotated_radial = GaussianRadialBasis(6, 6.0)(rotated_edges.distance)
    rotated = topology_carrier(
        weights,
        rotated_radial,
        rotated_edges.direction,
        rotated_edges.target,
        batch,
        coordinates.shape[0],
        1,
    )
    assert torch.allclose(rotated, carrier @ rotation.transpose(0, 1), atol=3e-6, rtol=3e-6)


def test_standardized_ridge_recovers_a_linear_clean_field() -> None:
    generator = torch.Generator().manual_seed(57)
    features = torch.randn((256, 9), generator=generator, dtype=torch.float64)
    coefficient = torch.randn(9, generator=generator, dtype=torch.float64)
    target = 0.37 + features @ coefficient
    edge_graph = torch.arange(256) // 16
    accumulator = ScalarRidgeAccumulator.create(features.shape[-1])
    accumulator.update(features, target, edge_graph, 16)
    fitted = fit_standardized_ridge(accumulator, 1.0e-10)
    prediction = fitted.predict(features)
    assert torch.mean((prediction - target).square()) < 1.0e-14
    assert fitted.rank == features.shape[-1]
