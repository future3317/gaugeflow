import torch

from gaugeflow.geometry import GaussianRadialBasis, periodic_radius_multigraph
from scripts.audit_h1a_latent_clean_topology import (
    ScalarRidgeAccumulator,
    _aggregate_decision,
    _bootstrap_improvement,
    all_pair_image_features,
    exact_all_pair_geometry,
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
    clean_source, clean_target, clean_distance, _ = exact_all_pair_geometry(
        clean, lattice, batch
    )
    noisy_source, noisy_target, noisy_distance, _ = exact_all_pair_geometry(
        noisy, lattice, batch
    )
    assert torch.equal(clean_source, noisy_source)
    assert torch.equal(clean_target, noisy_target)
    clean_field, noisy_field, coverage = topology_fields(
        clean_distance,
        noisy_distance,
        clean_target,
        batch,
        cutoff=8.0,
        multiplier=1.25,
        relative_width=0.08,
    )

    translated_clean = torch.remainder(clean + torch.tensor([0.31, -0.27, 0.19]), 1.0)
    translated_noisy = torch.remainder(noisy + torch.tensor([0.31, -0.27, 0.19]), 1.0)
    _, translated_target, translated_clean_distance, _ = exact_all_pair_geometry(
        translated_clean, lattice, batch
    )
    _, _, translated_noisy_distance, _ = exact_all_pair_geometry(
        translated_noisy, lattice, batch
    )
    translated_clean_field, translated_noisy_field, translated_coverage = topology_fields(
        translated_clean_distance,
        translated_noisy_distance,
        translated_target,
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
    _, rotated_target, rotated_clean_distance, _ = exact_all_pair_geometry(
        clean, rotated_lattice, batch
    )
    _, _, rotated_noisy_distance, _ = exact_all_pair_geometry(
        noisy, rotated_lattice, batch
    )
    rotated_clean_field, rotated_noisy_field, rotated_coverage = topology_fields(
        rotated_clean_distance,
        rotated_noisy_distance,
        rotated_target,
        batch,
        cutoff=8.0,
        multiplier=1.25,
        relative_width=0.08,
    )
    assert torch.allclose(clean_field, rotated_clean_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(noisy_field, rotated_noisy_field, atol=2e-6, rtol=2e-6)
    assert torch.allclose(coverage, rotated_coverage, atol=2e-6, rtol=2e-6)


def test_all_pair_image_features_have_complete_nonself_support() -> None:
    coordinates = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [0.7, 0.6, 0.5]])
    lattice = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.0]]])
    batch = torch.zeros(3, dtype=torch.long)
    model_edges = periodic_radius_multigraph(coordinates, lattice, batch, cutoff=8.0)
    edge_state = torch.randn(
        (model_edges.source.numel(), 5), generator=torch.Generator().manual_seed(9)
    )
    pairs = all_pair_image_features(
        coordinates,
        lattice,
        batch,
        model_edges,
        edge_state,
        GaussianRadialBasis(6, 8.0),
        temperature=0.2,
    )
    assert pairs.source.numel() == 3 * 2
    assert bool((pairs.source != pairs.target).all())
    assert bool(pairs.production_edge_present.all())
    assert torch.isfinite(pairs.radial).all()
    assert torch.isfinite(pairs.vector_radial).all()


def test_topology_carrier_is_rotation_covariant_and_translation_horizontal() -> None:
    coordinates = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [0.7, 0.6, 0.5]])
    lattice = torch.tensor([[[4.0, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.0]]])
    batch = torch.zeros(3, dtype=torch.long)
    edges = periodic_radius_multigraph(coordinates, lattice, batch, cutoff=6.0)
    radial = GaussianRadialBasis(6, 6.0)(edges.distance)
    weights = torch.exp(-0.4 * edges.distance)
    carrier = topology_carrier(
        weights,
        radial[:, :, None] * edges.direction[:, None, :],
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
        rotated_radial[:, :, None] * rotated_edges.direction[:, None, :],
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


def test_bootstrap_improvement_preserves_float64_dtype() -> None:
    baseline = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    corrected = 0.8 * baseline
    interval = _bootstrap_improvement(
        baseline, corrected, seed=91, samples=64
    )
    assert all(abs(value - 0.2) < 1.0e-12 for value in interval)


def test_missing_clean_mass_invalidates_decision_before_topology_interpretation() -> None:
    specification = {
        "diagnostic": {"middle_times": [0.4, 0.5, 0.6]},
        "acceptance": {
            "middle_soft_jaccard_max": 0.8,
            "middle_hard_switch_fraction_min": 0.2,
            "clean_topology_mass_coverage_min": 0.95,
            "oracle_middle_mean_improvement_min": 0.1,
            "oracle_minus_noisy_middle_mean_min": 0.05,
            "oracle_middle_supporting_times_min": 2,
            "oracle_each_supporting_time_improvement_min": 0.05,
            "probe_middle_mean_explained_fraction_min": 0.2,
            "probe_middle_mean_auc_min": 0.8,
            "probe_middle_mean_improvement_over_noisy_min": 0.1,
            "learned_middle_mean_improvement_min": 0.05,
            "learned_to_oracle_improvement_ratio_min": 0.5,
        },
    }
    topology_rows = [
        {
            "time": time,
            "soft_jaccard": 0.1,
            "hard_switch_fraction": 0.05,
            "clean_mass_coverage": 0.58,
        }
        for time in [0.4, 0.5, 0.6]
    ]
    probe_rows = [
        {
            "time": time,
            "explained_fraction": 0.0,
            "auc": 0.5,
            "improvement_over_noisy": 0.0,
        }
        for time in [0.4, 0.5, 0.6]
    ]
    carrier_rows = [
        {"time": time, "variant": variant, "relative_improvement": 0.0}
        for time in [0.4, 0.5, 0.6]
        for variant in ["clean_oracle", "noisy_current", "learned_probe"]
    ]
    _, _, decision = _aggregate_decision(
        topology_rows, probe_rows, carrier_rows, specification
    )
    assert decision == "audit_invalid_clean_topology_mass_not_covered"
