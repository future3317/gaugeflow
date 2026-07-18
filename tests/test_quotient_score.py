import torch

from gaugeflow.production.quotient_score import (
    factorized_translation_quotient_scaled_score,
)
from gaugeflow.production.schedules import wrapped_normal_score
from gaugeflow.production.state_projection import project_translation_state
from gaugeflow.production.wrapped_coordinates import AdaptiveWrappedQuotient


def _exact_scaled_score(
    displacement: torch.Tensor, sigma: float
) -> torch.Tensor:
    clean = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.68], [0.15, 0.73, 0.45]],
        dtype=torch.float64,
    )
    return sigma * AdaptiveWrappedQuotient(
        absolute_tail_tolerance=1e-12,
        relative_tail_tolerance=1e-10,
        max_images=2_000_000,
    ).evaluate(clean + displacement, clean, torch.eye(3, dtype=torch.float64), sigma).fractional_score


def test_factorized_quotient_score_matches_exact_image_oracle():
    generator = torch.Generator().manual_seed(8201)
    batch = torch.zeros(3, dtype=torch.long)
    for sigma in (0.005, 0.10, 0.25):
        for _ in range(4):
            displacement = sigma * torch.randn(
                (3, 3), generator=generator, dtype=torch.float64
            )
            observed = factorized_translation_quotient_scaled_score(
                displacement,
                torch.tensor([sigma], dtype=torch.float64),
                batch,
                1,
            )
            expected = _exact_scaled_score(displacement, sigma)
            assert torch.allclose(observed, expected, atol=2e-10, rtol=2e-10)


def test_quotient_score_is_translation_permutation_invariant_and_horizontal():
    displacement = torch.tensor(
        [[0.12, -0.07, 0.31], [-0.22, 0.18, -0.11], [0.09, 0.27, -0.38]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    sigma = torch.tensor([0.25], dtype=torch.float64)
    observed = factorized_translation_quotient_scaled_score(
        displacement, sigma, batch, 1
    )
    shifted = factorized_translation_quotient_scaled_score(
        displacement + torch.tensor([0.31, -0.27, 1.19], dtype=torch.float64),
        sigma,
        batch,
        1,
    )
    order = torch.tensor([2, 0, 1])
    permuted = factorized_translation_quotient_scaled_score(
        displacement[order], sigma, batch, 1
    )
    assert torch.allclose(shifted, observed, atol=2e-12, rtol=2e-12)
    assert torch.allclose(permuted, observed[order], atol=2e-12, rtol=2e-12)
    assert torch.allclose(observed.mean(0), torch.zeros(3, dtype=torch.float64), atol=2e-12)


def test_32_point_quotient_rule_matches_64_point_refinement_at_twenty_sites():
    generator = torch.Generator().manual_seed(8202)
    batch = torch.zeros(20, dtype=torch.long)
    for sigma in (0.10, 0.15, 0.25, 0.50):
        displacement = sigma * torch.randn(
            (20, 3), generator=generator, dtype=torch.float64
        )
        coarse = factorized_translation_quotient_scaled_score(
            displacement,
            torch.tensor([sigma], dtype=torch.float64),
            batch,
            1,
            quadrature_points=32,
        )
        refined = factorized_translation_quotient_scaled_score(
            displacement,
            torch.tensor([sigma], dtype=torch.float64),
            batch,
            1,
            quadrature_points=64,
        )
        assert torch.allclose(coarse, refined, atol=2e-9, rtol=2e-9)


def test_batched_quotient_score_matches_independent_graphs_at_low_noise():
    generator = torch.Generator().manual_seed(8203)
    counts = (4, 20)
    batch = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.long), torch.tensor(counts)
    )
    sigma = torch.tensor([0.005, 0.02], dtype=torch.float64)
    displacement = sigma[batch, None] * torch.randn(
        (sum(counts), 3), generator=generator, dtype=torch.float64
    )
    observed = factorized_translation_quotient_scaled_score(
        displacement, sigma, batch, len(counts)
    )
    start = 0
    for graph, count in enumerate(counts):
        selected = displacement[start : start + count]
        expected = factorized_translation_quotient_scaled_score(
            selected,
            sigma[graph : graph + 1],
            torch.zeros(count, dtype=torch.long),
            1,
        )
        assert torch.allclose(
            observed[start : start + count], expected, atol=2e-12, rtol=2e-12
        )
        # Far from the cut locus, the quotient heat kernel reduces to the
        # horizontal Euclidean Gaussian score at these narrow scales.
        horizontal = selected - selected.mean(dim=0, keepdim=True)
        analytic = -horizontal / sigma[graph]
        assert torch.allclose(expected, analytic, atol=2e-7, rtol=2e-7)
        start += count


def test_quotient_marginalization_removes_high_noise_nuisance_energy():
    displacement = 0.25 * torch.tensor(
        [[1.2, -0.7, 0.4], [-0.8, 1.1, -1.3], [0.5, 0.9, -0.6]],
        dtype=torch.float64,
    )
    batch = torch.zeros(3, dtype=torch.long)
    sigma = torch.tensor([0.25], dtype=torch.float64)
    quotient = factorized_translation_quotient_scaled_score(
        displacement, sigma, batch, 1
    )
    site = 0.25 * wrapped_normal_score(
        displacement, torch.full_like(displacement, 0.25)
    )
    site = project_translation_state(site, batch, 1)
    assert quotient.square().mean() < site.square().mean()
