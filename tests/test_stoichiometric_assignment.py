import torch

from gaugeflow.assignment import (
    assignment_energies,
    automorphism_orbit_targets,
    automorphism_quotient_nll,
    count_constrained_assignment_quotient_nll,
    count_constrained_log_partition,
    enumerate_count_assignments,
    exact_assignment_distribution,
    exact_assignment_distribution_permutation_log_probability_error,
    exact_assignment_permutation_log_probability_error,
    exact_assignment_quotient_nll,
    expand_composition_counts,
    gumbel_sinkhorn_assignment,
    hungarian_assignment,
    residual_automorphism_permutations,
    sample_exact_assignment,
    site_species_probabilities,
    sinkhorn_bistochastic,
)


def test_sinkhorn_assignment_converges_to_site_and_slot_marginals_and_composition():
    scores = torch.tensor(
        [[3.0, 0.0, 1.0, -1.0], [0.0, 2.0, 1.0, -1.0], [1.0, 0.0, 3.0, -1.0], [0.0, 1.0, -1.0, 3.0]]
    )
    slots = expand_composition_counts(torch.tensor([1, 2, 1]))
    # This is a soft training relaxation, so use a non-degenerate temperature
    # and enough fixed iterations to test its stated converged regime.  The
    # hard Hungarian path below is the exact discrete-count assertion.
    assignment = sinkhorn_bistochastic(scores, temperature=1.0, iterations=120)
    probabilities = site_species_probabilities(assignment, slots, species_count=3)
    assert torch.allclose(assignment.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(assignment.sum(dim=0), torch.ones(4), atol=1e-5)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(probabilities.sum(dim=0), torch.tensor([1.0, 2.0, 1.0]), atol=1e-5)


def test_gumbel_sinkhorn_is_equivariant_when_exchangeable_noise_is_relabelled():
    torch.manual_seed(9)
    scores = torch.randn(4, 4)
    noise = torch.randn(4, 4)
    site_permutation = torch.tensor([2, 0, 3, 1])
    slot_permutation = torch.tensor([1, 3, 0, 2])
    reference = gumbel_sinkhorn_assignment(scores, temperature=0.4, iterations=100, noise=noise)
    relabelled = gumbel_sinkhorn_assignment(
        scores[site_permutation][:, slot_permutation],
        temperature=0.4,
        iterations=100,
        noise=noise[site_permutation][:, slot_permutation],
    )
    assert torch.allclose(relabelled, reference[site_permutation][:, slot_permutation], atol=1e-5)


def test_hungarian_assignment_exactly_respects_the_generated_species_slot_counts():
    scores = torch.tensor([[4.0, 1.0, 0.0, 0.0], [3.0, 2.0, 0.0, 0.0], [0.0, 0.0, 4.0, 1.0], [0.0, 0.0, 1.0, 4.0]])
    slots = expand_composition_counts(torch.tensor([2, 1, 1]))
    assignment = hungarian_assignment(scores)
    probabilities = site_species_probabilities(assignment, slots, species_count=3)
    assert torch.equal(assignment.sum(dim=-1), torch.ones(4))
    assert torch.equal(assignment.sum(dim=0), torch.ones(4))
    assert torch.equal(probabilities.sum(dim=0), torch.tensor([2.0, 1.0, 1.0]))


def test_quotient_loss_accepts_an_automorphically_equivalent_labeling_without_double_counting():
    # The prediction assigns site 0 -> species 1 and site 1 -> species 0.
    # This is wrong under the fixed CIF labeling [0, 1] but correct under its
    # geometry automorphism that swaps the two unlabeled sites.
    logits = torch.tensor([[-12.0, 12.0], [12.0, -12.0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    target = torch.tensor([0, 1])
    operations = torch.tensor([[0, 1], [1, 0]])
    fixed_nll = -log_probs[torch.arange(2), target].sum()
    quotient_nll = automorphism_quotient_nll(log_probs, target, operations)
    assert fixed_nll > 40.0
    assert quotient_nll < 1e-5
    # A repeated-species target has only one unique orbit labeling; adding a
    # symmetry operation must not manufacture an extra likelihood factor.
    repeated = torch.tensor([0, 0])
    labels = automorphism_orbit_targets(repeated, operations)
    assert labels.shape == (1, 2)
    assert torch.allclose(
        automorphism_quotient_nll(log_probs, repeated, operations),
        -log_probs[:, 0].sum(),
    )


def test_exact_assignment_enumerates_unique_count_constrained_labelings_not_species_slots():
    assignments = enumerate_count_assignments(torch.tensor([2, 2]))
    # 4!/(2!2!)=6 chemical assignments, not 4! permutations of artificial slots.
    assert assignments.shape == (6, 4)
    assert torch.unique(assignments, dim=0).shape == assignments.shape
    assert torch.equal(
        torch.stack([(assignment == 0).sum() for assignment in assignments]),
        torch.full((6,), 2),
    )
    assert torch.equal(
        torch.stack([(assignment == 1).sum() for assignment in assignments]),
        torch.full((6,), 2),
    )


def test_exact_quotient_uses_only_the_state_dependent_residual_automorphism_group():
    operations = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([0, 1])
    # Masked states expose no species and retain the entire geometry group.
    masked = torch.tensor([2, 2])
    assert torch.equal(residual_automorphism_permutations(masked, operations), operations)
    # Revealing one species makes the site swap incompatible.  The quotient
    # therefore collapses to the actual fixed labeling at this state.
    revealed = torch.tensor([0, 2])
    residual = residual_automorphism_permutations(revealed, operations)
    assert torch.equal(residual, torch.tensor([[0, 1]]))
    assert torch.equal(
        residual_automorphism_permutations(target, operations), torch.tensor([[0, 1]])
    )
    scores = torch.tensor([[-12.0, 12.0], [12.0, -12.0]])
    full = exact_assignment_quotient_nll(scores, torch.tensor([1, 1]), target, operations, masked)
    partial = exact_assignment_quotient_nll(scores, torch.tensor([1, 1]), target, operations, revealed)
    # This is the intended fixed-CIF-low / quotient-high sanity case: the
    # score chooses the automorphically swapped labeling, not the arbitrary
    # input row labeling.
    assert full.fixed_cif_log_probability < -40.0
    assert full.quotient_nll < 1e-5
    assert partial.quotient_nll > 40.0


def test_exact_quotient_deduplicates_repeated_species_labelings_and_samples_exact_counts():
    target = torch.tensor([0, 0, 1, 1])
    # The two group elements produce the same target labeling because they swap
    # equal species.  Its likelihood must occur once, not twice.
    operations = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    scores = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    result = exact_assignment_quotient_nll(
        scores, torch.tensor([2, 2]), target, operations, torch.full((4,), 2)
    )
    assert result.unique_orbit_targets.shape == (1, 4)
    assert torch.allclose(result.target_log_probability, result.fixed_cif_log_probability)
    generator = torch.Generator().manual_seed(17)
    for _ in range(16):
        sampled, _ = sample_exact_assignment(result.distribution, generator=generator)
        assert torch.equal(torch.bincount(sampled, minlength=2), torch.tensor([2, 2]))


def test_exact_assignment_probability_vector_is_node_relabeling_consistent_in_fp32():
    scores = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2], [0.2, 0.0, -0.3], [0.1, -0.4, 0.5]],
        dtype=torch.float32,
    )
    distribution = exact_assignment_distribution(scores, torch.tensor([2, 1, 1]))
    assert distribution.assignments.shape == (12, 4)
    error = exact_assignment_permutation_log_probability_error(
        scores, torch.tensor([2, 1, 1]), torch.tensor([2, 0, 3, 1])
    )
    assert error <= 2e-6


def test_exact_assignment_probability_vector_is_stable_after_float32_score_saturation():
    scores = torch.zeros((4, 5), dtype=torch.float32)
    scores[:, 1] = torch.tensor([1.0e7, -1.0e7, 1.0e7, -1.0e7])
    scores[:, 3] = torch.tensor([-1.0e7, 1.0e7, -1.0e7, 1.0e7])
    permutation = torch.tensor([2, 0, 3, 1])
    error = exact_assignment_distribution_permutation_log_probability_error(
        scores,
        scores[permutation],
        torch.tensor([0, 2, 0, 2, 0]),
        permutation,
    )
    assert error <= 2e-6


def test_count_dynamic_program_matches_tiny_exact_enumeration_and_has_gradients():
    torch.manual_seed(29)
    counts = torch.tensor([2, 1, 1])
    scores = torch.randn(4, 3, requires_grad=True)
    dynamic = count_constrained_log_partition(scores, counts)
    assignments = enumerate_count_assignments(counts)
    direct = torch.logsumexp(assignment_energies(scores, assignments).to(scores), dim=0)
    assert torch.allclose(dynamic, direct, atol=1e-6, rtol=1e-6)
    dynamic.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_dynamic_quotient_nll_uses_residual_group_without_factorial_support():
    scores = torch.tensor(
        [[1.2, -0.5], [-0.3, 1.0], [0.4, -0.1], [0.2, 0.7]], requires_grad=True
    )
    counts = torch.tensor([2, 2])
    target = torch.tensor([0, 1, 0, 1])
    operations = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]])
    partial = torch.tensor([2, 2, 2, 2])
    result = count_constrained_assignment_quotient_nll(scores, counts, target, operations, partial)
    reference = exact_assignment_quotient_nll(scores, counts, target, operations, partial)
    assert torch.allclose(result.quotient_nll, reference.quotient_nll.to(result.quotient_nll), atol=1e-6, rtol=1e-6)
    result.quotient_nll.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
