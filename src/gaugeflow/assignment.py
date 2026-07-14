"""Stoichiometry-constrained, permutation-equivariant site assignments.

These primitives are intentionally separate from the frozen A5--A10 runtime.
They implement the A11-S/Q mathematical objects before any new training is
authorized: a balanced site-to-species-slot assignment and likelihood
supervision marginalized over unlabeled-geometry automorphisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import torch


def expand_composition_counts(counts: torch.Tensor) -> torch.Tensor:
    """Expand non-negative element counts into indistinguishable species slots."""
    if counts.ndim != 1:
        raise ValueError("counts must have shape [species]")
    if counts.dtype.is_floating_point:
        if not torch.isfinite(counts).all():
            raise ValueError("counts must be finite")
        if not torch.equal(counts, counts.round()):
            raise ValueError("counts must be integral")
        counts = counts.to(torch.long)
    else:
        counts = counts.to(torch.long)
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device, dtype=torch.long), counts
    )


def gumbel_noise_like(scores: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Independent exchangeable Gumbel noise for a site--slot score matrix.

    Exchangeability means that a relabeling of sites/slots relabels this noise
    with them.  Callers performing an equivariance check must therefore reuse
    the correspondingly permuted noise, rather than draw an unrelated sample.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [sites, slots] matrix")
    if not scores.dtype.is_floating_point:
        raise ValueError("scores must be floating point")
    uniform = torch.rand(
        scores.shape, dtype=scores.dtype, device=scores.device, generator=generator
    ).clamp_(torch.finfo(scores.dtype).tiny, 1.0 - torch.finfo(scores.dtype).eps)
    return -torch.log(-torch.log(uniform))


def sinkhorn_bistochastic(
    scores: torch.Tensor,
    *,
    temperature: float = 0.1,
    iterations: int = 40,
) -> torch.Tensor:
    """Differentiable finite-step relaxation of the assignment polytope.

    Finite Sinkhorn iterations approach, but do not claim to equal, doubly
    stochastic marginals.  The configured iteration count/tolerance must be
    reported for a training protocol.  Exact discrete count conservation is
    enforced at sampling by a balanced hard assignment, not by this soft
    relaxation alone.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [sites, slots] matrix")
    if not scores.dtype.is_floating_point or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite floating point")
    if temperature <= 0 or iterations < 1:
        raise ValueError("temperature and iterations must be positive")
    log_assignment = scores / temperature
    for _ in range(iterations):
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-1, keepdim=True)
        log_assignment = log_assignment - torch.logsumexp(log_assignment, dim=-2, keepdim=True)
    return log_assignment.exp()


def gumbel_sinkhorn_assignment(
    scores: torch.Tensor,
    *,
    temperature: float = 0.1,
    iterations: int = 40,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a soft, balanced assignment without using site indices as labels."""
    if noise is None:
        noise = gumbel_noise_like(scores, generator=generator)
    if noise.shape != scores.shape:
        raise ValueError("noise must match scores")
    return sinkhorn_bistochastic(scores + noise, temperature=temperature, iterations=iterations)


def site_species_probabilities(
    assignment: torch.Tensor,
    species_slots: torch.Tensor,
    species_count: int,
) -> torch.Tensor:
    """Collapse slot assignments to element probabilities and soft count marginals."""
    if assignment.ndim != 2 or assignment.shape[0] != assignment.shape[1]:
        raise ValueError("assignment must be a square [sites, slots] matrix")
    if species_slots.ndim != 1 or species_slots.numel() != assignment.shape[1]:
        raise ValueError("species_slots must contain one species index per slot")
    if species_count < 1 or (species_slots < 0).any() or (species_slots >= species_count).any():
        raise ValueError("species_slots are outside the declared species range")
    one_hot = torch.nn.functional.one_hot(species_slots.to(torch.long), species_count).to(assignment)
    return assignment @ one_hot


@torch.no_grad()
def hungarian_assignment(scores: torch.Tensor) -> torch.Tensor:
    """CPU diagnostic hard assignment that exactly uses every species slot once.

    This is intentionally a diagnostic/evaluation routine rather than a GPU
    training operator.  A future sampler may use Gumbel--Sinkhorn or a device
    native assignment solver, but both must retain the same slot constraints.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [sites, slots] matrix")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - pymatgen normally requires scipy.
        raise RuntimeError("Hungarian diagnostic assignment requires scipy") from error
    row, column = linear_sum_assignment((-scores.detach().cpu()).numpy())
    result = torch.zeros_like(scores)
    result[
        torch.as_tensor(row, dtype=torch.long, device=scores.device),
        torch.as_tensor(column, dtype=torch.long, device=scores.device),
    ] = 1.0
    return result


def _inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def automorphism_orbit_targets(
    target_types: torch.Tensor,
    automorphism_permutations: torch.Tensor,
) -> torch.Tensor:
    """Enumerate unique target labelings under site automorphisms.

    Each row of ``automorphism_permutations`` maps an original site index to
    its transformed site index.  Labels are pulled back with the inverse map,
    so every output row is a labeling on the original input-site order.  Equal
    species labelings are deduplicated before quotient likelihoods are summed.
    """
    if target_types.ndim != 1:
        raise ValueError("target_types must have shape [sites]")
    if target_types.dtype.is_floating_point:
        if not torch.isfinite(target_types).all() or not torch.equal(target_types, target_types.round()):
            raise ValueError("target_types must be integral")
    target_types = target_types.to(torch.long)
    if automorphism_permutations.ndim != 2 or automorphism_permutations.shape[1] != target_types.numel():
        raise ValueError("automorphism_permutations must have shape [operations, sites]")
    permutations = automorphism_permutations.to(dtype=torch.long, device=target_types.device)
    expected = torch.arange(target_types.numel(), device=target_types.device)
    if not torch.all(torch.sort(permutations, dim=-1).values == expected):
        raise ValueError("Every automorphism row must be a site permutation")
    transformed = torch.stack(
        [target_types[_inverse_permutation(permutation)] for permutation in permutations]
    )
    return torch.unique(transformed, dim=0)


def automorphism_quotient_nll(
    site_log_probs: torch.Tensor,
    target_types: torch.Tensor,
    automorphism_permutations: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood of the automorphism-equivalence class ``[Y]``.

    ``site_log_probs`` is a normalized ``[sites, species]`` log-probability
    matrix.  The result computes ``-log sum_{Y' in [Y]} p(Y' | X, c)`` over
    unique labelings, avoiding an artificial factor from operations that leave
    a repeated-species target unchanged.
    """
    if site_log_probs.ndim != 2 or site_log_probs.shape[0] != target_types.numel():
        raise ValueError("site_log_probs must have shape [sites, species]")
    if not site_log_probs.dtype.is_floating_point or not torch.isfinite(site_log_probs).all():
        raise ValueError("site_log_probs must be finite floating point")
    target_types = target_types.to(device=site_log_probs.device, dtype=torch.long)
    if target_types.numel() == 0 or (target_types < 0).any() or (target_types >= site_log_probs.shape[1]).any():
        raise ValueError("target_types are outside the declared species range")
    targets = automorphism_orbit_targets(target_types, automorphism_permutations)
    sites = torch.arange(target_types.numel(), device=site_log_probs.device)
    log_likelihood = torch.stack([site_log_probs[sites, labels].sum() for labels in targets])
    return -torch.logsumexp(log_likelihood, dim=0)


def _assignment_count(counts: torch.Tensor) -> int:
    total = int(counts.sum().item())
    denominator = 1
    for value in counts.detach().cpu().tolist():
        denominator *= factorial(int(value))
    return factorial(total) // denominator


def enumerate_count_assignments(
    counts: torch.Tensor,
    *,
    max_assignments: int = 4096,
) -> torch.Tensor:
    """Enumerate unique labeled-site assignments with an exact composition.

    This is intentionally restricted to the tiny A11-Q panel.  Repeated slots
    of the same species are never enumerated separately, so a 2+2 composition
    has six assignments rather than 24 slot permutations.
    """
    slots = expand_composition_counts(counts)
    site_count = int(slots.numel())
    assignment_count = _assignment_count(counts.to(torch.long))
    if assignment_count > max_assignments:
        raise ValueError(
            f"Exact enumeration would create {assignment_count} assignments; "
            f"configured maximum is {max_assignments}"
        )
    active = [
        (species, int(count))
        for species, count in enumerate(counts.to(torch.long).detach().cpu().tolist())
        if count
    ]
    rows: list[list[int]] = []

    def visit(species_index: int, available: tuple[int, ...], row: list[int]) -> None:
        if species_index == len(active):
            rows.append(row.copy())
            return
        species, count = active[species_index]
        for selected in combinations(available, count):
            next_row = row.copy()
            for site in selected:
                next_row[site] = species
            selected_set = set(selected)
            visit(
                species_index + 1,
                tuple(site for site in available if site not in selected_set),
                next_row,
            )

    visit(0, tuple(range(site_count)), [-1] * site_count)
    assignments = torch.tensor(rows, dtype=torch.long, device=counts.device)
    if assignments.shape != (assignment_count, site_count) or bool((assignments < 0).any()):
        raise RuntimeError("exact assignment enumeration was incomplete")
    return assignments


def assignment_energies(site_species_scores: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
    """Return ``S(Y)=sum_i C[i, Y_i]`` for every exact assignment ``Y``."""
    if site_species_scores.ndim != 2:
        raise ValueError("site_species_scores must have shape [sites, species]")
    if not site_species_scores.dtype.is_floating_point or not torch.isfinite(site_species_scores).all():
        raise ValueError("site_species_scores must be finite floating point")
    if assignments.ndim != 2 or assignments.shape[1] != site_species_scores.shape[0]:
        raise ValueError("assignments must have shape [assignments, sites]")
    assignments = assignments.to(device=site_species_scores.device, dtype=torch.long)
    if (assignments < 0).any() or (assignments >= site_species_scores.shape[1]).any():
        raise ValueError("assignments are outside the declared species range")
    sites = torch.arange(site_species_scores.shape[0], device=site_species_scores.device).unsqueeze(0)
    return site_species_scores[sites, assignments].sum(dim=-1)


@dataclass(frozen=True)
class ExactAssignmentDistribution:
    """Categorical probability over unique count-constrained assignments."""

    assignments: torch.Tensor
    energies: torch.Tensor
    log_probabilities: torch.Tensor


def exact_assignment_distribution(
    site_species_scores: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_assignments: int = 4096,
) -> ExactAssignmentDistribution:
    """Normalize the exact categorical distribution on ``A(counts)``."""
    if counts.ndim != 1 or counts.numel() != site_species_scores.shape[1]:
        raise ValueError("counts must contain one non-negative integer per score species")
    assignments = enumerate_count_assignments(
        counts.to(device=site_species_scores.device), max_assignments=max_assignments
    )
    energies = assignment_energies(site_species_scores, assignments)
    return ExactAssignmentDistribution(assignments, energies, torch.log_softmax(energies, dim=0))


def residual_automorphism_permutations(
    partial_tokens: torch.Tensor,
    automorphism_permutations: torch.Tensor,
) -> torch.Tensor:
    """Return ``Gamma_t={gamma in Aut(X): gamma y_t=y_t}``.

    ``partial_tokens`` includes both chemical indices and the absorbing mask.
    Compatibility is therefore evaluated against the *current* revealed state,
    not the initial full automorphism group.  A permutation maps source index
    ``i`` to transformed index ``permutation[i]``; invariance is equivalent to
    ``partial_tokens[permutation] == partial_tokens``.
    """
    if partial_tokens.ndim != 1:
        raise ValueError("partial_tokens must have shape [sites]")
    if automorphism_permutations.ndim != 2 or automorphism_permutations.shape[1] != partial_tokens.numel():
        raise ValueError("automorphism_permutations must have shape [operations, sites]")
    permutations = automorphism_permutations.to(device=partial_tokens.device, dtype=torch.long)
    expected = torch.arange(partial_tokens.numel(), device=partial_tokens.device)
    if not torch.all(torch.sort(permutations, dim=-1).values == expected):
        raise ValueError("Every automorphism row must be a site permutation")
    compatible = (partial_tokens[permutations] == partial_tokens.unsqueeze(0)).all(dim=-1)
    residual = permutations[compatible]
    if residual.numel() == 0:
        raise RuntimeError("Residual automorphism group unexpectedly omitted identity")
    return residual


@dataclass(frozen=True)
class ExactQuotientResult:
    """Exact target-orbit probability and diagnostics for an A11-Q state."""

    distribution: ExactAssignmentDistribution
    residual_automorphisms: torch.Tensor
    unique_orbit_targets: torch.Tensor
    target_log_probability: torch.Tensor
    fixed_cif_log_probability: torch.Tensor
    quotient_nll: torch.Tensor


def exact_assignment_quotient_nll(
    site_species_scores: torch.Tensor,
    counts: torch.Tensor,
    target_types: torch.Tensor,
    automorphism_permutations: torch.Tensor,
    partial_tokens: torch.Tensor,
    *,
    max_assignments: int = 4096,
) -> ExactQuotientResult:
    """Compute exact ``-log p([Y])`` on count-constrained assignments.

    The target orbit is formed only under the residual group compatible with
    ``partial_tokens``.  Identical-species labelings are deduplicated before
    their categorical probabilities are summed.
    """
    if target_types.ndim != 1 or target_types.numel() != site_species_scores.shape[0]:
        raise ValueError("target_types must contain one species per site")
    target_types = target_types.to(device=site_species_scores.device, dtype=torch.long)
    partial_tokens = partial_tokens.to(device=site_species_scores.device, dtype=torch.long)
    if not torch.equal(torch.bincount(target_types, minlength=counts.numel()), counts.to(target_types)):
        raise ValueError("target_types must exactly match the provided composition counts")
    distribution = exact_assignment_distribution(
        site_species_scores, counts, max_assignments=max_assignments
    )
    residual = residual_automorphism_permutations(partial_tokens, automorphism_permutations)
    orbit_targets = automorphism_orbit_targets(target_types, residual)
    matches = (distribution.assignments[:, None, :] == orbit_targets[None, :, :]).all(dim=-1).any(dim=-1)
    fixed = (distribution.assignments == target_types.unsqueeze(0)).all(dim=-1)
    if not bool(matches.any()) or int(fixed.sum()) != 1:
        raise RuntimeError("Exact support did not contain the target orbit exactly once per labeling")
    target_log_probability = torch.logsumexp(distribution.log_probabilities[matches], dim=0)
    fixed_log_probability = distribution.log_probabilities[fixed].squeeze(0)
    return ExactQuotientResult(
        distribution=distribution,
        residual_automorphisms=residual,
        unique_orbit_targets=orbit_targets,
        target_log_probability=target_log_probability,
        fixed_cif_log_probability=fixed_log_probability,
        quotient_nll=-target_log_probability,
    )


@torch.no_grad()
def sample_exact_assignment(
    distribution: ExactAssignmentDistribution,
    *,
    gumbel: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Gumbel-max sample from the complete exact assignment categorical law."""
    if gumbel is None:
        uniform = torch.rand(
            distribution.energies.shape,
            dtype=distribution.energies.dtype,
            device=distribution.energies.device,
            generator=generator,
        ).clamp_(torch.finfo(distribution.energies.dtype).tiny, 1.0 - torch.finfo(distribution.energies.dtype).eps)
        gumbel = -torch.log(-torch.log(uniform))
    if gumbel.shape != distribution.energies.shape:
        raise ValueError("assignment-level Gumbels must match the exact support")
    index = int((distribution.energies + gumbel).argmax())
    return distribution.assignments[index], index


def exact_assignment_permutation_log_probability_error(
    site_species_scores: torch.Tensor,
    counts: torch.Tensor,
    node_permutation: torch.Tensor,
    *,
    max_assignments: int = 4096,
) -> torch.Tensor:
    """Maximum log-probability error after a node relabeling.

    The returned scalar supports the pre-registered FP32 permutation-consistency
    test.  ``node_permutation`` specifies the old site carried by each new
    site, so scores and assignment label vectors are indexed by it together.
    """
    if node_permutation.ndim != 1 or node_permutation.numel() != site_species_scores.shape[0]:
        raise ValueError("node_permutation must have one entry per site")
    permutation = node_permutation.to(device=site_species_scores.device, dtype=torch.long)
    if not torch.equal(torch.sort(permutation).values, torch.arange(permutation.numel(), device=permutation.device)):
        raise ValueError("node_permutation must be a permutation")
    original = exact_assignment_distribution(site_species_scores, counts, max_assignments=max_assignments)
    relabelled = exact_assignment_distribution(
        site_species_scores[permutation], counts, max_assignments=max_assignments
    )
    transformed = original.assignments[:, permutation]
    matches = (transformed[:, None, :] == relabelled.assignments[None, :, :]).all(dim=-1)
    if not bool(matches.any(dim=-1).all()) or bool((matches.sum(dim=-1) != 1).any()):
        raise RuntimeError("Relabeled exact assignment support was not bijective")
    relabelled_index = matches.to(torch.int64).argmax(dim=-1)
    return (original.log_probabilities - relabelled.log_probabilities[relabelled_index]).abs().max()
