"""Graphwise composition prediction and exact count-constrained assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


@dataclass(frozen=True)
class AssignmentLogProbability:
    """Exact conditional assignment likelihood and normalization audit."""

    log_probability: torch.Tensor
    log_partition: torch.Tensor
    target_score: torch.Tensor
    dynamic_program_states: torch.Tensor


@dataclass(frozen=True)
class AssignmentSample:
    """One exact-count site assignment sampled from the normalized law."""

    tokens: torch.Tensor
    log_probability: torch.Tensor
    log_partition: torch.Tensor


def _mixed_radix_states(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all used-count states and their exact mixed-radix strides."""
    if counts.ndim != 1 or counts.dtype != torch.long or bool((counts < 1).any()):
        raise ValueError("active composition counts must be a positive int64 vector")
    bases = counts + 1
    strides = torch.cat((counts.new_ones(1), torch.cumprod(bases[:-1], dim=0)))
    states = int(torch.prod(bases).item())
    encoded = torch.arange(states, dtype=torch.long, device=counts.device)
    digits = torch.div(encoded.unsqueeze(1), strides.unsqueeze(0), rounding_mode="floor")
    digits = digits.remainder(bases.unsqueeze(0))
    return digits, strides


def _occupation_blocks(
    site_scores: torch.Tensor,
    block_index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse indivisible occupation blocks without losing site-score mass."""
    nodes = site_scores.shape[0]
    if block_index is None:
        inverse = torch.arange(nodes, dtype=torch.long, device=site_scores.device)
    else:
        if block_index.shape != (nodes,) or block_index.dtype != torch.long:
            raise ValueError("occupation block index must contain one int64 value per site")
        _, inverse = torch.unique(block_index, sorted=True, return_inverse=True)
    blocks = int(inverse.max()) + 1
    scores = site_scores.new_zeros((blocks, site_scores.shape[1]))
    scores.index_add_(0, inverse, site_scores)
    multiplicity = torch.bincount(inverse, minlength=blocks)
    return scores, multiplicity, inverse


def _backward_assignment_messages(
    block_scores: torch.Tensor,
    multiplicity: torch.Tensor,
    counts: torch.Tensor,
    *,
    maximum_states: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Vectorized exact coefficient DP for a repeated-column permanent."""
    digits, strides = _mixed_radix_states(counts)
    state_count, species = digits.shape
    if state_count > maximum_states:
        raise ValueError(
            f"assignment DP requires {state_count} states, above limit {maximum_states}"
        )
    target = (counts * strides).sum()
    working_dtype = torch.float64 if block_scores.dtype == torch.float64 else torch.float32
    beta = torch.full(
        (state_count,),
        -torch.inf,
        dtype=working_dtype,
        device=block_scores.device,
    )
    beta[int(target)] = 0.0
    messages = [beta]
    scores = block_scores.to(working_dtype)
    for block in range(block_scores.shape[0] - 1, -1, -1):
        width = multiplicity[block]
        valid = digits + width <= counts.unsqueeze(0)
        successor = (
            torch.arange(state_count, device=counts.device).unsqueeze(1)
            + width * strides.unsqueeze(0)
        ).clamp_max(state_count - 1)
        candidate = scores[block].unsqueeze(0) + beta.index_select(0, successor.reshape(-1)).reshape(
            state_count, species
        )
        reachable = valid & torch.isfinite(candidate)
        has_successor = reachable.any(dim=1)
        safe_candidate = torch.where(
            has_successor.unsqueeze(1),
            candidate.masked_fill(~reachable, -torch.inf),
            torch.zeros_like(candidate),
        )
        reduced = torch.logsumexp(safe_candidate, dim=1)
        beta = torch.where(
            has_successor,
            reduced,
            torch.full_like(reduced, -torch.inf),
        )
        messages.append(beta)
    messages.reverse()
    if not torch.isfinite(messages[0][0]):
        raise ValueError("composition counts are incompatible with occupation-block multiplicities")
    return messages, digits, strides


def _backward_assignment_max_messages(
    block_scores: torch.Tensor,
    multiplicity: torch.Tensor,
    counts: torch.Tensor,
    *,
    maximum_states: int,
) -> list[torch.Tensor]:
    """Return max-sum messages for the exact global assignment MAP.

    Sampling uses log-sum-exp messages, whereas a MAP traceback must use the
    max-plus semiring.  Reusing conditional probability modes is only a greedy
    decoder and can miss the highest-probability complete assignment.
    """
    digits, strides = _mixed_radix_states(counts)
    state_count = digits.shape[0]
    if state_count > maximum_states:
        raise ValueError(
            f"assignment DP requires {state_count} states, above limit {maximum_states}"
        )
    target = (counts * strides).sum()
    working_dtype = torch.float64 if block_scores.dtype == torch.float64 else torch.float32
    beta = torch.full(
        (state_count,),
        -torch.inf,
        dtype=working_dtype,
        device=block_scores.device,
    )
    beta[int(target)] = 0.0
    messages = [beta]
    scores = block_scores.to(working_dtype)
    state_index = torch.arange(state_count, device=counts.device).unsqueeze(1)
    for block in range(block_scores.shape[0] - 1, -1, -1):
        valid = digits + multiplicity[block] <= counts.unsqueeze(0)
        successor = state_index + multiplicity[block] * strides.unsqueeze(0)
        safe_successor = successor.clamp_max(state_count - 1)
        candidate = scores[block].unsqueeze(0) + beta.index_select(
            0, safe_successor.reshape(-1)
        ).reshape(state_count, -1)
        beta = candidate.masked_fill(~valid, -torch.inf).amax(dim=1)
        messages.append(beta)
    messages.reverse()
    if not torch.isfinite(messages[0][0]):
        raise ValueError("composition counts are incompatible with occupation-block multiplicities")
    return messages


class CountConstrainedAssignmentLaw:
    r"""Exact ``p(A | C, carrier)`` with optional indivisible occupation blocks.

    For block scores ``S_bk`` and exact species counts ``n_k``, the normalized
    law is

    ``p(A)=exp(sum_b S_b,A_b) / Z(C,carrier)``.

    ``Z`` is the coefficient DP over used-count states, not a factorial-size
    assignment enumeration.  P1 is represented by one block per site; a
    Wyckoff/occupation carrier can make a legal orbit indivisible by assigning
    all its sites the same block index.
    """

    def __init__(self, *, maximum_active_species: int = 7, maximum_states: int = 100_000) -> None:
        if maximum_active_species < 1 or maximum_states < 2:
            raise ValueError("assignment-law bounds must be positive")
        self.maximum_active_species = maximum_active_species
        self.maximum_states = maximum_states

    def _graph_problem(
        self,
        site_scores: torch.Tensor,
        counts: torch.Tensor,
        block_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if site_scores.ndim != 2 or site_scores.shape[1] != counts.numel():
            raise ValueError("site scores and dense composition vocabulary disagree")
        if counts.ndim != 1 or counts.dtype != torch.long or bool((counts < 0).any()):
            raise ValueError("composition must be a nonnegative int64 vector")
        if int(counts.sum()) != site_scores.shape[0]:
            raise ValueError("composition counts do not match the carrier site count")
        active_species = torch.nonzero(counts > 0, as_tuple=False).flatten()
        if not 1 <= active_species.numel() <= self.maximum_active_species:
            raise ValueError("composition species support exceeds the qualified bound")
        active_counts = counts.index_select(0, active_species)
        block_scores, multiplicity, inverse = _occupation_blocks(
            site_scores.index_select(1, active_species), block_index
        )
        messages, _, strides = _backward_assignment_messages(
            block_scores,
            multiplicity,
            active_counts,
            maximum_states=self.maximum_states,
        )
        return active_species, active_counts, block_scores, multiplicity, messages, inverse

    def _one_log_prob(
        self,
        site_scores: torch.Tensor,
        counts: torch.Tensor,
        assignment: torch.Tensor,
        block_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        active, _, block_scores, _, messages, inverse = self._graph_problem(
            site_scores, counts, block_index
        )
        if assignment.shape != (site_scores.shape[0],) or assignment.dtype != torch.long:
            raise ValueError("target assignment must contain one int64 token per site")
        blocks = block_scores.shape[0]
        block_token = assignment.new_empty(blocks)
        for block in range(blocks):
            selected = assignment[inverse == block]
            if not torch.all(selected == selected[0]):
                raise ValueError("target assignment violates an indivisible occupation block")
            block_token[block] = selected[0]
        match = active.unsqueeze(0) == block_token.unsqueeze(1)
        if bool((~match.any(dim=1)).any()):
            raise ValueError("target assignment uses species outside the supplied composition")
        local = match.long().argmax(dim=1)
        target_score = block_scores.gather(1, local.unsqueeze(1)).sum()
        log_partition = messages[0][0]
        return target_score - log_partition, target_score, messages[0].numel()

    def log_prob(
        self,
        site_scores: torch.Tensor,
        batch: torch.Tensor,
        counts: torch.Tensor,
        assignment: torch.Tensor,
        *,
        block_index: torch.Tensor | None = None,
    ) -> AssignmentLogProbability:
        """Evaluate the normalized probability for a packed graph batch."""
        _validate_assignment_batch(site_scores, batch, counts)
        if assignment.shape != batch.shape or assignment.dtype != torch.long:
            raise ValueError("assignment must align with the packed site batch")
        if block_index is not None and block_index.shape != batch.shape:
            raise ValueError("occupation blocks must align with packed sites")
        logp: list[torch.Tensor] = []
        score: list[torch.Tensor] = []
        partition: list[torch.Tensor] = []
        states: list[int] = []
        for graph in range(counts.shape[0]):
            selected = batch == graph
            one_logp, one_score, state_count = self._one_log_prob(
                site_scores[selected],
                counts[graph],
                assignment[selected],
                None if block_index is None else block_index[selected],
            )
            logp.append(one_logp)
            score.append(one_score)
            partition.append(one_score - one_logp)
            states.append(state_count)
        return AssignmentLogProbability(
            log_probability=torch.stack(logp),
            log_partition=torch.stack(partition),
            target_score=torch.stack(score),
            dynamic_program_states=torch.tensor(states, dtype=torch.long, device=batch.device),
        )

    def quotient_log_prob(
        self,
        site_scores: torch.Tensor,
        batch: torch.Tensor,
        counts: torch.Tensor,
        assignment: torch.Tensor,
        parent_permutations: Sequence[torch.Tensor],
        *,
        block_index: torch.Tensor | None = None,
    ) -> AssignmentLogProbability:
        """Marginalize unique target labelings under each parent site action."""
        _validate_assignment_batch(site_scores, batch, counts)
        if len(parent_permutations) != counts.shape[0]:
            raise ValueError("one parent permutation catalogue is required per graph")
        logp: list[torch.Tensor] = []
        score: list[torch.Tensor] = []
        partition: list[torch.Tensor] = []
        states: list[int] = []
        for graph, permutations in enumerate(parent_permutations):
            selected = batch == graph
            graph_assignment = assignment[selected]
            nodes = graph_assignment.numel()
            if permutations.ndim != 2 or permutations.shape[1] != nodes:
                raise ValueError("parent permutations must have shape [operations,sites]")
            expected = torch.arange(nodes, device=permutations.device)
            if not torch.equal(torch.sort(permutations, dim=1).values, expected.expand_as(permutations)):
                raise ValueError("parent action contains a non-permutation row")
            if not bool(torch.all(permutations == expected.unsqueeze(0), dim=1).any()):
                raise ValueError("parent action must contain the identity permutation")
            labelings = torch.unique(graph_assignment.to(permutations.device)[permutations], dim=0)
            active, _, block_scores, _, messages, inverse = self._graph_problem(
                site_scores[selected],
                counts[graph],
                None if block_index is None else block_index[selected],
            )
            if block_index is not None:
                relabeled_blocks = inverse.to(permutations.device)[permutations]
                source_blocks = inverse.to(permutations.device)
                for source_block in range(block_scores.shape[0]):
                    image_values = relabeled_blocks[:, source_blocks == source_block]
                    if not torch.all(image_values == image_values[:, :1]):
                        raise ValueError(
                            "parent action does not preserve the indivisible block partition"
                        )
            orbit_labelings = labelings.to(assignment.device)
            block_tokens = torch.empty(
                (orbit_labelings.shape[0], block_scores.shape[0]),
                dtype=torch.long,
                device=assignment.device,
            )
            for block in range(block_scores.shape[0]):
                block_values = orbit_labelings[:, inverse == block]
                if not torch.all(block_values == block_values[:, :1]):
                    raise ValueError("target orbit violates an indivisible occupation block")
                block_tokens[:, block] = block_values[:, 0]
            lookup = torch.full(
                (CHEMICAL_ELEMENT_COUNT,),
                -1,
                dtype=torch.long,
                device=assignment.device,
            )
            lookup[active] = torch.arange(active.numel(), device=assignment.device)
            active_index = lookup[block_tokens]
            if bool((active_index < 0).any()):
                raise ValueError("target orbit uses species outside the supplied composition")
            orbit_score = block_scores.unsqueeze(0).expand(labelings.shape[0], -1, -1).gather(
                2, active_index.unsqueeze(-1)
            ).squeeze(-1).sum(dim=1)
            graph_partition = messages[0][0]
            graph_logp = torch.logsumexp(orbit_score - graph_partition, dim=0)
            logp.append(graph_logp)
            score.append(torch.logsumexp(orbit_score, dim=0))
            partition.append(graph_partition)
            states.append(messages[0].numel())
        return AssignmentLogProbability(
            torch.stack(logp),
            torch.stack(partition),
            torch.stack(score),
            torch.tensor(states, dtype=torch.long, device=batch.device),
        )

    @torch.no_grad()
    def entropy(
        self,
        site_scores: torch.Tensor,
        batch: torch.Tensor,
        counts: torch.Tensor,
        *,
        block_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact graphwise assignment entropy from the DP conditionals."""
        _validate_assignment_batch(site_scores, batch, counts)
        if block_index is not None and block_index.shape != batch.shape:
            raise ValueError("occupation blocks must align with packed sites")
        graph_entropies: list[torch.Tensor] = []
        for graph in range(counts.shape[0]):
            selected = batch == graph
            _, active_counts, block_scores, multiplicity, messages, _ = self._graph_problem(
                site_scores[selected],
                counts[graph],
                None if block_index is None else block_index[selected],
            )
            digits, strides = _mixed_radix_states(active_counts)
            state_count = digits.shape[0]
            state_probability = torch.zeros_like(messages[0])
            state_probability[0] = 1.0
            graph_entropy = torch.zeros((), dtype=messages[0].dtype, device=messages[0].device)
            state_index = torch.arange(state_count, device=digits.device).unsqueeze(1)
            for block in range(block_scores.shape[0]):
                successor = state_index + multiplicity[block] * strides.unsqueeze(0)
                valid = digits + multiplicity[block] <= active_counts.unsqueeze(0)
                safe_successor = successor.clamp_max(state_count - 1)
                candidate = block_scores[block].to(messages[0].dtype).unsqueeze(0)
                candidate = candidate + messages[block + 1].index_select(
                    0, safe_successor.reshape(-1)
                ).reshape_as(successor)
                reachable = (
                    valid
                    & torch.isfinite(candidate)
                    & torch.isfinite(messages[block]).unsqueeze(1)
                )
                log_conditional = torch.where(
                    reachable,
                    candidate - messages[block].unsqueeze(1),
                    torch.zeros_like(candidate),
                )
                conditional = torch.where(
                    reachable,
                    log_conditional.exp(),
                    torch.zeros_like(log_conditional),
                )
                joint = state_probability.unsqueeze(1) * conditional
                graph_entropy = graph_entropy - (joint * log_conditional).sum()
                next_probability = torch.zeros_like(state_probability)
                next_probability.index_add_(
                    0,
                    safe_successor.reshape(-1),
                    joint.reshape(-1),
                )
                state_probability = next_probability
            target_state = int((active_counts * strides).sum())
            if not torch.allclose(
                state_probability[target_state],
                torch.ones((), dtype=state_probability.dtype, device=state_probability.device),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError("assignment entropy propagation lost probability mass")
            graph_entropies.append(graph_entropy)
        return torch.stack(graph_entropies)

    @torch.no_grad()
    def sample(
        self,
        site_scores: torch.Tensor,
        batch: torch.Tensor,
        counts: torch.Tensor,
        *,
        block_index: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        mode: bool = False,
    ) -> AssignmentSample:
        """Draw a full assignment categorical, or take its exact global MAP."""
        _validate_assignment_batch(site_scores, batch, counts)
        output = torch.empty_like(batch)
        log_probabilities: list[torch.Tensor] = []
        partitions: list[torch.Tensor] = []
        for graph in range(counts.shape[0]):
            selected = batch == graph
            active, active_counts, block_scores, multiplicity, messages, inverse = self._graph_problem(
                site_scores[selected],
                counts[graph],
                None if block_index is None else block_index[selected],
            )
            digits, strides = _mixed_radix_states(active_counts)
            max_messages = (
                _backward_assignment_max_messages(
                    block_scores,
                    multiplicity,
                    active_counts,
                    maximum_states=self.maximum_states,
                )
                if mode
                else None
            )
            state = 0
            choices: list[torch.Tensor] = []
            selected_score = torch.zeros(
                (), dtype=messages[0].dtype, device=block_scores.device
            )
            for block in range(block_scores.shape[0]):
                successor = state + multiplicity[block] * strides
                valid = digits[state] + multiplicity[block] <= active_counts
                continuation = messages if max_messages is None else max_messages
                candidate = block_scores[block].to(messages[0].dtype) + continuation[
                    block + 1
                ].index_select(
                    0, successor.clamp_max(messages[block + 1].numel() - 1)
                )
                candidate = candidate.masked_fill(~valid, -torch.inf)
                if mode:
                    choice = torch.argmax(candidate)
                else:
                    log_categorical = candidate - torch.logsumexp(candidate, dim=0)
                    choice = torch.multinomial(
                        log_categorical.exp(), 1, generator=generator
                    ).squeeze(0)
                selected_score = selected_score + block_scores[block, choice].to(
                    messages[0].dtype
                )
                choices.append(choice)
                state = int(successor[choice])
            block_tokens = active.index_select(0, torch.stack(choices))
            output[selected] = block_tokens.index_select(0, inverse)
            log_probabilities.append(selected_score - messages[0][0])
            partitions.append(messages[0][0])
        observed = composition_counts_from_tokens(output, batch, counts.shape[0])
        if not torch.equal(observed, counts):
            raise RuntimeError("exact assignment sampler changed composition counts")
        return AssignmentSample(output, torch.stack(log_probabilities), torch.stack(partitions))


def _validate_assignment_batch(
    site_scores: torch.Tensor,
    batch: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    if site_scores.shape != (batch.numel(), CHEMICAL_ELEMENT_COUNT):
        raise ValueError("site scores must have shape [nodes,118]")
    if batch.ndim != 1 or batch.dtype != torch.long:
        raise ValueError("batch must be a rank-one int64 tensor")
    if counts.ndim != 2 or counts.shape[1] != CHEMICAL_ELEMENT_COUNT:
        raise ValueError("composition counts must have shape [graphs,118]")
    if counts.dtype != torch.long or bool((counts < 0).any()):
        raise ValueError("composition counts must be nonnegative int64 values")
    if bool(((batch < 0) | (batch >= counts.shape[0])).any()):
        raise ValueError("batch contains an out-of-range graph index")
    node_counts = torch.bincount(batch, minlength=counts.shape[0])
    if not torch.equal(counts.sum(dim=1), node_counts):
        raise ValueError("composition counts do not match graph node counts")


def composition_counts_from_tokens(
    tokens: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    """Return a dense ``[graphs,118]`` integer histogram."""

    if tokens.shape != batch.shape or tokens.dtype != torch.long or batch.dtype != torch.long:
        raise ValueError("tokens and batch must be equal-shape int64 vectors")
    flat = batch * CHEMICAL_ELEMENT_COUNT + tokens
    return torch.bincount(
        flat,
        minlength=graph_count * CHEMICAL_ELEMENT_COUNT,
    ).reshape(graph_count, CHEMICAL_ELEMENT_COUNT)


def rounded_graph_composition(
    composition_logits: torch.Tensor,
    node_counts: torch.Tensor,
) -> torch.Tensor:
    """Round a predicted graph-level abundance distribution to exact counts."""

    graphs = int(node_counts.numel())
    if composition_logits.shape != (graphs, CHEMICAL_ELEMENT_COUNT):
        raise ValueError("composition logits must have shape [graphs,118]")
    probability = torch.softmax(composition_logits.float(), dim=-1)
    expected = probability * node_counts.unsqueeze(-1)
    counts = expected.floor().long()
    remainder = node_counts - counts.sum(dim=-1)
    if bool((remainder < 0).any()) or bool((remainder > node_counts).any()):
        raise RuntimeError("largest-remainder composition rounding lost node count")
    order = (expected - counts).argsort(dim=-1, descending=True)
    ranks = torch.arange(
        CHEMICAL_ELEMENT_COUNT,
        device=composition_logits.device,
    ).expand(graphs, -1)
    additions = ranks < remainder.unsqueeze(-1)
    counts.scatter_add_(1, order, additions.long())
    if not torch.equal(counts.sum(dim=-1), node_counts):
        raise RuntimeError("predicted composition does not preserve graph size")
    return counts


def count_projected_assignment(
    clean_logits: torch.Tensor,
    composition_logits: torch.Tensor,
    batch: torch.Tensor,
    node_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MAP-assign sites under model-predicted, never target, composition counts.

    The only CPU operation is one terminal Hungarian solve per graph of at most
    ``20 x 20`` for Alex-MP-20.  All probability and count construction stays
    batched on the accelerator; no reverse-step fallback or per-edge transfer
    is introduced.
    """

    counts = rounded_graph_composition(composition_logits, node_counts)
    return count_constrained_assignment(clean_logits, batch, counts), counts


def count_constrained_assignment(
    clean_logits: torch.Tensor,
    batch: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """MAP-assign sites under an explicitly supplied integer composition.

    Production sampling supplies model-predicted counts through
    :func:`count_projected_assignment`. Supplying observed counts is reserved
    for offline attribution: it measures the site-assignment ceiling after
    composition error has been removed and is never an input to the denoiser
    or production sampler.
    """

    return CountConstrainedAssignmentLaw().sample(
        clean_logits,
        batch,
        counts,
        mode=True,
    ).tokens
