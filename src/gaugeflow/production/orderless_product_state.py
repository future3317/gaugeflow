"""Exact-count partial occupation states for product-space reverse diffusion.

The categorical part of the active product-space reverse process is not an
independent site-token chain.  A graph carries its sampled composition counts
throughout the path; a uniformly random, unobserved reveal order turns the
categorical state into a partially revealed exact-count assignment.  The
remaining-count base measure makes every step normalized on legal species and
guarantees terminal count closure without a projection pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .assignment_training import sample_uniform_reveal_ranks
from .autoregressive_assignment import RemainingCountAssignmentLaw


@dataclass(frozen=True)
class OrderlessPartialOccupation:
    """One partially revealed packed assignment and its legal next decisions."""

    partial_tokens: torch.Tensor
    composition_counts: torch.Tensor
    remaining_counts: torch.Tensor
    batch: torch.Tensor
    reveal_rank: torch.Tensor
    reveal_count: torch.Tensor
    next_site: torch.Tensor
    next_token: torch.Tensor
    mask_token: int

    @property
    def graphs(self) -> int:
        return int(self.composition_counts.shape[0])

    def validate(self, *, vocabulary_size: int = CHEMICAL_ELEMENT_COUNT) -> None:
        nodes = self.batch.numel()
        graphs = self.graphs
        if graphs < 1 or self.composition_counts.shape != (graphs, vocabulary_size):
            raise ValueError("composition counts have an invalid graph/vocabulary shape")
        if self.composition_counts.dtype != torch.long or self.remaining_counts.dtype != torch.long:
            raise ValueError("composition counts must use int64")
        if self.remaining_counts.shape != self.composition_counts.shape:
            raise ValueError("remaining counts do not align with composition counts")
        if bool((self.composition_counts < 0).any()) or bool((self.remaining_counts < 0).any()):
            raise ValueError("composition counts must be nonnegative")
        if self.batch.shape != (nodes,) or self.batch.dtype != torch.long:
            raise ValueError("packed batch must be an int64 node vector")
        if int(self.batch.min()) != 0 or int(self.batch.max()) != graphs - 1 or not bool(
            (self.batch[1:] >= self.batch[:-1]).all()
        ):
            raise ValueError("packed batch must be nonempty, contiguous and sorted")
        if self.partial_tokens.shape != (nodes,) or self.partial_tokens.dtype != torch.long:
            raise ValueError("partial tokens must align with packed nodes")
        if self.mask_token < vocabulary_size:
            raise ValueError("mask token must lie outside the element vocabulary")
        valid_token = (self.partial_tokens == self.mask_token) | (
            (self.partial_tokens >= 0) & (self.partial_tokens < vocabulary_size)
        )
        if not bool(valid_token.all()):
            raise ValueError("partial tokens contain an invalid categorical state")
        if self.reveal_rank.shape != (nodes,) or self.reveal_rank.dtype != torch.long:
            raise ValueError("reveal rank must align with packed nodes")
        node_count = torch.bincount(self.batch, minlength=graphs)
        offsets = torch.cumsum(node_count, dim=0) - node_count
        if not torch.equal(
            torch.sort(self.reveal_rank + offsets[self.batch]).values,
            torch.arange(nodes, device=self.batch.device),
        ):
            raise ValueError("reveal rank is not one permutation per graph")
        if self.reveal_count.shape != (graphs,) or self.reveal_count.dtype != torch.long:
            raise ValueError("reveal count must contain one int64 depth per graph")
        if bool((self.reveal_count < 0).any()) or bool((self.reveal_count >= node_count).any()):
            raise ValueError("every partial state must leave exactly one legal next reveal")
        if self.next_site.shape != (graphs,) or self.next_site.dtype != torch.long:
            raise ValueError("next site must contain one int64 node index per graph")
        if self.next_token.shape != (graphs,) or self.next_token.dtype != torch.long:
            raise ValueError("next token must contain one int64 element per graph")
        if bool(((self.next_site < 0) | (self.next_site >= nodes)).any()):
            raise ValueError("next site lies outside packed nodes")
        if bool((self.batch[self.next_site] != torch.arange(graphs, device=self.batch.device)).any()):
            raise ValueError("next sites do not cover graphs in order")
        if not torch.equal(self.reveal_rank[self.next_site], self.reveal_count):
            raise ValueError("next site does not match the declared reveal depth")
        if bool(((self.next_token < 0) | (self.next_token >= vocabulary_size)).any()):
            raise ValueError("next token lies outside the element vocabulary")
        if not bool((self.remaining_counts.gather(1, self.next_token[:, None]).squeeze(1) > 0).all()):
            raise ValueError("next target is absent from the remaining composition")
        revealed = self.partial_tokens != self.mask_token
        observed = torch.bincount(
            self.batch[revealed] * vocabulary_size + self.partial_tokens[revealed],
            minlength=graphs * vocabulary_size,
        ).reshape(graphs, vocabulary_size)
        if not torch.equal(self.composition_counts - observed, self.remaining_counts):
            raise ValueError("remaining counts do not equal composition minus revealed tokens")


def _validate_complete_assignment(
    assignment: torch.Tensor,
    batch: torch.Tensor,
    composition_counts: torch.Tensor,
    *,
    vocabulary_size: int,
) -> tuple[int, torch.Tensor]:
    if assignment.ndim != 1 or assignment.dtype != torch.long or batch.shape != assignment.shape:
        raise ValueError("assignment and batch must be aligned int64 vectors")
    if assignment.numel() < 1 or bool(((assignment < 0) | (assignment >= vocabulary_size)).any()):
        raise ValueError("assignment contains an invalid element token")
    if batch.dtype != torch.long or int(batch.min()) != 0 or not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("batch must be contiguous and sorted")
    graphs = int(batch[-1]) + 1
    if composition_counts.shape != (graphs, vocabulary_size) or composition_counts.dtype != torch.long:
        raise ValueError("composition counts have the wrong packed shape")
    if bool((composition_counts < 0).any()):
        raise ValueError("composition counts must be nonnegative")
    observed = torch.bincount(
        batch * vocabulary_size + assignment,
        minlength=graphs * vocabulary_size,
    ).reshape(graphs, vocabulary_size)
    if not torch.equal(observed, composition_counts):
        raise ValueError("assignment does not realize the supplied composition")
    node_count = torch.bincount(batch, minlength=graphs)
    return graphs, node_count


def partial_occupation_from_reveal_rank(
    assignment: torch.Tensor,
    batch: torch.Tensor,
    composition_counts: torch.Tensor,
    reveal_rank: torch.Tensor,
    reveal_count: torch.Tensor,
    *,
    vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
    mask_token: int = 118,
) -> OrderlessPartialOccupation:
    """Construct one exact partial assignment from a target-independent order.

    ``reveal_count[g]`` is in ``[0,N_g-1]``.  The next target is therefore
    defined for every graph, including a one-site cell.
    """

    graphs, node_count = _validate_complete_assignment(
        assignment,
        batch,
        composition_counts,
        vocabulary_size=vocabulary_size,
    )
    if reveal_rank.shape != assignment.shape or reveal_rank.dtype != torch.long:
        raise ValueError("reveal rank must align with assignment")
    if reveal_count.shape != (graphs,) or reveal_count.dtype != torch.long:
        raise ValueError("reveal count must contain one value per graph")
    if bool((reveal_count < 0).any()) or bool((reveal_count >= node_count).any()):
        raise ValueError("reveal count must be strictly below each node count")
    revealed = reveal_rank < reveal_count[batch]
    partial = torch.where(revealed, assignment, torch.full_like(assignment, mask_token))
    observed = torch.bincount(
        batch[revealed] * vocabulary_size + assignment[revealed],
        minlength=graphs * vocabulary_size,
    ).reshape(graphs, vocabulary_size)
    next_site = torch.nonzero(reveal_rank == reveal_count[batch], as_tuple=False).flatten()
    if next_site.shape != (graphs,) or not torch.equal(batch[next_site], torch.arange(graphs, device=batch.device)):
        raise RuntimeError("reveal order did not supply one next site per graph")
    state = OrderlessPartialOccupation(
        partial_tokens=partial,
        composition_counts=composition_counts,
        remaining_counts=composition_counts - observed,
        batch=batch,
        reveal_rank=reveal_rank,
        reveal_count=reveal_count,
        next_site=next_site,
        next_token=assignment[next_site],
        mask_token=mask_token,
    )
    state.validate(vocabulary_size=vocabulary_size)
    return state


def sample_orderless_partial_occupation(
    assignment: torch.Tensor,
    batch: torch.Tensor,
    composition_counts: torch.Tensor,
    element_time: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
    mask_token: int = 118,
) -> OrderlessPartialOccupation:
    """Sample a random order and the partial state implied by categorical time."""

    graphs, node_count = _validate_complete_assignment(
        assignment,
        batch,
        composition_counts,
        vocabulary_size=vocabulary_size,
    )
    if element_time.shape != (graphs,) or not torch.isfinite(element_time).all():
        raise ValueError("element time must provide one finite value per graph")
    if bool(((element_time < 0.0) | (element_time > 1.0)).any()):
        raise ValueError("element time must lie in [0,1]")
    reveal_count = torch.floor((1.0 - element_time) * node_count.to(element_time)).long()
    reveal_count = torch.minimum(reveal_count, node_count - 1)
    rank = sample_uniform_reveal_ranks(batch, generator=generator)
    return partial_occupation_from_reveal_rank(
        assignment,
        batch,
        composition_counts,
        rank,
        reveal_count,
        vocabulary_size=vocabulary_size,
        mask_token=mask_token,
    )


def orderless_next_reveal_nll(
    logits: torch.Tensor,
    state: OrderlessPartialOccupation,
    *,
    vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return graphwise and mean exact-count next-reveal NLLs.

    The count factor in :class:`RemainingCountAssignmentLaw` is intentionally
    retained: at zero logits, it yields the exchangeable uniform law over all
    legal assignments, rather than an artificial uniform-over-species law.
    """

    state.validate(vocabulary_size=vocabulary_size)
    if logits.shape != (state.batch.numel(), vocabulary_size) or not torch.isfinite(logits).all():
        raise ValueError("assignment logits must be finite [nodes,vocabulary]")
    law = RemainingCountAssignmentLaw(vocabulary_size=vocabulary_size)
    selected = law.batched_step_log_probabilities(logits[state.next_site], state.remaining_counts)
    graph_nll = -selected.gather(1, state.next_token[:, None]).squeeze(1)
    if not torch.isfinite(graph_nll).all():
        raise FloatingPointError("exact-count next-reveal likelihood is non-finite")
    return graph_nll, graph_nll.mean()
