"""Vectorized training objective for count-exact orderless assignment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
)
from .state_projection import sorted_segment_sum


@dataclass(frozen=True)
class AssignmentCarrierBatch:
    """Packed target-free carrier geometry and its occupational label."""

    site_features: torch.Tensor
    graph_features: torch.Tensor
    batch: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_rbf: torch.Tensor
    composition_counts: torch.Tensor
    target_assignment: torch.Tensor
    parent_space_group: torch.Tensor
    cell_index: torch.Tensor

    @property
    def graph_count(self) -> int:
        return int(self.composition_counts.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.batch.numel())

    def validate(self, *, vocabulary_size: int, require_target: bool = True) -> None:
        graphs = self.graph_count
        nodes = self.node_count
        if graphs < 1 or nodes < 1:
            raise ValueError("assignment batch must contain at least one carrier")
        if self.batch.shape != (nodes,) or self.batch.dtype != torch.long:
            raise ValueError("assignment batch index must be one-dimensional int64")
        if int(self.batch.min()) != 0 or int(self.batch.max()) != graphs - 1:
            raise ValueError("assignment batch index must cover every packed graph")
        if not bool((self.batch[1:] >= self.batch[:-1]).all()):
            raise ValueError("assignment nodes must be contiguous by graph")
        if require_target:
            if self.target_assignment.shape != (nodes,) or self.target_assignment.dtype != torch.long:
                raise ValueError("target assignment must contain one int64 token per node")
            if bool(((self.target_assignment < 0) | (self.target_assignment >= vocabulary_size)).any()):
                raise ValueError("target assignment token lies outside the vocabulary")
        if self.composition_counts.shape != (graphs, vocabulary_size):
            raise ValueError("composition counts have the wrong packed shape")
        if self.composition_counts.dtype != torch.long or bool((self.composition_counts < 0).any()):
            raise ValueError("composition counts must be nonnegative int64")
        node_counts = torch.bincount(self.batch, minlength=graphs)
        if not torch.equal(node_counts, self.composition_counts.sum(dim=1)):
            raise ValueError("composition counts do not close on carrier nodes")
        if require_target:
            observed = torch.bincount(
                self.batch * vocabulary_size + self.target_assignment,
                minlength=graphs * vocabulary_size,
            ).reshape(graphs, vocabulary_size)
            if not torch.equal(observed, self.composition_counts):
                raise ValueError("target assignment does not realize the supplied composition")
        if self.site_features.shape[0] != nodes or self.graph_features.shape[0] != graphs:
            raise ValueError("assignment features do not align with the packed carriers")
        if self.parent_space_group.shape != (graphs,) or self.cell_index.shape != (graphs,):
            raise ValueError("assignment parent metadata does not align with the packed carriers")
        if self.edge_source.shape != self.edge_target.shape or self.edge_source.ndim != 1:
            raise ValueError("assignment edge indices have incompatible shapes")
        if self.edge_rbf.shape[0] != self.edge_source.numel():
            raise ValueError("assignment edge features do not align with edge indices")
        if self.edge_source.numel() and (
            int(self.edge_source.min()) < 0
            or int(self.edge_target.min()) < 0
            or int(self.edge_source.max()) >= nodes
            or int(self.edge_target.max()) >= nodes
            or not torch.equal(self.batch[self.edge_source], self.batch[self.edge_target])
        ):
            raise ValueError("assignment edge crosses a packed carrier boundary")
        if self.edge_source.numel() > 1:
            edge_key = self.edge_target * nodes + self.edge_source
            if not bool((edge_key[1:] > edge_key[:-1]).all()):
                raise ValueError("assignment edges must be unique and target-major")


@dataclass(frozen=True)
class OrderlessAssignmentObjective:
    """One Monte Carlo sample of the uniform reveal-order lower bound."""

    loss: torch.Tensor
    graph_nll: torch.Tensor
    graph_log_probability: torch.Tensor
    step_log_probability: torch.Tensor
    reveal_rank: torch.Tensor


def sample_uniform_reveal_ranks(
    batch: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one target-independent uniform site order for every graph."""
    if batch.ndim != 1 or batch.dtype != torch.long or batch.numel() < 1:
        raise ValueError("assignment batch index must be a nonempty int64 vector")
    if int(batch.min()) != 0 or not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("assignment batch index must be contiguous and sorted")
    graphs = int(batch[-1]) + 1
    node_counts = torch.bincount(batch, minlength=graphs)
    graph_offsets = torch.cumsum(node_counts, dim=0) - node_counts
    random_key = torch.rand(batch.numel(), device=batch.device, generator=generator)
    permutation = torch.argsort(batch.to(random_key.dtype) + random_key, stable=True)
    rank = torch.empty_like(batch)
    rank[permutation] = torch.arange(batch.numel(), device=batch.device) - graph_offsets[batch[permutation]]
    return rank


def orderless_assignment_objective(
    scorer: GeometryAwareRemainingCountScorer,
    carrier: AssignmentCarrierBatch,
    *,
    reveal_rank: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> OrderlessAssignmentObjective:
    """Evaluate a Rao--Blackwellized uniform-order bound in one model call.

    The sampled path objective is

    ``sum_d E_{S_d} E_{i not-in S_d}[-log p(A_i | A_S, C)]``.

    One uniform reveal order supplies a prefix ``S_d`` at every depth.  The
    expectation over the next site is evaluated exactly from logits already
    produced for that prefix, removing next-site Monte Carlo variance without
    another scorer call.  This has the same expectation as the old one-next-
    site path estimator.  It remains an orderless training bound, not the
    exact order-marginal likelihood; exact subset marginalization is retained
    only as a bounded audit for small carriers.
    """
    vocabulary_size = scorer.species_embedding.num_embeddings
    carrier.validate(vocabulary_size=vocabulary_size)
    batch = carrier.batch
    graphs = carrier.graph_count
    nodes = carrier.node_count
    node_counts = torch.bincount(batch, minlength=graphs)
    graph_offsets = torch.cumsum(node_counts, dim=0) - node_counts
    if reveal_rank is None:
        reveal_rank = sample_uniform_reveal_ranks(batch, generator=generator)
    if reveal_rank.shape != (nodes,) or reveal_rank.dtype != torch.long:
        raise ValueError("reveal rank must contain one int64 rank per node")
    if not torch.equal(
        torch.sort(reveal_rank + graph_offsets[batch]).values,
        torch.arange(nodes, device=batch.device),
    ):
        raise ValueError("reveal rank must be a site permutation within each graph")

    # There is one partial-state replica per next site.  Each replica contains
    # the full carrier, so all reveal depths can share one scorer invocation.
    replica_size = node_counts[batch]
    replica = torch.repeat_interleave(torch.arange(nodes, device=batch.device), replica_size)
    replica_start = torch.cumsum(replica_size, dim=0) - replica_size
    local_node = torch.arange(replica.numel(), device=batch.device) - torch.repeat_interleave(
        replica_start, replica_size
    )
    original_node = graph_offsets[batch[replica]] + local_node
    expanded_batch = replica

    revealed = reveal_rank[original_node] < reveal_rank[replica]
    partial_assignment = torch.where(
        revealed,
        carrier.target_assignment[original_node],
        torch.full_like(original_node, -1),
    )
    observed = torch.bincount(
        replica[revealed] * vocabulary_size + carrier.target_assignment[original_node[revealed]],
        minlength=nodes * vocabulary_size,
    ).reshape(nodes, vocabulary_size)
    expanded_composition = carrier.composition_counts[batch]
    remaining_counts = expanded_composition - observed

    edge_graph = batch[carrier.edge_source]
    edge_counts = torch.bincount(edge_graph, minlength=graphs)
    edge_offsets = torch.cumsum(edge_counts, dim=0) - edge_counts
    replica_edge_count = edge_counts[batch]
    edge_replica = torch.repeat_interleave(
        torch.arange(nodes, device=batch.device),
        replica_edge_count,
    )
    replica_edge_start = torch.cumsum(replica_edge_count, dim=0) - replica_edge_count
    local_edge = torch.arange(edge_replica.numel(), device=batch.device) - torch.repeat_interleave(
        replica_edge_start,
        replica_edge_count,
    )
    original_edge = edge_offsets[batch[edge_replica]] + local_edge
    source_local = carrier.edge_source[original_edge] - graph_offsets[batch[edge_replica]]
    target_local = carrier.edge_target[original_edge] - graph_offsets[batch[edge_replica]]
    expanded_edge_source = replica_start[edge_replica] + source_local
    expanded_edge_target = replica_start[edge_replica] + target_local

    logits = scorer(
        carrier.site_features[original_node],
        carrier.graph_features[batch],
        expanded_batch,
        expanded_edge_source,
        expanded_edge_target,
        carrier.edge_rbf[original_edge],
        partial_assignment,
        expanded_composition,
        remaining_counts,
        carrier.parent_space_group[batch],
        carrier.cell_index[batch],
    )
    law = RemainingCountAssignmentLaw(vocabulary_size=vocabulary_size)
    step_log_distribution = law.batched_step_log_probabilities(
        logits,
        remaining_counts[replica],
    )
    candidate_log_probability = step_log_distribution[
        torch.arange(replica.numel(), device=batch.device),
        carrier.target_assignment[original_node],
    ]
    eligible = ~revealed
    step_log_probability = sorted_segment_sum(
        candidate_log_probability[eligible],
        replica[eligible],
        nodes,
    )
    eligible_count = torch.bincount(replica[eligible], minlength=nodes)
    step_log_probability = step_log_probability / eligible_count.clamp_min(1).to(
        step_log_probability.dtype
    )
    graph_log_probability = sorted_segment_sum(step_log_probability, batch, graphs)
    graph_nll = -graph_log_probability
    return OrderlessAssignmentObjective(
        loss=graph_nll.mean(),
        graph_nll=graph_nll,
        graph_log_probability=graph_log_probability,
        step_log_probability=step_log_probability,
        reveal_rank=reveal_rank,
    )


@torch.no_grad()
def sample_orderless_assignment(
    scorer: GeometryAwareRemainingCountScorer,
    carrier: AssignmentCarrierBatch,
    *,
    reveal_rank: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one exact-count coloring per packed carrier.

    Sites are revealed in an independently uniform order for every carrier.
    The scorer sees only the partial coloring, target-free carrier features,
    total composition and remaining counts. All active carriers at a reveal
    depth share one vectorized model call.
    """

    vocabulary_size = scorer.species_embedding.num_embeddings
    carrier.validate(vocabulary_size=vocabulary_size, require_target=False)
    graphs = carrier.graph_count
    nodes = carrier.node_count
    batch = carrier.batch
    node_counts = torch.bincount(batch, minlength=graphs)
    graph_offsets = torch.cumsum(node_counts, dim=0) - node_counts
    if reveal_rank is None:
        reveal_rank = sample_uniform_reveal_ranks(batch, generator=generator)
    if reveal_rank.shape != (nodes,) or reveal_rank.dtype != torch.long:
        raise ValueError("reveal rank must contain one int64 rank per node")
    if not torch.equal(
        torch.sort(reveal_rank + graph_offsets[batch]).values,
        torch.arange(nodes, device=batch.device),
    ):
        raise ValueError("reveal rank must be a site permutation within each graph")

    partial = torch.full((nodes,), -1, dtype=torch.long, device=batch.device)
    remaining = carrier.composition_counts.clone()
    law = RemainingCountAssignmentLaw(vocabulary_size=vocabulary_size)
    for depth in range(int(node_counts.max())):
        active_graph = torch.nonzero(node_counts > depth, as_tuple=False).squeeze(1)
        active_node = torch.nonzero(reveal_rank == depth, as_tuple=False).squeeze(1)
        if active_node.shape != active_graph.shape or not torch.equal(batch[active_node], active_graph):
            raise ValueError("reveal ranks do not select one site per active carrier")
        logits = scorer(
            carrier.site_features,
            carrier.graph_features,
            batch,
            carrier.edge_source,
            carrier.edge_target,
            carrier.edge_rbf,
            partial,
            carrier.composition_counts,
            remaining,
            carrier.parent_space_group,
            carrier.cell_index,
        )
        log_probability = law.batched_step_log_probabilities(
            logits[active_node],
            remaining[active_graph],
        )
        token = torch.multinomial(
            log_probability.exp(),
            1,
            generator=generator,
        ).squeeze(1)
        partial[active_node] = token
        remaining[active_graph, token] -= 1
    if bool((partial < 0).any()) or bool((remaining != 0).any()):
        raise RuntimeError("assignment sampling did not exhaust the exact composition")
    return partial


class OrderlessAssignmentTrainingModule(nn.Module):
    """DDP-safe wrapper whose forward pass is the per-graph orderless NLL."""

    def __init__(self, scorer: GeometryAwareRemainingCountScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(
        self,
        carrier: AssignmentCarrierBatch,
        reveal_rank: torch.Tensor,
    ) -> torch.Tensor:
        return orderless_assignment_objective(
            self.scorer,
            carrier,
            reveal_rank=reveal_rank,
        ).graph_nll
