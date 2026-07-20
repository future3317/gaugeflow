"""Graphwise composition prediction and exact count-constrained assignment."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import scatter

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


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


def rounded_expected_composition(
    clean_logits: torch.Tensor,
    batch: torch.Tensor,
    node_counts: torch.Tensor,
) -> torch.Tensor:
    """Predict integer counts by largest-remainder rounding of site marginals."""

    graphs = int(node_counts.numel())
    if clean_logits.shape != (batch.numel(), CHEMICAL_ELEMENT_COUNT):
        raise ValueError("element logits must have shape [nodes,118]")
    probability = torch.softmax(clean_logits.float(), dim=-1)
    expected = scatter(
        probability,
        batch,
        dim=0,
        dim_size=graphs,
        reduce="sum",
    )
    counts = expected.floor().long()
    remainder = node_counts - counts.sum(dim=-1)
    if bool((remainder < 0).any()) or bool((remainder > node_counts).any()):
        raise RuntimeError("largest-remainder composition rounding lost node count")
    order = (expected - counts).argsort(dim=-1, descending=True)
    ranks = torch.arange(
        CHEMICAL_ELEMENT_COUNT,
        device=clean_logits.device,
    ).expand(graphs, -1)
    additions = ranks < remainder.unsqueeze(-1)
    counts.scatter_add_(1, order, additions.long())
    if not torch.equal(counts.sum(dim=-1), node_counts):
        raise RuntimeError("predicted composition does not preserve graph size")
    return counts


def count_projected_assignment(
    clean_logits: torch.Tensor,
    batch: torch.Tensor,
    node_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MAP-assign sites under model-predicted, never target, composition counts.

    The only CPU operation is one terminal Hungarian solve per graph of at most
    ``20 x 20`` for Alex-MP-20.  All probability and count construction stays
    batched on the accelerator; no reverse-step fallback or per-edge transfer
    is introduced.
    """

    counts = rounded_expected_composition(clean_logits, batch, node_counts)
    return count_constrained_assignment(clean_logits, batch, counts), counts


def count_constrained_assignment(
    clean_logits: torch.Tensor,
    batch: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """MAP-assign sites under an explicitly supplied integer composition.

    Production sampling supplies model-predicted counts through
    :func:`count_projected_assignment`.  Supplying observed counts is reserved
    for offline attribution: it measures the site-assignment ceiling after
    composition error has been removed and is never an input to the denoiser or
    production sampler.
    """

    if clean_logits.shape != (batch.numel(), CHEMICAL_ELEMENT_COUNT):
        raise ValueError("element logits must have shape [nodes,118]")
    if batch.dtype != torch.long or batch.ndim != 1:
        raise ValueError("batch must be a rank-one int64 tensor")
    if counts.ndim != 2 or counts.shape[1] != CHEMICAL_ELEMENT_COUNT:
        raise ValueError("composition counts must have shape [graphs,118]")
    if counts.dtype != torch.long or bool((counts < 0).any()):
        raise ValueError("composition counts must be nonnegative int64 values")
    graph_count = int(counts.shape[0])
    node_counts = torch.bincount(batch, minlength=graph_count)
    if not torch.equal(counts.sum(dim=-1), node_counts):
        raise ValueError("composition counts do not match graph node counts")

    assigned = torch.empty_like(batch)
    elements = torch.arange(
        CHEMICAL_ELEMENT_COUNT,
        dtype=torch.long,
        device=clean_logits.device,
    )
    for graph in range(graph_count):
        selected = batch == graph
        slots = torch.repeat_interleave(elements, counts[graph])
        graph_logits = clean_logits[selected][:, slots]
        source, target = linear_sum_assignment((-graph_logits.float()).detach().cpu().numpy())
        source_tensor = torch.as_tensor(source, dtype=torch.long, device=assigned.device)
        target_tensor = torch.as_tensor(target, dtype=torch.long, device=assigned.device)
        graph_assignment = torch.empty_like(slots)
        graph_assignment[source_tensor] = slots[target_tensor]
        assigned[selected] = graph_assignment
    observed = composition_counts_from_tokens(assigned, batch, graph_count)
    if not torch.equal(observed, counts):
        raise RuntimeError("count-constrained assignment changed predicted composition")
    return assigned
