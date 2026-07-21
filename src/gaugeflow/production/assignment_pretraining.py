"""Target-free masked-occupation pretraining on species-free crystal geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .assignment_training import AssignmentCarrierBatch
from .autoregressive_assignment import complete_pair_rbf
from .state_projection import sorted_segment_sum


@dataclass(frozen=True)
class MaskedAssignmentCompilation:
    """One packed exact-count assignment batch and its CVP certificate."""

    carrier: AssignmentCarrierBatch
    maximum_image_shell: int
    shell_by_edge: torch.Tensor


def complete_pair_indices(
    node_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target-major directed complete-pair indices for packed graphs."""
    if node_counts.ndim != 1 or node_counts.dtype != torch.long or bool((node_counts < 1).any()):
        raise ValueError("complete-pair node counts must be positive int64")
    device = node_counts.device
    graphs = node_counts.numel()
    graph_offset = torch.cumsum(node_counts, dim=0) - node_counts
    edge_counts = node_counts * (node_counts - 1)
    edge_offset = torch.cumsum(edge_counts, dim=0) - edge_counts
    edge_graph = torch.repeat_interleave(torch.arange(graphs, device=device), edge_counts)
    if edge_graph.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty
    local_rank = torch.arange(edge_graph.numel(), device=device) - torch.repeat_interleave(
        edge_offset,
        edge_counts,
    )
    alternatives = node_counts[edge_graph] - 1
    target_local = torch.div(local_rank, alternatives, rounding_mode="floor")
    source_local = torch.remainder(local_rank, alternatives)
    source_local = source_local + (source_local >= target_local)
    source = graph_offset[edge_graph] + source_local
    target = graph_offset[edge_graph] + target_local
    return source, target, edge_graph, source_local, target_local


@torch.no_grad()
def certified_periodic_pair_distances(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    *,
    maximum_shell: int = 4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Solve packed three-dimensional CVPs by certified finite enumeration.

    Coordinates are first moved into ``[-1/2,1/2)^3``.  After searching the
    integer cube of radius ``m``, every unseen image has fractional norm at
    least ``m+1/2``.  Therefore

    ``best <= sigma_min(L) * (m + 1/2)``

    certifies that the current image is globally closest.  Uncertified pairs
    alone expand to the next shell; exhausting the registered bound fails
    closed.
    """
    if (
        fractional_coordinates.ndim != 2
        or fractional_coordinates.shape[1] != 3
        or lattice.ndim != 3
        or lattice.shape[1:] != (3, 3)
        or batch.shape != fractional_coordinates.shape[:1]
        or batch.dtype != torch.long
        or maximum_shell < 1
    ):
        raise ValueError("certified periodic-distance inputs are invalid")
    graphs = lattice.shape[0]
    if graphs < 1 or batch.numel() < 1 or int(batch.min()) != 0 or int(batch.max()) != graphs - 1:
        raise ValueError("periodic-distance batch must cover every graph")
    if not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("periodic-distance nodes must be contiguous by graph")
    if not torch.isfinite(fractional_coordinates).all() or not torch.isfinite(lattice).all():
        raise ValueError("periodic-distance inputs must be finite")
    determinant = torch.linalg.det(lattice)
    if bool((determinant <= 0).any()):
        raise ValueError("periodic-distance lattices must have positive volume")

    node_counts = torch.bincount(batch, minlength=graphs)
    source, target, edge_graph, source_local, target_local = complete_pair_indices(node_counts)
    if source.numel() == 0:
        empty_distance = lattice.new_empty(0)
        empty_shell = torch.empty(0, dtype=torch.long, device=lattice.device)
        return source, target, edge_graph, source_local, target_local, empty_distance, empty_shell

    wrapped = fractional_coordinates[target] - fractional_coordinates[source]
    wrapped = wrapped - torch.floor(wrapped + 0.5)
    singular_minimum = torch.linalg.svdvals(lattice)[:, -1]
    lattice_scale = torch.linalg.matrix_norm(lattice, ord=2, dim=(-2, -1)).clamp_min(1.0)
    tolerance = 64.0 * torch.finfo(lattice.dtype).eps * lattice_scale
    distance = lattice.new_full((source.numel(),), torch.inf)
    shell_by_edge = torch.zeros(source.numel(), dtype=torch.long, device=lattice.device)
    unresolved = torch.arange(source.numel(), device=lattice.device)
    for shell in range(1, maximum_shell + 1):
        axis = torch.arange(-shell, shell + 1, device=lattice.device, dtype=lattice.dtype)
        shifts = torch.cartesian_prod(axis, axis, axis)
        candidate = wrapped[unresolved, None, :] - shifts[None, :, :]
        cartesian = torch.einsum("eci,eij->ecj", candidate, lattice[edge_graph[unresolved]])
        best = torch.linalg.vector_norm(cartesian, dim=-1).min(dim=1).values
        distance[unresolved] = best
        lower_bound = singular_minimum[edge_graph[unresolved]] * (shell + 0.5)
        certified = best + tolerance[edge_graph[unresolved]] <= lower_bound
        shell_by_edge[unresolved[certified]] = shell
        unresolved = unresolved[~certified]
        if unresolved.numel() == 0:
            break
    if unresolved.numel():
        raise RuntimeError(
            f"periodic CVP certificate exceeded image shell {maximum_shell} "
            f"for {unresolved.numel()} pairs"
        )
    return source, target, edge_graph, source_local, target_local, distance, shell_by_edge


def _identity_site_features(
    node_counts: torch.Tensor,
    batch: torch.Tensor,
    *,
    maximum_sites: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    nodes = batch.numel()
    features = torch.zeros(
        nodes,
        4 + 3 * maximum_sites,
        dtype=dtype,
        device=batch.device,
    )
    graph_nodes = node_counts[batch]
    features[:, 0] = graph_nodes.to(dtype).reciprocal()
    features[:, 1] = 1.0
    features[:, 2] = 1.0 / 192.0
    features[:, 3] = graph_nodes.to(dtype) / maximum_sites
    features[:, 4] = 1.0
    features[
        torch.arange(nodes, device=batch.device),
        4 + maximum_sites + graph_nodes - 1,
    ] = 1.0
    features[:, 4 + 2 * maximum_sites] = 1.0
    return features


def _identity_graph_features(
    normalized_distance: torch.Tensor,
    edge_graph: torch.Tensor,
    node_counts: torch.Tensor,
    volume: torch.Tensor,
    *,
    radial_channels: int,
    maximum_sites: int,
) -> torch.Tensor:
    graphs = node_counts.numel()
    dtype = volume.dtype
    edge_counts = node_counts * (node_counts - 1)
    safe_count = edge_counts.clamp_min(1).to(dtype)
    if normalized_distance.numel():
        total = sorted_segment_sum(normalized_distance, edge_graph, graphs)
        square_total = sorted_segment_sum(normalized_distance.square(), edge_graph, graphs)
        mean = total / safe_count
        variance = (square_total / safe_count - mean.square()).clamp_min(0.0)
        minimum = torch.segment_reduce(normalized_distance, "min", lengths=edge_counts)
        maximum = torch.segment_reduce(normalized_distance, "max", lengths=edge_counts)

        maximum_edges = int(edge_counts.max())
        edge_offset = torch.cumsum(edge_counts, dim=0) - edge_counts
        local_edge = torch.arange(edge_graph.numel(), device=edge_graph.device) - torch.repeat_interleave(
            edge_offset,
            edge_counts,
        )
        padded = normalized_distance.new_full((graphs, maximum_edges), torch.inf)
        padded[edge_graph, local_edge] = normalized_distance
        ordered = torch.sort(padded, dim=1).values
        quantiles = []
        for quantile in (0.25, 0.75):
            position = (edge_counts - 1).clamp_min(0).to(dtype) * quantile
            lower = position.floor().long()
            upper = position.ceil().long()
            weight = position - lower.to(dtype)
            quantiles.append(
                ordered.gather(1, lower[:, None]).squeeze(1) * (1.0 - weight)
                + ordered.gather(1, upper[:, None]).squeeze(1) * weight
            )
        empty = edge_counts == 0
        for value in (minimum, maximum, mean, variance, *quantiles):
            value[empty] = 0.0
        summary = torch.stack(
            (mean, variance.sqrt(), minimum, maximum, quantiles[0], quantiles[1]),
            dim=1,
        )
        radial = complete_pair_rbf(
            normalized_distance,
            radial_channels=radial_channels,
            maximum_normalized_distance=1.5,
        )
        radial = sorted_segment_sum(radial, edge_graph, graphs) / safe_count[:, None]
    else:
        summary = volume.new_zeros(graphs, 6)
        radial = volume.new_zeros(graphs, radial_channels)

    global_features = torch.stack(
        (
            torch.log(volume) / 8.0,
            node_counts.to(dtype) / maximum_sites,
            node_counts.to(dtype) / maximum_sites,
            torch.full_like(volume, 0.25),
            torch.full_like(volume, 1.0 / 192.0),
            node_counts.to(dtype) / maximum_sites,
            torch.full_like(volume, 1.0 / maximum_sites),
        ),
        dim=1,
    )
    return torch.cat((summary, radial, global_features), dim=1)


@torch.no_grad()
def compile_masked_assignment_batch(
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    atom_tokens: torch.Tensor,
    *,
    maximum_sites: int = 20,
    radial_channels: int = 16,
    maximum_image_shell: int = 4,
) -> MaskedAssignmentCompilation:
    """Compile a full-Alex batch without parent or target metadata inputs."""
    if atom_tokens.shape != batch.shape or atom_tokens.dtype != torch.long:
        raise ValueError("masked assignment needs one int64 atom token per node")
    if bool(((atom_tokens < 0) | (atom_tokens >= CHEMICAL_ELEMENT_COUNT)).any()):
        raise ValueError("masked assignment atom token lies outside the vocabulary")
    graphs = lattice.shape[0]
    node_counts = torch.bincount(batch, minlength=graphs)
    if bool((node_counts > maximum_sites).any()):
        raise ValueError("masked assignment graph exceeds the site limit")
    (
        source,
        target,
        edge_graph,
        source_local,
        target_local,
        distance,
        shell_by_edge,
    ) = certified_periodic_pair_distances(
        fractional_coordinates,
        lattice,
        batch,
        maximum_shell=maximum_image_shell,
    )
    volume = torch.linalg.det(lattice)
    normalized_distance = distance / volume[edge_graph].pow(1.0 / 3.0)
    base_rbf = complete_pair_rbf(normalized_distance, radial_channels=radial_channels)
    if source.numel():
        maximum_nodes = int(node_counts.max())
        dense = base_rbf.new_zeros(
            graphs,
            maximum_nodes,
            maximum_nodes,
            radial_channels,
        )
        dense[edge_graph, source_local, target_local] = base_rbf
        context = torch.einsum("bikc,bjkd->bijcd", dense, dense)
        context = 0.5 * (context + context.transpose(-1, -2))
        upper = torch.triu_indices(radial_channels, radial_channels, device=lattice.device)
        edge_features = torch.cat(
            (
                base_rbf,
                context[edge_graph, source_local, target_local][:, upper[0], upper[1]],
            ),
            dim=1,
        )
    else:
        feature_dim = radial_channels + radial_channels * (radial_channels + 1) // 2
        edge_features = lattice.new_empty((0, feature_dim))
    composition_counts = torch.bincount(
        batch * CHEMICAL_ELEMENT_COUNT + atom_tokens,
        minlength=graphs * CHEMICAL_ELEMENT_COUNT,
    ).reshape(graphs, CHEMICAL_ELEMENT_COUNT)
    carrier = AssignmentCarrierBatch(
        site_features=_identity_site_features(
            node_counts,
            batch,
            maximum_sites=maximum_sites,
            dtype=lattice.dtype,
        ),
        graph_features=_identity_graph_features(
            normalized_distance,
            edge_graph,
            node_counts,
            volume,
            radial_channels=radial_channels,
            maximum_sites=maximum_sites,
        ),
        batch=batch,
        edge_source=source,
        edge_target=target,
        edge_rbf=edge_features,
        composition_counts=composition_counts,
        target_assignment=atom_tokens,
        parent_space_group=torch.ones(graphs, dtype=torch.long, device=lattice.device),
        cell_index=torch.ones(graphs, dtype=torch.long, device=lattice.device),
    )
    carrier.validate(vocabulary_size=CHEMICAL_ELEMENT_COUNT)
    return MaskedAssignmentCompilation(
        carrier=carrier,
        maximum_image_shell=int(shell_by_edge.max()) if shell_by_edge.numel() else 0,
        shell_by_edge=shell_by_edge,
    )
