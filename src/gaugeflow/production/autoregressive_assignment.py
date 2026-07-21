"""Count-exact orderless autoregressive occupation on a parent carrier."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

from .state_projection import sorted_segment_sum

AssignmentScore = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class RemainingCountAssignmentLaw:
    """Normalized categorical path with exact terminal composition."""

    def __init__(self, *, vocabulary_size: int = CHEMICAL_ELEMENT_COUNT) -> None:
        if vocabulary_size < 2:
            raise ValueError("assignment vocabulary must contain at least two species")
        self.vocabulary_size = vocabulary_size

    def step_log_probabilities(
        self,
        logits: torch.Tensor,
        remaining_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Add the exchangeable count base measure and normalize one step."""
        if logits.shape != (self.vocabulary_size,) or remaining_counts.shape != (self.vocabulary_size,):
            raise ValueError("assignment step inputs have the wrong vocabulary shape")
        if remaining_counts.dtype != torch.long or bool((remaining_counts < 0).any()):
            raise ValueError("remaining counts must be a nonnegative int64 vector")
        if int(remaining_counts.sum()) < 1:
            raise ValueError("assignment step has no remaining atom")
        valid = remaining_counts > 0
        log_weight = torch.where(
            valid,
            logits + torch.log(remaining_counts.clamp_min(1).to(logits.dtype)),
            torch.full_like(logits, -torch.inf),
        )
        return torch.log_softmax(log_weight, dim=0)

    def batched_step_log_probabilities(
        self,
        logits: torch.Tensor,
        remaining_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized step law for independent partially revealed carriers."""
        if (
            logits.ndim != 2
            or logits.shape[1] != self.vocabulary_size
            or remaining_counts.shape != logits.shape
        ):
            raise ValueError("batched assignment steps have the wrong shape")
        if remaining_counts.dtype != torch.long or bool((remaining_counts < 0).any()):
            raise ValueError("remaining counts must be nonnegative int64")
        if bool((remaining_counts.sum(dim=1) < 1).any()):
            raise ValueError("every batched assignment step needs a remaining atom")
        valid = remaining_counts > 0
        log_weight = torch.where(
            valid,
            logits + torch.log(remaining_counts.clamp_min(1).to(logits.dtype)),
            torch.full_like(logits, -torch.inf),
        )
        return torch.log_softmax(log_weight, dim=1)

    def path_log_probability(
        self,
        score: AssignmentScore,
        assignment: torch.Tensor,
        reveal_order: torch.Tensor,
        composition_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``log p(A|Z)`` for one target-independent reveal order."""
        self._validate_complete_inputs(assignment, reveal_order, composition_counts)
        partial = torch.full_like(assignment, -1)
        remaining = composition_counts.clone()
        value = torch.zeros((), dtype=torch.float64, device=assignment.device)
        for site in reveal_order.tolist():
            logits = score(partial, remaining)
            if logits.shape != (assignment.numel(), self.vocabulary_size):
                raise ValueError("assignment scorer must return [sites,vocabulary]")
            token = assignment[site]
            value = (
                value
                + self.step_log_probabilities(
                    logits[site].to(torch.float64),
                    remaining,
                )[token]
            )
            partial[site] = token
            remaining[token] -= 1
        if int(remaining.sum()) != 0 or not torch.equal(partial, assignment):
            raise RuntimeError("assignment path did not consume the exact composition")
        return value

    @torch.no_grad()
    def sample(
        self,
        score: AssignmentScore,
        composition_counts: torch.Tensor,
        reveal_order: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one assignment with no terminal repair or mask state."""
        sites = int(composition_counts.sum())
        if (
            composition_counts.shape != (self.vocabulary_size,)
            or composition_counts.dtype != torch.long
            or bool((composition_counts < 0).any())
            or sites < 1
        ):
            raise ValueError("sampling composition counts are invalid")
        self._validate_order(sites, reveal_order)
        placeholder = torch.zeros(
            sites,
            dtype=torch.long,
            device=composition_counts.device,
        )
        partial = torch.full_like(placeholder, -1)
        remaining = composition_counts.clone()
        for site in reveal_order.tolist():
            logits = score(partial, remaining)
            log_probability = self.step_log_probabilities(logits[site], remaining)
            token = torch.multinomial(
                log_probability.exp(),
                1,
                generator=generator,
            )[0]
            partial[site] = token
            remaining[token] -= 1
        if int(remaining.sum()) != 0 or bool((partial < 0).any()):
            raise RuntimeError("assignment sampler failed exact-count closure")
        return partial

    @torch.no_grad()
    def exact_order_marginal_probability(
        self,
        score: AssignmentScore,
        assignment: torch.Tensor,
        composition_counts: torch.Tensor,
    ) -> float:
        """Compute ``p(A)`` by subset DP for a small qualification system."""
        sites = assignment.numel()
        if sites > 20:
            raise ValueError("exact subset audit is bounded to at most 20 sites")
        self._validate_counts(assignment, composition_counts)
        states = 1 << sites
        dynamic = torch.zeros(states, dtype=torch.float64)
        dynamic[0] = 1.0
        assignment_cpu = assignment.detach().to(device="cpu", dtype=torch.long)
        counts_cpu = composition_counts.detach().to(device="cpu", dtype=torch.long)
        for mask in range(states - 1):
            if float(dynamic[mask]) == 0.0:
                continue
            partial = torch.full((sites,), -1, dtype=torch.long)
            revealed = [site for site in range(sites) if mask & (1 << site)]
            if revealed:
                index = torch.tensor(revealed, dtype=torch.long)
                partial[index] = assignment_cpu[index]
            remaining = counts_cpu - torch.bincount(
                assignment_cpu[revealed] if revealed else assignment_cpu.new_empty(0),
                minlength=self.vocabulary_size,
            )
            unassigned = sites - len(revealed)
            logits = score(partial, remaining).detach().to(dtype=torch.float64, device="cpu")
            if logits.shape != (sites, self.vocabulary_size):
                raise ValueError("assignment scorer must return [sites,vocabulary]")
            for site in range(sites):
                if mask & (1 << site):
                    continue
                token = assignment_cpu[site]
                probability = self.step_log_probabilities(
                    logits[site],
                    remaining,
                )[token].exp()
                dynamic[mask | (1 << site)] += dynamic[mask] * probability / unassigned
        return float(dynamic[-1])

    @torch.no_grad()
    def exact_quotient_probability(
        self,
        score: AssignmentScore,
        assignment: torch.Tensor,
        composition_counts: torch.Tensor,
        parent_permutations: torch.Tensor,
    ) -> float:
        """Sum unique group-orbit labelings without operation multiplicity."""
        sites = assignment.numel()
        if parent_permutations.ndim != 2 or parent_permutations.shape[1] != sites:
            raise ValueError("parent action does not cover the target assignment")
        expected = torch.arange(sites, device=parent_permutations.device)
        if not torch.equal(
            torch.sort(parent_permutations, dim=1).values,
            expected.expand_as(parent_permutations),
        ):
            raise ValueError("parent action contains a non-permutation")
        orbit = torch.unique(assignment[parent_permutations], dim=0)
        return sum(
            self.exact_order_marginal_probability(
                score,
                member,
                composition_counts,
            )
            for member in orbit
        )

    def _validate_counts(
        self,
        assignment: torch.Tensor,
        composition_counts: torch.Tensor,
    ) -> None:
        if assignment.ndim != 1 or assignment.dtype != torch.long:
            raise ValueError("assignment must be a one-dimensional int64 tensor")
        if composition_counts.shape != (self.vocabulary_size,):
            raise ValueError("composition has the wrong vocabulary shape")
        if composition_counts.dtype != torch.long or bool((composition_counts < 0).any()):
            raise ValueError("composition counts must be nonnegative int64")
        if bool(((assignment < 0) | (assignment >= self.vocabulary_size)).any()):
            raise ValueError("assignment token lies outside the vocabulary")
        observed = torch.bincount(assignment, minlength=self.vocabulary_size)
        if not torch.equal(observed.to(composition_counts.device), composition_counts):
            raise ValueError("assignment does not realize the supplied composition")

    def _validate_complete_inputs(
        self,
        assignment: torch.Tensor,
        reveal_order: torch.Tensor,
        composition_counts: torch.Tensor,
    ) -> None:
        self._validate_counts(assignment, composition_counts)
        self._validate_order(assignment.numel(), reveal_order)

    @staticmethod
    def _validate_order(sites: int, reveal_order: torch.Tensor) -> None:
        if reveal_order.shape != (sites,) or reveal_order.dtype != torch.long:
            raise ValueError("reveal order must be a site permutation")
        if not torch.equal(
            torch.sort(reveal_order).values,
            torch.arange(sites, device=reveal_order.device),
        ):
            raise ValueError("reveal order is not a permutation of the sites")


class _AssignmentMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, radial_channels: int) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(radial_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        graph_state: torch.Tensor,
        batch: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        edge = self.edge_encoder(edge_rbf)
        message = self.message(
            torch.cat(
                (node_state[edge_source], node_state[edge_target], edge),
                dim=-1,
            )
        )
        aggregated = sorted_segment_sum(message, edge_target, node_state.shape[0])
        degree = torch.bincount(
            edge_target,
            minlength=node_state.shape[0],
        ).to(message.dtype)
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        update = self.update(torch.cat((node_state, aggregated, graph_state[batch]), dim=-1))
        return self.norm(node_state + update)


class GeometryAwareRemainingCountScorer(nn.Module):
    """All-pair equivariant scorer for partially revealed occupations."""

    def __init__(
        self,
        *,
        site_feature_dim: int,
        graph_feature_dim: int,
        radial_channels: int = 16,
        hidden_dim: int = 96,
        message_blocks: int = 3,
        maximum_sites: int = 20,
        maximum_cell_index: int = 4,
    ) -> None:
        super().__init__()
        if site_feature_dim < 1 or graph_feature_dim < 1 or radial_channels < 2 or hidden_dim < 8 or message_blocks < 1:
            raise ValueError("assignment scorer dimensions are invalid")
        self.site_feature_dim = site_feature_dim
        self.graph_feature_dim = graph_feature_dim
        self.radial_channels = radial_channels
        self.maximum_sites = maximum_sites
        self.maximum_cell_index = maximum_cell_index
        self.hidden_dim = hidden_dim

        self.site_encoder = nn.Sequential(
            nn.Linear(site_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.partial_embedding = nn.Embedding(CHEMICAL_ELEMENT_COUNT + 1, hidden_dim)
        self.species_embedding = nn.Embedding(CHEMICAL_ELEMENT_COUNT, hidden_dim)
        self.total_count_embedding = nn.Embedding(maximum_sites + 1, hidden_dim)
        self.remaining_count_embedding = nn.Embedding(maximum_sites + 1, hidden_dim)
        self.space_group_embedding = nn.Embedding(231, hidden_dim)
        self.cell_embedding = nn.Embedding(maximum_cell_index + 1, hidden_dim)
        self.blocks = nn.ModuleList(_AssignmentMessageBlock(hidden_dim, radial_channels) for _ in range(message_blocks))
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.partial_embedding,
            self.species_embedding,
            self.total_count_embedding,
            self.remaining_count_embedding,
            self.space_group_embedding,
            self.cell_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(
        self,
        site_features: torch.Tensor,
        graph_features: torch.Tensor,
        batch: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_rbf: torch.Tensor,
        partial_assignment: torch.Tensor,
        composition_counts: torch.Tensor,
        remaining_counts: torch.Tensor,
        parent_space_group: torch.Tensor,
        cell_index: torch.Tensor,
    ) -> torch.Tensor:
        graphs = composition_counts.shape[0]
        nodes = site_features.shape[0]
        if site_features.shape != (nodes, self.site_feature_dim):
            raise ValueError("assignment site features have the wrong shape")
        if graph_features.shape != (graphs, self.graph_feature_dim):
            raise ValueError("assignment graph features have the wrong shape")
        if batch.shape != (nodes,) or batch.dtype != torch.long:
            raise ValueError("assignment batch has the wrong shape or dtype")
        if partial_assignment.shape != (nodes,) or partial_assignment.dtype != torch.long:
            raise ValueError("partial assignment has the wrong shape or dtype")
        if bool(((partial_assignment < -1) | (partial_assignment >= CHEMICAL_ELEMENT_COUNT)).any()):
            raise ValueError("partial assignment token lies outside the vocabulary")
        expected_counts = (graphs, CHEMICAL_ELEMENT_COUNT)
        if (
            composition_counts.shape != expected_counts
            or remaining_counts.shape != expected_counts
            or composition_counts.dtype != torch.long
            or remaining_counts.dtype != torch.long
            or bool((composition_counts < 0).any())
            or bool((remaining_counts < 0).any())
            or bool((remaining_counts > composition_counts).any())
        ):
            raise ValueError("assignment count states are invalid")
        if parent_space_group.shape != (graphs,) or cell_index.shape != (graphs,):
            raise ValueError("assignment parent metadata has the wrong shape")
        if bool(((parent_space_group < 1) | (parent_space_group > 230)).any()):
            raise ValueError("parent space group lies outside 1..230")
        if bool(((cell_index < 1) | (cell_index > self.maximum_cell_index)).any()):
            raise ValueError("cell index lies outside the qualified support")
        if (
            edge_source.shape != edge_target.shape
            or edge_source.ndim != 1
            or edge_rbf.shape != (edge_source.numel(), self.radial_channels)
        ):
            raise ValueError("assignment complete-pair graph has invalid shapes")
        node_counts = torch.bincount(batch, minlength=graphs)
        if not torch.equal(node_counts, composition_counts.sum(dim=1)):
            raise ValueError("composition counts do not close on carrier nodes")
        revealed = partial_assignment >= 0
        observed = torch.bincount(
            batch[revealed] * CHEMICAL_ELEMENT_COUNT + partial_assignment[revealed],
            minlength=graphs * CHEMICAL_ELEMENT_COUNT,
        ).reshape(graphs, CHEMICAL_ELEMENT_COUNT)
        if not torch.equal(composition_counts - observed, remaining_counts):
            raise ValueError("partial assignment and remaining counts disagree")
        if edge_source.numel() and (
            int(edge_source.min()) < 0
            or int(edge_target.min()) < 0
            or int(edge_source.max()) >= nodes
            or int(edge_target.max()) >= nodes
            or not torch.equal(batch[edge_source], batch[edge_target])
        ):
            raise ValueError("assignment pair edge crosses a graph boundary")
        if edge_source.numel() > 1:
            edge_key = edge_target * nodes + edge_source
            if not bool((edge_key[1:] > edge_key[:-1]).all()):
                raise ValueError("assignment pair edges must be unique and target-major")

        token = torch.arange(CHEMICAL_ELEMENT_COUNT, device=site_features.device)
        species = self.species_embedding(token).unsqueeze(0).expand(graphs, -1, -1)
        total_species = species + self.total_count_embedding(composition_counts.clamp_max(self.maximum_sites))
        total_context = (total_species * composition_counts.to(site_features.dtype).unsqueeze(-1)).sum(dim=1)
        total_context /= composition_counts.sum(dim=1, keepdim=True).clamp_min(1)
        remaining_species = species + self.remaining_count_embedding(remaining_counts.clamp_max(self.maximum_sites))
        remaining_context = (remaining_species * remaining_counts.to(site_features.dtype).unsqueeze(-1)).sum(dim=1)
        remaining_context /= remaining_counts.sum(dim=1, keepdim=True).clamp_min(1)
        graph_state = (
            self.graph_encoder(graph_features)
            + total_context
            + remaining_context
            + self.space_group_embedding(parent_space_group)
            + self.cell_embedding(cell_index)
        )
        node_state = (
            self.site_encoder(site_features) + self.partial_embedding(partial_assignment + 1) + graph_state[batch]
        )
        for block in self.blocks:
            node_state = block(
                node_state,
                graph_state,
                batch,
                edge_source,
                edge_target,
                edge_rbf,
            )
        query = torch.nn.functional.normalize(self.query(node_state), dim=-1)
        key = torch.nn.functional.normalize(self.key(remaining_species), dim=-1)
        return torch.einsum("nd,nkd->nk", query, key[batch]) * math.sqrt(self.hidden_dim)


def complete_pair_rbf(
    distances: torch.Tensor,
    *,
    radial_channels: int,
    maximum_normalized_distance: float = 2.5,
) -> torch.Tensor:
    """Vectorized smooth basis for precomputed dimensionless pair distances."""
    if distances.ndim != 1 or not torch.isfinite(distances).all():
        raise ValueError("complete-pair distances must be a finite vector")
    if radial_channels < 2 or maximum_normalized_distance <= 0.0:
        raise ValueError("complete-pair RBF bounds are invalid")
    centers = torch.linspace(
        0.0,
        maximum_normalized_distance,
        radial_channels,
        dtype=distances.dtype,
        device=distances.device,
    )
    width = maximum_normalized_distance / (radial_channels - 1)
    return torch.exp(-0.5 * ((distances[:, None] - centers) / width) ** 2)


def complete_pair_context_features(
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    edge_rbf: torch.Tensor,
    *,
    node_count: int,
) -> torch.Tensor:
    """Add a target-free two-point view of every third carrier site.

    For a pair ``(i,j)``, the added feature is the upper triangle of

    ``sum_k sym(rbf(d_ik) outer rbf(d_jk))``.

    It is invariant to endpoint exchange, node relabeling, rigid motion and
    ``GL(3,Z)`` cell-basis changes.  The cubic contraction is performed once
    while compiling a carrier with at most 20 sites; model training and
    sampling still consume one fixed feature per directed pair.
    """
    if node_count < 2:
        raise ValueError("pair context requires at least two carrier sites")
    if (
        edge_source.shape != edge_target.shape
        or edge_source.ndim != 1
        or edge_rbf.ndim != 2
        or edge_rbf.shape[0] != edge_source.numel()
        or edge_rbf.shape[1] < 2
    ):
        raise ValueError("complete-pair context inputs have incompatible shapes")
    if edge_source.numel() != node_count * (node_count - 1):
        raise ValueError("pair context requires the complete directed non-self graph")
    if (
        int(edge_source.min()) < 0
        or int(edge_target.min()) < 0
        or int(edge_source.max()) >= node_count
        or int(edge_target.max()) >= node_count
        or bool((edge_source == edge_target).any())
        or not torch.isfinite(edge_rbf).all()
    ):
        raise ValueError("complete-pair context contains an invalid edge")
    encoded = edge_source * node_count + edge_target
    if torch.unique(encoded).numel() != edge_source.numel():
        raise ValueError("complete-pair context contains duplicate edges")
    channels = edge_rbf.shape[1]
    dense = edge_rbf.new_zeros(node_count, node_count, channels)
    dense[edge_source, edge_target] = edge_rbf
    context = torch.einsum("ikr,jks->ijrs", dense, dense)
    context = 0.5 * (context + context.transpose(-1, -2))
    upper = torch.triu_indices(channels, channels, device=edge_rbf.device)
    return torch.cat(
        (
            edge_rbf,
            context[edge_source, edge_target][:, upper[0], upper[1]],
        ),
        dim=1,
    )
