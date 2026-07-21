"""Exact sparse stochastic composition state for ordered crystals.

The scientific object is an unordered element-count multiset.  Increasing
element-token order is only its unique serialization; it is unrelated to CIF
row order or site assignment.  The decoder is autoregressive over at most a
few distinct species while every conditional categorical distribution is
exactly normalized over mathematically valid actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


@dataclass(frozen=True)
class SparseCompositionState:
    """Canonical padded species-count serialization of a graph composition."""

    species: torch.Tensor
    counts: torch.Tensor
    length: torch.Tensor
    node_count: torch.Tensor

    @property
    def graphs(self) -> int:
        return int(self.species.shape[0])

    @property
    def maximum_species(self) -> int:
        return int(self.species.shape[1])

    def validate(
        self,
        *,
        vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
        maximum_atoms: int | None = None,
    ) -> None:
        if self.species.ndim != 2 or self.counts.shape != self.species.shape:
            raise ValueError("species and counts must have the same [graphs, positions] shape")
        if self.length.shape != (self.graphs,) or self.node_count.shape != (self.graphs,):
            raise ValueError("length and node_count must contain one value per graph")
        if self.species.dtype != torch.long or self.counts.dtype != torch.long:
            raise ValueError("species and counts must use torch.long")
        if self.length.dtype != torch.long or self.node_count.dtype != torch.long:
            raise ValueError("length and node_count must use torch.long")
        if self.species.device != self.counts.device:
            raise ValueError("composition tensors must share a device")
        if self.length.device != self.species.device or self.node_count.device != self.species.device:
            raise ValueError("composition tensors must share a device")
        if self.graphs < 1 or self.maximum_species < 1:
            raise ValueError("composition state must contain at least one graph and position")
        if bool(((self.length < 1) | (self.length > self.maximum_species)).any()):
            raise ValueError("composition length is outside the padded support")
        if bool((self.node_count < 1).any()):
            raise ValueError("node_count must be positive")
        if maximum_atoms is not None and bool((self.node_count > maximum_atoms).any()):
            raise ValueError("node_count exceeds the configured maximum")

        positions = torch.arange(self.maximum_species, device=self.species.device)
        active = positions.unsqueeze(0) < self.length.unsqueeze(1)
        if bool(((self.species < 0) | (self.species >= vocabulary_size))[active].any()):
            raise ValueError("active species token is outside the vocabulary")
        if bool((self.counts[active] < 1).any()):
            raise ValueError("active species counts must be positive")
        if bool((self.species[~active] != -1).any()) or bool((self.counts[~active] != 0).any()):
            raise ValueError("inactive composition positions must use species=-1 and count=0")
        adjacent = active[:, 1:] & active[:, :-1]
        if bool((self.species[:, 1:][adjacent] <= self.species[:, :-1][adjacent]).any()):
            raise ValueError("active species tokens must be strictly increasing")
        if not torch.equal((self.counts * active).sum(dim=1), self.node_count):
            raise ValueError("composition counts do not sum to node_count")

    def to_dense(self, vocabulary_size: int = CHEMICAL_ELEMENT_COUNT) -> torch.Tensor:
        """Return the exact dense integer histogram without padding semantics."""
        self.validate(vocabulary_size=vocabulary_size)
        positions = torch.arange(self.maximum_species, device=self.species.device)
        active = positions.unsqueeze(0) < self.length.unsqueeze(1)
        dense = torch.zeros(
            (self.graphs, vocabulary_size),
            dtype=torch.long,
            device=self.species.device,
        )
        dense.scatter_add_(1, self.species.clamp_min(0), self.counts * active)
        return dense

    def to(self, device: torch.device | str) -> SparseCompositionState:
        """Move the compact state without changing its canonical serialization."""
        return SparseCompositionState(
            self.species.to(device),
            self.counts.to(device),
            self.length.to(device),
            self.node_count.to(device),
        )

    def index_select(self, index: torch.Tensor) -> SparseCompositionState:
        """Select graph rows while preserving the fixed sparse width."""
        if index.ndim != 1 or index.dtype != torch.long:
            raise ValueError("composition index must be a one-dimensional long tensor")
        return SparseCompositionState(
            self.species.index_select(0, index),
            self.counts.index_select(0, index),
            self.length.index_select(0, index),
            self.node_count.index_select(0, index),
        )

    @classmethod
    def from_dense(
        cls,
        dense_counts: torch.Tensor,
        *,
        maximum_species: int,
    ) -> SparseCompositionState:
        """Encode dense nonnegative integer counts in canonical token order."""
        if dense_counts.ndim != 2 or dense_counts.dtype != torch.long:
            raise ValueError("dense_counts must be a [graphs, vocabulary] torch.long tensor")
        if maximum_species < 1 or bool((dense_counts < 0).any()):
            raise ValueError("maximum_species must be positive and counts nonnegative")
        active = dense_counts > 0
        length = active.sum(dim=1)
        if bool((length < 1).any()) or bool((length > maximum_species).any()):
            raise ValueError("dense composition is outside sparse support")
        graphs = dense_counts.shape[0]
        species = torch.full(
            (graphs, maximum_species),
            -1,
            dtype=torch.long,
            device=dense_counts.device,
        )
        counts = torch.zeros_like(species)
        graph, token = torch.nonzero(active, as_tuple=True)
        position = active.long().cumsum(dim=1)[graph, token] - 1
        species[graph, position] = token
        counts[graph, position] = dense_counts[graph, token]
        state = cls(
            species=species,
            counts=counts,
            length=length.long(),
            node_count=dense_counts.sum(dim=1).long(),
        )
        state.validate(vocabulary_size=dense_counts.shape[1])
        return state

    @classmethod
    def from_element_tokens(
        cls,
        element_tokens: torch.Tensor,
        batch: torch.Tensor,
        *,
        graphs: int,
        maximum_species: int,
        vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
    ) -> SparseCompositionState:
        """Encode a graph-contiguous site-token list without using site order."""
        if element_tokens.ndim != 1 or batch.shape != element_tokens.shape:
            raise ValueError("element_tokens and batch must be one-dimensional and aligned")
        if element_tokens.dtype != torch.long or batch.dtype != torch.long:
            raise ValueError("element_tokens and batch must use torch.long")
        if graphs < 1 or element_tokens.numel() < graphs:
            raise ValueError("every graph must contain at least one element token")
        if bool(((element_tokens < 0) | (element_tokens >= vocabulary_size)).any()):
            raise ValueError("element token is outside the vocabulary")
        if bool(((batch < 0) | (batch >= graphs)).any()):
            raise ValueError("batch index is outside graph support")
        flat = batch * vocabulary_size + element_tokens
        dense = torch.bincount(flat, minlength=graphs * vocabulary_size).reshape(graphs, vocabulary_size)
        return cls.from_dense(dense.long(), maximum_species=maximum_species)

    @classmethod
    def from_packed_element_tokens(
        cls,
        element_tokens: torch.Tensor,
        offsets: torch.Tensor,
        *,
        maximum_species: int,
        vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
    ) -> SparseCompositionState:
        """Encode a complete packed split without allocating ``graphs*K`` counts."""
        if element_tokens.ndim != 1 or offsets.ndim != 1 or offsets.dtype != torch.long:
            raise ValueError("packed tokens/offsets must be one-dimensional with long offsets")
        if element_tokens.dtype != torch.long:
            raise ValueError("packed element tokens must use torch.long")
        graphs = offsets.numel() - 1
        if graphs < 1 or int(offsets[0]) != 0 or int(offsets[-1]) != element_tokens.numel():
            raise ValueError("packed offsets do not span the token array")
        node_count = offsets.diff()
        if bool((node_count < 1).any()):
            raise ValueError("every packed graph must contain at least one token")
        if bool(((element_tokens < 0) | (element_tokens >= vocabulary_size)).any()):
            raise ValueError("packed element token is outside the vocabulary")
        batch = torch.repeat_interleave(torch.arange(graphs, device=element_tokens.device), node_count)
        flat = batch * vocabulary_size + element_tokens
        unique, pair_count = torch.unique(flat, sorted=True, return_counts=True)
        graph = torch.div(unique, vocabulary_size, rounding_mode="floor")
        species_token = unique.remainder(vocabulary_size)
        length = torch.bincount(graph, minlength=graphs)
        if bool((length > maximum_species).any()):
            raise ValueError("packed composition exceeds maximum_species")
        starts = torch.cat((length.new_zeros(1), length.cumsum(dim=0)[:-1]))
        position = torch.arange(unique.numel(), device=unique.device) - torch.repeat_interleave(starts, length)
        species = torch.full(
            (graphs, maximum_species),
            -1,
            dtype=torch.long,
            device=element_tokens.device,
        )
        counts = torch.zeros_like(species)
        species[graph, position] = species_token
        counts[graph, position] = pair_count
        state = cls(species, counts, length.long(), node_count.long())
        state.validate(vocabulary_size=vocabulary_size)
        return state


@dataclass(frozen=True)
class SparseCompositionLogProbability:
    """Per-graph normalized log-probability decomposition."""

    total: torch.Tensor
    support: torch.Tensor
    species: torch.Tensor
    counts: torch.Tensor


@dataclass(frozen=True)
class SparseCompositionSample:
    """One exact composition sample and its joint log probability."""

    state: SparseCompositionState
    log_probability: torch.Tensor


@dataclass(frozen=True)
class IntegerPartitionCatalogue:
    """Finite catalogue of positive integer partitions for bounded crystals."""

    counts: torch.Tensor
    length: torch.Tensor
    node_count: torch.Tensor
    key: torch.Tensor
    sorted_key: torch.Tensor
    sorted_index: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.counts.shape[0])

    @property
    def maximum_species(self) -> int:
        return int(self.counts.shape[1])

    def to(self, device: torch.device | str) -> IntegerPartitionCatalogue:
        return IntegerPartitionCatalogue(
            counts=self.counts.to(device),
            length=self.length.to(device),
            node_count=self.node_count.to(device),
            key=self.key.to(device),
            sorted_key=self.sorted_key.to(device),
            sorted_index=self.sorted_index.to(device),
        )

    @classmethod
    def build(
        cls,
        *,
        maximum_atoms: int,
        maximum_species: int,
    ) -> IntegerPartitionCatalogue:
        if maximum_atoms < 1 or maximum_species < 1:
            raise ValueError("integer-partition bounds must be positive")

        def recurse(
            remaining: int,
            maximum: int,
            prefix: tuple[int, ...],
        ) -> list[tuple[int, ...]]:
            if remaining == 0:
                return [prefix]
            if len(prefix) == maximum_species:
                return []
            output: list[tuple[int, ...]] = []
            for value in range(min(remaining, maximum), 0, -1):
                output.extend(recurse(remaining - value, value, prefix + (value,)))
            return output

        partitions = [partition for atoms in range(1, maximum_atoms + 1) for partition in recurse(atoms, atoms, ())]
        counts = torch.zeros((len(partitions), maximum_species), dtype=torch.long)
        for row, partition in enumerate(partitions):
            counts[row, : len(partition)] = torch.tensor(partition, dtype=torch.long)
        length = (counts > 0).sum(dim=1)
        node_count = counts.sum(dim=1)
        key = _integer_partition_key(counts, maximum_atoms=maximum_atoms)
        sorted_key, sorted_index = torch.sort(key)
        if torch.unique_consecutive(sorted_key).numel() != len(partitions):
            raise RuntimeError("integer-partition catalogue key is not injective")
        return cls(counts, length, node_count, key, sorted_key, sorted_index)

    def encode(self, state: SparseCompositionState, *, maximum_atoms: int) -> torch.Tensor:
        ordered = torch.sort(state.counts, dim=1, descending=True).values
        key = _integer_partition_key(ordered, maximum_atoms=maximum_atoms)
        location = torch.searchsorted(self.sorted_key.to(key.device), key)
        if bool((location >= self.sorted_key.numel()).any()):
            raise ValueError("composition partition is outside the catalogue")
        matched = self.sorted_key.to(key.device).index_select(0, location)
        if not torch.equal(matched, key):
            raise ValueError("composition partition is outside the catalogue")
        return self.sorted_index.to(key.device).index_select(0, location)


def _integer_partition_key(counts: torch.Tensor, *, maximum_atoms: int) -> torch.Tensor:
    if counts.ndim != 2 or counts.dtype != torch.long:
        raise ValueError("partition counts must be a two-dimensional long tensor")
    base = maximum_atoms + 1
    powers = base ** torch.arange(counts.shape[1], dtype=torch.long, device=counts.device)
    return (counts * powers.unsqueeze(0)).sum(dim=1)


def fit_integer_partition_log_prior(
    state: SparseCompositionState,
    catalogue: IntegerPartitionCatalogue,
    *,
    maximum_atoms: int,
    smoothing: float,
) -> torch.Tensor:
    """Fit a train-only empirical ``p(lambda | N)`` with symmetric smoothing."""
    if smoothing <= 0.0:
        raise ValueError("partition smoothing must be positive")
    index = catalogue.encode(state, maximum_atoms=maximum_atoms)
    observed = torch.bincount(index.cpu(), minlength=catalogue.size).double()
    log_probability = torch.full((catalogue.size,), -torch.inf, dtype=torch.float64)
    for atoms in range(1, maximum_atoms + 1):
        valid = catalogue.node_count == atoms
        weights = observed[valid] + smoothing
        if weights.numel() == 0:
            raise RuntimeError("integer-partition catalogue has an empty atom-count support")
        log_probability[valid] = weights.log() - weights.sum().log()
    return log_probability


class StoichiometryFirstCompositionModel(nn.Module):
    """Exact composition law factored as partition first, then distinct species.

    A composition is serialized by decreasing count.  Species tokens are used
    only to break ties between equal counts, where they are strictly increasing.
    This gives every unordered element-count multiset exactly one sequence and
    avoids both target-composition input and identical-count multiplicity bias.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        partition_log_prior: torch.Tensor,
        *,
        maximum_atoms: int = 20,
        maximum_species: int = 7,
        vocabulary_size: int = CHEMICAL_ELEMENT_COUNT,
        active_vocabulary_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if context_dim < 1 or hidden_dim < 1:
            raise ValueError("context and hidden dimensions must be positive")
        catalogue = IntegerPartitionCatalogue.build(
            maximum_atoms=maximum_atoms,
            maximum_species=maximum_species,
        )
        if partition_log_prior.shape != (catalogue.size,):
            raise ValueError("partition_log_prior does not match the exact catalogue")
        if not torch.isfinite(partition_log_prior).all():
            raise ValueError("partition_log_prior must be finite on every valid partition")
        if active_vocabulary_mask is None:
            active_vocabulary_mask = torch.ones(vocabulary_size, dtype=torch.bool)
        if active_vocabulary_mask.shape != (vocabulary_size,) or active_vocabulary_mask.dtype != torch.bool:
            raise ValueError("active_vocabulary_mask must be a boolean vocabulary vector")
        if int(active_vocabulary_mask.sum()) < maximum_species:
            raise ValueError("active vocabulary is smaller than maximum_species")
        for atoms in range(1, maximum_atoms + 1):
            valid = catalogue.node_count == atoms
            normalization = torch.logsumexp(partition_log_prior[valid].double(), dim=0)
            # The qualified checkpoint is stored in FP32 even though the
            # empirical prior is fitted in FP64.  Keep a strict FP64 contract
            # while allowing the unavoidable FP32 serialization rounding when
            # reconstructing the frozen runtime model.
            tolerance = 1.0e-10 if partition_log_prior.dtype == torch.float64 else 2.0e-6
            if not torch.allclose(normalization, torch.zeros_like(normalization), atol=tolerance):
                raise ValueError("partition_log_prior is not normalized for every node count")

        self.context_dim = context_dim
        self.maximum_atoms = maximum_atoms
        self.maximum_species = maximum_species
        self.vocabulary_size = vocabulary_size
        self.register_buffer("partition_counts", catalogue.counts)
        self.register_buffer("partition_length", catalogue.length)
        self.register_buffer("partition_node_count", catalogue.node_count)
        self.register_buffer("partition_key", catalogue.key)
        self.register_buffer("partition_sorted_key", catalogue.sorted_key)
        self.register_buffer("partition_sorted_index", catalogue.sorted_index)
        self.register_buffer("partition_log_prior", partition_log_prior.clone())
        self.register_buffer("active_vocabulary_mask", active_vocabulary_mask.clone())

        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.node_count_embedding = nn.Embedding(maximum_atoms + 1, hidden_dim)
        self.species_embedding = nn.Embedding(vocabulary_size + 1, hidden_dim)
        self.count_embedding = nn.Embedding(maximum_atoms + 1, hidden_dim)
        self.position_embedding = nn.Embedding(maximum_species, hidden_dim)
        self.partition_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.species_head = nn.Linear(hidden_dim, vocabulary_size)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def _catalogue(self) -> IntegerPartitionCatalogue:
        return IntegerPartitionCatalogue(
            self.partition_counts,
            self.partition_length,
            self.partition_node_count,
            self.partition_key,
            self.partition_sorted_key,
            self.partition_sorted_index,
        )

    def _initial_hidden(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        partition_index: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape [graphs, context_dim]")
        if node_count.shape != (context.shape[0],) or partition_index.shape != node_count.shape:
            raise ValueError("node_count and partition_index must contain one value per graph")
        counts = self.partition_counts.index_select(0, partition_index)
        length = self.partition_length.index_select(0, partition_index)
        positions = torch.arange(self.maximum_species, device=context.device)
        active = positions.unsqueeze(0) < length.unsqueeze(1)
        partition_token = self.count_embedding(counts) + self.position_embedding(positions).unsqueeze(0)
        partition_summary = (self.partition_encoder(partition_token) * active.unsqueeze(2)).sum(dim=1) / length.to(
            context.dtype
        ).sqrt().unsqueeze(1)
        return torch.tanh(self.context_projection(context) + self.node_count_embedding(node_count) + partition_summary)

    def _partition_log_probability(self, node_count: torch.Tensor) -> torch.Tensor:
        valid = self.partition_node_count.unsqueeze(0) == node_count.unsqueeze(1)
        return self.partition_log_prior.unsqueeze(0).expand(node_count.shape[0], -1).masked_fill(~valid, -torch.inf)

    def count_first_order(self, state: SparseCompositionState) -> tuple[torch.Tensor, torch.Tensor]:
        """Return species/count pairs in the exact autoregressive serialization."""
        rank_key = state.counts * (self.vocabulary_size + 1) - state.species
        order = torch.argsort(rank_key, dim=1, descending=True, stable=True)
        return state.species.gather(1, order), state.counts.gather(1, order)

    def _species_validity(
        self,
        used: torch.Tensor,
        previous: torch.Tensor,
        same_count_as_previous: torch.Tensor,
        remaining_equal_counts: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        token = torch.arange(self.vocabulary_size, device=used.device)
        valid = self.active_vocabulary_mask.unsqueeze(0) & ~used
        valid = valid & (~same_count_as_previous.unsqueeze(1) | (token > previous.unsqueeze(1)))
        available_greater = (
            torch.flip(torch.cumsum(torch.flip(valid.long(), dims=(1,)), dim=1), dims=(1,)) - valid.long()
        )
        valid = valid & (available_greater >= remaining_equal_counts.unsqueeze(1))
        return _dummy_support_for_inactive(valid, active)

    def species_validity_by_slot(self, state: SparseCompositionState) -> torch.Tensor:
        """Return the exact legal categorical support for every count-rank slot."""
        state.validate(vocabulary_size=self.vocabulary_size, maximum_atoms=self.maximum_atoms)
        species, counts = self.count_first_order(state)
        return self._species_validity_by_ordered_state(species, counts, state.length)

    def _species_validity_by_ordered_state(
        self,
        species: torch.Tensor,
        counts: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(self.maximum_species, device=species.device)
        active = positions.unsqueeze(0) < length.unsqueeze(1)
        one_hot = torch.nn.functional.one_hot(species.clamp_min(0), num_classes=self.vocabulary_size).bool()
        one_hot = one_hot & active.unsqueeze(2)
        used = torch.cat(
            (
                torch.zeros_like(one_hot[:, :1]),
                one_hot[:, :-1].long().cumsum(dim=1).bool(),
            ),
            dim=1,
        )
        previous = torch.cat((species.new_full((species.shape[0], 1), -1), species[:, :-1]), dim=1)
        previous_count = torch.cat((counts.new_zeros((species.shape[0], 1)), counts[:, :-1]), dim=1)
        same_count = active & (positions.unsqueeze(0) > 0) & (counts == previous_count)
        remaining_equal = torch.zeros_like(counts)
        for position in range(self.maximum_species - 1):
            remaining_equal[:, position] = (counts[:, position + 1 :] == counts[:, position : position + 1]).sum(dim=1)
        return self._species_validity(
            used.reshape(-1, self.vocabulary_size),
            previous.reshape(-1),
            same_count.reshape(-1),
            remaining_equal.reshape(-1),
            active.reshape(-1),
        ).reshape(species.shape[0], self.maximum_species, self.vocabulary_size)

    def log_prob(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        state: SparseCompositionState,
    ) -> SparseCompositionLogProbability:
        state.validate(vocabulary_size=self.vocabulary_size, maximum_atoms=self.maximum_atoms)
        if state.graphs != context.shape[0] or not torch.equal(state.node_count, node_count):
            raise ValueError("composition state and context/node_count disagree")
        chosen = state.species.clamp_min(0)
        if bool((~self.active_vocabulary_mask[chosen])[state.species >= 0].any()):
            raise ValueError("composition uses an inactive vocabulary token")
        catalogue = self._catalogue()
        partition_index = catalogue.encode(state, maximum_atoms=self.maximum_atoms)
        partition_log = self._partition_log_probability(node_count).gather(1, partition_index.unsqueeze(1)).squeeze(1)
        selected = self._species_log_probability_by_slot(
            context,
            node_count,
            state,
            partition_index,
        )
        species_total = selected.sum(dim=1)
        return SparseCompositionLogProbability(
            total=partition_log + species_total,
            support=partition_log,
            species=species_total,
            counts=torch.zeros_like(partition_log),
        )

    def species_log_probability_by_slot(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        state: SparseCompositionState,
    ) -> torch.Tensor:
        """Return selected conditional species log probabilities by count-rank slot.

        Slots follow decreasing stoichiometric count with increasing element
        token used only to break equal-count ties.  Inactive padded slots are
        exactly zero.
        """
        state.validate(vocabulary_size=self.vocabulary_size, maximum_atoms=self.maximum_atoms)
        if state.graphs != context.shape[0] or not torch.equal(state.node_count, node_count):
            raise ValueError("composition state and context/node_count disagree")
        chosen = state.species.clamp_min(0)
        if bool((~self.active_vocabulary_mask[chosen])[state.species >= 0].any()):
            raise ValueError("composition uses an inactive vocabulary token")
        partition_index = self._catalogue().encode(state, maximum_atoms=self.maximum_atoms)
        return self._species_log_probability_by_slot(context, node_count, state, partition_index)

    def _species_log_probability_by_slot(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        state: SparseCompositionState,
        partition_index: torch.Tensor,
    ) -> torch.Tensor:
        species, counts = self.count_first_order(state)
        hidden = self._initial_hidden(context, node_count, partition_index)
        positions = torch.arange(self.maximum_species, device=state.species.device)
        active = positions.unsqueeze(0) < state.length.unsqueeze(1)
        recurrent_input = (
            self.species_embedding(torch.where(active, species, species.new_full((), self.vocabulary_size)))
            + self.count_embedding(counts)
            + self.position_embedding(positions).unsqueeze(0)
        )
        if self.maximum_species > 1:
            recurrent_output, _ = self.recurrent(recurrent_input[:, :-1], hidden.unsqueeze(0))
            head_hidden = torch.cat((hidden.unsqueeze(1), recurrent_output), dim=1)
        else:
            head_hidden = hidden.unsqueeze(1)
        slot_token = self.count_embedding(counts) + self.position_embedding(positions).unsqueeze(0)
        head_hidden = head_hidden + self.slot_projection(slot_token)

        valid = self._species_validity_by_ordered_state(species, counts, state.length)
        species_log = _masked_log_softmax(self.species_head(head_hidden), valid)
        selected = species_log.gather(2, species.clamp_min(0).unsqueeze(2)).squeeze(2)
        return torch.where(active, selected, torch.zeros_like(selected))

    @staticmethod
    def _draw(
        log_probability: torch.Tensor,
        *,
        mode: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if mode:
            return log_probability.argmax(dim=-1)
        return torch.multinomial(log_probability.float().exp(), num_samples=1, generator=generator).squeeze(1)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        node_count: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        mode: bool = False,
    ) -> SparseCompositionSample:
        partition_log = self._partition_log_probability(node_count)
        partition_index = self._draw(partition_log, mode=mode, generator=generator)
        log_probability = partition_log.gather(1, partition_index.unsqueeze(1)).squeeze(1)
        conditional = self.sample_species_given_partition(
            context,
            partition_index,
            generator=generator,
            mode=mode,
        )
        return SparseCompositionSample(
            state=conditional.state,
            log_probability=log_probability + conditional.log_probability,
        )

    @torch.no_grad()
    def sample_species_given_partition(
        self,
        context: torch.Tensor,
        partition_index: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        mode: bool = False,
    ) -> SparseCompositionSample:
        """Sample only the species law for an explicitly selected partition.

        This is the exact conditional kernel used by :meth:`sample`; it exists
        so diagnostics can hold the count partition fixed without duplicating
        the autoregressive mathematics.  It is not a production source of
        target composition: free sampling still draws the partition first.
        """
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape [graphs, context_dim]")
        if partition_index.shape != (context.shape[0],) or partition_index.dtype != torch.long:
            raise ValueError("partition_index must contain one long value per graph")
        if bool(((partition_index < 0) | (partition_index >= self.partition_counts.shape[0])).any()):
            raise ValueError("partition_index is outside the exact catalogue")
        node_count = self.partition_node_count.index_select(0, partition_index)
        counts = self.partition_counts.index_select(0, partition_index)
        length = self.partition_length.index_select(0, partition_index)
        hidden = self._initial_hidden(context, node_count, partition_index)
        graphs = context.shape[0]
        log_probability = torch.zeros(graphs, dtype=context.dtype, device=context.device)
        species = node_count.new_full((graphs, self.maximum_species), -1)
        used = torch.zeros((graphs, self.vocabulary_size), dtype=torch.bool, device=context.device)
        previous = node_count.new_full((graphs,), -1)
        previous_count = node_count.new_zeros((graphs,))
        for position in range(self.maximum_species):
            active = position < length
            current_count = counts[:, position]
            same_count = active & (position > 0) & (current_count == previous_count)
            if position + 1 < self.maximum_species:
                remaining_equal = (counts[:, position + 1 :] == current_count.unsqueeze(1)).sum(dim=1)
            else:
                remaining_equal = torch.zeros_like(node_count)
            valid = self._species_validity(used, previous, same_count, remaining_equal, active)
            slot_token = self.count_embedding(current_count) + self.position_embedding.weight[position].unsqueeze(0)
            species_log = _masked_log_softmax(self.species_head(hidden + self.slot_projection(slot_token)), valid)
            selected = self._draw(species_log, mode=mode, generator=generator)
            species[:, position] = torch.where(active, selected, species[:, position])
            log_probability = log_probability + species_log.gather(1, selected.unsqueeze(1)).squeeze(1) * active
            used.scatter_(1, selected.unsqueeze(1), active.unsqueeze(1))
            recurrent_input = (
                self.species_embedding(torch.where(active, selected, selected.new_full((), self.vocabulary_size)))
                + self.count_embedding(current_count)
                + self.position_embedding.weight[position].unsqueeze(0)
            )
            updated, _ = self.recurrent(recurrent_input.unsqueeze(1), hidden.unsqueeze(0))
            hidden = torch.where(active.unsqueeze(1), updated.squeeze(1), hidden)
            previous = torch.where(active, selected, previous)
            previous_count = torch.where(active, current_count, previous_count)

        species_order = torch.argsort(
            torch.where(species >= 0, species, species.new_full((), self.vocabulary_size)),
            dim=1,
        )
        canonical_species = species.gather(1, species_order)
        canonical_counts = counts.gather(1, species_order)
        state = SparseCompositionState(
            canonical_species,
            canonical_counts,
            length.long(),
            node_count,
        )
        state.validate(vocabulary_size=self.vocabulary_size, maximum_atoms=self.maximum_atoms)
        return SparseCompositionSample(state=state, log_probability=log_probability)


def _masked_log_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("categorical logits and validity mask must have matching shapes")
    if bool((~valid.any(dim=-1)).any()):
        raise ValueError("categorical support is empty")
    return torch.log_softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)


def _dummy_support_for_inactive(valid: torch.Tensor, needed: torch.Tensor) -> torch.Tensor:
    """Give inactive batch rows one harmless category for vectorized kernels."""
    if needed.shape != valid.shape[:-1]:
        raise ValueError("active-row mask does not match categorical batch shape")
    selected = torch.where(needed.unsqueeze(-1), valid, torch.zeros_like(valid))
    selected[..., 0] = selected[..., 0] | ~needed
    return selected
