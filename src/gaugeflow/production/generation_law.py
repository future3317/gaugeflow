"""Closed probability interfaces for the GaugeFlow crystal state.

The unambiguous parent-conditioned order is ``N -> C -> P -> G -> A -> D``.
``P`` is a discrete parent blueprint, ``G`` is its species-free reference
geometry, ``A`` is the count-exact occupation, and ``D`` contains continuous
strain/mode/residual coordinates.  In the parent stratum the terminal
``(L,F)`` is the deterministic reconstruction ``T(P,G,A,D)`` and is not sampled
again.  The explicitly modelled flexible/P1 stratum instead samples ``(L,F)``
directly after ``A``.  Flexible is a genuine state in ``p(P|N,C)``, not a
runtime error fallback.

The factorization is an audit interface rather than a claim that production
must be implemented as disconnected networks.  A heterogeneous joint field
may implement the same conditionals, but every sampled child must close on the
exact shared variables ``(N,C,A,L,F)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


@dataclass(frozen=True)
class ParentDeltaNodeCountLaw:
    """Exact ``p(N | parent)=delta[N=sum Wyckoff multiplicities]``."""

    parent_atom_count: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.parent_atom_count.ndim != 1
            or self.parent_atom_count.dtype != torch.long
            or bool((self.parent_atom_count < 1).any())
        ):
            raise ValueError("parent atom counts must be a positive int64 vector")

    def log_prob(self, node_counts: torch.Tensor) -> torch.Tensor:
        if node_counts.shape != self.parent_atom_count.shape or node_counts.dtype != torch.long:
            raise ValueError("node counts do not align with the sampled parent")
        zeros = torch.zeros(node_counts.shape, device=node_counts.device, dtype=torch.float32)
        return torch.where(node_counts == self.parent_atom_count.to(node_counts.device), zeros, -torch.inf)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.parent_atom_count.clone(), torch.zeros(
            self.parent_atom_count.shape,
            dtype=torch.float32,
            device=self.parent_atom_count.device,
        )


@dataclass(frozen=True)
class CarrierSelectionSample:
    """One exact draw from support-masked ``p(P | N,C)``."""

    index: torch.Tensor
    log_probability: torch.Tensor


class SupportedCarrierSelectionLaw:
    """Normalized carrier categorical law with an explicit universal stratum.

    ``feasible[g,p]`` must be computed from target-independent carrier data and
    the sampled ``(N,C)``.  Candidate ``universal_candidate_index`` represents
    the flexible/P1 stratum and must be feasible for every graph.  Consequently
    the support is never empty and sampling needs no rejection or composition
    resampling.
    """

    def __init__(self, *, universal_candidate_index: int = 0) -> None:
        if universal_candidate_index < 0:
            raise ValueError("universal carrier index must be nonnegative")
        self.universal_candidate_index = universal_candidate_index

    def log_probabilities(
        self,
        logits: torch.Tensor,
        feasible: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or feasible.shape != logits.shape or feasible.dtype != torch.bool:
            raise ValueError("carrier logits and feasibility mask must have shape [graphs,candidates]")
        if logits.shape[1] <= self.universal_candidate_index:
            raise ValueError("universal carrier index lies outside the candidate axis")
        if not torch.isfinite(logits).all():
            raise ValueError("carrier logits must be finite")
        if not bool(feasible[:, self.universal_candidate_index].all()):
            raise ValueError("the declared flexible carrier must be universally feasible")
        masked = logits.masked_fill(~feasible, -torch.inf)
        return torch.log_softmax(masked, dim=1)

    def log_prob(
        self,
        logits: torch.Tensor,
        feasible: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        if index.shape != (logits.shape[0],) or index.dtype != torch.long:
            raise ValueError("carrier choice must be one int64 index per graph")
        if bool(((index < 0) | (index >= logits.shape[1])).any()):
            raise ValueError("carrier choice lies outside the candidate axis")
        return self.log_probabilities(logits, feasible).gather(1, index.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def sample(
        self,
        logits: torch.Tensor,
        feasible: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> CarrierSelectionSample:
        log_probability = self.log_probabilities(logits, feasible)
        index = torch.multinomial(
            log_probability.exp(),
            1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
        if not bool(feasible.gather(1, index.unsqueeze(1)).all()):
            raise RuntimeError("support-masked carrier sampler selected an infeasible state")
        return CarrierSelectionSample(
            index=index,
            log_probability=log_probability.gather(1, index.unsqueeze(1)).squeeze(1),
        )


@dataclass(frozen=True)
class CrystalGenerationState:
    """One packed sample whose five random objects share an exact row grain."""

    node_count: torch.Tensor
    composition_counts: torch.Tensor
    assignment: torch.Tensor
    batch: torch.Tensor
    lattice: torch.Tensor
    fractional_coordinates: torch.Tensor

    def validate(self, *, vocabulary_size: int = CHEMICAL_ELEMENT_COUNT) -> None:
        graphs = self.node_count.numel()
        if self.node_count.ndim != 1 or self.node_count.dtype != torch.long:
            raise ValueError("N must be a rank-one int64 tensor")
        if self.composition_counts.shape != (graphs, vocabulary_size):
            raise ValueError("C must have shape [graphs,vocabulary]")
        if self.composition_counts.dtype != torch.long or bool((self.composition_counts < 0).any()):
            raise ValueError("C must contain nonnegative integer counts")
        if not torch.equal(self.composition_counts.sum(dim=1), self.node_count):
            raise ValueError("C does not close on N")
        nodes = int(self.node_count.sum())
        if self.assignment.shape != (nodes,) or self.assignment.dtype != torch.long:
            raise ValueError("A must contain one int64 species per site")
        if self.batch.shape != (nodes,) or self.batch.dtype != torch.long:
            raise ValueError("packed graph index does not align with A")
        if bool(((self.assignment < 0) | (self.assignment >= vocabulary_size)).any()):
            raise ValueError("A contains an out-of-vocabulary species")
        expected_batch = torch.repeat_interleave(
            torch.arange(graphs, device=self.node_count.device), self.node_count
        )
        if not torch.equal(self.batch, expected_batch.to(self.batch.device)):
            raise ValueError("packed sites are not graph contiguous")
        flat = self.batch * vocabulary_size + self.assignment
        observed = torch.bincount(flat, minlength=graphs * vocabulary_size).reshape(
            graphs, vocabulary_size
        )
        if not torch.equal(observed, self.composition_counts):
            raise ValueError("A does not preserve C exactly")
        if self.lattice.shape != (graphs, 3, 3) or not torch.isfinite(self.lattice).all():
            raise ValueError("L must be a finite [graphs,3,3] tensor")
        if bool((torch.linalg.det(self.lattice) <= 0).any()):
            raise ValueError("L must be right handed with positive volume")
        if self.fractional_coordinates.shape != (nodes, 3):
            raise ValueError("F must contain one three-vector per site")
        if not torch.isfinite(self.fractional_coordinates).all():
            raise ValueError("F contains non-finite coordinates")


@dataclass(frozen=True)
class FactorizedGenerationLogProbability:
    """Auditable chain-rule decomposition of ``log p(N,C,A,L,F)``."""

    node_count: torch.Tensor
    composition: torch.Tensor
    assignment: torch.Tensor
    lattice: torch.Tensor
    coordinates: torch.Tensor
    carrier: torch.Tensor | None = None

    @property
    def total(self) -> torch.Tensor:
        values = (
            self.node_count,
            self.composition,
            self.assignment,
            self.lattice,
            self.coordinates,
        )
        if any(value.shape != self.node_count.shape for value in values):
            raise ValueError("all log-probability terms must contain one value per graph")
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("qualified log-probability terms must be finite")
        total = sum(values[1:], values[0])
        if self.carrier is not None:
            if self.carrier.shape != self.node_count.shape or not torch.isfinite(self.carrier).all():
                raise ValueError("carrier log probability must contain one finite value per graph")
            total = total + self.carrier
        return total
