"""Leakage-free P1 blueprints for tensor-free substrate qualification."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def trace_free_projector(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Kelvin-coordinate projector onto symmetric trace-free matrices."""
    identity = torch.eye(6, dtype=dtype, device=device)
    trace = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    return identity - torch.outer(trace, trace) / 3.0


@dataclass(frozen=True)
class P1BlueprintBatch:
    """Minimal valid blueprint with every generated site asymmetric.

    S1a uses P1 deliberately: it qualifies the joint generative substrate
    without reading a paired target space group or Wyckoff labeling.  This is
    not a replacement for the separately planned 230-space-group sampler.
    """

    node_counts: torch.Tensor
    batch: torch.Tensor
    shape_projector: torch.Tensor
    fractional_to_cartesian: torch.Tensor

    @classmethod
    def from_counts(
        cls,
        node_counts: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "P1BlueprintBatch":
        selected_device = torch.device(device) if device is not None else node_counts.device
        counts = node_counts.to(device=selected_device, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 1).any()):
            raise ValueError("P1 blueprint node counts must be a nonempty positive vector")
        graphs = counts.numel()
        graph_ids = torch.arange(graphs, device=selected_device)
        batch = torch.repeat_interleave(graph_ids, counts)
        projector = trace_free_projector(dtype=dtype, device=selected_device)
        chart = torch.eye(3, dtype=dtype, device=selected_device)
        return cls(
            node_counts=counts,
            batch=batch,
            shape_projector=projector.expand(graphs, -1, -1).clone(),
            fractional_to_cartesian=chart.expand(graphs, -1, -1).clone(),
        )


class EmpiricalNodeCountPrior:
    """Categorical node-count prior fitted only to training split counts."""

    def __init__(self, support: torch.Tensor, probabilities: torch.Tensor) -> None:
        if support.ndim != 1 or probabilities.shape != support.shape or support.numel() < 1:
            raise ValueError("node-count support and probabilities must be equal nonempty vectors")
        if support.dtype != torch.long or bool((support < 1).any()):
            raise ValueError("node-count support must contain positive integers")
        if bool((probabilities < 0).any()) or not torch.isfinite(probabilities).all():
            raise ValueError("node-count probabilities must be finite and nonnegative")
        total = probabilities.sum()
        if float(total) <= 0.0:
            raise ValueError("node-count probabilities must have positive mass")
        self.support = support.detach().cpu()
        self.probabilities = (probabilities / total).detach().cpu()

    @classmethod
    def fit(cls, node_counts: torch.Tensor) -> "EmpiricalNodeCountPrior":
        counts = node_counts.detach().to(device="cpu", dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 1).any()):
            raise ValueError("training node counts must be a nonempty positive vector")
        support, frequency = torch.unique(counts, sorted=True, return_counts=True)
        return cls(support, frequency.to(torch.float64))

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("sample count must be positive")
        indices = torch.multinomial(self.probabilities, count, replacement=True, generator=generator)
        return self.support[indices].to(device=device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"support": self.support.clone(), "probabilities": self.probabilities.clone()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "EmpiricalNodeCountPrior":
        return cls(state["support"], state["probabilities"])
