"""Training-only quotient couplings for periodic crystal flow paths.

The sampler never receives a target structure.  These utilities only choose a
low-cost correspondence between an i.i.d. source state and the paired training
endpoint, which is a valid coupling for conditional flow matching and removes
the arbitrary CIF atom-order supervision from the vector-field target.
"""

from __future__ import annotations

import torch

from .manifold import torus_logmap


def periodic_assignment(
    source_frac: torch.Tensor,
    target_frac: torch.Tensor,
    *,
    source_types: torch.Tensor | None = None,
    target_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map every source atom to one target atom by minimum torus transport cost.

    If types are supplied, assignment is carried out independently in every
    species.  This is used only when atom types are fixed (A5 geometry-only);
    joint generation uses the all-atom form because source type probabilities
    are noise at the beginning of the path.
    """
    if source_frac.ndim != 2 or source_frac.shape[-1] != 3:
        raise ValueError("source_frac must have shape [atoms, 3]")
    if target_frac.shape != source_frac.shape:
        raise ValueError("source and target coordinate shapes must agree")
    if (source_types is None) != (target_types is None):
        raise ValueError("source_types and target_types must either both be set or both be absent")
    atoms = source_frac.shape[0]
    assignment = torch.empty(atoms, dtype=torch.long, device=source_frac.device)
    if source_types is None:
        groups = [(torch.arange(atoms, device=source_frac.device), torch.arange(atoms, device=source_frac.device))]
    else:
        if source_types.shape != (atoms,) or target_types.shape != (atoms,):
            raise ValueError("type arrays must have one entry per atom")
        if not torch.equal(torch.sort(source_types).values, torch.sort(target_types).values):
            raise ValueError("typewise matching requires equal source/target compositions")
        groups = []
        for element in torch.unique(source_types, sorted=True):
            source = torch.nonzero(source_types == element, as_tuple=False).flatten()
            target = torch.nonzero(target_types == element, as_tuple=False).flatten()
            groups.append((source, target))
    # Scipy's Hungarian method is exact and deterministic for the tiny
    # pre-registered A5 panel (four atoms). The cost is detached: matching is a
    # path coupling, not a differentiable model layer.
    from scipy.optimize import linear_sum_assignment

    for source, target in groups:
        displacement = torus_logmap(
            source_frac[source].unsqueeze(1), target_frac[target].unsqueeze(0)
        )
        cost = displacement.square().sum(dim=-1).detach().cpu().numpy()
        row, column = linear_sum_assignment(cost)
        assignment[source[torch.as_tensor(row, device=source.device)]] = target[
            torch.as_tensor(column, device=target.device)
        ]
    return assignment


def periodic_assignment_cost(source_frac: torch.Tensor, target_frac: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Mean squared periodic displacement under a source-to-target assignment."""
    if assignment.shape != (source_frac.shape[0],):
        raise ValueError("assignment must have one target index per source atom")
    return torus_logmap(source_frac, target_frac[assignment]).square().sum(dim=-1).mean()


def remove_graphwise_translation(velocity: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    """Project periodic-coordinate tangent vectors to the no-global-drift quotient."""
    from torch_geometric.utils import scatter

    mean = scatter(velocity, batch, dim=0, dim_size=graphs, reduce="mean")
    return velocity - mean[batch]


def translation_aligned_torus_rms(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMS modulo one common fractional translation for one equal-size structure."""
    if value.shape != target.shape or value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError("value and target must both have shape [atoms, 3]")
    displacement = torus_logmap(value, target)
    centered = displacement - displacement.mean(dim=0, keepdim=True)
    return centered.square().mean().sqrt()
