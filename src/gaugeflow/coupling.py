"""Training-only quotient couplings for periodic crystal flow paths.

The sampler never receives a target structure.  These utilities only choose a
low-cost correspondence between an i.i.d. source state and the paired training
endpoint, which is a valid coupling for conditional flow matching and removes
the arbitrary CIF atom-order supervision from the vector-field target.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch

from .manifold import torus_logmap


@dataclass
class FixedLiftCoupling:
    """One immutable source-endpoint coupling on the fractional universal cover."""

    assignment: torch.Tensor
    integer_lift: torch.Tensor
    translation: torch.Tensor
    endpoint_lift: torch.Tensor
    velocity: torch.Tensor
    cost: torch.Tensor
    second_cost: torch.Tensor


def _type_preserving_assignments(source_types: torch.Tensor, target_types: torch.Tensor) -> list[torch.Tensor]:
    if source_types.ndim != 1 or target_types.shape != source_types.shape:
        raise ValueError("type arrays must have the same one-dimensional shape")
    if not torch.equal(torch.sort(source_types).values, torch.sort(target_types).values):
        raise ValueError("type-preserving coupling requires equal compositions")
    groups = []
    for element in torch.unique(source_types, sorted=True):
        source = torch.nonzero(source_types == element, as_tuple=False).flatten().tolist()
        target = torch.nonzero(target_types == element, as_tuple=False).flatten().tolist()
        groups.append((source, list(itertools.permutations(target))))
    assignments = []
    for choices in itertools.product(*(permutations for _, permutations in groups)):
        assignment = torch.empty_like(source_types)
        for (source, _), target in zip(groups, choices):
            assignment[torch.tensor(source, device=source_types.device)] = torch.tensor(target, device=source_types.device)
        assignments.append(assignment)
    return assignments


def fixed_lift_coupling(
    source_frac: torch.Tensor,
    target_frac: torch.Tensor,
    lattice: torch.Tensor,
    *,
    source_types: torch.Tensor,
    target_types: torch.Tensor,
) -> FixedLiftCoupling:
    """Solve and freeze ``(pi, K, tau)`` for a tiny periodic endpoint pair.

    This exact component is intentionally limited to at most four sites.  It
    enumerates unique type-preserving assignments and every relative integer
    lift inside a rigorous finite optimum bound, fixes the redundant integer
    gauge with ``K[0] = 0``, and analytically eliminates the common translation.  The returned endpoint
    and velocity live on the universal cover; callers must not re-run a torus
    Log, assignment, or translation alignment along the path.
    """
    if source_frac.ndim != 2 or source_frac.shape[-1] != 3 or target_frac.shape != source_frac.shape:
        raise ValueError("source_frac and target_frac must have shape [sites, 3]")
    sites = source_frac.shape[0]
    if not 1 <= sites <= 4:
        raise ValueError("exact fixed-lift coupling is registered only for one to four sites")
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape [3, 3]")
    assignments = _type_preserving_assignments(source_types, target_types)
    costs = []
    lifts = []
    translations = []
    for assignment in assignments:
        difference = source_frac - target_frac[assignment]
        if sites == 1:
            integer_lifts = torch.zeros((1, 1, 3), dtype=torch.long, device=source_frac.device)
        else:
            relative = difference[1:] - difference[:1]
            feasible = torch.round(relative).to(torch.long)
            seeds = [feasible]
            for node in range(sites - 1):
                for dim in range(3):
                    for sign in (-1, 1):
                        neighbor = feasible.clone()
                        neighbor[node, dim] += sign
                        seeds.append(neighbor)
            feasible_relative = torch.stack(seeds)
            feasible_anchor = torch.zeros(
                (feasible_relative.shape[0], 1, 3), dtype=torch.long, device=source_frac.device
            )
            feasible_lifts = torch.cat((feasible_anchor, feasible_relative), dim=1)
            feasible_unaligned = difference[None] - feasible_lifts.to(source_frac.dtype)
            feasible_residual = feasible_unaligned - feasible_unaligned.mean(dim=1, keepdim=True)
            feasible_cost = (feasible_residual @ lattice).square().sum(dim=(-1, -2))
            second_feasible_cost = torch.topk(feasible_cost, k=2, largest=False).values[1]
            # Exact finite radius from
            # sum_i ||r_i-r_bar||_L^2 = (1/n) sum_{i<j} ||r_i-r_j||_L^2.
            # Any relative integer lift outside this ball exceeds a known
            # upper bound on the second-best cost and cannot enter the top two.
            sigma_min = torch.linalg.svdvals(lattice).min().clamp_min(1.0e-12)
            radius = float((sites * second_feasible_cost).clamp_min(0.0).sqrt() / sigma_min + 1.0e-6)
            candidate_lists = []
            for node in range(sites - 1):
                center = relative[node]
                lower = torch.floor(center - radius).to(torch.long)
                upper = torch.ceil(center + radius).to(torch.long)
                axes = [torch.arange(int(lower[dim]), int(upper[dim]) + 1, device=source_frac.device) for dim in range(3)]
                vectors = torch.cartesian_prod(*axes)
                if vectors.ndim == 1:
                    vectors = vectors.unsqueeze(-1)
                within = (vectors.to(source_frac.dtype) - center).square().sum(dim=-1) <= radius**2 + 1.0e-6
                vectors = vectors[within]
                if vectors.numel() == 0:
                    raise RuntimeError("exact lift bound unexpectedly removed every feasible integer vector")
                candidate_lists.append(vectors)
            choices = torch.cartesian_prod(*(torch.arange(values.shape[0], device=source_frac.device) for values in candidate_lists))
            if choices.ndim == 1:
                choices = choices.unsqueeze(-1)
            candidates = torch.stack([candidate_lists[node][choices[:, node]] for node in range(sites - 1)], dim=1)
            anchor = torch.zeros((candidates.shape[0], 1, 3), dtype=torch.long, device=source_frac.device)
            integer_lifts = torch.cat((anchor, candidates), dim=1)
        unaligned = difference[None] - integer_lifts.to(source_frac.dtype)
        translation = unaligned.mean(dim=1)
        residual = unaligned - translation[:, None]
        cartesian = torch.einsum("cni,ij->cnj", residual, lattice)
        costs.append(cartesian.square().sum(dim=(-1, -2)))
        lifts.append(integer_lifts)
        translations.append(translation)
    flattened = torch.cat(costs)
    count = min(2, flattened.numel())
    top = torch.topk(flattened, k=count, largest=False)
    flat_index = int(top.indices[0])
    assignment_index = 0
    lift_index = flat_index
    while lift_index >= costs[assignment_index].numel():
        lift_index -= costs[assignment_index].numel()
        assignment_index += 1
    cost = top.values[0]
    second_cost = top.values[1] if count == 2 else torch.tensor(torch.inf, dtype=source_frac.dtype, device=source_frac.device)
    assignment = assignments[assignment_index]
    integer_lift = lifts[assignment_index][lift_index]
    translation = translations[assignment_index][lift_index]
    endpoint_lift = target_frac[assignment] + integer_lift.to(target_frac.dtype) + translation
    velocity = endpoint_lift - source_frac
    return FixedLiftCoupling(assignment, integer_lift, translation, endpoint_lift, velocity, cost, second_cost)


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
