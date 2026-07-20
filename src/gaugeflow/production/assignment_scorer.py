"""Permutation-equivariant scores for exact count-constrained occupation.

The scorer consumes only an upstream composition and a species-free parent
carrier.  Target coloring, CIF row order, child symmetry, and occupational
classes are deliberately absent.  Parent operations are reduced to their
faithful finite-site action before invariant site signatures are constructed.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn

from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def faithful_parent_action(permutations: torch.Tensor) -> torch.Tensor:
    """Validate and deduplicate the image of a parent action in ``S_N``."""
    if permutations.ndim != 2 or permutations.dtype != torch.long:
        raise ValueError("parent action must be a rank-two int64 tensor")
    operations, sites = permutations.shape
    if operations < 1 or sites < 1:
        raise ValueError("parent action must contain an operation and a site")
    expected = torch.arange(sites, device=permutations.device)
    if not torch.equal(
        torch.sort(permutations, dim=1).values,
        expected.expand_as(permutations),
    ):
        raise ValueError("parent action contains a non-permutation row")
    image = torch.unique(permutations, dim=0)
    if not bool(torch.all(image == expected.unsqueeze(0), dim=1).any()):
        raise ValueError("parent action image does not contain the identity")
    return image


def _cycle_histogram(permutations: torch.Tensor, maximum_sites: int) -> torch.Tensor:
    """Node-mass histogram of cycle lengths, averaged over operations."""
    sites = permutations.shape[1]
    histogram = torch.zeros(maximum_sites, dtype=torch.float64)
    for row in permutations.detach().cpu().tolist():
        unseen = set(range(sites))
        while unseen:
            start = min(unseen)
            current = start
            length = 0
            while current in unseen:
                unseen.remove(current)
                length += 1
                current = row[current]
            histogram[length - 1] += length
    return histogram / float(permutations.shape[0] * sites)


def parent_action_site_features(
    permutations: torch.Tensor,
    *,
    maximum_sites: int = 20,
) -> torch.Tensor:
    """Construct relabeling-equivariant site signatures from a group action.

    For each site, the feature records its parent-orbit size, point-stabilizer
    size, the point-stabilizer suborbit-size distribution, operation fixed-point
    distribution, and cycle-length distribution.  Sites related by the parent
    action receive identical features; different action orbits may differ
    without introducing an arbitrary orbit index.
    """
    image = faithful_parent_action(permutations).detach().cpu()
    operations, sites = image.shape
    if sites > maximum_sites:
        raise ValueError("parent carrier exceeds the qualified site bound")
    features: list[torch.Tensor] = []
    for site in range(sites):
        orbit_size = torch.unique(image[:, site]).numel()
        stabilizer = image[image[:, site] == site]
        if stabilizer.shape[0] * orbit_size != operations:
            raise ValueError("parent action violates orbit-stabilizer closure")

        suborbit_sizes = torch.tensor(
            [torch.unique(stabilizer[:, other]).numel() for other in range(sites)],
            dtype=torch.long,
        )
        suborbit_histogram = torch.bincount(
            suborbit_sizes,
            minlength=maximum_sites + 1,
        )[1 : maximum_sites + 1].to(torch.float64)
        suborbit_histogram /= float(sites)

        fixed_points = (stabilizer == torch.arange(sites)).sum(dim=1)
        fixed_histogram = torch.bincount(
            fixed_points,
            minlength=maximum_sites + 1,
        )[1 : maximum_sites + 1].to(torch.float64)
        fixed_histogram /= float(stabilizer.shape[0])

        features.append(
            torch.cat(
                (
                    torch.tensor(
                        [
                            orbit_size / sites,
                            stabilizer.shape[0] / operations,
                            operations / 192.0,
                            sites / maximum_sites,
                        ],
                        dtype=torch.float64,
                    ),
                    suborbit_histogram,
                    fixed_histogram,
                    _cycle_histogram(stabilizer, maximum_sites),
                )
            )
        )
    return torch.stack(features).to(dtype=torch.float32, device=permutations.device)


def _minimum_periodic_distances(
    fractional: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    if fractional.ndim != 2 or fractional.shape[1] != 3:
        raise ValueError("parent fractional coordinates must have shape [sites,3]")
    if lattice.shape != (3, 3):
        raise ValueError("parent lattice must have shape [3,3]")
    if not torch.isfinite(fractional).all() or not torch.isfinite(lattice).all():
        raise ValueError("parent geometry must be finite")
    if torch.linalg.det(lattice) <= 0:
        raise ValueError("parent lattice must have positive volume")
    sites = fractional.shape[0]
    if sites < 2:
        return lattice.new_empty(0)
    shifts = torch.tensor(
        list(itertools.product((-1.0, 0.0, 1.0), repeat=3)),
        dtype=lattice.dtype,
        device=lattice.device,
    )
    delta = fractional[:, None, :] - fractional[None, :, :]
    cartesian = (delta[..., None, :] - shifts) @ lattice
    distance = torch.linalg.vector_norm(cartesian, dim=-1).amin(dim=-1)
    mask = ~torch.eye(sites, dtype=torch.bool, device=lattice.device)
    return distance[mask]


def parent_carrier_graph_features(
    parent_fractional: torch.Tensor,
    parent_lattice: torch.Tensor,
    parent_permutations: torch.Tensor,
    *,
    cell_index: int,
    maximum_sites: int = 20,
    radial_channels: int = 16,
) -> torch.Tensor:
    """Rotation/relabeling-invariant global features of a species-free carrier."""
    if not 1 <= cell_index <= 4:
        raise ValueError("cell index lies outside the qualified HNF domain")
    image = faithful_parent_action(parent_permutations)
    sites = image.shape[1]
    if sites > maximum_sites:
        raise ValueError("parent carrier exceeds the qualified site bound")

    lattice = parent_lattice.to(torch.float64)
    singular = torch.sort(torch.linalg.svdvals(lattice)).values
    log_singular = torch.log(singular.clamp_min(1e-12))
    centered_log_singular = log_singular - log_singular.mean()
    volume = torch.linalg.det(lattice)
    row_norm = torch.linalg.vector_norm(lattice, dim=1).clamp_min(1e-12)
    cosine = torch.stack(
        (
            torch.dot(lattice[0], lattice[1]) / (row_norm[0] * row_norm[1]),
            torch.dot(lattice[0], lattice[2]) / (row_norm[0] * row_norm[2]),
            torch.dot(lattice[1], lattice[2]) / (row_norm[1] * row_norm[2]),
        )
    ).sort().values

    distances = _minimum_periodic_distances(
        parent_fractional.to(torch.float64),
        lattice,
    )
    length_scale = volume.pow(1.0 / 3.0)
    centers = torch.linspace(0.0, 1.5, radial_channels, dtype=torch.float64)
    if distances.numel():
        normalized = distances / length_scale
        width = 1.5 / max(radial_channels - 1, 1)
        radial = torch.exp(-0.5 * ((normalized[:, None] - centers) / width) ** 2).mean(dim=0)
    else:
        radial = torch.zeros(radial_channels, dtype=torch.float64)

    orbit_sizes: list[int] = []
    unseen = set(range(sites))
    image_cpu = image.detach().cpu()
    while unseen:
        seed = min(unseen)
        orbit = set(map(int, image_cpu[:, seed].tolist()))
        orbit_sizes.append(len(orbit))
        unseen.difference_update(orbit)
    global_values = torch.tensor(
        [
            math.log(float(volume)) / 8.0,
            parent_fractional.shape[0] / maximum_sites,
            sites / maximum_sites,
            cell_index / 4.0,
            image.shape[0] / 192.0,
            len(orbit_sizes) / maximum_sites,
            max(orbit_sizes) / maximum_sites,
        ],
        dtype=torch.float64,
    )
    return torch.cat((centered_log_singular, cosine, radial, global_values)).to(
        dtype=torch.float32,
        device=parent_lattice.device,
    )


class OrbitAwareAssignmentScorer(nn.Module):
    """Composition-conditioned unary energy on a species-free carrier.

    The normalized assignment law is supplied separately by
    :class:`CountConstrainedAssignmentLaw`.  This module only produces the
    permutation-equivariant site--species energy matrix.
    """

    def __init__(
        self,
        *,
        maximum_sites: int = 20,
        maximum_cell_index: int = 4,
        hidden_dim: int = 96,
        radial_channels: int = 16,
    ) -> None:
        super().__init__()
        if maximum_sites < 1 or maximum_cell_index < 1 or hidden_dim < 8:
            raise ValueError("assignment scorer bounds are invalid")
        self.maximum_sites = maximum_sites
        self.maximum_cell_index = maximum_cell_index
        self.hidden_dim = hidden_dim
        self.site_feature_dim = 4 + 3 * maximum_sites
        self.graph_feature_dim = 3 + 3 + radial_channels + 7

        self.species_embedding = nn.Embedding(CHEMICAL_ELEMENT_COUNT, hidden_dim)
        self.count_embedding = nn.Embedding(maximum_sites + 1, hidden_dim)
        self.space_group_embedding = nn.Embedding(231, hidden_dim)
        self.cell_embedding = nn.Embedding(maximum_cell_index + 1, hidden_dim)
        self.site_encoder = nn.Sequential(
            nn.Linear(self.site_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(self.graph_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.site_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.species_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.species_embedding.weight, std=0.02)
        nn.init.normal_(self.count_embedding.weight, std=0.02)
        nn.init.normal_(self.space_group_embedding.weight, std=0.02)
        nn.init.normal_(self.cell_embedding.weight, std=0.02)

    def forward(
        self,
        site_features: torch.Tensor,
        graph_features: torch.Tensor,
        batch: torch.Tensor,
        composition_counts: torch.Tensor,
        parent_space_group: torch.Tensor,
        cell_index: torch.Tensor,
    ) -> torch.Tensor:
        graphs = composition_counts.shape[0]
        if site_features.ndim != 2 or site_features.shape[1] != self.site_feature_dim:
            raise ValueError("site action features have the wrong shape")
        if graph_features.shape != (graphs, self.graph_feature_dim):
            raise ValueError("carrier graph features have the wrong shape")
        if batch.shape != (site_features.shape[0],) or batch.dtype != torch.long:
            raise ValueError("packed site batch has the wrong shape")
        if composition_counts.shape != (graphs, CHEMICAL_ELEMENT_COUNT):
            raise ValueError("composition counts have the wrong shape")
        if composition_counts.dtype != torch.long or bool((composition_counts < 0).any()):
            raise ValueError("composition counts must be nonnegative int64")
        if parent_space_group.shape != (graphs,) or parent_space_group.dtype != torch.long:
            raise ValueError("parent space groups must be a graphwise int64 vector")
        if bool(((parent_space_group < 1) | (parent_space_group > 230)).any()):
            raise ValueError("parent space group lies outside 1..230")
        if cell_index.shape != (graphs,) or cell_index.dtype != torch.long:
            raise ValueError("cell indices must be a graphwise int64 vector")
        if bool(((cell_index < 1) | (cell_index > self.maximum_cell_index)).any()):
            raise ValueError("cell index lies outside the qualified support")
        observed_nodes = torch.bincount(batch, minlength=graphs)
        if not torch.equal(observed_nodes, composition_counts.sum(dim=1)):
            raise ValueError("composition counts do not close on carrier sites")

        token = torch.arange(CHEMICAL_ELEMENT_COUNT, device=site_features.device)
        species = self.species_embedding(token).unsqueeze(0).expand(graphs, -1, -1)
        species = species + self.count_embedding(composition_counts.clamp_max(self.maximum_sites))
        count_weight = composition_counts.to(species.dtype).unsqueeze(-1)
        composition_context = (species * count_weight).sum(dim=1)
        composition_context /= composition_counts.sum(dim=1, keepdim=True).clamp_min(1)

        graph = composition_context
        graph = graph + self.space_group_embedding(parent_space_group)
        graph = graph + self.cell_embedding(cell_index)
        scale_shift = self.graph_encoder(graph_features) + torch.cat((graph, graph), dim=-1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        site = self.site_encoder(site_features)
        site = site * (1.0 + 0.1 * torch.tanh(scale[batch])) + shift[batch]
        query = torch.nn.functional.normalize(self.site_query(site), dim=-1)
        key = torch.nn.functional.normalize(self.species_key(species), dim=-1)
        return torch.einsum("nd,nkd->nk", query, key[batch]) * math.sqrt(self.hidden_dim)
