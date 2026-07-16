"""Low-order harmonic conditioning primitives for the next GaugeFlow protocol.

This module deliberately separates two objects that were conflated by the
finite-frame implementation:

* an SO(3)-invariant condition token available even at an all-noise source;
* a state-dependent relative-alignment posterior over a deterministic,
  refinable set of proper rotations.

The grid is a deterministic Haar *quasi-Monte-Carlo* rule.  It is not claimed
to integrate arbitrary functions exactly; a protocol using it must report grid
refinement.  The score itself is band-limited through l <= 3, exactly matching
the irreducible content of a piezoelectric rank-three polar tensor.

For a geometry query ``q_l(x)`` with ``q_l(gx)=rho_l(g)q_l(x)`` and a polar
condition ``e`` the unnormalised continuous score is

``s(R; x, e) = sum_l,m w_lm <rho_l(R)e_lm, q_l(x)> / sqrt(2l+1)``.

Orthogonality of each ``rho_l`` gives the covariance theorem used by the
alignment posterior:

``s(R; g x, h e) = s(g^{-1} R h; x, e)`` for ``g,h,R in SO(3)``.

The theorem concerns the continuous score.  A finite Hopf grid is generally
not closed under left/right multiplication, so its sampled softmax is only a
numerical approximation; callers must measure shift and refinement error.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import nn
from torch_geometric.utils import scatter

from .tensor import (
    PIEZO_IRREPS,
    piezo_from_irreps,
    piezo_to_irreps,
    rotate_rank3,
    tensor_orbit_shape_magnitude,
)

_PIEZO_SLICES = (slice(0, 6), slice(6, 11), slice(11, 18))


def deterministic_so3_grid(
    count: int, *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return deterministic low-discrepancy Haar SO(3) nodes.

    A Hopf-coordinate construction turns three deterministic uniform sequences
    into unit quaternions.  Equal weights approximate Haar integration and are
    suitable for a *refinement study*, not an assertion of exact quadrature.
    The identity is explicitly included so small grids retain an unambiguous
    reference node.
    """
    if count < 1:
        raise ValueError("count must be positive")
    target_device = torch.device(device) if device is not None else None
    index = torch.arange(count, dtype=dtype, device=target_device)
    u = (index + 0.5) / count
    # Irrational rotations give deterministic low-discrepancy phases without
    # using a pseudorandom generator or an input representative.
    v = torch.remainder((index + 0.5) * ((5.0**0.5 - 1.0) / 2.0), 1.0)
    w = torch.remainder((index + 0.5) * (2.0**0.5 - 1.0), 1.0)
    q = torch.stack(
        (
            (1.0 - u).sqrt() * torch.sin(2.0 * math.pi * v),
            (1.0 - u).sqrt() * torch.cos(2.0 * math.pi * v),
            u.sqrt() * torch.sin(2.0 * math.pi * w),
            u.sqrt() * torch.cos(2.0 * math.pi * w),
        ),
        dim=-1,
    )
    x, y, z, r = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    rotation = torch.stack(
        (
            1 - two * (y.square() + z.square()), two * (x * y - z * r), two * (x * z + y * r),
            two * (x * y + z * r), 1 - two * (x.square() + z.square()), two * (y * z - x * r),
            two * (x * z - y * r), two * (y * z + x * r), 1 - two * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(count, 3, 3)
    rotation[0] = torch.eye(3, dtype=dtype, device=rotation.device)
    return rotation


def piezo_irrep_blocks(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``2x1o + 1x2o + 1x3o`` into ``[B,mul,2l+1]`` blocks."""
    if value.ndim != 2 or value.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError("piezo irreps must have shape [batch, 18]")
    return (
        value[:, _PIEZO_SLICES[0]].reshape(-1, 2, 3),
        value[:, _PIEZO_SLICES[1]].reshape(-1, 1, 5),
        value[:, _PIEZO_SLICES[2]].reshape(-1, 1, 7),
    )


def join_piezo_irrep_blocks(blocks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`piezo_irrep_blocks`."""
    first, second, third = blocks
    if first.ndim < 3 or second.ndim < 3 or third.ndim < 3:
        raise ValueError("all irrep blocks need batch, multiplicity, and irrep axes")
    if first.shape[-2:] != (2, 3) or second.shape[-2:] != (1, 5) or third.shape[-2:] != (1, 7):
        raise ValueError("unexpected piezo irrep block shapes")
    return torch.cat((first.flatten(-2), second.flatten(-2), third.flatten(-2)), dim=-1)


def rotate_piezo_irreps_on_grid(piezo_irreps: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Apply every proper rotation to every piezo condition in exact e3nn bases.

    e3nn 0.5's direct Wigner-D helper leaves its generator constants on CPU,
    which makes otherwise-valid CUDA rotations fail.  We therefore rotate the
    equivalent Cartesian rank-three tensor on-device and return to the same
    irreducible basis.  This is algebraically the same representation action,
    retains l<=3 exactly, and is covered by a Cartesian/irrep consistency test.
    """
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [frames, 3, 3]")
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=2e-5, rtol=2e-5):
        raise ValueError("harmonic conditioning accepts proper SO(3) rotations only")
    tensor = piezo_from_irreps(piezo_irreps)
    return piezo_to_irreps(rotate_rank3(tensor.unsqueeze(1), rotations.to(piezo_irreps).unsqueeze(0)))


def geometric_harmonic_queries(
    directions: torch.Tensor, edge_graph: torch.Tensor, graph_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean l=1,2,3 spherical-harmonic geometry queries from periodic edges.

    The query depends only on the current generated geometry.  In particular,
    it neither accepts a paired target CIF nor treats integer cell changes as
    tensor rotations.  At a perfectly isotropic/noise state these queries can
    be zero; the invariant condition channel handles that legitimate case.
    """
    if directions.ndim != 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [edges, 3]")
    if edge_graph.shape != directions.shape[:1]:
        raise ValueError("edge_graph must contain one graph index per edge")
    if graph_count < 1:
        raise ValueError("graph_count must be positive")
    dims = (3, 5, 7)
    if directions.numel() == 0:
        return tuple(directions.new_zeros((graph_count, dim)) for dim in dims)  # type: ignore[return-value]
    unit = torch.nn.functional.normalize(directions, dim=-1)
    harmonics = o3.spherical_harmonics([1, 2, 3], unit, normalize=True, normalization="component")
    return tuple(
        scatter(harmonics[:, start:stop], edge_graph, dim=0, dim_size=graph_count, reduce="mean")
        for start, stop in ((0, 3), (3, 8), (8, 15))
    )  # type: ignore[return-value]


def harmonic_alignment_scores(
    piezo_irreps: torch.Tensor,
    directions: torch.Tensor,
    edge_graph: torch.Tensor,
    rotations: torch.Tensor,
    *,
    weight_l1: torch.Tensor,
    weight_l2: torch.Tensor,
    weight_l3: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Evaluate the band-limited continuous harmonic score at arbitrary SO(3) nodes.

    This is deliberately separate from grid softmaxing so the covariance
    theorem can be tested at the exact transformed nodes ``g^{-1} R h``.  The
    return value contains one score per graph and supplied rotation node plus
    the geometry queries used to construct it.
    """
    if piezo_irreps.ndim != 2 or piezo_irreps.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError("piezo_irreps must have shape [graphs,18]")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [frames,3,3]")
    determinant = torch.linalg.det(rotations)
    if not torch.allclose(determinant, torch.ones_like(determinant), atol=2e-5, rtol=2e-5):
        raise ValueError("harmonic scores accept proper SO(3) rotations only")
    if weight_l1.shape != (2,) or weight_l2.shape != (1,) or weight_l3.shape != (1,):
        raise ValueError("harmonic score weights must have shapes [2], [1], [1]")
    graph_count = piezo_irreps.shape[0]
    rotated = rotate_piezo_irreps_on_grid(piezo_irreps, rotations)
    queries = geometric_harmonic_queries(directions, edge_graph, graph_count)
    blocks = piezo_irrep_blocks(rotated.reshape(-1, PIEZO_IRREPS.dim))
    first = blocks[0].reshape(graph_count, rotations.shape[0], 2, 3)
    second = blocks[1].reshape(graph_count, rotations.shape[0], 1, 5)
    third = blocks[2].reshape(graph_count, rotations.shape[0], 1, 7)
    score = (
        torch.einsum("bfmi,bi,m->bf", first, queries[0], weight_l1.to(first)) / math.sqrt(3.0)
        + torch.einsum("bfmi,bi,m->bf", second, queries[1], weight_l2.to(second)) / math.sqrt(5.0)
        + torch.einsum("bfmi,bi,m->bf", third, queries[2], weight_l3.to(third)) / math.sqrt(7.0)
    )
    return score, queries


def finite_grid_shift_residual(
    rotations: torch.Tensor,
    *,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each shifted node's nearest-node Frobenius residual on a finite grid.

    ``left`` and ``right`` implement ``left^{-1} R right``.  A zero result
    means that particular transformation reindexes the declared finite grid;
    a nonzero result is expected in general and quantifies why a fixed-grid
    posterior cannot be advertised as exactly left/right covariant.
    """
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape [frames,3,3]")
    device, dtype = rotations.device, rotations.dtype
    left_matrix = torch.eye(3, device=device, dtype=dtype) if left is None else left.to(rotations)
    right_matrix = torch.eye(3, device=device, dtype=dtype) if right is None else right.to(rotations)
    if left_matrix.shape != (3, 3) or right_matrix.shape != (3, 3):
        raise ValueError("left and right must have shape [3,3]")
    shifted = left_matrix.transpose(-1, -2).unsqueeze(0) @ rotations @ right_matrix.unsqueeze(0)
    pairwise = torch.linalg.matrix_norm(
        shifted[:, None] - rotations[None], ord="fro", dim=(-2, -1)
    )
    return pairwise.amin(dim=-1)


def low_order_orbit_invariants(piezo_irreps: torch.Tensor) -> torch.Tensor:
    """Return low-order SO(3) invariants plus a parity-odd pseudoscalar.

    This is intentionally a compact discriminator, not a claim to be a full
    integrity basis for rank-three piezoelectric tensors.  The final two
    entries are ``0o`` pseudoscalars obtained by coupling ``l=2`` and ``l=3``
    to an axial ``l=1`` vector and pairing it with each polar ``l=1`` copy.
    They are invariant under proper rotations and flip under reflection.
    """
    first, second, third = piezo_irrep_blocks(piezo_irreps)
    first_gram = torch.einsum("bmi,bni->bmn", first, first)
    quadratic = torch.cat(
        (first_gram[:, 0, 0:1], first_gram[:, 1, 1:2], first_gram[:, 0, 1:2],
         second.square().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1),
         third.square().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1)),
        dim=-1,
    )
    # FullTensorProduct has fixed Clebsch--Gordan coefficients, unlike a
    # learned fully-connected tensor product.  ``2o x 3o`` decomposes as
    # ``1e + 2e + ... + 5e``, so its first three coordinates are the unique
    # axial l=1 coupling used below.
    coupling = o3.FullTensorProduct("1x2o", "1x3o").to(device=piezo_irreps.device, dtype=piezo_irreps.dtype)
    axial = coupling(second.flatten(1), third.flatten(1))[:, :3]
    pseudoscalars = torch.einsum("bmi,bi->bm", first, axial)
    magnitude = piezo_irreps.square().sum(dim=-1, keepdim=True).sqrt()
    return torch.cat((quadratic, pseudoscalars, torch.log1p(magnitude)), dim=-1)


def normalized_low_order_orbit_invariants(piezo_irreps: torch.Tensor) -> torch.Tensor:
    """Low-order shape invariants, log magnitude, and an explicit zero class.

    Quadratic invariants scale as ``||e||^2`` and pseudoscalars as ``||e||^3``.
    Dividing those powers out for nonzero inputs prevents tensor amplitude from
    dominating the early orbit descriptor.  Exact/declared physical zeros use
    a finite all-zero shape descriptor plus a distinct flag; they are never
    mapped to the classifier-free null condition.
    """
    raw = low_order_orbit_invariants(piezo_irreps)
    decomposition = tensor_orbit_shape_magnitude(piezo_irreps)
    magnitude = torch.linalg.vector_norm(piezo_irreps, dim=-1, keepdim=True)
    safe = magnitude.clamp_min(torch.finfo(piezo_irreps.dtype).tiny)
    quadratic = raw[:, :5] / safe.square()
    pseudoscalar = raw[:, 5:7] / safe.pow(3)
    normalized = torch.cat(
        (
            quadratic,
            pseudoscalar,
            decomposition.log_magnitude.unsqueeze(-1),
            decomposition.physical_zero.to(piezo_irreps).unsqueeze(-1),
        ),
        dim=-1,
    )
    return torch.where(decomposition.physical_zero.unsqueeze(-1), torch.cat(
        (torch.zeros_like(normalized[:, :7]), normalized[:, 7:]), dim=-1
    ), normalized)


class OrbitInvariantConditionEncoder(nn.Module):
    """A low-order, proper-SO(3)-invariant early condition channel."""

    feature_dim = 9

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, piezo_irreps: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_low_order_orbit_invariants(piezo_irreps))


class HarmonicRelativeAlignment(nn.Module):
    """State-derived l<=3 relative alignment posterior over an SO(3) grid."""

    def __init__(self, grid_size: int = 60, temperature: float = 1.0) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("harmonic alignment needs at least two grid nodes")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.register_buffer("rotations", deterministic_so3_grid(grid_size))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
        # One state query per degree.  The l=1 condition has two multiplicity
        # channels and consequently has two independently learnable couplings.
        self.weight_l1 = nn.Parameter(torch.ones(2))
        self.weight_l2 = nn.Parameter(torch.ones(1))
        self.weight_l3 = nn.Parameter(torch.ones(1))

    def forward(
        self, piezo_irreps: torch.Tensor, directions: torch.Tensor, edge_graph: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        rotations = self.rotations.to(piezo_irreps)
        score, _ = harmonic_alignment_scores(
            piezo_irreps,
            directions,
            edge_graph,
            rotations,
            weight_l1=self.weight_l1,
            weight_l2=self.weight_l2,
            weight_l3=self.weight_l3,
        )
        # Keep the sampled irreps for posterior averaging.  The score helper
        # is intentionally pure so it can also be evaluated at arbitrary
        # continuous nodes in the covariance tests.
        rotated = rotate_piezo_irreps_on_grid(piezo_irreps, rotations)
        temperature = self.log_temperature.exp().clamp_min(1e-4).to(score)
        posterior = torch.softmax(score / temperature, dim=-1)
        aligned = (posterior.unsqueeze(-1) * rotated).sum(dim=1)
        entropy = -(posterior * posterior.clamp_min(torch.finfo(posterior.dtype).tiny).log()).sum(dim=-1)
        diagnostics = {
            "scores": score,
            "posterior": posterior,
            "entropy": entropy,
            "top_mode_mass": posterior.max(dim=-1).values,
            "rotations": rotations,
        }
        return aligned, posterior, entropy, diagnostics


class HarmonicDoubleCosetConditionEncoder(nn.Module):
    """Early invariant token plus coherent harmonic response-field alignment.

    The invariant branch is present from the noise source.  The relative-frame
    response branch is multiplied by a confidence/time gate, rather than being
    forced to invent a meaningful crystal frame before geometry exists.  This
    class has the same high-level output contract as the legacy conditioning
    encoder and can therefore be used by a separately versioned vector field.
    """

    def __init__(self, hidden_dim: int, grid_size: int = 60) -> None:
        super().__init__()
        self.invariant = OrbitInvariantConditionEncoder(hidden_dim)
        self.alignment = HarmonicRelativeAlignment(grid_size=grid_size)
        self.aligned_token = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Linear(2, 1)
        # At t=0 with a uniform posterior, lambda=sigmoid(-2) ~= 0.12.  The
        # parameters remain learnable, but the declared initialization makes
        # the intended early-invariant behavior reproducible.
        with torch.no_grad():
            self.gate.weight.copy_(torch.tensor([[2.0, 4.0]]))
            self.gate.bias.fill_(-2.0)
        self.null_condition = nn.Parameter(torch.zeros(hidden_dim))
        self.present_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        piezo_irreps: torch.Tensor,
        present: torch.Tensor,
        directions: torch.Tensor,
        edge_graph: torch.Tensor,
        time: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        if time.shape != (piezo_irreps.shape[0],):
            raise ValueError("time must provide one value per tensor condition")
        aligned, posterior, entropy, posterior_diagnostics = self.alignment(
            piezo_irreps, directions, edge_graph
        )
        grid_log = math.log(float(posterior.shape[-1]))
        confidence = 1.0 - entropy / grid_log
        gate = torch.sigmoid(self.gate(torch.stack((time, confidence), dim=-1))).squeeze(-1)
        invariant_token = self.invariant(piezo_irreps)
        aligned_features = torch.cat(
            (low_order_orbit_invariants(aligned), entropy.unsqueeze(-1), posterior.max(dim=-1).values.unsqueeze(-1)),
            dim=-1,
        )
        graph_condition = invariant_token + gate.unsqueeze(-1) * self.aligned_token(aligned_features)
        graph_condition = graph_condition + self.present_bias
        mask = present.to(dtype=torch.bool)
        graph_condition = torch.where(
            mask, graph_condition, self.null_condition.unsqueeze(0).expand_as(graph_condition)
        )
        aligned_tensor = piezo_from_irreps(aligned)
        if directions.numel():
            edge_tensor = aligned_tensor[edge_graph]
            response = torch.einsum("eijk,ej,ek->ei", edge_tensor, directions, directions)
            response = response * gate[edge_graph].unsqueeze(-1)
            response = torch.where(mask[edge_graph], response, torch.zeros_like(response))
        else:
            response = directions.new_empty((0, 3))
        auxiliary = torch.zeros_like(response)
        outputs = (graph_condition, response, auxiliary, posterior)
        if not return_diagnostics:
            return outputs
        diagnostics = {
            "invariant_embedding": invariant_token,
            "aligned_embedding": graph_condition,
            "alignment_gate": gate,
            "alignment_confidence": confidence,
            "aligned_irreps": aligned,
            **posterior_diagnostics,
        }
        return (*outputs, diagnostics)
