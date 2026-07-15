"""Proper crystallographic stabilizers used to quotient tensor alignments.

Only proper rotations are pooled.  A piezoelectric tensor is polar, so an
improper spatial operation (a mirror or inversion) must not be silently
identified with a proper SO(3) gauge transformation.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch
from torch.profiler import record_function
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.utils import to_dense_batch

from .tensor import rotate_rank3
from .manifold import torus_logmap


@lru_cache(maxsize=1)
def proper_unimodular_candidates() -> torch.Tensor:
    """Small-integer, finite-order proper lattice-action candidates.

    A crystallographic point-group action has finite order, restricted in
    three dimensions to 1, 2, 3, 4, or 6.  Filtering out infinite-order
    shear/hyperbolic matrices is both physically necessary and substantially
    cheaper than polar-projecting every determinant-one integer matrix.  The
    catalogue includes the standard small-integer representatives for common
    crystallographic lattice actions. The matrices act on row
    fractional coordinates.  Keeping the candidate catalogue fixed makes the
    subsequent score depend only on the current generated state, never on a
    target CIF or a non-differentiable space-group call.
    """
    with record_function("stabilizer.candidate_catalogue"):
        matrices = torch.tensor(
            tuple(product((-1, 0, 1), repeat=9)), dtype=torch.int64
        ).reshape(-1, 3, 3)
        a, b, c = matrices[:, 0].unbind(-1)
        d, e, f = matrices[:, 1].unbind(-1)
        g, h, i = matrices[:, 2].unbind(-1)
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        matrices = matrices[determinant == 1]
        identity = torch.eye(3, dtype=torch.int64)
        power = identity.expand(matrices.shape[0], -1, -1).clone()
        finite_order = torch.zeros(matrices.shape[0], dtype=torch.bool)
        for order in range(1, 7):
            power = power @ matrices
            if order in {1, 2, 3, 4, 6}:
                finite_order |= (power == identity).all(dim=(-1, -2))
        return matrices[finite_order].float()


def orthogonal_polar_factor(value: torch.Tensor, *, iterations: int = 7) -> torch.Tensor:
    """Orthogonal polar factor with finite gradients at repeated singular values.

    Scaled Newton iteration converges to ``U @ Vh`` without the undefined SVD
    derivative at repeated singular values.  It preserves the input
    determinant sign: positive matrices yield SO(3) factors and negative
    matrices yield the corresponding improper O(3) factor.
    """
    if value.shape[-2:] != (3, 3):
        raise ValueError("Expected [...,3,3] matrices")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    scale = torch.linalg.matrix_norm(value, ord="fro", dim=(-2, -1), keepdim=True) / (3.0**0.5)
    rotation = value / scale.clamp_min(torch.finfo(value.dtype).tiny)
    identity = torch.eye(3, dtype=value.dtype, device=value.device).expand_as(rotation)
    for _ in range(iterations):
        inverse_transpose = torch.linalg.solve(rotation.transpose(-1, -2), identity)
        rotation = 0.5 * (rotation + inverse_transpose)
    return rotation


def proper_polar_rotation(value: torch.Tensor, *, iterations: int = 7) -> torch.Tensor:
    """SO(3) polar factor for the determinant-positive alignment catalogue."""
    rotation = orthogonal_polar_factor(value, iterations=iterations)
    if not torch.all(torch.linalg.det(rotation) > 0):
        raise ValueError("proper_polar_rotation requires determinant-positive matrices")
    return rotation


def soft_crystal_stabilizer_actions(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    type_state: torch.Tensor,
    *,
    candidates: torch.Tensor | None = None,
    candidate_chunk_size: int = 128,
    lattice_temperature: float = 0.02,
    match_temperature: float = 0.10,
    chemical_penalty: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate proper stabilizer actions from the *current* periodic state.

    ``U`` is only a *lattice-basis proposal*, never a physical rotation.  For
    a row-vector lattice ``A`` we first form ``Q_U = A^-1 U A`` and project it
    with a proper polar decomposition onto ``R_U.T in SO(3)``.  We then score
    how closely this rotation realizes the proposed lattice action and how
    well it self-matches the type-aware periodic point set after marginalizing
    a translation.  Consequently every action used to rotate a tensor is
    rigorously in SO(3), while the soft weights are a posterior over *latent
    automorphisms*, not a claim that noisy ``x_t`` has an exact space group.
    """
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        raise ValueError("Expected [atoms, 3] fractional coordinates")
    if lattice.shape != (3, 3):
        raise ValueError("Expected one [3, 3] row-vector lattice")
    if type_state.shape[0] != frac_coords.shape[0]:
        raise ValueError("Atom-type state and coordinates must have the same atom count")
    actions, weights = batched_soft_crystal_stabilizer_actions(
        frac_coords,
        lattice.unsqueeze(0),
        type_state,
        torch.zeros(frac_coords.shape[0], dtype=torch.long, device=frac_coords.device),
        candidates=candidates,
        candidate_chunk_size=candidate_chunk_size,
        lattice_temperature=lattice_temperature,
        match_temperature=match_temperature,
        chemical_penalty=chemical_penalty,
    )
    return actions[0], weights[0]


def batched_soft_crystal_stabilizer_actions(
    frac_coords: torch.Tensor,
    lattices: torch.Tensor,
    type_state: torch.Tensor,
    batch: torch.Tensor,
    *,
    candidates: torch.Tensor | None = None,
    candidate_chunk_size: int = 128,
    lattice_temperature: float = 0.02,
    match_temperature: float = 0.10,
    chemical_penalty: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched exact posterior over all finite-order lattice proposals.

    Candidate chunking controls peak memory only.  Every one of the 792
    candidates receives the same polar, lattice, translation-marginalized
    periodic type-match, and softmax definition; no top-k or sampling is used.
    """
    if frac_coords.ndim != 2 or frac_coords.shape[-1] != 3:
        raise ValueError("Expected [atoms, 3] fractional coordinates")
    if lattices.ndim != 3 or lattices.shape[-2:] != (3, 3):
        raise ValueError("Expected [graphs, 3, 3] row-vector lattices")
    if type_state.shape[0] != frac_coords.shape[0] or batch.shape != frac_coords.shape[:1]:
        raise ValueError("Atom-type state, batch, and coordinates must have the same atom count")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    if lattice_temperature <= 0 or match_temperature <= 0:
        raise ValueError("Soft-stabilizer temperatures must be positive")

    graphs = lattices.shape[0]
    with record_function("stabilizer.candidate_transfer"):
        if candidates is None:
            candidates = proper_unimodular_candidates().to(lattices)
        else:
            candidates = candidates.to(lattices)

    with record_function("stabilizer.polar_projection"):
        proposed_actions = torch.linalg.solve(
            lattices.unsqueeze(1), candidates.unsqueeze(0) @ lattices.unsqueeze(1)
        )
        row_rotations = proper_polar_rotation(proposed_actions)
        lattice_error = (proposed_actions - row_rotations).square().mean((-1, -2))

    with record_function("stabilizer.periodic_type_self_match"):
        dense_frac, atom_mask = to_dense_batch(frac_coords, batch, batch_size=graphs)
        dense_types, _ = to_dense_batch(type_state, batch, batch_size=graphs)
        inverse_lattices = torch.linalg.inv(lattices)
        cartesian = dense_frac @ lattices
        rotated_cartesian = torch.einsum("bni,bkij->bknj", cartesian, row_rotations)
        rotated = torch.einsum("bkni,bij->bknj", rotated_cartesian, inverse_lattices)
        probabilities = torch.softmax(dense_types, dim=-1)
        type_mismatch = 1.0 - torch.einsum("bnc,bmc->bnm", probabilities, probabilities)
        atom_count = atom_mask.sum(dim=-1).clamp_min(1).to(lattices.dtype)
        atom_errors = []
        for start in range(0, candidates.shape[0], candidate_chunk_size):
            stop = min(start + candidate_chunk_size, candidates.shape[0])
            rotated_chunk = rotated[:, start:stop]
            # Every possible translation maps the first source atom to one
            # target atom, exactly matching the original marginalization.
            translations = dense_frac[:, None, :, :] - rotated_chunk[:, :, :1, :]
            transformed = rotated_chunk[:, :, None, :, :] + translations[:, :, :, None, :]
            delta = torus_logmap(
                transformed.unsqueeze(-2), dense_frac[:, None, None, None, :, :]
            )
            cartesian_delta = torch.einsum("bktnmi,bij->bktnmj", delta, lattices)
            distances = cartesian_delta.square().sum(dim=-1)
            distances = distances + chemical_penalty * type_mismatch[:, None, None, :, :]
            distances = distances.masked_fill(
                ~atom_mask[:, None, None, None, :], torch.inf
            )
            nearest = -match_temperature * torch.logsumexp(
                -distances / match_temperature, dim=-1
            )
            translation_error = (
                nearest * atom_mask[:, None, None, :]
            ).sum(dim=-1) / atom_count[:, None, None]
            translation_error = translation_error.masked_fill(
                ~atom_mask[:, None, :], torch.inf
            )
            atom_errors.append(
                -match_temperature
                * torch.logsumexp(-translation_error / match_temperature, dim=-1)
            )
        atom_error = torch.cat(atom_errors, dim=1)
        logits = -lattice_error / lattice_temperature - atom_error / match_temperature
        weights = torch.softmax(logits, dim=-1)

    # `rotate_rank3` and response vectors use column-vector rotations.
    return row_rotations.transpose(-1, -2), weights


def crystal_point_group_operations(
    structure: Structure,
    *,
    proper_only: bool,
    symprec: float = 1e-3,
    max_cartesian_orthogonality_error: float = 1e-2,
) -> torch.Tensor:
    """Return unique Cartesian point-group matrices of ``structure``.

    These matrices act on Cartesian *column* vectors.  They are for a
    crystal-compatibility calculation, rather than automatically an orbit
    quotient: when ``proper_only=False`` they intentionally retain mirrors
    and inversions.  That distinction is essential for a polar rank-three
    piezoelectric tensor.  Its condition orbit is an ``SO(3)`` orbit, whereas
    Neumann compatibility with a crystal point group is an ``O(3)`` statement.
    """
    if max_cartesian_orthogonality_error <= 0:
        raise ValueError("max_cartesian_orthogonality_error must be positive")
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    rotations: list[torch.Tensor] = []
    for operation in analyzer.get_symmetry_operations(cartesian=True):
        rotation = torch.as_tensor(operation.rotation_matrix, dtype=torch.float32)
        determinant = torch.linalg.det(rotation)
        if not torch.isclose(determinant.abs(), torch.ones((), dtype=rotation.dtype), atol=1e-4):
            raise ValueError(
                "Cartesian symmetry operation has determinant outside {+1,-1}; "
                "it is not an O(3) point-group action"
            )
        if proper_only and not torch.isclose(
            determinant, torch.ones((), dtype=rotation.dtype), atol=1e-4
        ):
            continue
        # CIF decimal precision can make an otherwise orthogonal Cartesian
        # operation slightly non-orthogonal (for example ~3e-4 in its Gram
        # residual). A tensor must never be acted on by that raw matrix:
        # validate the finite residual, then use its nearest orthogonal polar
        # factor. Newton polar iteration preserves the determinant sign, so an
        # improper crystal operation remains improper instead of being folded
        # into the SO(3) condition orbit.
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        error = (rotation @ rotation.T - identity).abs().max()
        if error > max_cartesian_orthogonality_error:
            raise ValueError(
                f"Cartesian symmetry operation has Gram residual {float(error):.3e}, exceeding the declared tolerance"
            )
        rotation = orthogonal_polar_factor(rotation.unsqueeze(0))[0]
        if not any(torch.allclose(rotation, seen, atol=1e-5, rtol=1e-5) for seen in rotations):
            rotations.append(rotation)
    if not rotations:
        scope = "proper" if proper_only else "orthogonal"
        raise ValueError(f"Symmetry analysis returned no {scope} point-group operations")
    return torch.stack(rotations)


def proper_stabilizer_rotations(
    structure: Structure, *, symprec: float = 1e-3, max_cartesian_orthogonality_error: float = 1e-2
) -> torch.Tensor:
    """Return only the proper Cartesian point-group rotations of ``structure``.

    This is intentionally the SO(3)-only helper used for latent tensor-orbit
    alignment.  Full O(3) point-group compatibility belongs to
    :func:`crystal_point_group_operations` with ``proper_only=False``.
    """
    return crystal_point_group_operations(
        structure,
        proper_only=True,
        symprec=symprec,
        max_cartesian_orthogonality_error=max_cartesian_orthogonality_error,
    )


def observed_tensor_stabilizer_rotations(
    tensor: torch.Tensor,
    crystal_rotations: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> torch.Tensor:
    """Return the response-preserving subset of a crystal's proper rotations.

    This is a discrete, data-time estimator of ``H_e`` restricted to the
    crystallographic rotations observed for the paired structure.  It is not
    presented as a solver for the full continuous tensor stabilizer; generic
    tensors have only the identity anyway, while a zero tensor has a
    continuous stabilizer that no finite catalogue can enumerate.  Identity
    is required so malformed labels fail loudly instead of yielding an empty
    quotient action.
    """
    if tensor.shape[-3:] != (3, 3, 3):
        raise ValueError("Expected a symmetric rank-three Cartesian tensor")
    if crystal_rotations.ndim != 3 or crystal_rotations.shape[-2:] != (3, 3):
        raise ValueError("Expected [count,3,3] proper crystal rotations")
    transformed = rotate_rank3(tensor, crystal_rotations.to(tensor))
    keep = torch.isclose(transformed, tensor, atol=atol, rtol=rtol).reshape(
        crystal_rotations.shape[0], -1
    ).all(dim=-1)
    selected = crystal_rotations[keep]
    identity = torch.eye(3, dtype=crystal_rotations.dtype, device=crystal_rotations.device)
    has_identity = any(
        torch.allclose(value, identity, atol=atol, rtol=rtol) for value in selected
    )
    if not selected.numel() or not has_identity:
        raise ValueError("Tensor is inconsistent with the identity rotation")
    return selected
