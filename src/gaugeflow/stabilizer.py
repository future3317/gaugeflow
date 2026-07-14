"""Proper crystallographic stabilizers used to quotient tensor alignments.

Only proper rotations are pooled.  A piezoelectric tensor is polar, so an
improper spatial operation (a mirror or inversion) must not be silently
identified with a proper SO(3) gauge transformation.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
    matrices: list[torch.Tensor] = []
    identity = torch.eye(3, dtype=torch.int64)
    for entries in product((-1, 0, 1), repeat=9):
        integer_matrix = torch.tensor(entries, dtype=torch.int64).reshape(3, 3)
        determinant = round(float(torch.linalg.det(integer_matrix.float()).item()))
        if determinant != 1:
            continue
        power = identity
        finite_order = False
        for order in range(1, 7):
            power = power @ integer_matrix
            if order in {1, 2, 3, 4, 6} and torch.equal(power, identity):
                finite_order = True
                break
        if finite_order:
            matrices.append(integer_matrix.float())
    return torch.stack(matrices)


def soft_crystal_stabilizer_actions(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    type_state: torch.Tensor,
    *,
    max_actions: int = 24,
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
    if not 1 <= max_actions:
        raise ValueError("max_actions must be positive")
    if lattice_temperature <= 0 or match_temperature <= 0:
        raise ValueError("Soft-stabilizer temperatures must be positive")

    candidates = proper_unimodular_candidates().to(lattice)
    # Q_U is generally not orthogonal. Project it before it can act on a
    # tensor; using Q_U itself would mix SL(3,Z) gauge changes with SO(3).
    proposed_actions = torch.linalg.solve(lattice, candidates @ lattice)
    left, _, right_t = torch.linalg.svd(proposed_actions)
    row_rotations = left @ right_t
    improper = torch.linalg.det(row_rotations) < 0
    if improper.any():
        correction = torch.eye(3, dtype=lattice.dtype, device=lattice.device).expand_as(row_rotations).clone()
        correction[improper, -1, -1] = -1
        row_rotations = left @ correction @ right_t
    lattice_error = (proposed_actions - row_rotations).square().mean((-1, -2))
    count = min(max_actions, candidates.shape[0])
    selected = torch.topk(lattice_error.detach(), k=count, largest=False).indices
    row_rotations = row_rotations[selected]
    lattice_error = lattice_error[selected]

    cartesian = frac_coords @ lattice
    rotated = (cartesian @ row_rotations) @ torch.linalg.inv(lattice)
    # Marginalize the translation using images which send the first atom onto
    # each candidate target atom.  This remains exact for a true symmetry.
    translations = frac_coords.unsqueeze(0) - rotated[:, :1, :]
    transformed = rotated.unsqueeze(1) + translations.unsqueeze(2)
    delta = torus_logmap(transformed.unsqueeze(3), frac_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    cartesian_delta = torch.einsum("stnij, jk -> stnik", delta, lattice)
    distances = cartesian_delta.square().sum(dim=-1)
    probabilities = torch.softmax(type_state, dim=-1)
    type_mismatch = 1.0 - probabilities @ probabilities.transpose(0, 1)
    distances = distances + chemical_penalty * type_mismatch.unsqueeze(0).unsqueeze(0)
    nearest = -match_temperature * torch.logsumexp(-distances / match_temperature, dim=-1)
    translation_error = nearest.mean(dim=-1)
    atom_error = -match_temperature * torch.logsumexp(-translation_error / match_temperature, dim=-1)
    logits = -lattice_error / lattice_temperature - atom_error / match_temperature
    weights = torch.softmax(logits, dim=0)
    # ``rotate_rank3`` and Cartesian response vectors use column-vector
    # rotations, hence transpose the row action before returning it.
    return row_rotations.transpose(-1, -2), weights


def proper_stabilizer_rotations(
    structure: Structure, *, symprec: float = 1e-3
) -> torch.Tensor:
    """Return unique proper Cartesian symmetry rotations of ``structure``.

    The returned matrices act on Cartesian column vectors.  Translations are
    intentionally omitted: they are handled by the periodic crystal graph,
    while this set removes only the residual rotational alignment ambiguity.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    rotations: list[torch.Tensor] = []
    for operation in analyzer.get_symmetry_operations(cartesian=True):
        rotation = torch.as_tensor(operation.rotation_matrix, dtype=torch.float32)
        determinant = torch.linalg.det(rotation)
        if not torch.isclose(determinant, torch.ones((), dtype=rotation.dtype), atol=1e-4):
            continue
        if not any(torch.allclose(rotation, seen, atol=1e-5, rtol=1e-5) for seen in rotations):
            rotations.append(rotation)
    if not rotations:
        raise ValueError("Symmetry analysis returned no proper stabilizer rotations")
    return torch.stack(rotations)


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
