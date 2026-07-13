"""Proper crystallographic stabilizers used to quotient tensor alignments.

Only proper rotations are pooled.  A piezoelectric tensor is polar, so an
improper spatial operation (a mirror or inversion) must not be silently
identified with a proper SO(3) gauge transformation.
"""

from __future__ import annotations

import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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
