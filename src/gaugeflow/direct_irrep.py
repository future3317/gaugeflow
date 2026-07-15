"""Complete direct irreducible-tensor baseline components.

This is a *new* baseline component for future versioned experiments.  The
historical Gate-A ``direct_irrep`` checkpoint exposed only two Cartesian
contractions and remains frozen.  For a piezo condition
``2x1o + 1x2o + 1x3o`` and local symmetric dyad ``0e + 1x2e``, there are six
independent output-vector CG pathways: two from ``1o x 0e``, two from
``1o x 2e``, one from ``2o x 2e``, and one from ``3o x 2e``.
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.io import CartesianTensor
from torch import nn

from .tensor import piezo_to_irreps


_CONDITION_IRREPS = o3.Irreps("2x1o + 1x2o + 1x3o")
_GEOMETRY_IRREPS = CartesianTensor("ij=ji")


class CompleteDirectIrrepCoupling(nn.Module):
    """Exact e3nn CG coupling from a rank-three tensor and edge dyad to 6 vectors."""

    pathway_count = 6

    def __init__(self) -> None:
        super().__init__()
        self.product = o3.FullTensorProduct(_CONDITION_IRREPS, _GEOMETRY_IRREPS)
        basis = _GEOMETRY_IRREPS.reduced_tensor_products().change_of_basis.detach().contiguous()
        self.register_buffer("geometry_change_of_basis", basis, persistent=False)

    def forward(self, tensors: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Return all six ``1o`` pathways with shape ``[edges, 6, 3]``.

        ``tensors`` contains one Cartesian piezo tensor for each edge and
        ``directions`` is the corresponding physical unit vector.  The local
        geometry is the symmetric dyad ``n outer n``.  No spherical-harmonic
        sampling approximation is used; e3nn supplies the exact CG basis.
        """
        if tensors.ndim != 4 or tensors.shape[-3:] != (3, 3, 3):
            raise ValueError("tensors must have shape [edges, 3, 3, 3]")
        if directions.shape != (tensors.shape[0], 3):
            raise ValueError("directions must have one [3] vector per tensor")
        if directions.numel() == 0:
            return directions.new_empty((0, self.pathway_count, 3))
        if not torch.allclose(tensors, tensors.transpose(-1, -2), atol=1e-5, rtol=1e-5):
            raise ValueError("piezo tensors must be symmetric in the final two indices")
        unit = torch.nn.functional.normalize(directions, dim=-1)
        dyad = unit.unsqueeze(-1) * unit.unsqueeze(-2)
        geometry = dyad.flatten(-2) @ self.geometry_change_of_basis.to(dyad).flatten(1).transpose(0, 1)
        output = self.product(piezo_to_irreps(tensors), geometry)
        # Product output instructions are deliberately pinned to the four
        # pathways stated in the module docstring. The two multiplicity-two
        # terms provide four vectors; the remaining two instructions provide
        # one vector each.
        return torch.cat(
            (output[..., 1:7].reshape(-1, 2, 3), output[..., 7:13].reshape(-1, 2, 3),
             output[..., 13:16].reshape(-1, 1, 3), output[..., 16:19].reshape(-1, 1, 3)),
            dim=1,
        )
