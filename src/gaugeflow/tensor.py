"""Tensor-orbit utilities for GaugeFlow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn.io import CartesianTensor

PIEZO_IRREPS = CartesianTensor("ijk=ikj")
VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
# `CartesianTensor.from_cartesian/to_cartesian` constructs a full
# ReducedTensorProducts object when no basis is supplied.  The formula is
# fixed, so constructing it inside every graph/flow step is pure overhead.
_PIEZO_CHANGE_OF_BASIS = (
    PIEZO_IRREPS.reduced_tensor_products().change_of_basis.detach().contiguous()
)


@dataclass(frozen=True)
class TensorOrbitShapeMagnitude:
    """A nonzero-orbit shape, log magnitude, and physical-zero indicator."""

    shape: torch.Tensor
    log_magnitude: torch.Tensor
    physical_zero: torch.Tensor


def piezo_irrep_blocks(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``2x1o + 1x2o + 1x3o`` into multiplicity/irrep blocks."""
    if value.ndim != 2 or value.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError("piezo irreps must have shape [batch,18]")
    return (
        value[:, :6].reshape(-1, 2, 3),
        value[:, 6:11].reshape(-1, 1, 5),
        value[:, 11:18].reshape(-1, 1, 7),
    )


def tensor_orbit_shape_magnitude(
    piezo_irreps: torch.Tensor, *, near_zero_tolerance: float = 0.0
) -> TensorOrbitShapeMagnitude:
    """Separate orbit shape from magnitude without turning a physical zero into null.

    For nonzero conditions, ``shape`` has unit Frobenius norm and
    ``log_magnitude=log(||e||)``.  For a declared physical/near zero condition
    the shape and log magnitude are finite placeholders and the explicit
    ``physical_zero`` flag carries the semantic distinction.  The tolerance is
    an input to a future data protocol, not an inferred DFPT uncertainty model.
    """
    if piezo_irreps.shape[-1] != PIEZO_IRREPS.dim or not piezo_irreps.dtype.is_floating_point:
        raise ValueError("piezo irreps must be floating tensors with final dimension 18")
    if near_zero_tolerance < 0:
        raise ValueError("near_zero_tolerance must be non-negative")
    magnitude = torch.linalg.vector_norm(piezo_irreps, dim=-1)
    physical_zero = magnitude <= near_zero_tolerance
    safe = magnitude.clamp_min(torch.finfo(piezo_irreps.dtype).tiny)
    shape = torch.where(physical_zero.unsqueeze(-1), torch.zeros_like(piezo_irreps), piezo_irreps / safe.unsqueeze(-1))
    log_magnitude = torch.where(physical_zero, torch.zeros_like(magnitude), safe.log())
    return TensorOrbitShapeMagnitude(shape=shape, log_magnitude=log_magnitude, physical_zero=physical_zero)


def piezo_change_of_basis(
    *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return the exact fixed e3nn Cartesian/irrep change-of-basis tensor."""
    value = _PIEZO_CHANGE_OF_BASIS
    if dtype is not None or device is not None:
        value = value.to(dtype=dtype or value.dtype, device=device or value.device)
    return value


def piezo_voigt_to_cartesian(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] != (3, 6):
        raise ValueError(f"Expected [..., 3, 6] Voigt tensor, got {tuple(value.shape)}")
    out = value.new_zeros(*value.shape[:-2], 3, 3, 3)
    out[..., :, 0, 0] = value[..., :, 0]
    out[..., :, 1, 1] = value[..., :, 1]
    out[..., :, 2, 2] = value[..., :, 2]
    out[..., :, 1, 2] = out[..., :, 2, 1] = value[..., :, 3]
    out[..., :, 0, 2] = out[..., :, 2, 0] = value[..., :, 4]
    out[..., :, 0, 1] = out[..., :, 1, 0] = value[..., :, 5]
    return out


def piezo_cartesian_to_voigt(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-3:] != (3, 3, 3):
        raise ValueError(f"Expected [..., 3, 3, 3] tensor, got {tuple(value.shape)}")
    if not torch.allclose(value, value.transpose(-1, -2), atol=1e-6, rtol=1e-6):
        raise ValueError("Piezo tensor must be symmetric in its final two indices")
    return torch.stack(
        (value[..., :, 0, 0], value[..., :, 1, 1], value[..., :, 2, 2],
         value[..., :, 1, 2], value[..., :, 0, 2], value[..., :, 0, 1]),
        dim=-1,
    )


def piezo_to_irreps(
    value: torch.Tensor, change_of_basis: torch.Tensor | None = None
) -> torch.Tensor:
    basis = (
        piezo_change_of_basis(dtype=value.dtype, device=value.device)
        if change_of_basis is None
        else change_of_basis.to(value)
    )
    return value.flatten(-3) @ basis.flatten(1).transpose(0, 1)


def piezo_from_irreps(
    value: torch.Tensor, change_of_basis: torch.Tensor | None = None
) -> torch.Tensor:
    if value.shape[-1] != PIEZO_IRREPS.dim:
        raise ValueError(f"Expected {PIEZO_IRREPS.dim} irreps coordinates, got {value.shape[-1]}")
    basis = (
        piezo_change_of_basis(dtype=value.dtype, device=value.device)
        if change_of_basis is None
        else change_of_basis.to(value)
    )
    return (value @ basis.flatten(1)).reshape(*value.shape[:-1], 3, 3, 3)


def rotate_rank3(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ia,...jb,...kc,...abc->...ijk", rotation, rotation, rotation, value)


def fixed_so3_frames(count: int, *, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic, seeded Haar-Monte-Carlo SO(3) nodes, including identity.

    These nodes are not a deterministic quadrature rule and carry no finite-K
    integration guarantee.  New alignment protocols must report frame/grid
    refinement rather than describe this routine as exact quadrature.
    """
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrices = torch.randn(count, 3, 3, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrices)
    signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1.0, 1.0).to(dtype)
    q = q * signs.unsqueeze(-2)
    negative = torch.linalg.det(q) < 0
    q[negative, :, -1] *= -1
    q[0] = torch.eye(3, dtype=dtype)
    return q


def fixed_lossless_response_probes(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Six directions whose symmetric dyads span ``Sym?(R?)``.

    Thus ``F_e`` on this fixed set determines every component of a tensor that
    is symmetric in its final two indices.  These probes are independent of a
    noisy generated crystal; local bonds remain complementary geometry probes.
    """
    directions = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
         (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
        dtype=dtype,
    )
    return torch.nn.functional.normalize(directions, dim=-1)


def icosahedral_response_probes(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Six antipodal icosahedral axes for a well-conditioned probe basis."""
    golden = (1.0 + 5.0**0.5) / 2.0
    directions = torch.tensor(
        ((0.0, 1.0, golden), (0.0, 1.0, -golden),
         (1.0, golden, 0.0), (1.0, -golden, 0.0),
         (golden, 0.0, 1.0), (golden, 0.0, -1.0)),
        dtype=dtype,
    )
    return torch.nn.functional.normalize(directions, dim=-1)


def response_probe_measurement_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Return the Kelvin-coordinate measurement matrix for ``n outer n``.

    Each row maps the six independent strain coordinates to the response of
    one Cartesian output component.  Its condition number diagnoses probe
    reconstruction stability independently of any neural network.
    """
    if directions.ndim != 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [probes, 3]")
    unit = torch.nn.functional.normalize(directions, dim=-1)
    root2 = 2.0**0.5
    return torch.stack(
        (
            unit[:, 0].square(), unit[:, 1].square(), unit[:, 2].square(),
            root2 * unit[:, 1] * unit[:, 2],
            root2 * unit[:, 0] * unit[:, 2],
            root2 * unit[:, 0] * unit[:, 1],
        ),
        dim=-1,
    )


def orbit_irreps(value: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Return a finite SO(3) orbit set with shape [batch, frames, 18]."""
    tensor = piezo_from_irreps(value).unsqueeze(1)
    rotated = rotate_rank3(tensor, rotations.to(value).unsqueeze(0))
    return piezo_to_irreps(rotated)


def response_field(tensor: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Evaluate F_e(n)=e:(n outer n), retaining all tensor degrees of freedom."""
    if tensor.shape[-3:] != (3, 3, 3) or directions.shape[-1] != 3:
        raise ValueError("Expected rank-three tensors and 3D directions")
    return torch.einsum("...ijk,...j,...k->...i", tensor, directions, directions)


def polarized_response(tensor: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Evaluate the symmetric bilinear constitutive form from ``F_e``.

    For a piezoelectric tensor symmetric in its final two indices,
    ``e(u, v) = (F_e(u + v) - F_e(u) - F_e(v)) / 2``.  Keeping this
    construction explicit makes the response field a lossless constitutive
    query, rather than an engineered directional summary.
    """
    return 0.5 * (
        response_field(tensor, left + right)
        - response_field(tensor, left)
        - response_field(tensor, right)
    )


def isotypic_slices() -> tuple[slice, slice, slice]:
    """The two l=1 copies are a single isotypic component for scaling."""
    return slice(0, 6), slice(6, 11), slice(11, 18)


def normalize_isotypic(irreps: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Scale 2x1o, 1x2o, and 1x3o components without per-coordinate z-scoring."""
    if scales.shape[-1] != 3:
        raise ValueError("Expected three isotypic scales")
    expanded = torch.cat(
        [scales[..., 0:1].expand(*scales.shape[:-1], 6),
         scales[..., 1:2].expand(*scales.shape[:-1], 5),
         scales[..., 2:3].expand(*scales.shape[:-1], 7)],
        dim=-1,
    )
    return irreps / expanded.to(irreps).clamp_min(torch.finfo(irreps.dtype).eps)


def response_field_error(prediction: torch.Tensor, target: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Mean squared complete vector-field error over directions."""
    delta = prediction - target
    field = torch.einsum("...ijk,mj,mk->...mi", delta, directions, directions)
    return field.square().sum(dim=-1).mean(dim=-1)


def maximum_response_field_error(
    prediction: torch.Tensor, target: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Worst directional response discrepancy over a declared probe set.

    Unlike the sphere-averaged field error, this exposes a device-relevant
    failure mode.  The result is explicitly a finite-probe approximation to
    the maximum over the unit sphere.
    """
    delta = prediction - target
    field = torch.einsum("...ijk,mj,mk->...mi", delta, directions, directions)
    return field.square().sum(dim=-1).sqrt().amax(dim=-1)
