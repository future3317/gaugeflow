"""Chemical-token conventions for new versioned generator protocols.

Historical GaugeFlow checkpoints use raw atomic numbers in a 119-wide vector:
class zero is therefore unreachable in training but reachable at sampling.  New
protocols must use this explicit, dense vocabulary instead.  It deliberately
does not reinterpret historical checkpoints.
"""

from __future__ import annotations

import torch


CHEMICAL_ELEMENT_COUNT = 118
"""The supported periodic-table elements, indexed by dense tokens 0..117."""

MASK_TOKEN = CHEMICAL_ELEMENT_COUNT
"""Internal absorbing-mask token; it is never decoded as a chemical element."""

TYPE_STATE_DIM = CHEMICAL_ELEMENT_COUNT + 1


def atomic_numbers_to_tokens(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Map physical atomic numbers ``1..118`` to dense categorical tokens.

    This rejects zero rather than silently reserving an untrained class in a
    production categorical distribution.
    """
    values = torch.as_tensor(atomic_numbers)
    if values.dtype.is_floating_point:
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError("atomic numbers must be finite integers")
    values = values.to(torch.long)
    if values.numel() and ((values < 1) | (values > CHEMICAL_ELEMENT_COUNT)).any():
        raise ValueError("atomic numbers must lie in 1..118")
    return values - 1


def tokens_to_atomic_numbers(tokens: torch.Tensor) -> torch.Tensor:
    """Decode dense chemical tokens ``0..117`` to physical atomic numbers."""
    values = torch.as_tensor(tokens)
    if values.dtype.is_floating_point:
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError("chemical tokens must be finite integers")
    values = values.to(torch.long)
    if values.numel() and ((values < 0) | (values >= CHEMICAL_ELEMENT_COUNT)).any():
        raise ValueError("mask or out-of-range token cannot be decoded as an element")
    return values + 1


def validate_type_tokens(tokens: torch.Tensor, *, allow_mask: bool = False) -> torch.Tensor:
    """Validate a dense token tensor and return its ``long`` representation."""
    values = torch.as_tensor(tokens)
    if values.dtype.is_floating_point:
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError("type tokens must be finite integers")
    values = values.to(torch.long)
    maximum = MASK_TOKEN if allow_mask else CHEMICAL_ELEMENT_COUNT - 1
    if values.numel() and ((values < 0) | (values > maximum)).any():
        raise ValueError("type token lies outside the declared vocabulary")
    return values
