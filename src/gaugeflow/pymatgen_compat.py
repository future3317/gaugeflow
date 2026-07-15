"""Narrow compatibility helpers for the pinned Pymatgen environment."""

from __future__ import annotations

import numpy as np


def enable_structure_matcher_numpy2_compatibility() -> None:
    """Restore Pymatgen's removed NumPy boolean alias locally.

    The pinned Pymatgen release's :class:`StructureMatcher` accesses the
    removed ``np.bool`` and ``np.int`` aliases.  NumPy 2 removed them, while
    ``np.bool_`` and Python's integer type retain the semantics its matcher
    needs.  This is deliberately a compatibility shim, not a replacement for
    the requested species-aware periodic matcher.
    """
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
