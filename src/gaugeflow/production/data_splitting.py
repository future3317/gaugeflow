"""Shared leakage-safe deterministic split primitives."""

from __future__ import annotations

import hashlib
from typing import Literal

DataSplit = Literal["train", "calibration", "test"]


def deterministic_iid_split(
    group_key: str,
    *,
    seed: int = 5705,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> DataSplit:
    """Map an immutable group key to one stable IID split."""

    if not group_key:
        raise ValueError("IID split requires an immutable grouping key")
    if calibration_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("IID split fractions must be positive")
    if calibration_fraction + test_fraction >= 1.0:
        raise ValueError("IID split fractions leave no training support")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    unit = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + calibration_fraction:
        return "calibration"
    return "train"
