from __future__ import annotations

import torch

from scripts.build_h1a_e1_absolute_calibration_split import _partition_stratified_labels


def test_partition_stratified_iid_split_is_deterministic_and_keeps_fit_support() -> None:
    keys = torch.tensor([1] * 8 + [2] * 20 + [3] * 100 + [4] * 3, dtype=torch.long)
    kwargs = {
        "seed": 7727,
        "calibration_fraction": 0.05,
        "test_fraction": 0.05,
        "minimum_partition_for_panels": 20,
        "frequent_partition_threshold": 100,
        "frequent_partition_panel_floor": 3,
    }
    first = _partition_stratified_labels(keys, **kwargs)
    second = _partition_stratified_labels(keys, **kwargs)
    assert torch.equal(first, second)
    assert torch.equal(first[keys == 1], torch.zeros(8, dtype=torch.int8))
    assert torch.equal(first[keys == 4], torch.zeros(3, dtype=torch.int8))
    for key in (2, 3):
        selected = first[keys == key]
        assert bool((selected == 0).any())
        assert bool((selected == 1).any())
        assert bool((selected == 2).any())
    assert int((first[keys == 3] == 1).sum()) >= 3
    assert int((first[keys == 3] == 2).sum()) >= 3
