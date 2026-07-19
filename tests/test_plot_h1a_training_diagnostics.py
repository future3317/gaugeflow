from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "plot_h1a_training_diagnostics.py"
    spec = importlib.util.spec_from_file_location("plot_h1a_training_diagnostics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_jsonl_is_strict_and_numeric(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "training_metrics.jsonl"
    path.write_text('{"step": 1, "coordinate_loss": 0.5}\n', encoding="utf-8")
    assert module._read_jsonl(path) == [{"step": 1.0, "coordinate_loss": 0.5}]
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        module._read_jsonl(path)


def test_slot_matrix_retains_checkpoint_layer_and_time() -> None:
    module = _load_script()
    checkpoints = {
        "0": {
            "0.1": [{"effective_slot_count": 8.0}, {"effective_slot_count": 7.0}],
            "0.5": [{"effective_slot_count": 6.0}, {"effective_slot_count": 5.0}],
        },
        "10": {
            "0.1": [{"effective_slot_count": 4.0}, {"effective_slot_count": 3.0}],
            "0.5": [{"effective_slot_count": 2.0}, {"effective_slot_count": 1.0}],
        },
    }
    matrix, rows, times = module._slot_matrix(checkpoints, "effective_slot_count")
    assert rows == ["0/L0", "0/L1", "10/L0", "10/L1"]
    assert times == ["0.1", "0.5"]
    np.testing.assert_allclose(matrix, [[8.0, 6.0], [7.0, 5.0], [4.0, 2.0], [3.0, 1.0]])
