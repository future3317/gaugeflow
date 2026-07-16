from pathlib import Path

import torch
import yaml

from gaugeflow.vnext.experiments.q0_1_partial_legacy_audit import _corrected_solver_rows
from gaugeflow.vnext.experiments.q0_c0_audit import run as frozen_q0_run
from gaugeflow.vnext.processes import translation_horizontal_basis


def test_q0_1_protocol_is_versioned_without_reclassifying_original_q0():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "gates" / "q0_1_partial_legacy_audit.yaml").read_text(encoding="utf-8"))
    original = yaml.safe_load((root / "configs" / "gates" / "q0_c0_audit.yaml").read_text(encoding="utf-8"))
    assert config["gate"] == "Q0.1"
    assert config["outcome_contract"]["execution_status"] == "complete_partial_legacy"
    assert original["outcome_contract"]["allowed_statuses"] == ["complete", "blocked"]


def test_solver_columns_separate_solution_error_from_target_residual():
    generator = torch.Generator().manual_seed(8101)
    source = torch.randn((4, 4, 3), dtype=torch.float64, generator=generator)
    endpoint = torch.randn((4, 4, 3), dtype=torch.float64, generator=generator)
    settings = {
        "solver_terminal_epsilon": 1.0e-6,
        "euler_steps": [32],
        "rk4_steps": [16],
        "adaptive_rtol": 1.0e-9,
        "adaptive_atol": 1.0e-11,
    }
    rows = _corrected_solver_rows(
        source,
        endpoint,
        translation_horizontal_basis(4, dtype=torch.float64),
        settings,
    )
    assert all("solution_error_rms" in row and "target_residual_rms" in row for row in rows)
    assert all("endpoint_rms" not in row for row in rows)
    assert all(row["target_residual_rms"] > row["solution_error_rms"] for row in rows)


def test_original_q0_runner_is_frozen_after_published_bundle(tmp_path):
    try:
        frozen_q0_run(tmp_path / "unused.yaml", device=torch.device("cpu"))
    except RuntimeError as error:
        assert "original Q0 is frozen" in str(error)
    else:
        raise AssertionError("published Q0 must not be silently rerun")
