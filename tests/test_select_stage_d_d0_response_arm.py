from __future__ import annotations

from scripts.select_stage_d_d0_response_arm import select_response_arm


def _result(arm: str, *, probe: float, piezo: float, other: float) -> dict[str, object]:
    return {
        "schema": "gaugeflow.stage_d_d0_response_arm.v1",
        "status": "complete",
        "arm": arm,
        "steps": 2000,
        "seed": 5731,
        "source_checkpoint_step": 40523,
        "source_checkpoint_sha256": "source",
        "cache_sha256": "cache",
        "normalizer_sha256": "normalizer",
        "protocol_sha256": "protocol",
        "validation": {
            "piezoelectric_loss": piezo,
            "piezoelectric_probe_loss": probe,
            "dielectric_loss": other,
            "born_loss": other,
            "gamma_loss": other,
            "internal_strain_loss": other,
        },
    }


def test_response_d0_selects_probe_only_when_all_checks_pass():
    protocol = {
        "protocol": "stage_d_d0_response_probe_v1",
        "selection": {
            "probe_error_relative_improvement_minimum": 0.05,
            "full_piezoelectric_loss_relative_degradation_maximum": 0.02,
            "other_task_macro_relative_degradation_maximum": 0.02,
        },
    }
    baseline = _result("baseline", probe=1.0, piezo=1.0, other=1.0)
    passing = select_response_arm(
        protocol,
        baseline,
        _result("probe", probe=0.9, piezo=1.01, other=1.01),
    )
    failing = select_response_arm(
        protocol,
        baseline,
        _result("probe", probe=0.9, piezo=1.03, other=1.01),
    )
    assert passing["qualified"] is True
    assert passing["selected_arm"] == "probe"
    assert failing["qualified"] is False
    assert failing["selected_arm"] == "baseline"
