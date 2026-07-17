from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_h0_d_opd_catalogue import audit


def _config() -> dict:
    path = Path("configs/gates/h0_d_opd_physical_path_catalogue_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_h0_d_fails_closed_when_catalogue_is_missing(tmp_path: Path):
    result = audit(_config(), tmp_path)
    assert not result["qualified"]
    assert result["decision"] == "H0-D_failed_stop_before_H0-E"
    assert not result["checks"]["catalogue_manifest_present"]


def test_h0_d_rejects_point_group_only_single_k_manifest(tmp_path: Path):
    config = _config()
    path = tmp_path / config["required_catalogue_manifest"]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "protocol": config["protocol"],
                "checks": {name: True for name in config["required_checks"]},
                "counts": {
                    "parent_space_groups": 230,
                    "exact_branches": 230,
                    "distorted_physical_classes": 1,
                },
                "measure": {
                    "maximum_parent_mass_sum_abs_error": 0.0,
                    "tuple_multiplicity_affects_mass": False,
                },
                "schema": {
                    "stabilizer_operation": "point_operation_index",
                    "mode_representation": "single_k_little_group",
                    "mode_occurrence": "abstract_irrep_only",
                },
            }
        ),
        encoding="utf-8",
    )
    result = audit(config, tmp_path)
    assert not result["qualified"]
    assert not result["checks"]["affine_operation_schema"]
    assert not result["checks"]["full_real_k_star_schema"]
    assert not result["checks"]["displacement_occurrence_schema"]
