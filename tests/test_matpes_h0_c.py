import json
from pathlib import Path

import numpy as np

from scripts.audit_matpes_h0_c import (
    _checkpoint_files_match,
    axis_angle_rotation,
    full_to_voigt,
    select_held_out_rows,
    voigt_kbar_to_full_gpa,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_matpes_stress_conversion_preserves_voigt_order_and_changes_sign():
    source = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    converted = voigt_kbar_to_full_gpa(source)
    assert np.allclose(
        converted,
        [[-1.0, -6.0, -5.0], [-6.0, -2.0, -4.0], [-5.0, -4.0, -3.0]],
    )
    assert np.allclose(full_to_voigt(converted), -0.1 * np.asarray(source))


def test_matpes_selection_is_prediction_independent_and_order_invariant(tmp_path):
    rows = [
        {"matpes_id": f"id-{index}", "prediction_like_field": 10 - index}
        for index in range(10)
    ]
    forward = tmp_path / "forward.jsonl"
    reverse = tmp_path / "reverse.jsonl"
    forward.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reverse.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8"
    )
    selected_forward, count_forward = select_held_out_rows(
        forward, protocol="frozen", sample_size=4
    )
    selected_reverse, count_reverse = select_held_out_rows(
        reverse, protocol="frozen", sample_size=4
    )
    assert count_forward == count_reverse == 10
    assert [row["matpes_id"] for row in selected_forward] == [
        row["matpes_id"] for row in selected_reverse
    ]


def test_h0_c_rotation_is_proper_and_orthogonal():
    rotation = axis_angle_rotation([1.0, 2.0, -1.0], 0.731)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_h0_c_v2_changes_only_the_independent_teacher_not_sample_or_thresholds():
    config_root = REPOSITORY_ROOT / "configs" / "gates"
    v1 = json.loads(
        (config_root / "h0_c_matpes_teacher_qualification_v1.json").read_text()
    )
    v2 = json.loads(
        (config_root / "h0_c_matpes_teacher_qualification_v2.json").read_text()
    )
    assert v2["selection"]["selection_seed_string"] == v1["selection"][
        "selection_seed_string"
    ]
    assert v2["selection"]["sample_size"] == v1["selection"]["sample_size"]
    assert v2["selection"]["invariance_sample_size"] == v1["selection"][
        "invariance_sample_size"
    ]
    assert v2["thresholds"] == v1["thresholds"]
    assert v2["teachers"]["primary"] == v1["teachers"]["primary"]
    assert v1["teachers"]["disagreement"]["architecture"] == "M3GNet"
    assert v2["teachers"]["disagreement"]["architecture"] == "QET"


def test_checkpoint_manifest_rehash_rejects_mutated_weight(tmp_path):
    teacher_dir = tmp_path / "primary"
    teacher_dir.mkdir()
    weight = teacher_dir / "state.pt"
    weight.write_bytes(b"frozen")
    import hashlib

    manifest = {
        "teachers": {
            "primary": {
                "local_dir": "primary",
                "files": [
                    {
                        "path": "state.pt",
                        "sha256": hashlib.sha256(b"frozen").hexdigest(),
                    }
                ],
            }
        }
    }
    assert _checkpoint_files_match(tmp_path, manifest)
    weight.write_bytes(b"mutated")
    assert not _checkpoint_files_match(tmp_path, manifest)
