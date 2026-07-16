import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.audit_alex_mp20_source import audit_source
from scripts.audit_h0_activation import audit_activation, render_markdown


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_h0_audit_distinguishes_source_presence_from_qualification(tmp_path):
    alex = b"alex"
    phonon_index = b"index"
    matpes = b"matpes"
    for relative, payload in (
        ("alex.parquet", alex),
        ("phonon-index.parquet", phonon_index),
        ("matpes.jsonl", matpes),
    ):
        path = tmp_path / relative
        path.write_bytes(payload)
    derived = {
        "phonopy_version": "4.3.1",
        "fc_calculator": "traditional",
        "posthoc_symmetrized": False,
        "primitive_matrix_policy": "P",
        "audit_imaginary_tolerance_thz": -0.05,
        "counts": {"successful": 1, "failed": 0},
        "validation": {"max_asr_residual": 1e-5},
    }
    derived_path = tmp_path / "phonon-manifest.json"
    derived_bytes = (json.dumps(derived) + "\n").encode()
    derived_path.write_bytes(derived_bytes)
    files = {}
    for relative in ("alex.parquet", "phonon-index.parquet", "phonon-manifest.json", "matpes.jsonl"):
        payload = (tmp_path / relative).read_bytes()
        files[relative] = {"bytes": len(payload), "sha256": _digest(payload)}
    root_manifest = {"files": files}
    root_manifest_path = tmp_path / "MANIFEST.json"
    root_manifest_bytes = json.dumps(root_manifest).encode()
    root_manifest_path.write_bytes(root_manifest_bytes)

    def entry(relative: str) -> dict[str, object]:
        return {"path": relative, **files[relative]}

    config = {
        "protocol": "test_h0",
        "data_center_manifest": "MANIFEST.json",
        "data_center_manifest_sha256": _digest(root_manifest_bytes),
        "h0_a": {
            "source_files": [entry("alex.parquet")],
            "gaugeflow_split_manifest": "missing-split.json",
        },
        "h0_b": {
            "source_files": [entry("phonon-manifest.json"), entry("phonon-index.parquet")],
            "derived_manifest": "phonon-manifest.json",
            "expected_materials": 1,
            "phonopy_version": "4.3.1",
            "required_attestations": [
                "primitive_supercell_mapping",
                "force_constant_solver",
                "symmetrization_policy",
                "acoustic_sum_rule",
                "imaginary_frequency_convention",
                "translational_zero_mode_test",
            ],
        },
        "h0_c": {"source_files": [entry("matpes.jsonl")], "teacher_checkpoint": None},
        "h0_d": {"catalogue_manifest": "missing-catalogue.json"},
        "h0_e": {"pilot_manifest": "missing-pilot.json"},
    }
    result = audit_activation(config, tmp_path)
    assert result["data_center_manifest_matches"] is True
    assert result["components"]["H0-A"]["passed"] is True
    assert result["components"]["H0-A"]["status"] == "blocked_split_not_frozen"
    assert result["components"]["H0-B"]["missing_attestations"] == [
        "translational_zero_mode_test"
    ]
    assert result["components"]["H0-C"]["status"] == "blocked_frozen_teacher_missing"
    assert result["h0_passed"] is False
    assert "H0_not_passed_stop_before_H1" in render_markdown(result)


def test_alex_source_audit_detects_formula_leakage_without_false_corruption(tmp_path):
    def write(split: str, rows: list[dict[str, object]]) -> None:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, tmp_path / f"{split}.parquet")

    def row(material_id: str, atomic_numbers: list[int]) -> dict[str, object]:
        return {
            "positions": [[0.0, 0.0, 0.0] for _ in atomic_numbers],
            "cell": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "atomic_numbers": atomic_numbers,
            "material_id": material_id,
            "space_group": "P1",
        }

    write("train", [row("train-a", [5, 7]), row("train-b", [14])])
    write("val", [row("val-a", [5, 5, 7, 7])])
    write("test", [row("test-a", [8])])
    result = audit_source(tmp_path)
    assert result["source_structure_validity_passed"] is True
    assert result["cross_split_overlap"]["train--val"]["reduced_formula_groups"] == 1
    assert result["cross_split_overlap"]["train--val"]["material_ids"] == 0
    assert result["upstream_split_formula_disjoint"] is False
    assert result["decision"] == "source_valid_but_rebuild_child_split"
