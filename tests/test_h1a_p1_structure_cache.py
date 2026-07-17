from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from scripts.audit_h1a_p1_structure_cache import audit_cache, independent_reduce
from scripts.build_h1a_p1_structure_cache import build_cache, reduce_alex_row


def _structure(
    lattice: np.ndarray, fractional: np.ndarray, numbers: list[int]
) -> dict[str, object]:
    return {
        "positions": (fractional @ lattice).tolist(),
        "cell": lattice.tolist(),
        "atomic_numbers": numbers,
    }


def test_builder_and_independent_niggli_formulations_agree():
    base = np.array(
        [[3.1, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.3, 4.2]], dtype=np.float64
    )
    change = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    lattice = change @ base
    fractional = np.array([[0.1, 0.2, 0.3], [0.7, 0.5, 0.9]])
    row = _structure(lattice, fractional, [8, 14])
    built = reduce_alex_row(
        row["positions"], row["cell"], row["atomic_numbers"]  # type: ignore[arg-type]
    )
    audited = independent_reduce(
        row["positions"],  # type: ignore[arg-type]
        row["cell"],  # type: ignore[arg-type]
        row["atomic_numbers"],  # type: ignore[arg-type]
        epsilon=1e-5,
    )
    assert np.array_equal(built.tokens, audited.tokens)
    assert np.array_equal(built.fractional, audited.fractional)
    assert np.array_equal(built.lattice, audited.lattice)
    assert np.array_equal(built.transform, audited.transform)
    assert built.cache_equivalence_error <= 1e-5


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(
        pa.table(
            {
                "positions": [row["positions"] for row in rows],
                "cell": [row["cell"] for row in rows],
                "atomic_numbers": [row["atomic_numbers"] for row in rows],
                "material_id": [row["material_id"] for row in rows],
            }
        ),
        path,
    )


def _write_tiny_protocol(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw/huggingface/Alex-MP-20"
    processed = data_root / "processed/gaugeflow_h0_v2"
    o1 = data_root / "processed/gaugeflow_h0_v8/occupational_order_o1_v1"
    raw_root.mkdir(parents=True)
    processed.mkdir(parents=True)
    o1.mkdir(parents=True)
    identity = np.eye(3) * 3.0
    rows = {
        "train": [
            {
                **_structure(identity, np.array([[0.1, 0.2, 0.3]]), [6]),
                "material_id": "train-a",
            },
            {
                **_structure(
                    np.array([[2.5, 0.0, 0.0], [0.3, 3.2, 0.0], [0.1, 0.2, 4.0]]),
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                    [8, 14],
                ),
                "material_id": "train-b",
            },
        ],
        "val": [
            {
                **_structure(identity, np.array([[0.4, 0.3, 0.2]]), [7]),
                "material_id": "val-a",
            }
        ],
        "test": [
            {
                **_structure(identity, np.array([[0.8, 0.1, 0.6]]), [13]),
                "material_id": "test-a",
            }
        ],
    }
    raw_records = []
    for split, split_rows in rows.items():
        path = raw_root / f"{split}.parquet"
        _write_raw(path, split_rows)
        raw_records.append(
            {
                "source_split": split,
                "path": f"raw/huggingface/Alex-MP-20/{split}.parquet",
                "rows": len(split_rows),
                "sha256": sha256_file(path),
            }
        )
    assignment_path = processed / "assignments.parquet"
    pq.write_table(
        pa.table(
            {
                "material_id": ["train-a", "train-b", "val-a", "test-a"],
                "source_split": ["train", "train", "val", "test"],
                "gaugeflow_split": ["val", "train", "test", "train"],
                "primitive_sites": [1, 2, 1, 1],
            }
        ),
        assignment_path,
    )
    split_manifest = processed / "split.json"
    split_manifest.write_text("{}\n", encoding="utf-8")
    split_audit = processed / "split_audit.json"
    split_audit.write_text(
        json.dumps(
            {
                "qualified": True,
                "split_counts": {"train": 2, "val": 1, "test": 1},
            }
        ),
        encoding="utf-8",
    )
    o1_manifest = o1 / "manifest.json"
    o1_manifest.write_text("{}\n", encoding="utf-8")
    o1_audit = o1 / "independent_audit.json"
    o1_audit.write_text("{}\n", encoding="utf-8")
    protocol = {
        "protocol": "h1a_p1_structure_cache_v1",
        "status_before_run": "frozen_not_run",
        "h0_dependency": {
            "o1_manifest_sha256": sha256_file(o1_manifest),
            "o1_independent_audit_sha256": sha256_file(o1_audit),
            "archive_tag": "archive/h0-e-v4-o1-v1-qualified-20260718",
        },
        "source": {
            "raw_files": raw_records,
            "assignment_path": "processed/gaugeflow_h0_v2/assignments.parquet",
            "assignment_sha256": sha256_file(assignment_path),
            "split_manifest_path": "processed/gaugeflow_h0_v2/split.json",
            "split_manifest_sha256": sha256_file(split_manifest),
            "split_audit_path": "processed/gaugeflow_h0_v2/split_audit.json",
            "split_audit_sha256": sha256_file(split_audit),
        },
        "selection": {
            "gaugeflow_split_counts": {"train": 2, "val": 1, "test": 1},
            "gaugeflow_split_node_counts": {"train": 3, "val": 1, "test": 1},
        },
        "canonical_cell": {"epsilon": 1e-5},
        "thresholds": {
            "source_rows": 4,
            "source_cartesian_equivalence_max_error_angstrom": 1e-7,
            "float32_cache_cartesian_equivalence_max_error_angstrom": 1e-5,
        },
        "required_outputs": {"root": "processed/cache"},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol_path, data_root


def test_tiny_cache_build_and_reverse_order_independent_audit(tmp_path: Path):
    protocol_path, data_root = _write_tiny_protocol(tmp_path)
    output_root = data_root / "processed/cache"
    manifest = build_cache(
        protocol_path=protocol_path,
        data_root=data_root,
        output_root=output_root,
        workers=2,
        batch_size=2,
    )
    assert manifest["builder_qualified"]
    assert not manifest["qualified"]
    audit = audit_cache(
        protocol_path=protocol_path,
        data_root=data_root,
        output_root=output_root,
        report_path=tmp_path / "report.md",
        batch_size=2,
    )
    assert audit["qualified"]
    assert audit["full_rebuild_matches"] == 4
    dataset = PackedAlexP1Dataset(output_root, "train")
    assert len(dataset) == 2
    assert dataset.node_counts.tolist() == [2, 1]
