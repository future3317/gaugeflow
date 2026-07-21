import json
from pathlib import Path

import pytest
import torch
from build_matpes_teacher_feature_cache import _iter_completed_rows

from gaugeflow.production.matpes_index import IndexedMatPESDataset, build_matpes_index
from gaugeflow.production.teacher_feature_cache import (
    MatPESTeacherFeatureCache,
    write_matpes_teacher_feature_cache,
)


def _row(index: int) -> dict:
    return {
        "matpes_id": f"teacher-{index}",
        "functional": "PBE",
        "nsites": 2,
        "structure": {
            "lattice": {"matrix": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]},
            "sites": [
                {"species": [{"element": "Na", "occu": 1.0}], "abc": [0.0, 0.0, 0.0]},
                {"species": [{"element": "Cl", "occu": 1.0}], "abc": [0.5, 0.5, 0.5]},
            ],
        },
        "cohesive_energy_per_atom": -2.0,
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "stress": [0.0] * 6,
    }


def test_teacher_cache_is_index_aligned_and_keeps_ids_out_of_batches(tmp_path: Path) -> None:
    source = tmp_path / "pbe.jsonl"
    source.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(50)), encoding="utf-8")
    index = tmp_path / "index"
    build_matpes_index({"PBE": [source]}, index, max_rows_per_source=50)
    teacher_manifest = tmp_path / "teacher.json"
    teacher_manifest.write_text('{"qualified": true}\n', encoding="utf-8")
    checkpoint_manifest = tmp_path / "checkpoint.json"
    checkpoint_manifest.write_text('{"qualified": true}\n', encoding="utf-8")
    features = [
        (row, torch.full((2, 3), float(row)) if row % 2 == 0 else None)
        for row in range(50)
    ]
    cache_root = tmp_path / "cache"
    manifest = write_matpes_teacher_feature_cache(
        cache_root,
        features,
        row_count=50,
        feature_dim=3,
        index_manifest=index / "manifest.json",
        teacher_manifest=teacher_manifest,
        teacher_checkpoint_manifest=checkpoint_manifest,
        teacher_model_sha256="0" * 64,
        functional_scope=("PBE",),
        expected_feature_rows=25,
        bounded_smoke=True,
    )
    assert not manifest["qualified"] and manifest["feature_rows"] == 25
    cache = MatPESTeacherFeatureCache(
        cache_root,
        index_manifest=index / "manifest.json",
        require_qualified=False,
    )
    assert torch.equal(cache.get(0, 2), torch.zeros(2, 3))
    assert cache.get(1, 2) is None
    dataset = IndexedMatPESDataset(
        index,
        "train",
        require_qualified=False,
        teacher_feature_cache=cache_root,
        require_qualified_teacher_cache=False,
    )
    for local_row in range(len(dataset)):
        record = dataset[local_row]
        global_row = int(dataset.indices[local_row])
        if global_row % 2 == 0:
            assert record.teacher_features is not None
            assert torch.equal(
                record.teacher_features,
                torch.full((2, 3), float(global_row)),
            )
        else:
            assert record.teacher_features is None


def test_teacher_shard_resume_fails_closed_on_provenance_change(tmp_path: Path) -> None:
    contract = {
        "index_manifest_sha256": "a" * 64,
        "teacher_manifest_sha256": "b" * 64,
        "teacher_model_sha256": "c" * 64,
        "functional": "PBE",
        "feature_dim": 3,
        "graphs_per_batch": 2,
        "nodes_per_batch": 8,
    }
    part = tmp_path / "part.pt"
    torch.save(
        {
            "schema": 2,
            "contract": contract,
            "start": 0,
            "stop": 2,
            "node_offsets": torch.tensor([0, 1, 1]),
            "features": torch.ones(1, 3, dtype=torch.float16),
        },
        part,
    )
    rows = list(
        _iter_completed_rows([part], row_count=2, feature_dim=3, contract=contract)
    )
    assert rows[0][1] is not None and rows[1][1] is None
    changed = dict(contract, teacher_model_sha256="d" * 64)
    with pytest.raises(ValueError, match="sequence"):
        list(_iter_completed_rows([part], row_count=2, feature_dim=3, contract=changed))
