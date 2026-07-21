import json
from pathlib import Path

import torch

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
