from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset


def _write_cache(root: Path) -> None:
    root.mkdir()
    tensor_path = root / "train.pt"
    index_path = root / "train_index.parquet"
    torch.save(
        {
            "schema": 1,
            "atom_tokens": torch.tensor([4, 7, 12], dtype=torch.uint8),
            "fractional_coordinates": torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4]],
                dtype=torch.float32,
            ),
            "lattice": torch.stack((2.0 * torch.eye(3), 3.0 * torch.eye(3))),
            "offsets": torch.tensor([0, 2, 3], dtype=torch.long),
            "niggli_transform": torch.eye(3, dtype=torch.int16)
            .expand(2, -1, -1)
            .clone(),
        },
        tensor_path,
    )
    pq.write_table(
        pa.table(
            {
                "material_id": ["a", "b"],
                "cache_row": [0, 1],
            }
        ),
        index_path,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "h1a_p1_structure_cache_v1",
                "qualified": True,
                "splits": {
                    "train": {
                        "rows": 2,
                        "nodes": 3,
                        "tensor_file": tensor_path.name,
                        "tensor_sha256": sha256_file(tensor_path),
                        "index_file": index_path.name,
                        "index_sha256": sha256_file(index_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_packed_alex_dataset_exposes_only_model_state_by_default(tmp_path: Path):
    _write_cache(tmp_path / "cache")
    dataset = PackedAlexP1Dataset(tmp_path / "cache", "train")
    assert len(dataset) == 2
    assert torch.equal(dataset.node_counts, torch.tensor([2, 1]))
    first = dataset[0]
    assert first.atom_types.tolist() == [4, 7]
    assert first.frac_coords.shape == (2, 3)
    assert first.lattice.shape == (1, 3, 3)
    assert "material_id" not in first
    assert "niggli_transform" not in first


def test_packed_alex_dataset_can_load_ids_for_offline_evaluation(tmp_path: Path):
    _write_cache(tmp_path / "cache")
    dataset = PackedAlexP1Dataset(
        tmp_path / "cache", "train", include_material_id=True
    )
    assert dataset[0].material_id == "a"
    assert dataset[-1].material_id == "b"


def test_packed_alex_dataset_rejects_unqualified_manifest(tmp_path: Path):
    root = tmp_path / "cache"
    _write_cache(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["qualified"] = False
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        PackedAlexP1Dataset(root, "train")
    except ValueError as error:
        assert "not qualified" in str(error)
    else:
        raise AssertionError("unqualified packed cache was accepted")

