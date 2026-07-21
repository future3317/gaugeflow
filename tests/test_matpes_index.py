import json
from pathlib import Path

from torch.utils.data import DataLoader

from gaugeflow.production.matpes_index import (
    IndexedMatPESDataset,
    MatPESBatchCollator,
    build_matpes_index,
)


def _row(index: int, functional: str) -> dict:
    return {
        "matpes_id": f"shared-{index}",
        "functional": functional,
        "nsites": 1,
        "structure": {
            "lattice": {"matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]},
            "sites": [{"species": [{"element": "Si", "occu": 1.0}], "abc": [0.0, 0.0, 0.0]}],
        },
        "energy": -4.0,
        "cohesive_energy_per_atom": -2.0 - 0.01 * index,
        "forces": [[0.1, 0.2, 0.3]],
        "stress": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
    }


def _write_jsonl(path: Path, functional: str, start: int = 0, stop: int = 200) -> None:
    path.write_text(
        "".join(json.dumps(_row(index, functional)) + "\n" for index in range(start, stop)),
        encoding="utf-8",
    )


def test_matpes_index_groups_functionals_and_seeks_records(tmp_path: Path) -> None:
    pbe = tmp_path / "pbe.jsonl"
    r2scan = tmp_path / "r2scan.jsonl"
    _write_jsonl(pbe, "PBE")
    _write_jsonl(r2scan, "r2SCAN")
    root = tmp_path / "index"
    manifest = build_matpes_index(
        {"PBE": [pbe], "r2SCAN": [r2scan]},
        root,
        max_rows_per_source=200,
    )
    assert manifest["bounded_smoke"] and not manifest["qualified"]
    assert manifest["unique_material_ids"] == 200
    assert sum(manifest["split_counts"].values()) == 400

    train = IndexedMatPESDataset(
        root,
        "train",
        verify_hashes=True,
        require_qualified=False,
    )
    first = train[0]
    assert first.functional in {"PBE", "r2SCAN"}
    assert first.energy_present and first.energy_per_atom_ev < -1.0
    loader = DataLoader(
        train,
        batch_size=4,
        collate_fn=MatPESBatchCollator({"PBE": 0, "r2SCAN": 1}, teacher_dim=3),
    )
    batch = next(iter(loader))
    assert batch.element_tokens.shape == (4,)
    assert batch.functional_index.shape == (4,)
    assert batch.targets.energy_mask.all()


def test_matpes_index_accepts_multiple_artifacts_per_functional(tmp_path: Path) -> None:
    pbe_train = tmp_path / "pbe-train.jsonl"
    pbe_valid = tmp_path / "pbe-valid.jsonl"
    r2scan_train = tmp_path / "r2scan-train.jsonl"
    r2scan_valid = tmp_path / "r2scan-valid.jsonl"
    _write_jsonl(pbe_train, "PBE", 0, 100)
    _write_jsonl(pbe_valid, "PBE", 100, 200)
    _write_jsonl(r2scan_train, "r2SCAN", 0, 100)
    _write_jsonl(r2scan_valid, "r2SCAN", 100, 200)

    manifest = build_matpes_index(
        {
            "PBE": [pbe_train, pbe_valid],
            "r2SCAN": [r2scan_train, r2scan_valid],
        },
        tmp_path / "multi-index",
        max_rows_per_source=100,
    )

    assert len(manifest["sources"]) == 4
    assert manifest["unique_material_ids"] == 200
    assert sum(manifest["split_counts"].values()) == 400
    assert {source["functional"] for source in manifest["sources"]} == {"PBE", "r2SCAN"}
