from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gaugeflow.production.lemat_data import (
    lemat_stress_kbar_to_kelvin_gpa,
    normalize_external_material_id,
    parse_lemat_row,
)
from gaugeflow.production.lemat_index import IndexedLeMatDataset, build_lemat_index


def _row(index: int, *, compatible: bool = True, forces: object = None) -> dict:
    return {
        "immutable_id": f"agm{index:09d}",
        "entalpic_fingerprint": f"fingerprint-{index}",
        "nsites": 2,
        "functional": "pbe",
        "cross_compatibility": compatible,
        "nperiodic_dimensions": 3,
        "dimension_types": [1, 1, 1],
        "lattice_vectors": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "cartesian_site_positions": [[1.0, 1.5, 2.0], [0.0, 0.0, 0.0]],
        "species_at_sites": ["Si", "O"],
        "energy": -10.0,
        "forces": [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]] if forces is None else forces,
        "stress_tensor": [[1.0, 2.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]],
    }


def test_lemat_parser_converts_geometry_units_and_masks() -> None:
    record = parse_lemat_row(_row(1))
    assert torch.allclose(record.fractional_coordinates[0], torch.full((3,), 0.5))
    assert torch.isclose(record.energy_per_atom_ev, torch.tensor(-5.0))
    expected = lemat_stress_kbar_to_kelvin_gpa(
        [[1.0, 2.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
    )
    assert torch.allclose(record.stress_kelvin_gpa.double(), expected)
    assert record.energy_present and record.forces_present and record.stress_present

    incompatible = parse_lemat_row(_row(2, compatible=False))
    assert not incompatible.energy_present
    assert not incompatible.forces_present
    assert not incompatible.stress_present
    incomplete = parse_lemat_row(_row(3, forces=[]))
    assert incomplete.energy_present and incomplete.stress_present
    assert not incomplete.forces_present


def test_lemat_index_reads_parquet_and_excludes_wrapped_alex_id(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(200)]
    rows[0]["immutable_id"] = "agm-overlap"
    rows[1]["entalpic_fingerprint"] = rows[0]["entalpic_fingerprint"]
    rows[2]["lattice_vectors"] = [
        [0.4, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    rows[3]["lattice_vectors"] = [
        [0.6, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 100.0],
    ]
    rows[4]["lattice_vectors"] = [
        [-2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    parquet = tmp_path / "pbe.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=20)
    root = tmp_path / "index"
    manifest = build_lemat_index(
        {"pbe": [parquet]},
        root,
        excluded_material_ids={"alex<agm-overlap>"},
        excluded_material_ids_artifact_sha256="a" * 64,
        max_row_groups_per_source=10,
    )
    assert manifest["bounded_smoke"] and not manifest["qualified"]
    assert manifest["excluded_external_overlap"] == 2
    assert manifest["excluded_direct_id_rows"] == 1
    assert manifest["excluded_cross_id_fingerprint_rows"] == 1
    assert manifest["excluded_benchmark_fingerprints"] == 1
    assert manifest["excluded_material_ids_count"] == 1
    assert manifest["excluded_material_ids_artifact_sha256"] == "a" * 64
    assert manifest["exclusion_artifact_bound"]
    assert manifest["excluded_degenerate_lattice_rows"] == 3
    assert manifest["excluded_minimum_lattice_width_rows"] == 1
    assert manifest["excluded_lattice_metric_condition_rows"] == 1
    assert manifest["excluded_nonpositive_lattice_volume_rows"] == 1
    assert sum(manifest["split_counts"].values()) == 195
    dataset = IndexedLeMatDataset(root, "train", require_qualified=False)
    assert dataset[0].functional == "pbe"
    assert dataset[0].element_tokens.numel() == 2
    assert dataset.functional_names == ("pbe",)
    assert torch.equal(dataset.functional_group_index, torch.zeros(len(dataset), dtype=torch.long))
    blocks = dataset.sampling_block_index
    assert blocks.shape == (len(dataset),)
    assert int(blocks.min()) == 0
    assert int(blocks.max()) + 1 == torch.unique(blocks).numel()
    assert normalize_external_material_id("alex<AGM001>") == "agm001"
