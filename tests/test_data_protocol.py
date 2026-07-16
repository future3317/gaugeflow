import hashlib
import json

import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

from gaugeflow.data import PiezoCrystalDataset
from gaugeflow.tensor import piezo_from_irreps


def _cache_path(cache_dir, material_id: str):
    digest = hashlib.sha256(material_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.pt"


def test_piezojet_v2_cache_and_manifest_are_used(tmp_path):
    material_ids = ["material-b", "material-a"]
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0.0, 0.0, 0.0]])
    pd.DataFrame(
        {
            "material_id": material_ids,
            "cif": [structure.to(fmt="cif")] * 2,
            # A deliberately incorrect raw column makes cache precedence observable.
            "piezo_irreps_raw": [json.dumps([3.0] * 18)] * 2,
        }
    ).to_csv(tmp_path / "paired.csv", index=False)
    manifest = tmp_path / "splits_formula_stratified_v2.json"
    manifest.write_text(json.dumps({"train": ["material-a", "material-b"]}))
    cache_dir = tmp_path / "piezo_symmetry_targets_v2"
    cache_dir.mkdir()
    zero = torch.zeros(3, 3, 3)
    large = zero.clone()
    large[0, 0, 0] = 2.0
    for material_id, target in (("material-a", zero), ("material-b", large)):
        torch.save(
            {"schema": 2, "target": target, "rotations": torch.eye(3).unsqueeze(0), "residual": 0.0},
            _cache_path(cache_dir, material_id),
        )

    dataset = PiezoCrystalDataset(
        tmp_path / "paired.csv",
        split_manifest=manifest,
        split="train",
        target_cache_dir=cache_dir,
    )
    assert dataset.frame.material_id.tolist() == ["material-a", "material-b"]
    assert dataset.condition_bins().tolist() == [0, 4]
    assert torch.allclose(piezo_from_irreps(dataset.condition_irreps()[1]), large, atol=1e-5)
    assert not torch.allclose(dataset.condition_irreps()[0], torch.full((18,), 3.0))
    record = dataset[0]
    assert record.material_id == "material-a"
    assert record.atom_types.tolist() == [13]
    assert not hasattr(record, "stabilizer_rotations")
    assert not hasattr(record, "tensor_stabilizer_rotations")


def test_structure_only_dataset_does_not_require_or_emit_tensor_condition(tmp_path):
    structure = Structure(Lattice.cubic(4.0), ["Si", "O"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    pd.DataFrame(
        {"material_id": ["structure-only"], "cif": [structure.to(fmt="cif")]}
    ).to_csv(tmp_path / "structures.csv", index=False)
    dataset = PiezoCrystalDataset(tmp_path / "structures.csv", condition_column=None)
    record = dataset[0]
    assert torch.equal(record.piezo_irreps, torch.zeros((1, 18)))
    assert not bool(record.condition_present.item())
    assert record.response_stratum.item() == -1
    try:
        dataset.condition_irreps()
    except RuntimeError as error:
        assert "disabled" in str(error)
    else:
        raise AssertionError("structure-only data must not expose tensor conditions")
