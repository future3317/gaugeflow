"""Build a pinned TensorOrbit-JARVIS-v2 cache from explicit raw records.

The script intentionally cannot fetch an unspecified "latest JARVIS" release.
It consumes a raw JSON/JSONL export plus a release manifest whose download hash
is checked, so data acquisition, excluded records, Voigt conventions, tensor
units and Cartesian projection are all reproducible rather than inferred from
an old cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pymatgen.core import Structure

from gaugeflow.data import SYMMETRY_TARGET_CACHE_SCHEMA, _target_cache_file
from gaugeflow.provenance import canonicalize_engineering_piezo_voigt, reynolds_project_proper_rank3
from gaugeflow.stabilizer import proper_stabilizer_rotations


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("raw records must be a JSON list/object.records or JSONL dictionaries")
    return records


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = {name: [str(value) for value in payload[name]] for name in ("train", "val", "test")}
    if {name: len(values) for name, values in split.items()} != {"train": 4000, "val": 499, "test": 499}:
        raise ValueError("v2 raw builder requires the registered 4000/499/499 candidate split")
    flattened = [value for values in split.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("split material IDs must be globally unique")
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/tensororbit_jarvis_v2_raw_build_v1.json"))
    parser.add_argument("--raw-records", type=Path, required=True)
    parser.add_argument("--raw-release-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json"))
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tensororbit_jarvis_v2"))
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol).read_text(encoding="utf-8"))
    if protocol.get("name") != "TensorOrbit-JARVIS-v2 raw acquisition and target-cache build v1":
        raise ValueError("raw builder must be invoked with its matching versioned protocol")
    release = json.loads(args.raw_release_manifest.read_text(encoding="utf-8"))
    required_manifest = set(protocol["required_raw_release_manifest"]["fields"])
    missing_manifest = sorted(required_manifest.difference(release))
    if missing_manifest:
        raise ValueError(f"raw release manifest is missing: {missing_manifest}")
    if release["download_sha256"] != sha256_file(args.raw_records):
        raise ValueError("raw record file hash differs from the pinned release manifest")
    split = load_split(args.split)
    expected_ids = {material_id for values in split.values() for material_id in values}
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    if not isinstance(exclusions, dict) or not all(isinstance(value, str) and value.strip() for value in exclusions.values()):
        raise ValueError("exclusions must map each excluded material ID to a nonempty reason")
    records = load_raw_records(args.raw_records)
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        material_id = str(record.get("material_id", ""))
        if not material_id:
            raise ValueError("raw record lacks material_id")
        if material_id in keyed:
            raise ValueError(f"duplicate raw material_id: {material_id}")
        keyed[material_id] = record
    raw_ids = set(keyed)
    missing = sorted(expected_ids - raw_ids)
    unexpected = sorted(raw_ids - expected_ids)
    if missing:
        raise ValueError(f"registered split IDs are missing from raw release: {missing[:5]}")
    if set(unexpected) != set(exclusions):
        omitted = sorted(set(unexpected) - set(exclusions))
        stale = sorted(set(exclusions) - set(unexpected))
        raise ValueError(f"exclusion coverage mismatch; missing={omitted[:5]}, stale={stale[:5]}")

    output_dir = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    csv_dir, target_dir = output_dir / "piezo", output_dir / "reynolds_projected_targets"
    csv_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    required_record = {"material_id", "cif", "piezo_voigt", "voigt_order", "engineering_shear", "unit"}
    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in split}
    target_index: dict[str, str] = {}
    zero_count = 0
    for split_name, ids in split.items():
        for material_id in ids:
            record = keyed[material_id]
            missing_fields = sorted(required_record.difference(record))
            if missing_fields:
                raise ValueError(f"{material_id} raw record missing {missing_fields}")
            if str(record["unit"]) != str(release["tensor_unit"]):
                raise ValueError(f"{material_id} unit differs from pinned release manifest")
            source = torch.tensor(record["piezo_voigt"], dtype=torch.float32)
            canonical = canonicalize_engineering_piezo_voigt(
                source, record["voigt_order"], engineering_shear=bool(record["engineering_shear"])
            )
            structure = Structure.from_str(str(record["cif"]), fmt="cif")
            rotations = proper_stabilizer_rotations(structure)
            target, residual = reynolds_project_proper_rank3(canonical, rotations)
            if not torch.allclose(target, target.transpose(-1, -2), atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"{material_id} projection lost strain-index symmetry")
            cache_path = _target_cache_file(target_dir, material_id)
            torch.save(
                {
                    "schema": SYMMETRY_TARGET_CACHE_SCHEMA,
                    "target": target.cpu(),
                    "rotations": rotations.cpu(),
                    "residual": float(residual),
                    "canonical_voigt": canonical.cpu(),
                },
                cache_path,
            )
            target_index[material_id] = sha256_file(cache_path)
            zero_count += int(torch.count_nonzero(target) == 0)
            rows_by_split[split_name].append(
                {
                    "material_id": material_id,
                    "cif": record["cif"],
                    "source_piezo_voigt": json.dumps(record["piezo_voigt"]),
                    "source_voigt_order": json.dumps(record["voigt_order"]),
                    "source_engineering_shear": bool(record["engineering_shear"]),
                    "source_unit": record["unit"],
                    "canonical_voigt_order": json.dumps(["xx", "yy", "zz", "yz", "xz", "xy"]),
                    "raw_record_sha256": canonical_hash(record),
                }
            )
    source_files = {}
    for name, rows in rows_by_split.items():
        destination = csv_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(destination, index=False)
        source_files[name] = sha256_file(destination)
    exclusions_path = output_dir / "exclusions_applied.json"
    exclusions_path.write_text(json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": "built_not_oracle_qualified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol),
        "raw_release_manifest_sha256": sha256_file(args.raw_release_manifest),
        "raw_records_sha256": sha256_file(args.raw_records),
        "split_sha256": sha256_file(args.split),
        "split_counts": {name: len(values) for name, values in split.items()},
        "excluded_record_count": len(exclusions),
        "exclusions_sha256": sha256_file(exclusions_path),
        "output_csv_sha256": source_files,
        "target_cache_sha256": canonical_hash(target_index),
        "physical_zero_target_count": zero_count,
        "tensor_convention": "canonical engineering Voigt [xx,yy,zz,yz,xz,xy] -> Cartesian ijk=ikj -> proper-SO(3) Reynolds projection",
    }
    (output_dir / "raw_release_manifest.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    (output_dir / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "tensororbit_jarvis_v2_raw_build_report.md"
    report.write_text(
        "# TensorOrbit-JARVIS-v2 raw build\n\n"
        f"Status: `{build_manifest['status']}`. This build does not qualify an external oracle or a GaugeFlow generator.\n\n"
        f"- Split counts: `{build_manifest['split_counts']}`\n"
        f"- Explicit exclusions: `{build_manifest['excluded_record_count']}`\n"
        f"- Physical zero targets retained: `{zero_count}`\n"
        f"- Raw release SHA-256: `{build_manifest['raw_records_sha256']}`\n"
        f"- Target-cache index SHA-256: `{build_manifest['target_cache_sha256']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(build_manifest, indent=2))


if __name__ == "__main__":
    main()
