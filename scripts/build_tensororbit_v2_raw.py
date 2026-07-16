"""Build a pinned TensorOrbit-JARVIS-v2 cache from explicit raw records.

The script intentionally cannot fetch an unspecified "latest JARVIS" release.
It consumes a raw JSON/JSONL export plus a release manifest whose download hash
is checked, so data acquisition, excluded records, Voigt conventions, tensor
units and Cartesian projection are all reproducible rather than inferred from
an old cache.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pymatgen.core import Structure

from gaugeflow.data import (
    FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA,
    SYMMETRY_TARGET_CACHE_SCHEMA,
    _target_cache_file,
)
from gaugeflow.file_utils import canonical_json_hash, sha256_file
from gaugeflow.provenance import (
    canonicalize_engineering_piezo_voigt,
    reynolds_project_crystal_rank3,
    reynolds_project_proper_rank3,
)
from gaugeflow.stabilizer import crystal_point_group_operations, proper_stabilizer_rotations

ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json"),
    )
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Override the protocol-declared output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only schema- and canonical-Voigt-verified target files from an interrupted identical build.",
    )
    parser.add_argument(
        "--cache-only-new-target-limit",
        type=int,
        help="Build at most this many missing target files, then exit without writing an incomplete manifest/CSV.",
    )
    parser.add_argument(
        "--cache-only-start-index",
        type=int,
        help="Inclusive index in the fixed train/val/test target order.",
    )
    parser.add_argument(
        "--cache-only-stop-index",
        type=int,
        help="Exclusive index in the fixed train/val/test target order.",
    )
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_name = protocol.get("name")
    if protocol_name not in {
        "TensorOrbit-JARVIS-v2 raw acquisition and target-cache build v1",
        "TensorOrbit-JARVIS-v2 raw acquisition and full-O(3) target-cache build v2",
    }:
        raise ValueError("raw builder must be invoked with its matching versioned protocol")
    compatibility_scope = str(protocol.get("crystal_compatibility_group", "legacy_proper_so3"))
    cache_schema = int(protocol.get("target_cache_schema", SYMMETRY_TARGET_CACHE_SCHEMA))
    if compatibility_scope == "legacy_proper_so3":
        if cache_schema != SYMMETRY_TARGET_CACHE_SCHEMA:
            raise ValueError("legacy proper-SO(3) builds must retain schema 2")
    elif compatibility_scope == "full_o3_crystal_point_group":
        if cache_schema != FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA:
            raise ValueError("full-O(3) crystal builds must use schema 3")
    else:
        raise ValueError("crystal_compatibility_group must be legacy_proper_so3 or full_o3_crystal_point_group")
    if args.cache_only_new_target_limit is not None:
        if not args.resume or args.cache_only_new_target_limit < 1:
            raise ValueError("cache-only target limits require --resume and a positive limit")
    if (args.cache_only_start_index is None) != (args.cache_only_stop_index is None):
        raise ValueError("cache-only start/stop indices must be supplied together")
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
    valid_exclusions = isinstance(exclusions, dict) and all(
        isinstance(value, str) and value.strip() for value in exclusions.values()
    )
    if not valid_exclusions:
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

    configured_output = Path(str(protocol.get("default_output_dir", "data/tensororbit_jarvis_v2")))
    requested_output = args.output_dir or configured_output
    output_dir = ROOT / requested_output if not requested_output.is_absolute() else requested_output
    csv_dir, target_dir = output_dir / "piezo", output_dir / "reynolds_projected_targets"
    csv_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    required_record = {"material_id", "cif", "piezo_voigt", "voigt_order", "engineering_shear", "unit"}
    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in split}
    target_index: dict[str, str] = {}
    zero_count = 0
    newly_built = 0
    ordered_targets = [(split_name, material_id) for split_name, ids in split.items() for material_id in ids]
    if args.cache_only_start_index is not None:
        if not args.resume:
            raise ValueError("cache-only start/stop indices require --resume")
        start, stop = args.cache_only_start_index, args.cache_only_stop_index
        if start < 0 or stop <= start or stop > len(ordered_targets):
            raise ValueError(f"cache-only indices must satisfy 0 <= start < stop <= {len(ordered_targets)}")
        ordered_targets = ordered_targets[start:stop]
    for split_name, material_id in ordered_targets:
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
            cache_path = _target_cache_file(target_dir, material_id)
            cached: dict[str, Any] | None = None
            if args.resume and cache_path.is_file():
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                except TypeError:
                    cached = torch.load(cache_path, map_location="cpu")
                cached_canonical = torch.as_tensor(cached.get("canonical_voigt"), dtype=canonical.dtype)
                if (
                    cached.get("schema") != cache_schema
                    or cached_canonical.shape != (3, 6)
                    or not torch.allclose(cached_canonical, canonical, atol=1e-7, rtol=1e-7)
                ):
                    raise ValueError(f"{material_id}: existing target cache cannot be resumed safely")
            if cached is None:
                structure = Structure.from_str(str(record["cif"]), fmt="cif")
                try:
                    if compatibility_scope == "full_o3_crystal_point_group":
                        rotations = crystal_point_group_operations(structure, proper_only=False)
                        target, residual = reynolds_project_crystal_rank3(canonical, rotations)
                        projection_label = "full_o3_crystal_point_group_reynolds_v2"
                    else:
                        rotations = proper_stabilizer_rotations(structure)
                        target, residual = reynolds_project_proper_rank3(canonical, rotations)
                        projection_label = "legacy_proper_so3_reynolds_v1"
                except ValueError as error:
                    raise ValueError(f"{material_id}: {error}") from error
                torch.save(
                    {
                        "schema": cache_schema,
                        "target": target.cpu(),
                        "rotations": rotations.cpu(),
                        "residual": float(residual),
                        "canonical_voigt": canonical.cpu(),
                        "rotation_projection": projection_label,
                        "crystal_compatibility_group": compatibility_scope,
                        "tensor_orbit_group": "proper_so3_only",
                    },
                    cache_path,
                )
                newly_built += 1
            else:
                target = torch.as_tensor(cached["target"], dtype=canonical.dtype)
                residual = torch.as_tensor(cached["residual"], dtype=canonical.dtype)
            if not torch.allclose(target, target.transpose(-1, -2), atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"{material_id} projection lost strain-index symmetry")
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
                    "raw_record_sha256": canonical_json_hash(record),
                }
            )
            if args.cache_only_new_target_limit is not None and newly_built >= args.cache_only_new_target_limit:
                print(json.dumps({"status": "partial_target_cache_complete", "newly_built": newly_built}, indent=2))
                return
    if args.cache_only_start_index is not None:
        print(
            json.dumps(
                {
                    "status": "partial_target_cache_range_complete",
                    "start": args.cache_only_start_index,
                    "stop": args.cache_only_stop_index,
                    "newly_built": newly_built,
                },
                indent=2,
            )
        )
        return
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
        "target_cache_sha256": canonical_json_hash(target_index),
        "physical_zero_target_count": zero_count,
        "target_cache_schema": cache_schema,
        "crystal_compatibility_group": compatibility_scope,
        "tensor_orbit_group": "proper_so3_only",
        "tensor_convention": (
            "canonical engineering Voigt [xx,yy,zz,yz,xz,xy] -> Cartesian ijk=ikj -> "
            + (
                "full-O(3) crystal-point-group Reynolds projection"
                if compatibility_scope == "full_o3_crystal_point_group"
                else "legacy proper-SO(3) Reynolds projection"
            )
        ),
    }
    (output_dir / "raw_release_manifest.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    (output_dir / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8")
    configured_report = Path(str(protocol.get("report_path", "reports/tensororbit_jarvis_v2_raw_build_report.md")))
    report = ROOT / configured_report if not configured_report.is_absolute() else configured_report
    report.write_text(
        "# TensorOrbit-JARVIS-v2 raw build\n\n"
        f"Status: `{build_manifest['status']}`. This build does not qualify an external oracle "
        "or a GaugeFlow generator.\n\n"
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
