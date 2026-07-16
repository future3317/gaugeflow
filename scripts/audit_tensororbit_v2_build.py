"""Validate a materialized TensorOrbit-JARVIS-v2 raw build without training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core import Structure

from gaugeflow.data import (
    FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA,
    SYMMETRY_TARGET_CACHE_SCHEMA,
    _target_cache_file,
    response_stratum,
)
from gaugeflow.file_utils import sha256_file
from gaugeflow.tensor import piezo_cartesian_to_voigt, piezo_voigt_to_cartesian, rotate_rank3

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("data/tensororbit_jarvis_v2"))
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("artifacts/tensororbit_jarvis_formula_grouped_candidate_v2/splits.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tensororbit_jarvis_v2_activation_audit"))
    parser.add_argument(
        "--attestation",
        type=Path,
        default=Path("artifacts/tensororbit_jarvis_v2_raw_build_v1/attestation.json"),
    )
    args = parser.parse_args()
    build_dir = ROOT / args.build_dir if not args.build_dir.is_absolute() else args.build_dir
    split_path = ROOT / args.split if not args.split.is_absolute() else args.split
    output_dir = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    attestation_path = ROOT / args.attestation if not args.attestation.is_absolute() else args.attestation
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_path = build_dir / "build_manifest.json"
    build_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_schema = int(build_manifest.get("target_cache_schema", SYMMETRY_TARGET_CACHE_SCHEMA))
    compatibility_scope = str(build_manifest.get("crystal_compatibility_group", "legacy_proper_so3"))
    if expected_schema not in {SYMMETRY_TARGET_CACHE_SCHEMA, FULL_O3_SYMMETRY_TARGET_CACHE_SCHEMA}:
        raise ValueError(f"Unsupported target cache schema {expected_schema}")
    if compatibility_scope not in {"legacy_proper_so3", "full_o3_crystal_point_group"}:
        raise ValueError(f"Unknown crystal compatibility scope {compatibility_scope}")
    frames = []
    for name in ("train", "val", "test"):
        frame = pd.read_csv(build_dir / "piezo" / f"{name}.csv")
        frame["split"] = name
        expected = [str(value) for value in split[name]]
        if frame.material_id.astype(str).tolist() != expected:
            raise ValueError(f"{name} CSV IDs differ from the frozen split order")
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    if source.material_id.duplicated().any() or len(source) != 4998:
        raise ValueError("v2 build must have exactly 4,998 unique material IDs")
    formula_splits: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    rows: list[dict[str, object]] = []
    for record in source.itertuples(index=False):
        material_id = str(record.material_id)
        structure = Structure.from_str(record.cif, fmt="cif")
        formula = structure.composition.reduced_formula
        formula_splits[str(record.split)].add(formula)
        cache_path = _target_cache_file(build_dir / "reynolds_projected_targets", material_id)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        target, rotations = torch.as_tensor(payload["target"]), torch.as_tensor(payload["rotations"])
        if payload.get("schema") != expected_schema or target.shape != (3, 3, 3):
            raise ValueError(f"{material_id}: cache schema/target shape is invalid")
        if compatibility_scope == "full_o3_crystal_point_group":
            if payload.get("crystal_compatibility_group") != compatibility_scope:
                raise ValueError(f"{material_id}: cache did not record full-O(3) compatibility")
            determinant = torch.linalg.det(rotations)
            if not torch.allclose(determinant.abs(), torch.ones_like(determinant), atol=1e-4, rtol=1e-4):
                raise ValueError(f"{material_id}: full-O(3) cache contains a non-orthogonal operation")
        symmetry_error = float((target - target.transpose(-1, -2)).abs().max())
        reynolds_error = float((rotate_rank3(target, rotations) - target).abs().max())
        voigt_error = float((piezo_voigt_to_cartesian(piezo_cartesian_to_voigt(target)) - target).abs().max())
        if symmetry_error > 2e-6 or reynolds_error > 5e-4 or voigt_error > 2e-6:
            raise ValueError(f"{material_id}: tensor conversion/projection audit failed")
        norm = float(torch.linalg.vector_norm(target))
        rows.append(
            {
                "material_id": material_id,
                "split": record.split,
                "formula": formula,
                "atom_count": len(structure),
                "tensor_norm": norm,
                "response_stratum": response_stratum(norm),
                "physical_zero": bool(torch.count_nonzero(target) == 0),
                "target_cache_sha256": sha256_file(cache_path),
            }
        )
    overlaps = {
        "train_val": sorted(formula_splits["train"] & formula_splits["val"]),
        "train_test": sorted(formula_splits["train"] & formula_splits["test"]),
        "val_test": sorted(formula_splits["val"] & formula_splits["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"formula-disjoint v2 split leaked: {overlaps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "raw_build_rows.csv"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    summary = {
        "schema": 1,
        "status": "raw_build_audit_passed_oracle_still_unqualified",
        "split_counts": Counter(source["split"]),
        "formula_overlap": overlaps,
        "physical_zero_counts": Counter(row["split"] for row in rows if row["physical_zero"]),
        "response_strata": {
            name: dict(Counter(row["response_stratum"] for row in rows if row["split"] == name))
            for name in ("train", "val", "test")
        },
        "rows_sha256": sha256_file(rows_path),
        "build_manifest_sha256": sha256_file(build_dir / "build_manifest.json"),
        "target_cache_schema": expected_schema,
        "crystal_compatibility_group": compatibility_scope,
        "split_sha256": sha256_file(split_path),
    }
    (output_dir / "raw_build_activation_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "raw_build_activation_report.md").write_text(
        "# TensorOrbit-JARVIS-v2 raw-build activation audit\n\n"
        f"Status: `{summary['status']}`. Formula overlaps are `{overlaps}`. "
        "This verifies the source-pinned local build only; it does not qualify GMTNet, "
        "the architecture-distinct oracle, or GaugeFlow generation.\n",
        encoding="utf-8",
    )
    source_release = json.loads((build_dir / "raw_release_manifest.json").read_text(encoding="utf-8"))
    exclusions = json.loads((build_dir / "exclusions_applied.json").read_text(encoding="utf-8"))
    attestation = {
        "schema": 1,
        "name": "TensorOrbit-JARVIS-v2 raw-build attestation v1",
        "status": summary["status"],
        "source": {
            key: source_release[key]
            for key in (
                "source_name", "source_release", "source_url", "source_copy_status", "retrieved_utc",
                "source_pickle_sha256", "download_sha256", "record_count", "tensor_unit",
                "source_voigt_order", "engineering_shear", "frozen_split_sha256",
            )
        },
        "build": {
            key: build_manifest[key]
            for key in (
                "protocol_sha256", "raw_release_manifest_sha256", "raw_records_sha256",
                "split_sha256", "split_counts", "excluded_record_count", "exclusions_sha256",
                "output_csv_sha256", "target_cache_sha256", "physical_zero_target_count", "tensor_convention",
            )
        },
        "exclusions": exclusions,
        "audit": summary,
        "limitations": (
            ([] if source_release.get("retrieved_utc") else [
                "The raw source direct-download timestamp is unavailable."
            ])
            + ["This attestation does not qualify either external tensor oracle or any GaugeFlow generation result."]
        ),
    }
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.write_text(json.dumps(attestation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
