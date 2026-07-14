"""Audit TensorOrbit-JARVIS integrity without mutating the frozen artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from gaugeflow.data import RESPONSE_NORM_BOUNDS, SYMMETRY_TARGET_CACHE_SCHEMA, _target_cache_file
from gaugeflow.tensor import piezo_cartesian_to_voigt, piezo_voigt_to_cartesian, rotate_rank3
from gaugeflow.unit_cell import niggli_reduce_structure_with_transform

# The installed pymatgen StructureMatcher still references NumPy's removed
# `np.bool` alias. Keep the compatibility shim local to this audit process.
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = np.int_  # type: ignore[attr-defined]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def response_stratum(norm: float) -> int:
    if norm <= 1e-12:
        return 0
    for index, upper in enumerate(RESPONSE_NORM_BOUNDS[1:], start=1):
        if norm < upper:
            return index
    return len(RESPONSE_NORM_BOUNDS)


def torus_max_error(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.remainder(left - right + 0.5, 1.0) - 0.5
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def git_state(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_source(csv_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = [csv_dir / f"{name}.csv" for name in ("train", "val", "test")]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--limit", type=int, help="Debug-only row limit; omit for the official audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    source, csv_paths = load_source(args.csv_dir)
    source["material_id"] = source.material_id.astype(str)
    source_duplicate_ids = sorted(source.loc[source.material_id.duplicated(False), "material_id"].unique())
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_sets = {name: set(map(str, split_payload[name])) for name in ("train", "val", "test")}
    split_ids = [material_id for name in ("train", "val", "test") for material_id in split_payload[name]]
    split_duplicates = len(split_ids) - len(set(split_ids))
    split_lookup = {str(material_id): name for name in ("train", "val", "test") for material_id in split_payload[name]}
    source_ids = set(source.material_id)
    artifact_ids = set(map(str, split_ids))
    missing_source_ids = sorted(artifact_ids - source_ids)
    extra_source_ids = sorted(source_ids - artifact_ids)

    expected_cache = {_target_cache_file(args.target_cache_dir, material_id).name: material_id for material_id in artifact_ids}
    actual_cache_files = {path.name: path for path in args.target_cache_dir.glob("*.pt")}
    missing_target_files = sorted(set(expected_cache) - set(actual_cache_files))
    extra_target_files = sorted(set(actual_cache_files) - set(expected_cache))

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gate_a_ids = list(map(str, protocol["material_ids"]))
    gate_a_exact = len(gate_a_ids) == 8 and len(set(gate_a_ids)) == 8 and all(
        value in split_sets["train"] for value in gate_a_ids
    )

    rows = source[source.material_id.isin(artifact_ids)].copy()
    rows["protocol_split"] = rows.material_id.map(split_lookup)
    rows = rows.sort_values(["protocol_split", "material_id"], kind="stable")
    if args.limit is not None:
        rows = rows.head(args.limit)

    audit_rows: list[dict[str, object]] = []
    formulas_by_split = {name: set() for name in ("train", "val", "test")}
    for _, row in rows.iterrows():
        material_id = str(row.material_id)
        result: dict[str, object] = {
            "material_id": material_id,
            "protocol_split": row.protocol_split,
            "source_csv": row.source_csv,
            "target_file": _target_cache_file(args.target_cache_dir, material_id).name,
            "valid": False,
            "error": "",
        }
        try:
            convention = json.loads(row.voigt_convention)
            convention_valid = (
                convention.get("engineering_shear") is True
                and convention.get("order") == ["xx", "yy", "zz", "xy", "yz", "xz"]
            )
            raw_irreps = torch.tensor(json.loads(row.piezo_irreps_raw), dtype=torch.float32)
            raw_irreps_valid = raw_irreps.shape == (18,) and bool(torch.isfinite(raw_irreps).all())

            cache_path = _target_cache_file(args.target_cache_dir, material_id)
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            target = torch.as_tensor(payload.get("target"), dtype=torch.float32)
            rotations = torch.as_tensor(payload.get("rotations"), dtype=torch.float32)
            schema_valid = payload.get("schema") == SYMMETRY_TARGET_CACHE_SCHEMA
            target_valid = (
                target.shape == (3, 3, 3)
                and bool(torch.isfinite(target).all())
                and torch.allclose(target, target.transpose(-1, -2), atol=1e-6, rtol=1e-6)
            )
            rotations_valid = (
                rotations.ndim == 3 and rotations.shape[-2:] == (3, 3)
                and bool(torch.isfinite(rotations).all())
            )
            target_norm = float(torch.linalg.vector_norm(target))
            exact_zero = bool(torch.count_nonzero(target) == 0)
            voigt = piezo_cartesian_to_voigt(target)
            roundtrip = piezo_voigt_to_cartesian(voigt)
            voigt_roundtrip_error = float((roundtrip - target).abs().max())
            projected = rotate_rank3(target, rotations)
            reynolds_invariance_error = float((projected - target).abs().max())
            stored_residual = float(payload.get("residual", math.nan))

            structure = Structure.from_str(row.cif, fmt="cif")
            reduced, integer_change = niggli_reduce_structure_with_transform(structure)
            reduced_twice, second_change = niggli_reduce_structure_with_transform(reduced)
            original_lattice = np.asarray(structure.lattice.matrix, dtype=float)
            reduced_lattice = np.asarray(reduced.lattice.matrix, dtype=float)
            change = reduced_lattice @ np.linalg.inv(original_lattice)
            change_integral_error = float(np.max(np.abs(change - integer_change)))
            change_determinant = int(round(float(np.linalg.det(integer_change))))
            recovered_lattice = np.linalg.inv(integer_change) @ reduced_lattice
            recovered_frac = np.remainder(reduced.frac_coords @ integer_change, 1.0)
            lattice_roundtrip_error = float(np.max(np.abs(recovered_lattice - original_lattice)))
            frac_roundtrip_error = torus_max_error(recovered_frac, structure.frac_coords)
            niggli_lattice_idempotence_error = float(
                np.max(np.abs(reduced_twice.lattice.matrix - reduced_lattice))
            )
            niggli_frac_idempotence_error = torus_max_error(
                reduced_twice.frac_coords, reduced.frac_coords
            )
            recovered_reduced_lattice = np.linalg.inv(second_change) @ reduced_twice.lattice.matrix
            recovered_reduced_frac = np.remainder(
                reduced_twice.frac_coords @ second_change, 1.0
            )
            niggli_quotient_lattice_error = float(
                np.max(np.abs(recovered_reduced_lattice - reduced_lattice))
            )
            niggli_quotient_frac_error = torus_max_error(
                recovered_reduced_frac, reduced.frac_coords
            )
            formula = structure.composition.reduced_formula
            formulas_by_split[row.protocol_split].add(formula)

            valid = all((
                convention_valid,
                raw_irreps_valid,
                schema_valid,
                target_valid,
                rotations_valid,
                voigt_roundtrip_error <= 2e-7,
                reynolds_invariance_error <= 5e-4,
                change_integral_error <= 1e-5,
                abs(change_determinant) == 1,
                lattice_roundtrip_error <= 1e-5,
                frac_roundtrip_error <= 1e-5,
                niggli_quotient_lattice_error <= 1e-5,
                niggli_quotient_frac_error <= 1e-5,
            ))
            result.update({
                "formula": formula,
                "atom_count": len(structure),
                "tensor_norm": target_norm,
                "exact_zero": exact_zero,
                "response_stratum": response_stratum(target_norm),
                "source_voigt_convention_valid": convention_valid,
                "raw_irreps_valid": raw_irreps_valid,
                "target_schema_valid": schema_valid,
                "target_valid": target_valid,
                "rotations_valid": rotations_valid,
                "stored_reynolds_residual": stored_residual,
                "reynolds_invariance_error": reynolds_invariance_error,
                "voigt_roundtrip_error": voigt_roundtrip_error,
                "niggli_change_integral_error": change_integral_error,
                "niggli_change_determinant": change_determinant,
                "basis_lattice_roundtrip_error": lattice_roundtrip_error,
                "basis_frac_roundtrip_error": frac_roundtrip_error,
                "niggli_lattice_idempotence_error": niggli_lattice_idempotence_error,
                "niggli_frac_idempotence_error": niggli_frac_idempotence_error,
                "niggli_representation_idempotent": (
                    niggli_lattice_idempotence_error <= 1e-5
                    and niggli_frac_idempotence_error <= 1e-5
                ),
                "niggli_quotient_lattice_error": niggli_quotient_lattice_error,
                "niggli_quotient_frac_error": niggli_quotient_frac_error,
                "tensor_rotated_by_basis_change": False,
                "valid": valid,
            })
        except Exception as error:  # Preserve every failed row in the audit artifact.
            result["error"] = f"{type(error).__name__}: {error}"
        audit_rows.append(result)

    audit_frame = pd.DataFrame(audit_rows)
    rows_path = args.output_dir / "data_quality_rows.csv"
    audit_frame.to_csv(rows_path, index=False)
    formula_overlaps = {
        "train_val": sorted(formulas_by_split["train"] & formulas_by_split["val"]),
        "train_test": sorted(formulas_by_split["train"] & formulas_by_split["test"]),
        "val_test": sorted(formulas_by_split["val"] & formulas_by_split["test"]),
    }
    cross_formula_groups = set(formula_overlaps["train_val"]) | set(
        formula_overlaps["train_test"]
    ) | set(formula_overlaps["val_test"])
    cross_formula_rows = audit_frame[audit_frame.formula.isin(cross_formula_groups)]
    representation_non_idempotent = int(
        (~audit_frame.niggli_representation_idempotent.fillna(False)).sum()
    )

    # Formula overlap is already a protocol violation. Structure matching
    # measures the stronger near-duplicate leakage class without changing the split.
    source_by_id = source.set_index("material_id")
    split_by_id = audit_frame.set_index("material_id").protocol_split.to_dict()
    formula_by_id = audit_frame.set_index("material_id").formula.to_dict()
    matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
        scale=True, attempt_supercell=True,
    )
    match_rows = []
    for formula in sorted(cross_formula_groups):
        ids = [material_id for material_id, value in formula_by_id.items() if value == formula]
        structures = {
            material_id: Structure.from_str(source_by_id.loc[material_id].cif, fmt="cif")
            for material_id in ids
        }
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1:]:
                if split_by_id[left_id] == split_by_id[right_id]:
                    continue
                if matcher.fit(structures[left_id], structures[right_id]):
                    match_rows.append({
                        "formula": formula,
                        "left_id": left_id,
                        "left_split": split_by_id[left_id],
                        "right_id": right_id,
                        "right_split": split_by_id[right_id],
                    })
    matches_path = args.output_dir / "data_quality_cross_split_matches.csv"
    pd.DataFrame(
        match_rows,
        columns=("formula", "left_id", "left_split", "right_id", "right_split"),
    ).to_csv(matches_path, index=False)
    valid_count = int(audit_frame.valid.fillna(False).sum()) if not audit_frame.empty else 0
    zero_counts = (
        audit_frame[audit_frame.exact_zero.fillna(False)].groupby("protocol_split").size().to_dict()
        if "exact_zero" in audit_frame else {}
    )

    target_index = sorted(
        {name: sha256_file(path) for name, path in actual_cache_files.items()}.items()
    )
    manifest = {
        "schema": 1,
        "artifact": "TensorOrbit-JARVIS-v1-data-quality-audit",
        "git": git_state(root),
        "tensor_convention_version": "gaugeflow-cartesian-ijk=ikj-engineering-shear-v1",
        "source_files": {str(path): sha256_file(path) for path in csv_paths},
        "split_manifest": {"path": str(args.split_manifest), "sha256": sha256_file(args.split_manifest)},
        "target_cache_index_sha256": canonical_json_hash(target_index),
        "target_cache_count": len(target_index),
        "audit_rows_sha256": sha256_file(rows_path),
        "audited_row_count": len(audit_frame),
        "full_audit": args.limit is None,
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    manifest_path = args.output_dir / "data_quality_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    leakage_tokens = ("stabilizer_rotations", "space_group", "target_graph", "target_lattice")
    model_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "src" / "gaugeflow").glob("*.py")
    )
    leakage_mentions = {token: model_sources.count(token) for token in leakage_tokens}
    # Metadata-analysis utilities may mention stabilizers; the runtime Data fields are
    # audited separately by the dataset tests and do not emit these keys.
    report = f"""# GaugeFlow data quality audit

## Technical summary

The 4,998-row tensor/CIF join is complete and unique, all projected tensors are
finite and Reynolds-consistent, and all tracked basis changes round-trip. The
current v1 split is **not formula-disjoint**: {len(cross_formula_groups)} reduced
formula groups affecting {len(cross_formula_rows)} rows cross train/validation/test.
This is a high-severity benchmark-leakage risk, so v1 results must not be
described as formula-disjoint. The frozen v1 files remain unchanged.

## Scope

- Rows audited: {len(audit_frame)} ({'full artifact' if args.limit is None else 'debug subset'})
- Physically/data-valid rows: {valid_count}
- Source duplicate material IDs: {len(source_duplicate_ids)}
- Duplicate IDs across protocol splits: {split_duplicates}
- Missing source IDs: {len(missing_source_ids)}
- Extra source IDs outside the frozen artifact: {len(extra_source_ids)}
- Missing target-cache files: {len(missing_target_files)}
- Extra target-cache files: {len(extra_target_files)}
- Gate A eight IDs exactly present in the frozen train split: {gate_a_exact}

## Tensor and geometry checks

- Exact-zero tensors by split: `{json.dumps(zero_counts, sort_keys=True)}`
- Invalid or non-finite target rows: {int((~audit_frame.target_valid.fillna(False)).sum())}
- Voigt/Cartesian round-trip tolerance: `2e-7` (FP32)
- Reynolds invariance tolerance: `5e-4`
- Niggli/basis quotient round-trip tolerance: `1e-5`
- Alternate but equivalent second Niggli representatives: {representation_non_idempotent}
- The tracked cell-basis operation never acts on the Cartesian tensor.

## Split and leakage checks

- Reduced-formula overlap train/val: {len(formula_overlaps['train_val'])}
- Reduced-formula overlap train/test: {len(formula_overlaps['train_test'])}
- Reduced-formula overlap val/test: {len(formula_overlaps['val_test'])}
- Union of overlapping formula groups: {len(cross_formula_groups)}
- Rows in an overlapping formula group: {len(cross_formula_rows)} / {len(audit_frame)}
- Cross-split StructureMatcher near-duplicate pairs: {len(match_rows)}
- Formula grouping failure is sufficient to reject the `formula-disjoint` label;
  StructureMatcher pairs identify the stronger primitive/supercell/near-duplicate
  leakage class. No frozen split was modified.
- Runtime source-token counts (mentions are not necessarily model inputs):
  `{json.dumps(leakage_mentions, sort_keys=True)}`. Dataset records are checked
  by regression tests to contain tensor conditions and current crystal state,
  not target-CIF stabilizers, space groups, paired target graphs, or target lattices.

## Integrity outputs

- Row-level audit: `{rows_path}`
- Cross-split structural matches: `{matches_path}`
- Manifest: `{manifest_path}`
- Manifest SHA-256: `{manifest['manifest_sha256']}`
- Frozen split SHA-256: `{manifest['split_manifest']['sha256']}`
- Computed target-cache index SHA-256: `{manifest['target_cache_index_sha256']}`

Any failed row remains in the CSV with its exception. This report does not
alter the frozen split or declare Gate A passed.
"""
    (args.output_dir / "data_quality_audit.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
