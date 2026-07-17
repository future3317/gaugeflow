"""Rebuild PhononDB force constants with a physical Hessian projection.

The v1 compact cache enforced only the row acoustic sum rule.  A force-
constant Hessian must additionally obey permutation symmetry.  Hermitianizing
the resulting dynamical matrix after the fact does not preserve the one-sided
sum rule.  This version reconstructs the full supercell Hessian from the
source displacement/force YAML, projects it onto permutation symmetry and both
translational sum rules, and only then stores a compact representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import phonopy
import pyarrow as pa
import pyarrow.parquet as pq
from phonopy.harmonic.force_constants import (
    full_fc_to_compact_fc,
    symmetrize_force_constants,
)

PROTOCOL = "phonondb_force_constants_v2_full_hessian_projection"
EXPECTED_MATERIALS = 10_034
FORMAT_VERSION = 4
SYMMETRIZATION_LEVEL = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _primitive_matrix(policy: str) -> str | None:
    if policy == "yaml":
        return None
    if policy == "P":
        return "P"
    raise ValueError(f"unknown primitive matrix policy: {policy}")


def _project_full_hessian(force_constants: np.ndarray) -> np.ndarray:
    """Project a full FC Hessian onto permutation symmetry and both ASRs."""
    projected = np.ascontiguousarray(force_constants, dtype=np.float64).copy()
    if projected.ndim != 4 or projected.shape[0] != projected.shape[1]:
        raise ValueError("full force constants must have shape [n,n,3,3]")
    symmetrize_force_constants(projected, level=SYMMETRIZATION_LEVEL, lang="C")
    return projected


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _build_one(task: dict[str, Any]) -> dict[str, Any]:
    material_id = str(task["materials_project_id"])
    try:
        core_path = Path(task["core_path"])
        v1_cache_path = Path(task["v1_cache_path"])
        output_path = Path(task["output_path"])
        phonon = phonopy.load(
            core_path,
            primitive_matrix=_primitive_matrix(str(task["primitive_matrix_policy"])),
            produce_fc=True,
            is_compact_fc=False,
            fc_calculator="traditional",
            symmetrize_fc=False,
            log_level=0,
        )
        raw_full = np.ascontiguousarray(phonon.force_constants, dtype=np.float64)
        raw_compact = full_fc_to_compact_fc(phonon.primitive, raw_full)
        with np.load(v1_cache_path, allow_pickle=False) as cached:
            v1_compact = np.asarray(cached["force_constants"], dtype=np.float64)
        source_reproduction_error = float(np.max(np.abs(raw_compact - v1_compact)))

        projected_full = _project_full_hessian(raw_full)
        projected_compact = np.ascontiguousarray(
            full_fc_to_compact_fc(phonon.primitive, projected_full), dtype=np.float64
        )
        row_asr = float(np.max(np.abs(projected_full.sum(axis=1))))
        column_asr = float(np.max(np.abs(projected_full.sum(axis=0))))
        permutation = float(
            np.max(np.abs(projected_full - projected_full.transpose(1, 0, 3, 2)))
        )
        delta = projected_full - raw_full
        relative_delta = float(np.linalg.norm(delta) / max(np.linalg.norm(raw_full), 1e-30))
        maximum_delta = float(np.max(np.abs(delta)))
        if not np.isfinite(projected_compact).all():
            raise ValueError("projected compact force constants contain non-finite values")

        _write_npz_atomic(
            output_path,
            force_constants=projected_compact,
            format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
            protocol=np.asarray(PROTOCOL),
            source_id=np.asarray(task["source_id"]),
            materials_project_id=np.asarray(material_id),
            nims_dataset_id=np.asarray(task["nims_dataset_id"]),
            core_sha256=np.asarray(task["core_sha256"]),
            v1_cache_sha256=np.asarray(task["v1_cache_sha256"]),
            phonopy_version=np.asarray(phonopy.__version__),
            fc_calculator=np.asarray("traditional"),
            projection=np.asarray("full_hessian_permutation_and_bilateral_asr_level_3_C"),
            primitive_matrix_policy=np.asarray(task["primitive_matrix_policy"]),
            force_constants_unit=np.asarray("eV/angstrom^2"),
        )
        return {
            "materials_project_id": material_id,
            "source_id": str(task["source_id"]),
            "nims_dataset_id": str(task["nims_dataset_id"]),
            "core_relpath": str(task["core_relpath"]),
            "core_sha256": str(task["core_sha256"]),
            "v1_cache_relpath": str(task["v1_cache_relpath"]),
            "v1_cache_sha256": str(task["v1_cache_sha256"]),
            "cache_relpath": str(task["output_relpath"]),
            "cache_sha256": sha256_file(output_path),
            "cache_bytes": output_path.stat().st_size,
            "primitive_matrix_policy": str(task["primitive_matrix_policy"]),
            "n_primitive_atoms": len(phonon.primitive),
            "n_supercell_atoms": len(phonon.supercell),
            "fc_shape": "x".join(map(str, projected_compact.shape)),
            "source_reproduction_max_abs": source_reproduction_error,
            "raw_permutation_max_abs": float(
                np.max(np.abs(raw_full - raw_full.transpose(1, 0, 3, 2)))
            ),
            "projection_relative_l2": relative_delta,
            "projection_max_abs": maximum_delta,
            "projected_row_asr_max_abs": row_asr,
            "projected_column_asr_max_abs": column_asr,
            "projected_permutation_max_abs": permutation,
            "error": None,
        }
    except Exception as error:
        return {
            "materials_project_id": material_id,
            "error": f"{type(error).__name__}: {error}",
        }


def build(data_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    v1_root = data_root / "processed" / "phonondb_force_constants_v1"
    mode_root = data_root / "processed" / "phonondb_mode_v1"
    rows = pq.read_table(v1_root / "index.parquet").to_pylist()
    if len(rows) != EXPECTED_MATERIALS:
        raise ValueError(f"expected {EXPECTED_MATERIALS} v1 rows, found {len(rows)}")
    tasks = []
    for row in rows:
        output_relpath = Path("force_constants") / Path(str(row["cache_relpath"])).name
        tasks.append(
            {
                **row,
                "core_path": str(mode_root / "phonopy_params" / Path(str(row["core_relpath"])).name),
                "v1_cache_path": str(v1_root / str(row["cache_relpath"])),
                "v1_cache_relpath": str(row["cache_relpath"]),
                "v1_cache_sha256": str(row["cache_sha256"]),
                "output_path": str(output_root / output_relpath),
                "output_relpath": output_relpath.as_posix(),
            }
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_build_one, tasks, chunksize=4))
    errors = [row for row in records if row.get("error")]
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.parquet"
    pq.write_table(pa.Table.from_pylist(records), index_path, compression="zstd", version="2.6")
    successful = [row for row in records if not row.get("error")]
    manifest = {
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phonopy_version": phonopy.__version__,
        "fc_calculator": "traditional",
        "projection": {
            "space": "full supercell Hessian",
            "constraints": ["permutation symmetry", "row ASR", "column ASR"],
            "algorithm": "phonopy symmetrize_force_constants level=3 C backend",
            "compact_after_projection": True,
            "v1_preserved": True,
        },
        "counts": {
            "expected": EXPECTED_MATERIALS,
            "successful": len(successful),
            "failed": len(errors),
        },
        "observed": {
            "max_source_reproduction_error": max(
                (float(row["source_reproduction_max_abs"]) for row in successful), default=0.0
            ),
            "max_projection_relative_l2": max(
                (float(row["projection_relative_l2"]) for row in successful), default=0.0
            ),
            "max_projected_row_asr": max(
                (float(row["projected_row_asr_max_abs"]) for row in successful), default=0.0
            ),
            "max_projected_column_asr": max(
                (float(row["projected_column_asr_max_abs"]) for row in successful), default=0.0
            ),
            "max_projected_permutation_residual": max(
                (float(row["projected_permutation_max_abs"]) for row in successful), default=0.0
            ),
        },
        "index_path": index_path.name,
        "index_sha256": sha256_file(index_path),
        "v1_index_sha256": sha256_file(v1_root / "index.parquet"),
        "builder_sha256": sha256_file(Path(__file__)),
        "errors": errors[:20],
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = build(args.data_root, args.output_root, workers=args.workers)
    print(json.dumps({"counts": result["counts"], "observed": result["observed"]}))
    if result["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
