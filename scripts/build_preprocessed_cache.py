"""Build the versioned TensorOrbit-JARVIS PyG-compatible preprocessing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from gaugeflow.data import (
    PREPROCESSED_CRYSTAL_CACHE_SCHEMA,
    TENSOR_CONVENTION_VERSION,
    PiezoCrystalDataset,
    _target_cache_file,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_state(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", help="Base commit when the worktree metadata is not WSL-readable")
    parser.add_argument("--git-dirty", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    split_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    ordered_ids = [
        str(material_id)
        for split in ("train", "val", "test")
        for material_id in split_payload[split]
    ]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Frozen split manifest contains duplicate material IDs")

    started = time.perf_counter()
    dataset = PiezoCrystalDataset(
        args.csv_dir, target_cache_dir=args.target_cache_dir
    )
    lookup = {str(value): index for index, value in enumerate(dataset.frame.material_id)}
    missing = sorted(set(ordered_ids).difference(lookup))
    if missing:
        raise ValueError(f"Source CSV is missing frozen IDs: {missing[:5]}")
    records: dict[str, dict[str, object]] = {}
    for position, material_id in enumerate(ordered_ids, start=1):
        record = dataset[lookup[material_id]]
        records[material_id] = {
            "atom_types": record.atom_types.cpu(),
            "frac_coords": record.frac_coords.cpu(),
            "lattice": record.lattice[0].cpu(),
            "piezo_irreps": record.piezo_irreps[0].cpu(),
            "niggli_transform": record.niggli_transform[0].cpu(),
            "response_norm": float(dataset._condition_norm_cache[lookup[material_id]]),
            "zero_response": bool(record.zero_response[0]),
            "response_stratum": int(record.response_stratum[0]),
        }
        if position % 500 == 0:
            print({"cached": position, "total": len(ordered_ids)})

    source_paths = [args.csv_dir / f"{name}.csv" for name in ("train", "val", "test")]
    target_index = [
        (
            _target_cache_file(args.target_cache_dir, material_id).name,
            sha256_file(_target_cache_file(args.target_cache_dir, material_id)),
        )
        for material_id in sorted(ordered_ids)
    ]
    code_paths = [
        root / "src" / "gaugeflow" / "data.py",
        root / "src" / "gaugeflow" / "unit_cell.py",
        root / "src" / "gaugeflow" / "tensor.py",
        Path(__file__).resolve(),
    ]
    preprocessing_git = git_state(root)
    if args.git_commit is not None:
        preprocessing_git = {"commit": args.git_commit, "dirty": args.git_dirty}
    manifest = {
        "schema": PREPROCESSED_CRYSTAL_CACHE_SCHEMA,
        "name": "TensorOrbit-JARVIS-v1-preprocessed-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "material_ids_sha256": canonical_hash(ordered_ids),
        "source_files": {str(path): sha256_file(path) for path in source_paths},
        "preprocessing_code": {str(path.relative_to(root)): sha256_file(path) for path in code_paths},
        "preprocessing_git": preprocessing_git,
        "tensor_convention_version": TENSOR_CONVENTION_VERSION,
        "split_manifest": {
            "path": str(args.split_manifest),
            "sha256": sha256_file(args.split_manifest),
        },
        "target_cache_index_sha256": canonical_hash(target_index),
        "niggli_transform_stored_per_record": True,
        "sample_id_stored_as_record_key": True,
        "zero_response_and_stratum_stored": True,
        "build_seconds": time.perf_counter() - started,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    torch.save(
        {"schema": PREPROCESSED_CRYSTAL_CACHE_SCHEMA, "manifest": manifest, "records": records},
        args.output,
    )
    sidecar = dict(manifest)
    sidecar["cache_path"] = str(args.output)
    sidecar["cache_sha256"] = sha256_file(args.output)
    sidecar_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
