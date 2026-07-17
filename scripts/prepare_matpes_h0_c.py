"""Download and freeze the exact H0-C MatPES teacher snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_identity(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(files, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _freeze_one(role: str, spec: dict[str, Any], output_root: Path) -> dict[str, Any]:
    local_dir = output_root / role
    required_paths = [local_dir / str(relative) for relative in spec["required_files"]]
    if not all(path.is_file() for path in required_paths):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=str(spec["revision"]),
            local_dir=local_dir,
            allow_patterns=list(spec["required_files"]),
        )
    files = []
    for relative in spec["required_files"]:
        path = local_dir / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen checkpoint file: {path}")
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    model_json = json.loads((local_dir / "model.json").read_text(encoding="utf-8"))
    model_spec = model_json["kwargs"]["model"]
    architecture = str(model_spec["@class"])
    if architecture != spec["architecture"]:
        raise ValueError(
            f"{role} architecture mismatch: expected {spec['architecture']}, got {architecture}"
        )
    readme = (local_dir / "README.md").read_text(encoding="utf-8")
    if str(spec["dataset"]) not in readme:
        raise ValueError(f"{role} model card does not declare {spec['dataset']}")
    init_args = model_spec["init_args"]
    return {
        "role": role,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "local_dir": role,
        "architecture": architecture,
        "model_module": model_spec["@module"],
        "model_version": model_spec.get("@model_version"),
        "dataset": spec["dataset"],
        "cutoff_angstrom": init_args["cutoff"],
        "element_types": init_args["element_types"],
        "files": files,
        "snapshot_sha256": _snapshot_identity(files),
    }


def prepare(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    teachers = {
        role: _freeze_one(role, spec, output_root)
        for role, spec in config["teachers"].items()
    }
    primary = teachers["primary"]
    disagreement = teachers["disagreement"]
    checks = {
        "architectures_distinct": primary["architecture"] != disagreement["architecture"],
        "snapshots_distinct": primary["snapshot_sha256"] != disagreement["snapshot_sha256"],
        "dataset_matches": all(
            record["dataset"] == config["dataset"]["name"] for record in teachers.values()
        ),
        "cutoffs_explicit": all(float(record["cutoff_angstrom"]) > 0 for record in teachers.values()),
        "element_vocabulary_matches": (
            primary["element_types"] == disagreement["element_types"]
        ),
    }
    manifest = {
        "protocol": config["protocol"],
        "qualified_snapshot_identity": all(checks.values()),
        "checks": checks,
        "teachers": teachers,
        "runtime": {
            "python_packages": {
                name: version(name)
                for name in ("torch", "matgl", "ase", "huggingface-hub")
            },
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config_sha256": file_sha256(config_path),
        "preparer_sha256": file_sha256(Path(__file__)),
    }
    manifest_path = output_root / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.config, args.output_root)
    print(
        json.dumps(
            {
                "qualified_snapshot_identity": manifest["qualified_snapshot_identity"],
                "architectures": {
                    role: record["architecture"]
                    for role, record in manifest["teachers"].items()
                },
            }
        )
    )
    if not manifest["qualified_snapshot_identity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
