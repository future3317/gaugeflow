"""Prepare, but never start, matched TensorOrbit-JARVIS-v2 oracle training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Best-effort provenance for Windows-hosted worktrees called from WSL."""
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # A Windows worktree can carry an ``E:/...`` gitdir pointer that Linux
        # subprocess Git cannot resolve. Preparation remains valid but cannot
        # certify a commit; activation is therefore still blocked.
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("configs/tensororbit_jarvis_v2_oracle_qualification_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/tensororbit_jarvis_v2_oracle_qualification_v1"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/tensororbit_jarvis_v2_oracle_qualification_preparation.md"),
    )
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    protocol: dict[str, Any] = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prepared_not_started":
        raise ValueError("Only the v2 prepared-not-started oracle protocol may be materialized")
    parent = protocol["activation_parent"]
    candidate_split = _resolve(parent["candidate_split"])
    audit_rows = _resolve(parent["source_audit_rows"])
    activation_manifest = _resolve(parent["activation_manifest"])
    if _sha256(candidate_split) != parent["candidate_split_sha256"]:
        raise ValueError("v2 candidate split hash does not match the pre-registered protocol")
    if _sha256(audit_rows) != parent["source_audit_rows_sha256"]:
        raise ValueError("Audit-row hash does not match the pre-registered protocol")
    activation = json.loads(activation_manifest.read_text(encoding="utf-8"))
    if activation.get("status") != "candidate_not_active_audit_complete":
        raise ValueError("v2 activation audit is missing or has an unexpected status")
    split = json.loads(candidate_split.read_text(encoding="utf-8"))
    expected_counts = protocol["data"]["splits"]
    actual_counts = {name: len(split[name]) for name in ("train", "val", "test")}
    if actual_counts != expected_counts:
        raise ValueError(f"v2 split counts drifted: {actual_counts}")
    if len(set(split["train"]) & set(split["val"])) or len(set(split["train"]) & set(split["test"])):
        raise ValueError("v2 split ID overlap detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    shared = {
        "schema": 1,
        "status": "prepared_not_started",
        "protocol_sha256": _sha256(protocol_path),
        "candidate_split": str(candidate_split),
        "candidate_split_sha256": _sha256(candidate_split),
        "split_counts": actual_counts,
        "source_csv_directory": str(_resolve(protocol["data"]["source_csv_directory"])),
        "target_cache_dir": str(_resolve(protocol["data"]["target_cache_dir"])),
        "target_definition": protocol["data"]["tensor_target"],
        "coordinate_convention": protocol["data"]["tensor_coordinate_convention"],
        "forbidden": ["GaugeFlow training", "PiezoJet as primary oracle", "v1 validation/test"],
    }
    oracle_manifests: dict[str, str] = {}
    for oracle in protocol["oracles"]:
        manifest = {
            **shared,
            "oracle": oracle,
            "required_before_training": oracle["required_external_pin"],
            "required_after_training": [
                "checkpoint_sha256",
                "prediction_file_sha256",
                "exact_val_test_id_join",
                "frozen_qualification_report",
            ],
        }
        path = output_dir / f"{oracle['id']}_training_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        oracle_manifests[oracle["id"]] = _sha256(path)
    status = {
        "schema": 1,
        "name": protocol["name"],
        "status": "prepared_commit_required_before_external_training",
        "protocol_sha256": _sha256(protocol_path),
        "activation_audit_manifest_sha256": _sha256(activation_manifest),
        "candidate_split_sha256": _sha256(candidate_split),
        "audit_rows_sha256": _sha256(audit_rows),
        "oracle_training_manifests": oracle_manifests,
        "git_head_at_preparation": _git("rev-parse", "HEAD"),
        "working_tree_clean_at_preparation": (
            None if _git("status", "--porcelain") is None
            else not bool(_git("status", "--porcelain"))
        ),
        "external_training_started": False,
        "gaugeflow_training_started": False,
        "piezojet_primary_oracle": False,
    }
    status_path = output_dir / "manifest.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# TensorOrbit-JARVIS-v2 external oracle qualification preparation

## Status

`{status['status']}`. The v2 candidate split remains inactive for GaugeFlow.
This preparation creates matched **external oracle** input manifests only; it
does not start GMTNet, the SE(3)-Transformer, PiezoJet, GaugeFlow, S2, or a
4,000/499/499 run.

## Frozen data identity

- Candidate split SHA-256: `{status['candidate_split_sha256']}`
- Protocol SHA-256: `{status['protocol_sha256']}`
- Split counts: `{actual_counts}`
- Oracle manifests: `{oracle_manifests}`

## Activation boundary

Before either external training job starts, pin its source repository and
commit, environment lock, entrypoint, this protocol/manifest commit, and the
same v2 split hash. Both GMTNet and the architecture-distinct e3nn
SE(3)-Transformer must complete matched v2 validation before any frozen oracle
ensemble is qualified. PiezoJet is explicitly not the primary oracle.
""",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
