"""Deterministic inventory and verification of frozen legacy evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

LEGACY_SOURCE_COMMIT = "57a5f76f740e7e1f44f7a0d6f6466079a195fd57"
EXECUTION_CONTRACT_SHA256 = "3bdac52ba00a14c40e8bb6f9de732d16d8a91eb5f81e1a9a2e9b2334e8dd952b"


@dataclass(frozen=True)
class FrozenFile:
    path: str
    sha256: str
    bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_legacy_evidence(path: str) -> bool:
    item = PurePosixPath(path)
    value = item.as_posix()
    prefixes = (
        "configs/gate_a",
        "configs/gate_p5",
        "reports/gate_a",
        "reports/gate_p5",
        "artifacts/gate_a",
        "scripts/audit_gate_a",
        "scripts/evaluate_gate_a",
        "scripts/run_gate_a",
        "scripts/run_gate_p5",
        "tests/test_gate_a",
        "tests/test_p5",
    )
    if value.startswith(prefixes):
        return True
    return value.startswith(("configs/substrate_v2_decoration", "reports/substrate_v2_decoration"))


def tracked_legacy_paths(root: Path) -> list[str]:
    # The Windows worktree stores an absolute Windows gitdir in its .git file,
    # which WSL git cannot resolve.  Inventory the narrow, immutable evidence
    # prefixes directly instead of making the verifier platform-dependent.
    allowed_suffixes = {".csv", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    paths = []
    for top_level in ("configs", "reports", "artifacts", "scripts", "tests"):
        for file_path in (root / top_level).rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed_suffixes:
                continue
            relative = file_path.relative_to(root).as_posix()
            if _is_legacy_evidence(relative):
                paths.append(relative)
    return sorted(paths)


def build_manifest(root: Path) -> dict[str, Any]:
    files = [
        FrozenFile(path, sha256_file(root / path), (root / path).stat().st_size) for path in tracked_legacy_paths(root)
    ]
    return {
        "schema": 1,
        "name": "GaugeFlow vNext frozen Gate A--P5-C0 evidence",
        "legacy_source_commit": LEGACY_SOURCE_COMMIT,
        "execution_contract": {
            "source_path": "../CODEX_IMPLEMENTATION_SPEC.md",
            "sha256": EXECUTION_CONTRACT_SHA256,
            "tracked_in_this_repository": False,
        },
        "scientific_status": {
            "gate_a_through_a11": "frozen_historical_negative_or_diagnostic_only",
            "p5": "frozen_not_passed",
            "p5_c0": "frozen_not_passed",
            "real_tensor_authorized": False,
            "oracle_authorized": False,
            "relaxation_dft_dfpt_authorized": False,
        },
        "q0_input_availability": {
            "sources_64_by_times_33": True,
            "fixed_lift_couplings": True,
            "legacy_checkpoint": False,
            "checkpoint_search_note": (
                "No P5-C0 or D0.4--D0.8 checkpoint exists under outputs, reports, or artifacts. "
                "Retraining is forbidden by Q0."
            ),
        },
        "file_count": len(files),
        "files": [entry.__dict__ for entry in files],
    }


def verify_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if manifest.get("legacy_source_commit") != LEGACY_SOURCE_COMMIT:
        failures.append("legacy source commit changed")
    if manifest.get("execution_contract", {}).get("sha256") != EXECUTION_CONTRACT_SHA256:
        failures.append("execution contract hash changed")
    current_paths = tracked_legacy_paths(root)
    recorded = {entry["path"]: entry for entry in manifest.get("files", [])}
    if current_paths != sorted(recorded):
        failures.append("tracked legacy evidence path inventory changed")
    for path in current_paths:
        entry = recorded.get(path)
        if entry is None:
            continue
        file_path = root / path
        if sha256_file(file_path) != entry["sha256"]:
            failures.append(f"hash changed: {path}")
        if file_path.stat().st_size != entry["bytes"]:
            failures.append(f"size changed: {path}")
    return failures


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
