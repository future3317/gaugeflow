"""Reproducible release audit for the revised-paper S0 implementation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .equivariant_denoiser import HybridCrystalDenoiser

DESIGN_SHA256 = "9ad4ed018600a62b5f663255a1e0a4d59abcdc26303e523a4f151bdfaf07dd31"
FORBIDDEN_SIGNATURE_FIELDS = {
    "target_cif",
    "target_lattice",
    "target_space_group",
    "target_stabilizer",
    "source_id",
    "endpoint_id",
    "target_metadata",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(repo: Path, arguments: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(
        arguments,
        cwd=repo,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(repo / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode == 0, completed.stdout


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()


def run_audit(repo: Path, design_source: Path, output: Path) -> None:
    if _git(repo, "status", "--short"):
        raise RuntimeError("S0 release audit requires a clean working tree")
    if not design_source.is_file() or _sha256(design_source).lower() != DESIGN_SHA256:
        raise RuntimeError("revised-paper source is missing or its SHA-256 changed")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing S0 report: {output}")
    output.mkdir(parents=True)

    commands = {
        "targeted_tests": [sys.executable, "-m", "pytest", "-q", "tests/test_paper_s0_production.py"],
        "full_tests": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "."],
        "mypy": [sys.executable, "-m", "mypy", "src"],
    }
    checks: dict[str, bool] = {}
    for name, command in commands.items():
        passed, command_output = _run(repo, command)
        checks[name] = passed
        (output / f"{name}.txt").write_text(command_output, encoding="utf-8")

    parameters = set(inspect.signature(HybridCrystalDenoiser.forward).parameters)
    checks["no_target_metadata_signature"] = parameters.isdisjoint(FORBIDDEN_SIGNATURE_FIELDS)
    production_source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((repo / "src/gaugeflow/production").glob("*.py"))
    )
    checks["no_legacy_probability_path_import"] = all(
        forbidden not in production_source
        for forbidden in ("from gaugeflow.flow import", "from gaugeflow.discrete import", "torus_logmap")
    )
    checks["no_fixed_image_cube"] = "product((-2, -1, 0, 1, 2)" not in production_source
    all_passed = all(checks.values())
    commit = _git(repo, "rev-parse", "HEAD")
    timestamp = datetime.now(timezone.utc).isoformat()
    status = {
        "schema": 1,
        "gate": "S0",
        "name": "paper_architecture_mathematical_qualification_v1",
        "status": "passed" if all_passed else "failed",
        "all_passed": all_passed,
        "checks": checks,
        "design_sha256": DESIGN_SHA256,
        "git_commit": commit,
        "timestamp_utc": timestamp,
        "successor_authorization": {"S1": all_passed},
        "real_tensor_or_physical_validation_allowed": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(f"| {name} | {value} |" for name, value in checks.items())
    report = (
        "# Revised-paper S0 qualification\n\n"
        f"Status: **{'passed' if all_passed else 'failed'}**. Design SHA-256: `{DESIGN_SHA256}`.\n\n"
        "This gate qualifies mathematical and software contracts only. It does not establish crystal-generation "
        "quality and does not authorize real tensor, MLIP screening, relaxation, DFT, or DFPT.\n\n"
        "| Check | Passed |\n|---|---:|\n"
        f"{rows}\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest_paths = sorted(path for path in output.iterdir() if path.name != "manifest.sha256")
    manifest = "\n".join(f"{_sha256(path)}  {path.name}" for path in manifest_paths) + "\n"
    (output / "manifest.sha256").write_text(manifest, encoding="utf-8")
    if not all_passed:
        raise RuntimeError("S0 audit failed; inspect the versioned report")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--design-source",
        type=Path,
        default=Path("/mnt/e/Downloads/GaugeFlow_PiezoGen_Revised.tex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_s0_mathematical_qualification_v1"),
    )
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    output = arguments.output if arguments.output.is_absolute() else repo / arguments.output
    run_audit(repo, arguments.design_source, output)


if __name__ == "__main__":
    main()
