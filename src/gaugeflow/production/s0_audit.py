"""Reproducible release audit for the revised-paper S0 implementation."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from gaugeflow.file_utils import sha256_file as _sha256

from .equivariant_denoiser import HybridCrystalDenoiser

DESIGN_SHA256 = "9ad4ed018600a62b5f663255a1e0a4d59abcdc26303e523a4f151bdfaf07dd31"
FORBIDDEN_SIGNATURE_FIELDS = {
    # Dataset bookkeeping and audit labels may remain on the PyG batch, but
    # they are never production-model inputs. Naming them here makes that
    # quarantine executable instead of relying on a future trainer to
    # remember which fields are target-derived.
    "material_id",
    "niggli_transform",
    "response_stratum",
    "zero_response",
    "target_cif",
    "target_lattice",
    "target_space_group",
    "target_stabilizer",
    "source_id",
    "endpoint_id",
    "target_metadata",
}


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


def _worktree_head(repo: Path) -> str:
    """Read HEAD even when a Windows-created worktree is audited from WSL."""
    git_pointer = (repo / ".git").read_text(encoding="utf-8").strip()
    if not git_pointer.startswith("gitdir: "):
        raise RuntimeError("S0 audit expects an explicit Git worktree pointer")
    raw = git_pointer.removeprefix("gitdir: ")
    if len(raw) >= 3 and raw[1:3] == ":/":
        git_directory = Path(f"/mnt/{raw[0].lower()}/{raw[3:]}")
    else:
        git_directory = Path(raw)
        if not git_directory.is_absolute():
            git_directory = repo / git_directory
    head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        common_pointer = git_directory / "commondir"
        common_directory = (
            (git_directory / common_pointer.read_text(encoding="utf-8").strip()).resolve()
            if common_pointer.is_file()
            else git_directory
        )
        loose_reference = common_directory / reference
        if loose_reference.is_file():
            head = loose_reference.read_text(encoding="utf-8").strip()
        else:
            packed = common_directory / "packed-refs"
            matches = [
                line.split(" ", 1)[0]
                for line in packed.read_text(encoding="utf-8").splitlines()
                if line.endswith(f" {reference}")
            ] if packed.is_file() else []
            if len(matches) != 1:
                raise RuntimeError("could not resolve symbolic worktree HEAD")
            head = matches[0]
    return head


def run_audit(
    repo: Path,
    design_source: Path,
    output: Path,
    *,
    git_commit: str,
    clean_attestation: str,
) -> None:
    if clean_attestation != "host_git_status_clean":
        raise RuntimeError("S0 audit requires a host-Git clean-worktree attestation")
    if git_commit != _worktree_head(repo):
        raise RuntimeError("supplied host Git commit does not match the worktree HEAD")
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
    runtime_paths = [
        path
        for path in sorted((repo / "src/gaugeflow/production").glob("*.py"))
        if path.name != "s0_audit.py"
    ]
    production_source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_paths)
    checks["no_legacy_probability_path_import"] = all(
        forbidden not in production_source
        for forbidden in ("from gaugeflow.flow import", "from gaugeflow.discrete import", "torus_logmap")
    )
    checks["no_fixed_image_cube"] = "product((-2, -1, 0, 1, 2)" not in production_source
    all_passed = all(checks.values())
    commit = git_commit
    timestamp = datetime.now(timezone.utc).isoformat()
    status = {
        "schema": 1,
        "gate": "S0.1",
        "name": "paper_architecture_mathematical_qualification_v1_1",
        "status": "passed" if all_passed else "failed",
        "all_passed": all_passed,
        "checks": checks,
        "design_sha256": DESIGN_SHA256,
        "git_commit": commit,
        "clean_worktree_attestation": clean_attestation,
        "timestamp_utc": timestamp,
        "successor_authorization": {"S1": all_passed},
        "supersedes_audit_definition_only": "paper_s0_mathematical_qualification_v1",
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
        "# Revised-paper S0.1 qualification\n\n"
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
        default=Path("reports/paper_s0_mathematical_qualification_v1_1"),
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--clean-attestation", required=True)
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    output = arguments.output if arguments.output.is_absolute() else repo / arguments.output
    run_audit(
        repo,
        arguments.design_source,
        output,
        git_commit=arguments.git_commit,
        clean_attestation=arguments.clean_attestation,
    )


if __name__ == "__main__":
    main()
