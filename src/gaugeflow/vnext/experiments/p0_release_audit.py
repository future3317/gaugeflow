"""Generate the pre-Q1 P0 release checklist from executable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from gaugeflow.checkpoints import load_safe_checkpoint, save_safe_checkpoint
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.geometry import periodic_closest_image_edges
from gaugeflow.stabilizer import proper_unimodular_candidates
from gaugeflow.vnext.experiments.q0_c0_audit import ROOT, _git_commit
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT, tokens_to_atomic_numbers


def _verify_run_manifest(directory: Path) -> bool:
    manifest = directory / "manifest.sha256"
    if not manifest.is_file():
        return False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def _coordinate_horizontal() -> bool:
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch = type(
        "Batch",
        (),
        {"batch": torch.tensor([0, 0, 0, 1, 1]), "num_graphs": 2},
    )()
    value = matcher._coordinate_velocity(torch.randn(5, 3), batch)
    return all(
        torch.allclose(value[batch.batch == graph].mean(dim=0), torch.zeros(3), atol=1.0e-7) for graph in range(2)
    )


def _cvp_counterexample() -> bool:
    lattice = torch.tensor(
        [[[1.0, 0.0, 0.0], [14.554425, 0.061259, 0.0], [5.614603, -0.193389, 0.164521]]],
        dtype=torch.float64,
    )
    delta = torch.tensor([0.825511, 0.213272, 0.458993], dtype=torch.float64)
    frac = torch.stack((torch.zeros(3, dtype=torch.float64), delta))
    edges = periodic_closest_image_edges(frac, lattice, torch.zeros(2, dtype=torch.long))
    index = int(torch.nonzero((edges.source == 0) & (edges.target == 1))[0])
    return bool(edges.distance[index].square() < 0.0267454)


def _dense_decode() -> bool:
    atomic_numbers = tokens_to_atomic_numbers(torch.arange(CHEMICAL_ELEMENT_COUNT))
    return torch.equal(atomic_numbers, torch.arange(1, 119))


def _safe_checkpoint() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.pt"
        model = torch.nn.Linear(2, 2)
        save_safe_checkpoint(
            path,
            model_state=model.state_dict(),
            isotypic_scales=torch.ones(3),
            training_step=1,
            metadata={"config": {"hidden_dim": 2}},
        )
        payload, metadata = load_safe_checkpoint(path, map_location="cpu")
        return payload["training_step"] == 1 and metadata["config"] == {"hidden_dim": 2}


def _run(command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = completed.stdout + completed.stderr
    return completed.returncode == 0, output


def run(output: Path) -> Path:
    if output.exists():
        raise FileExistsError("P0 release audit output already exists")
    output.mkdir(parents=True)
    python = sys.executable
    diagnostic_tests, diagnostic_output = _run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_vnext_diagnostics.py",
            "tests/test_vnext_p0_release.py",
            "tests/test_coordinate_geometry_v1.py",
        ]
    )
    full_tests, full_output = _run([python, "-m", "pytest", "-q"])
    ruff, ruff_output = _run([python, "-m", "ruff", "check", "."])
    mypy, mypy_output = _run([python, "-m", "mypy", "src/gaugeflow"])
    direct_checks: dict[str, Callable[[], bool]] = {
        "translation_horizontal_coordinate_velocity": _coordinate_horizontal,
        "exact_adaptive_triclinic_cvp": _cvp_counterexample,
        "all_792_actions_contract": lambda: proper_unimodular_candidates().shape == (792, 3, 3),
        "dense_element_token_decode_contract": _dense_decode,
        "safe_checkpoint_loading_contract": _safe_checkpoint,
        "published_run_manifest_hashes": lambda: _verify_run_manifest(
            ROOT / "runs" / "Q0" / "20260715T182701Z_7af3ca57bff6"
        ),
    }
    checks = {
        "q0_corrected_metric_regressions": diagnostic_tests,
        "explicit_time_every_conditioning_mode": diagnostic_tests,
        "explicit_time_every_message_block": diagnostic_tests,
        **{name: function() for name, function in direct_checks.items()},
    }
    checks["full_test_suite"] = full_tests
    checks["ruff"] = ruff
    checks["mypy"] = mypy
    all_passed = all(checks.values())
    (output / "diagnostic_test_output.txt").write_text(diagnostic_output, encoding="utf-8")
    (output / "full_test_output.txt").write_text(full_output, encoding="utf-8")
    (output / "ruff_output.txt").write_text(ruff_output, encoding="utf-8")
    (output / "mypy_output.txt").write_text(mypy_output, encoding="utf-8")
    status = {
        "schema": 1,
        "name": "vnext_p0_release_v1",
        "all_passed": all_passed,
        "checks": checks,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "q1_execution_allowed_after_q0_1": all_passed,
        "real_tensor_or_physical_validation_allowed": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(f"| {name} | {passed} |" for name, passed in checks.items())
    report = (
        "# vNext P0 release audit\n\n"
        f"All checks passed: **{all_passed}**. This only permits Q0.1 to consider Q1v2 authorization. "
        "It does not authorize real tensor, an external tensor oracle, relaxation, DFT, or DFPT.\n\n"
        "| Check | Passed |\n|---|---:|\n" + rows + "\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/vnext_p0_release_v1"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(run(output))


if __name__ == "__main__":
    main()
