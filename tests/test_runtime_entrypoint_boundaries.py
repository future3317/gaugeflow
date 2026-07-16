from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_legacy_training_entrypoint_fails_closed_without_acknowledgement() -> None:
    result = _run_script(
        "train.py",
        ["--train-csv", "unused.csv", "--checkpoint", "unused.pt"],
    )
    assert result.returncode != 0
    assert "archived legacy-prototype entry point" in result.stdout
    assert "cannot start revised-paper S1a training" in result.stdout


def test_legacy_sampling_entrypoint_fails_closed_without_acknowledgement() -> None:
    result = _run_script(
        "sample.py",
        [
            "--checkpoint",
            "unused.pt",
            "--target",
            "unused.json",
            "--num-samples",
            "1",
            "--num-atoms",
            "1",
            "--output",
            "unused.pt",
        ],
    )
    assert result.returncode != 0
    assert "archived legacy-prototype entry point" in result.stdout
    assert "cannot serve as the revised-paper reverse sampler" in result.stdout
