"""Plot the Stage-E common-noise orbit-mimic mechanism screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    args = parse_args()
    arms = (
        ("baseline", "Baseline"),
        ("orbit_mimic", "+ orbit mimic"),
        ("orbit_mimic_retention", "+ soft retention"),
        ("orbit_mimic_exact_null", "Atlas-only exact null"),
        ("orbit_mimic_residual_adapter", "Low-rank exact null"),
        ("orbit_mimic_centered_adapter", "Centered block route"),
    )
    available = [item for item in arms if (args.report_dir / item[0] / "result.json").is_file()]
    results = [_read_json(args.report_dir / key / "result.json") for key, _ in available]
    labels = [label for _, label in available]
    colors = ["#7f8c8d", "#2f6f9f", "#d98c3f", "#a65d7b", "#3f8f65", "#8c6d31"][: len(available)]
    x = np.arange(len(available))

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    fine = [result["validation"]["fine"] for result in results]
    axes[0, 0].bar(x, fine, color=colors)
    axes[0, 0].set_ylabel("Validation fine loss")
    axes[0, 0].set_title("A  Endpoint denoising")

    width = 0.36
    residual = [max(result["validation"]["orbit_mimic"], 1e-12) for result in results]
    information = [max(result["validation"]["posterior_information"], 1e-12) for result in results]
    axes[0, 1].bar(x - width / 2, residual, width, label="typed orbit residual", color="#4c78a8")
    axes[0, 1].bar(x + width / 2, information, width, label="posterior information", color="#f58518")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Value (log scale)")
    axes[0, 1].set_title("B  Orbit consistency and atlas use")
    axes[0, 1].legend(frameon=False, fontsize=8)

    swap = [result["validation"]["target_swap_separation"] for result in results]
    drift = [max(result["validation"]["null_retention"], 1e-12) for result in results]
    axes[1, 0].bar(x - width / 2, swap, width, label="target-swap separation", color="#54a24b")
    axes[1, 0].bar(x + width / 2, drift, width, label="null typed drift", color="#e45756")
    axes[1, 0].set_yscale("symlog", linthresh=1e-5)
    axes[1, 0].set_ylabel("Typed distance")
    axes[1, 0].set_title("C  Conditional signal versus base retention")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for (key, label), color in zip(available, colors, strict=True):
        records = _read_jsonl(args.report_dir / key / "training_metrics.jsonl")
        steps = [record["step"] for record in records]
        values = [record["fine"] for record in records]
        axes[1, 1].plot(steps, values, label=label, color=color, linewidth=1.4, alpha=0.9)
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Minibatch fine loss")
    axes[1, 1].set_title("D  One-seed mechanism trajectories")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        if axis is not axes[1, 1]:
            axis.set_xticks(x, labels, rotation=18, ha="right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
