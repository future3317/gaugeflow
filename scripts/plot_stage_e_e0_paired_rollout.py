"""Plot the paired Stage-C versus Stage-E direct-condition rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    base = result["metrics"]["base"]
    conditioned = result["metrics"]["conditioned"]
    labels = ("Stage-C base", "E0 conditioned")
    colors = ("#7f8c8d", "#2f6f9f")

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(
        x,
        [base["mean_tensor_orbit_error"], conditioned["mean_tensor_orbit_error"]],
        color=colors,
    )
    axes[0].set_ylabel("Normalized orbit RMSE")
    axes[0].set_title("A  Independent Stage-D target match")

    width = 0.34
    axes[1].bar(
        x - width / 2,
        [base["normalized_nearest_neighbor_wasserstein"], conditioned["normalized_nearest_neighbor_wasserstein"]],
        width,
        label="NN-W1",
        color="#e45756",
    )
    axes[1].bar(
        x + width / 2,
        [base["normalized_volume_wasserstein"], conditioned["normalized_volume_wasserstein"]],
        width,
        label="volume-W1",
        color="#54a24b",
    )
    axes[1].set_ylabel("Normalized Wasserstein")
    axes[1].set_title("B  Generated geometry")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(
        x,
        [base["minimum_distance_fraction_at_0_5_angstrom"], conditioned["minimum_distance_fraction_at_0_5_angstrom"]],
        color=colors,
    )
    axes[2].axhline(0.99, color="#d62728", linestyle="--", linewidth=1.2, label="frozen floor")
    axes[2].set_ylim(0.975, 1.002)
    axes[2].set_ylabel("Fraction with distance >= 0.5 Å")
    axes[2].set_title("C  Collision guardrail")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=13, ha="right")
        axis.grid(axis="y", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
