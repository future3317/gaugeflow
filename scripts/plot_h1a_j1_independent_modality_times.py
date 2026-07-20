"""Plot the frozen J1 result and optimization diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in args.training_log.read_text(encoding="utf-8").splitlines()
    ]
    names = ["clean-clean", "noisy element", "noisy lattice", "diagonal", "interior"]
    keys = ["clean_clean", "noisy_element", "noisy_lattice", "diagonal", "interior"]
    initial = np.array([result["corner_results"][key]["initial_coordinate_mse"] for key in keys])
    final = np.array([result["corner_results"][key]["final_coordinate_mse"] for key in keys])
    ratios = np.array([result["corner_results"][key]["validation_ratio"] for key in keys])
    low = np.array([result["corner_results"][key]["bootstrap_ratio"]["q025"] for key in keys])
    high = np.array([result["corner_results"][key]["bootstrap_ratio"]["q975"] for key in keys])

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
    x = np.arange(len(keys))
    colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#AA3377"]

    axes[0, 0].bar(x, ratios, color=colors)
    axes[0, 0].errorbar(x, ratios, yerr=(ratios - low, high - ratios), fmt="none", color="black", capsize=2)
    axes[0, 0].axhline(0.518513457052391, color="#4477AA", linestyle="--", linewidth=1, label="clean limit")
    axes[0, 0].axhline(0.6645334548626645, color="#AA3377", linestyle=":", linewidth=1, label="diagonal limit")
    axes[0, 0].set_ylabel("final / initial coordinate MSE")
    axes[0, 0].set_xticks(x, names, rotation=24, ha="right")
    axes[0, 0].set_title("(a) Frozen corner validation")
    axes[0, 0].legend(frameon=False, fontsize=7)

    width = 0.38
    axes[0, 1].bar(x - width / 2, initial, width, color="#BBBBBB", label="step 0")
    axes[0, 1].bar(x + width / 2, final, width, color=colors, label="step 2111")
    axes[0, 1].set_ylabel("coordinate MSE")
    axes[0, 1].set_xticks(x, names, rotation=24, ha="right")
    axes[0, 1].set_title("(b) Absolute held-out error")
    axes[0, 1].legend(frameon=False, fontsize=7)

    steps = np.array([row["step"] for row in records])
    for label, color in zip(("coordinate", "element", "lattice", "fusion"), colors[:4], strict=True):
        values = np.array([row["modality_time_gradient_norms"][label] for row in records])
        axes[1, 0].plot(steps, values, label=label, color=color, linewidth=1.3)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("post-clip gradient norm")
    axes[1, 0].set_title("(c) All modality clocks learn")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=7)

    clip = np.array([row["clip_fraction"] for row in records])
    speed = np.array([row["graphs_per_second"] for row in records])
    axes[1, 1].plot(steps, clip, color="#AA3377", label="clip fraction")
    axes[1, 1].set_ylim(0.85, 1.005)
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("cumulative clip fraction", color="#AA3377")
    speed_axis = axes[1, 1].twinx()
    speed_axis.spines["right"].set_visible(True)
    speed_axis.plot(steps[1:], speed[1:], color="#4477AA", alpha=0.8, label="throughput")
    speed_axis.set_ylabel("graphs/s", color="#4477AA")
    axes[1, 1].set_title("(d) Optimization and efficiency")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
