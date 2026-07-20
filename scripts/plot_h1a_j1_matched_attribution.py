"""Plot matched-clock attribution and zero-step gradient geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REGIMES = (
    "clean_clean",
    "noisy_element",
    "noisy_lattice",
    "diagonal",
    "interior",
)
LABELS = ("clean", "element", "lattice", "diagonal", "interior")
COLORS = {"C0": "#4477AA", "C1": "#EE6677", "C2": "#228833"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-result", type=Path, required=True)
    parser.add_argument("--gradient-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matched = json.loads(args.matched_result.read_text(encoding="utf-8"))
    gradient = json.loads(args.gradient_result.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)

    x = np.arange(len(REGIMES))
    width = 0.25
    for offset, arm in zip((-1, 0, 1), ("C0", "C1", "C2"), strict=True):
        values = [matched["arms"][arm]["corners"][name]["final_coordinate_mse"] for name in REGIMES]
        axes[0, 0].bar(x + offset * width, values, width, label=arm, color=COLORS[arm])
    axes[0, 0].set(
        title="A  Parameter-matched final error",
        ylabel="coordinate MSE",
        xticks=x,
        xticklabels=LABELS,
    )
    axes[0, 0].legend(frameon=False, ncols=3)

    paired = matched["paired_final_mse_differences"]["C2_minus_C0"]
    means = np.array([paired[name]["mean"] for name in REGIMES])
    lower = means - np.array([paired[name]["q025"] for name in REGIMES])
    upper = np.array([paired[name]["q975"] for name in REGIMES]) - means
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].errorbar(
        x,
        means,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        color=COLORS["C2"],
        capsize=3,
    )
    axes[0, 1].set(
        title="B  C2 minus C0 (paired)",
        ylabel="final MSE difference",
        xticks=x,
        xticklabels=LABELS,
    )

    matrix = np.eye(len(REGIMES))
    for left_index, left in enumerate(REGIMES):
        for right_index in range(left_index + 1, len(REGIMES)):
            right = REGIMES[right_index]
            key = f"{left}__{right}"
            value = gradient["gradient_cosines"][key]["median"]
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    image = axes[1, 0].imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1, 0].set(
        title="C  Median pre-clip gradient cosine",
        xticks=x,
        yticks=x,
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    rows = gradient["rows"]
    norm_mean = []
    alpha_mean = []
    for regime in REGIMES:
        selected = [row for row in rows if row["regime"] == regime]
        norm_mean.append(np.mean([row["gradient_norm"] for row in selected]))
        alpha_mean.append(np.mean([row["clip_scale_alpha"] for row in selected]))
    axes[1, 1].bar(x, norm_mean, color="#AA4499", alpha=0.8, label="gradient norm")
    axes[1, 1].set(
        title="D  Norm scale, not persistent conflict",
        ylabel="pre-clip gradient norm",
        xticks=x,
        xticklabels=LABELS,
    )
    alpha_axis = axes[1, 1].twinx()
    alpha_axis.plot(x, alpha_mean, "o-", color="#66CCEE", label="clip scale")
    alpha_axis.axhline(0.2, color="#66CCEE", linestyle="--", linewidth=1.0)
    alpha_axis.set_ylabel(r"mean $\alpha=\min(1,c/\|g\|)$")
    handles = axes[1, 1].get_legend_handles_labels()[0] + alpha_axis.get_legend_handles_labels()[0]
    labels = axes[1, 1].get_legend_handles_labels()[1] + alpha_axis.get_legend_handles_labels()[1]
    axes[1, 1].legend(handles, labels, frameon=False, loc="upper right")

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)
        axis.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
