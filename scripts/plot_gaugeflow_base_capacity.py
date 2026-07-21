"""Plot the frozen GaugeFlow-base capacity screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "small_34m": "34M",
    "base_58m": "58M",
    "large_98m": "98M",
}
COLORS = {
    "small_34m": "#2574A9",
    "base_58m": "#E67E22",
    "large_98m": "#7D3C98",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.report_root / "result.json").read_text(encoding="utf-8"))
    rows = {row["candidate"]: row for row in result["candidates"]}
    order = ["small_34m", "base_58m", "large_98m"]
    x = np.arange(len(order))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=True)

    loss_axis = axes[0, 0]
    for name in order:
        records = _read_jsonl(args.report_root / f"{name}_training_metrics.jsonl")
        loss_axis.plot(
            [record["graphs_seen_this_invocation"] / 1000.0 for record in records],
            [record["coordinate_loss"] for record in records],
            color=COLORS[name],
            linewidth=1.35,
            alpha=0.9,
            label=LABELS[name],
        )
    loss_axis.set_title("(a) Equal-exposure coordinate training")
    loss_axis.set_xlabel("presented graphs (thousands)")
    loss_axis.set_ylabel("batch coordinate loss")
    loss_axis.set_ylim(bottom=0.0)
    loss_axis.grid(alpha=0.2)
    loss_axis.legend(frameon=False, ncol=3, loc="upper right")

    ratio_axis = axes[0, 1]
    ratios = [rows[name]["validation_coordinate_ratio"] for name in order]
    ratio_axis.bar(x, ratios, color=[COLORS[name] for name in order], width=0.62)
    ratio_axis.axhline(0.297983, color="black", linestyle="--", linewidth=1, label="frozen ceiling")
    ratio_axis.set_xticks(x, [LABELS[name] for name in order])
    ratio_axis.set_ylim(0.24, 0.31)
    ratio_axis.set_ylabel("final / initial validation loss")
    ratio_axis.set_title("(b) Teacher-forced validation ratio")
    ratio_axis.grid(axis="y", alpha=0.2)
    ratio_axis.legend(frameon=False, loc="upper left")
    for index, value in enumerate(ratios):
        ratio_axis.text(index, value + 0.0012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    quality_axis = axes[1, 0]
    explained = [
        next(
            score["score_explained_fraction"]
            for score in rows[name]["score_calibration"]
            if float(score["time"]) == 0.6
        )
        for name in order
    ]
    quality_axis.bar(x, explained, color=[COLORS[name] for name in order], width=0.62)
    quality_axis.axhline(0.64857, color="black", linestyle="--", linewidth=1, label="frozen floor")
    quality_axis.set_xticks(x, [LABELS[name] for name in order])
    quality_axis.set_ylim(0.62, 0.80)
    quality_axis.set_ylabel("explained fraction at $t=0.6$")
    quality_axis.set_title("(c) Mid-noise vector-field quality")
    quality_axis.grid(axis="y", alpha=0.2)
    quality_axis.legend(frameon=False, loc="upper left")
    for index, value in enumerate(explained):
        quality_axis.text(index, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    tradeoff_axis = axes[1, 1]
    w1 = [
        rows[name]["clean_side_conditional_rollout"]["node_nearest_w1_normalized"]
        for name in order
    ]
    throughput = [rows[name]["training_throughput_graphs_per_second"] for name in order]
    scatter = tradeoff_axis.scatter(
        throughput,
        w1,
        s=[105, 120, 135],
        c=[COLORS[name] for name in order],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    del scatter
    label_offsets = {
        "small_34m": (-28, 8),
        "base_58m": (5, 8),
        "large_98m": (5, 8),
    }
    for name, speed, value in zip(order, throughput, w1, strict=True):
        tradeoff_axis.annotate(
            LABELS[name],
            (speed, value),
            xytext=label_offsets[name],
            textcoords="offset points",
            fontsize=8,
        )
    tradeoff_axis.set_xlabel("training throughput (graphs/s; higher is better)")
    tradeoff_axis.set_ylabel("conditional-rollout NN-W1 (lower is better)")
    tradeoff_axis.set_title("(d) Sampling-quality / compute trade-off")
    tradeoff_axis.set_ylim(0.1285, 0.1520)
    tradeoff_axis.grid(alpha=0.2)
    tradeoff_axis.text(
        0.40,
        0.03,
        "selected: 34M (minimum sufficient)",
        transform=tradeoff_axis.transAxes,
        fontsize=8,
        color=COLORS["small_34m"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
