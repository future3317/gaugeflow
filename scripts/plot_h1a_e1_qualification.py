"""Render the frozen E1 element-path comparison for reports and the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


METHODS = {
    "absorbing": ("Absorbing mask", "#3B6FB6", "o"),
    "uniform": ("Uniform + pooled counts", "#D9822B", "s"),
    "graph": ("Graph composition", "#6B8E23", "^"),
    "histogram": ("Histogram residual", "#B5547C", "D"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_paths = {
        "absorbing": args.repo_root / "reports/h1a_e1_element_reverse_v1/result.json",
        "uniform": args.repo_root
        / "reports/h1a_e1_uniform_count_projection_v1/result.json",
        "graph": args.repo_root / "reports/h1a_e1_graph_composition_field_v1/result.json",
        "histogram": args.repo_root
        / "reports/h1a_e1_exchangeable_histogram_residual_v1/result.json",
    }
    results = {name: _load(path) for name, path in result_paths.items()}
    run_names = {
        "uniform": "h1a_e1_uniform_count_projection_v1",
        "graph": "h1a_e1_graph_composition_field_v1",
        "histogram": "h1a_e1_exchangeable_histogram_residual_v1",
    }
    training = {
        name: _training_rows(
            args.run_root / run / "seed_5705" / "training_metrics.jsonl"
        )
        for name, run in run_names.items()
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#5B6470",
            "xtick.color": "#3E4650",
            "ytick.color": "#3E4650",
            "text.color": "#222831",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    figure.patch.set_facecolor("white")

    axis = axes[0, 0]
    for name, result in results.items():
        label, color, marker = METHODS[name]
        rows = result["teacher_forced"]["final"]["by_time"]
        axis.plot(
            [row["element_time"] for row in rows],
            [row["top1_accuracy"] for row in rows],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
        )
    axis.scatter([0.5, 0.9], [0.5, 0.25], marker="*", s=65, color="#333333", label="Gate minimum")
    axis.set(title="Teacher-forced element retrieval", xlabel="Element noise time", ylabel="Top-1 accuracy")
    axis.set_ylim(0.0, 0.55)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.legend(frameon=False, ncol=2, loc="lower left")

    axis = axes[0, 1]
    histogram_rows = results["histogram"]["teacher_forced"]["final"]["by_time"]
    graph_rows = results["graph"]["teacher_forced"]["final"]["by_time"]
    times = [row["element_time"] for row in histogram_rows]
    axis.plot(
        times,
        [row["input_count_overlap_fraction"] for row in histogram_rows],
        label="Current noisy histogram",
        color="#6B7280",
        marker="o",
        linestyle="--",
        linewidth=1.6,
    )
    for name, rows in (("graph", graph_rows), ("histogram", histogram_rows)):
        label, color, marker = METHODS[name]
        axis.plot(
            times,
            [row["composition_count_overlap_fraction"] for row in rows],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
        )
    axis.set(title="Composition information across noise", xlabel="Element noise time", ylabel="Count-overlap fraction")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.legend(frameon=False, loc="upper right")

    axis = axes[1, 0]
    compared = ("uniform", "graph", "histogram")
    metrics = (
        ("site_accuracy", "Free site"),
        ("composition_count_overlap_fraction", "Composition overlap"),
        ("oracle_count_site_accuracy", "Oracle-count site"),
    )
    x_values = list(range(len(compared)))
    width = 0.23
    metric_colors = ("#3B6FB6", "#D9822B", "#6B8E23")
    for offset, ((key, label), color) in enumerate(zip(metrics, metric_colors, strict=True)):
        positions = [value + (offset - 1) * width for value in x_values]
        values = [results[name]["reverse"][key] for name in compared]
        bars = axis.bar(positions, values, width=width, label=label, color=color, edgecolor="#303842")
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=7, rotation=90)
    axis.axhline(0.15, color="#333333", linestyle="--", linewidth=1.0, label="Composition Gate")
    axis.set_xticks(x_values, [METHODS[name][0].replace(" + pooled counts", "") for name in compared])
    axis.set(title="Free-reverse and oracle-count attribution", ylabel="Fraction")
    axis.set_ylim(0.0, 0.82)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.legend(frameon=False, ncol=2, loc="upper left")

    axis = axes[1, 1]
    for name, rows in training.items():
        label, color, marker = METHODS[name]
        axis.plot(
            [row["step"] for row in rows],
            [row["composition_loss"] for row in rows],
            label=label,
            color=color,
            marker=marker,
            markevery=max(len(rows) // 6, 1),
            linewidth=1.7,
            markersize=3.5,
        )
    axis.set(title="Graph composition optimization", xlabel="Training step", ylabel="Composition cross entropy")
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.legend(frameon=False)

    figure.suptitle(
        "H1a E1 element-only qualification (seed 5705, 2,111 updates)",
        fontsize=12,
        fontweight="semibold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
