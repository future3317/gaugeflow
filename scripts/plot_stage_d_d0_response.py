"""Plot the paired Stage-D D0 mechanism screen for paper inclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an object")
    return value


def _other_macro(metrics: dict[str, Any]) -> float:
    names = ("dielectric_loss", "born_loss", "gamma_loss", "internal_strain_loss")
    return sum(float(metrics[name]) for name in names) / len(names)


def main() -> None:
    args = parse_args()
    baseline = _load(args.baseline)
    probe = _load(args.probe)
    selection = _load(args.selection)
    baseline_metrics = baseline["validation"]
    probe_metrics = probe["validation"]
    absolute = np.asarray(
        [
            [
                float(baseline_metrics["piezoelectric_probe_loss"]),
                float(baseline_metrics["piezoelectric_loss"]),
                _other_macro(baseline_metrics),
            ],
            [
                float(probe_metrics["piezoelectric_probe_loss"]),
                float(probe_metrics["piezoelectric_loss"]),
                _other_macro(probe_metrics),
            ],
        ]
    )
    metrics = selection["metrics"]
    thresholds = selection["thresholds"]
    margins = 100.0 * np.asarray(
        [
            float(metrics["probe_error_relative_improvement"])
            - float(thresholds["probe_error_relative_improvement_minimum"]),
            float(thresholds["full_piezoelectric_loss_relative_degradation_maximum"])
            - float(metrics["full_piezoelectric_loss_relative_degradation"]),
            float(thresholds["other_task_macro_relative_degradation_maximum"])
            - float(metrics["other_task_macro_relative_degradation"]),
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "figure.dpi": 180,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), constrained_layout=True)
    blue = "#386CB0"
    orange = "#E68633"
    ink = "#262626"
    grid = "#D9D9D9"

    x = np.arange(3)
    width = 0.34
    axes[0].bar(
        x - width / 2,
        absolute[0],
        width,
        color="white",
        edgecolor=blue,
        linewidth=1.2,
        hatch="///",
        label="Baseline",
    )
    axes[0].bar(
        x + width / 2,
        absolute[1],
        width,
        color=orange,
        edgecolor=ink,
        linewidth=0.7,
        label="+ response probe",
    )
    axes[0].set_xticks(x, ("Response\nprobe", "Full piezo\ntensor", "Other-task\nmacro"))
    axes[0].set_ylabel("Validation loss (normalized)")
    axes[0].set_title("(a) Hatched baseline vs solid +probe")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(axis="y", color=grid, linewidth=0.6)
    axes[0].set_axisbelow(True)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=2, fontsize=7)

    colors = [blue if value >= 0.0 else orange for value in margins]
    bars = axes[1].barh(x, margins, color=colors, edgecolor=ink, linewidth=0.7)
    axes[1].axvline(0.0, color=ink, linewidth=0.9)
    axes[1].set_yticks(
        x,
        ("Probe improvement", "Full-tensor retention", "Other-task retention"),
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Margin to frozen threshold (percentage points)")
    axes[1].set_title("(b) Positive margin passes")
    axes[1].grid(axis="x", color=grid, linewidth=0.6)
    axes[1].set_axisbelow(True)
    span = max(float(np.abs(margins).max()), 1.0)
    axes[1].set_xlim(-1.25 * span, 1.25 * span)
    for bar, value in zip(bars, margins, strict=True):
        inside = value < 0.0
        axes[1].annotate(
            f"{value:+.2f} pp",
            (
                value / 2.0 if inside else value + 0.15,
                bar.get_y() + bar.get_height() / 2,
            ),
            va="center",
            ha="center" if inside else "left",
            fontsize=7.5,
            color="white" if inside else ink,
            fontweight="bold" if inside else "normal",
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Stage-D response-field auxiliary screen (2,000 paired updates, one seed)",
        fontsize=10.5,
        color=ink,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
