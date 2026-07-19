"""Plot the frozen H1a middle-noise reciprocal-attribution audit.

The figure is descriptive only: it reads the archived CSV artifacts and does
not recompute metrics, select thresholds, or alter the frozen decision.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reports" / "h1a_midnoise_reciprocal_attribution_v1"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_REPORT / "h1a_reciprocal_attribution",
        help="Output path without extension; both PDF and PNG are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    retrieval = _read_csv(args.report_dir / "middle_noise_retrieval.csv")
    spectrum = _read_csv(args.report_dir / "reciprocal_spectrum.csv")
    probe = _read_csv(args.report_dir / "frozen_low_k_probe.csv")

    times = _float(retrieval, "time")
    accuracy = _float(retrieval, "top1_accuracy")
    chance = _float(retrieval, "chance_accuracy")

    spectrum_by_time: dict[float, list[dict[str, Any]]] = {}
    for row in spectrum:
        spectrum_by_time.setdefault(float(row["time"]), []).append(row)
    low_high_ratio = []
    for time in times:
        rows = spectrum_by_time[float(time)]
        low_values = [
            float(row["normalized_residual_ratio"])
            for row in rows
            if float(row["q_upper"]) <= 1.5
        ]
        high_values = [
            float(row["normalized_residual_ratio"])
            for row in rows
            if float(row["q_lower"]) >= 2.5
        ]
        low_high_ratio.append(float(np.mean(low_values) / np.mean(high_values)))
    low_high_ratio_array = np.asarray(low_high_ratio, dtype=np.float64)

    probe_by_band = {
        band: [row for row in probe if row["band"] == band]
        for band in ("low", "high_control")
    }
    low_rows = probe_by_band["low"]
    high_rows = probe_by_band["high_control"]
    low_improvement = 100.0 * _float(low_rows, "relative_improvement")
    high_improvement = 100.0 * _float(high_rows, "relative_improvement")
    low_ci = np.vstack(
        (
            low_improvement - 100.0 * _float(low_rows, "bootstrap_95_low"),
            100.0 * _float(low_rows, "bootstrap_95_high") - low_improvement,
        )
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#3974b8"
    orange = "#d8892b"
    gray = "#6e7781"
    dark = "#20252b"
    light_grid = "#d9dde2"

    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), constrained_layout=True)

    ax = axes[0]
    ax.plot(times, accuracy, marker="o", color=blue, label="Top-1 retrieval")
    ax.plot(times, chance, color=gray, linestyle=":", label="Chance")
    ax.axhline(0.75, color=orange, linestyle="--", label="Frozen bound")
    ax.set_ylim(0.15, 0.82)
    ax.set_xlabel("Noise time $t$")
    ax.set_ylabel("Endpoint retrieval accuracy")
    ax.set_title("(a) Middle-noise identifiability")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(times, low_high_ratio_array, marker="s", color=blue)
    ax.axhline(1.0, color=gray, linestyle=":", label="Equal residual power")
    ax.axhline(1.15, color=orange, linestyle="--", label="Frozen bound")
    ax.set_ylim(0.84, 1.19)
    ax.set_xlabel("Noise time $t$")
    ax.set_ylabel("Low/high normalized residual")
    ax.set_title("(b) Reciprocal-shell attribution")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[2]
    ax.errorbar(
        times,
        low_improvement,
        yerr=low_ci,
        marker="o",
        color=blue,
        capsize=2.2,
        linewidth=1.2,
        label=r"Low $k$ ($95\%$ bootstrap CI)",
    )
    ax.plot(
        times,
        high_improvement,
        marker="s",
        color=gray,
        linestyle="--",
        label=r"High-$k$ control",
    )
    ax.axhline(0.0, color=dark, linewidth=0.7)
    ax.axhline(3.0, color=orange, linestyle="--", label="Frozen low-high bound")
    ax.set_ylim(-0.7, 3.35)
    ax.set_xlabel("Noise time $t$")
    ax.set_ylabel("Held-out MSE improvement (%)")
    ax.set_title("(c) Frozen 12-channel probes")
    ax.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_xticks(times)
        ax.set_xticklabels([f"{time:.3g}" for time in times], rotation=25)
        ax.grid(axis="y", color=light_grid, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
