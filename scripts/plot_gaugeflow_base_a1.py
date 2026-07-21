"""Plot the frozen GaugeFlow-base A1 learning and free-generation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _smooth(values: list[float], width: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < width:
        return array
    left = width // 2
    padded = np.pad(array, (left, width - 1 - left), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [
        json.loads(line)
        for line in arguments.training_metrics.read_text(encoding="utf-8").splitlines()
    ]
    result = json.loads(arguments.evaluation.read_text(encoding="utf-8"))
    checkpoints = result["checkpoints"]
    steps = [int(step) for step in checkpoints]
    ordered = [checkpoints[str(step)] for step in steps]

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)

    loss_axis = axes[0, 0]
    graphs = np.asarray([row["graphs_seen_this_invocation"] for row in records]) / 1000.0
    for key, label, color in (
        ("element_loss", "assignment", "#7D3C98"),
        ("coordinate_loss", "coordinates", "#2574A9"),
        ("shape_loss", "lattice shape", "#E67E22"),
        ("volume_loss", "lattice volume", "#239B56"),
    ):
        loss_axis.plot(
            graphs,
            _smooth([float(row[key]) for row in records]),
            label=label,
            color=color,
            linewidth=1.25,
        )
    loss_axis.set_title("(a) One-pass joint product-space training")
    loss_axis.set_xlabel("presented structures (thousands)")
    loss_axis.set_ylabel("five-record smoothed batch loss")
    loss_axis.set_ylim(bottom=0.0)
    loss_axis.grid(alpha=0.2)
    loss_axis.legend(frameon=False, ncol=2)

    geometry_axis = axes[0, 1]
    width = 0.36
    x = np.arange(len(steps))
    nearest = [row["normalized_nearest_neighbor_wasserstein"] for row in ordered]
    volume = [row["normalized_volume_wasserstein"] for row in ordered]
    geometry_axis.bar(x - width / 2, nearest, width, label="nearest-neighbour", color="#2574A9")
    geometry_axis.bar(x + width / 2, volume, width, label="volume/atom", color="#E67E22")
    geometry_axis.axhline(0.75, color="#2574A9", linestyle="--", linewidth=0.9)
    geometry_axis.axhline(0.50, color="#E67E22", linestyle=":", linewidth=1.0)
    geometry_axis.set_xticks(x, [str(step) for step in steps])
    geometry_axis.set_xlabel("training step")
    geometry_axis.set_ylabel("normalized Wasserstein distance")
    geometry_axis.set_title("(b) Free-generation geometry by checkpoint")
    geometry_axis.grid(axis="y", alpha=0.2)
    geometry_axis.legend(frameon=False)

    closure_axis = axes[1, 0]
    final = ordered[-1]
    closure_labels = [
        "exact\ncomposition",
        "positive\nlattice",
        r"$d_{min}\geq0.5$",
        "formula\nuniqueness",
    ]
    closure_values = [
        final["exact_composition_fraction"],
        final["finite_positive_lattice_fraction"],
        final["minimum_distance_fraction_at_0_5_angstrom"],
        final["formula_uniqueness_fraction"],
    ]
    closure_floors = [1.0, 1.0, 0.98, 0.50]
    closure_x = np.arange(len(closure_values))
    closure_axis.bar(closure_x, closure_values, color=["#7D3C98", "#239B56", "#2574A9", "#A569BD"])
    closure_axis.scatter(closure_x, closure_floors, marker="_", s=260, color="black", label="frozen bound")
    closure_axis.set_xticks(closure_x, closure_labels)
    closure_axis.set_ylim(0.0, 1.06)
    closure_axis.set_ylabel("fraction")
    closure_axis.set_title("(c) Final discrete and validity closure")
    closure_axis.grid(axis="y", alpha=0.2)
    closure_axis.legend(frameon=False, loc="lower left")
    closure_axis.text(
        0.50,
        0.58,
        "JSD: element 0.0475; $N\\sim p_0$ 0.00392\n"
        "$N$ vs disjoint validation: 0.3636 (diagnostic)",
        transform=closure_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "none",
        },
    )

    quantile_axis = axes[1, 1]
    percentiles = np.asarray([0.0, 1.0, 5.0, 50.0, 95.0, 100.0])
    quantile_axis.plot(
        percentiles,
        final["reference_minimum_distance_quantiles_angstrom"],
        marker="o",
        label="Alex-MP validation",
        color="#333333",
    )
    quantile_axis.plot(
        percentiles,
        final["generated_minimum_distance_quantiles_angstrom"],
        marker="s",
        label="GaugeFlow-base",
        color="#2574A9",
    )
    quantile_axis.axhline(0.5, color="#C0392B", linestyle="--", linewidth=0.9)
    quantile_axis.set_xlabel("structure percentile")
    quantile_axis.set_ylabel(r"minimum periodic distance ($\AA$)")
    quantile_axis.set_title("(d) Final local-packing quantiles")
    quantile_axis.grid(alpha=0.2)
    quantile_axis.legend(frameon=False)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=220)
    figure.savefig(arguments.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
