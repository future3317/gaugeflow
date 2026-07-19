"""Render the frozen clean-topology attribution and learning-curve figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2563A6"
BLUE_LIGHT = "#78A6D0"
ORANGE = "#D97706"
INK = "#20252B"
GREY = "#6B7280"
LIGHT_GREY = "#D7DCE2"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GREY)
    axis.tick_params(colors=GREY, labelsize=8)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_topology(topology_dir: Path, output_stem: Path) -> None:
    state = _rows(topology_dir / "topology_state.csv")
    carrier = _rows(topology_dir / "topology_oracle_carrier.csv")
    probe = _rows(topology_dir / "topology_probe.csv")
    time = _float(state, "time")

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))
    figure.suptitle(
        "All-pair clean-topology attribution",
        color=INK,
        fontsize=13,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        "Frozen dynamic checkpoint; complete directed non-self pair support; zero optimizer steps",
        ha="center",
        va="top",
        color=GREY,
        fontsize=8,
    )

    axis = axes[0]
    axis.plot(time, 1.0 - _float(state, "soft_jaccard"), marker="o", color=BLUE, label="1 − soft Jaccard")
    axis.plot(
        time,
        _float(state, "hard_switch_fraction"),
        marker="s",
        linestyle="--",
        color=ORANGE,
        label="hard switch fraction",
    )
    axis.set(title="A  Clean/noisy topology disagreement", xlabel="noise time t", ylabel="fraction")
    axis.set_ylim(0.0, 0.65)
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    _style_axis(axis)

    axis = axes[1]
    variants = {
        "clean_oracle": ("clean oracle", BLUE, "o", "-"),
        "noisy_current": ("noisy control", GREY, "s", "--"),
        "learned_probe": ("probe plug-in", ORANGE, "^", "-."),
    }
    for variant, (label, color, marker, linestyle) in variants.items():
        selected = [row for row in carrier if row["variant"] == variant]
        axis.plot(
            _float(selected, "time"),
            100.0 * _float(selected, "relative_improvement"),
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    axis.axhline(0.0, color=INK, linewidth=0.8)
    axis.set(title="B  Residual-energy improvement", xlabel="noise time t", ylabel="relative improvement (%)")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    _style_axis(axis)

    axis = axes[2]
    axis.plot(_float(probe, "time"), _float(probe, "auc"), color=BLUE, marker="o", label="probe AUC")
    axis.plot(
        _float(probe, "time"),
        _float(probe, "explained_fraction"),
        color=ORANGE,
        marker="s",
        linestyle="--",
        label="explained fraction",
    )
    axis.set(title="C  Frozen-probe predictability", xlabel="noise time t", ylabel="score")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False, fontsize=7, loc="lower left")
    _style_axis(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86), w_pad=1.8)
    _save(figure, output_stem)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    return np.asarray([np.median(values[max(0, index - radius) : index + radius + 1]) for index in range(values.size)])


def _plot_learning_curve(learning_dir: Path, training_log: Path, output_stem: Path) -> None:
    learning = _rows(learning_dir / "learning_curve.csv")
    score = _rows(learning_dir / "score_endpoint_curve.csv")
    records = [json.loads(line) for line in training_log.read_text(encoding="utf-8").splitlines()]
    passes = _float(learning, "nominal_data_passes")
    ratios = _float(learning, "validation_coordinate_ratio")

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    figure.suptitle(
        "Fixed dynamic-architecture exposure curve",
        color=INK,
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.962,
        "Seed 5705; unchanged 5.03M-parameter coordinate model; 540,164-structure train split",
        ha="center",
        va="top",
        color=GREY,
        fontsize=8,
    )

    axis = axes[0, 0]
    axis.plot(passes, ratios, color=BLUE, marker="o", linewidth=1.8)
    for x, y in zip(passes, ratios, strict=True):
        axis.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK)
    axis.axhline(0.5, color=ORANGE, linestyle="--", linewidth=1.0, label="archived threshold (diagnostic only)")
    axis.set(
        title="A  EMA validation coordinate loss", xlabel="nominal train-data passes", ylabel="final / initial loss"
    )
    axis.set_xticks(passes)
    axis.set_ylim(0.42, 1.07)
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axis(axis)

    axis = axes[0, 1]
    series = {
        0.0: ("0 pass", GREY, ":", "o"),
        0.5: ("0.5 pass", BLUE_LIGHT, "--", "s"),
        1.0: ("1 pass", ORANGE, "-.", "^"),
        2.0: ("2 passes", BLUE, "-", "o"),
    }
    for pass_value, (label, color, linestyle, marker) in series.items():
        selected = [row for row in score if float(row["nominal_data_passes"]) == pass_value]
        axis.plot(
            _float(selected, "time"),
            _float(selected, "endpoint_rms_angstrom"),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(title="B  Teacher-forced endpoint RMS", xlabel="noise time t (log scale)", ylabel="RMS (Å, log scale)")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _style_axis(axis)

    axis = axes[1, 0]
    exposure = np.asarray([record["graphs_seen_this_invocation"] for record in records]) / 540_164.0
    train_loss = np.asarray([record["coordinate_loss"] for record in records])
    smoothing_window = 9
    axis.plot(exposure, train_loss, color=LIGHT_GREY, linewidth=0.8, label="logged batch")
    axis.plot(
        exposure,
        _rolling_median(train_loss, smoothing_window),
        color=BLUE,
        linewidth=1.8,
        label=f"rolling median ({smoothing_window} logs)",
    )
    axis.set(title="C  Coordinate training objective", xlabel="train-data passes", ylabel="batch loss")
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    _style_axis(axis)

    axis = axes[1, 1]
    gradient_styles = {
        "input_state_embeddings": ("input/state", GREY, "-"),
        "base_message_blocks": ("base blocks", BLUE_LIGHT, "--"),
        "dynamic_edge_angular": ("dynamic edge/angular", ORANGE, "-."),
        "coordinate_readout": ("coordinate readout", BLUE, "-"),
    }
    for field, (label, color, linestyle) in gradient_styles.items():
        values = np.asarray([record["clipped_module_gradient_norms"][field] for record in records])
        axis.plot(exposure, values, color=color, linestyle=linestyle, linewidth=1.2, label=label)
    axis.set_yscale("log")
    axis.set(title="D  Post-clip module gradient norms", xlabel="train-data passes", ylabel="L2 norm (log scale)")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    _style_axis(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925), h_pad=2.1, w_pad=1.8)
    _save(figure, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-dir", type=Path, required=True)
    parser.add_argument("--learning-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _plot_topology(args.topology_dir, args.output_dir / "h1a_all_pair_clean_topology_v2")
    _plot_learning_curve(
        args.learning_dir,
        args.training_log,
        args.output_dir / "h1a_fixed_dynamic_learning_curve_v1",
    )


if __name__ == "__main__":
    main()
