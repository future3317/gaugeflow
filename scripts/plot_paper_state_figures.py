#!/usr/bin/env python3
"""Render the current GaugeFlow paper figures from frozen report artifacts.

The script intentionally reads only versioned JSON/CSV evidence.  It does not
mix metrics across protocols and it writes a source/output hash manifest next
to the figures so that the manuscript graphics remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#24303B"
BLUE = "#356FB6"
GOLD = "#C88A16"
GREEN = "#4E8A72"
RED = "#C45E5E"
SLATE = "#74808C"
LIGHT = "#E8EDF2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="GaugeFlow report root containing the frozen JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/paper_current_state_figures_v1"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK, labelsize=8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)
    axis.grid(axis="y", color=LIGHT, linewidth=0.8, zorder=0)


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=240, bbox_inches="tight")
    plt.close(figure)
    return outputs


def plot_base_factorization(output_dir: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(12.0, 2.45))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        ("Carrier + count", r"$p(B,N)$", "sample both; never use\ntest-set truth"),
        ("Composition", r"$p(C\mid N)$", "unordered multiset;\nexact atom count"),
        ("Site assignment", r"$p(A\mid C,B)$", "count-exact; parent-action\nquotient"),
        ("Lattice", r"$p(L\mid A,C,N,B)$", "positive volume;\nperiodic chart"),
        ("Coordinates", r"$p(F\mid A,L,N,B)$", "wrapped translation\nquotient"),
    ]
    colors = [SLATE, GREEN, GOLD, BLUE, RED]
    width = 0.166
    gap = 0.035
    left = 0.015
    y = 0.25
    height = 0.58
    for index, ((title, law, note), color) in enumerate(zip(boxes, colors, strict=True)):
        x = left + index * (width + gap)
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 0.44, title, ha="center", va="center", fontsize=10, color=INK, weight="bold")
        axis.text(x + width / 2, y + 0.29, law, ha="center", va="center", fontsize=11, color=color)
        axis.text(x + width / 2, y + 0.105, note, ha="center", va="center", fontsize=6.8, color=INK, linespacing=1.2)
        if index < len(boxes) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.004, y + height / 2),
                (x + width + gap - 0.004, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=INK,
            )
            axis.add_patch(arrow)
    axis.text(
        0.5,
        0.94,
        r"GaugeFlow-base generative closure: $p(B,N,C,A,L,F)$",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    return _save(figure, output_dir, "base_probability_factorization")


def plot_interface_evidence(
    composition: dict[str, Any],
    q1: dict[str, Any],
    geometry: dict[str, Any],
    q0: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))
    figure.subplots_adjust(wspace=0.32, hspace=0.42)

    # (a) Absolute conditional composition likelihood.
    axis = axes[0, 0]
    likelihood = composition["conditional_species_likelihood"]
    labels = ["Model", "Legal empirical", "Legal uniform"]
    calibration = [
        likelihood["calibration"]["model_nll_per_decision"],
        likelihood["calibration"]["empirical_nll_per_decision"],
        likelihood["calibration"]["uniform_nll_per_decision"],
    ]
    test = [
        likelihood["test"]["model_nll_per_decision"],
        likelihood["test"]["empirical_nll_per_decision"],
        likelihood["test"]["uniform_nll_per_decision"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    bars_cal = axis.bar(
        x - width / 2, calibration, width, color=BLUE, edgecolor=INK, linewidth=0.6, label="Calibration", zorder=2
    )
    bars_test = axis.bar(
        x + width / 2, test, width, color=GOLD, edgecolor=INK, linewidth=0.6, hatch="//", label="Test", zorder=2
    )
    axis.bar_label(bars_cal, fmt="%.3f", fontsize=7, padding=2)
    axis.bar_label(bars_test, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 4.65)
    axis.set_ylabel("NLL per species decision")
    axis.set_title("(a) Qualified unordered composition law", loc="left", fontsize=10, weight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _style_axis(axis)

    # (b) The frozen unary Q1 failed its held-out probability criteria.
    axis = axes[0, 1]
    metrics = ["target_quotient_probability", "sample_retrieval", "sample_orbit_aligned_site_accuracy"]
    short = ["Target-orbit\nprobability", "Categorical\nretrieval", "Orbit-aligned\nsite accuracy"]
    validation = [q1["metrics"]["validation"][key] for key in metrics]
    test_values = [q1["metrics"]["test"][key] for key in metrics]
    thresholds = [0.25, 0.25, 0.8]
    x = np.arange(len(metrics))
    bars_val = axis.bar(
        x - width / 2, validation, width, color=BLUE, edgecolor=INK, linewidth=0.6, label="Validation", zorder=2
    )
    bars_test = axis.bar(
        x + width / 2, test_values, width, color=GOLD, edgecolor=INK, linewidth=0.6, hatch="//", label="Test", zorder=2
    )
    axis.scatter(
        x,
        thresholds,
        marker="D",
        s=36,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.2,
        label="Frozen minimum",
        zorder=4,
    )
    axis.bar_label(bars_val, fmt="%.3f", fontsize=7, padding=2)
    axis.bar_label(bars_test, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(x, short)
    axis.set_ylim(0, 0.92)
    axis.set_ylabel("Fraction / probability")
    axis.set_title("(b) Legacy unary assignment Q1: failed", loc="left", fontsize=10, weight="bold")
    axis.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    _style_axis(axis)

    # (c) Target-free geometry repairs the carrier representation.
    axis = axes[1, 0]
    expressivity = geometry["metrics"]["all"]
    values = [
        expressivity["geometry_unary_resolved_fraction"],
        expressivity["geometry_pair_resolved_collision_fraction"],
        expressivity["geometry_pair_mean_target_ceiling"],
    ]
    labels = [
        "Geometry unary\nresolved",
        "Two-point descriptor\nresolves unary collisions",
        "Two-point mean\ntarget ceiling",
    ]
    bars = axis.bar(np.arange(3), values, color=[SLATE, BLUE, GREEN], edgecolor=INK, linewidth=0.7, zorder=2)
    axis.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], fontsize=8, padding=3)
    axis.set_xticks(np.arange(3), labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Fraction")
    axis.set_title("(c) Geometry-complete carrier audit (454 carriers)", loc="left", fontsize=10, weight="bold")
    _style_axis(axis)

    # (d) Q0 closes the probability software, but does not claim learning.
    axis = axes[1, 1]
    errors = [
        q0["metrics"]["complete_distribution_normalization_error"],
        q0["metrics"]["subset_dp_bruteforce_error"],
        q0["metrics"]["fp32_neural_equivariance_error"],
        q0["metrics"]["residual_stabilizer_error"],
    ]
    labels = ["Normalization", "Subset DP / brute force", "FP32 relabel", "Residual stabilizer"]
    y = np.arange(len(labels))
    bars = axis.barh(y, errors, color=[GREEN, GREEN, BLUE, BLUE], edgecolor=INK, linewidth=0.6, zorder=2)
    axis.set_xscale("log")
    axis.set_xlim(1e-18, 1e-5)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Absolute numerical residual (log scale)")
    axis.set_title("(d) Orderless remaining-count Q0: passed", loc="left", fontsize=10, weight="bold")
    for bar, value in zip(bars, errors, strict=True):
        axis.text(value * 1.25, bar.get_y() + bar.get_height() / 2, f"{value:.1e}", va="center", fontsize=7, color=INK)
    _style_axis(axis)
    axis.grid(axis="x", color=LIGHT, linewidth=0.8, which="major", zorder=0)
    axis.grid(axis="y", visible=False)

    return _save(figure, output_dir, "composition_assignment_evidence")


def plot_coordinate_qualification(coordinate: dict[str, Any], output_dir: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    figure.subplots_adjust(wspace=0.31, hspace=0.42)

    validation = coordinate["validation"]
    steps = np.array(sorted(int(step) for step in validation))
    exposure = steps / steps[-1]
    coordinate_mse = np.array([validation[str(step)]["coordinate"] for step in steps])
    axis = axes[0, 0]
    axis.plot(exposure, coordinate_mse, marker="o", color=BLUE, linewidth=2, zorder=3)
    axis.axhline(
        coordinate["reference_validation_ratio"] * coordinate_mse[0],
        color=SLATE,
        linestyle="--",
        linewidth=1.2,
        label="Historical quarter-pass endpoint",
    )
    axis.scatter([exposure[-1]], [coordinate_mse[-1]], marker="D", color=GREEN, zorder=4)
    axis.set_xlabel("Fraction of one train-data pass")
    axis.set_ylabel("Validation coordinate MSE")
    axis.set_title("(a) One exact Alex-MP train pass", loc="left", fontsize=10, weight="bold")
    axis.legend(frameon=False, fontsize=7)
    _style_axis(axis)

    score = coordinate["score_calibration"]
    times = np.array([row["time"] for row in score])
    endpoint = np.array([row["endpoint_rms_angstrom"] for row in score])
    axis = axes[0, 1]
    axis.plot(times, endpoint, marker="o", color=BLUE, linewidth=2, zorder=3)
    axis.scatter(
        [0.005, 0.1], [0.04, 0.08], marker="D", s=38, facecolor="white", edgecolor=INK, label="Frozen maxima", zorder=4
    )
    axis.set_xlabel("Coordinate diffusion time $t$")
    axis.set_ylabel(r"Teacher-forced endpoint RMS ($\AA$)")
    axis.set_title("(b) Endpoint reconstruction", loc="left", fontsize=10, weight="bold")
    axis.legend(frameon=False, fontsize=7)
    _style_axis(axis)

    explained = np.array([row["score_explained_fraction"] for row in score])
    axis = axes[1, 0]
    axis.plot(times, explained, marker="o", color=GOLD, linewidth=2, zorder=3)
    axis.scatter(
        [0.6], [0.5], marker="D", s=38, facecolor="white", edgecolor=INK, label=r"Frozen $t=.6$ minimum", zorder=4
    )
    axis.set_xlabel("Coordinate diffusion time $t$")
    axis.set_ylabel("Score explained fraction")
    axis.set_ylim(0, 1.0)
    axis.set_title("(c) Conditional score information", loc="left", fontsize=10, weight="bold")
    axis.legend(frameon=False, fontsize=7)
    _style_axis(axis)

    rollout = coordinate["conditional_rollout"]
    starts = [row["start_time"] for row in rollout]
    means = [row["mean_endpoint_rms_angstrom"] for row in rollout]
    limits = [0.5, 1.0]
    ratios = np.asarray(means) / np.asarray(limits)
    axis = axes[1, 1]
    bars = axis.bar(np.arange(len(starts)), ratios, color=[BLUE, GOLD], edgecolor=INK, linewidth=0.7, zorder=2)
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, label="Frozen maximum")
    axis.set_xticks(np.arange(len(starts)), [f"start $t={value:.1f}$" for value in starts])
    axis.set_ylabel("Mean rollout RMS / frozen maximum")
    axis.set_ylim(0, 1.12)
    axis.set_title("(d) Reverse-SDE-100 rollout", loc="left", fontsize=10, weight="bold")
    axis.bar_label(bars, labels=[f"{value:.3f} $\\AA$" for value in means], fontsize=8, padding=3)
    axis.text(
        0.98,
        0.9,
        "0 sampling failures",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=GREEN,
        weight="bold",
    )
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    _style_axis(axis)

    return _save(figure, output_dir, "conditional_coordinate_qualification")


def _roadmap_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str,
    note: str,
    color: str,
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor="white",
        edgecolor=color,
        linewidth=2.0,
    )
    axis.add_patch(patch)
    axis.text(x + 0.015, y + height - 0.04, title, ha="left", va="top", fontsize=10, color=color, weight="bold")
    axis.text(x + 0.015, y + height - 0.115, subtitle, ha="left", va="top", fontsize=7.7, color=INK, weight="bold")
    axis.text(x + 0.015, y + 0.035, note, ha="left", va="bottom", fontsize=6.6, color=INK, linespacing=1.2)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=INK))


def plot_pretraining_roadmap(output_dir: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(12.0, 5.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.02, 0.96, "Evidence-gated GaugeFlow training program", fontsize=14, color=INK, weight="bold", va="top")
    axis.text(
        0.02,
        0.905,
        "Later stages cannot repair or reinterpret a failed prerequisite Gate.",
        fontsize=8.5,
        color=SLATE,
        va="top",
    )

    width, height = 0.205, 0.25
    top_y = 0.57
    bottom_y = 0.16
    xs = [0.025, 0.275, 0.525, 0.775]
    _roadmap_box(
        axis,
        (xs[0], top_y),
        (width, height),
        "A0",
        "Interface closure",
        "learned assignment\n$p(B,N)$, L1, on-policy $F$",
        SLATE,
    )
    _roadmap_box(
        axis,
        (xs[1], top_y),
        (width, height),
        "A1",
        "GaugeFlow-base",
        "Alex-MP-20: 540,164 train\n$N,C,A,L,F$ + joint sampling",
        GREEN,
    )
    _roadmap_box(
        axis,
        (xs[2], top_y),
        (width, height),
        "B",
        "Physical representation",
        r"MatPES + force labels" + "\n" + r"feature distillation; $E,F,\sigma$",
        GOLD,
    )
    _roadmap_box(
        axis,
        (xs[3], top_y),
        (width, height),
        "C",
        "Scale structural prior",
        "LeMat inventory: 5.438M\nsource-balanced replay",
        BLUE,
    )

    center_y = top_y + height / 2
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        _arrow(axis, (left + width + 0.003, center_y), (right - 0.006, center_y))

    d_x, d_y, d_w, d_h = xs[1], bottom_y, width, height
    _roadmap_box(
        axis,
        (d_x, d_y),
        (d_w, d_h),
        "D",
        "Independent response oracle",
        "JARVIS/MP/PhononDB\nauxiliary tensor targets; OOD split",
        GOLD,
    )
    _roadmap_box(
        axis,
        (xs[2], bottom_y),
        (width, height),
        "E",
        "Tensor-orbit adapter",
        "synthetic Gate first\nCartesian atlas + PEFT",
        RED,
    )
    _roadmap_box(
        axis,
        (xs[3], bottom_y),
        (width, height),
        "F",
        "Constrained post-training",
        "orbit, stability, retention\ntrust region + uncertainty",
        SLATE,
    )
    _arrow(axis, (xs[1] + width / 2, top_y - 0.006), (d_x + width / 2, bottom_y + height + 0.006))
    _arrow(axis, (d_x + width + 0.003, bottom_y + height / 2), (xs[2] - 0.006, bottom_y + height / 2))
    _arrow(axis, (xs[3] + width / 2, top_y - 0.006), (xs[2] + width / 2, bottom_y + height + 0.006))
    _arrow(axis, (xs[2] + width + 0.003, bottom_y + height / 2), (xs[3] - 0.006, bottom_y + height / 2))
    return _save(figure, output_dir, "gaugeflow_training_roadmap")


def main() -> None:
    args = _arguments()
    reports = args.reports_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "composition": reports / "h1a_e1_absolute_likelihood_v1" / "result.json",
        "legacy_assignment_q1": reports / "h1a_oracle_c_assignment_q1_v1" / "result.json",
        "geometry_expressivity": reports / "h1a_assignment_geometry_expressivity_audit_v1" / "result.json",
        "orderless_q0": reports / "h1a_assignment_orderless_law_q0_v1" / "result.json",
        "conditional_coordinate": reports / "h1a_coordinate_clean_side_information_one_pass_v1" / "result.json",
    }
    payloads = {name: _read_json(path) for name, path in source_paths.items()}

    outputs: list[Path] = []
    outputs += plot_base_factorization(output_dir)
    outputs += plot_interface_evidence(
        payloads["composition"],
        payloads["legacy_assignment_q1"],
        payloads["geometry_expressivity"],
        payloads["orderless_q0"],
        output_dir,
    )
    outputs += plot_coordinate_qualification(payloads["conditional_coordinate"], output_dir)
    outputs += plot_pretraining_roadmap(output_dir)

    manifest = {
        "protocol": "paper_current_state_figures_v1",
        "sources": {
            name: {"path": path.relative_to(reports).as_posix(), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claims": {
            "base_probability_factorization": "method definition; no qualification claim",
            "composition_assignment_evidence": "p(C|N) passed; unary Q1 failed; geometry/Q0 are no-training audits",
            "conditional_coordinate_qualification": "clean-A/L low/middle-noise conditional scope only",
            "gaugeflow_training_roadmap": "planned dependency graph; not completed evidence",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
