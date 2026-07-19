"""Render reproducible H1a training and mechanism diagnostics.

The JSONL/JSON artifacts remain the canonical numerical record.  This script
only turns those records into static, review-friendly PNG/PDF figures; it does
not smooth, filter, or alter any acceptance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

INK = "#20242A"
BLUE = "#2864A5"
GOLD = "#C58B18"
ORANGE = "#D66A2C"
OLIVE = "#71843F"
GRID = "#D9DDE3"


def _pyplot() -> Any:
    """Load the optional plotting stack only when a figure is requested."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append({str(key): item for key, item in value.items()})
    if not records:
        raise ValueError(f"{path} contains no metric records")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=INK, labelsize=8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def _save(figure: plt.Figure, output: Path, stem: str) -> list[Path]:
    plt = _pyplot()
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths


def plot_learning_curve(
    metrics: list[dict[str, Any]], result: dict[str, Any], output: Path
) -> list[Path]:
    plt = _pyplot()
    steps = np.asarray([record["step"] for record in metrics])
    coordinate = np.asarray([record["coordinate_loss"] for record in metrics])
    gradient = np.asarray([record["gradient_norm"] for record in metrics])
    throughput = np.asarray([record["graphs_per_second"] for record in metrics])
    memory = np.asarray([record["peak_cuda_memory_mib"] for record in metrics])

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(steps, coordinate, color=BLUE, linewidth=1.15, marker="o", markersize=2.5)
    axis.set(title="Coordinate DSM training loss", xlabel="Optimizer step", ylabel="MSE")
    _style_axis(axis)

    axis = axes[0, 1]
    validation = result.get("validation", {})
    validation_steps = np.asarray(sorted(int(step) for step in validation))
    validation_loss = np.asarray(
        [float(validation[str(step)]["coordinate"]) for step in validation_steps]
    )
    axis.plot(
        validation_steps,
        validation_loss,
        color=GOLD,
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="EMA validation",
    )
    if validation_loss.size:
        gate = 0.5 * validation_loss[0]
        axis.axhline(gate, color=INK, linestyle="--", linewidth=1.0, label="Frozen 0.5 ratio gate")
    axis.set(title="EMA validation learning curve", xlabel="Optimizer step", ylabel="Coordinate MSE")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)

    axis = axes[1, 0]
    axis.plot(steps, gradient, color=ORANGE, linewidth=1.2, marker="o", markersize=2.5)
    axis.axhline(1.0, color=INK, linestyle="--", linewidth=1.0, label="Clip norm")
    axis.set(title="Gradient norm", xlabel="Optimizer step", ylabel="L2 norm")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)

    axis = axes[1, 1]
    axis.plot(steps, throughput, color=OLIVE, linewidth=1.2, label="Throughput")
    axis.set(title="CUDA utilization proxies", xlabel="Optimizer step", ylabel="Graphs / second")
    _style_axis(axis)
    memory_axis = axis.twinx()
    memory_axis.plot(steps, memory, color=INK, linewidth=1.0, linestyle=":", label="Peak memory")
    memory_axis.set_ylabel("Peak allocated CUDA memory (MiB)", color=INK)
    memory_axis.tick_params(colors=INK, labelsize=8)
    lines = axis.lines + memory_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    figure.suptitle(
        "H1a coordinate pretraining diagnostics\nRaw logged values; no smoothing",
        color=INK,
        fontsize=13,
    )
    return _save(figure, output, "training_learning_curve")


def plot_score_and_rollout(result: dict[str, Any], output: Path) -> list[Path]:
    plt = _pyplot()
    score = result.get("score_calibration", [])
    rollout = result.get("conditional_rollout", [])
    times = np.asarray([float(item["time"]) for item in score])
    rms = np.asarray([float(item["endpoint_rms_angstrom"]) for item in score])
    explained = np.asarray([float(item["score_explained_fraction"]) for item in score])
    cosine = np.asarray([float(item["prediction_target_cosine"]) for item in score])

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(times, rms, color=BLUE, linewidth=1.7, marker="o")
    axis.scatter([0.005, 0.1], [0.04, 0.08], color=INK, marker="x", s=55, label="Frozen endpoint limits")
    axis.set(title="Teacher-forced endpoint error", xlabel="Diffusion time t", ylabel="Periodic RMS (Å)")
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)

    axis = axes[0, 1]
    axis.plot(times, explained, color=GOLD, linewidth=1.7, marker="o", label="Explained fraction")
    axis.set(title="Score explained fraction", xlabel="Diffusion time t", ylabel="Fraction")
    axis.set_ylim(0.0, max(1.0, float(explained.max(initial=0.0)) * 1.05))
    _style_axis(axis)

    axis = axes[1, 0]
    axis.plot(times, cosine, color=ORANGE, linewidth=1.7, marker="o")
    axis.set(title="Prediction–target alignment", xlabel="Diffusion time t", ylabel="Cosine")
    axis.set_ylim(-1.0, 1.0)
    _style_axis(axis)

    axis = axes[1, 1]
    starts = np.asarray([float(item["start_time"]) for item in rollout])
    means = np.asarray([float(item["mean_endpoint_rms_angstrom"]) for item in rollout])
    if rollout:
        minimum = np.asarray([float(item["endpoint_rms_quantiles_angstrom"][0]) for item in rollout])
        medians = np.asarray([float(item["endpoint_rms_quantiles_angstrom"][1]) for item in rollout])
        p90 = np.asarray([float(item["endpoint_rms_quantiles_angstrom"][2]) for item in rollout])
        p95 = np.asarray([float(item["endpoint_rms_quantiles_angstrom"][3]) for item in rollout])
        maximum = np.asarray([float(item["endpoint_rms_quantiles_angstrom"][4]) for item in rollout])
        errors = np.vstack([medians - minimum, maximum - medians])
        axis.errorbar(
            starts,
            medians,
            yerr=errors,
            color=OLIVE,
            marker="o",
            capsize=4,
            linewidth=1.5,
            label="Median and min–max",
        )
        axis.scatter(starts, means, color=INK, marker="x", label="Mean")
        axis.scatter(starts, p90, color=ORANGE, marker="^", label="P90")
        axis.scatter(starts, p95, facecolors="none", edgecolors=ORANGE, marker="s", label="P95")
    axis.set(
        title="100-step stochastic rollout",
        xlabel="Rollout start time",
        ylabel="Endpoint RMS (Å)",
    )
    axis.legend(frameon=False, fontsize=8)
    _style_axis(axis)

    figure.suptitle("H1a score and rollout diagnostics", color=INK, fontsize=13)
    return _save(figure, output, "score_and_rollout")


def _slot_matrix(
    checkpoints: dict[str, Any], metric: str
) -> tuple[np.ndarray, list[str], list[str]]:
    checkpoint_ids = sorted(checkpoints, key=int)
    times = sorted({time for checkpoint in checkpoints.values() for time in checkpoint}, key=float)
    maximum_layers = max(
        len(checkpoints[checkpoint][time])
        for checkpoint in checkpoint_ids
        for time in checkpoints[checkpoint]
    )
    rows = [(checkpoint, layer) for checkpoint in checkpoint_ids for layer in range(maximum_layers)]
    matrix = np.full((len(rows), len(times)), np.nan, dtype=np.float64)
    for row, (checkpoint, layer) in enumerate(rows):
        for column, time in enumerate(times):
            layers = checkpoints[checkpoint].get(time, [])
            if layer < len(layers):
                matrix[row, column] = float(layers[layer][metric])
    return matrix, [f"{checkpoint}/L{layer}" for checkpoint, layer in rows], times


def _heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt = _pyplot()
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=vmin, vmax=vmax)
    axis.set_title(title, color=INK, fontsize=10)
    axis.set_xticks(np.arange(len(columns)), columns, fontsize=7)
    axis.set_yticks(np.arange(len(rows)), rows, fontsize=6)
    axis.set_xlabel("Diffusion time t", color=INK)
    axis.set_ylabel("Checkpoint / layer", color=INK)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.03)


def plot_slot_diagnostics(slot_audit: dict[str, Any], output: Path) -> list[Path]:
    plt = _pyplot()
    checkpoints = slot_audit.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        return []
    specifications = [
        ("maximum_global_slot_mass", "Maximum global slot mass", 0.0, 1.0),
        ("normalized_assignment_entropy", "Normalized assignment entropy", 0.0, 1.0),
        ("effective_slot_count", "Effective occupied slots", 1.0, 8.0),
        ("slot_representation_effective_rank", "Slot representation effective rank", 1.0, 8.0),
        ("mean_absolute_inter_slot_cosine", "Mean absolute inter-slot cosine", 0.0, 1.0),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    for axis, (metric, title, vmin, vmax) in zip(axes.flat, specifications, strict=False):
        matrix, rows, times = _slot_matrix(checkpoints, metric)
        _heatmap(axis, matrix, rows, times, title, vmin=vmin, vmax=vmax)

    final_checkpoint = max(checkpoints, key=int)
    final_times = sorted(checkpoints[final_checkpoint], key=float)
    occupancy_rows: list[str] = []
    occupancy: list[list[float]] = []
    for time in final_times:
        for layer, value in enumerate(checkpoints[final_checkpoint][time]):
            occupancy_rows.append(f"t={time}/L{layer}")
            occupancy.append([float(item) for item in value["occupancy"]])
    _heatmap(
        axes[1, 2],
        np.asarray(occupancy),
        occupancy_rows,
        [str(index) for index in range(len(occupancy[0]))],
        f"Final checkpoint {final_checkpoint}: slot occupancy",
        vmin=0.0,
        vmax=max(0.25, float(np.max(occupancy))),
    )
    axes[1, 2].set_xlabel("Slot index", color=INK)

    figure.suptitle(
        "Induced-slot mechanism diagnostics\nRows retain checkpoint, layer, and time structure",
        color=INK,
        fontsize=13,
    )
    return _save(figure, output, "slot_diagnostics")


def plot_paper_summary(
    result: dict[str, Any], slot_audit: dict[str, Any], output: Path
) -> list[Path]:
    """Build one dense two-column figure without discarding audit dimensions."""
    plt = _pyplot()
    checkpoints = slot_audit.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        return []
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 7.2), constrained_layout=True)

    validation = result["validation_curve"]
    validation_steps = np.asarray(sorted(int(step) for step in validation))
    validation_loss = np.asarray(
        [float(validation[str(step)]["coordinate"]) for step in validation_steps]
    )
    axis = axes[0, 0]
    axis.plot(validation_steps, validation_loss, color=GOLD, linewidth=2.0, marker="o")
    axis.axhline(0.5 * validation_loss[0], color=INK, linestyle="--", linewidth=1.0)
    axis.set(
        title="(a) EMA validation learning curve",
        xlabel="Optimizer step",
        ylabel="Coordinate MSE",
    )
    _style_axis(axis)

    score = result["score_calibration"]
    times = np.asarray([float(item["time"]) for item in score])
    rms = np.asarray([float(item["endpoint_rms_angstrom"]) for item in score])
    explained = np.asarray([float(item["score_explained_fraction"]) for item in score])
    axis = axes[0, 1]
    axis.plot(times, rms, color=BLUE, linewidth=1.8, marker="o", label="Endpoint RMS")
    axis.scatter([0.005, 0.1], [0.04, 0.08], color=INK, marker="x", s=45, label="Frozen limits")
    axis.set(
        title="(b) Time-resolved coordinate field",
        xlabel="Diffusion time t",
        ylabel="Periodic RMS (Å)",
    )
    _style_axis(axis)
    explained_axis = axis.twinx()
    explained_axis.plot(
        times,
        explained,
        color=ORANGE,
        linewidth=1.4,
        marker="s",
        label="Explained fraction",
    )
    explained_axis.set_ylabel("Explained fraction", color=INK)
    explained_axis.set_ylim(0.0, 1.0)
    explained_axis.tick_params(colors=INK, labelsize=8)
    lines = axis.lines + explained_axis.lines
    labels = [line.get_label() for line in lines]
    handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(handles + lines[1:], legend_labels + labels[1:], frameon=False, fontsize=7)

    rollout = result["conditional_rollout"]
    starts = np.asarray([float(item["start_time"]) for item in rollout])
    means = np.asarray([float(item["mean_endpoint_rms_angstrom"]) for item in rollout])
    quantiles = np.asarray([item["endpoint_rms_quantiles_angstrom"] for item in rollout], dtype=float)
    axis = axes[0, 2]
    axis.errorbar(
        starts,
        quantiles[:, 1],
        yerr=np.vstack([quantiles[:, 1] - quantiles[:, 0], quantiles[:, 4] - quantiles[:, 1]]),
        color=OLIVE,
        marker="o",
        capsize=4,
        label="Median / min–max",
    )
    axis.scatter(starts, means, color=INK, marker="x", label="Mean")
    axis.scatter(starts, quantiles[:, 2], color=ORANGE, marker="^", label="P90")
    axis.set(
        title="(c) 100-step stochastic rollout",
        xlabel="Rollout start time",
        ylabel="Endpoint RMS (Å)",
    )
    axis.legend(frameon=False, fontsize=7)
    _style_axis(axis)

    heatmaps = [
        ("maximum_global_slot_mass", "(d) Maximum slot mass", 0.0, 0.25),
        ("slot_representation_effective_rank", "(e) Representation effective rank", 1.0, 8.0),
        ("mean_absolute_inter_slot_cosine", "(f) Mean absolute inter-slot cosine", 0.0, 1.0),
    ]
    for axis, (metric, title, vmin, vmax) in zip(axes[1], heatmaps, strict=True):
        matrix, rows, slot_times = _slot_matrix(checkpoints, metric)
        _heatmap(axis, matrix, rows, slot_times, title, vmin=vmin, vmax=vmax)

    return _save(figure, output, "paper_h1a_local_operator_closure")


def main() -> None:
    args = parse_args()
    metrics_path = args.run / "training_metrics.jsonl"
    result_path = args.report / "result.json"
    slot_path = args.report / "slot_audit.json"
    output = args.output or args.report / "figures"
    output.mkdir(parents=True, exist_ok=True)

    metrics = _read_jsonl(metrics_path)
    result = _read_json(result_path)
    figures = []
    figures.extend(plot_learning_curve(metrics, result, output))
    figures.extend(plot_score_and_rollout(result, output))
    source_paths = [metrics_path, result_path]
    if slot_path.exists():
        slot_audit = _read_json(slot_path)
        figures.extend(plot_slot_diagnostics(slot_audit, output))
        figures.extend(plot_paper_summary(result, slot_audit, output))
        source_paths.append(slot_path)
    manifest = {
        "protocol": result.get("protocol"),
        "canonical_sources": {
            str(path): _sha256(path) for path in source_paths
        },
        "figures": [str(path) for path in figures],
        "rendering": {
            "raw_training_values": True,
            "smoothing": None,
            "png_dpi": 200,
            "pdf_vector_output": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
