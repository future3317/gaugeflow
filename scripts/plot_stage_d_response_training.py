"""Plot the formal Stage-D multi-task learning trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} does not contain JSON metric records")
    return rows


def _series(rows: list[dict[str, float]], name: str) -> tuple[list[float], list[float]]:
    return [float(row["step"]) for row in rows], [float(row[name]) for row in rows]


def main() -> None:
    args = parse_args()
    training = _read(args.training)
    validation = _read(args.validation)
    plt.rcParams.update({"font.size": 8, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)

    train_step, train_total = _series(training, "loss")
    val_step, val_total = _series(validation, "loss")
    axes[0, 0].plot(train_step, train_total, color="#4C78A8", alpha=0.55, label="train minibatch")
    axes[0, 0].plot(val_step, val_total, "o-", color="#E45756", label="EMA validation")
    axes[0, 0].axvline(4500, color="black", linestyle="--", linewidth=0.9, label="selected")
    axes[0, 0].set(title="Composite response objective", ylabel="loss")
    axes[0, 0].legend(frameon=False)

    for name, label, color in (
        ("piezoelectric_loss", "piezoelectric", "#4C78A8"),
        ("dielectric_loss", "dielectric", "#F58518"),
        ("elastic_loss", "elastic", "#54A24B"),
    ):
        step, value = _series(validation, name)
        axes[0, 1].plot(step, value, "o-", label=label, color=color)
    axes[0, 1].set(title="Graph-level Cartesian tensors", ylabel="validation loss")
    axes[0, 1].legend(frameon=False)

    for name, label, color in (
        ("born_loss", "Born charge", "#B279A2"),
        ("gamma_loss", "Gamma modes", "#FF9DA6"),
        ("internal_strain_loss", "internal strain", "#9D755D"),
    ):
        step, value = _series(validation, name)
        axes[1, 0].plot(step, value, "o-", label=label, color=color)
    axes[1, 0].set(
        title="Atomic and soft-mode tasks",
        xlabel="optimizer step",
        ylabel="validation loss",
    )
    axes[1, 0].legend(frameon=False)

    probe_step, probe = _series(validation, "piezoelectric_probe_loss")
    throughput_step, throughput = _series(training, "graphs_per_second")
    axes[1, 1].plot(probe_step, probe, "o-", color="#72B7B2", label="response-field probe")
    axes[1, 1].set(
        title="Constitutive field and throughput",
        xlabel="optimizer step",
        ylabel="probe loss",
    )
    throughput_axis = axes[1, 1].twinx()
    throughput_axis.plot(
        throughput_step,
        throughput,
        color="#BAB0AC",
        alpha=0.7,
        label="graphs/s",
    )
    throughput_axis.set_ylabel("graphs/s")
    lines = axes[1, 1].lines + throughput_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.18, linewidth=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=240)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
