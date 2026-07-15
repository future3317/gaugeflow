"""Numerical, no-training audit for the versioned harmonic conditioning grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from gaugeflow.harmonic import HarmonicRelativeAlignment
from gaugeflow.manifold import lattice_to_log_vector
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import piezo_to_irreps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[24, 60, 120, 240])
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.grid_sizes) < 2 or min(args.grid_sizes) < 2:
        raise ValueError("provide at least two grid sizes, each at least two")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; do not silently substitute CPU")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    tensor = torch.randn((1, 3, 3, 3), generator=generator, device=device)
    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
    condition = piezo_to_irreps(tensor)
    directions = torch.nn.functional.normalize(torch.randn((48, 3), generator=generator, device=device), dim=-1)
    edge_graph = torch.zeros((directions.shape[0],), dtype=torch.long, device=device)

    values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for grid_size in sorted(set(args.grid_sizes)):
        module = HarmonicRelativeAlignment(grid_size=grid_size).to(device).eval()
        with torch.no_grad():
            aligned, posterior, entropy, _ = module(condition, directions, edge_graph)
        values[grid_size] = (aligned.cpu(), entropy.cpu())
    reference_size = max(values)
    reference, _ = values[reference_size]
    rows = []
    for grid_size, (aligned, entropy) in values.items():
        relative = torch.linalg.vector_norm(aligned - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        rows.append({
            "grid_size": grid_size,
            "mean_entropy": float(entropy.mean()),
            "relative_aligned_irrep_difference_to_max_grid": float(relative),
            "reference_grid_size": reference_size,
        })

    smoke: dict[str, bool] = {}
    type_state = F.one_hot(torch.tensor([6, 7, 48], device=device), num_classes=119).to(dtype=condition.dtype)
    frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.7, 0.1], [0.8, 0.5, 0.6]], device=device, dtype=condition.dtype)
    lattice_log = lattice_to_log_vector(torch.eye(3, device=device, dtype=condition.dtype).unsqueeze(0))
    batch = torch.zeros((3,), device=device, dtype=torch.long)
    present = torch.ones((1, 1), device=device, dtype=torch.bool)
    for mode in ("direct_irrep_complete_v1", "harmonic_alignment_v1"):
        model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=24, conditioning_mode=mode).to(device)
        outputs = model(type_state, frac_coords, lattice_log, batch, torch.tensor([0.5], device=device), condition, present)
        objective = sum(value.square().mean() for value in outputs[:3])
        objective.backward()
        smoke[mode] = bool(torch.isfinite(objective)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
        )
        if not smoke[mode]:
            raise RuntimeError(f"{mode} failed CUDA forward/backward smoke qualification")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "harmonic_grid_refinement.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = args.output_dir / "harmonic_grid_refinement.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# H1 harmonic-grid refinement operator audit\n\n")
        handle.write("This is a deterministic numerical operator check, not training or a conditional-generation gate. "
                     "The largest declared grid is a reference, not exact SO(3) integration.\n\n")
        handle.write(f"- device: `{device}`\n- seed: `{args.seed}`\n- reference grid: `{reference_size}`\n\n")
        handle.write(f"- CUDA forward/backward smoke: `{smoke}`\n\n")
        handle.write("| grid | mean posterior entropy | relative aligned-irrep difference to reference |\n")
        handle.write("|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['grid_size']} | {row['mean_entropy']:.8f} | "
                f"{row['relative_aligned_irrep_difference_to_max_grid']:.8e} |\n"
            )
        handle.write("\nNo threshold is declared here. A later causal training protocol must pre-register its grid "
                     "and an acceptance tolerance before inspecting generation results.\n")


if __name__ == "__main__":
    main()
