"""Audit the physical metric induced by the active fractional-torus path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.quotient_score import factorized_translation_quotient_scaled_score
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule
from gaugeflow.production.state_projection import project_translation_state


def physical_fractional_noise_covariance(lattice: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return Cov(delta f @ L) for isotropic row-vector fractional noise."""
    if lattice.ndim != 3 or lattice.shape[-2:] != (3, 3) or sigma <= 0.0:
        raise ValueError("lattice must be [graphs,3,3] and sigma must be positive")
    return float(sigma) ** 2 * lattice.transpose(-1, -2) @ lattice


def unimodular_shear_covariance_relative_difference(
    lattice: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Measure the path change under an equivalent non-orthogonal cell chart."""
    if transform.shape != (3, 3):
        raise ValueError("basis transform must be 3x3")
    determinant = torch.linalg.det(transform.double())
    if not torch.isclose(determinant.abs(), determinant.new_tensor(1.0), atol=1e-12):
        raise ValueError("basis transform must be unimodular")
    original = physical_fractional_noise_covariance(lattice, 1.0)
    transformed = physical_fractional_noise_covariance(
        transform.to(lattice).unsqueeze(0) @ lattice, 1.0
    )
    return torch.linalg.matrix_norm(transformed - original) / torch.linalg.matrix_norm(
        original
    ).clamp_min(torch.finfo(lattice.dtype).tiny)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double()
    right = right.double()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(torch.finfo(left.dtype).tiny))


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.double()
    probabilities = values.new_tensor([0.05, 0.5, 0.95])
    quantiles = torch.quantile(values, probabilities)
    return {
        "mean": float(values.mean()),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p95_to_p05": float(quantiles[2] / quantiles[0].clamp_min(1e-30)),
    }


@torch.no_grad()
def run_audit(
    protocol: dict[str, Any], dataset: PackedAlexP1Dataset, *, device: torch.device
) -> dict[str, Any]:
    sample = protocol["sample"]
    selected = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(sample["selection_seed"]))
    )[: int(sample["graphs"])]
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(protocol["path"]["coordinate_sigma_min"]),
        sigma_max=float(protocol["path"]["coordinate_sigma_max"]),
    )
    records: dict[float, dict[str, list[torch.Tensor]]] = {
        float(time): {"physical": [], "normalized": [], "target": [], "endpoint": [], "cell": []}
        for time in protocol["path"]["times"]
    }
    shear_values: list[torch.Tensor] = []
    transform = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
    )
    generator = torch.Generator(device=device).manual_seed(int(sample["noise_seed"]))
    for start in range(0, selected.numel(), int(sample["batch_size"])):
        indices = selected[start : start + int(sample["batch_size"])]
        packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
        graphs = int(packed.num_graphs)
        clean = project_translation_state(packed.frac_coords, packed.batch, graphs)
        counts = torch.bincount(packed.batch, minlength=graphs)
        volume = torch.linalg.det(packed.lattice)
        cell_scale = (volume / counts.to(volume)).pow(1.0 / 3.0)
        shear_values.append(
            unimodular_shear_covariance_relative_difference(packed.lattice, transform).cpu()
        )
        for time_value in records:
            time = packed.lattice.new_full((graphs,), time_value)
            sigma = schedule.sigma(time)
            for _ in range(int(sample["noise_replicates"])):
                noise = torch.randn(
                    clean.shape, dtype=clean.dtype, device=device, generator=generator
                )
                displacement = sigma[packed.batch, None] * noise
                horizontal = project_translation_state(displacement, packed.batch, graphs)
                cartesian = torch.einsum(
                    "ni,nij->nj", horizontal, packed.lattice[packed.batch]
                )
                physical_rms = scatter(
                    cartesian.square().sum(-1),
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                ).sqrt()
                scaled_score = factorized_translation_quotient_scaled_score(
                    displacement, sigma, packed.batch, graphs
                )
                target_energy = scatter(
                    scaled_score.square().sum(-1) / 3.0,
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                )
                score = scaled_score / sigma[packed.batch, None]
                estimate = clean + displacement + sigma[packed.batch, None].square() * score
                endpoint_delta = project_translation_state(estimate - clean, packed.batch, graphs)
                endpoint_cartesian = torch.einsum(
                    "ni,nij->nj", endpoint_delta, packed.lattice[packed.batch]
                )
                endpoint_rms = scatter(
                    endpoint_cartesian.square().sum(-1),
                    packed.batch,
                    dim=0,
                    dim_size=graphs,
                    reduce="mean",
                ).sqrt()
                bucket = records[time_value]
                bucket["physical"].append(physical_rms.cpu())
                bucket["normalized"].append((physical_rms / cell_scale).cpu())
                bucket["target"].append(target_energy.cpu())
                bucket["endpoint"].append(endpoint_rms.cpu())
                bucket["cell"].append(cell_scale.cpu())
    time_resolved: list[dict[str, Any]] = []
    for time_value, bucket in records.items():
        physical = torch.cat(bucket["physical"])
        normalized = torch.cat(bucket["normalized"])
        target = torch.cat(bucket["target"])
        endpoint = torch.cat(bucket["endpoint"])
        cell = torch.cat(bucket["cell"])
        time_resolved.append(
            {
                "time": time_value,
                "sigma_fractional": float(schedule.sigma(torch.tensor(time_value))),
                "physical_noise_rms_angstrom": _summary(physical),
                "noise_rms_over_per_atom_cell_scale": _summary(normalized),
                "scaled_score_target_energy": _summary(target),
                "oracle_endpoint_rms_angstrom": _summary(endpoint),
                "log_physical_rms_to_log_cell_scale_correlation": _correlation(
                    physical.log(), cell.log()
                ),
                "log_target_energy_to_log_cell_scale_correlation": _correlation(
                    target.clamp_min(1e-30).log(), cell.log()
                ),
            }
        )
    shear = torch.cat(shear_values)
    thresholds = protocol["diagnostic_thresholds"]
    maximum_spread = max(
        value["physical_noise_rms_angstrom"]["p95_to_p05"] for value in time_resolved
    )
    minimum_correlation = min(
        abs(value["log_physical_rms_to_log_cell_scale_correlation"])
        for value in time_resolved
    )
    shear_summary = _summary(shear)
    checks = {
        "physical_noise_spread_detected": maximum_spread
        >= float(thresholds["physical_rms_p95_to_p05_min"]),
        "cell_scale_correlation_detected": minimum_correlation
        >= float(thresholds["log_physical_rms_to_log_cell_scale_abs_correlation_min"]),
        "nonorthogonal_chart_dependence_detected": shear_summary["median"]
        >= float(thresholds["unimodular_shear_covariance_relative_median_min"]),
    }
    detected = all(checks.values())
    return {
        "protocol": protocol["protocol"],
        "graphs": int(selected.numel()),
        "noise_replicates": int(sample["noise_replicates"]),
        "time_resolved": time_resolved,
        "unimodular_shear_covariance_relative_difference": shear_summary,
        "checks": checks,
        "metric_mismatch_hypothesis_detected": detected,
        "decision": (
            "freeze_separate_memorization_comparison_before_path_replacement"
            if detected
            else "retain_active_path_and_run_fixed_subset_memorization"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_coordinate_path_metric_audit_v1":
        raise ValueError("coordinate-path metric protocol identity mismatch")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["prerequisites"]["cache_manifest_sha256"]
    ):
        raise ValueError("coordinate-path metric cache mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    dataset = PackedAlexP1Dataset(args.cache_root, str(protocol["sample"]["split"]))
    result = run_audit(protocol, dataset, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
