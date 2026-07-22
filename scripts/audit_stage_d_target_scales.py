"""Report train-only invariant scale distributions for Stage-D responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.production.response_data import (
    StageDResponseDataset,
    collate_response_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summary(value: torch.Tensor) -> dict[str, float | int]:
    value = value.double().flatten()
    value = value[torch.isfinite(value)]
    if not value.numel():
        return {"count": 0}
    probabilities = torch.tensor(
        [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0], dtype=torch.float64
    )
    quantiles = torch.quantile(value, probabilities)
    return {
        "count": value.numel(),
        "minimum": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "p999": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "mean": float(value.mean()),
        "rms": float(value.square().mean().sqrt()),
    }


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.double().square().flatten(1).mean(dim=-1).sqrt()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-D scale audit {args.output}")
    dataset = StageDResponseDataset(args.cache, "train")
    batch = collate_response_records([dataset[index] for index in range(len(dataset))])
    target = batch.targets
    dielectric = target.dielectric[target.dielectric_mask].double()
    dielectric_trace = torch.einsum("gii->g", dielectric) / 3.0
    identity = torch.eye(3, dtype=torch.float64)
    dielectric_anisotropic = dielectric - dielectric_trace[:, None, None] * identity
    born = target.born_effective_charge[target.born_mask]
    internal_node_mask = target.internal_strain_mask.flatten(1).all(dim=-1)
    result = {
        "schema": "gaugeflow.stage_d_target_scale_audit.v1",
        "cache_sha256": dataset.manifest["cache_sha256"],
        "split": "train",
        "graphs": len(dataset),
        "atoms": int(batch.element_tokens.numel()),
        "piezoelectric_rms_per_graph": _summary(
            _tensor_rms(target.piezoelectric[target.piezoelectric_mask])
        ),
        "piezoelectric_exact_zero_fraction": float(
            (_tensor_rms(target.piezoelectric) == 0.0).double().mean()
        ),
        "dielectric_isotropic_trace": _summary(dielectric_trace),
        "dielectric_total_rms_per_graph": _summary(_tensor_rms(dielectric)),
        "dielectric_anisotropic_rms_per_graph": _summary(
            _tensor_rms(dielectric_anisotropic)
        ),
        "elastic_rms_gpa_per_graph": _summary(
            _tensor_rms(target.elastic[target.elastic_mask])
        ),
        "born_rms_per_atom": _summary(_tensor_rms(born)),
        "gamma_log_magnitude": _summary(
            target.gamma_log_magnitude[target.gamma_mask]
        ),
        "internal_strain_rms_per_atom": _summary(
            _tensor_rms(target.internal_strain[internal_node_mask])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
