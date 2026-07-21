"""Qualify the train-only categorical node-count law for GaugeFlow-base."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.blueprint import EmpiricalNodeCountPrior


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("node-count Gate requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _distribution(counts: torch.Tensor, maximum_atoms: int) -> torch.Tensor:
    histogram = torch.bincount(counts, minlength=maximum_atoms + 1)[1 : maximum_atoms + 1]
    return histogram.to(torch.float64) / counts.numel()


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    midpoint = 0.5 * (left + right)
    value = torch.zeros((), dtype=torch.float64)
    for probability in (left, right):
        present = probability > 0
        value += 0.5 * (
            probability[present] * (probability[present].log() - midpoint[present].log())
        ).sum()
    return float(value)


def _integer_wasserstein(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.abs(torch.cumsum(left - right, dim=0)).sum())


def _bootstrap_nll_difference_ucb95(
    model_minus_uniform_nll: torch.Tensor,
    *,
    resamples: int,
    seed: int,
    chunk_size: int = 128,
) -> float:
    if model_minus_uniform_nll.ndim != 1 or model_minus_uniform_nll.numel() < 2:
        raise ValueError("node-count bootstrap needs at least two structures")
    if resamples < 100 or chunk_size < 1:
        raise ValueError("node-count bootstrap resolution is too small")
    values = model_minus_uniform_nll.to(device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    means = []
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        index = torch.randint(
            values.numel(),
            (count, values.numel()),
            generator=generator,
        )
        means.append(values[index].mean(dim=1))
    return float(torch.quantile(torch.cat(means), 0.95))


def _panel_metrics(
    law: EmpiricalNodeCountPrior,
    node_counts: torch.Tensor,
    *,
    maximum_atoms: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    model_nll = -law.log_prob(node_counts)
    uniform_nll = math.log(maximum_atoms)
    target = _distribution(node_counts, maximum_atoms)
    model = torch.zeros(maximum_atoms, dtype=torch.float64)
    model[law.support - 1] = law.probabilities
    return {
        "structures": node_counts.numel(),
        "mean_nll": float(model_nll.mean()),
        "uniform_nll": uniform_nll,
        "nll_gain_over_uniform": uniform_nll - float(model_nll.mean()),
        "model_minus_uniform_nll_ucb95": _bootstrap_nll_difference_ucb95(
            model_nll - uniform_nll,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "js_divergence_from_train_law": _js_divergence(target, model),
        "total_variation_from_train_law": float(0.5 * torch.abs(target - model).sum()),
        "integer_wasserstein_from_train_law": _integer_wasserstein(target, model),
        "support_coverage": float(target[model > 0].sum()),
    }


def _load_counts(cache_root: Path, split: str) -> torch.Tensor:
    payload: Any = torch.load(
        cache_root / f"{split}.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    offsets = payload["offsets"]
    return offsets[1:] - offsets[:-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--iid-split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_node_count_gate_v1" or protocol.get(
        "status_before_run"
    ) != "frozen_not_run":
        raise ValueError("unexpected or unfrozen node-count protocol")
    source = protocol["source"]
    paths = {
        "cache_manifest": args.cache_root / "manifest.json",
        "iid_manifest": args.iid_split_root / "manifest.json",
        "fit_index": args.iid_split_root / "fit_index.pt",
        "calibration_index": args.iid_split_root / "calibration_index.pt",
        "test_index": args.iid_split_root / "test_index.pt",
    }
    for name, path in paths.items():
        if sha256_file(path) != source[f"{name}_sha256"]:
            raise ValueError(f"node-count source identity changed: {name}")
    cache_manifest = load_json_object(paths["cache_manifest"])
    iid_manifest = load_json_object(paths["iid_manifest"])
    if cache_manifest.get("qualified") is not True or iid_manifest.get("qualified") is not True:
        raise ValueError("node-count source split is not qualified")
    for split in ("train", "val", "test"):
        split_manifest = cache_manifest["splits"][split]
        if sha256_file(args.cache_root / split_manifest["tensor_file"]) != split_manifest[
            "tensor_sha256"
        ]:
            raise ValueError(f"node-count cache tensor identity changed: {split}")
    implementation_commit = _git_identity(repository)

    maximum_atoms = int(protocol["law"]["maximum_atoms"])
    train_counts = _load_counts(args.cache_root, "train")
    indices = {
        role: torch.load(
            paths[f"{role}_index"],
            map_location="cpu",
            weights_only=True,
        ).to(torch.long)
        for role in ("fit", "calibration", "test")
    }
    joined_indices = torch.cat(tuple(indices.values()))
    if joined_indices.numel() != train_counts.numel() or not torch.equal(
        torch.sort(joined_indices).values,
        torch.arange(train_counts.numel()),
    ):
        raise ValueError("IID node-count split does not partition the training rows exactly")
    count_panels = {role: train_counts[index] for role, index in indices.items()}
    law = EmpiricalNodeCountPrior.fit(count_panels["fit"])
    expected_support = torch.arange(1, maximum_atoms + 1)
    if not torch.equal(law.support, expected_support):
        raise ValueError("train-only node-count law does not cover the declared support")
    normalization_error = abs(float(law.probabilities.sum()) - 1.0)

    evaluation = protocol["evaluation"]
    panels = {
        role: _panel_metrics(
            law,
            count_panels[role],
            maximum_atoms=maximum_atoms,
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            bootstrap_seed=int(evaluation["bootstrap_seed"]) + index,
        )
        for index, role in enumerate(("calibration", "test"))
    }
    ood_panels = {
        split: _panel_metrics(
            law,
            _load_counts(args.cache_root, split),
            maximum_atoms=maximum_atoms,
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            bootstrap_seed=int(evaluation["bootstrap_seed"]) + 100 + index,
        )
        for index, split in enumerate(("val", "test"))
    }

    sample_count = int(evaluation["sample_count"])
    sample_seed = int(evaluation["sample_seed"])
    first = law.sample(sample_count, generator=torch.Generator().manual_seed(sample_seed))
    second = law.sample(sample_count, generator=torch.Generator().manual_seed(sample_seed))
    sampled = _distribution(first, maximum_atoms)
    model_distribution = law.probabilities
    sampling = {
        "draws": sample_count,
        "reproducible": torch.equal(first, second),
        "minimum_count": int(first.min()),
        "maximum_count": int(first.max()),
        "invalid_counts": int(((first < 1) | (first > maximum_atoms)).sum()),
        "sampling_failures": 0,
        "js_divergence_from_law": _js_divergence(sampled, model_distribution),
        "integer_wasserstein_from_law": _integer_wasserstein(sampled, model_distribution),
    }
    acceptance = protocol["acceptance"]
    checks = {
        "normalized": normalization_error <= float(acceptance["normalization_error_max"]),
        "full_support": torch.equal(law.support, expected_support),
        "calibration_nll": panels["calibration"]["mean_nll"]
        <= float(acceptance["iid_mean_nll_max"]),
        "test_nll": panels["test"]["mean_nll"] <= float(acceptance["iid_mean_nll_max"]),
        "calibration_uniform_gain": panels["calibration"]["nll_gain_over_uniform"]
        >= float(acceptance["iid_nll_gain_over_uniform_min"]),
        "test_uniform_gain": panels["test"]["nll_gain_over_uniform"]
        >= float(acceptance["iid_nll_gain_over_uniform_min"]),
        "calibration_bootstrap": panels["calibration"]["model_minus_uniform_nll_ucb95"]
        <= float(acceptance["model_minus_uniform_nll_ucb95_max"]),
        "test_bootstrap": panels["test"]["model_minus_uniform_nll_ucb95"]
        <= float(acceptance["model_minus_uniform_nll_ucb95_max"]),
        "calibration_js": panels["calibration"]["js_divergence_from_train_law"]
        <= float(acceptance["iid_js_divergence_max"]),
        "test_js": panels["test"]["js_divergence_from_train_law"]
        <= float(acceptance["iid_js_divergence_max"]),
        "calibration_wasserstein": panels["calibration"][
            "integer_wasserstein_from_train_law"
        ]
        <= float(acceptance["iid_integer_wasserstein_max"]),
        "test_wasserstein": panels["test"]["integer_wasserstein_from_train_law"]
        <= float(acceptance["iid_integer_wasserstein_max"]),
        "support_coverage": min(
            float(panels[role]["support_coverage"]) for role in ("calibration", "test")
        )
        >= float(acceptance["support_coverage_min"]),
        "sampling_reproducible": bool(sampling["reproducible"]),
        "sampling_js": sampling["js_divergence_from_law"]
        <= float(acceptance["sampling_js_divergence_max"]),
        "sampling_wasserstein": sampling["integer_wasserstein_from_law"]
        <= float(acceptance["sampling_integer_wasserstein_max"]),
        "valid_samples": sampling["invalid_counts"] == int(acceptance["invalid_counts"]),
        "zero_failures": sampling["sampling_failures"]
        == int(acceptance["sampling_failures"]),
    }
    qualified = all(checks.values())
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "implementation_commit": implementation_commit,
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "law": {
            "support": law.support.tolist(),
            "probabilities": law.probabilities.tolist(),
            "normalization_error": normalization_error,
            "fit_structures": count_panels["fit"].numel(),
        },
        "iid_panels": panels,
        "ood_stress_panels": ood_panels,
        "sampling": sampling,
        "hardware": {"device": "CPU", "torch": torch.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
