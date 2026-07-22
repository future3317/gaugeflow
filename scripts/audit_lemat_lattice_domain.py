"""Audit lattice conditioning for every row selected by a LeMat index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gaugeflow.file_utils import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-width-angstrom", type=float, default=0.5)
    parser.add_argument("--maximum-metric-condition", type=float, default=1.0e4)
    return parser.parse_args()


def _lattice_statistics(
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = lattice @ np.swapaxes(lattice, -1, -2)
    eigenvalues = np.linalg.eigvalsh(metric)
    minimum_width = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0))
    condition = eigenvalues[:, -1] / np.maximum(
        eigenvalues[:, 0], np.finfo(np.float64).tiny
    )
    return minimum_width, condition, np.linalg.det(lattice)


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite LeMat audit: {args.output}")
    if args.minimum_width_angstrom <= 0.0 or args.maximum_metric_condition <= 1.0:
        raise ValueError("LeMat lattice-domain bounds are invalid")

    manifest_path = args.index / "manifest.json"
    index_path = args.index / "index.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(index_path, map_location="cpu", weights_only=True)
    required = {"source_index", "row_group", "row_in_group", "split_index"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("LeMat index payload is incomplete")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("LeMat index manifest has no sources")

    source_index = payload["source_index"].numpy().astype(np.int64, copy=False)
    row_group = payload["row_group"].numpy().astype(np.int64, copy=False)
    row_in_group = payload["row_in_group"].numpy().astype(np.int64, copy=False)
    split_index = payload["split_index"].numpy().astype(np.int64, copy=False)
    size = source_index.size
    if not (row_group.size == row_in_group.size == split_index.size == size):
        raise ValueError("LeMat index tensors have inconsistent lengths")

    ordering = np.lexsort((row_in_group, row_group, source_index))
    ordered_source = source_index[ordering]
    ordered_group = row_group[ordering]
    boundaries = np.flatnonzero(
        np.r_[
            True,
            (ordered_source[1:] != ordered_source[:-1])
            | (ordered_group[1:] != ordered_group[:-1]),
            True,
        ]
    )
    counters: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    minimum_observed = float("inf")
    maximum_observed = 0.0
    parquet_files: dict[int, pq.ParquetFile] = {}
    split_names = ("train", "calibration", "test")

    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = ordering[begin:end]
        source = int(source_index[selected[0]])
        group = int(row_group[selected[0]])
        if source not in parquet_files:
            parquet_files[source] = pq.ParquetFile(str(sources[source]["path"]))
        parquet = parquet_files[source]
        table = parquet.read_row_group(group, columns=["lattice_vectors"])
        rows = pa.array(row_in_group[selected])
        lattice = np.asarray(
            table.take(rows)["lattice_vectors"].to_pylist(), dtype=np.float64
        )
        width, condition, determinant = _lattice_statistics(lattice)
        minimum_observed = min(minimum_observed, float(width.min()))
        maximum_observed = max(maximum_observed, float(condition.max()))
        width_failure = width < args.minimum_width_angstrom
        condition_failure = condition > args.maximum_metric_condition
        volume_failure = determinant <= 0.0
        union = width_failure | condition_failure | volume_failure
        counters["selected_rows"] += selected.size
        counters["minimum_width_failures"] += int(width_failure.sum())
        counters["metric_condition_failures"] += int(condition_failure.sum())
        counters["nonpositive_volume_failures"] += int(volume_failure.sum())
        counters["union_failures"] += int(union.sum())
        for split_id, split_name in enumerate(split_names):
            counters[f"{split_name}_rows"] += int((split_index[selected] == split_id).sum())
            counters[f"{split_name}_union_failures"] += int(
                (union & (split_index[selected] == split_id)).sum()
            )
        if bool(union.any()) and len(examples) < 20:
            for local in np.flatnonzero(union):
                index_position = int(selected[local])
                examples.append(
                    {
                        "index_position": index_position,
                        "source_index": source,
                        "row_group": group,
                        "row_in_group": int(row_in_group[index_position]),
                        "split": split_names[int(split_index[index_position])],
                        "minimum_width_angstrom": float(width[local]),
                        "metric_condition": float(condition[local]),
                        "lattice_determinant": float(determinant[local]),
                    }
                )
                if len(examples) == 20:
                    break

    result = {
        "schema": "gaugeflow.lemat_lattice_domain_audit.v1",
        "index": str(args.index),
        "index_manifest_sha256": sha256_file(manifest_path),
        "index_sha256": sha256_file(index_path),
        "thresholds": {
            "minimum_width_angstrom": args.minimum_width_angstrom,
            "maximum_metric_condition": args.maximum_metric_condition,
        },
        "counts": dict(sorted(counters.items())),
        "minimum_observed_width_angstrom": minimum_observed,
        "maximum_observed_metric_condition": maximum_observed,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
