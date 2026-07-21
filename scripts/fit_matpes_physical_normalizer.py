"""Fit train-only covariance-preserving MatPES physical normalization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.matpes_data import (
    MatPESPhysicalBatch,
    fit_functional_physical_normalizer_from_batches,
)
from gaugeflow.production.matpes_index import (
    IndexedMatPESDataset,
    MatPESBatchCollator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")

    vocabulary = {"PBE": 0, "r2SCAN": 1}
    dataset = IndexedMatPESDataset(
        arguments.index,
        "train",
        verify_hashes=True,
        require_qualified=True,
    )
    if arguments.batch_size < 1 or arguments.num_workers < 0:
        raise ValueError("normalization loader dimensions are invalid")
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        collate_fn=MatPESBatchCollator(vocabulary, teacher_dim=1),
        persistent_workers=arguments.num_workers > 0,
    )
    counts = torch.zeros(len(vocabulary), dtype=torch.int64)

    def counted_batches() -> Iterator[MatPESPhysicalBatch]:
        for batch in loader:
            counts.add_(torch.bincount(batch.functional_index, minlength=len(vocabulary)))
            yield batch

    normalizer = fit_functional_physical_normalizer_from_batches(
        counted_batches(),
        functional_vocabulary=vocabulary,
    )
    if bool((counts == 0).any()) or int(counts.sum()) != len(dataset):
        raise RuntimeError("normalization stream did not cover the qualified train split")
    functional_counts = {
        functional: int(counts[index])
        for functional, index in vocabulary.items()
    }
    index_manifest = arguments.index / "manifest.json"
    payload = {
        "schema": "gaugeflow.matpes_physical_normalizer.v1",
        "qualified": True,
        "fit_split": "train",
        "index_manifest": str(index_manifest.resolve()),
        "index_manifest_sha256": sha256_file(index_manifest),
        "functional_vocabulary": vocabulary,
        "functional_row_counts": dict(sorted(functional_counts.items())),
        "energy_location": normalizer.energy_location.tolist(),
        "energy_scale": normalizer.energy_scale.tolist(),
        "force_scale": normalizer.force_scale.tolist(),
        "stress_isotropic_location": normalizer.stress_isotropic_location.tolist(),
        "stress_scale": normalizer.stress_scale.tolist(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
