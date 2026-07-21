"""Fit train-only covariance-preserving MatPES physical normalization."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.matpes_data import (
    MatPESPhysicalRecord,
    fit_functional_physical_normalizer,
)
from gaugeflow.production.matpes_index import IndexedMatPESDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    counts: Counter[str] = Counter()

    def counted_records() -> Iterator[MatPESPhysicalRecord]:
        for record in dataset:
            counts[record.functional] += 1
            yield record

    normalizer = fit_functional_physical_normalizer(
        counted_records(),
        functional_vocabulary=vocabulary,
    )
    if set(counts) != set(vocabulary) or sum(counts.values()) != len(dataset):
        raise RuntimeError("normalization stream did not cover the qualified train split")
    index_manifest = arguments.index / "manifest.json"
    payload = {
        "schema": "gaugeflow.matpes_physical_normalizer.v1",
        "qualified": True,
        "fit_split": "train",
        "index_manifest": str(index_manifest.resolve()),
        "index_manifest_sha256": sha256_file(index_manifest),
        "functional_vocabulary": vocabulary,
        "functional_row_counts": dict(sorted(counts.items())),
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
