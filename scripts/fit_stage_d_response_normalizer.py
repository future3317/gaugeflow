"""Fit and bind the train-only Stage-D covariant response normalizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.response_data import (
    StageDResponseDataset,
    collate_response_records,
)
from gaugeflow.production.response_normalization import fit_response_normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-D normalizer {args.output}")
    dataset = StageDResponseDataset(args.cache, "train")
    batch = collate_response_records([dataset[index] for index in range(len(dataset))])
    source_count = int(batch.source_index.max()) + 1
    normalizer = fit_response_normalizer(
        batch.targets,
        batch.source_index,
        batch.batch,
        source_count=source_count,
    )
    fields = {
        name: getattr(normalizer, name).tolist()
        for name in normalizer.__dataclass_fields__
    }
    result = {
        "schema": "gaugeflow.stage_d_response_normalizer.v1",
        "qualified": True,
        "fit_split": "train",
        "fit_graphs": len(dataset),
        "fit_atoms": int(batch.element_tokens.numel()),
        "source_count": source_count,
        "cache_sha256": dataset.manifest["cache_sha256"],
        "cache_manifest_sha256": sha256_file(args.cache / "MANIFEST.json"),
        "fields": fields,
        "convention": (
            "train-only O(3)-invariant robust per-object scale; rank-two median "
            "isotropic identity location; invertible radial-asinh magnitude chart; "
            "Gamma median/MAD; masks preserve physical zero"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
