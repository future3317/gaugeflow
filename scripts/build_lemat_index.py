"""Build the versioned LeMat row-group index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.lemat_index import build_lemat_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-atoms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument(
        "--physical-label-policy",
        choices=("compatible_only", "all_with_functional"),
        default="compatible_only",
    )
    parser.add_argument("--exclude-material-ids", type=Path)
    parser.add_argument("--max-row-groups-per-source", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded: set[str] = set()
    excluded_artifact_sha256: str | None = None
    if args.exclude_material_ids is not None:
        payload = json.loads(args.exclude_material_ids.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            raise ValueError("excluded material IDs must be a JSON string list")
        excluded = set(payload)
        excluded_artifact_sha256 = sha256_file(args.exclude_material_ids)
    sources = {
        functional.removeprefix("unique_"): sorted((args.root / functional).glob("*.parquet"))
        for functional in ("unique_pbe", "unique_pbesol", "unique_scan")
    }
    manifest = build_lemat_index(
        sources,
        args.output,
        maximum_atoms=args.maximum_atoms,
        seed=args.seed,
        physical_label_policy=args.physical_label_policy,
        excluded_material_ids=excluded,
        excluded_material_ids_artifact_sha256=excluded_artifact_sha256,
        max_row_groups_per_source=args.max_row_groups_per_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
