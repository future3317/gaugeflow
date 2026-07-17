"""No-write real-data smoke test for the frozen H0-E implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.catalogue.parent_decomposition import balanced_selection
from scripts.build_h0_e_parent_decomposition_pilot import (
    _load_assignment,
    _load_source_rows,
    _process_one,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=4)
    args = parser.parse_args()
    if args.per_split < 1:
        raise ValueError("per-split diagnostic count must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, assignment = _load_assignment(config, args.data_root)
    pilot = config["pilot"]
    frozen_selection = list(
        balanced_selection(
            assignment,
            split_counts={
                name: int(pilot["gaugeflow_split_counts"][name])
                for name in pilot["source_splits"]
            },
            seed=int(pilot["seed"]),
            site_boundaries=pilot["primitive_site_bins"],
        )
    )
    selected = []
    for split in pilot["source_splits"]:
        values = [
            value
            for value in frozen_selection
            if str(value["gaugeflow_split"]) == split
        ]
        indices = [
            min(len(values) - 1, int(round(index * (len(values) - 1) / max(args.per_split - 1, 1))))
            for index in range(args.per_split)
        ]
        selected.extend(values[index] for index in indices)
    source, missing = _load_source_rows(selected, args.source_root)
    if missing:
        raise ValueError(f"diagnostic source rows are missing: {missing}")
    records = [
        _process_one((record, source[str(record["material_id"])], config))
        for record in selected
    ]
    summary = {
        "protocol": f"{config['protocol']}_diagnostic_only",
        "scientific_status": "no_write_smoke_not_gate_evidence",
        "selected": len(records),
        "candidate_rows": sum(int(value["candidate_count"]) > 0 for value in records),
        "qualified_rows": sum(value["qualified_nontrivial"] is True for value in records),
        "processing_failures": sum(value["processing_failure"] is True for value in records),
        "records": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
