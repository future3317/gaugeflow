"""Bounded, no-write-by-default performance diagnostic for H0-D-v2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from gaugeflow.catalogue import standard_hall_numbers
from scripts.build_h0_d_opd_catalogue_v2 import (
    _build_parent,
    _spgrep_modulation_reference_agreement,
)

DEFAULT_PANEL = (1, 2, 62, 123, 194, 221, 225)


def diagnose(space_groups: tuple[int, ...]) -> dict[str, Any]:
    hall_numbers = standard_hall_numbers()
    if not space_groups or any(value not in hall_numbers for value in space_groups):
        raise ValueError("diagnostic space groups must lie in 1..230")
    started = time.perf_counter()
    records = []
    for space_group in space_groups:
        parent_started = time.perf_counter()
        parent = _build_parent((space_group, hall_numbers[space_group]))
        quotient_records = parent["records"]
        records.append(
            {
                "space_group": space_group,
                "elapsed_seconds": time.perf_counter() - parent_started,
                "supercell_orbits": parent["supercell_orbit_count"],
                "unique_cayley_tables": len(
                    {value["cayley_table_sha256"] for value in quotient_records}
                ),
                "real_irreps": sum(len(value["irreps"]) for value in quotient_records),
                "opd_classes": sum(
                    len(irrep["opd_classes"])
                    for value in quotient_records
                    for irrep in value["irreps"]
                ),
                "maximum_quotient_order": max(
                    value["quotient_order"] for value in quotient_records
                ),
            }
        )
    return {
        "protocol": "h0_d_opd_physical_path_catalogue_v2_bounded_diagnostic",
        "scientific_status": "diagnostic_only_not_a_gate_result",
        "space_groups": list(space_groups),
        "records": records,
        "spgrep_modulation_reference": _spgrep_modulation_reference_agreement(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-groups", type=int, nargs="+", default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(tuple(args.space_groups))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
