"""Read-only closure audit for MatPES physical-pretraining records."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from gaugeflow.production.matpes_data import parse_matpes_row


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "mean": None, "maximum": None}
    return {
        "minimum": min(values),
        "mean": math.fsum(values) / len(values),
        "maximum": max(values),
    }


def audit_file(path: Path, max_rows: int) -> dict[str, Any]:
    """Audit at most ``max_rows`` rows without mutating or materializing the dataset."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    failures: Counter[str] = Counter()
    functionals: Counter[str] = Counter()
    atom_counts: list[float] = []
    volumes: list[float] = []
    energies: list[float] = []
    force_rms: list[float] = []
    stress_norms: list[float] = []
    label_counts = Counter[str]()
    rows_seen = 0
    valid_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if rows_seen >= max_rows:
                break
            rows_seen += 1
            try:
                row = json.loads(line)
                record = parse_matpes_row(row)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                failures[f"{type(error).__name__}: {error}"] += 1
                continue
            valid_rows += 1
            functionals[record.functional] += 1
            atom_counts.append(float(record.element_tokens.numel()))
            volumes.append(float(torch.linalg.det(record.lattice).item()))
            if record.energy_present:
                label_counts["energy"] += 1
                energies.append(float(record.energy_per_atom_ev.item()))
            if record.forces_present:
                label_counts["forces"] += 1
                force_rms.append(float(record.forces_ev_per_angstrom.square().mean().sqrt().item()))
            if record.stress_present:
                label_counts["stress"] += 1
                stress_norms.append(float(torch.linalg.vector_norm(record.stress_kelvin_gpa).item()))
    return {
        "path": str(path.resolve()),
        "rows_seen": rows_seen,
        "valid_rows": valid_rows,
        "invalid_rows": rows_seen - valid_rows,
        "failure_reasons": dict(sorted(failures.items())),
        "functionals": dict(sorted(functionals.items())),
        "label_presence": {
            label: {"count": label_counts[label], "fraction": label_counts[label] / max(valid_rows, 1)}
            for label in ("energy", "forces", "stress")
        },
        "atom_count": _summary(atom_counts),
        "cell_volume_angstrom3": _summary(volumes),
        "energy_per_atom_ev": _summary(energies),
        "force_rms_ev_per_angstrom": _summary(force_rms),
        "stress_kelvin_norm_gpa": _summary(stress_norms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = {
        "schema": "gaugeflow.matpes_physical_data_audit.v1",
        "max_rows_per_file": arguments.max_rows,
        "datasets": [audit_file(path, arguments.max_rows) for path in arguments.paths],
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
