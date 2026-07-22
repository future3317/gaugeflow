"""Build the leakage-safe random-access MatPES physical-pretraining index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.production.matpes_index import build_matpes_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbe", type=Path, nargs="+", required=True)
    parser.add_argument("--r2scan", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-atoms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--calibration-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-rows-per-source", type=int)
    parser.add_argument("--minimum-lattice-width-angstrom", type=float, default=0.5)
    parser.add_argument("--maximum-lattice-metric-condition", type=float, default=1.0e4)
    arguments = parser.parse_args()
    manifest = build_matpes_index(
        {"PBE": arguments.pbe, "r2SCAN": arguments.r2scan},
        arguments.output,
        energy_target="cohesive_energy_per_atom",
        maximum_atoms=arguments.maximum_atoms,
        seed=arguments.seed,
        calibration_fraction=arguments.calibration_fraction,
        test_fraction=arguments.test_fraction,
        max_rows_per_source=arguments.max_rows_per_source,
        minimum_lattice_width_angstrom=arguments.minimum_lattice_width_angstrom,
        maximum_lattice_metric_condition=arguments.maximum_lattice_metric_condition,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
