"""Freeze Alex validation/test IDs for LeMat overlap removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaugeflow.production.benchmark_exclusion import build_alex_benchmark_exclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alex-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_alex_benchmark_exclusion(args.alex_cache, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
