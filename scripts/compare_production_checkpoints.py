"""Compare two production checkpoints for exact tensor/state equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.production.checkpointing import read_production_checkpoint_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _compare(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            mismatches.append(f"{path}: tensor schema")
        elif not torch.equal(left, right):
            mismatches.append(f"{path}: tensor value")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            mismatches.append(f"{path}: mapping keys")
            return
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", mismatches)
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            mismatches.append(f"{path}: sequence schema")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]", mismatches)
        return
    if type(left) is not type(right) or left != right:
        mismatches.append(f"{path}: scalar value")


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint comparison: {args.output}")
    left_metadata = read_production_checkpoint_metadata(args.left)
    right_metadata = read_production_checkpoint_metadata(args.right)
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    mismatches: list[str] = []
    _compare(left_metadata, right_metadata, "metadata", mismatches)
    _compare(left, right, "checkpoint", mismatches)
    result = {
        "protocol": "production_checkpoint_exact_resume_comparison_v1",
        "exact": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "left": str(args.left),
        "right": str(args.right),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if mismatches:
        raise RuntimeError("production checkpoint resume is not exact")


if __name__ == "__main__":
    main()
