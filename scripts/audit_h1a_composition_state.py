"""Audit sparse composition-state complexity on the qualified H1a cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _split_summary(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    tokens = value["atom_tokens"].long()
    offsets = value["offsets"].long()
    graph_count = int(offsets.numel() - 1)
    node_counts = offsets.diff()
    batch = torch.repeat_interleave(torch.arange(graph_count), node_counts)
    flat = batch * CHEMICAL_ELEMENT_COUNT + tokens
    unique_pair, pair_count = torch.unique(flat, sorted=False, return_counts=True)
    pair_graph = torch.div(unique_pair, CHEMICAL_ELEMENT_COUNT, rounding_mode="floor")
    distinct_species = torch.bincount(pair_graph, minlength=graph_count)
    active_tokens = torch.unique(unique_pair.remainder(CHEMICAL_ELEMENT_COUNT))
    species_histogram = torch.bincount(
        distinct_species,
        minlength=int(distinct_species.max()) + 1,
    )
    multiplicity_histogram = torch.bincount(pair_count, minlength=int(pair_count.max()) + 1)

    return {
        "graphs": graph_count,
        "nodes": int(tokens.numel()),
        "node_count_mean": float(node_counts.float().mean()),
        "node_count_max": int(node_counts.max()),
        "active_element_count": int(active_tokens.numel()),
        "active_element_tokens": [int(value) for value in active_tokens.tolist()],
        "distinct_species_mean": float(distinct_species.float().mean()),
        "distinct_species_max": int(distinct_species.max()),
        "distinct_species_histogram": {
            str(index): int(count)
            for index, count in enumerate(species_histogram.tolist())
            if count
        },
        "fraction_with_at_most_species": {
            str(limit): float((distinct_species <= limit).float().mean())
            for limit in (1, 2, 3, 4, 5, 6)
        },
        "species_multiplicity_histogram": {
            str(index): int(count)
            for index, count in enumerate(multiplicity_histogram.tolist())
            if count
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_explicit_composition_state_data_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen composition-state audit protocol")
    manifest = args.cache_root / "manifest.json"
    manifest_sha256 = sha256_file(manifest)
    if manifest_sha256 != protocol["source_cache_manifest_sha256"]:
        raise ValueError("composition-state audit cache manifest mismatch")
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "cache_manifest_sha256": manifest_sha256,
        "splits": {
            split: _split_summary(args.cache_root / f"{split}.pt")
            for split in ("train", "val", "test")
        },
        "decision_rule": {
            "sparse_exact_state_practical": (
                "train fraction with at most four species >= 0.95 "
                "and maximum species <= 10"
            ),
            "boundary": "data qualification only; does not authorize E1 training or later Gates",
        },
    }
    train = result["splits"]["train"]
    acceptance = protocol["acceptance"]
    result["sparse_exact_state_practical"] = bool(
        train["fraction_with_at_most_species"]["4"]
        >= acceptance["train_fraction_with_at_most_four_species_min"]
        and train["distinct_species_max"]
        <= acceptance["train_maximum_distinct_species_max"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
