"""Build a new IID E1 panel without reusing the archived calibration rows.

The source remains the qualified child-train cache.  Exact composition
partitions are stratified so every panel state has training support; small
partitions stay fit-only and therefore cannot create accidental zero-support
likelihood events.  This axis is independent of the frozen formula/prototype
novelty split.  Alex-MP-20 has no source timestamp, so the time axis is written
as explicitly unavailable rather than inferred from row order or material ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import (
    categorical_total_variation,
    load_compositions,
    partition_key,
)

LABELS = ("fit", "calibration", "test")


def _normalized_source_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _partition_stratified_labels(
    keys: torch.Tensor,
    *,
    seed: int,
    calibration_fraction: float,
    test_fraction: float,
    minimum_partition_for_panels: int,
    frequent_partition_threshold: int,
    frequent_partition_panel_floor: int,
) -> torch.Tensor:
    """Assign exact partitions with a deterministic random order within group."""
    if keys.ndim != 1 or keys.dtype != torch.long:
        raise ValueError("partition keys must be a rank-one int64 tensor")
    if calibration_fraction <= 0 or test_fraction <= 0 or calibration_fraction + test_fraction >= 1:
        raise ValueError("IID panel fractions must be positive and leave a fit split")
    labels = torch.zeros(keys.numel(), dtype=torch.int8)
    random_rank = torch.rand(keys.numel(), generator=torch.Generator().manual_seed(seed))
    order = torch.argsort(keys, stable=True)
    ordered_keys = keys.index_select(0, order)
    _, group_sizes = torch.unique_consecutive(ordered_keys, return_counts=True)
    offset = 0
    for size_tensor in group_sizes:
        size = int(size_tensor)
        group = order[offset : offset + size]
        offset += size
        if size < minimum_partition_for_panels:
            continue
        panel_floor = frequent_partition_panel_floor if size >= frequent_partition_threshold else 1
        calibration = max(panel_floor, round(calibration_fraction * size))
        test = max(panel_floor, round(test_fraction * size))
        if calibration + test >= size:
            raise RuntimeError("partition stratification would remove all fit support")
        shuffled = group[torch.argsort(random_rank.index_select(0, group), stable=True)]
        labels[shuffled[:calibration]] = 1
        labels[shuffled[calibration : calibration + test]] = 2
    if not all(bool((labels == value).any()) for value in range(3)):
        raise RuntimeError("IID split construction produced an empty label")
    return labels


def _element_graph_frequency(state: Any, selected: torch.Tensor) -> torch.Tensor:
    species = state.species.index_select(0, selected)
    length = state.length.index_select(0, selected)
    active = torch.arange(state.maximum_species).unsqueeze(0) < length.unsqueeze(1)
    return torch.bincount(species[active], minlength=118)


def _pair_graph_frequency(state: Any, selected: torch.Tensor) -> torch.Tensor:
    species = state.species.index_select(0, selected)
    length = state.length.index_select(0, selected)
    frequency = torch.zeros((118, 118), dtype=torch.long)
    for left in range(state.maximum_species):
        for right in range(left + 1, state.maximum_species):
            active = (left < length) & (right < length)
            if not bool(active.any()):
                continue
            first = species[active, left]
            second = species[active, right]
            low = torch.minimum(first, second)
            high = torch.maximum(first, second)
            flat = low * 118 + high
            frequency += torch.bincount(flat, minlength=118 * 118).reshape(118, 118)
    return frequency


def _profile(
    state: Any,
    keys: torch.Tensor,
    labels: torch.Tensor,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    quality = protocol["quality_floors"]
    all_index = torch.arange(state.graphs)
    element_total = _element_graph_frequency(state, all_index)
    pair_total = _pair_graph_frequency(state, all_index)
    eligible_element = element_total >= int(quality["eligible_element_graphs_min"])
    eligible_pair = pair_total >= int(quality["eligible_pair_graphs_min"])
    label_profile: dict[str, Any] = {}
    for value, name in enumerate(LABELS):
        index = torch.nonzero(labels == value, as_tuple=False).flatten()
        element = _element_graph_frequency(state, index)
        pair = _pair_graph_frequency(state, index)
        label_profile[name] = {
            "graphs": int(index.numel()),
            "partition_tv_from_fit": None,
            "minimum_eligible_element_graphs": int(element[eligible_element].min()),
            "minimum_eligible_pair_graphs": int(pair[eligible_pair].min()),
            "node_count_min": int(state.node_count[index].min()),
            "node_count_max": int(state.node_count[index].max()),
            "species_count_min": int(state.length[index].min()),
            "species_count_max": int(state.length[index].max()),
        }
    fit_keys = keys[labels == 0]
    for value, name in ((1, "calibration"), (2, "test")):
        label_profile[name]["partition_tv_from_fit"] = categorical_total_variation(
            fit_keys, keys[labels == value]
        )
    return {
        "labels": label_profile,
        "eligible_elements": int(eligible_element.sum()),
        "eligible_pairs": int(torch.triu(eligible_pair, diagonal=1).sum()),
        "element_floor_pass": all(
            label_profile[name]["minimum_eligible_element_graphs"]
            >= int(quality["panel_element_graphs_min"])
            for name in ("calibration", "test")
        ),
        "pair_floor_pass": all(
            label_profile[name]["minimum_eligible_pair_graphs"]
            >= int(quality["panel_pair_graphs_min"])
            for name in ("calibration", "test")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--novelty-assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_absolute_calibration_split_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen E1 split protocol")
    source = protocol["source"]
    if _normalized_source_sha256(Path(__file__)) != source["builder_sha256"]:
        raise ValueError("split builder does not match the preregistered implementation")
    expected = {
        args.cache_root / "manifest.json": source["cache_manifest_sha256"],
        args.cache_root / "train.pt": source["train_cache_sha256"],
        args.cache_root / "train_index.parquet": source["train_index_sha256"],
        args.novelty_assignment: source["novelty_assignment_sha256"],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"source identity changed: {path}")

    state = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=int(protocol["support"]["maximum_species"]),
        vocabulary_size=int(protocol["support"]["vocabulary_size"]),
    )
    index_table = pq.read_table(args.cache_root / "train_index.parquet")
    if index_table.num_rows != state.graphs or index_table.column_names.count("material_id") != 1:
        raise ValueError("cache index and composition rows do not align")
    cache_row = torch.as_tensor(index_table["cache_row"].to_numpy(), dtype=torch.long)
    if not torch.equal(cache_row, torch.arange(state.graphs)):
        raise ValueError("cache index is not in canonical row order")

    keys = partition_key(state)
    split = protocol["split"]
    labels = _partition_stratified_labels(
        keys,
        seed=int(split["seed"]),
        calibration_fraction=float(split["calibration_fraction"]),
        test_fraction=float(split["test_fraction"]),
        minimum_partition_for_panels=int(split["minimum_partition_for_panels"]),
        frequent_partition_threshold=int(split["frequent_partition_threshold"]),
        frequent_partition_panel_floor=int(split["frequent_partition_panel_floor"]),
    )
    profile = _profile(state, keys, labels, protocol)
    if not profile["element_floor_pass"] or not profile["pair_floor_pass"]:
        raise RuntimeError("preregistered IID panel element/pair floors are not met")

    args.output_root.mkdir(parents=True, exist_ok=False)
    material_id = index_table["material_id"].to_pylist()
    label_strings = [LABELS[int(value)] for value in labels]
    partition_frequency = torch.bincount(
        torch.unique(keys, sorted=True, return_inverse=True)[1]
    )
    _, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    row_partition_frequency = partition_frequency.index_select(0, inverse)
    frequency_tier = torch.bucketize(
        row_partition_frequency,
        torch.tensor([20, 100, 1000]),
        right=False,
    )
    assignment_path = args.output_root / "iid_assignment.parquet"
    pq.write_table(
        pa.table(
            {
                "material_id": material_id,
                "cache_row": cache_row.numpy(),
                "split_label": label_strings,
                "node_count": state.node_count.numpy(),
                "species_count": state.length.numpy(),
                "partition_key": keys.numpy(),
                "partition_fit_frequency_tier": frequency_tier.numpy(),
            }
        ),
        assignment_path,
        compression="zstd",
        version="2.6",
    )
    for value, name in enumerate(LABELS):
        torch.save(
            torch.nonzero(labels == value, as_tuple=False).flatten(),
            args.output_root / f"{name}_index.pt",
        )
    manifest = {
        "schema": 1,
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "dataset": "Alex-MP-20 qualified child-train cache",
        "rows": state.graphs,
        "source": source,
        "iid_axis": {
            "scientific_role": "probability_calibration",
            "assignment_path": assignment_path.name,
            "assignment_sha256": sha256_file(assignment_path),
            "labels": list(LABELS),
            "split_rule": split,
            "profile": profile,
        },
        "novelty_axis": {
            "scientific_role": "OOD novelty and coverage only",
            "assignment_path": str(args.novelty_assignment),
            "assignment_sha256": sha256_file(args.novelty_assignment),
            "labels": ["train", "val", "test"],
        },
        "time_axis": {
            "status": "unavailable",
            "reason": "Alex-MP-20 source schema has no auditable record/release timestamp",
            "prohibited_surrogates": ["material_id lexical order", "source row", "source split"],
        },
        "index_sha256": {
            name: sha256_file(args.output_root / f"{name}_index.pt") for name in LABELS
        },
        "builder_sha256": sha256_file(Path(__file__)),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
