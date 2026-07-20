"""Independently audit the hash-bound E1 IID calibration split artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.composition_metrics import load_compositions, partition_key
from scripts.build_h1a_e1_absolute_calibration_split import (
    LABELS,
    _partition_stratified_labels,
    _profile,
)


def _column(table: Any, name: str) -> list[Any]:
    if table.column_names.count(name) != 1:
        raise ValueError(f"split assignment must contain exactly one {name} column")
    return table[name].to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--novelty-assignment", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_e1_absolute_calibration_split_v2":
        raise ValueError("unexpected E1 calibration split protocol")
    manifest_path = args.split_root / "manifest.json"
    manifest = load_json_object(manifest_path)
    if manifest.get("protocol") != protocol["protocol"]:
        raise ValueError("split manifest uses a different protocol")
    if manifest.get("protocol_sha256") != canonical_json_hash(protocol):
        raise ValueError("split manifest protocol hash mismatch")
    if manifest.get("source") != protocol["source"]:
        raise ValueError("split manifest source identity mismatch")

    assignment_path = args.split_root / manifest["iid_axis"]["assignment_path"]
    artifact_hashes = {
        "manifest": sha256_file(manifest_path),
        "assignment": sha256_file(assignment_path),
        **{
            f"{name}_index": sha256_file(args.split_root / f"{name}_index.pt")
            for name in LABELS
        },
    }
    if artifact_hashes["assignment"] != manifest["iid_axis"]["assignment_sha256"]:
        raise ValueError("IID assignment hash mismatch")
    for name in LABELS:
        if artifact_hashes[f"{name}_index"] != manifest["index_sha256"][name]:
            raise ValueError(f"{name} index hash mismatch")
    if sha256_file(args.novelty_assignment) != manifest["novelty_axis"]["assignment_sha256"]:
        raise ValueError("novelty-axis identity changed")

    table = pq.read_table(assignment_path)
    cache_table = pq.read_table(args.cache_root / "train_index.parquet")
    rows = int(protocol["support"]["graphs"])
    if table.num_rows != rows or cache_table.num_rows != rows or manifest["rows"] != rows:
        raise ValueError("split, cache index, and protocol row grains disagree")
    material_id = list(map(str, _column(table, "material_id")))
    cache_material_id = list(map(str, _column(cache_table, "material_id")))
    if material_id != cache_material_id or len(set(material_id)) != rows:
        raise ValueError("material ID join is not one-to-one in canonical cache order")
    cache_row = torch.as_tensor(_column(table, "cache_row"), dtype=torch.long)
    if not torch.equal(cache_row, torch.arange(rows)):
        raise ValueError("split assignment cache_row is not a canonical primary key")

    state = load_compositions(
        args.cache_root / "train.pt",
        maximum_species=int(protocol["support"]["maximum_species"]),
        vocabulary_size=int(protocol["support"]["vocabulary_size"]),
    )
    keys = partition_key(state)
    split = protocol["split"]
    recomputed = _partition_stratified_labels(
        keys,
        seed=int(split["seed"]),
        calibration_fraction=float(split["calibration_fraction"]),
        test_fraction=float(split["test_fraction"]),
        minimum_partition_for_panels=int(split["minimum_partition_for_panels"]),
        frequent_partition_threshold=int(split["frequent_partition_threshold"]),
        frequent_partition_panel_floor=int(split["frequent_partition_panel_floor"]),
    )
    label_lookup = {name: value for value, name in enumerate(LABELS)}
    labels = torch.tensor(
        [label_lookup[str(value)] for value in _column(table, "split_label")],
        dtype=torch.int8,
    )
    if not torch.equal(labels, recomputed):
        raise ValueError("stored IID rows differ from the preregistered random assignment")
    if not torch.equal(torch.as_tensor(_column(table, "partition_key")), keys):
        raise ValueError("stored partition keys do not match the composition cache")

    index_checks: dict[str, bool] = {}
    label_counts: dict[str, int] = {}
    for value, name in enumerate(LABELS):
        expected = torch.nonzero(labels == value, as_tuple=False).flatten()
        actual = torch.load(
            args.split_root / f"{name}_index.pt", map_location="cpu", weights_only=True
        ).long()
        index_checks[name] = torch.equal(actual, expected)
        label_counts[name] = int(actual.numel())
    if not all(index_checks.values()) or sum(label_counts.values()) != rows:
        raise ValueError("saved split indices do not form the exact assignment partition")

    profile = _profile(state, keys, labels, protocol)
    if profile != manifest["iid_axis"]["profile"]:
        raise ValueError("recomputed split profile differs from the manifest")
    checks = {
        "canonical_material_id_join": material_id == cache_material_id,
        "material_id_unique": len(set(material_id)) == rows,
        "all_rows_assigned_once": sum(label_counts.values()) == rows,
        "stored_indices_match_assignment": all(index_checks.values()),
        "deterministic_assignment_replay": torch.equal(labels, recomputed),
        "fit_support_for_every_panel_partition": profile[
            "fit_support_for_every_panel_partition"
        ],
        "element_floor": profile["element_floor_pass"],
        "pair_identity_presence": profile["pair_presence_pass"],
        "novelty_axis_preserved": sha256_file(args.novelty_assignment)
        == protocol["source"]["novelty_assignment_sha256"],
        "time_axis_fail_closed": manifest["time_axis"]["status"] == "unavailable",
    }
    qualified = all(checks.values())
    result = {
        "protocol": "h1a_e1_absolute_calibration_split_v2_independent_audit",
        "source_protocol_sha256": canonical_json_hash(protocol),
        "artifact_hashes": artifact_hashes,
        "label_counts": label_counts,
        "profile": profile,
        "checks": checks,
        "qualified": qualified,
        "decision": (
            "freeze_manifest_and_preregister_absolute_likelihood_e1"
            if qualified
            else "repair_split_contract_and_do_not_train_e1"
        ),
        "boundary": (
            "This audit cannot pass E1 or authorize assignment, L1, M1, tensor/oracle work, "
            "relaxation, DFT, or DFPT."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
