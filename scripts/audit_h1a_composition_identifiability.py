"""Audit whether species-free E1 context uniquely identifies exact composition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _parse_reduced_formula(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split("|"):
        atomic_number, count = item.split(":", maxsplit=1)
        pairs.append((int(atomic_number) - 1, int(count)))
    if not pairs or any(token < 0 or token >= CHEMICAL_ELEMENT_COUNT or count < 1 for token, count in pairs):
        raise ValueError(f"invalid reduced formula key: {value}")
    if [token for token, _ in pairs] != sorted(token for token, _ in pairs):
        raise ValueError(f"formula tokens are not canonical: {value}")
    return pairs


def _full_formula_key(reduced: str, node_count: int) -> str:
    pairs = _parse_reduced_formula(reduced)
    reduced_count = sum(count for _, count in pairs)
    if node_count % reduced_count:
        raise ValueError(
            f"node count {node_count} is incompatible with reduced formula {reduced}"
        )
    scale = node_count // reduced_count
    return "|".join(f"{token}:{count * scale}" for token, count in pairs)


def _tokens_formula_key(tokens: torch.Tensor) -> str:
    counts = torch.bincount(tokens.long(), minlength=CHEMICAL_ELEMENT_COUNT)
    active = torch.nonzero(counts, as_tuple=False).flatten().tolist()
    return "|".join(f"{token}:{int(counts[token])}" for token in active)


def _conditional_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    joint = frame.groupby(columns + ["composition_key"], sort=False, observed=True).size()
    levels = list(range(len(columns)))
    totals = joint.groupby(level=levels, sort=False).sum()
    maxima = joint.groupby(level=levels, sort=False).max()
    distinct = joint.groupby(level=levels, sort=False).size()
    conditional_probability = joint / joint.groupby(level=levels, sort=False).transform("sum")
    conditional_entropy = float(
        (-(joint / len(frame)) * conditional_probability.map(math.log)).sum()
    )
    ambiguous = distinct > 1
    repeated = totals > 1
    return {
        "columns": columns,
        "contexts": int(totals.shape[0]),
        "bayes_exact_composition_accuracy": float(maxima.sum() / len(frame)),
        "conditional_entropy_nats": conditional_entropy,
        "ambiguous_context_fraction": float(ambiguous.mean()),
        "ambiguous_sample_fraction": float(totals[ambiguous].sum() / len(frame)),
        "repeated_context_sample_fraction": float(totals[repeated].sum() / len(frame)),
        "maximum_distinct_compositions": int(distinct.max()),
        "mean_distinct_compositions_in_ambiguous_context": (
            float(distinct[ambiguous].mean()) if bool(ambiguous.any()) else 1.0
        ),
    }


def _context_seen_fraction(
    train: pd.DataFrame,
    split: pd.DataFrame,
    columns: list[str],
) -> float:
    train_keys = pd.MultiIndex.from_frame(train[columns].drop_duplicates())
    split_keys = pd.MultiIndex.from_frame(split[columns])
    return float(split_keys.isin(train_keys).mean())


def _load_split(
    cache_root: Path,
    assignments: pd.DataFrame,
    split: str,
    verification_sample_size: int,
    verification_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    packed = torch.load(cache_root / f"{split}.pt", map_location="cpu", weights_only=True)
    offsets = packed["offsets"].long()
    index = pd.read_parquet(
        cache_root / f"{split}_index.parquet",
        columns=["material_id", "source_split", "gaugeflow_split", "cache_row"],
    )
    if len(index) != offsets.numel() - 1 or not index["cache_row"].is_unique:
        raise ValueError(f"{split} cache index is not a one-to-one row map")
    index = index.sort_values("cache_row", ignore_index=True)
    expected_rows = torch.arange(len(index)).numpy()
    if not (index["cache_row"].to_numpy() == expected_rows).all():
        raise ValueError(f"{split} cache rows are not contiguous")
    index["node_count"] = offsets.diff().numpy()
    metadata = assignments[assignments["gaugeflow_split"] == split]
    joined = index.merge(
        metadata,
        on=["material_id", "source_split", "gaugeflow_split"],
        how="left",
        validate="one_to_one",
    )
    if joined["reduced_formula_key"].isna().any():
        raise ValueError(f"{split} cache contains rows missing from assignment metadata")
    joined["composition_key"] = [
        _full_formula_key(formula, int(nodes))
        for formula, nodes in zip(
            joined["reduced_formula_key"], joined["node_count"], strict=True
        )
    ]

    generator = torch.Generator().manual_seed(verification_seed)
    sample_count = min(verification_sample_size, len(joined))
    selected = torch.randperm(len(joined), generator=generator)[:sample_count].tolist()
    atom_tokens = packed["atom_tokens"]
    mismatch = 0
    for row in selected:
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        mismatch += int(
            _tokens_formula_key(atom_tokens[start:stop])
            != joined.iloc[row]["composition_key"]
        )
    verification = {
        "sample_size": sample_count,
        "mismatches": mismatch,
        "passed": mismatch == 0,
    }
    if mismatch:
        raise ValueError(f"{split} reduced/full formula reconstruction mismatch")
    return joined, verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_json_object(args.protocol)
    if (
        protocol.get("protocol") != "h1a_e1_composition_identifiability_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen composition-identifiability protocol")
    cache_manifest = args.cache_root / "manifest.json"
    if sha256_file(cache_manifest) != protocol["source_cache_manifest_sha256"]:
        raise ValueError("composition-identifiability cache manifest mismatch")
    if sha256_file(args.assignment) != protocol["source_assignment_sha256"]:
        raise ValueError("composition-identifiability assignment hash mismatch")

    assignment_columns = [
        "material_id",
        "source_split",
        "gaugeflow_split",
        "reduced_formula_key",
        "prototype_key",
        "matcher_envelope_key",
        "anonymous_stoichiometry",
        "space_group_number",
        "primitive_sites",
    ]
    assignments = pd.read_parquet(args.assignment, columns=assignment_columns)
    if assignments.duplicated(["material_id", "source_split"]).any():
        raise ValueError("assignment metadata has duplicate material/source identifiers")

    frames: dict[str, pd.DataFrame] = {}
    verification: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        frames[split], verification[split] = _load_split(
            args.cache_root,
            assignments,
            split,
            int(protocol["formula_verification_sample_size_per_split"]),
            int(protocol["formula_verification_seed"]) + split_index,
        )

    contexts = {name: list(columns) for name, columns in protocol["contexts"].items()}
    summaries = {
        split: {
            name: _conditional_summary(frame, columns)
            for name, columns in contexts.items()
        }
        for split, frame in frames.items()
    }
    train = frames["train"]
    cross_split: dict[str, Any] = {}
    train_formulas = set(train["composition_key"])
    for split in ("val", "test"):
        frame = frames[split]
        cross_split[split] = {
            "exact_composition_seen_in_train_fraction": float(
                frame["composition_key"].isin(train_formulas).mean()
            ),
            "context_seen_in_train_fraction": {
                name: _context_seen_fraction(train, frame, columns)
                for name, columns in contexts.items()
            },
        }

    threshold = float(
        protocol["decision_rule"]["anonymous_geometry_ambiguous_sample_fraction_min"]
    )
    train_summary = summaries["train"]
    ambiguity = max(
        train_summary["anonymous_prototype_audit_oracle"]["ambiguous_sample_fraction"],
        train_summary["matcher_envelope_audit_oracle"]["ambiguous_sample_fraction"],
    )
    ambiguous = ambiguity >= threshold
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "assignment_sha256": sha256_file(args.assignment),
        "formula_reconstruction_verification": verification,
        "splits": {
            split: {
                "graphs": len(frame),
                "exact_compositions": int(frame["composition_key"].nunique()),
                "marginal_composition_entropy_nats": float(
                    -(frame["composition_key"].value_counts(normalize=True).map(math.log)
                      * frame["composition_key"].value_counts(normalize=True)).sum()
                ),
                "contexts": summaries[split],
            }
            for split, frame in frames.items()
        },
        "cross_split": cross_split,
        "maximum_anonymous_geometry_ambiguous_sample_fraction": ambiguity,
        "paired_exact_formula_is_not_uniquely_identified": ambiguous,
        "decision": (
            "use_explicit_stochastic_composition_and_treat_paired_exact_recovery_as_diagnostic"
            if ambiguous
            else "use_explicit_stochastic_composition_but_retain_paired_recovery_as_a_secondary_gate"
        ),
        "leakage_boundary": protocol["leakage_boundary"],
        "boundary": protocol["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
