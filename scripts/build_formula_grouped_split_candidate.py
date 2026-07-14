"""Build an inactive formula-disjoint split candidate without modifying v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def select_exact(groups, target: int, seed: int):
    order = list(groups)
    random.Random(seed).shuffle(order)
    previous: list[tuple[int, str] | None] = [None] * (target + 1)
    previous[0] = (-1, "")
    for formula, ids, _ in order:
        width = len(ids)
        if width > target:
            continue
        for total in range(target - width, -1, -1):
            if previous[total] is not None and previous[total + width] is None:
                previous[total + width] = (total, formula)
        if previous[target] is not None:
            break
    if previous[target] is None:
        raise RuntimeError(f"No exact formula-group subset sums to {target}")
    selected = set()
    total = target
    while total:
        prior, formula = previous[total]
        selected.add(formula)
        total = prior
    return selected


def stratum_counts(group_map, selected):
    counts = Counter()
    for formula in selected:
        counts.update(group_map[formula][1])
    return counts


def objective(group_map, assignments, global_counts, total_rows):
    score = 0.0
    for formulas, size in assignments:
        observed = stratum_counts(group_map, formulas)
        for stratum, global_count in global_counts.items():
            expected = global_count * size / total_rows
            score += ((observed[stratum] - expected) / max(expected, 1.0)) ** 2
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=Path("reports/data_quality_rows.csv"))
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_csv(args.rows)
    if len(frame) != 4998 or frame.material_id.duplicated().any():
        raise ValueError("Expected the full unique 4,998-row data-quality audit")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    gate_ids = set(map(str, protocol["material_ids"]))
    gate_formulas = set(frame.loc[frame.material_id.isin(gate_ids), "formula"])
    if len(frame.loc[frame.material_id.isin(gate_ids)]) != len(gate_ids):
        raise ValueError("Gate A IDs are missing from audited rows")

    groups = []
    group_map = {}
    for formula, group in frame.groupby("formula", sort=True):
        ids = sorted(map(str, group.material_id))
        strata = Counter(map(int, group.response_stratum))
        groups.append((formula, ids, strata))
        group_map[formula] = (ids, strata)
    eligible = [value for value in groups if value[0] not in gate_formulas]
    eligible_formulas = {value[0] for value in eligible}
    global_counts = Counter(map(int, frame.response_stratum))

    best = None
    for trial in range(args.trials):
        test = select_exact(eligible, 499, args.seed + 2 * trial)
        remaining = [value for value in eligible if value[0] not in test]
        val = select_exact(remaining, 499, args.seed + 2 * trial + 1)
        train = (eligible_formulas - test - val) | gate_formulas
        score = objective(
            group_map,
            ((train, 4000), (val, 499), (test, 499)),
            global_counts,
            len(frame),
        )
        candidate = (score, trial, train, val, test)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    score, trial, train_formulas, val_formulas, test_formulas = best

    splits = {}
    for name, formulas in (
        ("train", train_formulas), ("val", val_formulas), ("test", test_formulas)
    ):
        splits[name] = sorted(
            material_id for formula in formulas for material_id in group_map[formula][0]
        )
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 4000, "val": 499, "test": 499
    }
    assert gate_ids.issubset(splits["train"])
    assert train_formulas.isdisjoint(val_formulas)
    assert train_formulas.isdisjoint(test_formulas)
    assert val_formulas.isdisjoint(test_formulas)

    parent_bytes = args.parent_split.read_bytes()
    metadata = {
        "schema": 1,
        "status": "candidate_not_active",
        "name": "TensorOrbit-JARVIS formula-grouped candidate v2",
        "parent_split_path": str(args.parent_split),
        "parent_split_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "group_definition": "pymatgen Structure.composition.reduced_formula from source CIF",
        "seed": args.seed,
        "selected_trial": trial,
        "trials": args.trials,
        "stratum_balance_objective": score,
        "gate_a_ids_forced_to_train": sorted(gate_ids),
        "counts": {name: len(values) for name, values in splits.items()},
        "formula_group_counts": {
            "train": len(train_formulas), "val": len(val_formulas), "test": len(test_formulas)
        },
        "response_strata": {
            name: dict(sorted(stratum_counts(group_map, formulas).items()))
            for name, formulas in (
                ("train", train_formulas), ("val", val_formulas), ("test", test_formulas)
            )
        },
        "audit_rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
    }
    payload = {**splits, "_metadata": metadata}
    metadata["candidate_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
