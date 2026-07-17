"""Shared frozen-panel and provenance I/O for parent-occurrence protocols."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow.dataset as pds
import pyarrow.parquet as pq

from gaugeflow.catalogue.parent_decomposition import balanced_selection


def ordered_material_id_hash(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def clean_git_commit(repo_root: Path, *, protocol: str) -> str:
    """Return HEAD only when tracked, staged and untracked content is clean."""
    tracked = subprocess.run(
        ["git", "diff", "--ignore-space-at-eol", "--quiet"],
        cwd=repo_root,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked.returncode != 0 or staged.returncode != 0 or untracked.strip():
        raise RuntimeError(f"the frozen {protocol} run requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def frozen_e1a_selection(
    config: dict[str, object], data_root: Path
) -> list[dict[str, object]]:
    """Rebuild and verify the exact ordered E1a zero-candidate panel."""
    decompositions = pq.read_table(
        data_root
        / "processed/gaugeflow_h0_v4/parent_decomposition_pilot_v1/decompositions.parquet"
    ).to_pylist()
    eligible = [
        row
        for row in decompositions
        if int(row["candidate_count"]) == 0 and not bool(row["processing_failure"])
    ]
    selection = config["selection"]
    if not isinstance(selection, dict):
        raise ValueError("E1a selection must be a mapping")
    split_counts = selection["split_counts"]
    if not isinstance(split_counts, dict):
        raise ValueError("E1a split counts must be a mapping")
    reproduced = list(
        balanced_selection(
            eligible,
            split_counts={str(key): int(value) for key, value in split_counts.items()},
            seed=int(selection["seed"]),
            site_boundaries=selection["site_bins"],
        )
    )
    observed_ids = [str(row["material_id"]) for row in reproduced]
    if observed_ids != list(selection["material_ids"]):
        raise ValueError("frozen E1a material IDs do not reproduce from v1")
    if ordered_material_id_hash(observed_ids) != selection["ordered_material_ids_sha256"]:
        raise ValueError("frozen E1a ordered material-ID hash does not match")
    return reproduced


def join_alex_rows(
    selection: list[dict[str, object]], data_root: Path
) -> dict[str, dict[str, object]]:
    """Join one frozen material panel to immutable Alex source rows."""
    requested = {str(row["material_id"]) for row in selection}
    observed: dict[str, dict[str, object]] = {}
    columns = ["material_id", "positions", "cell", "atomic_numbers"]
    for split in ("train", "val", "test"):
        path = data_root / f"raw/huggingface/Alex-MP-20/{split}.parquet"
        table = pds.dataset(path, format="parquet").to_table(
            filter=pds.field("material_id").isin(requested), columns=columns
        )
        for row in table.to_pylist():
            material_id = str(row["material_id"])
            if material_id in observed:
                raise ValueError("Alex material ID occurs in more than one source split")
            row["source_split_observed"] = split
            observed[material_id] = row
    if set(observed) != requested:
        raise ValueError("not every frozen material joined to Alex")
    return observed
