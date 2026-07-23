from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.production.generated_state_replay import (
    GeneratedCarrierRole,
    GeneratedStateReplayEntry,
    GeneratedStateReplayKey,
    GeneratedStateReplayManifest,
    load_generated_state_replay_manifest,
    validate_no_forbidden_source_ids,
    write_generated_state_replay_manifest,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT

MASK_TOKEN = CHEMICAL_ELEMENT_COUNT
BASE_SHA = "synthetic-base-checkpoint-sha"
SAMPLER_COMMIT = "synthetic-sampler-commit"
PROTOCOL_SHA = "synthetic-sampler-protocol-sha"


def _counts() -> torch.Tensor:
    counts = torch.zeros((2, CHEMICAL_ELEMENT_COUNT), dtype=torch.long)
    counts[0, 4] = 1
    counts[0, 6] = 1
    counts[1, 6] = 1
    counts[1, 12] = 1
    counts[1, 15] = 1
    return counts


def _key(role: GeneratedCarrierRole, refresh_id: int) -> GeneratedStateReplayKey:
    return GeneratedStateReplayKey(
        source_structure_id=f"synthetic<{role}>",
        role=role,
        base_checkpoint_sha256=BASE_SHA,
        sampler_commit=SAMPLER_COMMIT,
        sampler_protocol_sha256=PROTOCOL_SHA,
        refresh_id=refresh_id,
        seed=5705,
        coordinate_time=0.35,
        element_time=0.45,
        lattice_time=0.55,
    )


def _lattice(role: GeneratedCarrierRole) -> torch.Tensor:
    offset = {
        "clean_clean": 0.0,
        "generated_assignment": 0.1,
        "generated_lattice": 0.2,
        "generated_joint": 0.3,
    }[role]
    return torch.stack(
        (
            torch.diag(torch.tensor([3.0 + offset, 3.2 + offset, 3.4 + offset])),
            torch.diag(torch.tensor([3.5 + offset, 3.8 + offset, 4.1 + offset])),
        )
    )


def _entry(role: GeneratedCarrierRole, refresh_id: int) -> GeneratedStateReplayEntry:
    clean_tokens = torch.tensor([4, 6, 6, 12, 15], dtype=torch.long)
    generated_tokens = torch.tensor([6, 4, 12, 15, 6], dtype=torch.long)
    partial_tokens = torch.tensor([4, MASK_TOKEN, MASK_TOKEN, 6, 15], dtype=torch.long)
    full_reveal = torch.tensor([2, 3], dtype=torch.long)
    partial_reveal = torch.tensor([1, 2], dtype=torch.long)

    assignment_tokens = clean_tokens
    assignment_source = "clean"
    assignment_reveal_count = full_reveal
    lattice_source = "clean"
    coordinate_source = "clean"

    if role == "generated_assignment":
        assignment_tokens = generated_tokens
        assignment_source = "generated_assignment"
    elif role == "generated_lattice":
        lattice_source = "generated_lattice"
    elif role == "generated_joint":
        assignment_tokens = partial_tokens
        assignment_source = "generated_joint"
        assignment_reveal_count = partial_reveal
        lattice_source = "generated_joint"
        coordinate_source = "generated_joint"

    return GeneratedStateReplayEntry(
        key=_key(role, refresh_id),
        source_split="train",
        parent_or_flexible_carrier_id="synthetic-flexible-p1",
        node_count=torch.tensor([2, 3], dtype=torch.long),
        composition_counts=_counts(),
        composition_source="clean",
        assignment_tokens=assignment_tokens,
        assignment_source=assignment_source,  # type: ignore[arg-type]
        assignment_reveal_rank=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
        assignment_reveal_count=assignment_reveal_count,
        lattice=_lattice(role),
        lattice_source=lattice_source,  # type: ignore[arg-type]
        lattice_log_volume=torch.tensor([3.0 + 0.01 * refresh_id, 4.0 + 0.01 * refresh_id]),
        lattice_log_shape=torch.zeros((2, 6)),
        fractional_coordinates=torch.tensor(
            [
                [0.05, 0.10, 0.15],
                [0.35, 0.25, 0.70],
                [0.15, 0.75, 0.45],
                [0.72, 0.55, 0.20],
                [0.42, 0.31, 0.82],
            ]
        ),
        coordinate_source=coordinate_source,  # type: ignore[arg-type]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the small synthetic manifest and report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roles: tuple[GeneratedCarrierRole, ...] = (
        "clean_clean",
        "generated_assignment",
        "generated_lattice",
        "generated_joint",
    )
    entries = [_entry(role, refresh_id=index) for index, role in enumerate(roles)]
    for entry in entries:
        entry.validate(
            expected_base_checkpoint_sha256=BASE_SHA,
            expected_sampler_commit=SAMPLER_COMMIT,
            expected_sampler_protocol_sha256=PROTOCOL_SHA,
        )
    validate_no_forbidden_source_ids(entries, {"synthetic<held-out>"})

    manifest = GeneratedStateReplayManifest.from_entries(entries)
    manifest.validate(
        expected_base_checkpoint_sha256=BASE_SHA,
        expected_sampler_commit=SAMPLER_COMMIT,
        expected_sampler_protocol_sha256=PROTOCOL_SHA,
        forbidden_source_ids={"synthetic<held-out>"},
    )
    manifest.validate_against_entries(entries)

    manifest_path = output_dir / "generated_state_replay_manifest.json"
    report_path = output_dir / "generated_state_replay_smoke_report.json"
    manifest_hash = write_generated_state_replay_manifest(manifest_path, manifest)
    loaded = load_generated_state_replay_manifest(manifest_path)
    loaded.validate_against_entries(entries)

    forbidden_overlap_rejected = False
    try:
        loaded.validate(forbidden_source_ids={"synthetic<generated_joint>"})
    except ValueError:
        forbidden_overlap_rejected = True

    report = {
        "status": "passed",
        "entry_count": len(entries),
        "roles": list(roles),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "round_trip_sha256": loaded.canonical_sha256(),
        "round_trip_matches": loaded.canonical_sha256() == manifest_hash,
        "forbidden_overlap_rejected": forbidden_overlap_rejected,
    }
    if not report["round_trip_matches"] or not forbidden_overlap_rejected:
        raise RuntimeError("generated-state replay manifest smoke failed")

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
