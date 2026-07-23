from __future__ import annotations

import pytest
import torch

from gaugeflow.production.generated_state_replay import (
    GeneratedStateReplayEntry,
    GeneratedStateReplayKey,
    GeneratedStateReplayManifest,
    GeneratedStateReplayManifestRow,
    load_generated_state_replay_cache,
    load_generated_state_replay_manifest,
    validate_no_forbidden_source_ids,
    write_generated_state_replay_cache,
    write_generated_state_replay_manifest,
)
from scripts.train_generated_state_replay_correctness import _parameter_update_norm, _role_weight


def _key(role: str = "generated_joint") -> GeneratedStateReplayKey:
    return GeneratedStateReplayKey(
        source_structure_id="alex<unit-test>",
        role=role,  # type: ignore[arg-type]
        base_checkpoint_sha256="base-sha",
        sampler_commit="commit-sha",
        sampler_protocol_sha256="protocol-sha",
        refresh_id=3,
        seed=123,
        coordinate_time=0.4,
        element_time=0.5,
        lattice_time=0.6,
    )


def _counts() -> torch.Tensor:
    counts = torch.zeros((2, 118), dtype=torch.long)
    counts[0, 4] = 1
    counts[0, 6] = 1
    counts[1, 6] = 1
    counts[1, 12] = 1
    counts[1, 15] = 1
    return counts


def _entry(*, role: str = "generated_joint") -> GeneratedStateReplayEntry:
    return GeneratedStateReplayEntry(
        key=_key(role),
        source_split="train",
        parent_or_flexible_carrier_id="flexible-p1",
        node_count=torch.tensor([2, 3], dtype=torch.long),
        composition_counts=_counts(),
        composition_source="clean",
        assignment_tokens=torch.tensor([4, 118, 118, 6, 15], dtype=torch.long),
        assignment_source="generated_assignment",
        assignment_reveal_rank=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
        assignment_reveal_count=torch.tensor([1, 2], dtype=torch.long),
        lattice=torch.stack((3.0 * torch.eye(3), torch.diag(torch.tensor([3.5, 4.0, 4.5])))),
        lattice_source="generated_lattice",
        lattice_log_volume=torch.tensor([3.0, 4.0]),
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
        coordinate_source="generated_joint",
    )


def test_generated_state_replay_accepts_orderless_partial_joint_entry() -> None:
    entry = _entry()
    projector = torch.eye(6).expand(2, -1, -1)
    entry.validate(
        shape_projector=projector,
        expected_base_checkpoint_sha256="base-sha",
        expected_sampler_commit="commit-sha",
        expected_sampler_protocol_sha256="protocol-sha",
    )


def test_generated_state_replay_rejects_stale_checkpoint_identity() -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="different base checkpoint"):
        entry.validate(expected_base_checkpoint_sha256="other-base")


def test_generated_state_replay_rejects_clean_assignment_leakage_for_generated_role() -> None:
    entry = _entry(
        role="generated_assignment",
    )
    leaked = GeneratedStateReplayEntry(
        **{
            **entry.__dict__,
            "assignment_source": "clean",
            "assignment_tokens": torch.tensor([4, 6, 6, 12, 15], dtype=torch.long),
            "assignment_reveal_count": torch.tensor([2, 3], dtype=torch.long),
        }
    )
    with pytest.raises(ValueError, match="assignment source is incompatible"):
        leaked.validate()


def test_generated_state_replay_rejects_count_mismatch_and_bad_reveal_order() -> None:
    entry = _entry()
    bad_counts = GeneratedStateReplayEntry(
        **{**entry.__dict__, "composition_counts": entry.composition_counts.clone()}
    )
    bad_counts.composition_counts[0, 4] = 2
    with pytest.raises(ValueError, match="composition counts do not close"):
        bad_counts.validate()

    bad_rank = GeneratedStateReplayEntry(
        **{**entry.__dict__, "assignment_reveal_rank": torch.tensor([0, 0, 2, 0, 1])}
    )
    with pytest.raises(ValueError, match="per-graph permutation"):
        bad_rank.validate()


def test_generated_state_replay_rejects_shape_subspace_violation() -> None:
    entry = _entry()
    bad_shape = entry.lattice_log_shape.clone()
    bad_shape[0, 0] = 1.0
    off_subspace = GeneratedStateReplayEntry(**{**entry.__dict__, "lattice_log_shape": bad_shape})
    projector = torch.eye(6).expand(2, -1, -1).clone()
    projector[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="shape subspace"):
        off_subspace.validate(shape_projector=projector)


def test_generated_state_replay_rejects_forbidden_source_overlap() -> None:
    entry = _entry()
    validate_no_forbidden_source_ids([entry], {"other"})
    with pytest.raises(ValueError, match="forbidden source ids"):
        validate_no_forbidden_source_ids([entry], {"alex<unit-test>"})


def test_generated_state_replay_manifest_round_trips_with_stable_hash(tmp_path) -> None:
    entry = _entry()
    manifest = GeneratedStateReplayManifest.from_entries([entry])
    manifest.validate(
        expected_base_checkpoint_sha256="base-sha",
        expected_sampler_commit="commit-sha",
        expected_sampler_protocol_sha256="protocol-sha",
    )
    manifest.validate_against_entries([entry])

    path = tmp_path / "generated-state-replay-manifest.json"
    written_hash = write_generated_state_replay_manifest(path, manifest)
    loaded = load_generated_state_replay_manifest(path)

    assert loaded.to_json_object() == manifest.to_json_object()
    assert loaded.canonical_sha256() == written_hash
    loaded.validate_against_entries([entry])


def test_generated_state_replay_manifest_rejects_payload_digest_mismatch() -> None:
    entry = _entry()
    row = GeneratedStateReplayManifestRow.from_entry(entry)
    bad_hashes = dict(row.tensor_sha256)
    bad_hashes["lattice_log_shape"] = "wrong"
    manifest = GeneratedStateReplayManifest(
        rows=(GeneratedStateReplayManifestRow(**{**row.__dict__, "tensor_sha256": bad_hashes}),)
    )

    with pytest.raises(ValueError, match="does not match replay entry payload"):
        manifest.validate_against_entries([entry])


def test_generated_state_replay_manifest_rejects_duplicate_keys_and_forbidden_ids() -> None:
    entry = _entry()
    manifest = GeneratedStateReplayManifest.from_entries([entry, entry])
    with pytest.raises(ValueError, match="duplicate cache keys"):
        manifest.validate()

    single = GeneratedStateReplayManifest.from_entries([entry])
    with pytest.raises(ValueError, match="forbidden source ids"):
        single.validate(forbidden_source_ids={"alex<unit-test>"})


def test_generated_state_replay_cache_round_trips_payload_and_manifest(tmp_path) -> None:
    entry = _entry()
    manifest_hash = write_generated_state_replay_cache(tmp_path, [entry])
    entries, manifest = load_generated_state_replay_cache(
        tmp_path,
        expected_base_checkpoint_sha256="base-sha",
        expected_sampler_commit="commit-sha",
        expected_sampler_protocol_sha256="protocol-sha",
    )

    assert manifest.canonical_sha256() == manifest_hash
    assert len(entries) == 1
    assert entries[0].key == entry.key
    assert torch.equal(entries[0].composition_counts, entry.composition_counts)
    assert torch.equal(entries[0].assignment_tokens, entry.assignment_tokens)
    assert torch.equal(entries[0].lattice, entry.lattice)
    assert torch.equal(entries[0].fractional_coordinates, entry.fractional_coordinates)


def test_generated_state_replay_cache_rejects_tampered_payload(tmp_path) -> None:
    entry = _entry()
    write_generated_state_replay_cache(tmp_path, [entry])
    tampered = GeneratedStateReplayEntry(
        **{
            **entry.__dict__,
            "lattice_log_shape": entry.lattice_log_shape.clone() + 0.25,
        }
    )
    payload = {
        "format_version": 1,
        "entries": [
            {
                "key": tampered.key.to_json_object(),
                "source_split": tampered.source_split,
                "parent_or_flexible_carrier_id": tampered.parent_or_flexible_carrier_id,
                "composition_source": tampered.composition_source,
                "assignment_source": tampered.assignment_source,
                "lattice_source": tampered.lattice_source,
                "coordinate_source": tampered.coordinate_source,
                "tensors": {
                    "node_count": tampered.node_count,
                    "composition_counts": tampered.composition_counts,
                    "assignment_tokens": tampered.assignment_tokens,
                    "assignment_reveal_rank": tampered.assignment_reveal_rank,
                    "assignment_reveal_count": tampered.assignment_reveal_count,
                    "lattice": tampered.lattice,
                    "lattice_log_volume": tampered.lattice_log_volume,
                    "lattice_log_shape": tampered.lattice_log_shape,
                    "fractional_coordinates": tampered.fractional_coordinates,
                },
            }
        ],
    }
    torch.save(payload, tmp_path / "generated_state_replay_payload.pt")

    with pytest.raises(ValueError, match="does not match replay entry payload"):
        load_generated_state_replay_cache(tmp_path)


def test_generated_state_replay_correctness_role_weight_is_equal() -> None:
    assert _role_weight(4) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="role count"):
        _role_weight(0)


def test_generated_state_replay_correctness_parameter_update_norm() -> None:
    module = torch.nn.Linear(2, 1)
    reference = {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}
    assert _parameter_update_norm(module, reference) == pytest.approx(0.0)
    with torch.no_grad():
        module.weight.add_(1.0)
    assert _parameter_update_norm(module, reference) > 0.0
