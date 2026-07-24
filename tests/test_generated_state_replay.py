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
from scripts.audit_generated_state_replay_training_contract import _iter_role_chunks, _pack_role_entries
from scripts.build_tiny_generated_state_replay_cache import (
    _read_forbidden_source_ids,
    _reject_forbidden_selection,
    _select_source_indices,
    _source_ids_for_indices,
)
from scripts.evaluate_generated_state_replay_correctness import _paired_bootstrap_w1_delta
from scripts.project_generated_state_replay_checkpoint import _alpha_for_key, _project_state_dict
from scripts.select_generated_state_replay_checkpoint import (
    SelectionContract,
    evaluate_candidate,
    select_candidate,
)
from scripts.train_gaugeflow_base_v2_generated_state_smoke import _model_config, _training_config
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


def test_base_v2_smoke_config_uses_orderless_joint_training() -> None:
    config = _training_config(
        {
            "learning_rate": 2.0e-4,
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "ema_decay": 0.999,
            "coordinate_sigma_min": 0.005,
            "coordinate_sigma_max": 0.5,
            "minimum_time": 0.001,
            "maximum_time": 0.999,
            "precision": "bf16",
            "categorical_path": "orderless_reveal",
            "composition_conditioning": True,
        }
    )

    assert config.objective == "joint"
    assert config.categorical_path == "orderless_reveal"
    assert config.composition_conditioning is True


def test_base_v2_smoke_model_config_maps_capacity_spec() -> None:
    config = _model_config(
        {
            "hidden_dim": 384,
            "vector_dim": 64,
            "layers": 8,
            "radial_dim": 24,
            "radial_cutoff_angstrom": 8.0,
            "edge_dim": 128,
            "angular_channels": 16,
            "edge_refresh_rank": 32,
            "modality_time_conditioning": "separate",
        }
    )

    assert config["radial_cutoff"] == 8.0
    assert config["atlas_residual_circle_samples"] == 8
    assert config["modality_time_conditioning"] == "separate"


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


def test_generated_state_replay_role_chunks_preserve_graph_local_state() -> None:
    terminal = _entry(role="generated_joint")
    terminal = GeneratedStateReplayEntry(
        **{
            **terminal.__dict__,
            "assignment_tokens": torch.tensor([4, 6, 6, 12, 15], dtype=torch.long),
            "assignment_reveal_count": torch.tensor([2, 3], dtype=torch.long),
        }
    )
    entries = [terminal, terminal]
    packed = _pack_role_entries("generated_joint", entries, device=torch.device("cpu"))

    chunks = _iter_role_chunks(packed, 1)

    assert [int(chunk.node_counts.numel()) for chunk in chunks] == [1, 1, 1, 1]
    assert sum(int(chunk.node_counts.sum()) for chunk in chunks) == int(packed.node_counts.sum())
    assert torch.equal(torch.cat([chunk.assignment_tokens for chunk in chunks]), packed.assignment_tokens)
    assert torch.equal(
        torch.cat([chunk.fractional_coordinates for chunk in chunks]),
        packed.fractional_coordinates,
    )
    assert torch.equal(torch.cat([chunk.composition_counts for chunk in chunks]), packed.composition_counts)
    for chunk in chunks:
        assert torch.equal(chunk.batch, torch.repeat_interleave(torch.arange(1), chunk.node_counts.cpu()))


def test_generated_state_replay_correctness_parameter_update_norm() -> None:
    module = torch.nn.Linear(2, 1)
    reference = {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}
    assert _parameter_update_norm(module, reference) == pytest.approx(0.0)
    with torch.no_grad():
        module.weight.add_(1.0)
    assert _parameter_update_norm(module, reference) > 0.0


def test_generated_state_replay_paired_bootstrap_rejects_unpaired_records() -> None:
    base = [{"sample_index": 0, "sampling_failed": False, "volume_per_atom": 1.0}]
    candidate = [{"sample_index": 1, "sampling_failed": False, "volume_per_atom": 1.0}]

    with pytest.raises(ValueError, match="not paired"):
        _paired_bootstrap_w1_delta(
            base,
            candidate,
            torch.tensor([1.0], dtype=torch.float64),
            metric="volume_per_atom",
            points=2,
            scale=1.0,
            bootstrap_samples=8,
            seed=1,
        )


def test_generated_state_replay_paired_bootstrap_zero_for_identical_records() -> None:
    records = [
        {"sample_index": 0, "sampling_failed": False, "minimum_distance": 1.0},
        {"sample_index": 1, "sampling_failed": False, "minimum_distance": 2.0},
        {"sample_index": 2, "sampling_failed": False, "minimum_distance": 3.0},
    ]

    report = _paired_bootstrap_w1_delta(
        records,
        records,
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        metric="minimum_distance",
        points=4,
        scale=1.0,
        bootstrap_samples=16,
        seed=1,
    )

    assert report["mean_delta"] == pytest.approx(0.0)
    assert report["p025_delta"] == pytest.approx(0.0)
    assert report["p975_delta"] == pytest.approx(0.0)
    assert report["probability_delta_le_zero"] == pytest.approx(1.0)


def test_generated_state_replay_projection_assigns_expected_parameter_blocks() -> None:
    assert _alpha_for_key(
        "element_head.0.weight",
        element_alpha=0.1,
        coordinate_alpha=0.2,
        lattice_head_alpha=0.3,
    ) == pytest.approx(0.1)
    assert _alpha_for_key(
        "coordinate_edge_encoder.weight",
        element_alpha=0.1,
        coordinate_alpha=0.2,
        lattice_head_alpha=0.3,
    ) == pytest.approx(0.2)
    assert _alpha_for_key(
        "volume_head.0.weight",
        element_alpha=0.1,
        coordinate_alpha=0.2,
        lattice_head_alpha=0.3,
    ) == pytest.approx(0.3)
    assert _alpha_for_key(
        "shared_trunk.0.weight",
        element_alpha=0.1,
        coordinate_alpha=0.2,
        lattice_head_alpha=0.3,
    ) == pytest.approx(0.0)


def test_generated_state_replay_projection_interpolates_only_declared_blocks() -> None:
    base = {
        "element_head.0.weight": torch.tensor([0.0]),
        "coordinate_carrier.weight": torch.tensor([0.0]),
        "shape_head.0.weight": torch.tensor([0.0]),
        "shared_trunk.0.weight": torch.tensor([0.0]),
        "integer_buffer": torch.tensor([1], dtype=torch.long),
    }
    candidate = {
        key: value.clone()
        for key, value in base.items()
    }
    for key in candidate:
        if candidate[key].is_floating_point():
            candidate[key] += 8.0
    candidate["integer_buffer"] = torch.tensor([2], dtype=torch.long)

    projected = _project_state_dict(
        base,
        candidate,
        element_alpha=0.0,
        coordinate_alpha=1.0,
        lattice_head_alpha=0.25,
    )

    assert projected["element_head.0.weight"].item() == pytest.approx(0.0)
    assert projected["coordinate_carrier.weight"].item() == pytest.approx(8.0)
    assert projected["shape_head.0.weight"].item() == pytest.approx(2.0)
    assert projected["shared_trunk.0.weight"].item() == pytest.approx(0.0)
    assert torch.equal(projected["integer_buffer"], torch.tensor([1], dtype=torch.long))


def test_tiny_replay_builder_reads_forbidden_source_ids(tmp_path) -> None:
    json_path = tmp_path / "forbidden.json"
    json_path.write_text('["mp-1", "mp-2", "mp-1"]\n', encoding="utf-8")
    assert _read_forbidden_source_ids(json_path) == {"mp-1", "mp-2"}

    text_path = tmp_path / "forbidden.txt"
    text_path.write_text("mp-3\n\nmp-4\n", encoding="utf-8")
    assert _read_forbidden_source_ids(text_path) == {"mp-3", "mp-4"}
    assert _read_forbidden_source_ids(None) is None


def test_tiny_replay_builder_selects_contiguous_or_seeded_sources() -> None:
    assert _select_source_indices(
        split_size=10,
        start_index=2,
        sample_count=3,
        selection_seed=None,
    ) == [2, 3, 4]

    selected = _select_source_indices(
        split_size=10,
        start_index=2,
        sample_count=3,
        selection_seed=5705,
    )
    repeated = _select_source_indices(
        split_size=10,
        start_index=2,
        sample_count=3,
        selection_seed=5705,
    )
    assert selected == repeated
    assert selected != [2, 3, 4]
    assert len(selected) == len(set(selected)) == 3
    assert all(0 <= index < 10 for index in selected)


def test_tiny_replay_builder_rejects_forbidden_selected_sources() -> None:
    material_ids = ["mp-0", "mp-1", "mp-2", "mp-3"]
    selected_ids = _source_ids_for_indices(material_ids, [3, 1])
    assert selected_ids == ["mp-3", "mp-1"]
    _reject_forbidden_selection(selected_ids, {"mp-0"})
    with pytest.raises(ValueError, match="selected replay sources overlap forbidden source ids"):
        _reject_forbidden_selection(selected_ids, {"mp-1"})


def test_tiny_replay_builder_rejects_out_of_range_selection() -> None:
    with pytest.raises(ValueError, match="outside the packed Alex split"):
        _select_source_indices(
            split_size=4,
            start_index=3,
            sample_count=2,
            selection_seed=None,
        )


def _selection_eval(
    *,
    nn_delta: float,
    volume_delta: float,
    replay_loss_delta: float = -0.1,
    distance_delta: float = 0.0,
    failures_delta: float = 0.0,
    masks_delta: float = 0.0,
) -> dict[str, object]:
    roles = ("clean_clean", "generated_assignment", "generated_lattice", "generated_joint")
    replay_deltas = {
        role: {
            "loss": replay_loss_delta,
            "element_loss": -0.01,
            "coordinate_loss": -0.01,
            "volume_loss": -0.01,
            "shape_loss": -0.01,
        }
        for role in roles
    }
    free_base = {
        "normalized_nearest_neighbor_wasserstein": 0.538407,
        "normalized_volume_wasserstein": 0.333797,
        "minimum_distance_fraction_at_0_5_angstrom": 1.0,
        "exact_composition_fraction": 1.0,
        "finite_positive_lattice_fraction": 1.0,
        "sampling_failures": 0.0,
        "terminal_masks": 0.0,
    }
    free_candidate = {
        **free_base,
        "normalized_nearest_neighbor_wasserstein": free_base["normalized_nearest_neighbor_wasserstein"]
        + nn_delta,
        "normalized_volume_wasserstein": free_base["normalized_volume_wasserstein"] + volume_delta,
        "minimum_distance_fraction_at_0_5_angstrom": free_base[
            "minimum_distance_fraction_at_0_5_angstrom"
        ]
        + distance_delta,
        "sampling_failures": free_base["sampling_failures"] + failures_delta,
        "terminal_masks": free_base["terminal_masks"] + masks_delta,
    }
    return {
        "schema": "gaugeflow.generated_state_replay_correctness_evaluation.v1",
        "checkpoint": "/runs/candidate.pt",
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_ema_used": True,
        "checkpoint_training_summary": {"steps": 100},
        "replay_role_losses": {
            "base": {},
            "candidate": {},
            "candidate_minus_base": replay_deltas,
        },
        "free_generation": {
            "base": free_base,
            "candidate": free_candidate,
            "candidate_minus_base": {
                "normalized_nearest_neighbor_wasserstein": nn_delta,
                "normalized_volume_wasserstein": volume_delta,
                "minimum_distance_fraction_at_0_5_angstrom": distance_delta,
                "exact_composition_fraction": 0.0,
                "finite_positive_lattice_fraction": 0.0,
                "sampling_failures": failures_delta,
                "terminal_masks": masks_delta,
            },
        },
    }


def _write_selection_eval(tmp_path, name: str, result: dict[str, object]):
    path = tmp_path / f"{name}.json"
    import json

    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_generated_state_replay_checkpoint_selector_prefers_lowest_nn_delta(tmp_path) -> None:
    contract = SelectionContract()
    path_100 = _write_selection_eval(
        tmp_path,
        "100_ema",
        _selection_eval(nn_delta=0.0067, volume_delta=-0.0068, replay_loss_delta=-0.1),
    )
    path_200 = _write_selection_eval(
        tmp_path,
        "200_ema",
        _selection_eval(nn_delta=0.0327, volume_delta=-0.0079, replay_loss_delta=-0.2),
    )
    evidence = {
        "100_ema": evaluate_candidate(
            "100_ema",
            path_100,
            _selection_eval(nn_delta=0.0067, volume_delta=-0.0068),
            contract,
        ),
        "200_ema": evaluate_candidate(
            "200_ema",
            path_200,
            _selection_eval(nn_delta=0.0327, volume_delta=-0.0079, replay_loss_delta=-0.2),
            contract,
        ),
    }

    assert evidence["100_ema"]["eligible"]
    assert evidence["200_ema"]["eligible"]
    selection = select_candidate(evidence)
    assert selection["status"] == "diagnostic_checkpoint_selected"
    assert selection["selected_label"] == "100_ema"


def test_generated_state_replay_checkpoint_selector_rejects_rollout_regressions(tmp_path) -> None:
    contract = SelectionContract()
    raw_path = _write_selection_eval(
        tmp_path,
        "100_raw",
        _selection_eval(nn_delta=0.431, volume_delta=-0.037),
    )
    volume_path = _write_selection_eval(
        tmp_path,
        "volume_bad",
        _selection_eval(nn_delta=0.01, volume_delta=0.02),
    )
    distance_path = _write_selection_eval(
        tmp_path,
        "distance_bad",
        _selection_eval(nn_delta=0.01, volume_delta=-0.01, distance_delta=-0.1),
    )
    evidence = {
        "100_raw": evaluate_candidate(
            "100_raw",
            raw_path,
            _selection_eval(nn_delta=0.431, volume_delta=-0.037),
            contract,
        ),
        "volume_bad": evaluate_candidate(
            "volume_bad",
            volume_path,
            _selection_eval(nn_delta=0.01, volume_delta=0.02),
            contract,
        ),
        "distance_bad": evaluate_candidate(
            "distance_bad",
            distance_path,
            _selection_eval(nn_delta=0.01, volume_delta=-0.01, distance_delta=-0.1),
            contract,
        ),
    }

    assert not evidence["100_raw"]["eligible"]
    assert not evidence["volume_bad"]["eligible"]
    assert not evidence["distance_bad"]["eligible"]
    assert select_candidate(evidence)["status"] == "no_eligible_checkpoint"


def test_generated_state_replay_checkpoint_selector_rejects_replay_non_improvement(tmp_path) -> None:
    contract = SelectionContract()
    path = _write_selection_eval(
        tmp_path,
        "flat_replay",
        _selection_eval(nn_delta=0.0, volume_delta=-0.01, replay_loss_delta=0.0),
    )
    evidence = {
        "flat_replay": evaluate_candidate(
            "flat_replay",
            path,
            _selection_eval(nn_delta=0.0, volume_delta=-0.01, replay_loss_delta=0.0),
            contract,
        )
    }

    assert not evidence["flat_replay"]["eligible"]
    assert select_candidate(evidence)["status"] == "no_eligible_checkpoint"
