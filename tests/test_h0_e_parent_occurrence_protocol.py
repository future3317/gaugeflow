from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_e1a_protocol_freezes_the_ordered_v1_no_candidate_panel():
    config = json.loads(Path("configs/gates/h0_e_maximal_t_parent_occurrence_e1a_v1.json").read_text(encoding="utf-8"))
    selection = config["selection"]
    material_ids = selection["material_ids"]
    payload = json.dumps(material_ids, separators=(",", ":")).encode()
    assert len(material_ids) == 64
    assert len(set(material_ids)) == 64
    assert hashlib.sha256(payload).hexdigest() == selection["ordered_material_ids_sha256"]
    assert config["thresholds"]["new_candidate_materials_min"] == 3
    assert config["advancement_rule"].startswith("E1a success permits only a separately frozen E1b")


def test_parent_path_quarantine_does_not_rewrite_frozen_e1a_or_child_data():
    frozen = json.loads(
        Path("configs/gates/h0_e_maximal_t_parent_occurrence_e1a_v1.json").read_text(
            encoding="utf-8"
        )
    )
    quarantine = json.loads(
        Path("configs/data_quality/parent_occurrence_quarantine_v1.json").read_text(
            encoding="utf-8"
        )
    )
    material_id = "alex<agm004639609>"
    assert material_id in frozen["selection"]["material_ids"]
    assert quarantine["material_exclusions"] == []
    assert "the frozen H0-A child split" in quarantine["scope"]["excluded"]
    assert "the frozen H0-E-v2 E1a panel and artifacts" in quarantine["scope"]["excluded"]
    entry = quarantine["path_quarantine"][0]
    assert entry["material_id"] == material_id
    assert (entry["child_space_group"], entry["parent_space_group"]) == (12, 71)
    assert entry["action"] == "exclude_all_matching_parent_embeddings"
    assert entry["evidence"]["source_hencky_norm"] > entry["evidence"]["frozen_hencky_limit"]
    assert len(entry["evidence"]["space_group_symprec_panel_angstrom"]) == 7


def test_h0_e_v3_k0_is_a_new_frozen_cell_changing_mechanism_gate():
    config = json.loads(
        Path("configs/gates/h0_e_v3_maximal_k_occurrence_k0_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status_before_run"] == "frozen_not_run"
    assert config["selection"]["ordered_material_ids_sha256"] == (
        "d3a9cb27338e1ec822f8194173b8d85d1b80ff5a1d8377e750b93b380dd615bf"
    )
    assert config["setting_and_search"]["expected_candidate_edges"] == 578
    assert config["thresholds"]["new_candidate_materials_min"] == 3
    assert config["thresholds"]["source_max_displacement_angstrom"] == 0.2
    assert config["thresholds"]["source_hencky_norm_max"] == 0.15
    assert "distance(x,Fix(G)) >= distance(x,Fix(M))" in config["rationale"][
        "multistep_t_dominance"
    ]
    assert config["rationale"]["not_e1b"].startswith(
        "this protocol is a separately frozen H0-E-v3 successor"
    )
    assert config["advancement_rule"].startswith(
        "K0 success permits only a separately frozen H0-E-v3 occurrence protocol"
    )


def test_h0_e_v4_o0_freezes_ordered_occupational_mechanism_without_h0_claim():
    config = json.loads(
        Path("configs/gates/h0_e_v4_occupational_order_o0_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status_before_run"] == "frozen_not_run"
    assert config["selection"]["ordered_material_ids_sha256"] == (
        "d3a9cb27338e1ec822f8194173b8d85d1b80ff5a1d8377e750b93b380dd615bf"
    )
    assert config["setting_and_search"]["expected_candidate_edges"] == 1008
    assert config["setting_and_search"]["expected_quarantined_edges"] == 3
    assert config["thresholds"]["new_candidate_materials_min"] == 3
    assert config["thresholds"]["occupationally_nontrivial_materials_min"] == 3
    assert config["mathematical_contract"]["terminal_subgroup"].startswith(
        "H_child=H_a intersect"
    )
    assert config["rationale"]["physical_scope"].startswith(
        "terminal structures remain fully ordered integer-element crystals"
    )
    assert config["rationale"]["not_h0_qualification"].endswith(
        "a separately frozen held-out O1"
    )
    assert config["advancement_rule"].startswith(
        "O0 success permits only a separately frozen held-out O1"
    )
