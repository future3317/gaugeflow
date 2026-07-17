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
