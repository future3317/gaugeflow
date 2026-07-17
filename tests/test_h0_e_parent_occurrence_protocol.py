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
