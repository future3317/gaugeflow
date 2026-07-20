from scripts.audit_h1a_j1_gradient_geometry import _module_group, _summary


def test_j1_gradient_partition_separates_fusion_and_dynamic_edges() -> None:
    assert _module_group("time_embedding.network.0.weight") == "input_time_embeddings"
    assert _module_group("modality_time_fusion.weight") == "time_fusion"
    assert _module_group("blocks.0.scalar_message.0.weight") == "base_message_blocks"
    assert _module_group("blocks.0.edge_update.0.weight") == "dynamic_edge_angular"
    assert _module_group("coordinate_control_gate.weight") == "coordinate_readout"
    assert _module_group("element_head.0.weight") == "inactive_other"


def test_j1_gradient_summary_is_deterministic() -> None:
    result = _summary([1.0, 2.0, 3.0, 4.0])
    assert result["min"] == 1.0
    assert result["median"] == 2.5
    assert result["max"] == 4.0
