from pathlib import Path

from scripts.evaluate_h1a_coordinate_clean_side_information import (
    _prerequisite_hash_contract,
)


def test_one_pass_hash_contract_uses_frozen_screen_without_transitive_key() -> None:
    prerequisites = {
        "cache_manifest_sha256": "cache",
        "qualification_result_sha256": "mechanism",
        "source_architecture_protocol_sha256": "architecture",
        "clean_side_screen_protocol_sha256": "screen-protocol",
        "clean_side_screen_result_sha256": "screen-result",
    }

    contract = _prerequisite_hash_contract(
        "h1a_coordinate_clean_side_information_one_pass_v1",
        prerequisites,
        Path("cache"),
    )

    assert contract[Path("configs/gates/h1a_coordinate_clean_side_information_v1.json")] == "screen-protocol"
    assert contract[Path("reports/h1a_coordinate_clean_side_information_v1/result.json")] == "screen-result"
    assert Path("reports/h1a_fixed_dynamic_coordinate_learning_curve_v1/result.json") not in contract


def test_quarter_pass_hash_contract_requires_historical_learning_curve() -> None:
    prerequisites = {
        "cache_manifest_sha256": "cache",
        "qualification_result_sha256": "mechanism",
        "source_architecture_protocol_sha256": "architecture",
        "historical_learning_curve_result_sha256": "history",
    }

    contract = _prerequisite_hash_contract(
        "h1a_coordinate_clean_side_information_v1",
        prerequisites,
        Path("cache"),
    )

    assert contract[Path("reports/h1a_fixed_dynamic_coordinate_learning_curve_v1/result.json")] == "history"
