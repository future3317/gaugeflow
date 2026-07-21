from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.diagnose_h1a_coordinate_generator import _coordinate_side_time_arguments
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


def test_clean_side_coordinate_evaluation_sets_both_side_clocks_to_zero() -> None:
    runtime = SimpleNamespace(
        model=SimpleNamespace(uses_side_modality_times=True),
        training_config={"coordinate_clean_side_information": True},
    )
    coordinate_time = torch.tensor([0.2, 0.7])

    arguments = _coordinate_side_time_arguments(runtime, coordinate_time)

    torch.testing.assert_close(arguments["element_time"], torch.zeros_like(coordinate_time))
    torch.testing.assert_close(arguments["lattice_time"], torch.zeros_like(coordinate_time))


def test_joint_coordinate_evaluation_uses_matching_side_clocks() -> None:
    runtime = SimpleNamespace(
        model=SimpleNamespace(uses_side_modality_times=True),
        training_config={"coordinate_clean_side_information": False},
    )
    coordinate_time = torch.tensor([0.2, 0.7])

    arguments = _coordinate_side_time_arguments(runtime, coordinate_time)

    torch.testing.assert_close(arguments["element_time"], coordinate_time)
    torch.testing.assert_close(arguments["lattice_time"], coordinate_time)
