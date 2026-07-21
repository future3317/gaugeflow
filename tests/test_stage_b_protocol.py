from pathlib import Path

from gaugeflow.file_utils import load_json_object, sha256_file


def test_stage_b_protocol_binds_active_training_implementation() -> None:
    protocol = load_json_object(Path("configs/gates/stage_b_physical_representation_v1.json"))
    prerequisites = protocol["prerequisites"]
    paths = {
        "runner_sha256": Path("scripts/train_physical_representation.py"),
        "physical_training_sha256": Path("src/gaugeflow/production/physical_training.py"),
        "physical_checkpointing_sha256": Path(
            "src/gaugeflow/production/physical_checkpointing.py"
        ),
        "rank_sharded_data_sha256": Path("src/gaugeflow/production/rank_sharded_data.py"),
        "evaluator_sha256": Path("scripts/evaluate_physical_representation.py"),
        "physical_evaluation_sha256": Path("src/gaugeflow/production/physical_evaluation.py"),
    }
    assert all(sha256_file(path) == prerequisites[name] for name, path in paths.items())
