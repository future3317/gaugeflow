from __future__ import annotations

import pytest

from gaugeflow.production.split_contract import (
    EvaluationRole,
    MultiAxisSplitSchema,
    SplitAxis,
    SplitAxisContract,
    validate_assignment_columns,
)


def _hash(character: str) -> str:
    return character * 64


def _schema() -> MultiAxisSplitSchema:
    return MultiAxisSplitSchema(
        schema_version=1,
        dataset="Alex-MP-20",
        axes=(
            SplitAxisContract(
                SplitAxis.IID_CALIBRATION,
                "iid.parquet",
                _hash("a"),
                ("fit", "calibration", "test"),
            ),
            SplitAxisContract(
                SplitAxis.FORMULA_PROTOTYPE_DISJOINT,
                "novelty.parquet",
                _hash("b"),
                ("train", "val", "test"),
            ),
            SplitAxisContract(
                SplitAxis.TIME,
                "time-unavailable.json",
                _hash("c"),
                ("past", "development", "future"),
                status="unavailable",
            ),
        ),
    )


def test_split_roles_cannot_be_silently_interchanged() -> None:
    schema = _schema()
    schema.validate()
    iid = schema.for_role(EvaluationRole.PROBABILITY_CALIBRATION)
    assert iid.axis is SplitAxis.IID_CALIBRATION
    with pytest.raises(ValueError, match="requires iid_calibration"):
        schema.axes[1].assert_role(EvaluationRole.PROBABILITY_CALIBRATION)
    with pytest.raises(ValueError, match="unavailable"):
        schema.for_role(EvaluationRole.TEMPORAL_REDISCOVERY)


def test_active_time_split_fails_closed_without_timestamp_lineage() -> None:
    invalid = SplitAxisContract(
        SplitAxis.TIME,
        "time.parquet",
        _hash("d"),
        ("past", "development", "future"),
    )
    with pytest.raises(ValueError, match="auditable timestamp"):
        invalid.validate()


def test_assignment_table_requires_unique_grain_and_complete_labels() -> None:
    contract = _schema().axes[0]
    validate_assignment_columns(
        {
            "material_id": ["a", "b", "c"],
            "split_label": ["fit", "calibration", "test"],
        },
        contract,
    )
    with pytest.raises(ValueError, match="unique"):
        validate_assignment_columns(
            {
                "material_id": ["a", "a", "c"],
                "split_label": ["fit", "calibration", "test"],
            },
            contract,
        )
