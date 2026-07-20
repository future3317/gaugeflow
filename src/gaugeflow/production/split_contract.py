"""Scientific-role contracts for independent crystal-data split axes.

One row may carry assignments on several axes, but an evaluation protocol must
bind to exactly one axis.  In particular, the formula/prototype-disjoint axis
is an OOD novelty test and is never a substitute for IID probability
calibration.  A chronological axis fails closed when the source has no
auditable timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class SplitAxis(str, Enum):
    IID_CALIBRATION = "iid_calibration"
    FORMULA_PROTOTYPE_DISJOINT = "formula_prototype_disjoint"
    TIME = "time"


class EvaluationRole(str, Enum):
    PROBABILITY_CALIBRATION = "probability_calibration"
    OOD_NOVELTY = "ood_novelty"
    TEMPORAL_REDISCOVERY = "temporal_rediscovery"


_ROLE_AXIS = {
    EvaluationRole.PROBABILITY_CALIBRATION: SplitAxis.IID_CALIBRATION,
    EvaluationRole.OOD_NOVELTY: SplitAxis.FORMULA_PROTOTYPE_DISJOINT,
    EvaluationRole.TEMPORAL_REDISCOVERY: SplitAxis.TIME,
}

_AXIS_LABELS = {
    SplitAxis.IID_CALIBRATION: frozenset({"fit", "calibration", "test"}),
    SplitAxis.FORMULA_PROTOTYPE_DISJOINT: frozenset({"train", "val", "test"}),
    SplitAxis.TIME: frozenset({"past", "development", "future"}),
}


@dataclass(frozen=True)
class SplitAxisContract:
    """Hash-bound schema for one scientifically distinct split axis."""

    axis: SplitAxis
    assignment_path: str
    assignment_sha256: str
    labels: tuple[str, ...]
    status: str = "active"
    timestamp_column: str | None = None
    timestamp_source: str | None = None

    def validate(self) -> None:
        if self.status not in {"active", "unavailable"}:
            raise ValueError("split-axis status must be active or unavailable")
        if not self.assignment_path or len(self.assignment_sha256) != 64:
            raise ValueError("active split axes require a path and SHA-256 identity")
        if set(self.labels) != _AXIS_LABELS[self.axis]:
            raise ValueError(f"{self.axis.value} labels do not match the frozen schema")
        if self.axis is SplitAxis.TIME:
            has_time_lineage = bool(self.timestamp_column and self.timestamp_source)
            if self.status == "active" and not has_time_lineage:
                raise ValueError("a time split requires an auditable timestamp and source")
            if self.status == "unavailable" and has_time_lineage:
                raise ValueError("an unavailable time split cannot claim timestamp lineage")
        elif self.timestamp_column is not None or self.timestamp_source is not None:
            raise ValueError("only the time axis may declare timestamp lineage")

    def assert_role(self, role: EvaluationRole) -> None:
        self.validate()
        expected = _ROLE_AXIS[role]
        if self.axis is not expected:
            raise ValueError(
                f"evaluation role {role.value} requires {expected.value}, not {self.axis.value}"
            )
        if self.status != "active":
            raise ValueError(f"split axis {self.axis.value} is unavailable")


@dataclass(frozen=True)
class MultiAxisSplitSchema:
    """The three split axes and their non-interchangeable scientific roles."""

    schema_version: int
    dataset: str
    axes: tuple[SplitAxisContract, ...]

    def validate(self) -> None:
        if self.schema_version != 1 or not self.dataset:
            raise ValueError("unsupported split schema or empty dataset name")
        by_axis = {entry.axis: entry for entry in self.axes}
        if len(by_axis) != len(self.axes) or set(by_axis) != set(SplitAxis):
            raise ValueError("the split schema must declare each scientific axis exactly once")
        for entry in self.axes:
            entry.validate()

    def for_role(self, role: EvaluationRole) -> SplitAxisContract:
        self.validate()
        selected = next(entry for entry in self.axes if entry.axis is _ROLE_AXIS[role])
        selected.assert_role(role)
        return selected


def validate_assignment_columns(
    columns: Mapping[str, Sequence[object]],
    contract: SplitAxisContract,
) -> None:
    """Validate row grain and labels without coupling to pandas or PyArrow."""

    contract.validate()
    required = {"material_id", "split_label"}
    if contract.axis is SplitAxis.TIME and contract.status == "active":
        assert contract.timestamp_column is not None
        required.add(contract.timestamp_column)
    missing = required - set(columns)
    if missing:
        raise ValueError(f"split assignment is missing columns: {sorted(missing)}")
    lengths = {len(columns[name]) for name in required}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ValueError("split assignment columns must share one nonempty row grain")
    material_ids = tuple(map(str, columns["material_id"]))
    if len(set(material_ids)) != len(material_ids) or any(not value for value in material_ids):
        raise ValueError("split assignment material IDs must be nonempty and unique")
    labels = set(map(str, columns["split_label"]))
    if not labels <= set(contract.labels) or labels != set(contract.labels):
        raise ValueError("split assignment must populate every and only declared label")
    if contract.axis is SplitAxis.TIME and contract.status == "active":
        assert contract.timestamp_column is not None
        timestamps = columns[contract.timestamp_column]
        if any(value is None or str(value) == "" for value in timestamps):
            raise ValueError("time split contains missing source timestamps")
