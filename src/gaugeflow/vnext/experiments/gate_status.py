"""Machine-enforced predecessor checks for the vNext gate sequence."""

from __future__ import annotations

import json
from pathlib import Path


class GateBlockedError(RuntimeError):
    """Raised before any work when a predecessor gate is not qualified."""


def require_gate_status(path: Path, *, gate: str, accepted: frozenset[str]) -> dict[str, object]:
    """Load and validate a predecessor ``status.json`` or block immediately."""
    if not path.is_file():
        raise GateBlockedError(f"{gate} predecessor status is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != gate:
        raise GateBlockedError(f"expected predecessor {gate}, found {payload.get('gate')!r}")
    status = payload.get("status")
    if status not in accepted:
        choices = ", ".join(sorted(accepted))
        raise GateBlockedError(f"{gate} status {status!r} is not one of the required states: {choices}")
    return payload


def require_gate_authorization(path: Path, *, gate: str, successor: str) -> dict[str, object]:
    """Require a versioned partial-legacy execution and explicit successor unlock."""
    if not path.is_file():
        raise GateBlockedError(f"{gate} authorization status is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != gate:
        raise GateBlockedError(f"expected authorization from {gate}, found {payload.get('gate')!r}")
    if payload.get("execution_status") != "complete_partial_legacy":
        raise GateBlockedError(f"{gate} partial-legacy execution is not complete")
    authorization = payload.get("successor_authorization")
    if not isinstance(authorization, dict) or authorization.get(successor) is not True:
        raise GateBlockedError(f"{gate} does not authorize {successor}")
    return payload
