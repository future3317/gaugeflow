"""Select a generated-state replay checkpoint by a declared retention rule.

This selector consumes existing
``gaugeflow.generated_state_replay_correctness_evaluation.v1`` JSON files.  It
does not run generation, train a model, or qualify Stage-E.  The purpose is to
prevent manual cherry-picking among replay-correctness dose checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "gaugeflow.generated_state_replay_correctness_evaluation.v1"
OUTPUT_SCHEMA = "gaugeflow.generated_state_replay_checkpoint_selection.v1"
NN_METRIC = "normalized_nearest_neighbor_wasserstein"
VOLUME_METRIC = "normalized_volume_wasserstein"
DISTANCE_VALID_METRIC = "minimum_distance_fraction_at_0_5_angstrom"
EXACT_COMPOSITION_METRIC = "exact_composition_fraction"
FINITE_POSITIVE_LATTICE_METRIC = "finite_positive_lattice_fraction"
SAMPLING_FAILURES_METRIC = "sampling_failures"
TERMINAL_MASKS_METRIC = "terminal_masks"


@dataclass(frozen=True)
class SelectionContract:
    max_nn_w1_delta: float = 0.05
    max_volume_w1_delta: float = 0.0
    min_exact_composition_fraction: float = 1.0
    min_finite_positive_lattice_fraction: float = 1.0
    min_distance_valid_delta: float = 0.0
    max_sampling_failures_delta: float = 0.0
    max_terminal_masks_delta: float = 0.0
    tolerance: float = 1e-12

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"selection contract {name} is not finite")
        if self.max_nn_w1_delta < 0.0:
            raise ValueError("max_nn_w1_delta must be nonnegative")
        if self.min_exact_composition_fraction < 0.0 or self.min_exact_composition_fraction > 1.0:
            raise ValueError("min_exact_composition_fraction must lie in [0, 1]")
        if (
            self.min_finite_positive_lattice_fraction < 0.0
            or self.min_finite_positive_lattice_fraction > 1.0
        ):
            raise ValueError("min_finite_positive_lattice_fraction must lie in [0, 1]")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate label and evaluation JSON path as LABEL=PATH.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nn-w1-delta", type=float, default=SelectionContract.max_nn_w1_delta)
    parser.add_argument("--max-volume-w1-delta", type=float, default=SelectionContract.max_volume_w1_delta)
    parser.add_argument(
        "--min-exact-composition-fraction",
        type=float,
        default=SelectionContract.min_exact_composition_fraction,
    )
    parser.add_argument(
        "--min-finite-positive-lattice-fraction",
        type=float,
        default=SelectionContract.min_finite_positive_lattice_fraction,
    )
    parser.add_argument(
        "--min-distance-valid-delta",
        type=float,
        default=SelectionContract.min_distance_valid_delta,
    )
    parser.add_argument(
        "--max-sampling-failures-delta",
        type=float,
        default=SelectionContract.max_sampling_failures_delta,
    )
    parser.add_argument(
        "--max-terminal-masks-delta",
        type=float,
        default=SelectionContract.max_terminal_masks_delta,
    )
    parser.add_argument("--tolerance", type=float, default=SelectionContract.tolerance)
    return parser.parse_args()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_leaves(value: Any) -> list[int | float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _numeric_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _numeric_leaves(child)]
    return []


def _finite_numeric_payload(result: dict[str, Any]) -> bool:
    leaves = _numeric_leaves(result)
    return bool(leaves) and all(math.isfinite(float(value)) for value in leaves)


def _require_metric(mapping: dict[str, Any], metric: str) -> float:
    value = mapping.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric: {metric}")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"metric is not finite: {metric}")
    return value_float


def _parse_candidates(items: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in items:
        label, separator, path_text = item.partition("=")
        if not separator or not label or not path_text:
            raise ValueError("candidate must be LABEL=PATH")
        if label in paths:
            raise ValueError(f"duplicate candidate label: {label}")
        paths[label] = Path(path_text)
    return paths


def _load_candidate(label: str, path: Path) -> dict[str, Any]:
    result = _read_json_object(path)
    if result.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"{label} is not a generated-state replay correctness evaluation")
    return result


def _replay_role_report(result: dict[str, Any], *, tolerance: float) -> dict[str, Any]:
    replay = result.get("replay_role_losses")
    if not isinstance(replay, dict):
        raise ValueError("evaluation lacks replay_role_losses")
    deltas = replay.get("candidate_minus_base")
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError("evaluation lacks replay role deltas")
    per_role: dict[str, Any] = {}
    total_delta_sum = 0.0
    all_total_lower = True
    for role, role_deltas in sorted(deltas.items()):
        if not isinstance(role_deltas, dict):
            raise ValueError(f"replay role {role} lacks metric deltas")
        loss_delta = _require_metric(role_deltas, "loss")
        total_delta_sum += loss_delta
        role_ok = loss_delta < -tolerance
        all_total_lower = all_total_lower and role_ok
        per_role[str(role)] = {
            "loss_delta": loss_delta,
            "total_loss_lower": role_ok,
        }
    return {
        "all_total_losses_lower": all_total_lower,
        "total_loss_delta_sum": total_delta_sum,
        "total_loss_improvement": -total_delta_sum,
        "roles": per_role,
    }


def _free_generation_report(result: dict[str, Any], contract: SelectionContract) -> dict[str, Any]:
    free = result.get("free_generation")
    if not isinstance(free, dict):
        raise ValueError("evaluation lacks free_generation")
    base = free.get("base")
    candidate = free.get("candidate")
    delta = free.get("candidate_minus_base")
    if not isinstance(base, dict) or not isinstance(candidate, dict) or not isinstance(delta, dict):
        raise ValueError("free_generation lacks base/candidate/delta sections")
    nn_delta = _require_metric(delta, NN_METRIC)
    volume_delta = _require_metric(delta, VOLUME_METRIC)
    distance_delta = _require_metric(delta, DISTANCE_VALID_METRIC)
    exact_candidate = _require_metric(candidate, EXACT_COMPOSITION_METRIC)
    exact_delta = _require_metric(delta, EXACT_COMPOSITION_METRIC)
    finite_candidate = _require_metric(candidate, FINITE_POSITIVE_LATTICE_METRIC)
    finite_delta = _require_metric(delta, FINITE_POSITIVE_LATTICE_METRIC)
    failures_delta = _require_metric(delta, SAMPLING_FAILURES_METRIC)
    masks_delta = _require_metric(delta, TERMINAL_MASKS_METRIC)
    tolerance = contract.tolerance
    checks = {
        "nn_w1_non_degraded": nn_delta <= contract.max_nn_w1_delta + tolerance,
        "volume_w1_non_inferior": volume_delta <= contract.max_volume_w1_delta + tolerance,
        "distance_valid_non_degraded": distance_delta + tolerance >= contract.min_distance_valid_delta,
        "sampling_failures_non_regressed": failures_delta <= contract.max_sampling_failures_delta + tolerance,
        "terminal_masks_non_regressed": masks_delta <= contract.max_terminal_masks_delta + tolerance,
        "exact_composition_unchanged": abs(exact_delta) <= tolerance
        and exact_candidate + tolerance >= contract.min_exact_composition_fraction,
        "finite_positive_lattice_unchanged": abs(finite_delta) <= tolerance
        and finite_candidate + tolerance >= contract.min_finite_positive_lattice_fraction,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "metrics": {
            "nn_w1_delta": nn_delta,
            "volume_w1_delta": volume_delta,
            "distance_valid_delta": distance_delta,
            "sampling_failures_delta": failures_delta,
            "terminal_masks_delta": masks_delta,
            "exact_composition_candidate": exact_candidate,
            "exact_composition_delta": exact_delta,
            "finite_positive_lattice_candidate": finite_candidate,
            "finite_positive_lattice_delta": finite_delta,
        },
    }


def evaluate_candidate(
    label: str,
    path: Path,
    result: dict[str, Any],
    contract: SelectionContract,
) -> dict[str, Any]:
    finite = _finite_numeric_payload(result)
    replay = _replay_role_report(result, tolerance=contract.tolerance)
    free = _free_generation_report(result, contract)
    eligible = finite and bool(replay["all_total_losses_lower"]) and bool(free["all_checks_passed"])
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256_file(path),
        "checkpoint": result.get("checkpoint"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "checkpoint_ema_used": result.get("checkpoint_ema_used"),
        "checkpoint_training_summary": result.get("checkpoint_training_summary"),
        "finite_numeric_payload": finite,
        "replay": replay,
        "free_generation": free,
        "eligible": eligible,
    }


def select_candidate(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in evidence.values() if item["eligible"]]
    if not eligible:
        return {
            "status": "no_eligible_checkpoint",
            "selected_label": None,
            "tie_break_order": [
                "fewest free NN-W1 delta",
                "largest replay total-loss improvement",
                "lexicographic label",
            ],
        }
    ordered = sorted(
        eligible,
        key=lambda item: (
            float(item["free_generation"]["metrics"]["nn_w1_delta"]),
            -float(item["replay"]["total_loss_improvement"]),
            str(item["label"]),
        ),
    )
    selected = ordered[0]
    return {
        "status": "diagnostic_checkpoint_selected",
        "selected_label": selected["label"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint_ema_used": selected["checkpoint_ema_used"],
        "selected_metrics": {
            "nn_w1_delta": selected["free_generation"]["metrics"]["nn_w1_delta"],
            "volume_w1_delta": selected["free_generation"]["metrics"]["volume_w1_delta"],
            "distance_valid_delta": selected["free_generation"]["metrics"]["distance_valid_delta"],
            "replay_total_loss_improvement": selected["replay"]["total_loss_improvement"],
        },
        "tie_break_order": [
            "fewest free NN-W1 delta",
            "largest replay total-loss improvement",
            "lexicographic label",
        ],
    }


def _contract_from_args(args: argparse.Namespace) -> SelectionContract:
    contract = SelectionContract(
        max_nn_w1_delta=float(args.max_nn_w1_delta),
        max_volume_w1_delta=float(args.max_volume_w1_delta),
        min_exact_composition_fraction=float(args.min_exact_composition_fraction),
        min_finite_positive_lattice_fraction=float(args.min_finite_positive_lattice_fraction),
        min_distance_valid_delta=float(args.min_distance_valid_delta),
        max_sampling_failures_delta=float(args.max_sampling_failures_delta),
        max_terminal_masks_delta=float(args.max_terminal_masks_delta),
        tolerance=float(args.tolerance),
    )
    contract.validate()
    return contract


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    contract = _contract_from_args(args)
    paths = _parse_candidates(args.candidate)
    evidence = {
        label: evaluate_candidate(label, path, _load_candidate(label, path), contract)
        for label, path in sorted(paths.items())
    }
    output = {
        "schema": OUTPUT_SCHEMA,
        "contract": contract.__dict__,
        "candidate_evidence": evidence,
        "selection": select_candidate(evidence),
        "boundary": (
            "diagnostic checkpoint selection for 34M generated-state replay correctness; "
            "not a Stage-E pass, not Stage-F authorization, and not a capacity result"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
