"""Verify the frozen two-target selection for Gate A3 before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gaugeflow.tensor import fixed_so3_frames, piezo_from_irreps, rotate_rank3  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: pd.Series) -> pd.Series:
    return value.astype(str).str.lower().isin(("true", "1"))


def _relative_orbit_distance(
    first: torch.Tensor, second: torch.Tensor, frames: torch.Tensor
) -> float:
    rotated = rotate_rank3(second.unsqueeze(0), frames)
    numerator = (first.unsqueeze(0) - rotated).square().sum(dim=(-1, -2, -3)).sqrt().amin()
    denominator = (first.square().sum() + second.square().sum()).sqrt().clamp_min(1e-12)
    return float(numerator / denominator)


def _lattice_shape_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first_gram = first @ first.T
    second_gram = second @ second.T
    first_shape = first_gram / first_gram.trace().clamp_min(1e-12)
    second_shape = second_gram / second_gram.trace().clamp_min(1e-12)
    return float(torch.linalg.vector_norm(first_shape - second_shape))


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
        raise ValueError(f"Unexpected preprocessed cache: {path}")
    return payload["records"]


def select_pair(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = protocol["selection"]
    data = protocol["data"]
    eligibility = selection["eligibility"]
    rows_path = _resolve(data["audit_rows"])
    split_path = _resolve(data["split_manifest"])
    cache_path = _resolve(data["preprocessed_cache"])
    frame = pd.read_csv(rows_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(str, split[data["split"]]))
    eligible = frame[
        frame.material_id.astype(str).isin(train_ids)
        & _as_bool(frame["valid"])
        & ~_as_bool(frame["exact_zero"])
        & frame.atom_count.between(eligibility["atom_count_min"], eligibility["atom_count_max"])
    ].copy()
    eligible["material_id"] = eligible.material_id.astype(str)
    response_threshold = float(
        eligible.tensor_norm.quantile(eligibility["response_norm_quantile_among_nonzero_candidates"])
    )
    expected_threshold = eligibility["expected_response_norm_threshold"]
    if abs(response_threshold - expected_threshold) > 1e-9:
        raise ValueError(
            f"Response threshold drifted: observed {response_threshold}, expected {expected_threshold}"
        )
    eligible = eligible[eligible.tensor_norm >= response_threshold].sort_values("material_id")
    records = _load_records(cache_path)
    frames = fixed_so3_frames(eligibility["fixed_so3_frames"])
    tensors = {
        material_id: piezo_from_irreps(torch.as_tensor(records[material_id]["piezo_irreps"]).float())
        for material_id in eligible.material_id
    }
    lattices = {
        material_id: torch.as_tensor(records[material_id]["lattice"]).float()
        for material_id in eligible.material_id
    }
    candidates: list[dict[str, Any]] = []
    for atom_count, group in eligible.groupby("atom_count", sort=True):
        group = group.sort_values("material_id").reset_index(drop=True)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                first_id = str(group.loc[left, "material_id"])
                second_id = str(group.loc[right, "material_id"])
                if group.loc[left, "formula"] == group.loc[right, "formula"]:
                    continue
                orbit_distance = _relative_orbit_distance(tensors[first_id], tensors[second_id], frames)
                shape_distance = _lattice_shape_distance(lattices[first_id], lattices[second_id])
                if orbit_distance < eligibility["relative_tensor_orbit_distance_min"]:
                    continue
                if shape_distance < eligibility["scale_invariant_lattice_shape_distance_min"]:
                    continue
                candidates.append({
                    "material_ids": [first_id, second_id],
                    "formulas": [str(group.loc[left, "formula"]), str(group.loc[right, "formula"])],
                    "atom_count": int(atom_count),
                    "response_norms": [float(group.loc[left, "tensor_norm"]), float(group.loc[right, "tensor_norm"])],
                    "relative_tensor_orbit_distance": orbit_distance,
                    "lattice_shape_distance": shape_distance,
                })
    if not candidates:
        raise RuntimeError("No pair satisfies the pre-registered Gate A3 selection criteria")
    winner = sorted(
        candidates,
        key=lambda value: (
            -value["relative_tensor_orbit_distance"],
            -min(value["response_norms"]),
            -value["lattice_shape_distance"],
            value["atom_count"],
            *value["material_ids"],
        ),
    )[0]
    return winner, {
        "response_norm_threshold": response_threshold,
        "eligible_records": int(len(eligible)),
        "eligible_pairs": int(len(candidates)),
        "audit_rows_sha256": _sha256(rows_path),
        "split_manifest_sha256": _sha256(split_path),
        "preprocessed_cache_sha256": _sha256(cache_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a3_early_branching_v1.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/gate_a3_early_branching_v1_selection/selection_manifest.json"),
    )
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_two_target_only":
        raise ValueError("Gate A3 selection only accepts the frozen two-target protocol")
    winner, evidence = select_pair(protocol)
    expected = protocol["selection"]["expected"]
    if winner["material_ids"] != protocol["selection"]["selected_material_ids"]:
        raise ValueError(f"Selected IDs drifted: {winner['material_ids']}")
    if winner["material_ids"] != protocol["material_ids"]:
        raise ValueError("Top-level training IDs do not match the verified selection")
    for key in ("formulas", "atom_count"):
        if winner[key] != expected[key]:
            raise ValueError(f"Selected {key} drifted: {winner[key]}")
    for key in ("response_norms", "relative_tensor_orbit_distance", "lattice_shape_distance"):
        observed = torch.as_tensor(winner[key], dtype=torch.float64)
        reference = torch.as_tensor(expected[key], dtype=torch.float64)
        if not torch.allclose(observed, reference, atol=1e-8, rtol=1e-8):
            raise ValueError(f"Selected {key} drifted: {winner[key]}")
    manifest = {
        "schema": 1,
        "name": "Gate A3 early-branching pair selection",
        "status": "verified_pre_registered_two_target_selection",
        "protocol_sha256": _sha256(protocol_path),
        "selection": winner,
        **evidence,
        "scope": "Selection only; this manifest does not start S2, full training, relaxation, DFT, or DFPT.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
