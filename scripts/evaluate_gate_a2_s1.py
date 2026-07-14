"""Evaluate the fixed Gate A2 S1 direct-irrep conditional-control screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from audit_gate_a1 import (  # noqa: E402
    _common_initial_state,
    _distance_diagnostics,
    _dummy_batch,
    _seed,
    _teacher_forced_ranking,
    _trajectory,
)
from evaluate_gate_a import (  # noqa: E402
    _flow_gap,
    _generated_distribution,
    _load_model,
    _load_panel,
)
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher  # noqa: E402
from gaugeflow.tensor import normalize_isotypic  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_norms(
    model, conditions: torch.Tensor, evaluation: dict[str, Any], variant: str, device: torch.device
) -> pd.DataFrame:
    targets, atoms = conditions.shape[0], evaluation["sample_atoms"]
    batch, _ = _dummy_batch(conditions, 1, atoms, device)
    initial = _common_initial_state(targets, 1, atoms, device, evaluation["seed"] + 50_000)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(evaluation["trajectory_steps"]):
            time_value = step / evaluation["trajectory_steps"]
            time = torch.full((targets,), time_value, device=device)
            outputs = model(
                initial.type_state, initial.frac_coords, initial.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present, return_velocity_components=True,
            )
            components = outputs[4]
            for head, base_key, residual_key, graph_level in (
                ("type_logit", "type_base", "type_conditional_residual", False),
                ("fractional_coordinate", "coordinate_base", "coordinate_conditional_residual", False),
                ("lattice_log", "lattice_base", "lattice_conditional_residual", True),
            ):
                base = components[base_key]
                residual = components[residual_key]
                if graph_level:
                    base = base.reshape(targets, -1)
                    residual = residual.reshape(targets, -1)
                else:
                    base = base.reshape(targets, atoms, -1)
                    residual = residual.reshape(targets, atoms, -1)
                for target in range(targets):
                    base_norm = float(base[target].square().mean().sqrt())
                    residual_norm = float(residual[target].square().mean().sqrt())
                    gate = float(components["gate"][target])
                    rows.append({
                        "variant": variant,
                        "time": time_value,
                        "target": target,
                        "head": head,
                        "base_velocity_rms": base_norm,
                        "conditional_residual_rms": residual_norm,
                        "gated_conditional_residual_rms": gate * residual_norm,
                        "gated_residual_over_base": gate * residual_norm / max(base_norm, 1e-12),
                        "g_t": gate,
                        "conditional_control": components["mode"],
                    })
    return pd.DataFrame(rows)


def _terminal_state_summary(trajectory_rows: list[dict[str, Any]], variant: str) -> tuple[pd.DataFrame, bool]:
    frame = pd.DataFrame(trajectory_rows)
    states = frame[frame.quantity.str.startswith("state_")]
    rows = []
    for quantity, group in states.groupby("quantity", sort=False):
        curve = group.groupby("time", as_index=False).pairwise_rms.mean().sort_values("time")
        peak = float(curve.pairwise_rms.max())
        terminal = float(curve.iloc[-1].pairwise_rms)
        rows.append({
            "variant": variant,
            "state_head": quantity.removeprefix("state_"),
            "terminal_pairwise_rms": terminal,
            "peak_pairwise_rms": peak,
            "terminal_over_peak": terminal / max(peak, 1e-12),
            "finite": math.isfinite(terminal),
            "retained_at_90pct_peak": terminal >= 0.9 * peak,
        })
    summary = pd.DataFrame(rows)
    passed = bool(
        (summary.terminal_pairwise_rms > 0).all()
        and summary.finite.all()
        and summary.retained_at_90pct_peak.all()
    )
    return summary, passed


def _guided_distribution(model, conditions: torch.Tensor, evaluation: dict[str, Any], scale: float, device: torch.device) -> dict[str, Any]:
    matcher = RiemannianCrystalFlowMatcher()
    batch, labels = _dummy_batch(conditions, evaluation["sample_replicates"], evaluation["sample_atoms"], device)
    _seed(evaluation["seed"])
    state = matcher.sample(model, batch, steps=evaluation["sample_steps"], guidance_scale=scale)
    features = []
    # Keep the same feature definition as the frozen Gate A evaluator.
    from evaluate_gate_a import _state_features  # local import avoids a public API expansion

    features = _state_features(state, batch)
    atoms = evaluation["sample_atoms"]
    finite = torch.stack((
        torch.isfinite(state.type_state).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(state.frac_coords).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(state.lattice_log).all(dim=1),
    )).all(dim=0)
    return {
        **_distance_diagnostics(features, labels),
        "sampling_failure_count": int((~finite).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a2_conditional_control_v1.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/gate_a2_conditional_control_v1/s1_direct_irrep"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a2_conditional_control_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_s1_direct_irrep_only":
        raise ValueError("Only the frozen A2 S1 protocol may be evaluated here")
    device = torch.device(args.device)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.checkpoint_root if args.checkpoint_root.is_absolute() else ROOT / args.checkpoint_root
    # Gate A2 keeps the S1 split inside its frozen training specification;
    # the legacy panel loader expects it in the data mapping.  Adapt only the
    # in-memory view so the already-hashed protocol file remains immutable.
    parent_protocol = {
        "data": {**protocol["data"], "split": "train"},
        "material_ids": protocol["material_ids"],
    }
    raw_batch = _load_panel(
        parent_protocol, ROOT, device,
        train_csv=_resolve(protocol["data"]["train_csv"]),
        split_manifest=_resolve(protocol["data"]["split_manifest"]),
        target_cache_dir=_resolve(protocol["data"]["target_cache_dir"]),
        preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]),
    )
    evaluation = protocol["s1"]["evaluation"]
    criterion = protocol["s1"]["pass_criteria"]
    metric_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    component_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    cfg_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}

    for variant in protocol["s1"]["variants"]:
        for step in protocol["s1"]["training"]["checkpoint_steps"]:
            checkpoint = root / variant["id"] / f"step_{step:04d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model, scales, checkpoint_mode = _load_model(checkpoint, device)
            if checkpoint_mode != "direct_irrep":
                raise ValueError(f"{checkpoint} is not a direct_irrep checkpoint")
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload["config"].get("conditional_control") != variant["conditional_control"]:
                raise ValueError(f"Conditional-control variant mismatch in {checkpoint}")
            batch = raw_batch.clone()
            batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales)
            flow = _flow_gap(model, batch, evaluation["loss_repeats"], evaluation["seed"])
            generated = _generated_distribution(
                model, batch.piezo_irreps, evaluation["sample_replicates"], evaluation["sample_atoms"],
                evaluation["sample_steps"], evaluation["seed"], device,
            )
            ranking_records, _ = _teacher_forced_ranking(
                model, batch, evaluation["ranking_times"], evaluation["seed"], variant["id"]
            )
            ranking = pd.DataFrame(ranking_records)
            ranking["checkpoint_step"] = step
            ranking_rows.extend(ranking.to_dict("records"))
            own_win_rate = float(ranking.own_target_win.mean())
            mean_margin = float(ranking.own_target_margin.mean())
            mean_own_error = float(ranking.own_velocity_error.mean())
            mean_wrong_error = float(ranking.shuffled_velocity_error.mean())
            trajectory, _, _, _ = _trajectory(
                model, batch.piezo_irreps, variant["id"], steps=evaluation["trajectory_steps"],
                atoms=evaluation["sample_atoms"], seed=evaluation["seed"] + 60_000,
                device=device, capture_posterior=False,
            )
            trajectory_frame = pd.DataFrame(trajectory)
            trajectory_frame["checkpoint_step"] = step
            state_frames.append(trajectory_frame)
            terminal, terminal_pass = _terminal_state_summary(trajectory, variant["id"])
            terminal["checkpoint_step"] = step
            terminal.to_csv(output_dir / f"{variant['id']}_step_{step:04d}_terminal_state.csv", index=False)
            components = _component_norms(model, batch.piezo_irreps, evaluation, variant["id"], device)
            components["checkpoint_step"] = step
            component_frames.append(components)
            main_sample = _guided_distribution(model, batch.piezo_irreps, evaluation, 0.0, device)
            for cfg_scale in variant["cfg_evaluation_scales"]:
                sample = main_sample if cfg_scale == 0.0 else _guided_distribution(
                    model, batch.piezo_irreps, evaluation, cfg_scale, device
                )
                cfg_rows.append({
                    "variant": variant["id"], "checkpoint_step": step, "cfg_scale": cfg_scale,
                    "analysis": "main" if cfg_scale == 0.0 else "pre_registered_dropout_only_supplement",
                    **sample,
                })
            residual_heads_recorded = set(components["head"]) == {
                "type_logit", "fractional_coordinate", "lattice_log"
            }
            passed = all((
                generated["between_within_distance_ratio"] >= criterion["generated_between_within_ratio_min"],
                own_win_rate >= criterion["own_target_win_rate_min"],
                mean_margin > criterion["mean_own_target_margin_min_exclusive"],
                terminal_pass,
                mean_own_error <= mean_wrong_error,
                main_sample["sampling_failure_count"] <= criterion["sampling_failure_count_max"],
                residual_heads_recorded,
            ))
            metric_rows.append({
                "variant": variant["id"],
                "checkpoint_step": step,
                "conditional_control": variant["conditional_control"],
                "condition_dropout": variant["condition_dropout"],
                "counterfactual_weight": variant["counterfactual_weight"],
                "final_training_loss": payload.get("last_terms", {}).get("loss", math.nan),
                "counterfactual_training_loss": payload.get("last_terms", {}).get("counterfactual", 0.0),
                "condition_shuffle_gap": flow["shuffled_flow_gap_fraction_median"],
                "generated_between_within_ratio": generated["between_within_distance_ratio"],
                "own_target_win_rate": own_win_rate,
                "mean_own_target_margin": mean_margin,
                "mean_own_flow_error": mean_own_error,
                "mean_wrong_condition_flow_error": mean_wrong_error,
                "own_not_worse_than_wrong": mean_own_error <= mean_wrong_error,
                "common_noise_terminal_pass": terminal_pass,
                "sampling_failure_count_cfg0": main_sample["sampling_failure_count"],
                "all_residual_heads_recorded": residual_heads_recorded,
                "s1_pass": passed,
                "checkpoint_sha256": _sha256_file(checkpoint),
            })
            checkpoint_hashes[f"{variant['id']}/step_{step:04d}"] = _sha256_file(checkpoint)

    metrics = pd.DataFrame(metric_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    component_frame = pd.concat(component_frames, ignore_index=True)
    state_frame = pd.concat(state_frames, ignore_index=True)
    cfg_frame = pd.DataFrame(cfg_rows)
    metrics.to_csv(output_dir / "s1_metrics.csv", index=False)
    ranking_frame.to_csv(output_dir / "teacher_forced_ranking.csv", index=False)
    component_frame.to_csv(output_dir / "velocity_component_curves.csv", index=False)
    state_frame.to_csv(output_dir / "common_noise_trajectory.csv", index=False)
    cfg_frame.to_csv(output_dir / "cfg_supplement.csv", index=False)
    main_800 = metrics[metrics.checkpoint_step == max(protocol["s1"]["training"]["checkpoint_steps"])]
    candidates = main_800[main_800.s1_pass]
    status = "s1_direct_irrep_passed_s2_still_not_run" if not candidates.empty else "s1_direct_irrep_not_passed"
    report = f"""# Gate A2 S1 conditional-control screen

## Status

`{status}`.  This report evaluates only the pre-registered direct-irrep S1
mechanism screen at 400 and 800 steps.  It does not change Gate A v1, activate
v2, run S2, or claim Gate A passage.

## Fixed-combination results

{metrics.round(5).to_markdown(index=False)}

## Interpretation boundary

The main sampling result is always CFG=0.  CFG=1 appears only for the
graphwise-dropout variant as a pre-registered supplement, never as a
replacement result.  `velocity_component_curves.csv` records base velocity,
conditional residual, g(t)-weighted residual, and residual/base ratio for all
three heads.  `common_noise_trajectory.csv` records the matched-noise terminal
state path; `teacher_forced_ranking.csv` retains every target/time comparison.

S2 remains locked.  It may be started only by a separate command after a
direct-irrep S1 result passes every registered criterion and the selected
conditional-control backbone/loss is explicitly frozen.
"""
    (output_dir / "gate_a2_s1_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": status,
        "protocol_sha256": _sha256_file(protocol_path),
        "checkpoint_sha256": checkpoint_hashes,
        "main_cfg": 0.0,
        "s2_started": False,
        "gate_a_v1_modified": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
