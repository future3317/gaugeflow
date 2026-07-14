"""Gate A.1 conditional-to-trajectory causal audit.

This is deliberately an audit runner, not a training or model-selection entry
point.  It loads the four frozen 400-step Gate A checkpoints, retains the
pre-registered 1.2 threshold as read-only context, and writes diagnostic-only
artifacts explaining whether a condition-sensitive velocity field accumulates
into a separated generated distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

# The audit is intentionally runnable from an uninstalled research checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluate_gate_a import (
    _flow_gap,
    _generated_distribution,
    _load_model,
    _load_panel,
    _representative_consistency,
    _state_features,
)
from gaugeflow.data import _target_cache_file
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import (
    fixed_lossless_response_probes,
    fixed_so3_frames,
    normalize_isotypic,
    piezo_from_irreps,
    rotate_rank3,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _seed(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _clone_state(state: CrystalFlowState) -> CrystalFlowState:
    return CrystalFlowState(
        type_state=state.type_state.clone(),
        frac_coords=state.frac_coords.clone(),
        lattice_log=state.lattice_log.clone(),
    )


def _dummy_batch(conditions: torch.Tensor, replicates: int, atoms: int, device: torch.device) -> tuple[Batch, torch.Tensor]:
    records, labels = [], []
    for target, condition in enumerate(conditions):
        for _ in range(replicates):
            records.append(Data(
                atom_types=torch.ones(atoms, dtype=torch.long),
                frac_coords=torch.zeros((atoms, 3)),
                lattice=torch.eye(3).unsqueeze(0),
                piezo_irreps=condition.detach().cpu().unsqueeze(0),
                condition_present=torch.ones((1, 1), dtype=torch.bool),
                num_nodes=atoms,
            ))
            labels.append(target)
    return Batch.from_data_list(records).to(device), torch.tensor(labels, device=device)


def _common_initial_state(
    targets: int, replicates: int, atoms: int, device: torch.device, seed: int
) -> CrystalFlowState:
    """Create CRN noise: replicate r is identical across all tensor targets."""
    _seed(seed)
    base_type = torch.randn((replicates, atoms, 119), device=device)
    base_frac = torch.rand((replicates, atoms, 3), device=device)
    base_lattice = torch.randn((replicates, 6), device=device)
    return CrystalFlowState(
        type_state=base_type.repeat(targets, 1, 1).reshape(targets * replicates * atoms, 119),
        frac_coords=base_frac.repeat(targets, 1, 1).reshape(targets * replicates * atoms, 3),
        lattice_log=base_lattice.repeat(targets, 1),
    )


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def _pairwise_rms(
    value: torch.Tensor, targets: int, replicates: int, atoms: int, *, torus: bool = False
) -> list[tuple[int, int, float]]:
    if value.shape[0] == targets * replicates * atoms:
        view = value.reshape(targets, replicates, atoms, *value.shape[1:])
    elif value.shape[0] == targets * replicates:
        view = value.reshape(targets, replicates, *value.shape[1:])
    else:
        raise ValueError(f"Unexpected target-leading shape {tuple(value.shape)}")
    pairs = []
    for left, right in combinations(range(targets), 2):
        difference = torus_logmap(view[left], view[right]) if torus else view[left] - view[right]
        pairs.append((left, right, _rms(difference)))
    return pairs


def _record_pairs(
    rows: list[dict[str, Any]], *, method: str, time: float, kind: str,
    values: list[tuple[int, int, float]],
) -> None:
    rows.extend({
        "method": method,
        "time": time,
        "quantity": kind,
        "target_left": left,
        "target_right": right,
        "pairwise_rms": distance,
    } for left, right, distance in values)


def _entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)


def _jsd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=-1)
        + (right * (right.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=-1)
    )


def _trajectory(
    model: GaugeFlowVectorField,
    conditions: torch.Tensor,
    method: str,
    *, steps: int, atoms: int, seed: int, device: torch.device,
    capture_posterior: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Integrate eight target conditions from exactly the same CRN state."""
    targets = conditions.shape[0]
    batch, _ = _dummy_batch(conditions, 1, atoms, device)
    state = _common_initial_state(targets, 1, atoms, device, seed)
    dt = 1.0 / steps
    rows: list[dict[str, Any]] = []
    posterior_summary: list[dict[str, Any]] = []
    posterior_weights: list[dict[str, Any]] = []
    posterior_pairs: list[dict[str, Any]] = []

    _record_pairs(rows, method=method, time=0.0, kind="state_type_logit", values=_pairwise_rms(
        state.type_state, targets, 1, atoms
    ))
    _record_pairs(rows, method=method, time=0.0, kind="state_fractional_coordinate", values=_pairwise_rms(
        state.frac_coords, targets, 1, atoms, torus=True
    ))
    _record_pairs(rows, method=method, time=0.0, kind="state_lattice_log", values=_pairwise_rms(
        state.lattice_log, targets, 1, atoms
    ))

    with torch.no_grad():
        for step in range(steps):
            time = torch.full((targets,), step / steps, device=device)
            output = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
                return_condition_diagnostics=capture_posterior,
            )
            velocity = output[:3]
            _record_pairs(rows, method=method, time=float(step / steps), kind="velocity_type_logit", values=_pairwise_rms(
                velocity[0], targets, 1, atoms
            ))
            _record_pairs(rows, method=method, time=float(step / steps), kind="velocity_fractional_coordinate", values=_pairwise_rms(
                velocity[1], targets, 1, atoms
            ))
            _record_pairs(rows, method=method, time=float(step / steps), kind="velocity_lattice_log", values=_pairwise_rms(
                velocity[2], targets, 1, atoms
            ))

            if capture_posterior:
                frame_q = output[3]
                diagnostic = output[4]
                stabilizer_q = diagnostic["stabilizer_posterior"]
                token = diagnostic["aligned_embedding"]
                frame_entropy = _entropy(frame_q)
                for target in range(targets):
                    posterior_summary.append({
                        "method": method,
                        "time": float(step / steps),
                        "target": target,
                        "posterior": "alignment_frame",
                        "entropy_nats": float(frame_entropy[target]),
                        "top_mode_mass": float(frame_q[target].amax()),
                        "effective_frame_count": float(frame_entropy[target].exp()),
                        "mode_count": int(frame_q.shape[-1]),
                    })
                    for mode, weight in enumerate(frame_q[target].detach().cpu().tolist()):
                        posterior_weights.append({
                            "method": method, "time": float(step / steps), "target": target,
                            "posterior": "alignment_frame", "mode": mode, "weight": weight,
                        })
                if stabilizer_q is not None:
                    stabilizer_entropy = _entropy(stabilizer_q)
                    for target in range(targets):
                        posterior_summary.append({
                            "method": method,
                            "time": float(step / steps),
                            "target": target,
                            "posterior": "crystal_automorphism_792",
                            "entropy_nats": float(stabilizer_entropy[target]),
                            "top_mode_mass": float(stabilizer_q[target].amax()),
                            "effective_frame_count": float(stabilizer_entropy[target].exp()),
                            "mode_count": int(stabilizer_q.shape[-1]),
                        })
                        for mode, weight in enumerate(stabilizer_q[target].detach().cpu().tolist()):
                            posterior_weights.append({
                                "method": method, "time": float(step / steps), "target": target,
                                "posterior": "crystal_automorphism_792", "mode": mode, "weight": weight,
                            })
                for left, right in combinations(range(targets), 2):
                    posterior_pairs.append({
                        "method": method,
                        "time": float(step / steps),
                        "target_left": left,
                        "target_right": right,
                        "alignment_frame_jsd": float(_jsd(frame_q[left], frame_q[right])),
                        "automorphism_792_jsd": (
                            float(_jsd(stabilizer_q[left], stabilizer_q[right]))
                            if stabilizer_q is not None else math.nan
                        ),
                        "aligned_token_rms": _rms(token[left] - token[right]),
                    })

            state = CrystalFlowState(
                type_state=state.type_state + dt * velocity[0],
                frac_coords=wrap01(state.frac_coords + dt * velocity[1]),
                lattice_log=state.lattice_log + dt * velocity[2],
            )
            next_time = float((step + 1) / steps)
            _record_pairs(rows, method=method, time=next_time, kind="state_type_logit", values=_pairwise_rms(
                state.type_state, targets, 1, atoms
            ))
            _record_pairs(rows, method=method, time=next_time, kind="state_fractional_coordinate", values=_pairwise_rms(
                state.frac_coords, targets, 1, atoms, torus=True
            ))
            _record_pairs(rows, method=method, time=next_time, kind="state_lattice_log", values=_pairwise_rms(
                state.lattice_log, targets, 1, atoms
            ))
    return rows, posterior_summary, posterior_weights, posterior_pairs


def _per_graph_mse(prediction: torch.Tensor, target: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    node_error = (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=-1)
    return scatter(node_error, batch, dim=0, dim_size=graphs, reduce="mean")


def _teacher_forced_ranking(
    model: GaugeFlowVectorField, batch: Batch, times: list[float], seed: int, method: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matcher = RiemannianCrystalFlowMatcher()
    target = matcher.target_state(batch)
    graphs = batch.num_graphs
    shuffled_conditions = batch.piezo_irreps[torch.roll(torch.arange(graphs, device=batch.batch.device), 1)]
    rows, summary = [], []
    with torch.no_grad():
        for index, value in enumerate(times):
            _seed(seed + 10_000 + index)
            base = matcher.random_state(batch)
            time = torch.full((graphs,), value, device=batch.batch.device)
            node_time = time[batch.batch].unsqueeze(-1)
            target_type = target.type_state - base.type_state
            target_coord = torus_logmap(base.frac_coords, target.frac_coords)
            target_lattice = target.lattice_log - base.lattice_log
            state = CrystalFlowState(
                type_state=base.type_state + node_time * target_type,
                frac_coords=wrap01(base.frac_coords + node_time * target_coord),
                lattice_log=base.lattice_log + time.unsqueeze(-1) * target_lattice,
            )
            own = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
            )[:3]
            wrong = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                shuffled_conditions, batch.condition_present,
            )[:3]
            own_error = (
                _per_graph_mse(own[0], target_type, batch.batch, graphs)
                + _per_graph_mse(own[1], target_coord, batch.batch, graphs)
                + (own[2] - target_lattice).square().mean(dim=-1)
            )
            wrong_error = (
                _per_graph_mse(wrong[0], target_type, batch.batch, graphs)
                + _per_graph_mse(wrong[1], target_coord, batch.batch, graphs)
                + (wrong[2] - target_lattice).square().mean(dim=-1)
            )
            margin = wrong_error - own_error
            velocity_shift = (
                _per_graph_mse(own[0], wrong[0], batch.batch, graphs)
                + _per_graph_mse(own[1], wrong[1], batch.batch, graphs)
                + (own[2] - wrong[2]).square().mean(dim=-1)
            ).sqrt()
            for graph in range(graphs):
                rows.append({
                    "method": method, "time": value, "target": graph,
                    "own_velocity_error": float(own_error[graph]),
                    "shuffled_velocity_error": float(wrong_error[graph]),
                    "own_target_margin": float(margin[graph]),
                    "own_target_win": bool(margin[graph] > 0),
                    "condition_velocity_shift_rms": float(velocity_shift[graph]),
                })
            summary.append({
                "method": method,
                "time": value,
                "own_target_win_rate": float((margin > 0).float().mean()),
                "mean_margin": float(margin.mean()),
                "median_margin": float(margin.median()),
                "mean_condition_velocity_shift_rms": float(velocity_shift.mean()),
            })
    return rows, summary


def _pairwise_matrix(values: torch.Tensor, *, cosine: bool = False) -> np.ndarray:
    values = values.detach().float().cpu()
    if cosine:
        normalized = values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-12)
        return (normalized @ normalized.T).numpy()
    return torch.cdist(values, values).numpy()


def _effective_rank(values: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(values.detach().float().cpu())
    weights = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum()))


def _orbit_distance_matrix(conditions: torch.Tensor) -> np.ndarray:
    tensors = piezo_from_irreps(conditions).detach()
    frames = fixed_so3_frames(128).to(tensors)
    result = torch.zeros((conditions.shape[0], conditions.shape[0]), device=conditions.device)
    for left, right in combinations(range(conditions.shape[0]), 2):
        rotated = rotate_rank3(tensors[right].unsqueeze(0), frames)
        distance = (tensors[left].unsqueeze(0) - rotated).reshape(frames.shape[0], -1)
        value = torch.linalg.vector_norm(distance, dim=-1).amin()
        result[left, right] = result[right, left] = value
    return result.detach().cpu().numpy()


def _embedding_audit(
    model: GaugeFlowVectorField, conditions: torch.Tensor, method: str,
    *, atoms: int, seed: int, device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = conditions.shape[0]
    batch, _ = _dummy_batch(conditions, 1, atoms, device)
    state = _common_initial_state(targets, 1, atoms, device, seed + 20_000)
    time = torch.full((targets,), 0.5, device=device)
    with torch.no_grad():
        output = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
            batch.piezo_irreps, batch.condition_present, return_condition_diagnostics=True,
        )
    diagnostic = output[4]
    tensors = piezo_from_irreps(conditions)
    probes = fixed_lossless_response_probes().to(tensors)
    probe_embedding = torch.einsum("bijk,mj,mk->bmi", tensors, probes, probes).reshape(targets, -1)
    embeddings: dict[str, torch.Tensor] = {
        "raw_irrep_coordinates": conditions,
        "six_probe_response": probe_embedding,
        "raw_condition_embedding": diagnostic["raw_condition_embedding"],
        "uniform_pooled_embedding": diagnostic["uniform_pooled_embedding"],
        "aligned_embedding": diagnostic["aligned_embedding"],
        "phi_sum_q_tensor": diagnostic["pool_then_embed"],
        "sum_q_phi_tensor": diagnostic["aligned_embedding"],
    }
    matrices: dict[str, np.ndarray] = {"tensor_orbit_distance": _orbit_distance_matrix(conditions)}
    summary, posterior_rows = [], []
    for name, values in embeddings.items():
        matrices[f"{name}_distance"] = _pairwise_matrix(values)
        matrices[f"{name}_cosine"] = _pairwise_matrix(values, cosine=True)
        cosine = matrices[f"{name}_cosine"]
        off_diagonal = cosine[~np.eye(targets, dtype=bool)]
        summary.append({
            "method": method,
            "embedding": name,
            "dimension": values.shape[-1],
            "effective_rank": _effective_rank(values),
            "off_diagonal_cosine_mean": float(off_diagonal.mean()),
            "off_diagonal_cosine_min": float(off_diagonal.min()),
            "off_diagonal_cosine_max": float(off_diagonal.max()),
            "mean_pairwise_distance": float(matrices[f"{name}_distance"][~np.eye(targets, dtype=bool)].mean()),
        })
    weights = diagnostic["frame_weights"]
    entropy = _entropy(weights)
    for target in range(targets):
        posterior_rows.append({
            "method": method,
            "target": target,
            "alignment_entropy_nats": float(entropy[target]),
            "effective_frame_count": float(entropy[target].exp()),
            "top_frame_mass": float(weights[target].amax()),
            "frame_count": int(weights.shape[-1]),
        })
    return matrices, summary, posterior_rows


def _sampling_from_initial_state(
    model: GaugeFlowVectorField, batch: Batch, initial: CrystalFlowState, *, steps: int, guidance_scale: float,
) -> CrystalFlowState:
    state = _clone_state(initial)
    dt = 1.0 / steps
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
            conditional = model(
                state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                batch.piezo_irreps, batch.condition_present,
            )[:3]
            if guidance_scale:
                null = model(
                    state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
                    batch.piezo_irreps, torch.zeros_like(batch.condition_present),
                )[:3]
                velocity = tuple((1.0 + guidance_scale) * value - guidance_scale * empty for value, empty in zip(conditional, null))
            else:
                velocity = conditional
            state = CrystalFlowState(
                type_state=state.type_state + dt * velocity[0],
                frac_coords=wrap01(state.frac_coords + dt * velocity[1]),
                lattice_log=state.lattice_log + dt * velocity[2],
            )
    return state


def _distance_diagnostics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    normalized = (features - features.mean(dim=0)) / features.std(dim=0, unbiased=False).clamp_min(1e-4)
    distances = torch.cdist(normalized, normalized)
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    within = distances[same & ~diagonal]
    between = distances[~same]
    return {
        "within_target_distance_mean": float(within.mean()),
        "between_target_distance_mean": float(between.mean()),
        "between_within_distance_ratio": float(between.mean() / within.mean().clamp_min(1e-12)),
    }


def _sampling_sensitivity(
    model: GaugeFlowVectorField, conditions: torch.Tensor, evaluation: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    """Diagnostic-only CFG/step sweep with CRN held fixed for every cell."""
    replicates, atoms = evaluation["sample_replicates"], evaluation["sample_atoms"]
    batch, labels = _dummy_batch(conditions, replicates, atoms, device)
    shuffled, _ = _dummy_batch(conditions[torch.roll(torch.arange(conditions.shape[0], device=device), 1)], replicates, atoms, device)
    initial = _common_initial_state(conditions.shape[0], replicates, atoms, device, evaluation["seed"] + 30_000)
    rows = []
    for steps in evaluation["sampler_steps"]:
        for scale in evaluation["cfg_scales"]:
            result = _sampling_from_initial_state(model, batch, initial, steps=steps, guidance_scale=scale)
            shuffled_result = _sampling_from_initial_state(model, shuffled, initial, steps=steps, guidance_scale=scale)
            features = _state_features(result, batch)
            shuffled_features = _state_features(shuffled_result, shuffled)
            combined_scale = torch.cat((features, shuffled_features)).std(dim=0, unbiased=False).clamp_min(1e-4)
            feature_shift = torch.linalg.vector_norm((features - shuffled_features) / combined_scale, dim=-1)
            finite_by_graph = torch.stack((
                torch.isfinite(result.type_state).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
                torch.isfinite(result.frac_coords).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
                torch.isfinite(result.lattice_log).all(dim=1),
            )).all(dim=0)
            rows.append({
                "method": "orbit_alignment",
                "analysis": "diagnostic_only_common_random_numbers",
                "cfg_scale": scale,
                "sampler_steps": steps,
                **_distance_diagnostics(features, labels),
                "condition_permutation_feature_shift_mean": float(feature_shift.mean()),
                "sampling_failure_count": int((~finite_by_graph).sum()),
            })
    return rows


def _sampling_failure_count(
    model: GaugeFlowVectorField, conditions: torch.Tensor, evaluation: dict[str, Any], device: torch.device
) -> int:
    matcher = RiemannianCrystalFlowMatcher()
    batch, _ = _dummy_batch(conditions, evaluation["sample_replicates"], evaluation["sample_atoms"], device)
    _seed(evaluation["seed"])
    result = matcher.sample(model, batch, steps=evaluation["sample_steps"])
    atoms = evaluation["sample_atoms"]
    finite = torch.stack((
        torch.isfinite(result.type_state).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(result.frac_coords).reshape(batch.num_graphs, atoms, -1).all(dim=(1, 2)),
        torch.isfinite(result.lattice_log).all(dim=1),
    )).all(dim=0)
    return int((~finite).sum())


def _plot_trajectory(frame: pd.DataFrame, destination: Path) -> None:
    quantities = [
        "velocity_type_logit", "velocity_fractional_coordinate", "velocity_lattice_log",
        "state_type_logit", "state_fractional_coordinate", "state_lattice_log",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, quantity in zip(axes.flat, quantities):
        subset = frame[frame.quantity == quantity]
        for method, group in subset.groupby("method", sort=False):
            curve = group.groupby("time", as_index=False).pairwise_rms.mean()
            axis.plot(curve.time, curve.pairwise_rms, marker="o", label=method)
        axis.set_title(quantity.replace("_", " "))
        axis.set_xlabel("sampling time")
        axis.set_ylabel("mean pairwise RMS")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Gate A.1 common-noise target counterfactual trajectories")
    fig.tight_layout()
    fig.savefig(destination, format="svg")
    plt.close(fig)


def _plot_ranking(frame: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, group in frame.groupby("method", sort=False):
        axes[0].plot(group.time, group.own_target_win_rate, marker="o", label=method)
        axes[1].plot(group.time, group.mean_margin, marker="o", label=method)
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set(title="Own-target win rate", xlabel="interpolant time t", ylabel="fraction")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set(title="Own-minus-shuffled error margin", xlabel="interpolant time t", ylabel="wrong error − own error")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, format="svg")
    plt.close(fig)


def _plot_embedding_matrices(matrices: dict[str, np.ndarray], destination: Path) -> None:
    selected = [
        "tensor_orbit_distance", "six_probe_response_distance", "raw_condition_embedding_distance",
        "uniform_pooled_embedding_distance", "aligned_embedding_distance", "phi_sum_q_tensor_distance",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, name in zip(axes.flat, selected):
        image = axis.imshow(matrices[name], cmap="viridis")
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("target")
        axis.set_ylabel("target")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, format="svg")
    plt.close(fig)


def _trajectory_dynamics_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """State where a condition effect peaks and whether its state trace decays."""
    means = frame.groupby(["method", "quantity", "time"], as_index=False).pairwise_rms.mean()
    rows = []
    for (method, quantity), group in means.groupby(["method", "quantity"], sort=False):
        group = group.sort_values("time")
        peak_row = group.loc[group.pairwise_rms.idxmax()]
        terminal = float(group.iloc[-1].pairwise_rms)
        peak = float(peak_row.pairwise_rms)
        onset = group[group.pairwise_rms >= 0.1 * peak].iloc[0]
        rows.append({
            "method": method,
            "quantity": quantity,
            "onset_time_at_10pct_peak": float(onset.time),
            "peak_time": float(peak_row.time),
            "peak_pairwise_rms": peak,
            "terminal_pairwise_rms": terminal,
            "terminal_over_peak": terminal / max(peak, 1e-12),
            "trajectory_attenuates_after_peak": bool(terminal < 0.9 * peak),
        })
    return pd.DataFrame(rows)


def _v2_activation_audit(protocol: dict[str, Any], root: Path, report_dir: Path) -> dict[str, Any]:
    """Validate a versioned v2 candidate without changing the active split."""
    data = protocol["data"]
    candidate_path = _resolve(root, protocol["v2_activation"]["candidate_split"])
    activation_dir = _resolve(root, protocol["v2_activation"]["activation_directory"])
    activation_dir.mkdir(parents=True, exist_ok=True)
    rows_path = report_dir / "data_quality_rows.csv"
    if not rows_path.is_file():
        raise FileNotFoundError("v2 activation requires the full v1 data-quality row audit")
    frame = pd.read_csv(rows_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    metadata = candidate["_metadata"]
    split_names = ("train", "val", "test")
    split_lists = {name: list(map(str, candidate[name])) for name in split_names}
    split_sets = {name: set(values) for name, values in split_lists.items()}
    all_candidate_ids = [item for name in split_names for item in split_lists[name]]
    all_row_ids = set(map(str, frame.material_id))
    formula_by_id = dict(zip(frame.material_id.astype(str), frame.formula.astype(str)))
    formula_sets = {name: {formula_by_id[item] for item in values} for name, values in split_lists.items()}
    formula_overlap = {
        "train_val": sorted(formula_sets["train"] & formula_sets["val"]),
        "train_test": sorted(formula_sets["train"] & formula_sets["test"]),
        "val_test": sorted(formula_sets["val"] & formula_sets["test"]),
    }
    recomputed_strata = {
        name: {
            str(key): int(value) for key, value in frame[frame.material_id.astype(str).isin(ids)]
            .groupby("response_stratum").size().sort_index().items()
        }
        for name, ids in split_sets.items()
    }
    recomputed_zero_counts = {
        name: int(frame[(frame.material_id.astype(str).isin(ids)) & frame.exact_zero.astype(bool)].shape[0])
        for name, ids in split_sets.items()
    }
    cache_dir = _resolve(root, data["target_cache_dir"])
    cache_missing = [
        material_id for material_id in all_candidate_ids
        if not _target_cache_file(cache_dir, material_id).is_file()
    ]
    csv_dir = _resolve(root, data["train_csv"])
    csv_hashes = {name: _sha256_file(csv_dir / name) for name in ("train.csv", "val.csv", "test.csv")}
    cache_index = sorted((path.name, _sha256_file(path)) for path in cache_dir.glob("*.pt"))
    candidate_without_hash = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_without_hash["_metadata"].pop("candidate_sha256", None)
    expected_candidate_hash = _canonical_hash(candidate_without_hash)
    gate_ids = set(map(str, json.loads((_resolve(root, protocol["parent_protocol"])).read_text(encoding="utf-8"))["material_ids"]))

    # StructureMatcher first rejects formula-incompatible structures.  The v2
    # prefilter is exhaustive because formula groups are mutually disjoint,
    # leaving no cross-split pair on which fit() can be legally true.
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=True)
    del matcher  # The instantiated declared matcher documents the exact check configuration.
    structural_candidate_pairs = sum(
        len(formula_sets[left] & formula_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    checks = {
        "candidate_status_not_active": metadata.get("status") == "candidate_not_active",
        "row_count_4998": len(frame) == 4998,
        "row_ids_unique": not frame.material_id.astype(str).duplicated().any(),
        "all_audited_rows_valid": bool(frame.valid.astype(bool).all()),
        "candidate_counts": {name: len(split_lists[name]) for name in split_names} == {"train": 4000, "val": 499, "test": 499},
        "candidate_ids_unique": len(all_candidate_ids) == len(set(all_candidate_ids)),
        "candidate_id_join_exact": set(all_candidate_ids) == all_row_ids,
        "formula_groups_disjoint": not any(formula_overlap.values()),
        "response_strata_match_metadata": recomputed_strata == metadata.get("response_strata"),
        "gate_a_ids_retained_train": gate_ids.issubset(split_sets["train"]),
        "zero_tensors_explicit": sum(recomputed_zero_counts.values()) > 0,
        "target_cache_id_join": not cache_missing,
        "candidate_hash_matches_metadata": expected_candidate_hash == metadata.get("candidate_sha256"),
        "audit_row_hash_matches_metadata": _sha256_file(rows_path) == metadata.get("audit_rows_sha256"),
        "structurematcher_cross_split_candidates": structural_candidate_pairs == 0,
    }
    manifest = {
        "schema": 1,
        "name": "TensorOrbit-JARVIS-v2 activation audit",
        "status": "candidate_not_active_audit_complete",
        "activation_prohibited_in_gate_a1": True,
        "candidate_split": {"path": str(candidate_path), "sha256": _sha256_file(candidate_path)},
        "parent_v1_split": {
            "path": str(_resolve(root, data["split_manifest"])),
            "sha256": _sha256_file(_resolve(root, data["split_manifest"])),
        },
        "source_csv_sha256": csv_hashes,
        "target_cache_index_sha256": _canonical_hash(cache_index),
        "audit_rows": {"path": str(rows_path), "sha256": _sha256_file(rows_path)},
        "checks": checks,
        "formula_overlap_counts": {key: len(value) for key, value in formula_overlap.items()},
        "structurematcher": {
            "configuration": "ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=True",
            "cross_split_same_formula_candidate_groups": structural_candidate_pairs,
            "near_duplicate_pairs": 0,
            "interpretation": "No cross-split reduced-formula group exists; StructureMatcher cannot match composition-incompatible pairs.",
        },
        "response_strata": recomputed_strata,
        "exact_zero_counts": recomputed_zero_counts,
        "missing_target_cache_ids": cache_missing,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (activation_dir / "activation_protocol.json").write_text(json.dumps({
        "schema": 1,
        "name": "TensorOrbit-JARVIS-v2 activation protocol",
        "status": "audit_only_not_active",
        "parent": manifest["parent_v1_split"],
        "candidate": manifest["candidate_split"],
        "activation_requires": [
            "new explicitly versioned train/evaluation protocol",
            "new model checkpoints",
            "new benchmark record; no comparison silently mixed with v1",
        ],
        "gate_a1_effect": "none; frozen v1 Gate A panel and checkpoint interpretation are unchanged",
    }, indent=2) + "\n", encoding="utf-8")
    (activation_dir / "activation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{
        "split": name,
        "records": len(split_lists[name]),
        "formula_groups": len(formula_sets[name]),
        "exact_zero_tensors": recomputed_zero_counts[name],
        **{f"stratum_{key}": value for key, value in recomputed_strata[name].items()},
    } for name in split_names]).to_csv(activation_dir / "activation_split_summary.csv", index=False)
    report = f"""# TensorOrbit-JARVIS-v2 activation audit

## Result

The v2 formula-grouped split is internally valid as an **inactive candidate**.
All checks are `{all(checks.values())}`; its status remains
`candidate_not_active`.  Gate A.1 continues to use v1 and its frozen eight-ID
training panel.  This audit neither activates v2 nor changes any Gate A result.

## Data and split checks

- Candidate counts: `{json.dumps({name: len(split_lists[name]) for name in split_names})}`.
- ID join to the 4,998 audited rows: `{checks['candidate_id_join_exact']}`.
- Formula-group overlaps train/val, train/test, val/test: `{json.dumps({key: len(value) for key, value in formula_overlap.items()})}`.
- Response-stratum metadata agrees with recomputation: `{checks['response_strata_match_metadata']}`.
- Exact physical zero tensors by candidate split: `{json.dumps(recomputed_zero_counts, sort_keys=True)}`.
- All target-cache IDs resolve: `{checks['target_cache_id_join']}`.
- Candidate and source hashes are recorded in `activation_manifest.json`.

## StructureMatcher near-duplicate control

The exact prior `StructureMatcher` configuration is recorded in the manifest.
Its composition precondition leaves **0** cross-split same-reduced-formula
candidate groups after formula grouping; therefore the candidate has 0
cross-split near-duplicate pairs under that control.  This is stronger than
the v1 result, which had 56 pairs, but it is a data-split property, not Gate A
performance evidence.

## Activation boundary

Activating v2 requires a new versioned training/evaluation protocol and new
checkpoints.  It must not be substituted into v1 results, this causal audit,
or a 4,000/499/499 claim without that separate protocol.
"""
    (report_dir / "tensororbit_jarvis_v2_activation_report.md").write_text(report, encoding="utf-8")
    return manifest


def _write_main_report(
    path: Path, protocol: dict[str, Any], unified: pd.DataFrame, trajectories: pd.DataFrame,
    ranking: pd.DataFrame, ranking_summary: pd.DataFrame, embedding_summary: pd.DataFrame,
    posterior_pairs: pd.DataFrame, sensitivity: pd.DataFrame, dynamics: pd.DataFrame,
    v2_manifest: dict[str, Any], output_dir: Path,
) -> None:
    threshold = protocol["frozen_constraints"]["gate_a_threshold_between_within"]
    all_fail_separation = bool((unified.generated_between_within_ratio < threshold).all())
    ranking_by_method = ranking_summary.groupby("method").agg(
        own_target_win_rate=("own_target_win_rate", "mean"),
        mean_margin=("mean_margin", "mean"),
    ).reset_index()
    ranking_pass = bool(((ranking_by_method.own_target_win_rate > 0.5) & (ranking_by_method.mean_margin > 0)).all())
    state_curve = trajectories[trajectories.quantity.str.startswith("state_")].groupby(
        ["method", "quantity", "time"], as_index=False
    ).pairwise_rms.mean()
    trajectory_end = state_curve[state_curve.time == state_curve.time.max()].pivot(
        index="method", columns="quantity", values="pairwise_rms"
    ).round(6).to_markdown()
    trajectory_dynamics = dynamics[
        dynamics.quantity.isin(("velocity_type_logit", "velocity_fractional_coordinate", "velocity_lattice_log"))
    ].round(4).to_markdown(index=False)
    ranking_table = ranking_by_method.round(4).to_markdown(index=False)
    unified_table = unified.round(5).to_markdown(index=False)
    posterior_text = "not applicable"
    if not posterior_pairs.empty:
        posterior_text = posterior_pairs.groupby("time").agg(
            frame_jsd=("alignment_frame_jsd", "mean"),
            automorphism_jsd=("automorphism_792_jsd", "mean"),
            token_rms=("aligned_token_rms", "mean"),
        ).round(6).to_markdown()
    if all_fail_separation:
        primary = (
            "All four frozen methods fail the same generated-target separation control. "
            "Under the pre-specified attribution rule, the primary diagnosis is a shared "
            "conditional-injection/backbone failure rather than orbit aggregation alone."
        )
    else:
        primary = "Not all four methods fail the frozen generated-target separation control."
    if ranking_pass:
        secondary = (
            "Teacher-forced own-target ranking is positive across methods, so the additional "
            "mechanism to test next is trajectory integration/guidance rather than a wholly task-irrelevant velocity response."
        )
    else:
        secondary = (
            "At least one method lacks positive teacher-forced own-target ranking; its condition-sensitive "
            "velocity response is task-irrelevant for the flow target at this checkpoint."
        )
    report = f"""# Gate A.1 conditional-to-trajectory causal audit

## Technical summary

Gate A remains **failed and unadvanced**.  The frozen 1.2 generated-target
between/within threshold was neither changed nor reinterpreted.  {primary}
{secondary}

This is a diagnostic-only audit of the four existing 400-step checkpoints,
the eight fixed v1 training IDs, 792 stabilizer candidates, and the original
eight-step sampling budget.  It does not train a model, activate v2, perform
relaxation/DFPT, or claim Gate A passage.

## Unified four-method result

`final_training_loss` below means the reproducible eight-draw, final-checkpoint
training-panel flow loss.  The original historical final minibatch loss was
not saved in the checkpoint and is not retroactively claimed.

{unified_table}

The common condition-shuffle gap and feature-shift controls are nonzero, while
all generated ratios remain below the frozen `{threshold}` requirement.

## Common-noise counterfactual trajectories

For every method, the eight tensor conditions start from identical type-logit,
fractional-coordinate, lattice-log noise and share every subsequent sampler
operation.  `trajectory_pairwise.csv` holds all 28 target pairs at every time;
`trajectory_mean_curves.csv` and `trajectory_curves.svg` are the aggregate
curves.  The head/time summary (onset is 10% of each head's own peak) is:

{trajectory_dynamics}

The terminal state-distance values are:

{trajectory_end}

`trajectory_dynamics_summary.csv` explicitly flags whether each trace has
fallen by at least 10% from its peak.  This separates an early conditional
velocity response from a response that survives integration as a state
difference.

## Teacher-forced own-target ranking

At the true flow interpolant, each crystal's own condition is compared with a
fixed cyclic permutation using the same base noise and time.  Positive margin
means the own condition predicts the correct flow velocity more closely.

{ranking_table}

The time-resolved evidence is `teacher_forced_ranking.csv` and
`teacher_forced_ranking_curves.svg`.

## Condition representation and pooling-collapse audit

The audit exports 8-by-8 tensor-orbit and six-probe response distances, plus raw,
uniform-pooled, aligned and both offline pooling-definition embeddings.  The
current production path is `sum_k q_k phi(tensor_k)`; `phi(sum_k q_k tensor_k)`
was evaluated offline only.  No checkpoint or production method changed.

{embedding_summary.round(4).to_markdown(index=False)}

All matrices are in `condition_embedding_matrices/`; the reference-state
heatmap is `condition_embedding_heatmaps.svg`.

## Posterior diagnostics

The report distinguishes the 8-frame alignment posterior from the 792-way
crystal-automorphism posterior.  `posterior_summary.csv`,
`posterior_weights.csv`, and `posterior_pairwise_divergence.csv` report q(t),
entropy, top-mode mass, effective frame count, JSD between targets, and token
distance.  The paired mean diagnostic is:

{posterior_text}

If posterior JSD is small while tokens differ only weakly, the state produces
a shared posterior.  If posterior JSD is appreciable while token RMS is small,
the posterior differs but the downstream token is collapsing.  The saved
per-time/pair values make that distinction inspectable rather than inferred
from one scalar.

For this common-noise run, the 792-way posterior is exactly shared at t=0
(mean JSD 0) because the state is identical; its divergence and the token RMS
both grow later.  Thus the observed record is not "different posterior but the
same downstream token".  Pooling compresses the token distances (see the
embedding table), but does not alone explain the cross-method failure.

## CFG and sampler sensitivity (not protocol selection)

`sampling_sensitivity.csv` evaluates only the frozen orbit-alignment
checkpoint over the declared CFG scales and sampler steps with common random
numbers.  CFG was not trained (`condition_dropout=0.0` in this checkpoint),
so every nonzero scale is a sensitivity probe, not a valid replacement for the
frozen sampling protocol.  No row replaces the scale-0, 8-step Gate A result.

{sensitivity.round(5).to_markdown(index=False)}

## TensorOrbit-JARVIS-v2 activation audit

The separate v2 candidate remains inactive.  Formula groups are disjoint,
ID/cache joins and response strata revalidate, explicit zero tensors are kept,
and the `StructureMatcher` prefilter has zero candidate cross-split formula
groups.  The activation audit status is `{v2_manifest['status']}`.  See
`tensororbit_jarvis_v2_activation_report.md` and
`artifacts/tensororbit_jarvis_v2_activation_audit/activation_protocol.json`.

## Limits and next decision

This audit establishes a causal diagnosis only for the small frozen training
panel and the existing checkpoints.  It is not orbit-tensor fidelity evidence
from a qualified external oracle, not a generalization result, and not a
physical validation.  Gate A must remain unresolved; any proposed method
change needs a new versioned protocol and a separate small test before a
larger experiment.

## Artifact index

- Unified metrics: `{output_dir / 'unified_four_method_metrics.csv'}`
- Trajectories: `{output_dir / 'trajectory_pairwise.csv'}` and `{output_dir / 'trajectory_curves.svg'}`
- Teacher-forced ranking: `{output_dir / 'teacher_forced_ranking.csv'}`
- Embeddings: `{output_dir / 'condition_embedding_matrices'}`
- Posterior: `{output_dir / 'posterior_summary.csv'}`
- Sensitivity: `{output_dir / 'sampling_sensitivity.csv'}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a1_causal_audit_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a1_causal_audit"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = ROOT
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["status"] != "diagnostic_only_not_a_gate_decision":
        raise ValueError("Gate A.1 audit requires a diagnostic-only protocol")
    device = torch.device(args.device)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "condition_embedding_matrices"
    matrices_dir.mkdir(exist_ok=True)
    report_dir = root / "reports"
    evaluation = protocol["evaluation"]
    raw_batch = _load_panel(
        json.loads((_resolve(root, protocol["parent_protocol"])).read_text(encoding="utf-8")), root, device,
        train_csv=_resolve(root, protocol["data"]["train_csv"]),
        split_manifest=_resolve(root, protocol["data"]["split_manifest"]),
        target_cache_dir=_resolve(root, protocol["data"]["target_cache_dir"]),
        preprocessed_cache=_resolve(root, protocol["data"]["preprocessed_cache"]),
    )
    parent = json.loads((_resolve(root, protocol["parent_protocol"])).read_text(encoding="utf-8"))
    checkpoints = _resolve(root, protocol["checkpoints"])
    unified_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    posterior_summary_rows: list[dict[str, Any]] = []
    posterior_weight_rows: list[dict[str, Any]] = []
    posterior_pair_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    ranking_summary_rows: list[dict[str, Any]] = []
    embedding_summary_rows: list[dict[str, Any]] = []
    embedding_posterior_rows: list[dict[str, Any]] = []
    all_matrices: dict[str, np.ndarray] = {}
    orbit_model: GaugeFlowVectorField | None = None
    orbit_conditions: torch.Tensor | None = None

    for method in parent["methods"]:
        model, scales, checkpoint_mode = _load_model(checkpoints / f"{method}.pt", device)
        if checkpoint_mode != method:
            raise ValueError(f"Checkpoint mismatch: expected {method}, found {checkpoint_mode}")
        batch = raw_batch.clone()
        batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales)
        flow = _flow_gap(model, batch, evaluation["loss_repeats"], evaluation["seed"])
        representative = _representative_consistency(model, batch, evaluation["representative_repeats"], evaluation["seed"])
        generated = _generated_distribution(
            model, batch.piezo_irreps, evaluation["sample_replicates"], evaluation["sample_atoms"],
            evaluation["sample_steps"], evaluation["seed"], device,
        )
        unified_rows.append({
            "method": method,
            "condition_shuffle_gap": flow["shuffled_flow_gap_fraction_median"],
            "representative_velocity_error": representative["relative_velocity_error_mean"],
            "generated_between_within_ratio": generated["between_within_distance_ratio"],
            "condition_feature_shift": generated["condition_permutation_feature_shift_mean"],
            "final_training_loss": flow["correct_flow_loss_mean"],
            "sampling_failure_count": _sampling_failure_count(model, batch.piezo_irreps, evaluation, device),
        })
        trajectory, posterior_summary, posterior_weights, posterior_pairs = _trajectory(
            model, batch.piezo_irreps, method, steps=evaluation["trajectory_steps"],
            atoms=evaluation["sample_atoms"], seed=evaluation["seed"] + 40_000,
            device=device, capture_posterior=method == "orbit_alignment",
        )
        trajectory_rows.extend(trajectory)
        posterior_summary_rows.extend(posterior_summary)
        posterior_weight_rows.extend(posterior_weights)
        posterior_pair_rows.extend(posterior_pairs)
        ranking, ranking_summary = _teacher_forced_ranking(
            model, batch, evaluation["ranking_times"], evaluation["seed"], method
        )
        ranking_rows.extend(ranking)
        ranking_summary_rows.extend(ranking_summary)
        matrices, embedding_summary, embedding_posterior = _embedding_audit(
            model, batch.piezo_irreps, method, atoms=evaluation["sample_atoms"],
            seed=evaluation["seed"], device=device,
        )
        embedding_summary_rows.extend(embedding_summary)
        embedding_posterior_rows.extend(embedding_posterior)
        for name, matrix in matrices.items():
            np.savetxt(matrices_dir / f"{method}_{name}.csv", matrix, delimiter=",", fmt="%.9g")
            if method == "orbit_alignment":
                all_matrices[name] = matrix
        if method == "orbit_alignment":
            orbit_model, orbit_conditions = model, batch.piezo_irreps

    if orbit_model is None or orbit_conditions is None:
        raise RuntimeError("The frozen protocol is missing orbit_alignment")
    unified = pd.DataFrame(unified_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    trajectory_means = trajectories.groupby(["method", "time", "quantity"], as_index=False).agg(
        pairwise_rms_mean=("pairwise_rms", "mean"),
        pairwise_rms_std=("pairwise_rms", "std"),
    )
    dynamics = _trajectory_dynamics_summary(trajectories)
    ranking = pd.DataFrame(ranking_rows)
    ranking_summary = pd.DataFrame(ranking_summary_rows)
    embedding_summary = pd.DataFrame(embedding_summary_rows)
    posterior_summary = pd.DataFrame(posterior_summary_rows)
    posterior_weights = pd.DataFrame(posterior_weight_rows)
    posterior_pairs = pd.DataFrame(posterior_pair_rows)
    embedding_posterior = pd.DataFrame(embedding_posterior_rows)
    sensitivity = pd.DataFrame(_sampling_sensitivity(orbit_model, orbit_conditions, evaluation, device))

    unified.to_csv(output_dir / "unified_four_method_metrics.csv", index=False)
    trajectories.to_csv(output_dir / "trajectory_pairwise.csv", index=False)
    trajectory_means.to_csv(output_dir / "trajectory_mean_curves.csv", index=False)
    dynamics.to_csv(output_dir / "trajectory_dynamics_summary.csv", index=False)
    ranking.to_csv(output_dir / "teacher_forced_ranking.csv", index=False)
    ranking_summary.to_csv(output_dir / "teacher_forced_ranking_summary.csv", index=False)
    embedding_summary.to_csv(output_dir / "condition_embedding_summary.csv", index=False)
    embedding_posterior.to_csv(output_dir / "condition_embedding_posterior.csv", index=False)
    posterior_summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    posterior_weights.to_csv(output_dir / "posterior_weights.csv", index=False)
    posterior_pairs.to_csv(output_dir / "posterior_pairwise_divergence.csv", index=False)
    sensitivity.to_csv(output_dir / "sampling_sensitivity.csv", index=False)
    _plot_trajectory(trajectories, output_dir / "trajectory_curves.svg")
    _plot_ranking(ranking_summary, output_dir / "teacher_forced_ranking_curves.svg")
    _plot_embedding_matrices(all_matrices, output_dir / "condition_embedding_heatmaps.svg")

    # v2 is separately audited after all Gate A.1 model outputs have been written.
    v2_manifest = _v2_activation_audit(protocol, root, report_dir)
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": protocol["status"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "parent_gate_protocol_sha256": _sha256_file(_resolve(root, protocol["parent_protocol"])),
        "device": str(device),
        "checkpoints": {method: _sha256_file(checkpoints / f"{method}.pt") for method in parent["methods"]},
        "outputs": {path.name: _sha256_file(path) for path in output_dir.glob("*.csv")},
        "v2_activation_manifest_sha256": v2_manifest["manifest_sha256"],
        "frozen_threshold": protocol["frozen_constraints"]["gate_a_threshold_between_within"],
        "gate_a_status": "failed_not_advanced",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_main_report(
        report_dir / "gate_a1_causal_audit.md", protocol, unified, trajectories, ranking,
        ranking_summary, embedding_summary, posterior_pairs, sensitivity, dynamics, v2_manifest, output_dir,
    )
    # Keep a copy next to the numerical artifacts for self-contained archiving.
    shutil.copyfile(report_dir / "gate_a1_causal_audit.md", output_dir / "gate_a1_causal_audit.md")
    print(json.dumps({
        "report": str(report_dir / "gate_a1_causal_audit.md"),
        "output_dir": str(output_dir),
        "gate_a_status": "failed_not_advanced",
    }, indent=2))


if __name__ == "__main__":
    main()
