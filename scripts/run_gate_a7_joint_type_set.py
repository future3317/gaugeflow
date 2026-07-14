"""Run a versioned graph-composition-constrained discrete atom-type audit.

The generated composition is a graph-level latent predicted from the endpoint
condition.  It is never passed into the model as a target-side condition.
"""

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

from audit_gate_a4_generator_substrate import (  # noqa: E402
    _decoded_metrics,
    _finite_state,
    _generated_batch,
    _load_panel,
    _localize_types,
    _set_endpoint_ids,
)
from gaugeflow.discrete import AbsorbingDiscreteTypeFlowMatcher, DiscreteSamplingNoise  # noqa: E402
from gaugeflow.flow import CrystalFlowState  # noqa: E402
from gaugeflow.model import GaugeFlowVectorField  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _decode_state(state: CrystalFlowState, matcher: AbsorbingDiscreteTypeFlowMatcher) -> CrystalFlowState:
    token = state.type_state.argmax(dim=-1)
    if bool((token == matcher.mask_index).any()):
        raise RuntimeError("terminal discrete state contains a non-chemical mask")
    return CrystalFlowState(state.type_state[:, : matcher.atom_types], state.frac_coords, state.lattice_log)


def _common_noise(matcher: AbsorbingDiscreteTypeFlowMatcher, *, steps: int, replicates: int, atoms: int, seed: int, device: torch.device) -> tuple[DiscreteSamplingNoise, torch.Tensor]:
    _seed(seed)
    reveal = torch.rand((steps, replicates, atoms), device=device)
    categorical = torch.rand((steps, replicates, atoms, matcher.atom_types), device=device)
    count_uniform = torch.rand((replicates, matcher.atom_types, atoms + 1), device=device)
    noise = DiscreteSamplingNoise(
        reveal_uniform=reveal.repeat(1, 2, 1).reshape(steps, 2 * replicates * atoms),
        categorical_uniform=categorical.repeat(1, 2, 1, 1).reshape(steps, 2 * replicates * atoms, matcher.atom_types),
    )
    gumbel = -torch.log(-torch.log(count_uniform.clamp(1e-7, 1.0 - 1e-7)))
    return noise, gumbel.repeat(2, 1, 1)


class _OraclePosterior(torch.nn.Module):
    def __init__(self, targets: torch.Tensor, state_dim: int):
        super().__init__()
        self.register_buffer("targets", targets.detach().clone())
        self.state_dim = state_dim

    def forward(self, type_state, frac_coords, lattice_log, batch, time, *args, **kwargs):
        del type_state, time, args, kwargs
        logits = torch.full((self.targets.numel(), self.state_dim), -30.0, device=self.targets.device)
        logits.scatter_(1, self.targets.unsqueeze(-1), 30.0)
        return logits, torch.zeros_like(frac_coords), torch.zeros_like(lattice_log), torch.ones((lattice_log.shape[0], 1), device=logits.device)


def _composition_logits(model, matcher: AbsorbingDiscreteTypeFlowMatcher, batch) -> torch.Tensor:
    state = matcher.mask_state(batch)
    time = torch.zeros((batch.num_graphs,), device=batch.batch.device)
    return model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, batch.condition_present, return_composition_counts=True,
    )[-1]


def _analytic_closure(batch, matcher: AbsorbingDiscreteTypeFlowMatcher, settings: dict[str, Any]) -> pd.DataFrame:
    samples = settings["analytic_closure_replicates"]
    generated, _ = _generated_batch(batch, samples, list(range(matcher.atom_types)), batch.batch.device)
    oracle = _OraclePosterior(generated.atom_types, matcher.state_dim)
    counts = matcher.composition_count_targets(generated)
    state = matcher.sample(oracle, generated, steps=settings["sampler_steps"], graph_counts=counts)
    token = state.type_state.argmax(-1)
    rows = []
    for graph in range(generated.num_graphs):
        nodes = generated.batch == graph
        rows.append({
            "graph": graph,
            "decoded_atom_type_accuracy": float((token[nodes] == generated.atom_types[nodes]).float().mean()),
            "terminal_mask_count": int((token[nodes] == matcher.mask_index).sum()),
            "composition_conserved": bool(torch.equal(torch.bincount(token[nodes], minlength=matcher.atom_types), counts[graph])),
        })
    return pd.DataFrame(rows)


def _train(batch, matcher: AbsorbingDiscreteTypeFlowMatcher, settings: dict[str, Any]):
    _seed(settings["seed"])
    model = GaugeFlowVectorField(
        hidden_dim=settings["hidden_dim"], layers=settings["layers"], atom_types=matcher.state_dim,
        conditioning_mode="endpoint_id", composition_max_atoms=4, composition_atom_types=matcher.atom_types,
    ).to(batch.batch.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    last = {}
    for _ in range(settings["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        discrete = matcher.loss(model, batch)
        composition_logits = _composition_logits(model, matcher, batch)
        composition_nll, _ = matcher.composition_count_loss(composition_logits, batch)
        total = discrete["loss"] + settings["composition_loss_weight"] * composition_nll
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {
            **{name: float(value.detach()) for name, value in discrete.items()},
            "composition_count_nll": float(composition_nll.detach()),
            "total_loss": float(total.detach()),
        }
    model.eval()
    return model, last


def _map_counts(model, matcher: AbsorbingDiscreteTypeFlowMatcher, batch, gumbel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _composition_logits(model, matcher, batch)
    atom_counts = torch.bincount(batch.batch, minlength=batch.num_graphs)
    mapped = torch.stack([
        matcher.map_composition_counts(logits[graph], int(atom_counts[graph]), gumbel[graph])
        for graph in range(batch.num_graphs)
    ])
    target = matcher.composition_count_targets(batch)
    return mapped, target, logits


def _trajectory_rows(trajectory: list[CrystalFlowState], matcher: AbsorbingDiscreteTypeFlowMatcher, atoms: int) -> pd.DataFrame:
    rows = []
    steps = len(trajectory) - 1
    for step, state in enumerate(trajectory):
        left, right = state.type_state[:atoms], state.type_state[atoms: 2 * atoms]
        rows.append({
            "time": step / steps,
            "pairwise_state_rms": float((left - right).square().mean().sqrt()),
            "pairwise_decoded_difference": float((left.argmax(-1) != right.argmax(-1)).float().mean()),
            "left_mask_fraction": float((left.argmax(-1) == matcher.mask_index).float().mean()),
            "right_mask_fraction": float((right.argmax(-1) == matcher.mask_index).float().mean()),
        })
    return pd.DataFrame(rows)


def _write_report(path: Path, protocol: dict[str, Any], closure: pd.DataFrame, result: dict[str, Any]):
    criteria = protocol["endpoint_id_type_substrate"]["criteria"]
    lines = [
        f"# {protocol['name']}",
        "",
        "This protocol retains the valid absorbing discrete path and uses a generated graph-level count latent plus count-constrained reveal matching. It does not modify prior gates or start geometry/tensor training.",
        "",
        "## Analytic closure",
        "",
        f"Exact-posterior atom accuracy: `{closure.decoded_atom_type_accuracy.mean():.3f}`; terminal masks: `{int(closure.terminal_mask_count.sum())}`; composition conserved: `{bool(closure.composition_conserved.all())}`.",
        "",
        "## Fixed endpoint-ID type-set result",
        "",
        f"Generated exact-composition accuracy: `{result['graph_composition_exact_accuracy']:.3f}` (minimum `{criteria['graph_composition_exact_accuracy_min']:.2f}`); decoded composition: `{result['type_decoded_composition_accuracy']:.3f}` (minimum `{criteria['type_decoded_composition_accuracy_min']:.2f}`); site atom accuracy: `{result['decoded_atom_type_accuracy']:.3f}` (minimum `{criteria['decoded_atom_type_accuracy_min']:.2f}`).",
        f"Common-noise terminal discrete difference: `{result['common_noise_terminal_discrete_difference']:.3f}`; terminal masks: `{result['terminal_mask_count']}`; sampling failures: `{result['sampling_failure_count']}`.",
        "",
        "A pass qualifies only type-set generation. Geometry, joint crystal generation, and tensor conditioning remain separate, versioned work.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a7_joint_type_set_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a7_joint_type_set_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "pre_registered_two_endpoint_joint_type_repair",
        "pre_registered_time_conditioned_joint_type_repair",
        "pre_registered_source_weighted_joint_type_repair",
    }:
        raise ValueError("the joint type-set runner requires a versioned pre-registered protocol")
    device = torch.device(args.device)
    report_name = protocol.get("report_filename", "gate_a7_joint_type_set_report.md")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = _load_panel(protocol, ROOT, device, preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]))
    if raw.num_graphs != 2 or any(int((raw.batch == graph).sum()) != 4 for graph in range(2)):
        raise ValueError("A7 requires the frozen equal-four-atom InN/BN pair")
    endpoint_batch = _localize_types(_set_endpoint_ids(raw), list(range(119)))
    path_settings = protocol["joint_type_set"]
    matcher = AbsorbingDiscreteTypeFlowMatcher(
        training_time_distribution=path_settings.get("training_time_distribution", "uniform")
    )
    closure = _analytic_closure(endpoint_batch, matcher, path_settings)
    closure.to_csv(output / "analytic_discrete_set_closure.csv", index=False)
    settings = protocol["endpoint_id_type_substrate"]
    model, training = _train(endpoint_batch, matcher, settings)
    generated, labels = _generated_batch(raw, settings["sample_replicates"], list(range(119)), device)
    atoms = int((raw.batch == 0).sum())
    noise, gumbel = _common_noise(
        matcher, steps=path_settings["sampler_steps"], replicates=settings["sample_replicates"], atoms=atoms,
        seed=settings["common_noise_seed"], device=device,
    )
    predicted_counts, target_counts, composition_logits = _map_counts(model, matcher, generated, gumbel)
    state = matcher.sample(model, generated, steps=path_settings["sampler_steps"], noise=noise, graph_counts=predicted_counts)
    decoded, summary = _decoded_metrics(_decode_state(state, matcher), generated, labels, raw, list(range(119)), "constrained_discrete_type_set")
    decoded.to_csv(output / "decoded_state_audit.csv", index=False)
    count_frame = pd.DataFrame({
        "sample": list(range(generated.num_graphs)), "target": labels.detach().cpu().tolist(),
        "exact_count_match": (predicted_counts == target_counts).all(dim=-1).detach().cpu().tolist(),
        "predicted_counts": [json.dumps(value.tolist()) for value in predicted_counts.detach().cpu()],
        "target_counts": [json.dumps(value.tolist()) for value in target_counts.detach().cpu()],
        "count_logits_max": composition_logits.max(dim=-1).values.mean(dim=-1).detach().cpu().tolist(),
    })
    count_frame.to_csv(output / "composition_count_audit.csv", index=False)
    trajectory_batch, _ = _generated_batch(raw, 1, list(range(119)), device)
    trajectory_noise, trajectory_gumbel = _common_noise(matcher, steps=path_settings["sampler_steps"], replicates=1, atoms=atoms, seed=settings["common_noise_seed"], device=device)
    trajectory_counts, _, _ = _map_counts(model, matcher, trajectory_batch, trajectory_gumbel)
    _, trajectory = matcher.sample(
        model, trajectory_batch, steps=path_settings["sampler_steps"], noise=trajectory_noise,
        graph_counts=trajectory_counts, return_trajectory=True,
    )
    trajectory_frame = _trajectory_rows(trajectory, matcher, atoms)
    trajectory_frame.to_csv(output / "common_noise_trajectory.csv", index=False)
    token = state.type_state.argmax(dim=-1)
    result = {
        "variant": "absorbing_discrete_flow_with_generated_count_constraint_full_119",
        "device": str(device), "chemical_vocabulary_size": matcher.atom_types,
        "internal_mask_index": matcher.mask_index, "train_steps": settings["train_steps"],
        "sampler_steps": path_settings["sampler_steps"], **training, **summary,
        "training_time_distribution": matcher.training_time_distribution,
        "graph_composition_exact_accuracy": float((predicted_counts == target_counts).all(dim=-1).float().mean()),
        "terminal_mask_count": int((token == matcher.mask_index).sum()),
        "common_noise_terminal_discrete_difference": float(trajectory_frame.iloc[-1].pairwise_decoded_difference),
        "sampling_failure_count": _finite_state(state),
    }
    pd.DataFrame([result]).to_csv(output / "endpoint_id_results.csv", index=False)
    (output / "literature_basis.json").write_text(json.dumps(protocol["evidence"]["literature"], indent=2) + "\n", encoding="utf-8")
    _write_report(output / report_name, protocol, closure, result)
    criteria = settings["criteria"]
    qualified = bool(
        closure.decoded_atom_type_accuracy.min() == 1.0 and closure.terminal_mask_count.sum() == 0 and bool(closure.composition_conserved.all())
        and result["type_decoded_composition_accuracy"] >= criteria["type_decoded_composition_accuracy_min"]
        and result["decoded_atom_type_accuracy"] >= criteria["decoded_atom_type_accuracy_min"]
        and result["graph_composition_exact_accuracy"] >= criteria["graph_composition_exact_accuracy_min"]
        and result["terminal_mask_count"] <= criteria["terminal_mask_count_max"]
        and result["sampling_failure_count"] <= criteria["sampling_failure_count_max"]
        and result["common_noise_terminal_discrete_difference"] >= criteria["common_noise_terminal_discrete_difference_min"]
    )
    files = [
        "analytic_discrete_set_closure.csv", "endpoint_id_results.csv", "decoded_state_audit.csv",
        "composition_count_audit.csv", "common_noise_trajectory.csv", "literature_basis.json",
        report_name,
    ]
    manifest = {
        "schema": 1, "name": protocol["name"], "protocol_sha256": _sha256(protocol_path),
        "status": "type_set_substrate_qualified_geometry_not_started" if qualified else "type_set_substrate_not_qualified",
        "analytic_discrete_set_closure_passed": bool(closure.decoded_atom_type_accuracy.min() == 1.0 and closure.terminal_mask_count.sum() == 0 and bool(closure.composition_conserved.all())),
        "type_set_substrate_qualified": qualified, "device": str(device),
        "historical_gate_evidence_modified": False, "tensor_conditioned_gate_started": False,
        "geometry_or_joint_crystal_repair_started": False,
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
