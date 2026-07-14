"""Run the fixed Gate A6 absorbing-discrete atom-type substrate audit.

This is endpoint-ID-only and type-only.  It does not start tensor conditioning,
geometry repair, or a full crystal benchmark.
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
    tokens = state.type_state.argmax(dim=-1)
    if bool((tokens == matcher.mask_index).any()):
        raise RuntimeError("cannot decode a terminal state containing a mask token")
    return CrystalFlowState(state.type_state[:, : matcher.atom_types], state.frac_coords, state.lattice_log)


def _common_noise(matcher: AbsorbingDiscreteTypeFlowMatcher, *, steps: int, replicates: int, atoms: int, seed: int, device: torch.device) -> DiscreteSamplingNoise:
    _seed(seed)
    reveal = torch.rand((steps, replicates, atoms), device=device)
    categorical = torch.rand((steps, replicates, atoms, matcher.atom_types), device=device)
    return DiscreteSamplingNoise(
        reveal_uniform=reveal.repeat(1, 2, 1).reshape(steps, 2 * replicates * atoms),
        categorical_uniform=categorical.repeat(1, 2, 1, 1).reshape(steps, 2 * replicates * atoms, matcher.atom_types),
    )


class _OraclePosterior(torch.nn.Module):
    """Exact endpoint posterior used only to verify the discrete sampler algebra."""

    def __init__(self, targets: torch.Tensor, state_dim: int):
        super().__init__()
        self.register_buffer("targets", targets.detach().clone())
        self.state_dim = state_dim

    def forward(self, type_state, frac_coords, lattice_log, batch, time, *args, **kwargs):
        del type_state, time, args, kwargs
        logits = torch.full((self.targets.numel(), self.state_dim), -30.0, device=self.targets.device)
        logits.scatter_(1, self.targets.unsqueeze(-1), 30.0)
        return logits, torch.zeros_like(frac_coords), torch.zeros_like(lattice_log), torch.ones((lattice_log.shape[0], 1), device=logits.device)


def _analytic_closure(batch, matcher: AbsorbingDiscreteTypeFlowMatcher, settings: dict[str, Any]) -> pd.DataFrame:
    rows = []
    samples = settings["analytic_closure_replicates"]
    clone = _generated_batch(batch, samples, list(range(matcher.atom_types)), batch.batch.device)[0]
    oracle = _OraclePosterior(clone.atom_types, matcher.state_dim)
    state = matcher.sample(oracle, clone, steps=settings["sampler_steps"])
    tokens = state.type_state.argmax(dim=-1)
    for graph in range(clone.num_graphs):
        nodes = torch.nonzero(clone.batch == graph, as_tuple=False).flatten()
        rows.append({
            "graph": graph,
            "decoded_atom_type_accuracy": float((tokens[nodes] == clone.atom_types[nodes]).float().mean()),
            "terminal_mask_count": int((tokens[nodes] == matcher.mask_index).sum()),
        })
    return pd.DataFrame(rows)


def _train(batch, matcher: AbsorbingDiscreteTypeFlowMatcher, settings: dict[str, Any]):
    _seed(settings["seed"])
    model = GaugeFlowVectorField(
        hidden_dim=settings["hidden_dim"], layers=settings["layers"], atom_types=matcher.state_dim,
        conditioning_mode="endpoint_id",
    ).to(batch.batch.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    last = {}
    for _ in range(settings["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        terms = matcher.loss(model, batch)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in terms.items()}
    model.eval()
    return model, last


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
        "# Gate A6 discrete atom-type substrate",
        "",
        "A6 replaces only the failed continuous atom-type path. It uses an absorbing non-chemical mask and an endpoint-posterior discrete flow sampler; it does not modify A4/A5 or start tensor conditioning.",
        "",
        "## Analytic discrete-path closure",
        "",
        f"Exact-posterior terminal atom accuracy: `{closure.decoded_atom_type_accuracy.mean():.3f}`; terminal masks: `{int(closure.terminal_mask_count.sum())}`.",
        "",
        "## Fixed endpoint-ID type result",
        "",
        f"Composition accuracy: `{result['type_decoded_composition_accuracy']:.3f}` (minimum `{criteria['type_decoded_composition_accuracy_min']:.2f}`); atom accuracy: `{result['decoded_atom_type_accuracy']:.3f}` (minimum `{criteria['decoded_atom_type_accuracy_min']:.2f}`).",
        f"Common-noise terminal discrete difference: `{result['common_noise_terminal_discrete_difference']:.3f}` (minimum `{criteria['common_noise_terminal_discrete_difference_min']:.2f}`); masks: `{result['terminal_mask_count']}`; failures: `{result['sampling_failure_count']}`.",
        "",
        "A passing A6 qualifies only the type substrate and requires a separate geometry protocol; it never passes Gate A or authorizes tensor-conditioned training.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a6_discrete_type_substrate_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a6_discrete_type_substrate_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_two_endpoint_discrete_type_repair":
        raise ValueError("A6 requires its versioned pre-registered protocol")
    device = torch.device(args.device)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = _load_panel(
        protocol, ROOT, device,
        preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]),
    )
    if raw.num_graphs != 2 or any(int((raw.batch == graph).sum()) != 4 for graph in range(2)):
        raise ValueError("A6 requires the frozen equal-four-atom InN/BN pair")
    endpoint_batch = _localize_types(_set_endpoint_ids(raw), list(range(119)))
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    path_settings = protocol["discrete_path"]
    closure = _analytic_closure(endpoint_batch, matcher, path_settings)
    closure.to_csv(output / "analytic_discrete_path_closure.csv", index=False)
    model, training = _train(endpoint_batch, matcher, protocol["endpoint_id_type_substrate"])
    settings = protocol["endpoint_id_type_substrate"]
    generated, labels = _generated_batch(raw, settings["sample_replicates"], list(range(119)), device)
    atoms = int((raw.batch == 0).sum())
    noise = _common_noise(
        matcher, steps=path_settings["sampler_steps"], replicates=settings["sample_replicates"], atoms=atoms,
        seed=settings["common_noise_seed"], device=device,
    )
    state = matcher.sample(model, generated, steps=path_settings["sampler_steps"], noise=noise)
    decoded, summary = _decoded_metrics(_decode_state(state, matcher), generated, labels, raw, list(range(119)), "absorbing_discrete_type")
    decoded.to_csv(output / "decoded_state_audit.csv", index=False)
    trajectory_batch, _ = _generated_batch(raw, 1, list(range(119)), device)
    trajectory_noise = _common_noise(matcher, steps=path_settings["sampler_steps"], replicates=1, atoms=atoms, seed=settings["common_noise_seed"], device=device)
    trajectory_state, trajectory = matcher.sample(model, trajectory_batch, steps=path_settings["sampler_steps"], noise=trajectory_noise, return_trajectory=True)
    trajectory_frame = _trajectory_rows(trajectory, matcher, atoms)
    trajectory_frame.to_csv(output / "common_noise_trajectory.csv", index=False)
    terminal_difference = float(trajectory_frame.iloc[-1].pairwise_decoded_difference)
    tokens = state.type_state.argmax(dim=-1)
    result = {
        "variant": "absorbing_mask_discrete_flow_full_119",
        "device": str(device),
        "chemical_vocabulary_size": matcher.atom_types,
        "internal_mask_index": matcher.mask_index,
        "train_steps": settings["train_steps"],
        "sampler_steps": path_settings["sampler_steps"],
        **training,
        **summary,
        "terminal_mask_count": int((tokens == matcher.mask_index).sum()),
        "common_noise_terminal_discrete_difference": terminal_difference,
        "sampling_failure_count": _finite_state(state),
    }
    pd.DataFrame([result]).to_csv(output / "endpoint_id_results.csv", index=False)
    (output / "literature_basis.json").write_text(json.dumps(protocol["evidence"]["literature"], indent=2) + "\n", encoding="utf-8")
    _write_report(output / "gate_a6_discrete_type_report.md", protocol, closure, result)
    criteria = settings["criteria"]
    qualified = bool(
        closure.decoded_atom_type_accuracy.min() == 1.0
        and closure.terminal_mask_count.sum() == 0
        and result["type_decoded_composition_accuracy"] >= criteria["type_decoded_composition_accuracy_min"]
        and result["decoded_atom_type_accuracy"] >= criteria["decoded_atom_type_accuracy_min"]
        and result["terminal_mask_count"] <= criteria["terminal_mask_count_max"]
        and result["sampling_failure_count"] <= criteria["sampling_failure_count_max"]
        and result["common_noise_terminal_discrete_difference"] >= criteria["common_noise_terminal_discrete_difference_min"]
    )
    files = [
        "analytic_discrete_path_closure.csv", "endpoint_id_results.csv", "decoded_state_audit.csv",
        "common_noise_trajectory.csv", "literature_basis.json", "gate_a6_discrete_type_report.md",
    ]
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "protocol_sha256": _sha256(protocol_path),
        "status": "type_substrate_qualified_geometry_not_started" if qualified else "type_substrate_not_qualified",
        "analytic_path_closure_passed": bool(closure.decoded_atom_type_accuracy.min() == 1.0 and closure.terminal_mask_count.sum() == 0),
        "type_substrate_qualified": qualified,
        "device": str(device),
        "historical_gate_evidence_modified": False,
        "tensor_conditioned_gate_started": False,
        "geometry_or_joint_repair_started": False,
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
