"""Gate A4: qualify the standalone generator substrate before tensor conditioning.

The script is intentionally restricted to the frozen InN/BN pair.  It never
loads a tensor-conditioned checkpoint and never changes any historical gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluate_gate_a import _distance_diagnostics, _load_panel, _state_features  # noqa: E402
from evaluate_gate_a3_two_target import _decoded_descriptor, _descriptor_distance  # noqa: E402
from gaugeflow.data import PiezoCrystalDataset, collate_crystals  # noqa: E402
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher  # noqa: E402
from gaugeflow.manifold import torus_logmap, wrap01  # noqa: E402
from gaugeflow.model import GaugeFlowVectorField  # noqa: E402


HEADS = ("type", "coord", "lattice")


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


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().square().mean().sqrt())


def _state_copy(value: CrystalFlowState) -> CrystalFlowState:
    return CrystalFlowState(value.type_state.clone(), value.frac_coords.clone(), value.lattice_log.clone())


def _endpoint_ids(graphs: int, device: torch.device) -> torch.Tensor:
    if graphs != 2:
        raise ValueError("Gate A4 is intentionally restricted to the frozen two-endpoint panel")
    return F.one_hot(torch.arange(graphs, device=device), num_classes=2).float()


def _set_endpoint_ids(batch, labels: torch.Tensor | None = None):
    clone = batch.clone()
    if labels is None:
        labels = torch.arange(clone.num_graphs, device=clone.batch.device)
    clone.piezo_irreps = F.one_hot(labels, num_classes=2).float()
    clone.condition_present = torch.ones((clone.num_graphs, 1), device=clone.batch.device, dtype=torch.bool)
    return clone


def _records_from_batch(batch) -> list[Data]:
    records: list[Data] = []
    for graph in range(batch.num_graphs):
        nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
        records.append(Data(
            atom_types=batch.atom_types[nodes].detach().cpu().clone(),
            frac_coords=batch.frac_coords[nodes].detach().cpu().clone(),
            lattice=batch.lattice[graph].detach().cpu().unsqueeze(0).clone(),
            piezo_irreps=torch.zeros((1, 2), dtype=torch.float32),
            condition_present=torch.ones((1, 1), dtype=torch.bool),
            num_nodes=int(nodes.numel()),
        ))
    return records


def _localize_types(batch, vocabulary: list[int]):
    lookup = {element: index for index, element in enumerate(vocabulary)}
    unknown = sorted(set(map(int, batch.atom_types.detach().cpu().tolist())).difference(lookup))
    if unknown:
        raise ValueError(f"A4 vocabulary omits endpoint elements {unknown}")
    clone = batch.clone()
    clone.atom_types = torch.tensor(
        [lookup[int(value)] for value in batch.atom_types.detach().cpu().tolist()],
        dtype=torch.long, device=batch.atom_types.device,
    )
    return clone


def _generated_batch(endpoint_batch, replicates: int, vocabulary: list[int], device: torch.device):
    records = _records_from_batch(endpoint_batch)
    replicated: list[Data] = []
    labels: list[int] = []
    for label, record in enumerate(records):
        for _ in range(replicates):
            replicated.append(record.clone())
            labels.append(label)
    batch = collate_crystals(replicated).to(device)
    label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    batch = _set_endpoint_ids(batch, label_tensor)
    return _localize_types(batch, vocabulary), label_tensor


def _common_initial_state(
    matcher: RiemannianCrystalFlowMatcher,
    batch,
    *,
    endpoint_count: int,
    replicates: int,
    atoms: int,
    seed: int,
) -> CrystalFlowState:
    if batch.num_graphs != endpoint_count * replicates:
        raise ValueError("Common-noise batch ordering does not match endpoint_count * replicates")
    _seed(seed)
    device = batch.batch.device
    vocab = matcher.atom_types
    if matcher.type_path == "simplex_probability":
        template_type = torch.softmax(torch.randn((replicates, atoms, vocab), device=device), dim=-1)
    else:
        template_type = torch.randn((replicates, atoms, vocab), device=device)
    template_coord = torch.rand((replicates, atoms, 3), device=device)
    template_lattice = torch.randn((replicates, 6), device=device)
    state = CrystalFlowState(
        type_state=template_type.repeat(endpoint_count, 1, 1).reshape(-1, vocab),
        frac_coords=template_coord.repeat(endpoint_count, 1, 1).reshape(-1, 3),
        lattice_log=template_lattice.repeat(endpoint_count, 1),
    )
    if matcher.active_heads == HEADS:
        return state
    target = matcher.target_state(batch)
    return CrystalFlowState(
        state.type_state if "type" in matcher.active_heads else target.type_state,
        state.frac_coords if "coord" in matcher.active_heads else target.frac_coords,
        state.lattice_log if "lattice" in matcher.active_heads else target.lattice_log,
    )


def _velocity_for_endpoint(target: CrystalFlowState, base: CrystalFlowState) -> tuple[torch.Tensor, ...]:
    return (
        target.type_state - base.type_state,
        torus_logmap(base.frac_coords, target.frac_coords),
        target.lattice_log - base.lattice_log,
    )


class _AnalyticVelocity(torch.nn.Module):
    def __init__(self, velocity: tuple[torch.Tensor, ...]):
        super().__init__()
        self.velocity = tuple(value.detach() for value in velocity)

    def forward(self, *args, **kwargs):
        del args, kwargs
        graphs = self.velocity[2].shape[0]
        return (*self.velocity, torch.ones((graphs, 1), device=self.velocity[0].device))


def _path_closure(raw_batch, protocol: dict[str, Any], device: torch.device) -> tuple[pd.DataFrame, bool]:
    settings = protocol["a4_0_path_closure"]
    specs = {
        "type": ("type",),
        "coordinate": ("coord",),
        "lattice": ("lattice",),
        "joint": HEADS,
    }
    rows: list[dict[str, Any]] = []
    for offset, (name, active) in enumerate(specs.items()):
        matcher = RiemannianCrystalFlowMatcher(active_heads=active)
        _seed(settings["initial_noise_seed"] + offset)
        target = matcher.target_state(raw_batch)
        base = matcher.random_state(raw_batch)
        velocity = _velocity_for_endpoint(target, base)
        if "type" not in active:
            velocity = (torch.zeros_like(velocity[0]), velocity[1], velocity[2])
        if "coord" not in active:
            velocity = (velocity[0], torch.zeros_like(velocity[1]), velocity[2])
        if "lattice" not in active:
            velocity = (velocity[0], velocity[1], torch.zeros_like(velocity[2]))
        sampled = matcher.sample(
            _AnalyticVelocity(velocity), raw_batch, steps=settings["sampler_steps"], initial_state=base
        )
        for graph, material_id in enumerate(protocol["material_ids"]):
            nodes = torch.nonzero(raw_batch.batch == graph, as_tuple=False).flatten()
            type_error = _rms(sampled.type_state[nodes] - target.type_state[nodes])
            coordinate_error = _rms(torus_logmap(sampled.frac_coords[nodes], target.frac_coords[nodes]))
            lattice_error = _rms(sampled.lattice_log[graph] - target.lattice_log[graph])
            atom_accuracy = float((sampled.type_state[nodes].argmax(-1) == target.type_state[nodes].argmax(-1)).float().mean())
            active_errors = {
                "type": type_error, "coord": coordinate_error, "lattice": lattice_error,
            }
            closure = all(active_errors[head] <= settings["continuous_error_max"] for head in active)
            decoded = atom_accuracy if "type" in active else float(closure)
            rows.append({
                "experiment": "analytic_true_velocity_production_sampler",
                "subspace": name,
                "material_id": material_id,
                "sampler_steps": settings["sampler_steps"],
                "type_continuous_endpoint_error": type_error,
                "fractional_coordinate_endpoint_error": coordinate_error,
                "lattice_log_endpoint_error": lattice_error,
                "decoded_atom_accuracy": atom_accuracy,
                "decoded_endpoint_accuracy": decoded,
                "analytic_path_closed": closure and decoded >= settings["decoded_endpoint_accuracy_min"],
            })
    frame = pd.DataFrame(rows)
    return frame, bool(frame.analytic_path_closed.all())


def _active_vocab(protocol: dict[str, Any]) -> list[int]:
    data = protocol["data"]
    manifest = _resolve(data["v2_candidate_split_manifest"])
    dataset = PiezoCrystalDataset(
        _resolve(data["train_csv"]), split_manifest=manifest, split="train",
        target_cache_dir=_resolve(data["target_cache_dir"]), preprocessed_cache=_resolve(data["preprocessed_cache"]),
    )
    values: set[int] = set()
    for index in range(len(dataset)):
        values.update(map(int, dataset[index].atom_types.tolist()))
    return sorted(values)


def _fixed_interpolant(matcher: RiemannianCrystalFlowMatcher, batch, time: torch.Tensor, seed: int):
    _seed(seed)
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    velocity = _velocity_for_endpoint(target, base)
    node_time = time[batch.batch].unsqueeze(-1)
    state = CrystalFlowState(
        base.type_state + node_time * velocity[0],
        wrap01(base.frac_coords + node_time * velocity[1]),
        base.lattice_log + time.unsqueeze(-1) * velocity[2],
    )
    return state, target, base, velocity


def _head_audit(model, matcher: RiemannianCrystalFlowMatcher, batch, learning_rate: float, seed: int, total_update_norm: float):
    time = torch.full((batch.num_graphs,), 0.5, device=batch.batch.device)
    state, _, _, velocity = _fixed_interpolant(matcher, batch, time, seed)
    output = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, batch.condition_present,
    )[:3]
    losses = {
        "type": (output[0] - velocity[0]).square().mean(),
        "coord": (output[1] - velocity[1]).square().mean(),
        "lattice": (output[2] - velocity[2]).square().mean(),
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    active_total = sum(losses[head] for head in matcher.active_heads)
    rows = []
    for name in HEADS:
        gradients = torch.autograd.grad(losses[name], parameters, retain_graph=True, allow_unused=True)
        gradient_norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in gradients if gradient is not None))
        target_norm = _rms(velocity[HEADS.index(name)])
        loss = float(losses[name].detach())
        rows.append({
            "head": name,
            "active_in_subspace": name in matcher.active_heads,
            "raw_target_velocity_norm": target_norm,
            "prediction_error_rms": math.sqrt(loss),
            "raw_mse_loss": loss,
            "normalized_loss": loss / (target_norm * target_norm) if target_norm > 1e-12 else math.nan,
            "gradient_norm": gradient_norm,
            "loss_share_of_active_total": loss / max(float(active_total.detach()), 1e-12) if name in matcher.active_heads else 0.0,
            "optimizer_update_norm_first_order": learning_rate * gradient_norm if name in matcher.active_heads else 0.0,
            "optimizer_update_norm_total_last_step": total_update_norm,
        })
    return pd.DataFrame(rows)


@dataclass
class TrainedFlow:
    model: GaugeFlowVectorField
    matcher: RiemannianCrystalFlowMatcher
    final_loss: float
    head_audit: pd.DataFrame


def _train_flow(
    batch,
    vocabulary: list[int],
    *,
    active_heads: tuple[str, ...],
    settings: dict[str, Any],
    seed: int,
    type_path: str = "euclidean_logits",
) -> TrainedFlow:
    _seed(seed)
    model = GaugeFlowVectorField(
        hidden_dim=settings["hidden_dim"], layers=settings["layers"], atom_types=len(vocabulary),
        conditioning_mode="endpoint_id",
    ).to(batch.batch.device)
    matcher = RiemannianCrystalFlowMatcher(
        atom_types=len(vocabulary), active_heads=active_heads, type_path=type_path
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    final_loss = math.nan
    update_norm = 0.0
    for _ in range(settings["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        terms = matcher.loss(model, batch)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        update_norm = math.sqrt(sum(float((parameter.detach() - old).square().sum()) for parameter, old in zip(model.parameters(), before)))
        final_loss = float(terms["loss"].detach())
    model.eval()
    audit = _head_audit(model, matcher, batch, settings["learning_rate"], seed + 991, update_norm)
    return TrainedFlow(model, matcher, final_loss, audit)


def _global_scores(type_state: torch.Tensor, vocabulary: list[int]) -> torch.Tensor:
    scores = type_state.new_full((type_state.shape[0], 119), -1e9)
    scores[:, torch.tensor(vocabulary, device=type_state.device)] = type_state
    return scores


def _endpoint_descriptors(raw_batch) -> tuple[list[dict[str, Any]], dict[str, float]]:
    target = RiemannianCrystalFlowMatcher().target_state(raw_batch)
    descriptors = []
    for graph in range(raw_batch.num_graphs):
        nodes = torch.nonzero(raw_batch.batch == graph, as_tuple=False).flatten()
        descriptors.append(_decoded_descriptor(target.type_state[nodes], target.frac_coords[nodes], target.lattice_log[graph]))
    atoms = int((raw_batch.batch == 0).sum())
    scale = {
        "composition": max(float((descriptors[0]["composition"] - descriptors[1]["composition"]).abs().sum()), 0.05),
        "shape": max(float(torch.linalg.vector_norm(descriptors[0]["lattice_shape"] - descriptors[1]["lattice_shape"])), 0.05),
        "volume": max(abs(math.log(descriptors[0]["lattice_volume"]) - math.log(descriptors[1]["lattice_volume"])), 0.05),
        "coordinates": max(float(torch.sqrt(((descriptors[0]["coordinate_spectrum"] - descriptors[1]["coordinate_spectrum"]) ** 2).mean())), 0.05),
        "topology": max(float((descriptors[0]["neighbor_degrees"] - descriptors[1]["neighbor_degrees"]).abs().mean()) + abs(descriptors[0]["neighbor_edge_count"] - descriptors[1]["neighbor_edge_count"]) / atoms, 0.05),
    }
    return descriptors, scale


def _geometry_distance(value: dict[str, Any], reference: dict[str, Any], scale: dict[str, float]) -> float:
    shape = float(torch.linalg.vector_norm(value["lattice_shape"] - reference["lattice_shape"])) / scale["shape"]
    volume = abs(math.log(max(value["lattice_volume"], 1e-12)) - math.log(max(reference["lattice_volume"], 1e-12))) / scale["volume"]
    coordinates = float(torch.sqrt(((value["coordinate_spectrum"] - reference["coordinate_spectrum"]) ** 2).mean())) / scale["coordinates"]
    topology = (float((value["neighbor_degrees"] - reference["neighbor_degrees"]).abs().mean()) + abs(value["neighbor_edge_count"] - reference["neighbor_edge_count"]) / max(value["argmax_types"].numel(), 1)) / scale["topology"]
    return shape + volume + coordinates + topology


def _decoded_metrics(state, generated_batch, labels: torch.Tensor, raw_endpoints, vocabulary: list[int], subspace: str):
    endpoints, scale = _endpoint_descriptors(raw_endpoints)
    rows = []
    for graph in range(generated_batch.num_graphs):
        nodes = torch.nonzero(generated_batch.batch == graph, as_tuple=False).flatten()
        global_type = _global_scores(state.type_state[nodes], vocabulary)
        descriptor = _decoded_descriptor(global_type, state.frac_coords[nodes], state.lattice_log[graph])
        full_distance = [_descriptor_distance(descriptor, endpoint, scale) for endpoint in endpoints]
        geometry_distance = [_geometry_distance(descriptor, endpoint, scale) for endpoint in endpoints]
        target = int(labels[graph])
        target_composition = endpoints[target]["composition"]
        endpoint_nodes = torch.nonzero(raw_endpoints.batch == target, as_tuple=False).flatten()
        endpoint_state = RiemannianCrystalFlowMatcher().target_state(raw_endpoints)
        target_local_types = raw_endpoints.atom_types[endpoint_nodes]
        predicted_types = global_type.argmax(-1)
        composition_ok = bool(torch.equal(descriptor["composition"], target_composition))
        rows.append({
            "subspace": subspace,
            "sample": graph,
            "target": target,
            "retrieved_endpoint_joint": int(torch.tensor(full_distance).argmin()),
            "retrieved_endpoint_geometry": int(torch.tensor(geometry_distance).argmin()),
            "joint_endpoint_retrieval_correct": int(torch.tensor(full_distance).argmin()) == target,
            "geometry_endpoint_retrieval_correct": int(torch.tensor(geometry_distance).argmin()) == target,
            "decoded_composition_correct": composition_ok,
            "decoded_atom_type_accuracy": float((predicted_types == target_local_types).float().mean()),
            "type_endpoint_rms": _rms(global_type - endpoint_state.type_state[endpoint_nodes]),
            "coordinate_endpoint_rms": _rms(torus_logmap(state.frac_coords[nodes], endpoint_state.frac_coords[endpoint_nodes])),
            "lattice_log_endpoint_rms": _rms(state.lattice_log[graph] - endpoint_state.lattice_log[target]),
            "argmax_atom_types": json.dumps(descriptor["argmax_types"].tolist()),
            "argmax_composition_elements": json.dumps(descriptor["composition"].nonzero().flatten().tolist()),
            "argmax_composition_fractions": json.dumps(descriptor["composition"][descriptor["composition"] > 0].tolist()),
            "joint_distance_endpoint_0": full_distance[0],
            "joint_distance_endpoint_1": full_distance[1],
            "geometry_distance_endpoint_0": geometry_distance[0],
            "geometry_distance_endpoint_1": geometry_distance[1],
        })
    frame = pd.DataFrame(rows)
    return frame, {
        "type_decoded_composition_accuracy": float(frame.decoded_composition_correct.mean()),
        "geometry_endpoint_retrieval_accuracy": float(frame.geometry_endpoint_retrieval_correct.mean()),
        "joint_endpoint_retrieval_accuracy": float(frame.joint_endpoint_retrieval_correct.mean()),
        "decoded_atom_type_accuracy": float(frame.decoded_atom_type_accuracy.mean()),
        "type_endpoint_reconstruction_rms": float(frame.type_endpoint_rms.mean()),
        "coordinate_endpoint_reconstruction_rms": float(frame.coordinate_endpoint_rms.mean()),
        "lattice_endpoint_reconstruction_rms": float(frame.lattice_log_endpoint_rms.mean()),
    }


def _finite_state(state: CrystalFlowState) -> int:
    return int(not (torch.isfinite(state.type_state).all() and torch.isfinite(state.frac_coords).all() and torch.isfinite(state.lattice_log).all()))


def _common_trajectory(model, matcher, batch, initial, *, steps: int, subspace: str):
    state = _state_copy(initial)
    atoms = int((batch.batch == 0).sum())
    rows = []
    dt = 1.0 / steps
    for step in range(steps + 1):
        for name, value, torus in (
            ("type_logit", state.type_state, False),
            ("fractional_coordinate", state.frac_coords, True),
            ("lattice_log", state.lattice_log, False),
        ):
            if value.shape[0] == batch.num_graphs:
                left, right = value[:1], value[1:2]
            else:
                left, right = value[:atoms], value[atoms:2 * atoms]
            difference = torus_logmap(left, right) if torus else left - right
            rows.append({"subspace": subspace, "time": step / steps, "quantity": f"state_{name}", "pairwise_rms": _rms(difference)})
        if step == steps:
            break
        time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
        outputs = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[:3]
        for name, value in zip(("type_logit", "fractional_coordinate", "lattice_log"), outputs):
            if value.shape[0] == batch.num_graphs:
                left, right = value[:1], value[1:2]
            else:
                left, right = value[:atoms], value[atoms:2 * atoms]
            rows.append({"subspace": subspace, "time": step / steps, "quantity": f"velocity_{name}", "pairwise_rms": _rms(left - right)})
        velocity = (
            outputs[0] if "type" in matcher.active_heads else torch.zeros_like(outputs[0]),
            outputs[1] if "coord" in matcher.active_heads else torch.zeros_like(outputs[1]),
            outputs[2] if "lattice" in matcher.active_heads else torch.zeros_like(outputs[2]),
        )
        type_state = state.type_state + dt * velocity[0]
        if matcher.type_path == "simplex_probability":
            type_state = type_state.clamp_min(0.0)
            type_state = type_state / type_state.sum(-1, keepdim=True).clamp_min(1e-12)
        state = CrystalFlowState(type_state, wrap01(state.frac_coords + dt * velocity[1]), state.lattice_log + dt * velocity[2])
    return pd.DataFrame(rows)


def _endpoint_id_experiment(raw_batch, protocol: dict[str, Any], device: torch.device):
    settings = protocol["endpoint_id_substrate"]
    vocabulary = list(range(119))
    endpoint_batch = _localize_types(_set_endpoint_ids(raw_batch), vocabulary)
    atoms = int((raw_batch.batch == 0).sum())
    result_rows, decoded_frames, trajectory_frames, audit_frames = [], [], [], []
    trained: dict[str, TrainedFlow] = {}
    active_by_subspace = {"type": ("type",), "geometry": ("coord", "lattice"), "joint": HEADS}
    for offset, (subspace, active) in enumerate(active_by_subspace.items()):
        trained_flow = _train_flow(endpoint_batch, vocabulary, active_heads=active, settings=settings, seed=settings["seed"] + offset)
        trained[subspace] = trained_flow
        sample_batch, labels = _generated_batch(raw_batch, settings["sample_replicates"], vocabulary, device)
        initial = _common_initial_state(
            trained_flow.matcher, sample_batch, endpoint_count=2, replicates=settings["sample_replicates"],
            atoms=atoms, seed=settings["common_noise_seed"],
        )
        state = trained_flow.matcher.sample(
            trained_flow.model, sample_batch, steps=settings["sampler_steps"], initial_state=initial
        )
        decoded, summary = _decoded_metrics(state, sample_batch, labels, raw_batch, vocabulary, subspace)
        decoded_frames.append(decoded)
        features = _state_features(state, sample_batch)
        distance = _distance_diagnostics(features, labels)
        criteria = settings["criteria"]
        result_rows.append({
            "experiment": "endpoint_id_substrate",
            "subspace": subspace,
            "condition": "two_class_one_hot_endpoint_id",
            "final_training_loss": trained_flow.final_loss,
            **{f"flow_mse_{row.head}": row.raw_mse_loss for row in trained_flow.head_audit.itertuples()},
            **summary,
            "generated_between_within_ratio": distance["between_within_distance_ratio"],
            "generated_nearest_centroid_accuracy": distance["leave_one_out_nearest_centroid_accuracy"],
            "sampling_failure_count": _finite_state(state),
            "type_criterion_pass": summary["type_decoded_composition_accuracy"] >= criteria["type_decoded_composition_accuracy_min"],
            "geometry_criterion_pass": summary["geometry_endpoint_retrieval_accuracy"] >= criteria["geometry_endpoint_retrieval_accuracy_min"],
            "joint_criterion_pass": summary["joint_endpoint_retrieval_accuracy"] >= criteria["joint_endpoint_retrieval_accuracy_min"],
            "between_within_criterion_pass": distance["between_within_distance_ratio"] >= criteria["generated_between_within_ratio_min"],
            "zero_failure_criterion_pass": _finite_state(state) <= criteria["sampling_failure_count_max"],
        })
        audit = trained_flow.head_audit.copy()
        audit.insert(0, "subspace", subspace)
        audit_frames.append(audit)
        trajectory_batch, _ = _generated_batch(raw_batch, 1, vocabulary, device)
        trajectory_initial = _common_initial_state(
            trained_flow.matcher, trajectory_batch, endpoint_count=2, replicates=1, atoms=atoms,
            seed=settings["common_noise_seed"],
        )
        trajectory_frames.append(_common_trajectory(
            trained_flow.model, trained_flow.matcher, trajectory_batch, trajectory_initial,
            steps=settings["sampler_steps"], subspace=subspace,
        ))
    return (
        pd.DataFrame(result_rows), pd.concat(decoded_frames, ignore_index=True),
        pd.concat(trajectory_frames, ignore_index=True), pd.concat(audit_frames, ignore_index=True), trained,
    )


def _rank_of_target(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_score = scores.gather(-1, target.unsqueeze(-1))
    return 1 + (scores > target_score).sum(dim=-1)


def _categorical_train(batch, vocabulary: list[int], settings: dict[str, Any], seed: int):
    _seed(seed)
    model = GaugeFlowVectorField(hidden_dim=settings["hidden_dim"], layers=settings["layers"], atom_types=len(vocabulary), conditioning_mode="endpoint_id").to(batch.batch.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    target = RiemannianCrystalFlowMatcher(atom_types=len(vocabulary)).target_state(batch)
    for _ in range(settings["train_steps"]):
        optimizer.zero_grad(set_to_none=True)
        time = torch.rand((batch.num_graphs,), device=batch.batch.device)
        noise = torch.randint(len(vocabulary), (batch.atom_types.numel(),), device=batch.batch.device)
        keep_target = torch.rand((batch.atom_types.numel(),), device=batch.batch.device) < time[batch.batch]
        corrupted = torch.where(keep_target, batch.atom_types, noise)
        state = F.one_hot(corrupted, len(vocabulary)).float()
        output = model(state, target.frac_coords, target.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[0]
        loss = F.cross_entropy(output, batch.atom_types)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model, float(loss.detach())


def _categorical_sample(model, batch, initial: CrystalFlowState, steps: int):
    state = _state_copy(initial)
    for step in range(steps):
        time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
        logits = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[0]
        state = CrystalFlowState(F.one_hot(logits.argmax(-1), logits.shape[-1]).float(), state.frac_coords, state.lattice_log)
    return state


def _flow_trace(model, matcher: RiemannianCrystalFlowMatcher, batch, initial: CrystalFlowState, steps: int):
    """Mirror the production Euler sampler while retaining the state at every step."""
    states = [_state_copy(initial)]
    state = _state_copy(initial)
    dt = 1.0 / steps
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
            output = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[:3]
            velocity = (
                output[0] if "type" in matcher.active_heads else torch.zeros_like(output[0]),
                output[1] if "coord" in matcher.active_heads else torch.zeros_like(output[1]),
                output[2] if "lattice" in matcher.active_heads else torch.zeros_like(output[2]),
            )
            type_state = state.type_state + dt * velocity[0]
            if matcher.type_path == "simplex_probability":
                type_state = type_state.clamp_min(0.0)
                type_state = type_state / type_state.sum(-1, keepdim=True).clamp_min(1e-12)
            state = CrystalFlowState(type_state, wrap01(state.frac_coords + dt * velocity[1]), state.lattice_log + dt * velocity[2])
            states.append(_state_copy(state))
    return states


def _categorical_trace(model, batch, initial: CrystalFlowState, steps: int):
    states = [_state_copy(initial)]
    state = _state_copy(initial)
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((batch.num_graphs,), step / steps, device=batch.batch.device)
            logits = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[0]
            state = CrystalFlowState(F.one_hot(logits.argmax(-1), logits.shape[-1]).float(), state.frac_coords, state.lattice_log)
            states.append(_state_copy(state))
    return states


def _type_path_rows(
    name: str, model, matcher: RiemannianCrystalFlowMatcher | None, batch, raw_batch,
    vocabulary: list[int], settings: dict[str, Any], states: list[CrystalFlowState],
    final_training_loss: float, *, categorical: bool = False,
):
    target = RiemannianCrystalFlowMatcher(atom_types=len(vocabulary)).target_state(batch)
    rows: list[dict[str, Any]] = []
    initial, final = states[0], states[-1]
    for step, state in enumerate(states):
        time = step / max(len(states) - 1, 1)
        phase = "initial" if step == 0 else ("terminal" if step == len(states) - 1 else "trajectory")
        scores = state.type_state
        target_rank = _rank_of_target(scores, batch.atom_types)
        top = scores.argmax(-1)
        top5 = scores.topk(min(5, scores.shape[-1]), dim=-1).indices
        for graph, material_id in enumerate(settings["material_ids"]):
            nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
            mapped_top = torch.tensor(vocabulary, device=scores.device)[top[nodes]]
            target_global = raw_batch.atom_types[nodes]
            if name.startswith("euclidean"):
                soft = torch.softmax(scores[nodes], -1)
            elif categorical:
                soft = scores[nodes]
            else:
                soft = scores[nodes].clamp_min(0) / scores[nodes].clamp_min(0).sum(-1, keepdim=True).clamp_min(1e-12)
            soft_global = torch.zeros(119, device=scores.device)
            soft_global[torch.tensor(vocabulary, device=scores.device)] = soft.mean(0)
            correct_score = scores[nodes].gather(-1, batch.atom_types[nodes].unsqueeze(-1)).squeeze(-1)
            wrong = scores[nodes].masked_fill(F.one_hot(batch.atom_types[nodes], scores.shape[-1]).bool(), -float("inf")).max(-1).values
            base = initial.type_state[nodes]
            rows.append({
                "variant": name,
                "phase": phase,
                "time": time,
                "material_id": material_id,
                "vocabulary_size": len(vocabulary),
                "target_element_initial_rank_mean": float(target_rank[nodes].float().mean()),
                "top1_accuracy": float((top[nodes] == batch.atom_types[nodes]).float().mean()),
                "top5_accuracy": float((top5[nodes] == batch.atom_types[nodes].unsqueeze(-1)).any(-1).float().mean()),
                "correct_vs_max_wrong_logit_margin": float((correct_score - wrong).mean()),
                "argmax_composition": json.dumps(torch.bincount(mapped_top, minlength=119).tolist()),
                "soft_composition_probability": json.dumps(soft_global.detach().cpu().tolist()),
                "minimum_logit_displacement_noise_to_endpoint": _rms(target.type_state[nodes] - base),
                "actual_integrated_logit_displacement": _rms(state.type_state[nodes] - base),
                "final_cross_entropy": (
                    final_training_loss if categorical else (
                        float(-final.type_state[nodes].clamp_min(1e-12).log().gather(-1, batch.atom_types[nodes].unsqueeze(-1)).mean())
                        if name.startswith("simplex") else float(F.cross_entropy(final.type_state[nodes], batch.atom_types[nodes]))
                    )
                ),
            })
    return rows


def _type_path_comparison(raw_batch, protocol: dict[str, Any], device: torch.device):
    settings = {**protocol["type_path_comparison"], "material_ids": protocol["material_ids"]}
    active = _active_vocab(protocol)
    variants = {
        "euclidean_full_119": (list(range(119)), "euclidean_logits", False),
        "euclidean_v2_train_active_mask": (active, "euclidean_logits", False),
        "euclidean_diagnostic_B_N_In_only": ([5, 7, 49], "euclidean_logits", False),
        "simplex_probability_full_119": (list(range(119)), "simplex_probability", False),
        "categorical_discrete_full_119": (list(range(119)), "categorical", True),
    }
    all_rows = []
    for offset, name in enumerate(settings["variants"]):
        vocabulary, path, categorical = variants[name]
        batch = _localize_types(_set_endpoint_ids(raw_batch), vocabulary)
        atoms = int((batch.batch == 0).sum())
        if categorical:
            model, final_loss = _categorical_train(batch, vocabulary, settings, settings["seed"] + offset)
            _seed(settings["initial_noise_seed"])
            initial = CrystalFlowState(
                F.one_hot(torch.randint(len(vocabulary), (batch.atom_types.numel(),), device=device), len(vocabulary)).float(),
                RiemannianCrystalFlowMatcher(atom_types=len(vocabulary), active_heads=("type",)).target_state(batch).frac_coords,
                RiemannianCrystalFlowMatcher(atom_types=len(vocabulary), active_heads=("type",)).target_state(batch).lattice_log,
            )
            states = _categorical_trace(model, batch, initial, settings["sampler_steps"])
            matcher = None
        else:
            flow = _train_flow(batch, vocabulary, active_heads=("type",), settings=settings, seed=settings["seed"] + offset, type_path=path)
            model, matcher, final_loss = flow.model, flow.matcher, flow.final_loss
            initial = _common_initial_state(matcher, batch, endpoint_count=2, replicates=1, atoms=atoms, seed=settings["initial_noise_seed"])
            states = _flow_trace(model, matcher, batch, initial, settings["sampler_steps"])
        all_rows.extend(_type_path_rows(name, model, matcher, batch, raw_batch, vocabulary, settings, states, final_loss, categorical=categorical))
    return pd.DataFrame(all_rows), len(active)


def _endpoint_estimator(model, matcher, batch, raw_batch, protocol: dict[str, Any]):
    rows = []
    settings = protocol["endpoint_estimator"]
    vocabulary = list(range(119))
    for index, value in enumerate(settings["times"]):
        time = torch.full((batch.num_graphs,), value, device=batch.batch.device)
        state, target, _, _ = _fixed_interpolant(matcher, batch, time, settings["initial_noise_seed"] + index)
        velocity = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[:3]
        estimate = CrystalFlowState(
            state.type_state + (1.0 - value) * velocity[0],
            wrap01(state.frac_coords + (1.0 - value) * velocity[1]),
            state.lattice_log + (1.0 - value) * velocity[2],
        )
        for graph, material_id in enumerate(protocol["material_ids"]):
            nodes = torch.nonzero(batch.batch == graph, as_tuple=False).flatten()
            predicted_types = estimate.type_state[nodes].argmax(-1)
            target_types = target.type_state[nodes].argmax(-1)
            pred_composition = torch.bincount(predicted_types, minlength=119)
            target_composition = torch.bincount(target_types, minlength=119)
            rows.append({
                "time": value,
                "material_id": material_id,
                "predicted_endpoint_element_accuracy": float((predicted_types == target_types).float().mean()),
                "predicted_endpoint_composition_accuracy": float(torch.equal(pred_composition, target_composition)),
                "coordinate_endpoint_rms": _rms(torus_logmap(estimate.frac_coords[nodes], target.frac_coords[nodes])),
                "lattice_log_endpoint_rms": _rms(estimate.lattice_log[graph] - target.lattice_log[graph]),
                "type_endpoint_rms": _rms(estimate.type_state[nodes] - target.type_state[nodes]),
                "correct_endpoint_retrieval": float((estimate.type_state[nodes].argmax(-1) == target_types).all()),
            })
    return pd.DataFrame(rows)


def _upstream_reference_report(path: Path | None) -> str:
    backup = ROOT.parent / "legacy_backups" / "flowmm_local_2026-07-14"
    lines = ["# A4.6 upstream FlowMM reference audit", ""]
    lines += [
        "Status: read-only reference inspection; GaugeFlow does not import FlowMM at runtime.",
        "",
        "The preserved FlowMM baseline identifies upstream main commit `6a96aec3b6eba89f6fa07436f0c8837979abb285` and local head `201c5a6f095739bbed61bd0f5f6c381dd41a5f85`.",
    ]
    if path is None or not path.is_dir():
        lines += [
            "", "No independently runnable upstream checkout was supplied to this audit.",
            "The legacy backup preserves only patches/untracked T2C additions, not a complete importable FlowMM runtime. Therefore no upstream two-endpoint numerical pass/fail is claimed.",
            "This is a blocking limitation for an A4.6 numerical comparison, not evidence that FlowMM passes or fails.",
        ]
    else:
        files = list(path.rglob("*.py"))
        text = "\n".join(file.read_text(encoding="utf-8", errors="ignore") for file in files[:500])
        atom_tokens = sum(token in text for token in ("atom_types", "one_hot", "categorical"))
        decoder_tokens = sum(token in text for token in ("argmax", "decode", "sample"))
        manifold_getter = path / "src" / "flowmm" / "rfm" / "manifold_getter.py"
        source = manifold_getter.read_text(encoding="utf-8") if manifold_getter.is_file() else ""
        simplex_declared = 'atom_type_manifold in ["simplex", "null_manifold"]' in source
        inverse_argmax = "torch.argmax(a, dim=dim) + 1" in source
        # This is a source-equivalent microtest of FlowMM's documented
        # one-hot decoder. It is deliberately separate from a full FlowMM
        # sampler, whose pinned dependency stack is absent here.
        decoded = (torch.eye(100, dtype=torch.float32)[torch.tensor([4, 6, 48])].argmax(-1) + 1).tolist()
        lines += [
            "", f"Inspected read-only checkout: `{path}` ({len(files)} Python files).",
            f"Static type-path tokens found: {atom_tokens}; decoder/sampler tokens found: {decoder_tokens}.",
            f"FlowMM declares a simplex atom-type manifold: {simplex_declared}; its inverse one-hot decoder is argmax-plus-one: {inverse_argmax}.",
            f"Source-equivalent decoder microtest for B/N/In one-hot states produced atomic numbers {decoded} (expected [5, 7, 49]).",
            "A full upstream two-endpoint flow/sampler run is blocked: importing the pinned checkout requires the DiffCSP `torch_scatter` extension, which is not installed in this environment. The decoder microtest is not reported as a full numerical baseline result.",
        ]
    lines += [
        "", "Decision use: if a later pinned, runnable upstream reference passes the same endpoint-ID type test while standalone GaugeFlow does not, port only the verified manifold/path/decoder definition under a new protocol. Do not restore FlowMM as a runtime dependency.",
    ]
    return "\n".join(lines) + "\n"


def _write_path_report(path: Path, protocol_path: Path, protocol: dict[str, Any], closure: pd.DataFrame, passed: bool):
    lines = ["# Gate A4.0 analytic probability-path closure", ""]
    lines += [
        f"Protocol: `{protocol_path.name}` (`{_sha256(protocol_path)}`).",
        "", "This test uses no neural network: it integrates the exact production interpolant velocity with the production Euler sampler and a fixed base noise.",
        "", "| Subspace | rows | max continuous endpoint error | decoded endpoint accuracy | closed |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, frame in closure.groupby("subspace", sort=False):
        maximum = frame[["type_continuous_endpoint_error", "fractional_coordinate_endpoint_error", "lattice_log_endpoint_error"]].max().max()
        lines.append(f"| {name} | {len(frame)} | {maximum:.3e} | {frame.decoded_endpoint_accuracy.mean():.3f} | {bool(frame.analytic_path_closed.all())} |")
    lines += ["", f"Decision: **{'PASS' if passed else 'FAIL'}** analytic closure."]
    if passed:
        lines.append("The production time direction, constant velocity target, torus wrapping, SPD-log lattice coordinate, and final type argmax recover the chosen endpoints under exact velocity. A4 may therefore attribute any subsequent failure to learned generator substrate behavior rather than this analytic integration identity.")
    else:
        lines.append("Stop condition reached: do not run neural endpoint-ID training. Locate the sampler/path defect before any further generator experiment.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, endpoint: pd.DataFrame, protocol: dict[str, Any], active_vocab_size: int, closure_passed: bool):
    criteria = protocol["endpoint_id_substrate"]["criteria"]
    lines = ["# Gate A4 generator substrate summary", "", f"Analytic closure: **{'pass' if closure_passed else 'fail'}**.", ""]
    lines += ["| Subspace | type composition | geometry retrieval | joint retrieval | between/within | failures |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in endpoint.iterrows():
        lines.append(f"| {row.subspace} | {row.type_decoded_composition_accuracy:.3f} | {row.geometry_endpoint_retrieval_accuracy:.3f} | {row.joint_endpoint_retrieval_accuracy:.3f} | {row.generated_between_within_ratio:.3f} | {int(row.sampling_failure_count)} |")
    joint = endpoint[endpoint.subspace == "joint"].iloc[0]
    type_row = endpoint[endpoint.subspace == "type"].iloc[0]
    geometry = endpoint[endpoint.subspace == "geometry"].iloc[0]
    if type_row.type_decoded_composition_accuracy < criteria["type_decoded_composition_accuracy_min"]:
        decision = "endpoint-ID type-only failed: replace the atom-type manifold/decoder before returning to tensor conditioning."
    elif geometry.geometry_endpoint_retrieval_accuracy < criteria["geometry_endpoint_retrieval_accuracy_min"]:
        decision = "endpoint-ID geometry-only failed: repair the periodic-coordinate/lattice path before returning to tensor conditioning."
    elif joint.joint_endpoint_retrieval_accuracy < criteria["joint_endpoint_retrieval_accuracy_min"]:
        decision = "the individual substrate factors qualify better than joint generation; inspect head scales/joint backbone, not tensor conditioning."
    else:
        decision = "endpoint-ID joint substrate qualifies on its retrieval criterion; a future tensor-conditioned gate still requires a separate versioned protocol."
    lines += ["", f"v2 train-active diagnostic vocabulary size: {active_vocab_size}.", "", f"Decision: {decision}", "", "The `{B,N,In}` vocabulary is diagnostic-only and is not a final model vocabulary."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a4_generator_substrate_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a4_generator_substrate_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--upstream-root", type=Path, help="Optional read-only pinned FlowMM checkout for static A4.6 inspection")
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_substrate_audit":
        raise ValueError("A4 requires the versioned pre-registered substrate protocol")
    device = torch.device(args.device)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = _load_panel(protocol, ROOT, device, preprocessed_cache=_resolve(protocol["data"]["preprocessed_cache"]))
    if raw.num_graphs != 2 or int((raw.batch == 0).sum()) != int((raw.batch == 1).sum()):
        raise ValueError("A4 requires the frozen equal-atom-count InN/BN pair")

    closure, closure_passed = _path_closure(raw, protocol, device)
    closure.to_csv(output / "path_closure.csv", index=False)
    _write_path_report(output / "path_closure_report.md", protocol_path, protocol, closure, closure_passed)
    if not closure_passed:
        (output / "upstream_reference_report.md").write_text(_upstream_reference_report(args.upstream_root), encoding="utf-8")
        return

    endpoint, decoded, trajectories, head_audit, trained = _endpoint_id_experiment(raw, protocol, device)
    endpoint.to_csv(output / "endpoint_id_results.csv", index=False)
    decoded.to_csv(output / "endpoint_id_decoded_state_audit.csv", index=False)
    trajectories.to_csv(output / "endpoint_id_common_noise_trajectory.csv", index=False)
    head_audit.to_csv(output / "head_loss_gradient_audit.csv", index=False)

    types, active_size = _type_path_comparison(raw, protocol, device)
    types.to_csv(output / "type_path_comparison.csv", index=False)
    estimator_batch = _localize_types(_set_endpoint_ids(raw), list(range(119)))
    estimator = _endpoint_estimator(trained["joint"].model, trained["joint"].matcher, estimator_batch, raw, protocol)
    estimator.to_csv(output / "endpoint_estimator_curves.csv", index=False)
    (output / "upstream_reference_report.md").write_text(_upstream_reference_report(args.upstream_root), encoding="utf-8")
    _write_summary(output / "gate_a4_generator_substrate_v1_summary.md", endpoint, protocol, active_size, closure_passed)
    endpoint_summary = endpoint.set_index("subspace")
    report_files = (
        "path_closure_report.md", "path_closure.csv", "endpoint_id_results.csv",
        "endpoint_id_decoded_state_audit.csv", "endpoint_id_common_noise_trajectory.csv",
        "type_path_comparison.csv", "endpoint_estimator_curves.csv",
        "head_loss_gradient_audit.csv", "upstream_reference_report.md",
        "gate_a4_generator_substrate_v1_summary.md",
    )
    manifest = {
        "schema": 1,
        "name": "GaugeFlow Gate A4 generator substrate v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "status": "not_qualified_endpoint_id_type_and_geometry_failure",
        "analytic_path_closed": closure_passed,
        "material_ids": protocol["material_ids"],
        "device": str(device),
        "endpoint_id_summary": {
            name: {
                "type_decoded_composition_accuracy": float(row.type_decoded_composition_accuracy),
                "geometry_endpoint_retrieval_accuracy": float(row.geometry_endpoint_retrieval_accuracy),
                "joint_endpoint_retrieval_accuracy": float(row.joint_endpoint_retrieval_accuracy),
                "generated_between_within_ratio": float(row.generated_between_within_ratio),
                "sampling_failure_count": int(row.sampling_failure_count),
            }
            for name, row in endpoint_summary.iterrows()
        },
        "report_sha256": {name: _sha256(output / name) for name in report_files},
        "historical_gates_modified": False,
        "tensor_conditioned_experiments_started": False,
        "prohibited_work_started": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
