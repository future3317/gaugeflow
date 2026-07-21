"""Qualify exact no-padding two-GPU assignment pretraining on real Alex data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel

from gaugeflow.file_utils import canonical_json_hash, load_json_object
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.assignment_pretraining import (
    compile_masked_assignment_batch,
    ddp_global_mean_loss,
    sample_rank_sharded_reveal_ranks,
)
from gaugeflow.production.assignment_training import (
    OrderlessAssignmentTrainingModule,
    orderless_assignment_objective,
    sample_uniform_reveal_ranks,
)
from gaugeflow.production.autoregressive_assignment import GeometryAwareRemainingCountScorer


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_sha256(index: torch.Tensor) -> str:
    return hashlib.sha256(index.contiguous().numpy().tobytes()).hexdigest()


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("assignment DDP qualification requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if not torch.cuda.is_available() or rank < 0 or local_rank < 0:
        raise RuntimeError("launch the assignment DDP qualification with torchrun")
    if world_size != expected_world_size:
        raise RuntimeError("assignment DDP qualification world size changed")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def _allowed_indices(dataset: PackedAlexP1Dataset, roles: dict[str, Any]) -> torch.Tensor:
    gold = {str(row["material_id"]) for row in roles["carrier_rows"]}
    return torch.tensor(
        [
            index
            for index, material_id in enumerate(dataset.material_ids_audit_only)
            if material_id not in gold
        ],
        dtype=torch.long,
    )


def _model(protocol: dict[str, Any], device: torch.device) -> GeometryAwareRemainingCountScorer:
    config = protocol["model"]
    return GeometryAwareRemainingCountScorer(
        site_feature_dim=int(config["site_feature_dim"]),
        graph_feature_dim=int(config["graph_feature_dim"]),
        radial_channels=int(config["pair_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        message_blocks=int(config["message_blocks"]),
        maximum_sites=int(config["maximum_sites"]),
        maximum_cell_index=int(config["maximum_cell_index"]),
    ).to(device)


def _flat_parameters(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in module.parameters()])


def _flat_gradients(module: torch.nn.Module) -> torch.Tensor:
    gradients = []
    for parameter in module.parameters():
        if parameter.grad is None:
            raise RuntimeError("assignment software qualification found an unused parameter")
        gradients.append(parameter.grad.detach().reshape(-1).cpu())
    return torch.cat(gradients)


def _compile(
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    device: torch.device,
    maximum_candidates: int,
):
    selected = dataset.select_model_batch(indices, device=device)
    return compile_masked_assignment_batch(
        selected.fractional_coordinates,
        selected.lattice,
        selected.batch,
        selected.atom_types,
        maximum_refinement_candidates=maximum_candidates,
    ).carrier


def _reference_step(
    protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    initial_state: dict[str, torch.Tensor],
    global_indices: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    training = protocol["training"]
    model = _model(protocol, device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    path_samples = int(training["path_samples_per_carrier"])
    microbatch = int(training["global_graph_microbatch_size"])
    denominator = global_indices.numel() * path_samples
    generator = torch.Generator().manual_seed(seed)
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, global_indices.numel(), microbatch):
        micro = global_indices[start : start + microbatch]
        carrier = _compile(
            dataset,
            micro,
            device=device,
            maximum_candidates=int(training["maximum_refinement_candidates_per_pair"]),
        )
        counts = dataset.node_counts[micro]
        graph = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        for _ in range(path_samples):
            reveal = sample_uniform_reveal_ranks(graph, generator=generator).to(device)
            nll = orderless_assignment_objective(model, carrier, reveal_rank=reveal).graph_nll
            (nll.sum() / denominator).backward()
    gradient = _flat_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
    before = _flat_parameters(model)
    optimizer.step()
    update = _flat_parameters(model) - before
    del optimizer, model
    torch.cuda.empty_cache()
    return gradient, update


class _DdpRun:
    def __init__(
        self,
        protocol: dict[str, Any],
        initial_state: dict[str, torch.Tensor],
        *,
        seed: int,
        device: torch.device,
    ) -> None:
        self.protocol = protocol
        scorer = _model(protocol, device)
        scorer.load_state_dict(initial_state)
        self.module = OrderlessAssignmentTrainingModule(scorer)
        training = protocol["training"]
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        self.generator = torch.Generator().manual_seed(seed)
        self.ddp = DistributedDataParallel(
            self.module,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=True,
        )

    def step(
        self,
        dataset: PackedAlexP1Dataset,
        global_indices: torch.Tensor,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        capture_gradient: bool = False,
    ) -> tuple[float, float, torch.Tensor | None]:
        training = self.protocol["training"]
        path_samples = int(training["path_samples_per_carrier"])
        microbatch = int(training["global_graph_microbatch_size"])
        global_paths = global_indices.numel() * path_samples
        self.optimizer.zero_grad(set_to_none=True)
        local_nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, global_indices.numel(), microbatch):
            global_micro = global_indices[start : start + microbatch]
            if global_micro.numel() < world_size:
                raise RuntimeError("software qualification produced an empty DDP rank")
            local_indices = global_micro[rank::world_size]
            carrier = _compile(
                dataset,
                local_indices,
                device=device,
                maximum_candidates=int(training["maximum_refinement_candidates_per_pair"]),
            )
            counts = dataset.node_counts[global_micro]
            for _ in range(path_samples):
                reveal = sample_rank_sharded_reveal_ranks(
                    counts,
                    rank=rank,
                    world_size=world_size,
                    generator=self.generator,
                    device=device,
                )
                local_nll = self.ddp(carrier, reveal)
                local_nll_sum += local_nll.detach().to(torch.float64).sum()
                ddp_global_mean_loss(
                    local_nll,
                    global_count=global_paths,
                    world_size=world_size,
                ).backward()
        gradient = _flat_gradients(self.module) if capture_gradient else None
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.module.parameters(),
            float(training["gradient_clip_norm"]),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("assignment DDP software qualification found a nonfinite gradient")
        self.optimizer.step()
        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        return float(local_nll_sum / global_paths), float(gradient_norm), gradient

    def state(self) -> dict[str, Any]:
        return {
            "model": copy.deepcopy(self.module.scorer.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "generator": self.generator.get_state(),
        }

    def load(self, state: dict[str, Any]) -> None:
        self.module.scorer.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(functional.cosine_similarity(left.double(), right.double(), dim=0))


def _run_four_steps(
    protocol: dict[str, Any],
    dataset: PackedAlexP1Dataset,
    initial_state: dict[str, torch.Tensor],
    batches: list[torch.Tensor],
    *,
    seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
    interrupt_after_two: bool,
) -> torch.Tensor:
    run = _DdpRun(protocol, initial_state, seed=seed, device=device)
    for batch in batches[:2]:
        run.step(dataset, batch, rank=rank, world_size=world_size, device=device)
    if interrupt_after_two:
        state = run.state()
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location=device, weights_only=False)
        del run
        torch.cuda.empty_cache()
        run = _DdpRun(protocol, initial_state, seed=seed, device=device)
        run.load(state)
    for batch in batches[2:]:
        run.step(dataset, batch, rank=rank, world_size=world_size, device=device)
    result = _flat_parameters(run.module)
    del run
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_assignment_ddp_software_qualification_v1":
        raise ValueError("unexpected assignment DDP software protocol")
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("assignment DDP software protocol is not frozen")
    commit = _git_identity(repository)
    source = protocol["source"]
    capacity_path = repository / source["capacity_result"]
    if _normalized_sha256(capacity_path) != source["capacity_result_normalized_sha256"]:
        raise ValueError("assignment capacity result identity changed")
    capacity = load_json_object(capacity_path)
    if capacity.get("qualified") is not True:
        raise ValueError("assignment capacity screen is not qualified")
    if int(capacity["selection"]["selected_hidden_dim"]) != int(protocol["model"]["hidden_dim"]):
        raise ValueError("assignment DDP model width differs from the capacity result")
    role_path = repository / source["iid_role_result"]
    if _normalized_sha256(role_path) != source["iid_role_result_normalized_sha256"]:
        raise ValueError("assignment role source identity changed")
    roles = load_json_object(role_path)

    training = protocol["training"]
    rank, world_size, device = _initialize_distributed(int(training["world_size"]))
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = PackedAlexP1Dataset(
        args.cache_root,
        "train",
        include_material_id=True,
        verify_hashes=True,
    )
    allowed = _allowed_indices(dataset, roles)
    if _index_sha256(allowed) != source["allowed_index_sha256"]:
        raise ValueError("assignment DDP allowed-index identity changed")
    permutation = allowed[
        torch.randperm(allowed.numel(), generator=torch.Generator().manual_seed(seed))
    ]
    structures = int(training["structures"])
    panel = permutation[:structures]
    batch_size = int(training["global_graph_batch_size"])
    updates = math.ceil(structures / batch_size)
    if updates != int(training["updates"]):
        raise ValueError("assignment DDP update count changed")
    if torch.unique(panel).numel() != panel.numel():
        raise RuntimeError("assignment DDP panel contains duplicate structures")
    gold = {str(row["material_id"]) for row in roles["carrier_rows"]}
    leakage = len(gold.intersection(dataset.material_ids_audit_only[index] for index in panel))

    initial_model = _model(protocol, torch.device("cpu"))
    initial_state = copy.deepcopy(initial_model.state_dict())
    initial_flat = _flat_parameters(initial_model)
    del initial_model
    first_batch = panel[:batch_size]

    reference_gradient: torch.Tensor | None = None
    reference_update: torch.Tensor | None = None
    if rank == 0:
        reference_gradient, reference_update = _reference_step(
            protocol,
            dataset,
            initial_state,
            first_batch,
            seed=seed + 11,
            device=device,
        )
    dist.barrier()
    ddp_one = _DdpRun(protocol, initial_state, seed=seed + 11, device=device)
    before = _flat_parameters(ddp_one.module)
    _, _, ddp_gradient = ddp_one.step(
        dataset,
        first_batch,
        rank=rank,
        world_size=world_size,
        device=device,
        capture_gradient=True,
    )
    ddp_update = _flat_parameters(ddp_one.module) - before
    del ddp_one
    torch.cuda.empty_cache()
    if rank == 0:
        if reference_gradient is None or reference_update is None or ddp_gradient is None:
            raise RuntimeError("rank zero lost the one-step reference")
        gradient_cosine = _cosine(reference_gradient, ddp_gradient)
        update_cosine = _cosine(reference_update, ddp_update)
        update_relative_l2 = float(
            torch.linalg.vector_norm(ddp_update - reference_update)
            / torch.linalg.vector_norm(reference_update).clamp_min(1e-12)
        )
    else:
        gradient_cosine = update_cosine = update_relative_l2 = 0.0
    dist.barrier()

    four_batches = [panel[start : start + batch_size] for start in range(0, 4 * batch_size, batch_size)]
    continuous = _run_four_steps(
        protocol,
        dataset,
        initial_state,
        four_batches,
        seed=seed + 29,
        rank=rank,
        world_size=world_size,
        device=device,
        interrupt_after_two=False,
    )
    resumed = _run_four_steps(
        protocol,
        dataset,
        initial_state,
        four_batches,
        seed=seed + 29,
        rank=rank,
        world_size=world_size,
        device=device,
        interrupt_after_two=True,
    )
    resume_max_abs = float((continuous - resumed).abs().max())
    dist.barrier()

    run = _DdpRun(protocol, initial_state, seed=seed + 47, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()
    losses: list[float] = []
    finite_updates = 0
    local_processed = 0
    for update in range(updates):
        batch = panel[update * batch_size : min((update + 1) * batch_size, structures)]
        loss, gradient_norm, _ = run.step(
            dataset,
            batch,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        losses.append(loss)
        finite_updates += int(math.isfinite(gradient_norm))
        local_processed += batch[rank::world_size].numel()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    processed = torch.tensor(local_processed, dtype=torch.long, device=device)
    dist.all_reduce(processed, op=dist.ReduceOp.SUM)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    devices: list[str | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(torch.cuda.get_device_name(device), devices, dst=0)
    del run

    if rank == 0:
        equivalence = protocol["equivalence"]
        acceptance = protocol["acceptance"]
        checks = {
            "one_step_gradient_equivalence": gradient_cosine
            >= float(equivalence["one_step_gradient_cosine_min"]),
            "one_step_update_equivalence": update_cosine
            >= float(equivalence["one_step_update_cosine_min"])
            and update_relative_l2 <= float(equivalence["one_step_update_relative_l2_max"]),
            "resume_equivalence": resume_max_abs
            <= float(equivalence["resume_parameter_max_abs_error_max"]),
            "exact_unique_coverage": int(processed) == structures,
            "finite_gradient_updates": finite_updates
            == int(acceptance["finite_gradient_updates"]),
            "loss_descent": losses[-1] / losses[0]
            <= float(acceptance["final_to_initial_nll_max"]),
            "throughput": structures / float(elapsed)
            >= float(acceptance["global_graphs_per_second_min"]),
            "peak_memory": float(peak) / (1024**2)
            <= float(acceptance["peak_cuda_mib_per_rank_max"]),
            "zero_gold_leakage": leakage == int(acceptance["gold_leakage"]),
        }
        qualified = all(checks.values())
        result = {
            "protocol": protocol["protocol"],
            "protocol_sha256": canonical_json_hash(protocol),
            "implementation_commit": commit,
            "qualified": qualified,
            "checks": checks,
            "equivalence": {
                "one_step_gradient_cosine": gradient_cosine,
                "one_step_update_cosine": update_cosine,
                "one_step_update_relative_l2": update_relative_l2,
                "resume_parameter_max_abs_error": resume_max_abs,
            },
            "smoke": {
                "structures": structures,
                "processed_structures": int(processed),
                "updates": updates,
                "initial_nll": losses[0],
                "second_update_nll": losses[1],
                "final_nll": losses[-1],
                "final_to_initial_nll": losses[-1] / losses[0],
                "global_graphs_per_second": structures / float(elapsed),
                "peak_cuda_mib_per_rank": float(peak) / (1024**2),
                "gold_leakage": leakage,
                "panel_index_sha256": _index_sha256(panel),
            },
            "model": {
                "parameters": initial_flat.numel(),
                "hidden_dim": int(protocol["model"]["hidden_dim"]),
            },
            "hardware": {
                "devices": devices,
                "physical_cuda_devices": training["physical_cuda_devices"],
                "world_size": world_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": "float32",
            },
            "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
            "boundary": protocol["boundary"],
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
