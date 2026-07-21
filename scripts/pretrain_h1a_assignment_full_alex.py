"""Two-GPU exact-pass masked-occupation pretraining on Alex-MP-20."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.assignment_pretraining import (
    compile_masked_assignment_batch,
    ddp_global_mean_loss,
    rank_shard_of_global_batch,
    sample_rank_sharded_reveal_ranks,
)
from gaugeflow.production.assignment_training import OrderlessAssignmentTrainingModule
from gaugeflow.production.autoregressive_assignment import GeometryAwareRemainingCountScorer


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_identity(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("full-Alex assignment pretraining requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initialize_distributed(expected_world_size: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("full-Alex assignment pretraining requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or local_rank < 0 or world_size != expected_world_size:
        raise RuntimeError("launch full-Alex assignment pretraining with the frozen torchrun world size")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def _allowed_indices(
    dataset: PackedAlexP1Dataset,
    role_result: dict[str, Any],
) -> torch.Tensor:
    gold = {str(row["material_id"]) for row in role_result["carrier_rows"]}
    allowed = torch.tensor(
        [
            index
            for index, material_id in enumerate(dataset.material_ids_audit_only)
            if material_id not in gold
        ],
        dtype=torch.long,
    )
    if allowed.numel() < 1 or torch.unique(allowed).numel() != allowed.numel():
        raise ValueError("full-Alex allowed-index set is empty or duplicated")
    return allowed


def _index_sha256(index: torch.Tensor) -> str:
    return hashlib.sha256(index.contiguous().numpy().tobytes()).hexdigest()


def _build_model(protocol: dict[str, Any]) -> GeometryAwareRemainingCountScorer:
    model = protocol["model"]
    return GeometryAwareRemainingCountScorer(
        site_feature_dim=int(model["site_feature_dim"]),
        graph_feature_dim=int(model["graph_feature_dim"]),
        radial_channels=int(model["pair_feature_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        message_blocks=int(model["message_blocks"]),
        maximum_sites=int(model["maximum_sites"]),
        maximum_cell_index=int(model["maximum_cell_index"]),
    )


def _gather_shared_generator_state(
    generator: torch.Generator,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor | None:
    local = generator.get_state().cpu()
    gathered: list[torch.Tensor | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if gathered is None:
        return None
    if any(value is None or not torch.equal(value, gathered[0]) for value in gathered):
        raise RuntimeError("rank reveal-order generator states diverged")
    return gathered[0]


def _save_checkpoint(
    path: Path,
    *,
    module: OrderlessAssignmentTrainingModule,
    optimizer: torch.optim.Optimizer,
    next_update: int,
    history: list[dict[str, float]],
    elapsed_seconds: float,
    generator_state: torch.Tensor | None,
    protocol_sha256: str,
    implementation_commit: str,
    allowed_index_sha256: str,
    permutation_sha256: str,
) -> None:
    if generator_state is None:
        raise ValueError("rank-zero checkpoint is missing the reveal-order generator state")
    payload = {
        "schema": 2,
        "task": "full_alex_masked_assignment_pretraining",
        "protocol_sha256": protocol_sha256,
        "implementation_commit": implementation_commit,
        "allowed_index_sha256": allowed_index_sha256,
        "permutation_sha256": permutation_sha256,
        "next_update": next_update,
        "elapsed_seconds": elapsed_seconds,
        "model": module.scorer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "reveal_generator_state": generator_state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    module: OrderlessAssignmentTrainingModule,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    protocol_sha256: str,
    implementation_commit: str,
    allowed_index_sha256: str,
    permutation_sha256: str,
    device: torch.device,
) -> tuple[int, list[dict[str, float]], float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "schema": 2,
        "task": "full_alex_masked_assignment_pretraining",
        "protocol_sha256": protocol_sha256,
        "implementation_commit": implementation_commit,
        "allowed_index_sha256": allowed_index_sha256,
        "permutation_sha256": permutation_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("assignment pretraining checkpoint identity mismatch")
    state = payload.get("reveal_generator_state")
    if not isinstance(state, torch.Tensor):
        raise ValueError("assignment pretraining checkpoint has no reveal-order RNG state")
    module.scorer.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(state.cpu())
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("assignment pretraining checkpoint history is invalid")
    return int(payload["next_update"]), history, float(payload["elapsed_seconds"])


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("assignment pretraining history is empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_assignment_full_alex_pretraining_v1":
        raise ValueError("unexpected full-Alex assignment pretraining protocol")
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("full-Alex assignment pretraining protocol is not frozen")
    implementation_commit = _git_identity(repository)
    protocol_sha256 = canonical_json_hash(protocol)

    source = protocol["source"]
    interface_path = repository / source["qualified_interface_result"]
    if _normalized_sha256(interface_path) != source["qualified_interface_normalized_sha256"]:
        raise ValueError("qualified full-Alex interface identity changed")
    interface = load_json_object(interface_path)
    if interface.get("qualified") is not True or not all(interface["checks"].values()):
        raise ValueError("full-Alex assignment interface is not qualified")
    role_path = repository / source["iid_role_result"]
    if _normalized_sha256(role_path) != source["iid_role_result_normalized_sha256"]:
        raise ValueError("assignment role source identity changed")
    roles = load_json_object(role_path)
    if roles.get("qualified") is not True:
        raise ValueError("assignment role source is unqualified")

    training = protocol["training"]
    rank, world_size, device = _initialize_distributed(int(training["world_size"]))
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    dataset = PackedAlexP1Dataset(
        args.cache_root,
        "train",
        include_material_id=True,
        verify_hashes=True,
    )
    allowed = _allowed_indices(dataset, roles)
    allowed_sha256 = _index_sha256(allowed)
    if allowed_sha256 != source["allowed_index_sha256"]:
        raise ValueError("full-Alex allowed-index identity changed")
    permutation_generator = torch.Generator().manual_seed(seed)
    permutation = allowed[torch.randperm(allowed.numel(), generator=permutation_generator)]
    permutation_sha256 = _index_sha256(permutation)
    if not torch.equal(torch.sort(permutation).values, torch.sort(allowed).values):
        raise RuntimeError("training permutation is not one exact allowed-index pass")

    global_batch_size = int(training["global_graph_batch_size"])
    global_microbatch_size = int(training["global_graph_microbatch_size"])
    updates = math.ceil(allowed.numel() / global_batch_size)
    if updates != int(training["updates"]) or int(training["exact_passes"]) != 1:
        raise ValueError("frozen updates do not equal one exact full-Alex pass")
    if (
        global_microbatch_size < world_size
        or global_batch_size % global_microbatch_size != 0
    ):
        raise ValueError("global assignment microbatch is incompatible with DDP")
    path_samples = int(training["path_samples_per_carrier"])
    maximum_candidates = int(protocol["geometry"]["maximum_refinement_candidates_per_pair"])

    scorer = _build_model(protocol).to(device)
    module = OrderlessAssignmentTrainingModule(scorer)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    reveal_generator = torch.Generator().manual_seed(seed + 100_003)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    history: list[dict[str, float]] = []
    start_update = 0
    elapsed_before_resume = 0.0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        start_update, history, elapsed_before_resume = _load_checkpoint(
            checkpoint_path,
            module=module,
            optimizer=optimizer,
            generator=reveal_generator,
            protocol_sha256=protocol_sha256,
            implementation_commit=implementation_commit,
            allowed_index_sha256=allowed_sha256,
            permutation_sha256=permutation_sha256,
            device=device,
        )
    ddp = DistributedDataParallel(
        module,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    interval = int(training["history_interval"])
    checkpoint_interval = int(training["checkpoint_interval"])
    finite_gradient_updates = start_update
    local_processed_graphs = sum(
        rank_shard_of_global_batch(
            permutation,
            update=update,
            global_batch_size=global_batch_size,
            rank=rank,
            world_size=world_size,
        ).numel()
        for update in range(start_update)
    )
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter() - elapsed_before_resume
    for update in range(start_update, updates):
        global_start = update * global_batch_size
        global_indices = permutation[
            global_start : min(global_start + global_batch_size, allowed.numel())
        ]
        global_graphs = global_indices.numel()
        global_paths = global_graphs * path_samples

        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        backward_calls = math.ceil(global_graphs / global_microbatch_size) * path_samples
        backward_call = 0
        for micro_start in range(0, global_graphs, global_microbatch_size):
            micro_global = global_indices[
                micro_start : micro_start + global_microbatch_size
            ]
            if micro_global.numel() < world_size:
                raise RuntimeError("a no-padding DDP microbatch cannot cover every rank")
            selected_indices = micro_global[rank::world_size]
            selected = dataset.select_model_batch(selected_indices, device=device)
            carrier = compile_masked_assignment_batch(
                selected.fractional_coordinates,
                selected.lattice,
                selected.batch,
                selected.atom_types,
                maximum_refinement_candidates=maximum_candidates,
            ).carrier
            global_node_counts = dataset.node_counts[micro_global]
            for _ in range(path_samples):
                reveal_rank = sample_rank_sharded_reveal_ranks(
                    global_node_counts,
                    rank=rank,
                    world_size=world_size,
                    generator=reveal_generator,
                    device=device,
                )
                backward_call += 1
                synchronization = nullcontext() if backward_call == backward_calls else ddp.no_sync()
                with synchronization:
                    local_nll = ddp(carrier, reveal_rank)
                    nll_sum += local_nll.detach().to(torch.float64).sum()
                    loss = ddp_global_mean_loss(
                        local_nll,
                        global_count=global_paths,
                        world_size=world_size,
                    )
                    loss.backward()
            local_processed_graphs += selected_indices.numel()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            ddp.module.parameters(),
            float(training["gradient_clip_norm"]),
        )
        finite_gradient = torch.isfinite(gradient_norm)
        if not bool(finite_gradient):
            raise RuntimeError("full-Alex assignment pretraining produced a nonfinite gradient")
        finite_gradient_updates += 1
        optimizer.step()

        completed = update + 1
        record = completed == 1 or completed % interval == 0 or completed == updates
        if record:
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            elapsed = time.perf_counter() - started
            if rank == 0:
                history.append(
                    {
                        "update": float(completed),
                        "graph_exposures": float(min(completed * global_batch_size, allowed.numel())),
                        "train_order_elbo_nll": float(nll_sum / global_paths),
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": elapsed,
                    }
                )
        save = completed % checkpoint_interval == 0 or completed == updates
        if save:
            generator_state = _gather_shared_generator_state(
                reveal_generator,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                _save_checkpoint(
                    checkpoint_path,
                    module=ddp.module,
                    optimizer=optimizer,
                    next_update=completed,
                    history=history,
                    elapsed_seconds=time.perf_counter() - started,
                    generator_state=generator_state,
                    protocol_sha256=protocol_sha256,
                    implementation_commit=implementation_commit,
                    allowed_index_sha256=allowed_sha256,
                    permutation_sha256=permutation_sha256,
                )
            dist.barrier()

    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    processed = torch.tensor(local_processed_graphs, dtype=torch.long, device=device)
    dist.all_reduce(processed, op=dist.ReduceOp.SUM)
    finite_updates = torch.tensor(finite_gradient_updates, dtype=torch.long, device=device)
    dist.all_reduce(finite_updates, op=dist.ReduceOp.MIN)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    devices: list[str | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(torch.cuda.get_device_name(device), devices, dst=0)

    if rank == 0:
        initial_nll = float(history[0]["train_order_elbo_nll"])
        final_nll = float(history[-1]["train_order_elbo_nll"])
        acceptance = protocol["acceptance"]
        checks = {
            "exact_graph_coverage": int(processed) == allowed.numel(),
            "finite_gradient_updates": int(finite_updates) == updates,
            "loss_descent": final_nll / initial_nll <= float(acceptance["final_to_initial_nll_max"]),
            "throughput": allowed.numel() / float(elapsed)
            >= float(acceptance["global_graphs_per_second_min"]),
            "peak_memory": float(peak) / (1024**2)
            <= float(acceptance["peak_cuda_mib_per_rank_max"]),
            "checkpoint": checkpoint_path.is_file(),
        }
        result = {
            "protocol": protocol["protocol"],
            "protocol_sha256": protocol_sha256,
            "implementation_commit": implementation_commit,
            "qualified": all(checks.values()),
            "checks": checks,
            "training": {
                "unique_graphs": allowed.numel(),
                "processed_graphs": int(processed),
                "updates": updates,
                "path_samples_per_graph": path_samples,
                "graph_path_exposures": allowed.numel() * path_samples,
                "initial_order_elbo_nll": initial_nll,
                "final_order_elbo_nll": final_nll,
                "final_to_initial_nll": final_nll / initial_nll,
                "elapsed_seconds": float(elapsed),
                "global_graphs_per_second": allowed.numel() / float(elapsed),
                "peak_cuda_mib_per_rank": float(peak) / (1024**2),
                "finite_gradient_updates": int(finite_updates),
            },
            "data": {
                "allowed_index_sha256": allowed_sha256,
                "permutation_sha256": permutation_sha256,
                "excluded_gold_train_rows": len(dataset) - allowed.numel(),
            },
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": sha256_file(checkpoint_path),
                "selection": "final_exact_pass_without_calibration_peeking",
            },
            "hardware": {
                "devices": devices,
                "world_size": world_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": "float32",
            },
            "decision": protocol["decision_rule"]["pass" if all(checks.values()) else "fail"],
            "boundary": protocol["boundary"],
        }
        _write_history(args.output_dir / "history.csv", history)
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
