"""Frozen no-Gold capacity screen for full-Alex assignment pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.assignment_pretraining import (
    compile_masked_assignment_batch,
)
from gaugeflow.production.assignment_training import orderless_assignment_objective
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
        raise ValueError("assignment capacity screen requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _model(protocol: dict[str, Any], hidden_dim: int, device: torch.device):
    config = protocol["model"]
    return GeometryAwareRemainingCountScorer(
        site_feature_dim=int(config["site_feature_dim"]),
        graph_feature_dim=int(config["graph_feature_dim"]),
        radial_channels=int(config["pair_feature_dim"]),
        hidden_dim=hidden_dim,
        message_blocks=int(config["message_blocks"]),
        maximum_sites=int(config["maximum_sites"]),
        maximum_cell_index=int(config["maximum_cell_index"]),
    ).to(device)


@torch.no_grad()
def _fixed_validation_nll(
    model: GeometryAwareRemainingCountScorer,
    dataset: PackedAlexP1Dataset,
    indices: torch.Tensor,
    *,
    batch_size: int,
    order_samples: int,
    seed: int,
    maximum_candidates: int,
    device: torch.device,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for start in range(0, indices.numel(), batch_size):
        selected = dataset.select_model_batch(indices[start : start + batch_size], device=device)
        carrier = compile_masked_assignment_batch(
            selected.fractional_coordinates,
            selected.lattice,
            selected.batch,
            selected.atom_types,
            maximum_refinement_candidates=maximum_candidates,
        ).carrier
        for _ in range(order_samples):
            objective = orderless_assignment_objective(model, carrier, generator=generator)
            total += objective.graph_nll.to(torch.float64).sum()
            count += carrier.graph_count
    return float(total / count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_assignment_full_alex_capacity_screen_v2":
        raise ValueError("unexpected assignment capacity-screen protocol")
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("assignment capacity-screen protocol is not frozen")
    commit = _git_identity(repository)
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
    seed = int(training["seed"])
    device = torch.device("cuda", int(training["cuda_device"]))
    torch.cuda.set_device(device)
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
        raise ValueError("capacity-screen allowed-index identity changed")
    permutation = allowed[
        torch.randperm(allowed.numel(), generator=torch.Generator().manual_seed(seed))
    ]
    train_count = int(training["train_structures"])
    validation_count = int(training["validation_structures"])
    train_indices = permutation[:train_count]
    validation_indices = permutation[train_count : train_count + validation_count]
    batch_size = int(training["graph_batch_size"])
    microbatch_size = int(training["graph_microbatch_size"])
    if train_count != int(training["updates"]) * batch_size:
        raise ValueError("capacity screen must use every fit structure exactly once")
    if microbatch_size < 1 or batch_size % microbatch_size != 0:
        raise ValueError("capacity screen microbatch must divide the graph batch")
    maximum_candidates = int(training["maximum_refinement_candidates_per_pair"])

    evaluation = protocol["evaluation"]
    rows: list[dict[str, float | int | bool]] = []
    for hidden_dim in map(int, protocol["model"]["hidden_dimensions"]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = _model(protocol, hidden_dim, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        initial_validation = _fixed_validation_nll(
            model,
            dataset,
            validation_indices,
            batch_size=int(evaluation["graph_batch_size"]),
            order_samples=int(evaluation["order_samples"]),
            seed=int(evaluation["fixed_reveal_seed"]),
            maximum_candidates=maximum_candidates,
            device=device,
        )
        generator = torch.Generator(device=device).manual_seed(seed + hidden_dim)
        finite_updates = 0
        first_train_nll = 0.0
        last_train_nll = 0.0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model.train()
        for update in range(int(training["updates"])):
            begin = update * batch_size
            optimizer.zero_grad(set_to_none=True)
            path_samples = int(training["path_samples_per_carrier"])
            nll_sum = torch.zeros((), dtype=torch.float64, device=device)
            for micro_start in range(0, batch_size, microbatch_size):
                micro_indices = train_indices[
                    begin + micro_start : begin + micro_start + microbatch_size
                ]
                selected = dataset.select_model_batch(micro_indices, device=device)
                carrier = compile_masked_assignment_batch(
                    selected.fractional_coordinates,
                    selected.lattice,
                    selected.batch,
                    selected.atom_types,
                    maximum_refinement_candidates=maximum_candidates,
                ).carrier
                for _ in range(path_samples):
                    objective = orderless_assignment_objective(
                        model,
                        carrier,
                        generator=generator,
                    )
                    nll_sum += objective.graph_nll.detach().to(torch.float64).sum()
                    (objective.graph_nll.sum() / (batch_size * path_samples)).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training["gradient_clip_norm"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("capacity screen produced a nonfinite gradient")
            finite_updates += 1
            optimizer.step()
            observed = float(nll_sum / (batch_size * path_samples))
            if update == 0:
                first_train_nll = observed
            last_train_nll = observed
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        final_validation = _fixed_validation_nll(
            model,
            dataset,
            validation_indices,
            batch_size=int(evaluation["graph_batch_size"]),
            order_samples=int(evaluation["order_samples"]),
            seed=int(evaluation["fixed_reveal_seed"]),
            maximum_candidates=maximum_candidates,
            device=device,
        )
        rows.append(
            {
                "hidden_dim": hidden_dim,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "initial_validation_nll": initial_validation,
                "final_validation_nll": final_validation,
                "validation_nll_reduction": initial_validation - final_validation,
                "first_train_nll": first_train_nll,
                "last_train_nll": last_train_nll,
                "finite_updates": finite_updates,
                "graphs_per_second": train_count / elapsed,
                "peak_cuda_mib": peak,
            }
        )

    selection = protocol["selection"]
    best_nll = min(float(row["final_validation_nll"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["final_validation_nll"])
        <= best_nll * float(selection["validation_nll_relative_to_best_max"])
        and float(row["peak_cuda_mib"]) <= float(selection["peak_cuda_mib_max"])
        and float(row["graphs_per_second"]) >= float(selection["graphs_per_second_min"])
        and int(row["finite_updates"]) == int(training["updates"])
    ]
    selected_width = min((int(row["hidden_dim"]) for row in eligible), default=None)
    checks = {
        "all_finite": all(int(row["finite_updates"]) == int(training["updates"]) for row in rows),
        "all_loss_descent": all(
            float(row["final_validation_nll"]) < float(row["initial_validation_nll"])
            for row in rows
        ),
        "resource_support": bool(eligible),
        "capacity_selected": selected_width is not None,
    }
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "implementation_commit": commit,
        "qualified": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "selection": {
            "selected_hidden_dim": selected_width,
            "best_final_validation_nll": best_nll,
            "rule": selection["rule"],
        },
        "data": {
            "allowed_index_sha256": _index_sha256(allowed),
            "fit_index_sha256": _index_sha256(train_indices),
            "validation_index_sha256": _index_sha256(validation_indices),
            "gold_leakage": 0,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "float32",
        },
        "decision": protocol["decision_rule"]["pass" if all(checks.values()) else "fail"],
        "boundary": protocol["boundary"],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
