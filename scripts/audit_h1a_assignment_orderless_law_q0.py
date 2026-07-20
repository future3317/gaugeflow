"""Run the frozen Q0 audit for the count-exact orderless assignment law."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.production.autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
    complete_pair_rbf,
)
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def _count_vector(values: tuple[int, ...]) -> torch.Tensor:
    return torch.bincount(
        torch.tensor(values, dtype=torch.long),
        minlength=CHEMICAL_ELEMENT_COUNT,
    )


def _unique_assignments(values: tuple[int, ...]) -> list[torch.Tensor]:
    return [
        torch.tensor(value, dtype=torch.long)
        for value in sorted(set(itertools.permutations(values)))
    ]


def _analytic_score(distance: torch.Tensor):
    interaction = torch.tensor(
        [[0.7, -0.4, 0.2], [-0.4, 0.9, -0.3], [0.2, -0.3, 0.5]],
        dtype=torch.float64,
    )

    def score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            partial.numel(),
            CHEMICAL_ELEMENT_COUNT,
            dtype=torch.float64,
        )
        revealed = torch.nonzero(partial >= 0, as_tuple=False).flatten()
        if revealed.numel():
            color = partial[revealed]
            logits[:, :3] = torch.einsum(
                "ij,jk->ik",
                torch.exp(-distance[:, revealed]),
                interaction[color],
            )
        logits[:, :3] += 0.03 * remaining[:3].to(torch.float64)
        return logits

    return score


def _brute_order_marginal(
    law: RemainingCountAssignmentLaw,
    score: Any,
    assignment: torch.Tensor,
    counts: torch.Tensor,
) -> float:
    values = []
    for order in itertools.permutations(range(assignment.numel())):
        log_probability = law.path_log_probability(
            score,
            assignment,
            torch.tensor(order, dtype=torch.long),
            counts,
        )
        values.append(float(log_probability.exp()))
    return sum(values) / math.factorial(assignment.numel())


def _law_metrics(config: dict[str, Any]) -> dict[str, float]:
    law = RemainingCountAssignmentLaw()
    zero_counts = _count_vector((0, 0, 1, 1))

    def zero_score(partial: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        del remaining
        return torch.zeros(
            partial.numel(),
            CHEMICAL_ELEMENT_COUNT,
            dtype=torch.float64,
        )

    zero_assignments = _unique_assignments((0, 0, 1, 1))
    zero_probabilities = [
        law.exact_order_marginal_probability(zero_score, value, zero_counts)
        for value in zero_assignments
    ]
    uniform_error = max(abs(value - 1.0 / 6.0) for value in zero_probabilities)

    assignments = _unique_assignments((0, 0, 1, 1, 2, 2))
    counts = _count_vector((0, 0, 1, 1, 2, 2))
    coordinates = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.8],
            [1.0, 1.7],
            [0.0, 1.5],
            [-0.5, 0.7],
        ],
        dtype=torch.float64,
    )
    distance = torch.cdist(coordinates, coordinates)
    score = _analytic_score(distance)
    probabilities = [
        law.exact_order_marginal_probability(score, value, counts)
        for value in assignments
    ]
    normalization_error = abs(sum(probabilities) - 1.0)
    selected = assignments[17]
    dp_value = probabilities[17]
    brute_value = _brute_order_marginal(law, score, selected, counts)

    relabel = torch.tensor([3, 0, 5, 2, 1, 4], dtype=torch.long)
    changed_score = _analytic_score(distance[relabel][:, relabel])
    changed_probability = law.exact_order_marginal_probability(
        changed_score,
        selected[relabel],
        counts,
    )

    action = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    quotient = law.exact_quotient_probability(
        zero_score,
        torch.tensor([0, 0, 1, 1]),
        zero_counts,
        action,
    )

    generator = torch.Generator().manual_seed(int(config["audit"]["seed"]))
    exact_samples = 0
    for _ in range(int(config["audit"]["sample_count"])):
        sampled = law.sample(
            score,
            counts,
            torch.randperm(6, generator=generator),
            generator=generator,
        )
        exact_samples += int(
            torch.equal(
                torch.bincount(sampled, minlength=CHEMICAL_ELEMENT_COUNT),
                counts,
            )
        )
    return {
        "uniform_assignment_error": uniform_error,
        "subset_dp_bruteforce_error": abs(dp_value - brute_value),
        "complete_distribution_normalization_error": normalization_error,
        "sample_exact_count_fraction": exact_samples
        / int(config["audit"]["sample_count"]),
        "relabel_marginal_error": abs(dp_value - changed_probability),
        "duplicate_orbit_probability_error": abs(quotient - 4.0 / 6.0),
    }


def _model_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, torch.Tensor]:
    torch.manual_seed(5705)
    nodes = 6
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=7,
        graph_feature_dim=5,
        radial_channels=8,
        hidden_dim=32,
        message_blocks=2,
    ).to(device=device, dtype=dtype)
    site = torch.randn(nodes, 7, device=device, dtype=dtype)
    graph = torch.randn(1, 5, device=device, dtype=dtype)
    batch = torch.zeros(nodes, dtype=torch.long, device=device)
    angle = torch.arange(nodes, device=device, dtype=torch.float64) * (2.0 * math.pi / nodes)
    positions = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    distance = torch.cdist(positions, positions).to(dtype)
    edge_source, edge_target = torch.nonzero(
        ~torch.eye(nodes, dtype=torch.bool, device=device),
        as_tuple=True,
    )
    edge_rbf = complete_pair_rbf(
        distance[edge_source, edge_target],
        radial_channels=8,
    )
    partial = torch.tensor([0, -1, 1, -1, 0, -1], device=device)
    composition = _count_vector((0, 0, 0, 1, 1, 1)).unsqueeze(0).to(device)
    remaining = _count_vector((0, 1, 1)).unsqueeze(0).to(device)
    parent = torch.tensor([1], dtype=torch.long, device=device)
    cell = torch.tensor([1], dtype=torch.long, device=device)
    logits = model(
        site,
        graph,
        batch,
        edge_source,
        edge_target,
        edge_rbf,
        partial,
        composition,
        remaining,
        parent,
        cell,
    )
    order = torch.tensor([3, 0, 5, 2, 1, 4], device=device)
    changed_distance = distance[order][:, order]
    changed_source, changed_target = torch.nonzero(
        ~torch.eye(nodes, dtype=torch.bool, device=device),
        as_tuple=True,
    )
    changed_rbf = complete_pair_rbf(
        changed_distance[changed_source, changed_target],
        radial_channels=8,
    )
    changed = model(
        site[order],
        graph,
        batch,
        changed_source,
        changed_target,
        changed_rbf,
        partial[order],
        composition,
        remaining,
        parent,
        cell,
    )
    equivariance = float((changed - logits[order]).abs().max())
    law = RemainingCountAssignmentLaw()
    loss = -law.step_log_probabilities(logits[1].float(), remaining[0])[1]
    loss.backward()
    gradient_norm = math.sqrt(
        sum(
            float(torch.square(value.grad.float()).sum())
            for value in model.parameters()
            if value.grad is not None
        )
    )
    return equivariance, gradient_norm, logits.detach().float().cpu()


def _residual_stabilizer_case(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Check sites related by an automorphism of the revealed state."""
    torch.manual_seed(5706)
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=3,
        graph_feature_dim=2,
        radial_channels=6,
        hidden_dim=16,
        message_blocks=2,
    ).to(device=device, dtype=dtype)
    nodes = 4
    distance = torch.tensor(
        [
            [0.0, 0.5, math.sqrt(0.5), 0.5],
            [0.5, 0.0, 0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), 0.5, 0.0, 0.5],
            [0.5, math.sqrt(0.5), 0.5, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    edge_source, edge_target = torch.nonzero(
        ~torch.eye(nodes, dtype=torch.bool, device=device),
        as_tuple=True,
    )
    edge_rbf = complete_pair_rbf(
        distance[edge_source, edge_target],
        radial_channels=6,
    )
    logits = model(
        torch.ones(nodes, 3, device=device, dtype=dtype),
        torch.zeros(1, 2, device=device, dtype=dtype),
        torch.zeros(nodes, dtype=torch.long, device=device),
        edge_source,
        edge_target,
        edge_rbf,
        torch.tensor([0, -1, 0, -1], dtype=torch.long, device=device),
        _count_vector((0, 0, 1, 1)).unsqueeze(0).to(device),
        _count_vector((1, 1)).unsqueeze(0).to(device),
        torch.tensor([1], dtype=torch.long, device=device),
        torch.tensor([1], dtype=torch.long, device=device),
    )
    return float((logits[1] - logits[3]).abs().max())


def _cuda_performance(config: dict[str, Any]) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("Q0 CUDA qualification requires a CUDA device")
    device = torch.device("cuda:0")
    torch.manual_seed(int(config["audit"]["seed"]))
    graphs = int(config["audit"]["cuda_batch_graphs"])
    sites = int(config["audit"]["cuda_sites_per_graph"])
    nodes = graphs * sites
    model = GeometryAwareRemainingCountScorer(
        site_feature_dim=64,
        graph_feature_dim=42,
        radial_channels=16,
        hidden_dim=96,
        message_blocks=3,
    ).to(device)
    site_feature = torch.randn(nodes, 64, device=device)
    graph_feature = torch.randn(graphs, 42, device=device)
    batch = torch.repeat_interleave(torch.arange(graphs, device=device), sites)
    local_source, local_target = torch.nonzero(
        ~torch.eye(sites, dtype=torch.bool, device=device),
        as_tuple=True,
    )
    offsets = (torch.arange(graphs, device=device) * sites)[:, None]
    edge_source = (local_source[None, :] + offsets).reshape(-1)
    edge_target = (local_target[None, :] + offsets).reshape(-1)
    distance = torch.rand(edge_source.numel(), device=device) * 2.0
    edge_rbf = complete_pair_rbf(distance, radial_channels=16)
    partial = torch.full((nodes,), -1, dtype=torch.long, device=device)
    local = torch.arange(sites, device=device)
    partial = partial.reshape(graphs, sites)
    partial[:, : sites // 2] = (local[: sites // 2] % 2)[None, :]
    partial = partial.reshape(-1)
    composition = torch.zeros(
        graphs,
        CHEMICAL_ELEMENT_COUNT,
        dtype=torch.long,
        device=device,
    )
    composition[:, 0] = sites // 2
    composition[:, 1] = sites - sites // 2
    revealed_zero = (sites // 2 + 1) // 2
    revealed_one = (sites // 2) // 2
    remaining = composition.clone()
    remaining[:, 0] -= revealed_zero
    remaining[:, 1] -= revealed_one
    parent = torch.ones(graphs, dtype=torch.long, device=device)
    cell = torch.ones(graphs, dtype=torch.long, device=device)

    @torch.no_grad()
    def forward() -> torch.Tensor:
        return model(
            site_feature,
            graph_feature,
            batch,
            edge_source,
            edge_target,
            edge_rbf,
            partial,
            composition,
            remaining,
            parent,
            cell,
        )

    for _ in range(int(config["audit"]["cuda_warmup"])):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fp32 = None
    for _ in range(int(config["audit"]["cuda_iterations"])):
        fp32 = forward()
    torch.cuda.synchronize()
    latency = (
        1000.0
        * (time.perf_counter() - started)
        / int(config["audit"]["cuda_iterations"])
    )
    memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    if fp32 is None:
        raise RuntimeError("CUDA performance loop produced no output")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        bf16 = forward()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            fp32.flatten().float(),
            bf16.flatten().float(),
            dim=0,
        )
    )
    return latency, memory, cosine


def _write_report(path: Path, result: dict[str, Any]) -> None:
    metric = result["metrics"]
    equivariance = (
        f"{metric['fp64_neural_equivariance_error']:.3e} / "
        f"{metric['fp32_neural_equivariance_error']:.3e}"
    )
    cuda_cost = (
        f"{metric['cuda_forward_latency_ms']:.3f} ms / "
        f"{metric['cuda_peak_memory_mib']:.3f} MiB"
    )
    path.write_text(
        f"""# Orderless remaining-count assignment Q0

Decision: **{'PASS' if result['qualified'] else 'FAIL'}**.

| metric | value |
|---|---:|
| complete distribution normalization error | {metric['complete_distribution_normalization_error']:.3e} |
| subset-DP vs order brute force | {metric['subset_dp_bruteforce_error']:.3e} |
| sample exact-count fraction | {metric['sample_exact_count_fraction']:.6f} |
| relabel marginal error | {metric['relabel_marginal_error']:.3e} |
| FP64 / FP32 neural equivariance | {equivariance} |
| residual-stabilizer error | {metric['residual_stabilizer_error']:.3e} |
| BF16 output cosine | {metric['bf16_output_cosine']:.6f} |
| RTX 4090 forward latency / peak memory | {cuda_cost} |

This is a mathematical/software qualification only. It performs no learning
and does not qualify assignment or connect generated composition.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    config = load_json_object(args.protocol)
    if (
        config.get("protocol") != "h1a_assignment_orderless_law_q0_v1"
        or config.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen assignment Q0 protocol")
    prerequisite = repository / config["prerequisite"]["geometry_expressivity_result"]
    if sha256_file(prerequisite) != config["prerequisite"][
        "geometry_expressivity_result_sha256"
    ]:
        raise ValueError("geometry-expressivity prerequisite identity changed")
    if load_json_object(prerequisite).get("qualified") is not True:
        raise ValueError("geometry-expressivity prerequisite is not qualified")

    metrics = _law_metrics(config)
    fp64, gradient64, _ = _model_case(
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    fp32, gradient32, _ = _model_case(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )
    stabilizer64 = _residual_stabilizer_case(
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    stabilizer32 = _residual_stabilizer_case(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
    )
    latency, memory, bf16_cosine = _cuda_performance(config)
    metrics.update(
        {
            "fp64_neural_equivariance_error": fp64,
            "fp32_neural_equivariance_error": fp32,
            "residual_stabilizer_error": max(stabilizer64, stabilizer32),
            "gradient_norm_fp64": gradient64,
            "gradient_norm_fp32": gradient32,
            "bf16_output_cosine": bf16_cosine,
            "cuda_forward_latency_ms": latency,
            "cuda_peak_memory_mib": memory,
            "cuda_measurement_mode": "no_grad_forward",
            "cuda_device": torch.cuda.get_device_name(0),
            "failures": 0,
        }
    )
    acceptance = config["acceptance"]
    checks = {
        "uniform_assignment": metrics["uniform_assignment_error"]
        <= acceptance["uniform_assignment_error_max"],
        "subset_dp_bruteforce": metrics["subset_dp_bruteforce_error"]
        <= acceptance["subset_dp_bruteforce_error_max"],
        "complete_normalization": metrics[
            "complete_distribution_normalization_error"
        ]
        <= acceptance["complete_distribution_normalization_error_max"],
        "sample_exact_counts": metrics["sample_exact_count_fraction"]
        == acceptance["sample_exact_count_fraction"],
        "relabel_marginal": metrics["relabel_marginal_error"]
        <= acceptance["relabel_marginal_error_max"],
        "duplicate_orbit": metrics["duplicate_orbit_probability_error"]
        <= acceptance["duplicate_orbit_probability_error_max"],
        "fp64_equivariance": metrics["fp64_neural_equivariance_error"]
        <= acceptance["fp64_neural_equivariance_error_max"],
        "fp32_equivariance": metrics["fp32_neural_equivariance_error"]
        <= acceptance["fp32_neural_equivariance_error_max"],
        "residual_stabilizer": metrics["residual_stabilizer_error"]
        <= acceptance["residual_stabilizer_error_max"],
        "finite_nonzero_gradient": all(
            math.isfinite(metrics[name]) and metrics[name] > 0.0
            for name in ("gradient_norm_fp64", "gradient_norm_fp32")
        ),
        "bf16_output": metrics["bf16_output_cosine"]
        >= acceptance["bf16_output_cosine_min"],
        "cuda_latency": metrics["cuda_forward_latency_ms"]
        <= acceptance["cuda_forward_latency_ms_max"],
        "cuda_memory": metrics["cuda_peak_memory_mib"]
        <= acceptance["cuda_peak_memory_mib_max"],
        "failures": metrics["failures"] == acceptance["failures"],
    }
    qualified = all(checks.values())
    result = {
        "protocol": config["protocol"],
        "protocol_sha256": canonical_json_hash(config),
        "qualified": qualified,
        "checks": checks,
        "metrics": metrics,
        "implementation_sha256": sha256_file(Path(__file__)),
        "law_sha256": sha256_file(
            repository / "src/gaugeflow/production/autoregressive_assignment.py"
        ),
        "decision": config["decision_rule"]["pass" if qualified else "fail"],
        "boundary": config["decision_rule"]["boundary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "README.md", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
