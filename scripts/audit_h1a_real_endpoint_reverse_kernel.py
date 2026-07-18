"""Audit analytic quotient reverse kernels on fixed real validation endpoints."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
from audit_h1a_wrapped_reverse_kernel import (
    REFERENCE_METHOD,
    SCORE_ONLY_METHODS,
    _categorical_sample,
    _integrate_score_step,
    _project_translation,
    _standard_normal,
)

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file
from gaugeflow.manifold import torus_logmap
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.quotient_score import (
    factorized_translation_quotient_log_density_and_scaled_score,
)
from gaugeflow.production.schedules import (
    ExponentialTorusNoiseSchedule,
    wrapped_normal_log_density_and_score,
)


def _graph_score(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    variance: torch.Tensor,
    quadrature_points: int,
) -> torch.Tensor:
    graphs, sites, dimensions = state.shape
    if endpoint.shape != state.shape or variance.ndim != 0:
        raise ValueError("real endpoint score has incompatible shapes")
    batch = torch.arange(graphs, device=state.device).repeat_interleave(sites)
    sigma = variance.sqrt().expand(graphs)
    _, scaled_score = factorized_translation_quotient_log_density_and_scaled_score(
        (state - endpoint).reshape(graphs * sites, dimensions),
        sigma,
        batch,
        graphs,
        quadrature_points=quadrature_points,
    )
    return (scaled_score / sigma[batch, None]).reshape_as(state)


def _paired_endpoint_rms(
    state: torch.Tensor, endpoint: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    difference = torus_logmap(endpoint, state)
    phase = 2.0 * math.pi * difference
    translation = torch.atan2(phase.sin().mean(dim=1), phase.cos().mean(dim=1)) / (
        2.0 * math.pi
    )
    residual = torus_logmap(translation[:, None], difference)
    cartesian = torch.einsum("bni,bij->bnj", residual, lattice)
    return cartesian.square().sum(dim=-1).mean(dim=-1).sqrt()


def _translation_posterior_margin(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    variance: torch.Tensor,
    quadrature_points: int,
) -> torch.Tensor:
    displacement = state - endpoint
    angle = 2.0 * math.pi * displacement
    center = torch.atan2(angle.sin().mean(dim=1), angle.cos().mean(dim=1)) / (
        2.0 * math.pi
    )
    grid = torch.arange(
        quadrature_points, dtype=state.dtype, device=state.device
    ) / quadrature_points
    translation = center[:, None, :] + grid[None, :, None]
    residual = displacement[:, :, None, :] - translation[:, None, :, :]
    log_kernel, _ = wrapped_normal_log_density_and_score(
        residual, variance.sqrt()
    )
    top = log_kernel.sum(dim=1).topk(2, dim=1).values
    return (top[:, 0] - top[:, 1]).amin(dim=-1)


def _graph_wrapped_bridge_step(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    quadrature_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    graphs, sites, dimensions = state.shape
    displacement = state - endpoint
    angle = 2.0 * math.pi * displacement
    center = torch.atan2(angle.sin().mean(dim=1), angle.cos().mean(dim=1)) / (
        2.0 * math.pi
    )
    grid = torch.arange(
        quadrature_points, dtype=state.dtype, device=state.device
    ) / quadrature_points
    translation = center[:, None, :] + grid[None, :, None]
    residual = displacement[:, :, None, :] - translation[:, None, :, :]
    log_kernel, _ = wrapped_normal_log_density_and_score(
        residual, variance_from.sqrt()
    )
    translation_posterior = torch.softmax(log_kernel.sum(dim=1), dim=1)
    translation_index = _categorical_sample(
        translation_posterior.permute(0, 2, 1).reshape(
            graphs * dimensions, quadrature_points
        ),
        generator,
    ).reshape(graphs, dimensions)
    selected_translation = translation.gather(
        1, translation_index[:, None, :]
    ).squeeze(1)
    centered = torch.remainder(
        displacement - selected_translation[:, None, :] + 0.5, 1.0
    ) - 0.5
    images = torch.arange(-4, 5, dtype=state.dtype, device=state.device)
    winding_logits = -0.5 * (
        (centered[..., None] + images) / variance_from.sqrt()
    ).square()
    winding_index = _categorical_sample(
        torch.softmax(winding_logits, dim=-1).reshape(-1, images.numel()),
        generator,
    ).reshape(graphs, sites, dimensions)
    lifted_final = centered + images[winding_index]
    ratio = variance_to / variance_from
    bridge_variance = variance_to * (variance_from - variance_to) / variance_from
    lifted_to = ratio * lifted_final
    if float(variance_to) > 0.0:
        lifted_to = lifted_to + bridge_variance.sqrt() * _standard_normal(
            lifted_to.shape, lifted_to, generator
        )
    return _project_translation(endpoint + lifted_to)


def _run_trajectory(
    method: str,
    initial: torch.Tensor,
    endpoint: torch.Tensor,
    variances: torch.Tensor,
    quadrature_points: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial.clone()
    preterminal = state

    def score_function(value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return _graph_score(value, endpoint, variance, quadrature_points)

    for index in range(variances.numel() - 1):
        preterminal = state
        if method == REFERENCE_METHOD:
            state = _graph_wrapped_bridge_step(
                state,
                endpoint,
                variances[index],
                variances[index + 1],
                quadrature_points,
                generator,
            )
        else:
            state = _integrate_score_step(
                method,
                state,
                variances[index],
                variances[index + 1],
                score_function,
                generator,
            )
    return state, preterminal


def _self_automorphism_count(
    coordinates: torch.Tensor,
    atom_types: torch.Tensor,
    lattice: torch.Tensor,
    tolerance: float,
) -> int:
    count = 0
    identity = tuple(range(coordinates.shape[0]))
    for permutation in itertools.permutations(identity):
        if permutation == identity:
            continue
        order = torch.tensor(permutation, dtype=torch.long)
        if not torch.equal(atom_types[order], atom_types):
            continue
        rms = _paired_endpoint_rms(
            coordinates[None], coordinates[order][None], lattice[None]
        )[0]
        if float(rms) <= tolerance:
            count += 1
    return count


def _quantiles(values: torch.Tensor) -> list[float]:
    return torch.quantile(
        values.double(),
        torch.tensor(
            [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
            dtype=torch.float64,
            device=values.device,
        ),
    ).cpu().tolist()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_real_endpoint_reverse_kernel_audit_v1":
        raise ValueError("unexpected real-endpoint reverse-kernel protocol")
    if sha256_file(args.cache_root / "manifest.json") != str(
        protocol["cache_manifest_sha256"]
    ):
        raise ValueError("real-endpoint audit cache manifest mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selection = protocol["endpoint_selection"]
    dataset = PackedAlexP1Dataset(args.cache_root, str(protocol["split"]), include_material_id=True)
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(selection["seed"]))
    )
    selected_indices = order[dataset.node_counts[order] == int(selection["site_count"])][
        : int(selection["endpoints"])
    ]
    if selected_indices.numel() != int(selection["endpoints"]):
        raise RuntimeError("real-endpoint selection did not produce enough structures")
    records = [dataset[int(index)] for index in selected_indices]
    material_ids = [str(record.material_id) for record in records]
    endpoint_base = torch.stack([record.frac_coords for record in records]).to(
        device=device, dtype=torch.float64
    )
    lattice_base = torch.cat([record.lattice for record in records]).to(
        device=device, dtype=torch.float64
    )
    atom_types = torch.stack([record.atom_types for record in records])
    automorphisms = [
        _self_automorphism_count(
            endpoint_base[index].cpu(),
            atom_types[index],
            lattice_base[index].cpu(),
            float(protocol["automorphism_tolerance_angstrom"]),
        )
        for index in range(endpoint_base.shape[0])
    ]
    repeats = int(selection["trajectories_per_endpoint"])
    endpoint = endpoint_base.repeat_interleave(repeats, dim=0)
    lattice = lattice_base.repeat_interleave(repeats, dim=0)
    endpoint_index = torch.arange(endpoint_base.shape[0]).repeat_interleave(repeats)
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(protocol["coordinate_sigma_min"]),
        sigma_max=float(protocol["coordinate_sigma_max"]),
    )
    terminal_variance = schedule.variance(
        torch.tensor(
            float(protocol["maximum_time"]), dtype=torch.float64, device=device
        )
    )
    result_records: list[dict[str, Any]] = []
    for initialization_index, initialization in enumerate(
        protocol["terminal_initializations"]
    ):
        initial_generator = torch.Generator(device=device).manual_seed(
            int(protocol["seed"]) + 100_000 * initialization_index
        )
        if initialization == "exact_forward":
            initial = endpoint + terminal_variance.sqrt() * _standard_normal(
                endpoint.shape, endpoint, initial_generator
            )
        elif initialization == "uniform_quotient":
            initial = torch.rand(
                endpoint.shape,
                dtype=endpoint.dtype,
                device=device,
                generator=initial_generator,
            )
        else:
            raise ValueError(f"unknown terminal initialization: {initialization}")
        initial = _project_translation(initial)
        for steps in map(int, protocol["step_counts"]):
            times = torch.linspace(
                float(protocol["maximum_time"]),
                0.0,
                steps + 1,
                dtype=torch.float64,
                device=device,
            )
            variances = schedule.variance(times)
            for method_index, method in enumerate(protocol["methods"]):
                generator = torch.Generator(device=device).manual_seed(
                    int(protocol["seed"])
                    + 100_000 * initialization_index
                    + 1_000 * steps
                    + (0 if method in SCORE_ONLY_METHODS else method_index)
                )
                final, preterminal = _run_trajectory(
                    str(method),
                    initial,
                    endpoint,
                    variances,
                    int(protocol["quadrature_points"]),
                    generator,
                )
                rms = _paired_endpoint_rms(final, endpoint, lattice)
                recovered = rms <= float(protocol["endpoint_rms_tolerance_angstrom"])
                margin = _translation_posterior_margin(
                    preterminal,
                    endpoint,
                    variances[-2],
                    int(protocol["quadrature_points"]),
                )
                failed_endpoint_indices = torch.unique(endpoint_index[~recovered.cpu()])
                result_records.append(
                    {
                        "initialization": initialization,
                        "method": method,
                        "steps": steps,
                        "mean_endpoint_rms_angstrom": float(rms.mean()),
                        "endpoint_rms_quantiles_angstrom": _quantiles(rms),
                        "endpoint_recovery_fraction": float(recovered.double().mean()),
                        "cut_locus_failure_fraction": float((~recovered).double().mean()),
                        "final_step_translation_log_margin_quantiles": _quantiles(margin),
                        "failed_translation_log_margin_quantiles": (
                            _quantiles(margin[~recovered]) if bool((~recovered).any()) else []
                        ),
                        "failed_endpoint_indices": failed_endpoint_indices.tolist(),
                        "failed_endpoint_material_ids": [
                            material_ids[int(index)] for index in failed_endpoint_indices
                        ],
                        "failed_endpoints_with_nonidentity_automorphism": sum(
                            automorphisms[int(index)] > 0
                            for index in failed_endpoint_indices
                        ),
                        "sampling_failures": 0,
                    }
                )

    acceptance = protocol["acceptance"]
    method_checks: dict[str, dict[str, Any]] = {}
    for method in SCORE_ONLY_METHODS:
        selected = [
            record for record in result_records if record["method"] == method
        ]
        at_200 = [record for record in selected if record["steps"] == 200]
        at_100 = {
            record["initialization"]: record["mean_endpoint_rms_angstrom"]
            for record in selected
            if record["steps"] == 100
        }
        increases = [
            record["mean_endpoint_rms_angstrom"]
            - at_100[str(record["initialization"])]
            for record in at_200
        ]
        checks = {
            "endpoint_recovery": min(
                record["endpoint_recovery_fraction"] for record in at_200
            )
            >= float(acceptance["score_only_endpoint_recovery_min"]),
            "cut_locus_failure": max(
                record["cut_locus_failure_fraction"] for record in at_200
            )
            <= float(acceptance["score_only_cut_locus_failure_fraction_max"]),
            "step_refinement": max(increases)
            <= float(
                acceptance[
                    "score_only_100_to_200_mean_rms_increase_angstrom_max"
                ]
            ),
            "sampling_failures": max(
                int(record["sampling_failures"]) for record in selected
            )
            == int(acceptance["sampling_failures"]),
        }
        method_checks[method] = {
            "checks": checks,
            "qualified": all(checks.values()),
            "max_100_to_200_mean_rms_increase_angstrom": max(increases),
        }
    reference_200 = [
        record
        for record in result_records
        if record["method"] == REFERENCE_METHOD and record["steps"] == 200
    ]
    reference_qualified = min(
        record["endpoint_recovery_fraction"] for record in reference_200
    ) >= float(acceptance["reference_endpoint_recovery_min"])
    qualified = [
        method for method, value in method_checks.items() if value["qualified"]
    ]
    if not reference_qualified or not qualified:
        decision = "no_score_only_kernel_qualified_stop_before_production_change"
    elif len(qualified) == 1:
        decision = "unique_score_only_kernel_may_enter_existing_checkpoint_diagnostic"
    else:
        decision = "multiple_kernels_pass_endpoint_closure_require_marginal_robustness_audit"
    result = {
        "protocol": protocol["protocol"],
        "cache_manifest_sha256": protocol["cache_manifest_sha256"],
        "selected_indices": selected_indices.tolist(),
        "selected_material_ids": material_ids,
        "selection_sha256": canonical_json_hash(
            {"indices": selected_indices.tolist(), "material_ids": material_ids}
        ),
        "endpoint_automorphism_counts": automorphisms,
        "endpoints_with_nonidentity_automorphism": sum(value > 0 for value in automorphisms),
        "reference_qualified": reference_qualified,
        "score_only_method_checks": method_checks,
        "qualified_score_only_methods": qualified,
        "records": result_records,
        "decision": decision,
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
