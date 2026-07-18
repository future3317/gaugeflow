"""Qualify score-only reverse kernels on an analytic quotient-torus panel."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from gaugeflow.file_utils import load_json_object
from gaugeflow.manifold import torus_logmap
from gaugeflow.production.quotient_score import (
    factorized_translation_quotient_log_density_and_scaled_score,
)
from gaugeflow.production.schedules import (
    ExponentialTorusNoiseSchedule,
    wrapped_normal_log_density_and_score,
)

SCORE_ONLY_METHODS = (
    "ancestral_gaussian",
    "reverse_sde_grw",
    "predictor_corrector_grw",
    "probability_flow_heun",
)
REFERENCE_METHOD = "endpoint_aware_wrapped_bridge_reference"


def _project_translation(state: torch.Tensor) -> torch.Tensor:
    return state - state.mean(dim=1, keepdim=True)


def _standard_normal(
    shape: torch.Size, reference: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _quotient_components(
    state: torch.Tensor,
    endpoints: torch.Tensor,
    variance: torch.Tensor,
    quadrature_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint log likelihoods and unscaled quotient scores."""
    samples, sites, dimensions = state.shape
    modes = endpoints.shape[0]
    if endpoints.shape[1:] != (sites, dimensions) or variance.ndim != 0:
        raise ValueError("analytic quotient panel has incompatible shapes")
    displacement = (
        state[:, None] - endpoints[None]
    ).reshape(samples * modes * sites, dimensions)
    graph_batch = torch.arange(
        samples * modes, device=state.device, dtype=torch.long
    ).repeat_interleave(sites)
    sigma = variance.sqrt().expand(samples * modes)
    log_density, scaled_score = (
        factorized_translation_quotient_log_density_and_scaled_score(
            displacement,
            sigma,
            graph_batch,
            samples * modes,
            quadrature_points=quadrature_points,
        )
    )
    return (
        log_density.reshape(samples, modes),
        (scaled_score / sigma[graph_batch, None]).reshape(
            samples, modes, sites, dimensions
        ),
    )


def _mixture_score(
    state: torch.Tensor,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    variance: torch.Tensor,
    quadrature_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_density, component_score = _quotient_components(
        state, endpoints, variance, quadrature_points
    )
    posterior = torch.softmax(log_density + weights.log()[None], dim=1)
    return (
        (posterior[:, :, None, None] * component_score).sum(dim=1),
        posterior,
    )


def _endpoint_rms(
    state: torch.Tensor, endpoints: torch.Tensor
) -> torch.Tensor:
    difference = torus_logmap(endpoints[None], state[:, None])
    phase = 2.0 * math.pi * difference
    translation = torch.atan2(phase.sin().mean(dim=2), phase.cos().mean(dim=2)) / (
        2.0 * math.pi
    )
    residual = torus_logmap(translation[:, :, None], difference)
    return residual.square().sum(dim=-1).mean(dim=-1).sqrt()


def _categorical_sample(
    probabilities: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    return torch.multinomial(
        probabilities, 1, replacement=True, generator=generator
    ).squeeze(-1)


def _endpoint_aware_wrapped_bridge_step(
    state: torch.Tensor,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    quadrature_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the exact finite bridge after endpoint/translation/winding expansion."""
    samples, sites, dimensions = state.shape
    log_density, _ = _quotient_components(
        state, endpoints, variance_from, quadrature_points
    )
    endpoint_index = _categorical_sample(
        torch.softmax(log_density + weights.log()[None], dim=1), generator
    )
    endpoint = endpoints[endpoint_index]
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
            samples * dimensions, quadrature_points
        ),
        generator,
    ).reshape(samples, dimensions)
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
    ).reshape(samples, sites, dimensions)
    lifted_final = centered + images[winding_index]
    ratio = variance_to / variance_from
    bridge_variance = variance_to * (variance_from - variance_to) / variance_from
    lifted_to = ratio * lifted_final
    if float(variance_to) > 0.0:
        lifted_to = lifted_to + bridge_variance.sqrt() * _standard_normal(
            lifted_to.shape, lifted_to, generator
        )
    return _project_translation(endpoint + lifted_to)


def _score_step(
    method: str,
    state: torch.Tensor,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    quadrature_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    def score_function(value: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        return _mixture_score(
            value, endpoints, weights, variance, quadrature_points
        )[0]

    return _integrate_score_step(
        method,
        state,
        variance_from,
        variance_to,
        score_function,
        generator,
    )


def _integrate_score_step(
    method: str,
    state: torch.Tensor,
    variance_from: torch.Tensor,
    variance_to: torch.Tensor,
    score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply one pre-registered score-only quotient integrator step."""
    score = score_function(state, variance_from)
    variance_drop = variance_from - variance_to
    tangent_noise = _project_translation(
        _standard_normal(state.shape, state, generator)
    )
    if method == "ancestral_gaussian":
        updated = state + variance_drop * score
        if float(variance_to) > 0.0:
            bridge_variance = variance_to * variance_drop / variance_from
            updated = updated + bridge_variance.sqrt() * tangent_noise
    elif method in {"reverse_sde_grw", "predictor_corrector_grw"}:
        updated = state + variance_drop * score
        if float(variance_to) > 0.0:
            updated = updated + variance_drop.sqrt() * tangent_noise
        if method == "predictor_corrector_grw" and float(variance_to) > 0.0:
            correction = 0.1 * torch.minimum(variance_drop, variance_to)
            corrected_score = score_function(updated, variance_to)
            correction_noise = _project_translation(
                _standard_normal(state.shape, state, generator)
            )
            updated = (
                updated
                + correction * corrected_score
                + (2.0 * correction).sqrt() * correction_noise
            )
    elif method == "probability_flow_heun":
        if float(variance_to) == 0.0:
            updated = state + variance_from * score
        else:
            predicted = _project_translation(
                state + 0.5 * variance_drop * score
            )
            predicted_score = score_function(predicted, variance_to)
            updated = state + 0.25 * variance_drop * (
                score + predicted_score
            )
    else:
        raise ValueError(f"unknown score-only method: {method}")
    return _project_translation(updated)


def _initial_state(
    initialization: str,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    samples: int,
    variance: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if initialization == "uniform_quotient":
        state = torch.rand(
            (samples, endpoints.shape[1], endpoints.shape[2]),
            dtype=endpoints.dtype,
            device=endpoints.device,
            generator=generator,
        )
        return _project_translation(state), None
    if initialization != "exact_forward":
        raise ValueError(f"unknown terminal initialization: {initialization}")
    labels = _categorical_sample(weights.expand(samples, -1), generator)
    noise = _standard_normal(
        (samples, endpoints.shape[1], endpoints.shape[2]), endpoints, generator
    )
    state = endpoints[labels] + variance.sqrt() * noise
    return _project_translation(state), labels


def _run_trajectory(
    method: str,
    initial: torch.Tensor,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    variances: torch.Tensor,
    quadrature_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    state = initial.clone()
    for index in range(variances.numel() - 1):
        if method == REFERENCE_METHOD:
            state = _endpoint_aware_wrapped_bridge_step(
                state,
                endpoints,
                weights,
                variances[index],
                variances[index + 1],
                quadrature_points,
                generator,
            )
        else:
            state = _score_step(
                method,
                state,
                endpoints,
                weights,
                variances[index],
                variances[index + 1],
                quadrature_points,
                generator,
            )
    return state


def _record(
    panel: str,
    initialization: str,
    method: str,
    steps: int,
    final: torch.Tensor,
    endpoints: torch.Tensor,
    initial_labels: torch.Tensor | None,
    tolerance: float,
) -> dict[str, Any]:
    rms = _endpoint_rms(final, endpoints)
    best_rms, retrieval = rms.min(dim=1)
    counts = torch.bincount(retrieval, minlength=endpoints.shape[0]).double()
    proportions = counts / counts.sum()
    value: dict[str, Any] = {
        "panel": panel,
        "initialization": initialization,
        "method": method,
        "steps": steps,
        "mean_endpoint_rms_fractional": float(best_rms.mean()),
        "endpoint_rms_quantiles_fractional": torch.quantile(
            best_rms,
            torch.tensor(
                [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
                dtype=best_rms.dtype,
                device=best_rms.device,
            ),
        ).cpu().tolist(),
        "endpoint_recovery_fraction": float((best_rms <= tolerance).double().mean()),
        "cut_locus_failure_fraction": float((best_rms > tolerance).double().mean()),
        "retrieved_endpoint_proportions": proportions.cpu().tolist(),
        "sampling_failures": 0,
    }
    if initial_labels is not None:
        value["initial_label_retention_fraction"] = float(
            (retrieval == initial_labels).double().mean()
        )
    return value


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_wrapped_reverse_kernel_audit_v1":
        raise ValueError("unexpected wrapped reverse-kernel protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float64
    samples = int(protocol["samples"])
    seed = int(protocol["seed"])
    quadrature_points = int(protocol["quadrature_points"])
    tolerance = float(protocol["endpoint_rms_tolerance_fractional"])
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(protocol["coordinate_sigma_min"]),
        sigma_max=float(protocol["coordinate_sigma_max"]),
    )
    maximum_time = float(protocol["maximum_time"])
    panels = {
        "single_endpoint": torch.tensor(
            [protocol["panels"]["single_endpoint"]], dtype=dtype, device=device
        ),
        "two_endpoint_mixture": torch.tensor(
            protocol["panels"]["two_endpoint_mixture"],
            dtype=dtype,
            device=device,
        ),
    }
    methods = (*SCORE_ONLY_METHODS, REFERENCE_METHOD)
    records: list[dict[str, Any]] = []
    for panel_index, (panel_name, endpoints) in enumerate(panels.items()):
        if endpoints.shape[0] == 1:
            weights = torch.ones(1, dtype=dtype, device=device)
        else:
            weights = torch.tensor(
                protocol["panels"]["mixture_weights"], dtype=dtype, device=device
            )
        for initialization_index, initialization in enumerate(
            protocol["panels"]["terminal_initializations"]
        ):
            initial_generator = torch.Generator(device=device).manual_seed(
                seed + 100_000 * panel_index + 10_000 * initialization_index
            )
            terminal_variance = schedule.variance(
                torch.tensor(maximum_time, dtype=dtype, device=device)
            )
            initial, labels = _initial_state(
                str(initialization),
                endpoints,
                weights,
                samples,
                terminal_variance,
                initial_generator,
            )
            for steps in map(int, protocol["step_counts"]):
                times = torch.linspace(
                    maximum_time, 0.0, steps + 1, dtype=dtype, device=device
                )
                variances = schedule.variance(times)
                for method_index, method in enumerate(methods):
                    generator = torch.Generator(device=device).manual_seed(
                        seed
                        + 1_000_000 * panel_index
                        + 100_000 * initialization_index
                        + 1_000 * steps
                        + (0 if method in SCORE_ONLY_METHODS else method_index)
                    )
                    final = _run_trajectory(
                        method,
                        initial,
                        endpoints,
                        weights,
                        variances,
                        quadrature_points,
                        generator,
                    )
                    records.append(
                        _record(
                            panel_name,
                            str(initialization),
                            method,
                            steps,
                            final,
                            endpoints,
                            labels,
                            tolerance,
                        )
                    )

    acceptance = protocol["acceptance"]
    method_checks: dict[str, dict[str, Any]] = {}
    for method in SCORE_ONLY_METHODS:
        selected = [record for record in records if record["method"] == method]
        at_200 = [record for record in selected if record["steps"] == 200]
        singles = [record for record in at_200 if record["panel"] == "single_endpoint"]
        mixtures = [
            record for record in at_200 if record["panel"] == "two_endpoint_mixture"
        ]
        rms_100 = {
            (record["panel"], record["initialization"]): record[
                "mean_endpoint_rms_fractional"
            ]
            for record in selected
            if record["steps"] == 100
        }
        refinement_increases = [
            record["mean_endpoint_rms_fractional"]
            - rms_100[(record["panel"], record["initialization"])]
            for record in at_200
        ]
        mixture_errors = [
            max(abs(value - 0.5) for value in record["retrieved_endpoint_proportions"])
            for record in mixtures
        ]
        checks = {
            "single_endpoint_recovery": min(
                record["endpoint_recovery_fraction"] for record in singles
            )
            >= float(acceptance["score_only_single_endpoint_recovery_min"]),
            "two_endpoint_recovery": min(
                record["endpoint_recovery_fraction"] for record in mixtures
            )
            >= float(acceptance["score_only_two_endpoint_recovery_min"]),
            "cut_locus_failure": max(
                record["cut_locus_failure_fraction"] for record in at_200
            )
            <= float(acceptance["score_only_cut_locus_failure_fraction_max"]),
            "mixture_weight": max(mixture_errors)
            <= float(acceptance["score_only_mixture_weight_absolute_error_max"]),
            "step_refinement": max(refinement_increases)
            <= float(acceptance["score_only_100_to_200_mean_rms_increase_max"]),
        }
        method_checks[method] = {
            "checks": checks,
            "qualified": all(checks.values()),
            "max_mixture_weight_absolute_error": max(mixture_errors),
            "max_100_to_200_mean_rms_increase": max(refinement_increases),
        }

    references = [record for record in records if record["method"] == REFERENCE_METHOD]
    reference_200 = [record for record in references if record["steps"] == 200]
    reference_mixture_errors = [
        max(abs(value - 0.5) for value in record["retrieved_endpoint_proportions"])
        for record in reference_200
        if record["panel"] == "two_endpoint_mixture"
    ]
    reference_checks = {
        "single_endpoint_recovery": min(
            record["endpoint_recovery_fraction"]
            for record in reference_200
            if record["panel"] == "single_endpoint"
        )
        >= float(acceptance["reference_single_endpoint_recovery_min"]),
        "two_endpoint_recovery": min(
            record["endpoint_recovery_fraction"]
            for record in reference_200
            if record["panel"] == "two_endpoint_mixture"
        )
        >= float(acceptance["reference_two_endpoint_recovery_min"]),
        "mixture_weight": max(reference_mixture_errors)
        <= float(acceptance["reference_mixture_weight_absolute_error_max"]),
    }
    qualified = [
        method for method, value in method_checks.items() if value["qualified"]
    ]
    result = {
        "protocol": protocol["protocol"],
        "reference_checks": reference_checks,
        "reference_qualified": all(reference_checks.values()),
        "score_only_method_checks": method_checks,
        "qualified_score_only_methods": qualified,
        "records": records,
        "decision": (
            "propose_one_qualified_score_only_kernel_for_checkpoint_diagnostic"
            if all(reference_checks.values()) and qualified
            else "no_score_only_kernel_qualified_stop_before_production_change"
        ),
        "decision_boundary": protocol["decision_rule"]["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
