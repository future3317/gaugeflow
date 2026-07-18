"""Audit quotient reverse kernels on the high-symmetry InN/BN endpoints."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from audit_h1a_real_endpoint_reverse_kernel import (
    _paired_endpoint_rms,
    _quantiles,
    _run_trajectory,
    _translation_posterior_margin,
)
from audit_h1a_wrapped_reverse_kernel import (
    REFERENCE_METHOD,
    SCORE_ONLY_METHODS,
    _project_translation,
    _standard_normal,
)

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.schedules import ExponentialTorusNoiseSchedule


def _permutation_rms(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    lattice: torch.Tensor,
    atom_types: torch.Tensor,
    *,
    preserve_types: bool,
) -> torch.Tensor:
    sites = endpoint.shape[1]
    values: list[torch.Tensor] = []
    identity = tuple(range(sites))
    base_types = atom_types[0]
    for permutation in itertools.permutations(identity):
        order = torch.tensor(permutation, dtype=torch.long, device=endpoint.device)
        if preserve_types and not torch.equal(base_types[order], base_types):
            continue
        values.append(_paired_endpoint_rms(state, endpoint[:, order], lattice))
    return torch.stack(values, dim=1).amin(dim=1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_json_object(args.protocol)
    if protocol.get("protocol") != "h1a_symmetric_endpoint_reverse_kernel_audit_v1":
        raise ValueError("unexpected symmetric-endpoint protocol")
    if sha256_file(args.source_csv) != str(protocol["source_csv_sha256"]):
        raise ValueError("symmetric-endpoint source hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.float64
    endpoints_value = protocol["endpoints"]
    endpoint_base = torch.tensor(
        [value["fractional_coordinates"] for value in endpoints_value],
        dtype=dtype,
        device=device,
    )
    lattice_base = torch.tensor(
        [value["lattice"] for value in endpoints_value], dtype=dtype, device=device
    )
    atom_types_base = torch.tensor(
        [value["atom_types"] for value in endpoints_value],
        dtype=torch.long,
        device=device,
    )
    repeats = int(protocol["trajectories_per_endpoint"])
    endpoint = endpoint_base.repeat_interleave(repeats, dim=0)
    lattice = lattice_base.repeat_interleave(repeats, dim=0)
    atom_types = atom_types_base.repeat_interleave(repeats, dim=0)
    endpoint_labels = torch.arange(endpoint_base.shape[0]).repeat_interleave(repeats)
    schedule = ExponentialTorusNoiseSchedule(
        sigma_min=float(protocol["coordinate_sigma_min"]),
        sigma_max=float(protocol["coordinate_sigma_max"]),
    )
    terminal_variance = schedule.variance(
        torch.tensor(
            float(protocol["maximum_time"]), dtype=dtype, device=device
        )
    )
    records: list[dict[str, Any]] = []
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
                dtype=dtype,
                device=device,
                generator=initial_generator,
            )
        else:
            raise ValueError(f"unknown initialization: {initialization}")
        initial = _project_translation(initial)
        for steps in map(int, protocol["step_counts"]):
            times = torch.linspace(
                float(protocol["maximum_time"]),
                0.0,
                steps + 1,
                dtype=dtype,
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
                fixed_rms = _paired_endpoint_rms(final, endpoint, lattice)
                type_rms = torch.empty_like(fixed_rms)
                full_rms = torch.empty_like(fixed_rms)
                for label in range(endpoint_base.shape[0]):
                    selected = endpoint_labels.to(device) == label
                    type_rms[selected] = _permutation_rms(
                        final[selected],
                        endpoint[selected],
                        lattice[selected],
                        atom_types[selected],
                        preserve_types=True,
                    )
                    full_rms[selected] = _permutation_rms(
                        final[selected],
                        endpoint[selected],
                        lattice[selected],
                        atom_types[selected],
                        preserve_types=False,
                    )
                tolerance = float(protocol["endpoint_rms_tolerance_angstrom"])
                type_recovered = type_rms <= tolerance
                full_recovered = full_rms <= tolerance
                margin = _translation_posterior_margin(
                    preterminal,
                    endpoint,
                    variances[-2],
                    int(protocol["quadrature_points"]),
                )
                by_endpoint = []
                for label, value in enumerate(endpoints_value):
                    selected = endpoint_labels.to(device) == label
                    by_endpoint.append(
                        {
                            "material_id": value["material_id"],
                            "formula": value["formula"],
                            "fixed_cif_recovery_fraction": float(
                                (fixed_rms[selected] <= tolerance).double().mean()
                            ),
                            "type_quotient_recovery_fraction": float(
                                type_recovered[selected].double().mean()
                            ),
                            "full_permutation_recovery_fraction": float(
                                full_recovered[selected].double().mean()
                            ),
                        }
                    )
                records.append(
                    {
                        "initialization": initialization,
                        "method": method,
                        "steps": steps,
                        "mean_fixed_cif_rms_angstrom": float(fixed_rms.mean()),
                        "mean_type_quotient_rms_angstrom": float(type_rms.mean()),
                        "mean_full_permutation_rms_angstrom": float(full_rms.mean()),
                        "fixed_cif_rms_quantiles_angstrom": _quantiles(fixed_rms),
                        "type_quotient_rms_quantiles_angstrom": _quantiles(type_rms),
                        "type_quotient_recovery_fraction": float(
                            type_recovered.double().mean()
                        ),
                        "species_invalid_branch_fraction": float(
                            ((~type_recovered) & full_recovered).double().mean()
                        ),
                        "unrecovered_even_full_permutation_fraction": float(
                            (~full_recovered).double().mean()
                        ),
                        "final_step_translation_log_margin_quantiles": _quantiles(
                            margin
                        ),
                        "by_endpoint": by_endpoint,
                        "sampling_failures": 0,
                    }
                )

    acceptance = protocol["acceptance"]
    method_checks: dict[str, dict[str, Any]] = {}
    for method in SCORE_ONLY_METHODS:
        selected = [record for record in records if record["method"] == method]
        at_200 = [record for record in selected if record["steps"] == 200]
        rms_100 = {
            record["initialization"]: record["mean_type_quotient_rms_angstrom"]
            for record in selected
            if record["steps"] == 100
        }
        increases = [
            record["mean_type_quotient_rms_angstrom"]
            - rms_100[str(record["initialization"])]
            for record in at_200
        ]
        checks = {
            "type_quotient_recovery": min(
                record["type_quotient_recovery_fraction"] for record in at_200
            )
            >= float(acceptance["score_only_type_quotient_recovery_min"]),
            "species_invalid_branch": max(
                record["species_invalid_branch_fraction"] for record in at_200
            )
            <= float(acceptance["score_only_species_invalid_branch_fraction_max"]),
            "step_refinement": max(increases)
            <= float(
                acceptance[
                    "score_only_100_to_200_type_quotient_rms_increase_angstrom_max"
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
            "max_100_to_200_type_quotient_rms_increase_angstrom": max(increases),
        }
    reference = [
        record
        for record in records
        if record["method"] == REFERENCE_METHOD and record["steps"] == 200
    ]
    reference_qualified = min(
        record["type_quotient_recovery_fraction"] for record in reference
    ) >= float(acceptance["reference_type_quotient_recovery_min"])
    qualified = [
        method for method, value in method_checks.items() if value["qualified"]
    ]
    if not reference_qualified or not qualified:
        decision = "no_score_only_kernel_qualified_on_symmetric_endpoints"
    elif len(qualified) == 1:
        decision = "unique_kernel_may_enter_existing_checkpoint_diagnostic"
    else:
        decision = "multiple_kernels_pass_require_learned_score_robustness_audit"
    result = {
        "protocol": protocol["protocol"],
        "source_csv_sha256": protocol["source_csv_sha256"],
        "reference_qualified": reference_qualified,
        "score_only_method_checks": method_checks,
        "qualified_score_only_methods": qualified,
        "records": records,
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
