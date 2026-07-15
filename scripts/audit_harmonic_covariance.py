"""Numerically audit the continuous harmonic covariance theorem and grid limits.

This is an operator-only audit.  It neither opens a dataset nor trains a
generator.  In particular, finite-grid left/right residuals are reported as a
limitation of the QMC discretisation, not used as a post-hoc selection rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gaugeflow.harmonic import (
    deterministic_so3_grid,
    finite_grid_shift_residual,
    harmonic_alignment_scores,
)
from gaugeflow.tensor import (
    piezo_from_irreps,
    piezo_to_irreps,
    piezo_voigt_to_cartesian,
    rotate_rank3,
)


ROOT = Path(__file__).resolve().parents[1]


def _weights(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "weight_l1": torch.tensor([0.7, -1.1], dtype=dtype),
        "weight_l2": torch.tensor([0.3], dtype=dtype),
        "weight_l3": torch.tensor([-0.5], dtype=dtype),
    }


def _high_symmetry_condition(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    voigt = torch.zeros(1, 3, 6, dtype=dtype)
    # e_zxx=e_zyy is C4z invariant and remains nonzero for a polar tensor.
    voigt[0, 2, 0] = 1.0
    voigt[0, 2, 1] = 1.0
    c4z = torch.tensor(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=dtype)
    return piezo_to_irreps(piezo_voigt_to_cartesian(voigt)), c4z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/gate_h1_harmonic_conditioning_v1")
    )
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    output = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dtype = torch.float64

    condition_tensor = torch.randn(2, 3, 3, 3, dtype=dtype)
    condition = piezo_to_irreps(0.5 * (condition_tensor + condition_tensor.transpose(-1, -2)))
    directions = torch.nn.functional.normalize(torch.randn(12, 3, dtype=dtype), dim=-1)
    edge_graph = torch.tensor([0] * 6 + [1] * 6)
    grid = deterministic_so3_grid(37, dtype=dtype)
    g, h = grid[11], grid[23]
    transformed_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), h))
    transformed_directions = directions @ g.transpose(-1, -2)
    left_score, _ = harmonic_alignment_scores(
        transformed_condition, transformed_directions, edge_graph, grid, **_weights(dtype)
    )
    transformed_nodes = g.transpose(-1, -2).unsqueeze(0) @ grid @ h.unsqueeze(0)
    right_score, _ = harmonic_alignment_scores(
        condition, directions, edge_graph, transformed_nodes, **_weights(dtype)
    )
    covariance_error = float((left_score - right_score).abs().max())

    high, c4z = _high_symmetry_condition(dtype)
    high_rotated = piezo_to_irreps(rotate_rank3(piezo_from_irreps(high), c4z))
    high_score, _ = harmonic_alignment_scores(high, directions[:6], torch.zeros(6, dtype=torch.long), grid, **_weights(dtype))
    high_rotated_score, _ = harmonic_alignment_scores(
        high_rotated, directions[:6], torch.zeros(6, dtype=torch.long), grid, **_weights(dtype)
    )
    high_symmetry_tensor_error = float((piezo_from_irreps(high) - piezo_from_irreps(high_rotated)).abs().max())
    high_symmetry_score_error = float((high_score - high_rotated_score).abs().max())

    zero = torch.zeros(1, 18, dtype=dtype)
    zero_score, _ = harmonic_alignment_scores(
        zero, directions[:6], torch.zeros(6, dtype=torch.long), grid, **_weights(dtype)
    )
    zero_posterior = torch.softmax(zero_score, dim=-1)
    zero_uniform_error = float((zero_posterior - 1.0 / grid.shape[0]).abs().max())

    finite_grid = deterministic_so3_grid(24, dtype=dtype)
    shift = deterministic_so3_grid(29, dtype=dtype)[17]
    identity_shift_error = float(finite_grid_shift_residual(finite_grid).max())
    left_shift_error = float(finite_grid_shift_residual(finite_grid, left=shift).max())
    right_shift_error = float(finite_grid_shift_residual(finite_grid, right=shift).max())

    records = [
        {"test": "continuous_score_covariance", "value": covariance_error, "expectation": "<= 5e-5", "result": covariance_error <= 5e-5},
        {"test": "high_symmetry_tensor_representative", "value": high_symmetry_tensor_error, "expectation": "<= 5e-7", "result": high_symmetry_tensor_error <= 5e-7},
        {"test": "high_symmetry_score_representative", "value": high_symmetry_score_error, "expectation": "<= 5e-5", "result": high_symmetry_score_error <= 5e-5},
        {"test": "zero_tensor_uniform_posterior", "value": zero_uniform_error, "expectation": "<= 1e-12", "result": zero_uniform_error <= 1e-12},
        {"test": "finite_grid_identity_shift", "value": identity_shift_error, "expectation": "<= 1e-12", "result": identity_shift_error <= 1e-12},
        {"test": "finite_grid_left_shift_nonclosure", "value": left_shift_error, "expectation": "> 1e-4 (reported limitation)", "result": left_shift_error > 1e-4},
        {"test": "finite_grid_right_shift_nonclosure", "value": right_shift_error, "expectation": "> 1e-4 (reported limitation)", "result": right_shift_error > 1e-4},
    ]
    frame = pd.DataFrame(records)
    csv_path = output / "harmonic_covariance_audit.csv"
    frame.to_csv(csv_path, index=False)
    status = bool(frame.result.all())
    report = output / "harmonic_covariance_audit.md"
    report.write_text(
        "# H1 harmonic covariance audit\n\n"
        "This is a deterministic operator audit, not a training or generation result.\n\n"
        "For the degree-`l` query `q_l(gx)=rho_l(g)q_l(x)` and the score "
        "`s(R;x,e)=sum_l,m w_lm <rho_l(R)e_lm,q_l(x)>/sqrt(2l+1)`, orthogonality "
        "of `rho_l` gives `s(R;gx,he)=s(g^{-1}Rh;x,e)`. The continuous score is "
        "tested directly at the transformed nodes; it is distinct from a sampled grid posterior.\n\n"
        f"- seed: `{args.seed}`\n"
        f"- continuous-score theorem status: `{covariance_error <= 5e-5}`\n"
        f"- overall audit status: `{status}`\n\n"
        + frame.to_markdown(index=False)
        + "\n\nThe positive nonidentity left/right residuals are expected: the finite Hopf QMC grid "
        "is not a group and therefore has no exact generic left/right reindexing. They are not "
        "a threshold for a generation gate; a later protocol must pre-register its grid/refinement choice.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": 1,
        "name": "H1 harmonic covariance numerical audit v1",
        "status": "passed_operator_only" if status else "failed_operator_only",
        "seed": args.seed,
        "continuous_score_theorem": "s(R;gx,he)=s(g^{-1}Rh;x,e)",
        "finite_grid_statement": "QMC grid is not assumed closed under generic left/right shifts",
        "outputs": {"csv": csv_path.name, "report": report.name},
    }
    (output / "harmonic_covariance_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if not status:
        raise SystemExit("Harmonic covariance operator audit failed")


if __name__ == "__main__":
    main()
