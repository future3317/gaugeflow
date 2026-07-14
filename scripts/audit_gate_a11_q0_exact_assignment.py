"""Read-only Gate A11-Q0 audit of exact assignment enumeration and group actions.

This script deliberately contains no neural network, optimizer, checkpoint, or
tensor condition.  It validates the finite categorical law that Q1 may use
only after this Q0 audit has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core import Element


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluate_gate_a import _load_panel  # noqa: E402
from gaugeflow.assignment import (  # noqa: E402
    exact_assignment_permutation_log_probability_error,
    exact_assignment_quotient_nll,
    residual_automorphism_permutations,
    sample_exact_assignment,
)


MASK_INDEX = 119


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbol(atomic_number: int) -> str:
    return Element.from_Z(int(atomic_number)).symbol


def _assignment_label(labels: torch.Tensor) -> str:
    return "[" + ", ".join(_symbol(int(value)) for value in labels.tolist()) + "]"


def _automorphisms(frame: pd.DataFrame, material_id: str, mode: str) -> torch.Tensor:
    rows = frame.loc[
        (frame["material_id"] == material_id)
        & (frame["representation_symmetry"] == mode)
    ].sort_values("operation_index")
    if rows.empty:
        raise ValueError(f"A11.0 does not contain {mode} automorphisms for {material_id}")
    return torch.tensor(
        [json.loads(value) for value in rows["site_permutation_zero_based"].tolist()], dtype=torch.long
    )


def _target_counts(target: torch.Tensor, vocabulary: int) -> torch.Tensor:
    if target.numel() == 0 or int(target.min()) < 0 or int(target.max()) >= vocabulary:
        raise ValueError("Endpoint species are outside the fixed full chemical vocabulary")
    return torch.bincount(target, minlength=vocabulary)


def _group_rows(
    material_id: str,
    target: torch.Tensor,
    automorphisms: torch.Tensor,
    group_scope: str,
) -> list[dict[str, object]]:
    states = {
        "all_mask": torch.full_like(target, MASK_INDEX),
        "one_revealed_species": torch.cat((target[:1], torch.full_like(target[1:], MASK_INDEX))),
        "fully_revealed_species": target,
    }
    records: list[dict[str, object]] = []
    for name, partial in states.items():
        residual = residual_automorphism_permutations(partial, automorphisms)
        records.append(
            {
                "material_id": material_id,
                "group_scope": group_scope,
                "partial_state": name,
                "partial_tokens": json.dumps(partial.tolist()),
                "full_group_size": int(automorphisms.shape[0]),
                "residual_group_size": int(residual.shape[0]),
                "residual_permutations_zero_based": json.dumps(residual.tolist()),
            }
        )
    return records


def _check_sampling(result: object, counts: torch.Tensor, *, seed: int = 20260715) -> int:
    # Assignment-level Gumbel-max is the Q1 sampler contract.  Q0 samples a
    # fixed finite number only to test count conservation, never to evaluate a
    # learned generator.
    generator = torch.Generator().manual_seed(seed)
    failures = 0
    for _ in range(32):
        sampled, _ = sample_exact_assignment(result.distribution, generator=generator)
        if not torch.equal(torch.bincount(sampled, minlength=counts.numel()), counts):
            failures += 1
    return failures


def _report(
    protocol: dict[str, object],
    summary: pd.DataFrame,
    residual: pd.DataFrame,
    permutation: pd.DataFrame,
) -> str:
    lines = [
        "# Gate A11-Q0 exact assignment and residual-group audit",
        "",
        "This is a read-only, untrained mathematical qualification for the A11-Q finite assignment law. It does not start Q1, Q2, A11-G, tensor conditioning, a full benchmark, relaxation, DFT, or DFPT.",
        "",
        "## Contract verified",
        "",
        "For each endpoint, Q0 enumerates the unique count-constrained chemical labelings, not permutations of artificial same-species slots. With the observed 2+2 endpoint composition, this is exactly six assignments. The neutral all-zero score probe makes this a uniform categorical law; its quoted masses are checks of group marginalization, not learned endpoint-ID accuracy.",
        "",
        "Production quotient calculations use only `proper_so3`. `full_o3_scalar` is printed alongside it only as the A11.0 O(3)-scalar decoder diagnostic. It is not an allowed production quotient: improper operations cannot be silently removed for a rank-three polar tensor condition.",
        "",
        "At each partial state the quotient group is recomputed as `Gamma_t = {gamma: gamma y_t = y_t}`. The all-mask state retains the geometry group; revealed species/mask tokens can only retain operations compatible with that actual current state.",
        "",
        "## Exact-law results",
        "",
        "| material | support | fixed-CIF p (diagnostic) | proper quotient p | full-O(3) quotient p (diagnostic) | proper quotient NLL | full quotient NLL | entropy | samples with wrong count | Q0 checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.material_id} | {row.assignment_support_size} | {row.fixed_cif_probability:.6f} | "
            f"{row.proper_so3_quotient_probability:.6f} | {row.full_o3_quotient_probability:.6f} | "
            f"{row.proper_so3_quotient_nll:.6f} | {row.full_o3_quotient_nll:.6f} | "
            f"{row.assignment_entropy:.6f} | {row.assignment_sampling_count_failures} | {row.q0_status} |"
        )
    lines += ["", "## Residual groups", ""]
    for row in residual.itertuples(index=False):
        lines.append(
            f"- `{row.material_id}` / `{row.group_scope}` / `{row.partial_state}`: "
            f"|Aut(X)|={row.full_group_size}, |Gamma_t|={row.residual_group_size}."
        )
    lines += ["", "## Relabeling consistency", ""]
    for row in permutation.itertuples(index=False):
        lines.append(
            f"- `{row.material_id}`: maximum FP32 log-probability-vector error = "
            f"`{row.max_log_probability_error:.3e}` against the pre-registered "
            f"`{row.threshold:.1e}` threshold; pass={row.passed}."
        )
    lines += [
        "",
        "## Decision",
        "",
        "Q0 passes only the exact-enumeration and group-action implementation checks. It does not produce a learned composition, exact-assignment, StructureMatcher, terminal-mask, or sampling-validity result. Q1 remains **not started**. If Q1 is ever authorized and passes, Q2 must first test materials with distinct proper-SO(3) orbit structures before any tensor condition is restored.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a11_q_exact_assignment_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a11_q_exact_assignment_v1"))
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    q0 = protocol.get("q0")
    if not isinstance(q0, dict) or q0.get("status") != "pre_registered_read_only_exact_enumeration_audit":
        raise ValueError("A11-Q0 requires the frozen read-only exact-enumeration protocol")
    if q0.get("training") is not False:
        raise ValueError("A11-Q0 must not train")
    if q0.get("chemical_vocabulary_size") != 119 or q0.get("automorphism_scope", {}).get("production_group") != "proper_so3":
        raise ValueError("A11-Q must retain the 119-class proper-SO(3) production contract")

    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = _resolve(str(q0["source_protocol"]))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if list(q0["material_ids"]) != list(source["material_ids"]):
        raise ValueError("A11-Q0 materials must match the frozen endpoint-ID source panel")
    automorphism_path = _resolve(str(q0["automorphism_source"]))
    automorphism_frame = pd.read_csv(automorphism_path)
    batch = _load_panel(
        source,
        ROOT,
        torch.device("cpu"),
        preprocessed_cache=_resolve(source["data"]["preprocessed_cache"]),
    )

    summary_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    vocabulary = int(q0["chemical_vocabulary_size"])
    threshold = float(q0["fp32_permutation_log_probability_error_max"])
    for target_index, material_id in enumerate(q0["material_ids"]):
        nodes = batch.batch == target_index
        target_types = batch.atom_types[nodes].to(torch.long).cpu()
        counts = _target_counts(target_types, vocabulary)
        active_counts = {str(index): int(value) for index, value in enumerate(counts.tolist()) if value}
        if sorted(active_counts.values()) != [2, 2]:
            raise ValueError(f"{material_id} is no longer the frozen 2+2 Q0 endpoint composition")
        scores = torch.zeros((target_types.numel(), vocabulary), dtype=torch.float32)
        proper = _automorphisms(automorphism_frame, str(material_id), "proper_so3")
        full = _automorphisms(automorphism_frame, str(material_id), "full_o3_scalar")
        mask_state = torch.full_like(target_types, MASK_INDEX)
        proper_result = exact_assignment_quotient_nll(scores, counts, target_types, proper, mask_state)
        full_result = exact_assignment_quotient_nll(scores, counts, target_types, full, mask_state)
        distribution = proper_result.distribution
        sampling_failures = _check_sampling(proper_result, counts, seed=20260715 + target_index)
        relabeling = torch.roll(torch.arange(target_types.numel()), shifts=1)
        error = exact_assignment_permutation_log_probability_error(scores, counts, relabeling)
        residual_rows.extend(_group_rows(str(material_id), target_types, proper, "proper_so3_production"))
        residual_rows.extend(_group_rows(str(material_id), target_types, full, "full_o3_diagnostic"))

        proper_labels = {tuple(row.tolist()) for row in proper_result.unique_orbit_targets}
        full_labels = {tuple(row.tolist()) for row in full_result.unique_orbit_targets}
        for index, labels in enumerate(distribution.assignments):
            label = tuple(labels.tolist())
            probability_rows.append(
                {
                    "material_id": material_id,
                    "assignment_index": index,
                    "assignment_atomic_numbers": json.dumps(labels.tolist()),
                    "assignment_elements": _assignment_label(labels),
                    "probability": float(distribution.log_probabilities[index].exp()),
                    "is_fixed_cif_label": label == tuple(target_types.tolist()),
                    "in_proper_so3_target_orbit": label in proper_labels,
                    "in_full_o3_target_orbit_diagnostic": label in full_labels,
                }
            )
        entropy = -(distribution.log_probabilities.exp() * distribution.log_probabilities).sum()
        every_assignment_has_exact_counts = bool(
            torch.stack(
                [torch.bincount(assignment, minlength=vocabulary) for assignment in distribution.assignments]
            ).eq(counts).all()
        )
        q0_ok = (
            distribution.assignments.shape[0] == 6
            and sampling_failures == 0
            and float(error) <= threshold
            and every_assignment_has_exact_counts
        )
        summary_rows.append(
            {
                "material_id": material_id,
                "endpoint_label": q0["endpoint_labels"][target_index],
                "chemical_vocabulary_size": vocabulary,
                "active_composition_atomic_number_counts": json.dumps(active_counts),
                "assignment_support_size": int(distribution.assignments.shape[0]),
                "all_enumerated_assignments_have_exact_counts": every_assignment_has_exact_counts,
                "fixed_cif_probability": float(proper_result.fixed_cif_log_probability.exp()),
                "proper_so3_target_orbit_size": int(proper_result.unique_orbit_targets.shape[0]),
                "proper_so3_quotient_probability": float(proper_result.target_log_probability.exp()),
                "proper_so3_quotient_nll": float(proper_result.quotient_nll),
                "full_o3_target_orbit_size_diagnostic": int(full_result.unique_orbit_targets.shape[0]),
                "full_o3_quotient_probability": float(full_result.target_log_probability.exp()),
                "full_o3_quotient_nll": float(full_result.quotient_nll),
                "assignment_entropy": float(entropy),
                "assignment_sampling_count_failures": sampling_failures,
                "q0_status": "passed_exact_law_checks" if q0_ok else "failed_exact_law_checks",
            }
        )
        permutation_rows.append(
            {
                "material_id": material_id,
                "node_permutation_new_to_old": json.dumps(relabeling.tolist()),
                "max_log_probability_error": float(error),
                "threshold": threshold,
                "passed": bool(float(error) <= threshold),
            }
        )

    summary = pd.DataFrame(summary_rows)
    probabilities = pd.DataFrame(probability_rows)
    residual = pd.DataFrame(residual_rows)
    permutation = pd.DataFrame(permutation_rows)
    summary.to_csv(output / "q0_exact_assignment_summary.csv", index=False)
    probabilities.to_csv(output / "q0_assignment_probabilities.csv", index=False)
    residual.to_csv(output / "q0_residual_groups.csv", index=False)
    permutation.to_csv(output / "q0_permutation_consistency.csv", index=False)
    report_path = output / "gate_a11_q0_exact_assignment_report.md"
    report_path.write_text(_report(protocol, summary, residual, permutation), encoding="utf-8")
    q0_passed = bool((summary["q0_status"] == "passed_exact_law_checks").all()) and bool(permutation["passed"].all())
    files = [
        "q0_exact_assignment_summary.csv",
        "q0_assignment_probabilities.csv",
        "q0_residual_groups.csv",
        "q0_permutation_consistency.csv",
        report_path.name,
    ]
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": "q0_passed_exact_enumeration_read_only" if q0_passed else "q0_failed_exact_enumeration_read_only",
        "q1_status": "not_started",
        "q2_status": "not_started",
        "training_started": False,
        "tensor_conditioned_training_started": False,
        "historical_gate_evidence_modified": False,
        "production_automorphism_group": "proper_so3",
        "full_o3_usage": "diagnostic_only",
        "protocol_sha256": _sha256(protocol_path),
        "source_protocol_sha256": _sha256(source_path),
        "a11_0_automorphism_sha256": _sha256(automorphism_path),
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(permutation.to_string(index=False))


if __name__ == "__main__":
    main()
