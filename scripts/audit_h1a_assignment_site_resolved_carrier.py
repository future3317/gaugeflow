"""Audit site-resolved geometry coverage of the assignment parent carrier."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from gaugeflow.file_utils import canonical_json_hash, load_json_object, sha256_file


def audit_candidate_geometry(candidate: dict[str, Any]) -> dict[str, Any]:
    nodes = int(candidate["child_site_count"])
    action = candidate["parent_action_permutations"]
    action_aligned = bool(action) and all(len(row) == nodes for row in action)
    parent_fractional = candidate.get("parent_fractional", [])
    full_geometry = len(parent_fractional) == nodes
    expanded = candidate.get("expanded_parent_fractional")
    expanded_geometry = isinstance(expanded, list) and len(expanded) == nodes
    cell_index = int(candidate["cell_index"])
    hnf = candidate.get("supercell_hnf")
    cosets = candidate.get("translation_cosets")
    expanded_lattice = candidate.get("expanded_parent_lattice")
    lattice = expanded_lattice if expanded_lattice is not None else candidate.get("parent_lattice")
    finite_lattice_shape = (
        isinstance(lattice, list)
        and len(lattice) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in lattice)
        and all(math.isfinite(float(value)) for row in lattice for value in row)
    )
    if finite_lattice_shape:
        a, b, c = ([float(value) for value in row] for row in lattice)
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    else:
        determinant = float("nan")
    finite_lattice = finite_lattice_shape and determinant > 0.0
    return {
        "cell_index": cell_index,
        "nodes": nodes,
        "parent_site_count": int(candidate["parent_site_count"]),
        "full_site_geometry": full_geometry,
        "expanded_geometry_field": expanded_geometry,
        "expanded_lattice_field": expanded_lattice is not None,
        "supercell_hnf_present": cell_index == 1 or hnf is not None,
        "translation_cosets_present": cell_index == 1 or cosets is not None,
        "action_node_aligned": action_aligned,
        "positive_finite_lattice": finite_lattice,
    }


def _fraction(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _validate_protocol(protocol: dict[str, Any], repository: Path, o1_root: Path) -> None:
    if (
        protocol.get("protocol") != "h1a_assignment_site_resolved_carrier_audit_v1"
        or protocol.get("status_before_run") != "frozen_not_run"
    ):
        raise ValueError("unexpected or unfrozen site-resolved carrier protocol")
    prerequisite = repository / "reports/h1a_assignment_orbital_set_expressivity_audit_v1/result.json"
    if sha256_file(prerequisite) != protocol["prerequisites"]["orbital_set_result_sha256"]:
        raise ValueError("orbital-set prerequisite identity changed")
    previous = load_json_object(prerequisite)
    if bool(previous.get("qualified")) is not bool(
        protocol["prerequisites"]["orbital_set_required_qualified"]
    ):
        raise ValueError("site-resolved audit has the wrong orbital-set prerequisite state")
    for name, expected in protocol["source"]["artifact_sha256"].items():
        if sha256_file(o1_root / name) != expected:
            raise ValueError(f"assignment source identity changed: {name}")


def _write_readme(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# H1a assignment site-resolved carrier audit v1",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| carriers | {metrics['carriers']} |",
        f"| full site geometry | {metrics['full_site_geometry_fraction']:.6f} |",
        f"| explicit expanded geometry | {metrics['expanded_geometry_field_fraction']:.6f} |",
        f"| explicit expanded lattice | {metrics['expanded_lattice_field_fraction']:.6f} |",
        "| HNF on nontrivial supercells | "
        f"{metrics['supercell_hnf_fraction_when_cell_index_gt_1']:.6f} |",
        "| translation cosets on nontrivial supercells | "
        f"{metrics['translation_coset_fraction_when_cell_index_gt_1']:.6f} |",
        f"| action-node alignment | {metrics['action_node_alignment_fraction']:.6f} |",
        "",
        "The archived O1 carrier stores primitive parent coordinates but omits the",
        "expanded species-free geometry, HNF and translation-coset ordering needed",
        "by 296 nontrivial-supercell carriers. These fields existed in the certified",
        "parent decomposition object but were not serialized into the assignment",
        "interface. They must be rebuilt at a versioned data boundary, not guessed",
        "from target coloring or patched with a model fallback.",
        "",
        f"Boundary: {result['boundary']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol, repository, args.o1_root)
    with gzip.open(args.o1_root / "results.json.gz", "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    rows = [
        {
            "material_id": str(record["material_id"]),
            "original_split": str(record["gaugeflow_split"]),
            "candidate_index": candidate_index,
            **audit_candidate_geometry(candidate),
        }
        for record in source
        for candidate_index, candidate in enumerate(record["candidates"])
    ]
    if len(rows) != int(protocol["source"]["candidate_carriers"]):
        raise ValueError("assignment carrier count changed")
    nontrivial = [row for row in rows if row["cell_index"] > 1]
    metrics = {
        "carriers": len(rows),
        "carriers_by_cell_index": dict(sorted(Counter(row["cell_index"] for row in rows).items())),
        "full_site_geometry_fraction": _fraction(rows, "full_site_geometry"),
        "expanded_geometry_field_fraction": _fraction(rows, "expanded_geometry_field"),
        "expanded_lattice_field_fraction": _fraction(rows, "expanded_lattice_field"),
        "supercell_hnf_fraction_when_cell_index_gt_1": _fraction(
            nontrivial, "supercell_hnf_present"
        ),
        "translation_coset_fraction_when_cell_index_gt_1": _fraction(
            nontrivial, "translation_cosets_present"
        ),
        "action_node_alignment_fraction": _fraction(rows, "action_node_aligned"),
        "positive_finite_lattice_fraction": _fraction(rows, "positive_finite_lattice"),
    }
    acceptance = protocol["acceptance"]
    checks = {
        key: metrics[key] == float(expected)
        for key, expected in acceptance.items()
    }
    qualified = all(checks.values())
    payload = {
        "protocol": protocol["protocol"],
        "protocol_sha256": canonical_json_hash(protocol),
        "qualified": qualified,
        "checks": checks,
        "decision": protocol["decision_rule"]["pass" if qualified else "fail"],
        "boundary": protocol["decision_rule"]["boundary"],
        "metrics": metrics,
        "carrier_rows": rows,
        "source_hashes": protocol["source"]["artifact_sha256"],
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(args.output_dir / "README.md", payload)
    print(json.dumps({key: payload[key] for key in ("qualified", "checks", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
