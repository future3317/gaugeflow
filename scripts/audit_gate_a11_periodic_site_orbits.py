"""A11.0 read-only audit of unlabeled periodic-site automorphism orbits."""

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
from gaugeflow.periodic_orbits import audit_unlabeled_periodic_site_orbits  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _species_counts(counts: dict[str, int]) -> str:
    return ", ".join(
        f"{Element.from_Z(int(number)).symbol}:{count}"
        for number, count in sorted(counts.items(), key=lambda item: int(item[0]))
    )


def _flatten_records(material_id: str, audit: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    orbits: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    for mode in ("proper_so3", "full_o3_scalar"):
        result = audit[mode]
        assert isinstance(result, dict)
        summary.append(
            {
                "material_id": material_id,
                "representation_symmetry": mode,
                "site_count": audit["site_count"],
                "operation_count": result["operation_count"],
                "site_orbit_count": len(result["orbits"]),
                "mixed_species_orbit_count": result["mixed_orbit_count"],
                "deterministic_equivariant_fixed_cif_accuracy_ceiling": result[
                    "deterministic_equivariant_fixed_cif_accuracy_ceiling"
                ],
                "a11_g_decision": audit["a11_g_decision"],
            }
        )
        for orbit in result["orbits"]:
            assert isinstance(orbit, dict)
            counts = orbit["atomic_number_counts"]
            assert isinstance(counts, dict)
            orbits.append(
                {
                    "material_id": material_id,
                    "representation_symmetry": mode,
                    "orbit_index": orbit["orbit_index"],
                    "site_indices_zero_based": json.dumps(orbit["site_indices"]),
                    "orbit_size": len(orbit["site_indices"]),
                    "species_counts": _species_counts(counts),
                    "is_species_mixed": orbit["is_species_mixed"],
                    "deterministic_constant_label_ceiling": orbit[
                        "deterministic_constant_label_ceiling"
                    ],
                }
            )
        for operation_index, operation in enumerate(result["operations"]):
            assert isinstance(operation, dict)
            operations.append(
                {
                    "material_id": material_id,
                    "representation_symmetry": mode,
                    "operation_index": operation_index,
                    "cartesian_determinant": operation["cartesian_determinant"],
                    "fractional_rotation": json.dumps(operation["fractional_rotation"]),
                    "fractional_translation": json.dumps(operation["fractional_translation"]),
                    "site_permutation_zero_based": json.dumps(operation["permutation"]),
                }
            )
    return summary, orbits, operations


def _report(protocol: dict[str, object], summaries: pd.DataFrame, orbit_rows: pd.DataFrame) -> str:
    lines = [
        "# Gate A11.0 periodic unlabeled-site automorphism audit",
        "",
        "This is a read-only geometric-identifiability audit. It starts no training and does not alter Gate A through A10.",
        "",
        "## Method",
        "",
        "Each endpoint is first Niggli reduced, which quotients integer lattice-basis representations. Every site is then replaced by the same dummy species before `SpacegroupAnalyzer` enumerates periodic operations. Operations are converted to explicit site permutations using only lattice and fractional coordinates. Target elements are inspected only after those permutations and site orbits have been fixed.",
        "",
        "Two partitions are reported. `proper_so3` keeps only determinant-positive Cartesian operations, matching GaugeFlow's tensor gauge. `full_o3_scalar` also retains improper operations because the proposed A11-G distance/dot-product type head is O(3)-invariant. The latter is therefore the conservative identifiability partition for A11-G.",
        "",
        "## Results",
        "",
        "| material | partition | operations | site orbits | mixed chemical orbits | fixed-CIF deterministic ceiling | decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.material_id} | {row.representation_symmetry} | {row.operation_count} | "
            f"{row.site_orbit_count} | {row.mixed_species_orbit_count} | "
            f"{row.deterministic_equivariant_fixed_cif_accuracy_ceiling:.3f} | {row.a11_g_decision} |"
        )
    lines += ["", "## Orbit-level labels", ""]
    for row in orbit_rows.itertuples(index=False):
        lines.append(
            f"- `{row.material_id}` / `{row.representation_symmetry}` orbit {row.orbit_index}: "
            f"sites {row.site_indices_zero_based}; {row.species_counts}; mixed={row.is_species_mixed}; "
            f"constant-label ceiling={row.deterministic_constant_label_ceiling:.3f}."
        )
    decisions = set(summaries.loc[summaries.representation_symmetry == "full_o3_scalar", "a11_g_decision"])
    lines += ["", "## Consequence", ""]
    if decisions == {"geometry_only_authorized"}:
        lines.append(
            "All full-O(3) site orbits are species-pure. A separately versioned A11-G geometry-only protocol is scientifically admissible; this audit does not train or pass A11-G."
        )
    else:
        lines.append(
            "At least one full-O(3) unlabeled site orbit contains multiple target species. A deterministic distance/dot-product decoder cannot be judged solely by fixed-CIF site accuracy on that orbit. Any successor must specify stochastic balanced assignment and automorphism-quotient supervision before training; A11-G is not authorized by this audit."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/gate_a11_0_periodic_site_orbits_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/gate_a11_0_periodic_site_orbits_v1"),
    )
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_read_only_geometric_identifiability_audit":
        raise ValueError("A11.0 requires its frozen read-only audit protocol")
    source_path = _resolve(str(protocol["source_protocol"]))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if list(protocol["material_ids"]) != list(source["material_ids"]):
        raise ValueError("A11.0 material IDs must exactly match the frozen endpoint-ID source protocol")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch = _load_panel(
        source,
        ROOT,
        torch.device("cpu"),
        preprocessed_cache=_resolve(source["data"]["preprocessed_cache"]),
    )
    geometry = protocol["geometry"]
    summaries, orbits, operations = [], [], []
    for target, material_id in enumerate(protocol["material_ids"]):
        nodes = batch.batch == target
        audit = audit_unlabeled_periodic_site_orbits(
            batch.lattice[target].numpy(),
            batch.frac_coords[nodes].numpy(),
            batch.atom_types[nodes].numpy(),
            symprec=float(geometry["symprec_angstrom"]),
            angle_tolerance=float(geometry["angle_tolerance_degrees"]),
            mapping_tolerance=float(geometry["site_mapping_tolerance_angstrom"]),
        )
        summary_rows, orbit_rows, operation_rows = _flatten_records(str(material_id), audit)
        summaries.extend(summary_rows)
        orbits.extend(orbit_rows)
        operations.extend(operation_rows)
    summary_frame = pd.DataFrame(summaries)
    orbit_frame = pd.DataFrame(orbits)
    operation_frame = pd.DataFrame(operations)
    summary_frame.to_csv(output / "summary.csv", index=False)
    orbit_frame.to_csv(output / "site_orbits.csv", index=False)
    operation_frame.to_csv(output / "automorphisms.csv", index=False)
    report_path = output / "a11_0_periodic_site_orbits_report.md"
    report_path.write_text(_report(protocol, summary_frame, orbit_frame), encoding="utf-8")
    files = ["summary.csv", "site_orbits.csv", "automorphisms.csv", report_path.name]
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": protocol["status"],
        "training_started": False,
        "historical_gate_evidence_modified": False,
        "protocol_sha256": _sha256(protocol_path),
        "source_protocol_sha256": _sha256(source_path),
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
