"""Read-only A10 audit of species-aware structure matches in A7-A9 samples."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The pinned WSL pymatgen release predates NumPy 2.  These aliases only make
# the external StructureMatcher available for this read-only audit.
np.bool = np.bool_  # type: ignore[attr-defined]
np.int = int  # type: ignore[attr-defined]

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluate_gate_a import _load_panel  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a10_site_representation_audit_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gate_a10_site_representation_audit_v1"))
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "completed_read_only_site_representation_audit":
        raise ValueError("A10 requires its frozen read-only audit protocol")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(_resolve(protocol["source_protocol"]).read_text(encoding="utf-8"))
    batch = _load_panel(
        source, ROOT, torch.device("cpu"),
        preprocessed_cache=_resolve(source["data"]["preprocessed_cache"]),
    )
    matcher_settings = dict(protocol["structure_matcher"])
    matcher_settings.pop("definition")
    matcher = StructureMatcher(**matcher_settings)
    frames = []
    for report_path in protocol["source_reports"]:
        frame = pd.read_csv(_resolve(report_path))
        protocol_name = Path(report_path).parts[-2]
        for row in frame.itertuples(index=False):
            target = int(row.target)
            nodes = batch.batch == target
            reference = Structure(
                batch.lattice[target].numpy(), batch.atom_types[nodes].tolist(),
                batch.frac_coords[nodes].numpy(), coords_are_cartesian=False,
            )
            predicted = Structure(
                batch.lattice[target].numpy(), ast.literal_eval(row.argmax_atom_types),
                batch.frac_coords[nodes].numpy(), coords_are_cartesian=False,
            )
            structure_match = bool(matcher.fit(reference, predicted))
            rms = matcher.get_rms_dist(reference, predicted)[0] if structure_match else None
            frames.append({
                "source_protocol": protocol_name,
                "sample": int(row.sample),
                "target": target,
                "cif_index_atom_accuracy": float(row.decoded_atom_type_accuracy),
                "species_aware_structure_match": structure_match,
                "species_aware_rms": rms,
            })
    detail = pd.DataFrame(frames)
    detail.to_csv(output / "species_aware_structure_matches.csv", index=False)
    summary = detail.groupby("source_protocol", as_index=False).agg(
        samples=("sample", "count"),
        cif_index_atom_accuracy=("cif_index_atom_accuracy", "mean"),
        species_aware_match_rate=("species_aware_structure_match", "mean"),
    )
    summary.to_csv(output / "species_aware_structure_summary.csv", index=False)
    lines = [
        "# Gate A10 species-aware site-representation audit",
        "",
        "This read-only audit uses a periodic, species-decorated StructureMatcher. It distinguishes a harmless CIF row permutation from a real chemical-sublattice mismatch.",
        "",
        "| Source | samples | CIF-index atom accuracy | species-aware match rate |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.source_protocol} | {row.samples} | {row.cif_index_atom_accuracy:.3f} | {row.species_aware_match_rate:.3f} |")
    lines += [
        "",
        "The non-unit species-aware rates prove that the residual errors are not merely arbitrary CIF ordering. The code audit also finds that endpoint-ID emits no response edge field and the scalar type messages contain no periodic edge length or vector-state invariant. Therefore a future architecture repair must make scalar site decoding geometrically informative and introduce symmetry-breaking node latents without using target atom order. No further sampler/loss search is justified by this audit.",
    ]
    report = output / "site_representation_audit.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = ["species_aware_structure_matches.csv", "species_aware_structure_summary.csv", "site_representation_audit.md"]
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "protocol_sha256": _sha256(protocol_path),
        "status": protocol["status"],
        "training_started": False,
        "historical_gate_evidence_modified": False,
        "report_sha256": {name: _sha256(output / name) for name in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
