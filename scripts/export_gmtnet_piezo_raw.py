"""Export a pinned GMTNet piezo pickle into GaugeFlow's raw-record contract.

The input is read as data only: this script imports no PiezoJet Python code and
creates no runtime dependency between the two projects.  It gives a local,
hash-pinned GMTNet/JARVIS source copy an explicit per-record schema before the
separate GaugeFlow v2 target-cache builder converts it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.core import Lattice, Structure


SOURCE_ORDER = ["xx", "yy", "zz", "xy", "yz", "xz"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_record(source: dict[str, Any]) -> dict[str, Any]:
    """Validate one GMTNet record and emit the standalone raw-record schema."""
    required = {"JARVIS_ID", "atoms", "piezoelectric_C_m2"}
    missing = sorted(required.difference(source))
    if missing:
        raise ValueError(f"GMTNet record is missing {missing}")
    atoms = source["atoms"]
    if not isinstance(atoms, dict):
        raise ValueError("GMTNet atoms field must be a dictionary")
    atom_fields = {"lattice_mat", "coords", "elements", "cartesian"}
    missing_atoms = sorted(atom_fields.difference(atoms))
    if missing_atoms:
        raise ValueError(f"GMTNet atoms field is missing {missing_atoms}")
    if bool(atoms["cartesian"]):
        raise ValueError("this v1 exporter expects fractional GMTNet coordinates")
    structure = Structure(
        Lattice(atoms["lattice_mat"]), atoms["elements"], atoms["coords"], coords_are_cartesian=False
    )
    piezo = source["piezoelectric_C_m2"]
    if len(piezo) != 3 or any(len(row) != 6 for row in piezo):
        raise ValueError(f"{source['JARVIS_ID']}: piezoelectric_C_m2 is not [3,6]")
    return {
        "material_id": str(source["JARVIS_ID"]),
        "cif": structure.to(fmt="cif"),
        "piezo_voigt": piezo,
        "voigt_order": SOURCE_ORDER,
        "engineering_shear": True,
        "unit": "C/m^2",
        "source_record_fields": sorted(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pickle", type=Path, required=True)
    parser.add_argument("--source-commit", type=str, required=True)
    parser.add_argument("--source-url", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        help="Optional frozen v2 split. When supplied, write an exhaustive exclusion map for source IDs outside it.",
    )
    parser.add_argument(
        "--retrieved-utc",
        help="Original download timestamp if known. Omit to record this as a locally observed pinned copy.",
    )
    args = parser.parse_args()
    with args.source_pickle.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, list) or not all(isinstance(record, dict) for record in raw):
        raise ValueError("GMTNet piezo source must be a list of dictionaries")
    records = [export_record(record) for record in raw]
    records.sort(key=lambda record: record["material_id"])
    ids = [record["material_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("GMTNet source contains duplicate JARVIS IDs")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "raw_records.json"
    records_path.write_text(json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8")
    raw_sha256 = sha256_file(records_path)
    manifest = {
        "source_name": "GMTNet JARVIS dielectric/piezo release",
        "source_release": f"GMTNet source commit {args.source_commit}",
        "source_url": args.source_url,
        "downloaded_filename": args.source_pickle.name,
        "download_sha256": raw_sha256,
        "normalized_records_sha256": raw_sha256,
        "source_pickle_sha256": sha256_file(args.source_pickle),
        "upstream_file_sha256": sha256_file(args.source_pickle),
        "retrieved_utc": args.retrieved_utc,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_copy_status": (
            "direct_download_timestamp_pinned" if args.retrieved_utc else "local_pinned_copy_direct_download_timestamp_unavailable"
        ),
        "license": "See the upstream YKQ98/GMTNet repository and JARVIS-DFT source terms.",
        "record_schema": "material_id,cif,piezo_voigt[3,6],voigt_order,engineering_shear,unit",
        "tensor_unit": "C/m^2",
        "source_voigt_order": SOURCE_ORDER,
        "engineering_shear": True,
        "record_count": len(records),
        "material_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    (output / "raw_release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.split is not None:
        split_payload = json.loads(args.split.read_text(encoding="utf-8"))
        split_ids = {
            str(material_id)
            for name in ("train", "val", "test")
            for material_id in split_payload[name]
        }
        missing = sorted(split_ids - set(ids))
        if missing:
            raise ValueError(f"frozen split IDs are absent from the GMTNet source: {missing[:5]}")
        exclusions = {
            material_id: "absent_from_frozen_tensororbit_4998_parent_population"
            for material_id in ids
            if material_id not in split_ids
        }
        (output / "exclusions.json").write_text(
            json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["frozen_split_sha256"] = sha256_file(args.split)
        manifest["frozen_split_record_count"] = len(split_ids)
        manifest["explicit_exclusion_count"] = len(exclusions)
        (output / "raw_release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
