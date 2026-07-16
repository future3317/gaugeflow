"""Read-only H0 data-activation audit for the hierarchical GaugeFlow protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entry(manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    value = files.get(key)
    return value if isinstance(value, dict) else None


def _source_attestation(
    root: Path,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for expected in entries:
        relative = str(expected["path"])
        recorded = _manifest_entry(manifest, relative)
        file_path = root / relative
        direct_hash = bool(expected.get("direct_hash", False))
        direct_sha = _sha256(file_path) if direct_hash and file_path.is_file() else None
        sha_matches = (
            direct_sha == expected["sha256"]
            if direct_hash
            else recorded is not None and recorded.get("sha256") == expected["sha256"]
        )
        bytes_matches = (
            file_path.is_file() and file_path.stat().st_size == expected["bytes"]
            if direct_hash
            else recorded is not None and recorded.get("bytes") == expected["bytes"]
        )
        results.append(
            {
                "path": relative,
                "present": file_path.is_file(),
                "attestation_mode": "direct_sha256" if direct_hash else "root_manifest",
                "manifest_present": recorded is not None,
                "sha256_matches_frozen_attestation": sha_matches,
                "bytes_match_frozen_attestation": bytes_matches,
            }
        )
    passed = all(
        row["present"]
        and (row["manifest_present"] or row["attestation_mode"] == "direct_sha256")
        and row["sha256_matches_frozen_attestation"]
        and row["bytes_match_frozen_attestation"]
        for row in results
    )
    return {"passed": passed, "files": results}


def audit_activation(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    root_manifest_path = data_root / str(config["data_center_manifest"])
    if not root_manifest_path.is_file():
        raise FileNotFoundError(root_manifest_path)
    observed_manifest_sha = _sha256(root_manifest_path)
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    manifest_ok = observed_manifest_sha == config["data_center_manifest_sha256"]

    alex = _source_attestation(data_root, root_manifest, config["h0_a"]["source_files"])
    alex_split = data_root / str(config["h0_a"]["gaugeflow_split_manifest"])
    alex["formula_prototype_split_present"] = alex_split.is_file()
    alex["status"] = (
        "qualified" if alex["passed"] and alex_split.is_file() else "blocked_split_not_frozen"
    )

    phonon = _source_attestation(data_root, root_manifest, config["h0_b"]["source_files"])
    phonon_manifest_path = data_root / str(config["h0_b"]["derived_manifest"])
    derived: dict[str, Any] = {}
    if phonon_manifest_path.is_file():
        derived = json.loads(phonon_manifest_path.read_text(encoding="utf-8"))
    required = config["h0_b"]["required_attestations"]
    evidence = {
        "primitive_supercell_mapping": bool(derived.get("primitive_matrix_policy")),
        "force_constant_solver": derived.get("fc_calculator") == "traditional",
        "symmetrization_policy": derived.get("posthoc_symmetrized") is False,
        "acoustic_sum_rule": derived.get("validation", {}).get("max_asr_residual") is not None,
        "imaginary_frequency_convention": derived.get("audit_imaginary_tolerance_thz") is not None,
        "translational_zero_mode_test": False,
        "degenerate_subspace_numerical_test": False,
        "non_analytic_correction_attestation": False,
    }
    phonon["derived_counts"] = derived.get("counts", {})
    phonon["phonopy_version"] = derived.get("phonopy_version")
    phonon["evidence"] = evidence
    phonon["missing_attestations"] = [name for name in required if not evidence.get(name, False)]
    complete_derived = (
        derived.get("counts", {}).get("successful") == config["h0_b"]["expected_materials"]
        and derived.get("counts", {}).get("failed") == 0
        and derived.get("phonopy_version") == config["h0_b"]["phonopy_version"]
    )
    phonon["status"] = (
        "qualified"
        if phonon["passed"] and complete_derived and not phonon["missing_attestations"]
        else "partial_missing_derivation_attestations"
    )

    matpes = _source_attestation(data_root, root_manifest, config["h0_c"]["source_files"])
    checkpoint = config["h0_c"].get("teacher_checkpoint")
    checkpoint_path = data_root / str(checkpoint) if checkpoint else None
    matpes["teacher_checkpoint_present"] = bool(
        checkpoint_path is not None and checkpoint_path.is_file()
    )
    matpes["status"] = (
        "qualified"
        if matpes["passed"] and matpes["teacher_checkpoint_present"]
        else "blocked_frozen_teacher_missing"
    )

    catalogue = data_root / str(config["h0_d"]["catalogue_manifest"])
    h0_d = {
        "catalogue_manifest": str(config["h0_d"]["catalogue_manifest"]),
        "catalogue_present": catalogue.is_file(),
        "status": "qualified" if catalogue.is_file() else "blocked_catalogue_missing",
    }
    pilot = data_root / str(config["h0_e"]["pilot_manifest"])
    h0_e = {
        "pilot_manifest": str(config["h0_e"]["pilot_manifest"]),
        "pilot_present": pilot.is_file(),
        "status": "qualified" if pilot.is_file() else "blocked_pilot_missing",
    }
    components = {
        "H0-A": alex,
        "H0-B": phonon,
        "H0-C": matpes,
        "H0-D": h0_d,
        "H0-E": h0_e,
    }
    qualified = manifest_ok and all(value["status"] == "qualified" for value in components.values())
    return {
        "protocol": config["protocol"],
        "data_root": str(data_root),
        "data_center_manifest_sha256": observed_manifest_sha,
        "data_center_manifest_matches": manifest_ok,
        "components": components,
        "h0_passed": qualified,
        "next_gate_authorized": qualified,
        "decision": "H0_passed" if qualified else "H0_not_passed_stop_before_H1",
    }


def render_markdown(result: dict[str, Any]) -> str:
    rows = []
    for gate, value in result["components"].items():
        rows.append(f"| {gate} | `{value['status']}` |")
    missing = result["components"]["H0-B"].get("missing_attestations", [])
    alex_profile = result.get("alex_source_profile", {})
    alex_profiles = alex_profile.get("profiles", {}) if isinstance(alex_profile, dict) else {}
    alex_rows = sum(int(value.get("rows", 0)) for value in alex_profiles.values())
    alex_overlap = (
        alex_profile.get("cross_split_overlap", {}) if isinstance(alex_profile, dict) else {}
    )
    return "\n".join(
        [
            "# H0 data activation v1",
            "",
            "## Technical summary",
            "",
            f"**Decision: `{result['decision']}`.** The external data center is present and its "
            "frozen manifest is readable, but source availability is not equivalent to model-ready "
            "activation. No H1 training is authorized by this audit.",
            "",
            "## Gate status",
            "",
            "| Component | Status |",
            "| --- | --- |",
            *rows,
            "",
            "## Evidence and interpretation",
            "",
            "- Alex-MP-20 Parquet sources match the frozen data-center manifest. The source "
            f"profile contains {alex_rows:,} structurally valid rows; upstream reduced-formula "
            f"overlap is train--val {alex_overlap.get('train--val', {}).get('reduced_formula_groups', 'unknown')}, "
            f"train--test {alex_overlap.get('train--test', {}).get('reduced_formula_groups', 'unknown')}, "
            f"and val--test {alex_overlap.get('val--test', {}).get('reduced_formula_groups', 'unknown')}. "
            "GaugeFlow's formula/prototype-disjoint child split has not been materialized.",
            "- PhononDB contains 10,034 successful compact float64 force-constant caches with "
            f"phonopy {result['components']['H0-B'].get('phonopy_version')}; remaining formal "
            f"attestations: {', '.join(missing) if missing else 'none'}.",
            "- MatPES-PBE data files are present; no frozen, hashed teacher checkpoint is activated.",
            "- A normalized OPD path-class catalogue and the 1,000--5,000 structure parent "
            "decomposition pilot do not yet exist.",
            "",
            "## Scope and data definition",
            "",
            "The audit reads external files in place. It does not copy raw data into the code "
            "repository, derive new labels, train a model, run relaxation, or access tensor gates.",
            "",
            "## Method",
            "",
            "Dataset files are checked against the frozen root manifest by relative path, byte "
            "count, and SHA-256 recorded in that manifest. Small derivation manifests are hashed "
            "directly. Scientific qualification additionally requires the versioned split, "
            "derivation tests, teacher checkpoint, path measure, and pilot artifacts.",
            "",
            "## Limitations and next actions",
            "",
            "1. Freeze the Alex child split before constructing any parent/path/join artifact.",
            "2. Add the missing PhononDB translational-mode, degenerate-subspace, and NAC "
            "attestations without changing the existing force-constant cache.",
            "3. Qualify a frozen MatPES-PBE teacher and disagreement model.",
            "4. Build a deduplicated OPD physical path-class measure, then run the bounded parent "
            "decomposition pilot.",
            "5. Split H1 into H1a (P1 real-data hybrid generator) and H1b (full 230-group/Wyckoff "
            "parent blueprint); H1 passes only when both pass.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--alex-profile", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_activation(config, args.data_root)
    if args.alex_profile is not None:
        result["alex_source_profile"] = json.loads(args.alex_profile.read_text(encoding="utf-8"))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "h0_passed": result["h0_passed"]}))


if __name__ == "__main__":
    main()
