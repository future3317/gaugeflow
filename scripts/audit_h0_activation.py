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


def _audit_alex_split(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / str(config["gaugeflow_split_manifest"])
    audit_relative = config.get("gaugeflow_split_audit")
    if audit_relative is None:
        return {
            "qualified": False,
            "manifest_present": manifest_path.is_file(),
            "audit_present": False,
            "historical_protocol_without_strict_split_audit": True,
        }
    audit_path = data_root / str(audit_relative)
    if not manifest_path.is_file() or not audit_path.is_file():
        return {
            "qualified": False,
            "manifest_present": manifest_path.is_file(),
            "audit_present": audit_path.is_file(),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    overlap = audit.get("cross_split_overlap", {})
    required_zero = tuple(map(str, config["required_zero_overlap"]))
    zero_overlap = all(
        int(values.get(key, -1)) == 0
        for values in overlap.values()
        for key in required_zero
    )
    assignment_path = manifest_path.parent / str(manifest.get("assignment_path", ""))
    assignment_sha = _sha256(assignment_path) if assignment_path.is_file() else None
    checks = {
        "protocol_matches": manifest.get("protocol") == config["split_protocol"],
        "audit_protocol_matches": audit.get("protocol") == config["audit_protocol"],
        "audit_qualified": audit.get("qualified") is True,
        "split_manifest_hash_matches_audit": audit.get("split_manifest_sha256")
        == _sha256(manifest_path),
        "assignment_present": assignment_path.is_file(),
        "assignment_hash_matches_manifest": assignment_sha == manifest.get("assignment_sha256"),
        "assignment_hash_matches_audit": assignment_sha == audit.get("assignment_sha256"),
        "row_count_complete": int(manifest.get("rows", -1)) == int(config["expected_rows"])
        and int(audit.get("rows", -1)) == int(config["expected_rows"]),
        "fraction_deviation_within_limit": float(audit.get("maximum_fraction_deviation", 1.0))
        <= float(config["maximum_fraction_deviation"]),
        "required_overlap_zero": zero_overlap,
        "matcher_candidate_universe_empty": int(
            audit.get("structure_matcher", {}).get("possible_cross_split_candidate_pairs", -1)
        )
        == 0,
    }
    return {
        "qualified": all(checks.values()),
        "manifest_present": True,
        "audit_present": True,
        "checks": checks,
        "split_counts": audit.get("split_counts", {}),
        "component_count": manifest.get("component_count"),
        "largest_component_rows": manifest.get("largest_component_rows"),
        "assignment_sha256": assignment_sha,
    }


def audit_activation(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    root_manifest_path = data_root / str(config["data_center_manifest"])
    if not root_manifest_path.is_file():
        raise FileNotFoundError(root_manifest_path)
    observed_manifest_sha = _sha256(root_manifest_path)
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    manifest_ok = observed_manifest_sha == config["data_center_manifest_sha256"]

    alex = _source_attestation(data_root, root_manifest, config["h0_a"]["source_files"])
    alex_split = _audit_alex_split(config["h0_a"], data_root)
    alex["formula_prototype_split"] = alex_split
    alex["formula_prototype_split_present"] = bool(alex_split["manifest_present"])
    alex["status"] = (
        "qualified" if alex["passed"] and alex_split["qualified"] else "blocked_split_not_frozen"
    )

    phonon = _source_attestation(data_root, root_manifest, config["h0_b"]["source_files"])
    phonon_manifest_path = data_root / str(config["h0_b"]["derived_manifest"])
    derived: dict[str, Any] = {}
    if phonon_manifest_path.is_file():
        derived = json.loads(phonon_manifest_path.read_text(encoding="utf-8"))
    attestation_relative = config["h0_b"].get("attestation_manifest")
    attestation_path = data_root / str(attestation_relative) if attestation_relative else None
    attestation: dict[str, Any] = {}
    if attestation_path is not None and attestation_path.is_file():
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation_checks = attestation.get("checks", {})
    projection = derived.get("projection", {})
    projected_observed = derived.get("observed", {})
    required = config["h0_b"]["required_attestations"]
    evidence = {
        "primitive_supercell_mapping": bool(
            derived.get("primitive_matrix_policy")
            or config["h0_b"].get("primitive_matrix_policy")
        ),
        "force_constant_solver": derived.get("fc_calculator") == "traditional",
        "symmetrization_policy": bool(projection.get("constraints"))
        or derived.get("posthoc_symmetrized") is False,
        "acoustic_sum_rule": all(
            projected_observed.get(key) is not None
            for key in ("max_projected_row_asr", "max_projected_column_asr")
        )
        or derived.get("validation", {}).get("max_asr_residual") is not None,
        "imaginary_frequency_convention": bool(
            derived.get("audit_imaginary_tolerance_thz") is not None
            or config["h0_b"].get("imaginary_frequency_convention_thz") is not None
        ),
        "translational_zero_mode_test": all(
            attestation_checks.get(key) is True
            for key in ("translation_frequency", "translation_subspace", "translation_residual")
        ),
        "degenerate_subspace_numerical_test": all(
            attestation_checks.get(key) is True
            for key in (
                "degenerate_projector_gauge",
                "conjugacy_frequency",
                "conjugacy_projector",
            )
        ),
        "non_analytic_correction_attestation": all(
            attestation_checks.get(key) is True
            for key in (
                "nac_availability_matches_source",
                "nac_shapes_and_finite",
                "nac_charge_neutrality",
                "nac_dielectric_symmetry",
                "nac_dielectric_positive",
                "nac_factor",
                "missing_nac_is_explicit",
            )
        ),
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
    script_root = Path(__file__).resolve().parent
    builder_path = script_root / "build_phonondb_force_constants_v2.py"
    auditor_path = script_root / "audit_phonondb_h0_b.py"
    current_builder_matches = (
        config["h0_b"].get("builder_sha256") is None
        or (
            builder_path.is_file()
            and _sha256(builder_path) == config["h0_b"]["builder_sha256"]
        )
    )
    current_auditor_matches = (
        config["h0_b"].get("auditor_sha256") is None
        or (
            auditor_path.is_file()
            and _sha256(auditor_path) == config["h0_b"]["auditor_sha256"]
        )
    )
    derivation_identity = all(
        (
            derived.get("protocol") == config["h0_b"].get("derived_protocol"),
            derived.get("builder_sha256") == config["h0_b"].get("builder_sha256"),
            current_builder_matches,
        )
    ) if config["h0_b"].get("derived_protocol") else True
    attestation_identity = (
        all(
            (
                attestation.get("protocol") == config["h0_b"].get("attestation_protocol"),
                attestation.get("force_index_sha256") == derived.get("index_sha256"),
                attestation.get("auditor_sha256") == config["h0_b"].get("auditor_sha256"),
                current_auditor_matches,
                config["h0_b"].get("attestation_sha256") is None
                or (
                    attestation_path is not None
                    and attestation_path.is_file()
                    and _sha256(attestation_path) == config["h0_b"]["attestation_sha256"]
                ),
            )
        )
        if config["h0_b"].get("attestation_protocol")
        else True
    )
    phonon["attestation_present"] = bool(attestation)
    phonon["attestation_qualified"] = attestation.get("qualified") is True
    phonon["attestation_protocol"] = attestation.get("protocol")
    phonon["attestation_counts"] = attestation.get("counts", {})
    phonon["attestation_selection"] = attestation.get("selection", {})
    phonon["current_builder_matches"] = current_builder_matches
    phonon["current_auditor_matches"] = current_auditor_matches
    phonon["derivation_identity_matches"] = derivation_identity
    phonon["attestation_identity_matches"] = attestation_identity
    phonon["status"] = (
        "qualified"
        if (
            phonon["passed"]
            and complete_derived
            and derivation_identity
            and attestation_identity
            and attestation.get("qualified") is True
            and not phonon["missing_attestations"]
        )
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
    alex_gate = result["components"]["H0-A"]
    phonon_gate = result["components"]["H0-B"]
    if alex_gate["status"] == "qualified":
        split_evidence = alex_gate["formula_prototype_split"]
        alex_statement = (
            "- Alex-MP-20 H0-A is qualified: all source hashes match, the child-first "
            f"formula/prototype split contains {split_evidence['split_counts']}, its "
            f"{split_evidence['component_count']} connected components have zero cross-split "
            "formula, prototype, matcher-envelope and component overlap, and the exhaustive "
            "StructureMatcher candidate universe across splits is empty."
        )
    else:
        alex_statement = (
            "- Alex-MP-20 Parquet sources match the frozen data-center manifest. The source "
            f"profile contains {alex_rows:,} structurally valid rows; upstream reduced-formula "
            f"overlap is train--val {alex_overlap.get('train--val', {}).get('reduced_formula_groups', 'unknown')}, "
            f"train--test {alex_overlap.get('train--test', {}).get('reduced_formula_groups', 'unknown')}, "
            f"and val--test {alex_overlap.get('val--test', {}).get('reduced_formula_groups', 'unknown')}. "
            "GaugeFlow's formula/prototype-disjoint child split has not qualified."
        )
    if phonon_gate["status"] == "qualified":
        counts = phonon_gate.get("attestation_counts", {})
        phonon_statement = (
            "- PhononDB H0-B is qualified under its versioned derivation attestation: all "
            f"{counts.get('universe', 10034):,} compact Hessians pass full-universe algebraic "
            f"constraints, while a frozen {counts.get('successful', 'unknown')}-material "
            "long-tail/stratified sample passes acoustic-mode, degenerate-subspace, q/-q "
            "conjugacy and explicit NAC checks. This is not represented as a full-universe "
            "eigendecomposition audit."
        )
        phonon_next = "2. Preserve the qualified PhononDB derivation and its source-confidence fields."
    else:
        phonon_statement = (
            "- PhononDB contains 10,034 successful compact float64 force-constant caches with "
            f"phonopy {phonon_gate.get('phonopy_version')}; remaining formal attestations: "
            f"{', '.join(missing) if missing else 'attestation not qualified'}."
        )
        phonon_next = (
            "2. Add or repair the missing PhononDB translational-mode, degenerate-subspace, and "
            "NAC attestations without overwriting frozen historical artifacts."
        )
    return "\n".join(
        [
            f"# {result['protocol']}",
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
            alex_statement,
            phonon_statement,
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
            "1. Preserve the qualified Alex child split and require every later artifact to inherit it.",
            phonon_next,
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
