"""Fail-closed H0-D OPD physical-path catalogue qualification.

This auditor deliberately rejects a point-group-only or single-k catalogue.
It does not build missing evidence and never upgrades file presence to a
scientific qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions(config: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, bool]]:
    observed: dict[str, str | None] = {}
    checks: dict[str, bool] = {}
    for name, source in config["qualified_sources"].items():
        try:
            version = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            version = None
        observed[name] = version
        checks[f"source_version_{name}"] = version == source["version"]
    return observed, checks


def audit(config: dict[str, Any], data_root: Path) -> dict[str, Any]:
    observed_versions, checks = _package_versions(config)
    manifest_path = data_root / config["required_catalogue_manifest"]
    checks["catalogue_manifest_present"] = manifest_path.is_file()
    manifest: dict[str, Any] = {}
    errors: list[str] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"catalogue manifest is unreadable: {exc}")

    required = set(config["required_checks"])
    manifest_checks = manifest.get("checks", {}) if isinstance(manifest, dict) else {}
    checks["protocol_matches"] = manifest.get("protocol") == config["protocol"]
    checks["required_check_names_complete"] = required <= set(manifest_checks)
    for name in sorted(required):
        checks[f"catalogue_{name}"] = manifest_checks.get(name) is True

    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    checks["all_230_parents_present"] = counts.get("parent_space_groups") == 230
    checks["all_230_exact_branches_present"] = counts.get("exact_branches") == 230
    checks["distorted_physical_classes_present"] = (
        isinstance(counts.get("distorted_physical_classes"), int)
        and counts["distorted_physical_classes"] > 0
    )

    measure = manifest.get("measure", {}) if isinstance(manifest, dict) else {}
    maximum_mass_error = measure.get("maximum_parent_mass_sum_abs_error")
    checks["class_mass_normalized"] = (
        isinstance(maximum_mass_error, (int, float))
        and math.isfinite(maximum_mass_error)
        and maximum_mass_error
        <= config["numerical_thresholds"]["mass_sum_abs_error_float64"]
    )
    checks["tuple_multiplicity_excluded_from_measure"] = (
        measure.get("tuple_multiplicity_affects_mass") is False
    )

    schema = manifest.get("schema", {}) if isinstance(manifest, dict) else {}
    checks["affine_operation_schema"] = schema.get("stabilizer_operation") == (
        "fractional_rotation_plus_translation_coset"
    )
    checks["full_real_k_star_schema"] = schema.get("mode_representation") == "full_real_k_star"
    checks["displacement_occurrence_schema"] = (
        schema.get("mode_occurrence") == "parent_wyckoff_displacement_representation"
    )

    qualified = not errors and all(checks.values())
    return {
        "protocol": config["protocol"],
        "qualified": qualified,
        "decision": "H0-D_qualified_H0-E_may_start" if qualified else "H0-D_failed_stop_before_H0-E",
        "catalogue_manifest": str(manifest_path),
        "catalogue_manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "observed_package_versions": observed_versions,
        "checks": checks,
        "counts": counts,
        "errors": errors,
        "scientific_note": (
            "spgrep-modulation is a cross-check for little-group isotropy branches; "
            "it is not accepted as a full real-k-star affine catalogue by itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(config, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
