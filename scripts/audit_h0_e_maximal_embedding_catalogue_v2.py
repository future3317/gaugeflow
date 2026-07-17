"""Independent exhaustive audit of the frozen H0-E-v2 E0 embedding catalogue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gaugeflow.catalogue import RationalAffineTransform, certify_affine_subgroup_inclusion
from gaugeflow.file_utils import sha256_file


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _groups() -> dict[int, dict[str, object]]:
    import spglib
    from pyxtal.symmetry import Group

    groups: dict[int, dict[str, object]] = {}
    for number in range(1, 231):
        group = Group(number)
        matrices = [np.asarray(value.affine_matrix, dtype=np.float64) for value in group[0].ops]
        rotations = np.rint([value[:3, :3] for value in matrices]).astype(np.int64)
        translations = np.asarray([value[:3, 3] for value in matrices], dtype=np.float64)
        identified = spglib.get_spacegroup_type_from_symmetry(
            rotations, translations, np.eye(3), symprec=1e-6
        )
        groups[number] = {
            "rotations": rotations,
            "translations": translations,
            "labels": {value.get_label() for value in group},
            "parent_label_count": len(group),
            "identified": int(identified.number) if identified is not None else None,
        }
    return groups


def _sort_key(value: dict[str, object]) -> tuple[object, ...]:
    return (
        int(value["child_space_group"]),
        int(value["parent_space_group"]),
        str(value["kind"]),
        int(value["subgroup_index"]),
        str(value["embedding_key"]),
    )


def audit(
    config: dict[str, Any],
    data_root: Path,
    pyxtal_root: Path,
    pyxtal_license: Path,
) -> dict[str, Any]:
    manifest_path = data_root / config["required_outputs"]["catalogue_manifest"]
    records_path = data_root / config["required_outputs"]["catalogue_records"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with gzip.open(records_path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("embedding catalogue records must be a JSON list")
    source_paths = {
        "t_subgroup": pyxtal_root / "database" / "t_subgroup.json",
        "k_subgroup": pyxtal_root / "database" / "k_subgroup.json",
        "wyckoff_list": pyxtal_root / "database" / "wyckoff_list.csv",
        "license": pyxtal_license,
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    groups = _groups()
    maximum_rotation_error = 0.0
    maximum_translation_error = 0.0
    inclusion_failures = 0
    key_failures = 0
    relation_failures = 0
    denominator_failures = 0
    representative_kernel_failures = 0
    seen_keys: set[str] = set()
    for record in records:
        denominator = int(record["transform_denominator"])
        compact = np.asarray(record["transform_numerators"], dtype=np.int64).reshape(3, 4)
        homogeneous = np.zeros((4, 4), dtype=np.int64)
        homogeneous[:3] = compact
        homogeneous[3, 3] = denominator
        transform = RationalAffineTransform(homogeneous, denominator)
        parent = int(record["parent_space_group"])
        child = int(record["child_space_group"])
        certificate = certify_affine_subgroup_inclusion(
            groups[parent]["rotations"],
            groups[parent]["translations"],
            groups[child]["rotations"],
            groups[child]["translations"],
            transform,
        )
        maximum_rotation_error = max(
            maximum_rotation_error, certificate.maximum_rotation_error
        )
        maximum_translation_error = max(
            maximum_translation_error,
            certificate.maximum_periodic_translation_error,
        )
        inclusion_failures += not certificate.passed
        representative_kernel_failures += (
            int(record["representative_image_order"])
            != certificate.representative_image_order
            or int(record["representative_kernel_size"])
            != certificate.representative_kernel_size
        )
        payload = {
            "parent": parent,
            "child": child,
            "kind": str(record["kind"]),
            "subgroup_index": int(record["subgroup_index"]),
            "denominator": denominator,
            "numerators": tuple(int(value) for value in compact.ravel()),
        }
        expected_key = _json_hash(payload)
        key_failures += expected_key != record["embedding_key"] or expected_key in seen_keys
        seen_keys.add(expected_key)
        denominator_failures += denominator > int(
            config["numerical_thresholds"]["maximum_rational_denominator"]
        )
        child_labels = groups[child]["labels"]
        for relation in record["relation_variants"]:
            if len(relation) != groups[parent]["parent_label_count"]:
                relation_failures += 1
                continue
            if any(label not in child_labels for children in relation for label in children):
                relation_failures += 1
    counts = {
        "raw_records_from_multiplicity": sum(int(value["raw_multiplicity"]) for value in records),
        "unique_affine_embeddings": len(records),
        "normalized_relation_variants": sum(len(value["relation_variants"]) for value in records),
        "embeddings_with_multiple_relation_variants": sum(
            len(value["relation_variants"]) > 1 for value in records
        ),
        "parent_space_groups": len({int(value["parent_space_group"]) for value in records}),
        "child_space_groups": len({int(value["child_space_group"]) for value in records}),
    }
    expected = config["expected_counts"]
    checks = {
        "source_hashes_independently_match": source_hashes == config["source"]["sha256"],
        "records_hash_matches_manifest": sha256_file(records_path)
        == manifest["records_sha256"],
        "manifest_protocol_matches": manifest.get("protocol") == config["protocol"],
        "all_230_settings_independently_identified": all(
            value["identified"] == number for number, value in groups.items()
        ),
        "all_compiled_embeddings_independently_certified": inclusion_failures == 0,
        "embedding_keys_unique_and_recomputed": key_failures == 0,
        "wyckoff_relation_labels_and_lengths_valid": relation_failures == 0,
        "rational_denominators_within_frozen_bound": denominator_failures == 0,
        "finite_representative_kernels_recomputed": representative_kernel_failures
        == 0,
        "records_are_reverse_index_sorted": records == sorted(records, key=_sort_key),
        "raw_multiplicity_does_not_duplicate_records": len(records) == len(seen_keys),
        "frozen_counts_reproduced": (
            counts["raw_records_from_multiplicity"] == int(expected["raw_records"])
            and counts["unique_affine_embeddings"] == int(expected["unique_affine_embeddings"])
            and counts["normalized_relation_variants"]
            == int(expected["normalized_relation_variants"])
            and counts["embeddings_with_multiple_relation_variants"]
            == int(expected["embeddings_with_multiple_relation_variants"])
        ),
        "all_parent_and_child_space_groups_covered": counts["parent_space_groups"] == 230
        and counts["child_space_groups"] == 230,
        "builder_manifest_was_qualified": manifest.get("qualified") is True,
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"] + "_independent_audit",
        "audit_passed": passed,
        "checks": checks,
        "counts": counts,
        "maximum_rotation_error": maximum_rotation_error,
        "maximum_periodic_translation_error": maximum_translation_error,
        "records_sha256": sha256_file(records_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_hashes,
        "decision": (
            "H0-E-v2-E0_qualified_parent_occurrence_E1_may_start"
            if passed
            else "H0-E-v2-E0_failed_stop_before_parent_occurrence_E1"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pyxtal-root", type=Path, required=True)
    parser.add_argument("--pyxtal-license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(config, args.data_root, args.pyxtal_root, args.pyxtal_license)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["audit_passed"] else 2)


if __name__ == "__main__":
    main()
