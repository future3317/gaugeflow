"""Compile a frozen compact maximal-subgroup/Wyckoff source for H0-E-v2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gaugeflow.catalogue import (
    RationalAffineTransform,
    certify_affine_subgroup_inclusion,
    normalized_relation_variant,
)
from gaugeflow.file_utils import sha256_file


def _json_hash(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _group_contracts() -> dict[int, dict[str, object]]:
    import spglib
    from pyxtal.symmetry import Group

    contracts: dict[int, dict[str, object]] = {}
    for space_group in range(1, 231):
        group = Group(space_group)
        operations = [np.asarray(value.affine_matrix, dtype=np.float64) for value in group[0].ops]
        rotations = np.rint([value[:3, :3] for value in operations]).astype(np.int64)
        translations = np.asarray([value[:3, 3] for value in operations], dtype=np.float64)
        identified = spglib.get_spacegroup_type_from_symmetry(
            rotations,
            translations,
            np.eye(3),
            symprec=1e-6,
        )
        contracts[space_group] = {
            "rotations": rotations,
            "translations": translations,
            "wyckoff_labels": tuple(value.get_label() for value in group),
            "spglib_identified_number": (
                int(identified.number) if identified is not None else None
            ),
        }
    return contracts


def _raw_entries(
    t_source: Path,
    k_source: Path,
    contracts: dict[int, dict[str, object]],
    *,
    maximum_k_index: int,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path, kind in ((t_source, "t"), (k_source, "k")):
        source = json.loads(path.read_text(encoding="utf-8"))
        if set(source) != {str(value) for value in range(1, 231)}:
            raise ValueError(f"{path.name} does not contain all 230 parent keys")
        for parent_text, row in source.items():
            parent = int(parent_text)
            count = len(row["subgroup"])
            if any(
                len(row[key]) != count
                for key in ("index", "relations", "transformation", "type", "cosets")
            ):
                raise ValueError(f"{path.name} parent {parent} has misaligned fields")
            parent_labels = contracts[parent]["wyckoff_labels"]
            for source_index in range(count):
                subgroup_index = int(row["index"][source_index])
                if kind == "k" and subgroup_index > maximum_k_index:
                    continue
                child = int(row["subgroup"][source_index])
                transform = RationalAffineTransform.from_array(
                    np.asarray(row["transformation"][source_index]["data"], dtype=np.float64)
                )
                raw_relations = list(reversed(row["relations"][source_index]))
                if len(raw_relations) != len(parent_labels):
                    raise ValueError("Wyckoff splitting does not cover every parent orbit")
                child_labels = set(contracts[child]["wyckoff_labels"])
                if any(label not in child_labels for values in raw_relations for label in values):
                    raise ValueError("Wyckoff splitting references an unknown child orbit")
                relation = normalized_relation_variant(raw_relations)
                certificate = certify_affine_subgroup_inclusion(
                    contracts[parent]["rotations"],
                    contracts[parent]["translations"],
                    contracts[child]["rotations"],
                    contracts[child]["translations"],
                    transform,
                )
                if not certificate.passed:
                    raise ValueError(
                        f"uncertified maximal-subgroup edge {parent}->{child} "
                        f"({kind}, source index {source_index})"
                    )
                entries.append(
                    {
                        "parent_space_group": parent,
                        "child_space_group": child,
                        "kind": kind,
                        "subgroup_index": subgroup_index,
                        "cell_index": 1 if kind == "t" else subgroup_index,
                        "transform_denominator": transform.denominator,
                        "transform_numerators": tuple(
                            int(value) for value in transform.compact_numerators().ravel()
                        ),
                        "relation_variant": relation,
                        "source_index": source_index,
                        "maximum_rotation_error": certificate.maximum_rotation_error,
                        "maximum_periodic_translation_error": (
                            certificate.maximum_periodic_translation_error
                        ),
                        "representative_image_order": certificate.representative_image_order,
                        "representative_kernel_size": certificate.representative_kernel_size,
                        "uniform_representative_kernel": (
                            certificate.uniform_representative_kernel
                        ),
                    }
                )
    return entries


def _embedding_identity(entry: dict[str, object]) -> tuple[object, ...]:
    return (
        int(entry["parent_space_group"]),
        int(entry["child_space_group"]),
        str(entry["kind"]),
        int(entry["subgroup_index"]),
        int(entry["transform_denominator"]),
        tuple(entry["transform_numerators"]),
    )


def _compile(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        grouped[_embedding_identity(entry)].append(entry)
    records: list[dict[str, object]] = []
    for identity, values in grouped.items():
        relations = sorted({tuple(value["relation_variant"]) for value in values})
        source_indices = sorted({int(value["source_index"]) for value in values})
        parent, child, kind, subgroup_index, denominator, numerators = identity
        key_payload = {
            "parent": parent,
            "child": child,
            "kind": kind,
            "subgroup_index": subgroup_index,
            "denominator": denominator,
            "numerators": numerators,
        }
        records.append(
            {
                "embedding_key": _json_hash(key_payload),
                "parent_space_group": parent,
                "child_space_group": child,
                "kind": kind,
                "subgroup_index": subgroup_index,
                "cell_index": 1 if kind == "t" else subgroup_index,
                "transform_denominator": denominator,
                "transform_numerators": list(numerators),
                "relation_variants": [
                    [list(children) for children in relation] for relation in relations
                ],
                "raw_multiplicity": len(values),
                "source_indices": source_indices,
                "maximum_rotation_error": max(
                    float(value["maximum_rotation_error"]) for value in values
                ),
                "maximum_periodic_translation_error": max(
                    float(value["maximum_periodic_translation_error"])
                    for value in values
                ),
                "representative_image_order": min(
                    int(value["representative_image_order"]) for value in values
                ),
                "representative_kernel_size": max(
                    int(value["representative_kernel_size"]) for value in values
                ),
            }
        )
    records.sort(
        key=lambda value: (
            int(value["child_space_group"]),
            int(value["parent_space_group"]),
            str(value["kind"]),
            int(value["subgroup_index"]),
            str(value["embedding_key"]),
        )
    )
    return records


def build(
    config: dict[str, Any], pyxtal_root: Path, pyxtal_license: Path
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    sources = config["source"]
    t_source = pyxtal_root / "database" / "t_subgroup.json"
    k_source = pyxtal_root / "database" / "k_subgroup.json"
    wyckoff_source = pyxtal_root / "database" / "wyckoff_list.csv"
    paths = {
        "t_subgroup": t_source,
        "k_subgroup": k_source,
        "wyckoff_list": wyckoff_source,
        "license": pyxtal_license,
    }
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if observed_hashes != sources["sha256"]:
        raise ValueError("PyXtal source hashes do not match the frozen protocol")
    if importlib.metadata.version("pyxtal") != sources["version"]:
        raise ValueError("PyXtal version does not match the frozen protocol")
    contracts = _group_contracts()
    entries = _raw_entries(
        t_source,
        k_source,
        contracts,
        maximum_k_index=int(config["scope"]["maximum_k_subgroup_index"]),
    )
    records = _compile(entries)
    reversed_records = _compile(list(reversed(entries)))
    relation_variant_count = sum(len(value["relation_variants"]) for value in records)
    multiple_relation_embeddings = sum(len(value["relation_variants"]) > 1 for value in records)
    counts = {
        "raw_records": len(entries),
        "raw_t_records": sum(value["kind"] == "t" for value in entries),
        "raw_k_records": sum(value["kind"] == "k" for value in entries),
        "unique_affine_embeddings": len(records),
        "normalized_relation_variants": relation_variant_count,
        "embeddings_with_multiple_relation_variants": multiple_relation_embeddings,
        "raw_duplicate_records": len(entries) - len(records),
        "parent_space_groups": len({int(value["parent_space_group"]) for value in records}),
        "child_space_groups": len({int(value["child_space_group"]) for value in records}),
        "maximum_transform_denominator": max(
            int(value["transform_denominator"]) for value in records
        ),
        "maximum_representative_kernel_size": max(
            int(value["representative_kernel_size"]) for value in records
        ),
    }
    expected = config["expected_counts"]
    checks = {
        "source_hashes_match": True,
        "source_license_is_mit": "MIT License" in pyxtal_license.read_text(encoding="utf-8"),
        "all_230_group_settings_identified_by_spglib": all(
            value["spglib_identified_number"] == key for key, value in contracts.items()
        ),
        "all_source_edges_affine_inclusion_certified": all(
            float(value["maximum_rotation_error"]) <= 1e-9
            and float(value["maximum_periodic_translation_error"]) <= 1e-9
            for value in entries
        ),
        "all_wyckoff_labels_resolve": True,
        "raw_counts_match_frozen_source": all(counts[key] == int(expected[key]) for key in expected),
        "enumeration_order_invariance": records == reversed_records,
        "duplicate_expansion_does_not_create_embedding_records": len(records)
        == len({_embedding_identity(value) for value in entries}),
        "compact_integer_affine_storage": all(
            len(value["transform_numerators"]) == 12 for value in records
        ),
        "reverse_child_index_covers_all_230_groups": counts["child_space_groups"] == 230,
    }
    manifest = {
        "protocol": config["protocol"],
        "qualified": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "source": {
            "package": "PyXtal",
            "version": sources["version"],
            "license": "MIT",
            "sha256": observed_hashes,
            "role": "offline maximal subgroup and Wyckoff-splitting source only",
        },
        "representation": config["representation"],
        "decision": (
            "H0-E-v2-E0_qualified_parent_occurrence_E1_may_start"
            if all(checks.values())
            else "H0-E-v2-E0_failed_stop_before_parent_occurrence_E1"
        ),
    }
    return manifest, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pyxtal-root", type=Path, required=True)
    parser.add_argument("--pyxtal-license", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, records = build(config, args.pyxtal_root, args.pyxtal_license)
    records_path = args.data_root / config["required_outputs"]["catalogue_records"]
    manifest_path = args.data_root / config["required_outputs"]["catalogue_manifest"]
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text_handle:
                json.dump(records, text_handle, separators=(",", ":"), sort_keys=True)
    manifest["records_sha256"] = sha256_file(records_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    raise SystemExit(0 if manifest["qualified"] else 2)


if __name__ == "__main__":
    main()
