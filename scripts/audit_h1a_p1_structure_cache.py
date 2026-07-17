"""Independently rebuild and qualify every row of the H1a packed structure cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Hundreds of thousands of independent 3x3 solves are latency-bound. Spawning
# a BLAS team for each one is dramatically slower than scalar kernels and can
# occupy every host core without useful parallel work.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pyarrow.parquet as pq
import spglib
import torch

from gaugeflow.file_utils import load_json_object, sha256_file
from gaugeflow.production.alex_p1_data import (
    PACKED_ALEX_P1_PROTOCOL,
    PACKED_ALEX_P1_SCHEMA,
)

SOURCE_SPLITS = ("train", "val", "test")
REVERSE_SOURCE_SPLITS = tuple(reversed(SOURCE_SPLITS))
TARGET_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class IndependentRow:
    tokens: np.ndarray
    fractional: np.ndarray
    lattice: np.ndarray
    transform: np.ndarray
    source_error: float
    cache_error: float


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")


def _periodic_error(
    cartesian: np.ndarray, fractional: np.ndarray, lattice: np.ndarray
) -> float:
    generated = fractional @ lattice
    difference = generated - cartesian
    fractional_difference = np.linalg.solve(lattice.T, difference.T).T
    fractional_difference -= np.rint(fractional_difference)
    residual = fractional_difference @ lattice
    return float(np.max(np.sqrt(np.sum(residual * residual, axis=1))))


def independent_reduce(
    positions: Iterable[Iterable[float]],
    cell: Iterable[Iterable[float]],
    atomic_numbers: Iterable[int],
    *,
    epsilon: float,
) -> IndependentRow:
    """Rebuild one cache row without calling the production builder transform."""
    source_lattice = np.array(cell, dtype=np.float64, copy=True)
    source_cartesian = np.array(positions, dtype=np.float64, copy=True)
    numbers = np.array(atomic_numbers, dtype=np.int64, copy=True)
    if source_lattice.shape != (3, 3):
        raise ValueError("auditor found an invalid source lattice shape")
    if source_cartesian.shape != (numbers.size, 3) or not 1 <= numbers.size <= 20:
        raise ValueError("auditor found invalid source coordinates")
    if not np.isfinite(source_lattice).all() or not np.isfinite(source_cartesian).all():
        raise ValueError("auditor found a nonfinite source structure")
    if np.any(numbers < 1) or np.any(numbers > 118):
        raise ValueError("auditor found an invalid atomic number")
    if float(np.linalg.det(source_lattice)) <= 0.0:
        raise ValueError("auditor found a nonpositive source volume")
    reduced_lattice = spglib.niggli_reduce(source_lattice, eps=epsilon)
    if reduced_lattice is None:
        raise ValueError("independent Niggli reduction failed")
    reduced_lattice = np.array(reduced_lattice, dtype=np.float64, copy=True)
    # Solve R = B L instead of multiplying by an explicitly formed inverse.
    transform_real = np.linalg.solve(source_lattice.T, reduced_lattice.T).T
    transform = np.rint(transform_real).astype(np.int64)
    if np.max(np.abs(transform_real - transform)) > 1e-8:
        raise ValueError("independent Niggli certificate is nonintegral")
    if abs(int(round(float(np.linalg.det(transform))))) != 1:
        raise ValueError("independent Niggli certificate is not unimodular")
    if np.max(np.abs(reduced_lattice - transform @ source_lattice)) > 1e-10:
        raise ValueError("independent Niggli certificate does not reconstruct the cell")
    if float(np.linalg.det(reduced_lattice)) <= 0.0:
        raise ValueError("independent reduced lattice is not right handed")
    # Independently verify the direct reduced-cell solve, but serialize the
    # protocol-defined expression f_red = f_source B^-1 exactly as frozen.
    fractional_direct = np.linalg.solve(
        reduced_lattice.T, source_cartesian.T
    ).T
    source_fractional = source_cartesian @ np.linalg.inv(source_lattice)
    transformed_fractional = source_fractional @ np.linalg.inv(transform)
    relation = fractional_direct - transformed_fractional
    relation -= np.rint(relation)
    if np.max(np.abs(relation)) > 1e-10:
        raise ValueError("independent coordinate basis relation failed")
    fractional = np.mod(transformed_fractional, 1.0)
    cache_fractional = np.remainder(
        np.asarray(fractional, dtype=np.float32), np.float32(1.0)
    )
    cache_lattice = np.asarray(reduced_lattice, dtype=np.float32)
    source_error = _periodic_error(source_cartesian, fractional, reduced_lattice)
    cache_error = _periodic_error(
        source_cartesian,
        cache_fractional.astype(np.float64),
        cache_lattice.astype(np.float64),
    )
    if np.any(transform < np.iinfo(np.int16).min) or np.any(
        transform > np.iinfo(np.int16).max
    ):
        raise ValueError("independent transform exceeds int16")
    return IndependentRow(
        tokens=np.asarray(numbers - 1, dtype=np.uint8),
        fractional=cache_fractional,
        lattice=cache_lattice,
        transform=np.asarray(transform, dtype=np.int16),
        source_error=source_error,
        cache_error=cache_error,
    )


def _source_counts(protocol: dict[str, Any]) -> dict[str, int]:
    return {
        str(record["source_split"]): int(record["rows"])
        for record in protocol["source"]["raw_files"]
    }


def _source_paths(protocol: dict[str, Any], data_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for record in protocol["source"]["raw_files"]:
        split = str(record["source_split"])
        path = data_root / str(record["path"])
        _verify_hash(path, str(record["sha256"]), f"raw {split}")
        output[split] = path
    return output


def _load_assignment_map(
    protocol: dict[str, Any], data_root: Path
) -> dict[str, tuple[str, str, int]]:
    source = protocol["source"]
    path = data_root / str(source["assignment_path"])
    _verify_hash(path, str(source["assignment_sha256"]), "H0-A assignments")
    table = pq.read_table(
        path,
        columns=["material_id", "source_split", "gaugeflow_split", "primitive_sites"],
        memory_map=True,
    )
    values = table.to_pydict()
    output: dict[str, tuple[str, str, int]] = {}
    for material_id, source_split, target_split, sites in zip(
        values["material_id"],
        values["source_split"],
        values["gaugeflow_split"],
        values["primitive_sites"],
        strict=True,
    ):
        key = str(material_id)
        if key in output:
            raise ValueError("independent assignment join found a duplicate ID")
        output[key] = (str(source_split), str(target_split), int(sites))
    return output


def _load_payloads(
    output_root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, torch.Tensor]]:
    payloads: dict[str, dict[str, torch.Tensor]] = {}
    for split in TARGET_SPLITS:
        split_manifest = manifest["splits"][split]
        tensor_path = output_root / str(split_manifest["tensor_file"])
        index_path = output_root / str(split_manifest["index_file"])
        _verify_hash(tensor_path, str(split_manifest["tensor_sha256"]), f"{split} tensor")
        _verify_hash(index_path, str(split_manifest["index_sha256"]), f"{split} index")
        payload = torch.load(
            tensor_path, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(payload, dict) or payload.get("schema") != PACKED_ALEX_P1_SCHEMA:
            raise ValueError(f"invalid {split} packed payload")
        payloads[split] = payload
    return payloads


def _build_source_index(
    protocol: dict[str, Any],
    output_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    counts = _source_counts(protocol)
    mappings: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in counts.items():
        material_ids = np.empty(rows, dtype=object)
        material_ids[:] = None
        mappings[split] = {
            "target_code": np.full(rows, 255, dtype=np.uint8),
            "cache_row": np.full(rows, -1, dtype=np.int64),
            "node_start": np.full(rows, -1, dtype=np.int64),
            "node_stop": np.full(rows, -1, dtype=np.int64),
            "material_id": material_ids,
        }
    target_codes = {name: code for code, name in enumerate(TARGET_SPLITS)}
    for target_split in TARGET_SPLITS:
        index_path = output_root / str(manifest["splits"][target_split]["index_file"])
        table = pq.read_table(index_path, memory_map=True)
        values = table.to_pydict()
        expected_cache_rows = list(range(table.num_rows))
        if values["cache_row"] != expected_cache_rows:
            raise ValueError(f"{target_split} index is not in cache-row order")
        for material_id, source_split, source_row, gaugeflow_split, cache_row, node_start, node_stop in zip(
            values["material_id"],
            values["source_split"],
            values["source_row"],
            values["gaugeflow_split"],
            values["cache_row"],
            values["node_start"],
            values["node_stop"],
            strict=True,
        ):
            source_split = str(source_split)
            source_row = int(source_row)
            if gaugeflow_split != target_split:
                raise ValueError("index target split column mismatch")
            mapping = mappings[source_split]
            if source_row < 0 or source_row >= mapping["cache_row"].size:
                raise ValueError("index source row is outside its raw split")
            if mapping["cache_row"][source_row] != -1:
                raise ValueError("source row appears more than once in cache indices")
            mapping["target_code"][source_row] = target_codes[target_split]
            mapping["cache_row"][source_row] = int(cache_row)
            mapping["node_start"][source_row] = int(node_start)
            mapping["node_stop"][source_row] = int(node_stop)
            mapping["material_id"][source_row] = str(material_id)
    for source_split, mapping in mappings.items():
        if np.any(mapping["cache_row"] < 0):
            raise ValueError(f"{source_split} has unindexed raw rows")
        if any(value is None for value in mapping["material_id"]):
            raise ValueError(f"{source_split} has missing material IDs")
    return mappings


def _write_report(
    path: Path, audit: dict[str, Any], manifest: dict[str, Any]
) -> None:
    decision = "qualified" if audit["qualified"] else "failed_stop_before_H1a"
    source_error = audit["maximum_source_cartesian_equivalence_error_angstrom"]
    cache_error = audit["maximum_float32_cache_cartesian_equivalence_error_angstrom"]
    split_rows = " / ".join(
        str(manifest["splits"][split]["rows"]) for split in TARGET_SPLITS
    )
    split_nodes = " / ".join(
        str(manifest["splits"][split]["nodes"]) for split in TARGET_SPLITS
    )
    lines = [
        "# Current H1a packed structure cache",
        "",
        f"**Decision: `{decision}`.**",
        "",
        "This is a data-plane result only. It does not train or qualify the generator.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Source rows | {audit['source_rows']} |",
        f"| Rebuilt rows | {audit['full_rebuild_matches']} |",
        f"| Processing failures | {audit['processing_failures']} |",
        f"| Maximum source equivalence error (A) | {source_error:.6g} |",
        f"| Maximum float32 cache equivalence error (A) | {cache_error:.6g} |",
        f"| Train/val/test rows | {split_rows} |",
        f"| Train/val/test nodes | {split_nodes} |",
        f"| Independent audit wall seconds | {audit['wall_seconds']:.2f} |",
        "",
        "A qualified result permits only freezing a separate H1a training protocol.",
        "H1b and H2--H6 remain unauthorized.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_cache(
    *,
    protocol_path: Path,
    data_root: Path,
    output_root: Path,
    report_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = load_json_object(protocol_path)
    if protocol.get("protocol") != PACKED_ALEX_P1_PROTOCOL:
        raise ValueError("auditor received the wrong protocol")
    manifest_path = output_root / "manifest.json"
    manifest = load_json_object(manifest_path)
    if manifest.get("protocol") != PACKED_ALEX_P1_PROTOCOL:
        raise ValueError("cache manifest protocol mismatch")
    if not bool(manifest.get("builder_qualified")) or bool(manifest.get("qualified")):
        raise ValueError("cache manifest is not awaiting independent audit")
    builder_manifest_sha256 = sha256_file(manifest_path)
    _verify_hash(protocol_path, str(manifest["protocol_sha256"]), "cache protocol")
    source_paths = _source_paths(protocol, data_root)
    assignments = _load_assignment_map(protocol, data_root)
    payloads = _load_payloads(output_root, manifest)
    source_index = _build_source_index(protocol, output_root, manifest)
    epsilon = float(protocol["canonical_cell"]["epsilon"])
    source_rows = int(protocol["thresholds"]["source_rows"])
    rebuilt = 0
    processing_failures = 0
    max_source_error = 0.0
    max_cache_error = 0.0
    for source_split in REVERSE_SOURCE_SPLITS:
        table = pq.read_table(
            source_paths[source_split],
            columns=["positions", "cell", "atomic_numbers", "material_id"],
            memory_map=True,
            use_threads=True,
        )
        mapping = source_index[source_split]
        for stop in range(table.num_rows, 0, -batch_size):
            start = max(0, stop - batch_size)
            rows = table.slice(start, stop - start).to_pylist()
            for local_index in range(len(rows) - 1, -1, -1):
                source_row = start + local_index
                row = rows[local_index]
                material_id = str(row["material_id"])
                try:
                    expected_id = str(mapping["material_id"][source_row])
                    if expected_id != material_id:
                        raise ValueError("raw/index material ID mismatch")
                    assignment = assignments.get(material_id)
                    if assignment is None:
                        raise ValueError("raw material ID is absent from H0-A assignments")
                    target_code = int(mapping["target_code"][source_row])
                    target_split = TARGET_SPLITS[target_code]
                    if assignment != (
                        source_split,
                        target_split,
                        len(row["atomic_numbers"]),
                    ):
                        raise ValueError("independent H0-A join mismatch")
                    rebuilt_row = independent_reduce(
                        row["positions"],
                        row["cell"],
                        row["atomic_numbers"],
                        epsilon=epsilon,
                    )
                    cache_row = int(mapping["cache_row"][source_row])
                    node_start = int(mapping["node_start"][source_row])
                    node_stop = int(mapping["node_stop"][source_row])
                    payload = payloads[target_split]
                    if node_stop - node_start != rebuilt_row.tokens.size:
                        raise ValueError("cache index node interval mismatch")
                    if int(payload["offsets"][cache_row]) != node_start or int(
                        payload["offsets"][cache_row + 1]
                    ) != node_stop:
                        raise ValueError("packed offsets disagree with index")
                    if not np.array_equal(
                        payload["atom_tokens"][node_start:node_stop].numpy(),
                        rebuilt_row.tokens,
                    ):
                        raise ValueError("independent atom-token rebuild mismatch")
                    if not np.array_equal(
                        payload["fractional_coordinates"][node_start:node_stop].numpy(),
                        rebuilt_row.fractional,
                    ):
                        raise ValueError("independent fractional-coordinate rebuild mismatch")
                    if not np.array_equal(
                        payload["lattice"][cache_row].numpy(), rebuilt_row.lattice
                    ):
                        raise ValueError("independent lattice rebuild mismatch")
                    if not np.array_equal(
                        payload["niggli_transform"][cache_row].numpy(),
                        rebuilt_row.transform,
                    ):
                        raise ValueError("independent Niggli-transform rebuild mismatch")
                    max_source_error = max(max_source_error, rebuilt_row.source_error)
                    max_cache_error = max(max_cache_error, rebuilt_row.cache_error)
                    rebuilt += 1
                except Exception:
                    processing_failures += 1
                    raise
            if rebuilt % 50000 < len(rows):
                print(f"independently rebuilt {rebuilt}/{source_rows} rows", flush=True)
    thresholds = protocol["thresholds"]
    checks = {
        "source_rows_exact": rebuilt == source_rows,
        "processing_failures_zero": processing_failures == 0,
        "full_rebuild_match_fraction_one": rebuilt == source_rows,
        "source_equivalence_within_threshold": max_source_error
        <= float(thresholds["source_cartesian_equivalence_max_error_angstrom"]),
        "cache_equivalence_within_threshold": max_cache_error
        <= float(thresholds["float32_cache_cartesian_equivalence_max_error_angstrom"]),
        "split_counts_exact": {
            split: int(manifest["splits"][split]["rows"])
            for split in TARGET_SPLITS
        }
        == protocol["selection"]["gaugeflow_split_counts"],
        "node_counts_exact": {
            split: int(manifest["splits"][split]["nodes"])
            for split in TARGET_SPLITS
        }
        == protocol["selection"]["gaugeflow_split_node_counts"],
    }
    qualified = all(checks.values())
    audit: dict[str, Any] = {
        "schema": 1,
        "protocol": "h1a_p1_structure_cache_independent_audit",
        "cache_protocol": PACKED_ALEX_P1_PROTOCOL,
        "builder_manifest_sha256": builder_manifest_sha256,
        "source_rows": source_rows,
        "full_rebuild_matches": rebuilt,
        "processing_failures": processing_failures,
        "maximum_source_cartesian_equivalence_error_angstrom": max_source_error,
        "maximum_float32_cache_cartesian_equivalence_error_angstrom": max_cache_error,
        "checks": checks,
        "qualified": qualified,
        "decision": (
            "cache_qualified_only_freeze_H1a_training_protocol"
            if qualified
            else "cache_failed_stop_H1a_and_later_gates"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    audit_path = output_root / "independent_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["independent_audit_file"] = audit_path.name
    manifest["independent_audit_sha256"] = sha256_file(audit_path)
    manifest["qualified"] = qualified
    manifest["status"] = "qualified" if qualified else "failed"
    manifest["decision"] = audit["decision"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(report_path, audit, manifest)
    if not qualified:
        raise RuntimeError("H1a packed cache failed independent qualification")
    return audit


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repo_root / "configs/gates/h1a_p1_structure_cache_v1.json",
    )
    parser.add_argument("--data-root", type=Path, default=Path("E:/DATA/T2C-Flow"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports/h1a_p1_structure_cache_v1/README.md",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error("batch-size must be positive")
    protocol = load_json_object(arguments.protocol)
    output_root = arguments.output_root or (
        arguments.data_root / protocol["required_outputs"]["root"]
    )
    result = audit_cache(
        protocol_path=arguments.protocol.resolve(),
        data_root=arguments.data_root.resolve(),
        output_root=output_root.resolve(),
        report_path=arguments.report.resolve(),
        batch_size=arguments.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
