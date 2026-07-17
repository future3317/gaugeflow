"""Build the leakage-safe packed Alex-MP-20 cache for the current H1a data gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import spglib
import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import (
    PACKED_ALEX_P1_PROTOCOL,
    PACKED_ALEX_P1_SCHEMA,
)

SOURCE_SPLITS = ("train", "val", "test")
TARGET_SPLITS = ("train", "val", "test")
H0_O1_ROOT = Path("processed/gaugeflow_h0_v8/occupational_order_o1_v1")


@dataclass(frozen=True)
class Assignment:
    source_split: str
    gaugeflow_split: str
    primitive_sites: int


@dataclass(frozen=True)
class ReducedRow:
    tokens: np.ndarray
    fractional: np.ndarray
    lattice: np.ndarray
    transform: np.ndarray
    source_equivalence_error: float
    cache_equivalence_error: float


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _provenance_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != PACKED_ALEX_P1_PROTOCOL:
        raise ValueError("unexpected H1a cache protocol")
    if protocol.get("status_before_run") != "frozen_not_run":
        raise ValueError("H1a cache protocol is not frozen before run")
    return protocol


def _verify_file(path: Path, expected_hash: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected_hash}")


def _verify_inputs(
    protocol: dict[str, Any], data_root: Path, repo_root: Path
) -> dict[str, Path]:
    source = protocol["source"]
    paths: dict[str, Path] = {}
    for record in source["raw_files"]:
        source_split = str(record["source_split"])
        path = data_root / str(record["path"])
        _verify_file(path, str(record["sha256"]), label=f"Alex {source_split}")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != int(record["rows"]):
            raise ValueError(f"Alex {source_split} row count mismatch")
        paths[f"raw_{source_split}"] = path
    for key, hash_key in (
        ("assignment_path", "assignment_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("split_audit_path", "split_audit_sha256"),
    ):
        path = data_root / str(source[key])
        _verify_file(path, str(source[hash_key]), label=key)
        paths[key] = path
    split_audit = json.loads(paths["split_audit_path"].read_text(encoding="utf-8"))
    if not bool(split_audit.get("qualified")):
        raise ValueError("H0-A split audit is not qualified")
    expected_split_counts = protocol["selection"]["gaugeflow_split_counts"]
    if split_audit.get("split_counts") != expected_split_counts:
        raise ValueError("H0-A split counts do not match the H1a protocol")
    o1_root = data_root / H0_O1_ROOT
    o1_manifest = o1_root / "manifest.json"
    o1_audit = o1_root / "independent_audit.json"
    dependency = protocol["h0_dependency"]
    _verify_file(
        o1_manifest,
        str(dependency["o1_manifest_sha256"]),
        label="H0 O1 manifest",
    )
    _verify_file(
        o1_audit,
        str(dependency["o1_independent_audit_sha256"]),
        label="H0 O1 independent audit",
    )
    tags = subprocess.run(
        ["git", "tag", "--list", str(dependency["archive_tag"])],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if tags != [str(dependency["archive_tag"])]:
        raise ValueError("required H0 archive tag is absent")
    return paths


def _load_assignments(path: Path, expected_rows: int) -> dict[str, Assignment]:
    table = pq.read_table(
        path,
        columns=["material_id", "source_split", "gaugeflow_split", "primitive_sites"],
        memory_map=True,
    )
    if table.num_rows != expected_rows:
        raise ValueError("assignment row count mismatch")
    columns = table.to_pydict()
    assignments: dict[str, Assignment] = {}
    for material_id, source_split, gaugeflow_split, primitive_sites in zip(
        columns["material_id"],
        columns["source_split"],
        columns["gaugeflow_split"],
        columns["primitive_sites"],
        strict=True,
    ):
        key = str(material_id)
        if key in assignments:
            raise ValueError(f"duplicate assignment material ID: {key}")
        if source_split not in SOURCE_SPLITS or gaugeflow_split not in TARGET_SPLITS:
            raise ValueError("assignment contains an invalid split")
        assignments[key] = Assignment(
            str(source_split), str(gaugeflow_split), int(primitive_sites)
        )
    return assignments


def _periodic_cartesian_error(
    reference_cartesian: np.ndarray,
    fractional: np.ndarray,
    lattice: np.ndarray,
) -> float:
    delta_cartesian = fractional @ lattice - reference_cartesian
    delta_fractional = delta_cartesian @ np.linalg.inv(lattice)
    delta_fractional -= np.rint(delta_fractional)
    return float(np.max(np.linalg.norm(delta_fractional @ lattice, axis=1)))


def reduce_alex_row(
    positions: Iterable[Iterable[float]],
    cell: Iterable[Iterable[float]],
    atomic_numbers: Iterable[int],
    *,
    epsilon: float = 1e-5,
) -> ReducedRow:
    """Apply one certified Niggli GL(3,Z) basis change to an Alex row."""
    cartesian = np.asarray(positions, dtype=np.float64)
    lattice = np.asarray(cell, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    nodes = numbers.size
    if lattice.shape != (3, 3) or cartesian.shape != (nodes, 3) or nodes < 1:
        raise ValueError("invalid Alex structure shapes")
    if nodes > 20 or not np.isfinite(lattice).all() or not np.isfinite(cartesian).all():
        raise ValueError("Alex structure lies outside the P1 finite 1--20 atom domain")
    if np.any(numbers < 1) or np.any(numbers > 118):
        raise ValueError("Alex atomic number lies outside 1..118")
    if float(np.linalg.det(lattice)) <= 0.0:
        raise ValueError("Alex source lattice is not right handed")
    reduced = spglib.niggli_reduce(lattice, eps=epsilon)
    if reduced is None:
        raise ValueError("spglib Niggli reduction failed")
    reduced = np.asarray(reduced, dtype=np.float64)
    transform_float = reduced @ np.linalg.inv(lattice)
    transform = np.rint(transform_float).astype(np.int64)
    if not np.allclose(transform_float, transform, atol=1e-8, rtol=0.0):
        raise ValueError("Niggli basis certificate is not integral")
    determinant = int(round(float(np.linalg.det(transform))))
    if abs(determinant) != 1:
        raise ValueError("Niggli basis certificate is not unimodular")
    if not np.allclose(reduced, transform @ lattice, atol=1e-10, rtol=1e-10):
        raise ValueError("Niggli lattice does not equal B L_source")
    if float(np.linalg.det(reduced)) <= 0.0:
        raise ValueError("Niggli lattice is not right handed")
    fractional_source = cartesian @ np.linalg.inv(lattice)
    unwrapped = fractional_source @ np.linalg.inv(transform)
    fractional = np.mod(unwrapped, 1.0)
    source_error = _periodic_cartesian_error(cartesian, fractional, reduced)
    cache_fractional = np.remainder(
        fractional.astype(np.float32), np.float32(1.0)
    )
    cache_lattice = reduced.astype(np.float32)
    cache_error = _periodic_cartesian_error(
        cartesian,
        cache_fractional.astype(np.float64),
        cache_lattice.astype(np.float64),
    )
    if np.any(transform < np.iinfo(np.int16).min) or np.any(
        transform > np.iinfo(np.int16).max
    ):
        raise ValueError("Niggli transform exceeds the packed int16 schema")
    return ReducedRow(
        tokens=(numbers - 1).astype(np.uint8),
        fractional=cache_fractional,
        lattice=cache_lattice,
        transform=transform.astype(np.int16),
        source_equivalence_error=source_error,
        cache_equivalence_error=cache_error,
    )


def _reduce_chunk(
    rows: list[dict[str, object]], epsilon: float
) -> list[tuple[str, ReducedRow]]:
    output: list[tuple[str, ReducedRow]] = []
    for row in rows:
        output.append(
            (
                str(row["material_id"]),
                reduce_alex_row(
                    row["positions"],  # type: ignore[arg-type]
                    row["cell"],  # type: ignore[arg-type]
                    row["atomic_numbers"],  # type: ignore[arg-type]
                    epsilon=epsilon,
                ),
            )
        )
    return output


def _chunk_rows(rows: list[dict[str, object]], chunks: int) -> list[list[dict[str, object]]]:
    if chunks <= 1 or len(rows) <= 1:
        return [rows]
    step = (len(rows) + chunks - 1) // chunks
    return [rows[start : start + step] for start in range(0, len(rows), step)]


def _allocate_split(rows: int, nodes: int) -> dict[str, torch.Tensor]:
    return {
        "atom_tokens": torch.empty(nodes, dtype=torch.uint8),
        "fractional_coordinates": torch.empty((nodes, 3), dtype=torch.float32),
        "lattice": torch.empty((rows, 3, 3), dtype=torch.float32),
        "offsets": torch.empty(rows + 1, dtype=torch.int64),
        "niggli_transform": torch.empty((rows, 3, 3), dtype=torch.int16),
    }


def _write_split(
    root: Path,
    split: str,
    payload: dict[str, torch.Tensor],
    index_columns: dict[str, list[object]],
) -> dict[str, object]:
    tensor_path = root / f"{split}.pt"
    index_path = root / f"{split}_index.parquet"
    torch.save({"schema": PACKED_ALEX_P1_SCHEMA, **payload}, tensor_path)
    pq.write_table(
        pa.table(index_columns),
        index_path,
        compression="zstd",
        use_dictionary=["source_split", "gaugeflow_split"],
    )
    return {
        "rows": int(payload["lattice"].shape[0]),
        "nodes": int(payload["atom_tokens"].shape[0]),
        "tensor_file": tensor_path.name,
        "tensor_bytes": tensor_path.stat().st_size,
        "tensor_sha256": sha256_file(tensor_path),
        "index_file": index_path.name,
        "index_bytes": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
    }


def build_cache(
    *,
    protocol_path: Path,
    data_root: Path,
    output_root: Path,
    workers: int,
    batch_size: int,
) -> dict[str, object]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    protocol = _load_protocol(protocol_path)
    paths = _verify_inputs(protocol, data_root, repo_root)
    expected_rows = int(protocol["thresholds"]["source_rows"])
    assignments = _load_assignments(paths["assignment_path"], expected_rows)
    split_counts = {k: int(v) for k, v in protocol["selection"]["gaugeflow_split_counts"].items()}
    node_counts = {
        k: int(v)
        for k, v in protocol["selection"]["gaugeflow_split_node_counts"].items()
    }
    payloads = {
        split: _allocate_split(split_counts[split], node_counts[split])
        for split in TARGET_SPLITS
    }
    indices: dict[str, dict[str, list[object]]] = {
        split: {
            "material_id": [],
            "source_split": [],
            "source_row": [],
            "gaugeflow_split": [],
            "cache_row": [],
            "node_start": [],
            "node_stop": [],
        }
        for split in TARGET_SPLITS
    }
    row_cursor = {split: 0 for split in TARGET_SPLITS}
    node_cursor = {split: 0 for split in TARGET_SPLITS}
    max_source_error = 0.0
    max_cache_error = 0.0
    processed = 0
    epsilon = float(protocol["canonical_cell"]["epsilon"])
    temporary_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or temporary_root.exists():
        raise FileExistsError(
            "H1a cache output already exists; formal builds never overwrite artifacts"
        )
    temporary_root.mkdir(parents=True)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for source_split in SOURCE_SPLITS:
            source_path = paths[f"raw_{source_split}"]
            source_row = 0
            parquet = pq.ParquetFile(source_path, memory_map=True)
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=["positions", "cell", "atomic_numbers", "material_id"],
                use_threads=True,
            ):
                rows = batch.to_pylist()
                chunks = _chunk_rows(rows, workers)
                if executor is None:
                    reduced_chunks = [_reduce_chunk(chunks[0], epsilon)]
                else:
                    reduced_chunks = list(
                        executor.map(
                            _reduce_chunk,
                            chunks,
                            [epsilon] * len(chunks),
                        )
                    )
                reduced_rows = [item for chunk in reduced_chunks for item in chunk]
                if len(reduced_rows) != len(rows):
                    raise RuntimeError("parallel Niggli transform lost rows")
                for material_id, reduced in reduced_rows:
                    assignment = assignments.pop(material_id, None)
                    if assignment is None:
                        raise ValueError(f"missing or duplicate assignment for {material_id}")
                    if assignment.source_split != source_split:
                        raise ValueError(f"source split mismatch for {material_id}")
                    if assignment.primitive_sites != reduced.tokens.size:
                        raise ValueError(f"node-count assignment mismatch for {material_id}")
                    split = assignment.gaugeflow_split
                    cache_row = row_cursor[split]
                    node_start = node_cursor[split]
                    node_stop = node_start + reduced.tokens.size
                    payload = payloads[split]
                    payload["atom_tokens"][node_start:node_stop] = torch.from_numpy(
                        reduced.tokens
                    )
                    payload["fractional_coordinates"][node_start:node_stop] = (
                        torch.from_numpy(reduced.fractional)
                    )
                    payload["lattice"][cache_row] = torch.from_numpy(reduced.lattice)
                    payload["niggli_transform"][cache_row] = torch.from_numpy(
                        reduced.transform
                    )
                    payload["offsets"][cache_row] = node_start
                    index = indices[split]
                    index["material_id"].append(material_id)
                    index["source_split"].append(source_split)
                    index["source_row"].append(source_row)
                    index["gaugeflow_split"].append(split)
                    index["cache_row"].append(cache_row)
                    index["node_start"].append(node_start)
                    index["node_stop"].append(node_stop)
                    row_cursor[split] += 1
                    node_cursor[split] = node_stop
                    source_row += 1
                    processed += 1
                    max_source_error = max(
                        max_source_error, reduced.source_equivalence_error
                    )
                    max_cache_error = max(max_cache_error, reduced.cache_equivalence_error)
                if processed % 50000 < len(rows):
                    print(f"processed {processed}/{expected_rows} rows", flush=True)
        if assignments:
            raise ValueError(f"{len(assignments)} assignment rows were not consumed")
        if processed != expected_rows:
            raise ValueError("source row count does not match the frozen protocol")
        for split in TARGET_SPLITS:
            if row_cursor[split] != split_counts[split]:
                raise ValueError(f"{split} row count mismatch")
            if node_cursor[split] != node_counts[split]:
                raise ValueError(f"{split} node count mismatch")
            payloads[split]["offsets"][-1] = node_cursor[split]
        thresholds = protocol["thresholds"]
        if max_source_error > float(
            thresholds["source_cartesian_equivalence_max_error_angstrom"]
        ):
            raise ValueError("source Cartesian-equivalence threshold failed")
        if max_cache_error > float(
            thresholds["float32_cache_cartesian_equivalence_max_error_angstrom"]
        ):
            raise ValueError("float32 cache Cartesian-equivalence threshold failed")
        split_manifest = {
            split: _write_split(
                temporary_root, split, payloads[split], indices[split]
            )
            for split in TARGET_SPLITS
        }
        manifest: dict[str, object] = {
            "schema": PACKED_ALEX_P1_SCHEMA,
            "protocol": PACKED_ALEX_P1_PROTOCOL,
            "status": "awaiting_independent_audit",
            "builder_qualified": True,
            "qualified": False,
            "source_commit": _git_commit(repo_root),
            "protocol_file": _provenance_path(protocol_path, repo_root),
            "protocol_sha256": sha256_file(protocol_path),
            "data_root": str(data_root),
            "source_rows": processed,
            "maximum_source_cartesian_equivalence_error_angstrom": max_source_error,
            "maximum_float32_cache_cartesian_equivalence_error_angstrom": max_cache_error,
            "splits": split_manifest,
            "wall_seconds": time.perf_counter() - started,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.rename(output_root)
        return manifest
    except Exception:
        (temporary_root / "FAILED").write_text(
            "The formal build failed closed. Do not use this directory.\n",
            encoding="utf-8",
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    protocol = _load_protocol(arguments.protocol)
    output_root = arguments.output_root or (
        arguments.data_root / protocol["required_outputs"]["root"]
    )
    result = build_cache(
        protocol_path=arguments.protocol.resolve(),
        data_root=arguments.data_root.resolve(),
        output_root=output_root.resolve(),
        workers=arguments.workers,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
