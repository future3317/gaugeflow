"""Build a resumable, batched TensorNet per-atom feature cache for Stage-B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from gaugeflow.file_utils import load_json_object
from gaugeflow.production.teacher_feature_cache import write_matpes_teacher_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--functional", default="PBE")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--graphs-per-batch", type=int, default=64)
    parser.add_argument("--nodes-per-batch", type=int, default=1024)
    parser.add_argument("--rows-per-shard", type=int, default=8192)
    parser.add_argument("--maximum-rows", type=int)
    return parser.parse_args()


def _load_index(root: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    manifest = load_json_object(root / "manifest.json")
    if not bool(manifest.get("qualified")):
        raise ValueError("teacher cache requires the qualified MatPES index")
    payload: Any = torch.load(
        root / str(manifest["index_file"]),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(name), torch.Tensor)
        for name in ("source_index", "byte_offset", "node_count")
    ):
        raise ValueError("MatPES index tensor is incomplete")
    return manifest, payload


class _SourceReader:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.handles: dict[int, Any] = {}

    def row(self, source: int, offset: int) -> dict[str, Any]:
        if source not in self.handles:
            self.handles[source] = self.paths[source].open("rb")
        handle = self.handles[source]
        handle.seek(offset)
        raw = handle.readline()
        if not raw:
            raise ValueError("indexed MatPES source row is unreadable")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("MatPES source row is not an object")
        return value


def _extract_shard(
    start: int,
    stop: int,
    *,
    payload: dict[str, torch.Tensor],
    source_functionals: list[str],
    reader: _SourceReader,
    functional: str,
    potential: Any,
    converter: Any,
    feature_dim: int,
    graphs_per_batch: int,
    nodes_per_batch: int,
) -> dict[str, Any]:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from torch_geometric.data import Batch

    row_node_counts = torch.zeros(stop - start, dtype=torch.int64)
    chunks: list[torch.Tensor] = []
    pending: list[tuple[int, Any, torch.Tensor, torch.Tensor, int]] = []
    pending_nodes = 0

    def flush() -> None:
        nonlocal pending_nodes
        if not pending:
            return
        graphs = Batch.from_data_list([item[1] for item in pending])
        lattice = torch.cat([item[2] for item in pending], dim=0)
        state = torch.stack([item[3] for item in pending])
        with torch.no_grad():
            potential(graphs, lattice, state)
        features = potential.model.feature_dict.get("readout")
        if (
            features is None
            or features.shape != (pending_nodes, feature_dim)
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("TensorNet returned invalid per-atom readout features")
        cursor = 0
        for local_row, _graph, _lattice, _state, nodes in pending:
            value = features[cursor : cursor + nodes].detach().cpu().to(torch.float16)
            chunks.append(value)
            row_node_counts[local_row] = nodes
            cursor += nodes
        pending.clear()
        pending_nodes = 0

    for row in range(start, stop):
        source = int(payload["source_index"][row])
        if source_functionals[source] != functional:
            continue
        source_row = reader.row(source, int(payload["byte_offset"][row]))
        nodes = int(payload["node_count"][row])
        if source_row.get("functional") != functional or int(source_row.get("nsites", 0)) != nodes:
            raise ValueError("teacher feature source disagrees with qualified index")
        if pending and (
            len(pending) >= graphs_per_batch or pending_nodes + nodes > nodes_per_batch
        ):
            flush()
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(source_row["structure"]))
        graph, lattice, state = converter.get_graph(atoms)
        pending.append(
            (row - start, graph, lattice, torch.as_tensor(state), nodes)
        )
        pending_nodes += nodes
    flush()
    features = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty((0, feature_dim), dtype=torch.float16)
    )
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), row_node_counts.cumsum(dim=0))
    )
    if features.shape[0] != int(offsets[-1]):
        raise AssertionError("teacher feature shard offsets do not close")
    return {
        "schema": 1,
        "start": start,
        "stop": stop,
        "node_offsets": offsets,
        "features": features,
    }


def _iter_completed_rows(
    parts: list[Path], *, row_count: int, feature_dim: int
) -> Iterator[tuple[int, torch.Tensor | None]]:
    next_row = 0
    for path in parts:
        shard: Any = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(shard, dict)
            or shard.get("schema") != 1
            or shard.get("start") != next_row
        ):
            raise ValueError("teacher feature shard sequence is invalid")
        stop = int(shard["stop"])
        offsets = shard["node_offsets"]
        features = shard["features"]
        if offsets.shape != (stop - next_row + 1,) or features.shape[1:] != (feature_dim,):
            raise ValueError("teacher feature shard dimensions are invalid")
        for local_row in range(stop - next_row):
            start_node = int(offsets[local_row])
            stop_node = int(offsets[local_row + 1])
            yield next_row + local_row, (
                features[start_node:stop_node].float()
                if stop_node > start_node
                else None
            )
        next_row = stop
    if next_row != row_count:
        raise ValueError("teacher feature shards do not cover the requested rows")


def main() -> None:
    arguments = parse_args()
    if arguments.feature_dim < 1 or arguments.graphs_per_batch < 1 or arguments.nodes_per_batch < 1:
        raise ValueError("teacher feature batch dimensions must be positive")
    if arguments.rows_per_shard < 1:
        raise ValueError("teacher feature shard size must be positive")
    manifest, payload = _load_index(arguments.index)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("MatPES index sources are invalid")
    source_paths = [Path(str(source["path"])) for source in sources]
    source_functionals = [str(source["functional"]) for source in sources]
    row_count = int(payload["byte_offset"].numel())
    bounded = arguments.maximum_rows is not None
    if arguments.maximum_rows is not None:
        if arguments.maximum_rows < 1:
            raise ValueError("maximum rows must be positive")
        row_count = min(row_count, arguments.maximum_rows)
    expected_feature_rows = sum(
        source_functionals[int(payload["source_index"][row])] == arguments.functional
        for row in range(row_count)
    )
    if expected_feature_rows < 1:
        raise ValueError("requested teacher functional has no indexed rows")

    import matgl
    from matgl.ext.ase import Atoms2Graph

    potential = matgl.load_model(arguments.teacher_model).to(arguments.device).eval()
    potential.calc_forces = False
    potential.calc_stresses = False
    potential.calc_hessian = False
    if potential.model.feature_dict is None:
        raise ValueError("TensorNet teacher does not expose a feature dictionary")
    converter = Atoms2Graph(
        element_types=tuple(potential.model.element_types),
        cutoff=float(potential.model.cutoff),
    )
    arguments.work.mkdir(parents=True, exist_ok=True)
    reader = _SourceReader(source_paths)
    parts: list[Path] = []
    for start in range(0, row_count, arguments.rows_per_shard):
        stop = min(start + arguments.rows_per_shard, row_count)
        path = arguments.work / f"rows_{start:09d}_{stop:09d}.pt"
        if path.exists():
            shard: Any = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(shard, dict) or shard.get("start") != start or shard.get("stop") != stop:
                raise ValueError(f"existing teacher shard is invalid: {path}")
        else:
            shard = _extract_shard(
                start,
                stop,
                payload=payload,
                source_functionals=source_functionals,
                reader=reader,
                functional=arguments.functional,
                potential=potential,
                converter=converter,
                feature_dim=arguments.feature_dim,
                graphs_per_batch=arguments.graphs_per_batch,
                nodes_per_batch=arguments.nodes_per_batch,
            )
            torch.save(shard, path)
        parts.append(path)
        print(json.dumps({"completed_rows": stop, "total_rows": row_count}), flush=True)
    write_matpes_teacher_feature_cache(
        arguments.output,
        _iter_completed_rows(parts, row_count=row_count, feature_dim=arguments.feature_dim),
        row_count=row_count,
        feature_dim=arguments.feature_dim,
        index_manifest=arguments.index / "manifest.json",
        teacher_manifest=arguments.teacher_manifest,
        functional_scope=(arguments.functional,),
        expected_feature_rows=expected_feature_rows,
        bounded_smoke=bounded,
    )


if __name__ == "__main__":
    main()
