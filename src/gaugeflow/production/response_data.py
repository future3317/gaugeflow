"""Hash-bound Stage-D response data and equivalent-view augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from gaugeflow.file_utils import sha256_file

from .response_multitask import ResponseTargets

_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class ResponseRecord:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    source_index: torch.Tensor
    targets: ResponseTargets


@dataclass(frozen=True)
class ResponseBatch:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor
    node_counts: torch.Tensor
    source_index: torch.Tensor
    targets: ResponseTargets

    @property
    def graph_count(self) -> int:
        return int(self.lattice.shape[0])

    def pin_memory(self) -> ResponseBatch:
        return _map_response_batch(self, lambda value: value.pin_memory())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ResponseBatch:
        return _map_response_batch(
            self,
            lambda value: value.to(device, non_blocking=non_blocking),
        )


def _map_targets(
    targets: ResponseTargets,
    operation: Any,
) -> ResponseTargets:
    return ResponseTargets(
        **{
            name: operation(getattr(targets, name))
            for name in ResponseTargets.__dataclass_fields__
        }
    )


def _map_response_batch(batch: ResponseBatch, operation: Any) -> ResponseBatch:
    return ResponseBatch(
        element_tokens=operation(batch.element_tokens),
        fractional_coordinates=operation(batch.fractional_coordinates),
        lattice=operation(batch.lattice),
        batch=operation(batch.batch),
        node_counts=operation(batch.node_counts),
        source_index=operation(batch.source_index),
        targets=_map_targets(batch.targets, operation),
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


class StageDResponseDataset(Dataset[ResponseRecord]):
    """Split-local reader for the immutable Stage-D packed tensor cache."""

    def __init__(self, root: Path, split: str) -> None:
        if split not in _SPLIT_INDEX:
            raise ValueError(f"unknown Stage-D split {split!r}")
        manifest_path = root / "MANIFEST.json"
        cache_path = root / "data.pt"
        manifest = _load_json(manifest_path)
        if (
            manifest.get("schema") != "gaugeflow.stage_d_jarvis_multitask.v1"
            or not bool(manifest.get("qualified"))
        ):
            raise ValueError("Stage-D response cache is not qualified")
        if sha256_file(cache_path) != manifest.get("cache_sha256"):
            raise ValueError("Stage-D response cache hash disagrees with its manifest")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != 1:
            raise ValueError("unexpected Stage-D response cache schema")
        self._manifest = manifest
        self.payload = payload
        split_index = self._tensor("split_index", torch.uint8)
        self.indices = torch.nonzero(split_index == _SPLIT_INDEX[split]).flatten()
        expected = int(manifest["selected_split_counts"][split])
        if self.indices.numel() != expected:
            raise ValueError("Stage-D split count disagrees with its manifest")
        offsets = self._tensor("node_offsets", torch.long)
        if offsets.shape != (split_index.numel() + 1,) or int(offsets[0]) != 0:
            raise ValueError("Stage-D node offsets are invalid")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise ValueError("Stage-D node offsets are not monotone")

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def _tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        value = self.payload.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise ValueError(f"Stage-D cache field {name!r} has the wrong type")
        return value

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> ResponseRecord:
        graph_index = int(self.indices[index])
        offsets = self._tensor("node_offsets", torch.long)
        start, stop = int(offsets[graph_index]), int(offsets[graph_index + 1])
        node_slice = slice(start, stop)
        graph_slice = slice(graph_index, graph_index + 1)
        targets = ResponseTargets(
            piezoelectric=self._tensor("piezoelectric", torch.float32)[graph_index],
            dielectric=self._tensor("dielectric", torch.float32)[graph_index],
            elastic=self._tensor("elastic", torch.float32)[graph_index],
            born_effective_charge=self._tensor("born_effective_charge", torch.float32)[
                node_slice
            ],
            gamma_soft=self._tensor("gamma_soft", torch.float32)[graph_index],
            gamma_log_magnitude=self._tensor("gamma_log_magnitude", torch.float32)[
                graph_index
            ],
            internal_strain=self._tensor("internal_strain", torch.float32)[node_slice],
            piezoelectric_mask=self._tensor("piezoelectric_mask", torch.bool)[graph_index],
            dielectric_mask=self._tensor("dielectric_mask", torch.bool)[graph_index],
            elastic_mask=self._tensor("elastic_mask", torch.bool)[graph_index],
            born_mask=self._tensor("born_mask", torch.bool)[node_slice],
            gamma_mask=self._tensor("gamma_mask", torch.bool)[graph_index],
            internal_strain_mask=self._tensor("internal_strain_mask", torch.bool)[
                node_slice
            ],
        )
        return ResponseRecord(
            element_tokens=self._tensor("element_tokens", torch.long)[node_slice],
            fractional_coordinates=self._tensor(
                "fractional_coordinates", torch.float32
            )[node_slice],
            lattice=self._tensor("lattice", torch.float32)[graph_index],
            source_index=self._tensor("source_index", torch.long)[graph_slice],
            targets=targets,
        )


def collate_response_records(records: Sequence[ResponseRecord]) -> ResponseBatch:
    if not records:
        raise ValueError("cannot collate an empty Stage-D response batch")
    node_counts = torch.tensor(
        [record.element_tokens.shape[0] for record in records], dtype=torch.long
    )
    if bool((node_counts < 1).any()):
        raise ValueError("Stage-D records must contain at least one atom")
    batch = torch.repeat_interleave(torch.arange(len(records)), node_counts)
    graph_target_names = (
        "piezoelectric",
        "dielectric",
        "elastic",
        "gamma_soft",
        "gamma_log_magnitude",
        "piezoelectric_mask",
        "dielectric_mask",
        "elastic_mask",
        "gamma_mask",
    )
    node_target_names = (
        "born_effective_charge",
        "internal_strain",
        "born_mask",
        "internal_strain_mask",
    )
    target_values = {
        name: torch.stack([getattr(record.targets, name) for record in records])
        for name in graph_target_names
    }
    target_values.update(
        {
            name: torch.cat([getattr(record.targets, name) for record in records])
            for name in node_target_names
        }
    )
    return ResponseBatch(
        element_tokens=torch.cat([record.element_tokens for record in records]),
        fractional_coordinates=torch.cat(
            [record.fractional_coordinates for record in records]
        ),
        lattice=torch.stack([record.lattice for record in records]),
        batch=batch,
        node_counts=node_counts,
        source_index=torch.cat([record.source_index for record in records]),
        targets=ResponseTargets(**target_values),
    )


def _random_orthogonal(
    graph_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    allow_improper: bool,
) -> torch.Tensor:
    raw = torch.randn(graph_count, 3, 3, dtype=dtype, device=device, generator=generator)
    orthogonal, _ = torch.linalg.qr(raw)
    if allow_improper:
        signs = torch.where(
            torch.rand(graph_count, device=device, generator=generator) < 0.5,
            -torch.ones(graph_count, device=device, dtype=dtype),
            torch.ones(graph_count, device=device, dtype=dtype),
        )
    else:
        signs = torch.ones(graph_count, device=device, dtype=dtype)
    determinant = torch.linalg.det(orthogonal)
    orthogonal[:, :, 0] *= (signs / determinant)[:, None]
    return orthogonal


def _random_elementary_sl3(
    graph_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(3, dtype=dtype, device=device).expand(graph_count, -1, -1).clone()
    row = torch.randint(0, 3, (graph_count,), device=device, generator=generator)
    offset = torch.randint(1, 3, (graph_count,), device=device, generator=generator)
    column = (row + offset) % 3
    sign = 2 * torch.randint(
        0, 2, (graph_count,), device=device, generator=generator
    ) - 1
    graph = torch.arange(graph_count, device=device)
    basis[graph, row, column] = sign.to(dtype)
    inverse = torch.eye(3, dtype=dtype, device=device).expand(graph_count, -1, -1).clone()
    inverse[graph, row, column] = -sign.to(dtype)
    return basis, inverse


def _rotate_rank(value: torch.Tensor, rotation: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 2:
        return torch.einsum("gia,gjb,gab->gij", rotation, rotation, value)
    if rank == 3:
        return torch.einsum("gia,gjb,gkc,gabc->gijk", rotation, rotation, rotation, value)
    if rank == 4:
        return torch.einsum(
            "gia,gjb,gkc,gld,gabcd->gijkl",
            rotation,
            rotation,
            rotation,
            rotation,
            value,
        )
    raise ValueError("only Cartesian rank-two through rank-four tensors are supported")


def augment_equivalent_response_batch(
    value: ResponseBatch,
    *,
    generator: torch.Generator,
    include_rotation: bool = True,
    include_improper: bool = False,
    include_basis: bool = False,
    include_origin: bool = True,
    include_permutation: bool = True,
) -> ResponseBatch:
    """Sample one equivalent presentation per graph with one forward cost.

    The production default uses symmetries already exact in the active input
    chart. Basis changes and improper actions remain explicit audit options.
    """

    if value.fractional_coordinates.device.type != "cpu":
        raise ValueError("equivalent response augmentation runs on the host before transfer")
    internal_mask = value.targets.internal_strain_mask.reshape(
        value.targets.internal_strain_mask.shape[0], -1
    )
    if bool(((internal_mask.any(dim=-1)) & (~internal_mask.all(dim=-1))).any()):
        raise ValueError(
            "Cartesian augmentation requires complete-or-missing internal-strain tensors"
        )
    graph_count = value.graph_count
    dtype = value.fractional_coordinates.dtype
    device = value.fractional_coordinates.device
    if include_improper and not include_rotation:
        raise ValueError("improper augmentation requires Cartesian rotation")
    if include_rotation:
        rotation = _random_orthogonal(
            graph_count,
            dtype=dtype,
            device=device,
            generator=generator,
            allow_improper=include_improper,
        )
    else:
        rotation = torch.eye(3, dtype=dtype, device=device).expand(
            graph_count, -1, -1
        ).clone()
    if include_basis:
        basis, basis_inverse = _random_elementary_sl3(
            graph_count, dtype=dtype, device=device, generator=generator
        )
    else:
        basis = torch.eye(3, dtype=dtype, device=device).expand(
            graph_count, -1, -1
        ).clone()
        basis_inverse = basis.clone()
    # An improper Cartesian action reverses the row-lattice handedness. Pair it
    # with an orientation-reversing integer basis change so the represented
    # reflected crystal remains inside the positive-determinant lattice chart.
    improper = torch.linalg.det(rotation) < 0.0
    orientation = torch.eye(3, dtype=dtype, device=device).expand(
        graph_count, -1, -1
    ).clone()
    orientation[improper, 0, 0] = -1.0
    basis = torch.bmm(orientation, basis)
    basis_inverse = torch.bmm(basis_inverse, orientation)
    if include_origin:
        origin = torch.rand(graph_count, 1, 3, dtype=dtype, generator=generator)
    else:
        origin = torch.zeros(graph_count, 1, 3, dtype=dtype)

    lattice = torch.bmm(basis, value.lattice)
    lattice = torch.bmm(lattice, rotation.transpose(-1, -2))
    fractional = torch.einsum(
        "ni,nij->nj",
        value.fractional_coordinates,
        basis_inverse[value.batch],
    )
    fractional = (fractional + origin[value.batch, 0]).remainder(1.0)

    # Sort random keys within each already contiguous graph; this is one global
    # vectorized permutation rather than a Python loop over structures.
    if include_permutation:
        key = torch.rand(value.batch.shape[0], generator=generator)
        permutation = torch.argsort(value.batch.to(key) + key / 2.0)
    else:
        permutation = torch.arange(value.batch.shape[0])
    node_rotation = rotation[value.batch]
    targets = value.targets
    rotated_targets = replace(
        targets,
        piezoelectric=_rotate_rank(targets.piezoelectric, rotation, 3),
        dielectric=_rotate_rank(targets.dielectric, rotation, 2),
        elastic=_rotate_rank(targets.elastic, rotation, 4),
        born_effective_charge=torch.einsum(
            "nia,njb,nab->nij",
            node_rotation,
            node_rotation,
            targets.born_effective_charge,
        )[permutation],
        internal_strain=torch.einsum(
            "nia,njb,nkc,nabc->nijk",
            node_rotation,
            node_rotation,
            node_rotation,
            targets.internal_strain,
        )[permutation],
        born_mask=targets.born_mask[permutation],
        internal_strain_mask=targets.internal_strain_mask[permutation],
    )
    return ResponseBatch(
        element_tokens=value.element_tokens[permutation],
        fractional_coordinates=fractional[permutation],
        lattice=lattice,
        batch=value.batch,
        node_counts=value.node_counts,
        source_index=value.source_index,
        targets=rotated_targets,
    )
