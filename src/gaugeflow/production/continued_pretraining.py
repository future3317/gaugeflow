"""Dataset-neutral structure batches for post-Stage-B continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .matpes_data import MatPESPhysicalRecord


@dataclass(frozen=True)
class StructureReplayBatch:
    element_tokens: torch.Tensor
    fractional_coordinates: torch.Tensor
    lattice: torch.Tensor
    batch: torch.Tensor
    node_counts: torch.Tensor

    def to(self, device: torch.device | str) -> StructureReplayBatch:
        return StructureReplayBatch(
            element_tokens=self.element_tokens.to(device, non_blocking=True),
            fractional_coordinates=self.fractional_coordinates.to(
                device, non_blocking=True
            ),
            lattice=self.lattice.to(device, non_blocking=True),
            batch=self.batch.to(device, non_blocking=True),
            node_counts=self.node_counts.to(device, non_blocking=True),
        )


def collate_structure_records(
    records: Sequence[MatPESPhysicalRecord],
) -> StructureReplayBatch:
    """Pack geometry only; audit IDs, functionals, and physical targets disappear."""

    if not records:
        raise ValueError("cannot collate an empty structure replay batch")
    counts = torch.tensor(
        [record.element_tokens.numel() for record in records], dtype=torch.long
    )
    if bool((counts < 1).any()):
        raise ValueError("structure replay contains an empty graph")
    return StructureReplayBatch(
        element_tokens=torch.cat([record.element_tokens for record in records]),
        fractional_coordinates=torch.cat(
            [record.fractional_coordinates for record in records]
        ),
        lattice=torch.stack([record.lattice for record in records]),
        batch=torch.repeat_interleave(torch.arange(len(records)), counts),
        node_counts=counts,
    )


def structure_replay_loss(
    diffusion: TensorFreeHybridDiffusion,
    clean: StructureReplayBatch,
    *,
    generator: torch.Generator | None = None,
    precision: str = "bf16",
) -> torch.Tensor:
    """Evaluate the unchanged GaugeFlow-base product-space denoising law."""

    if precision not in {"fp32", "bf16"}:
        raise ValueError("structure replay precision must be fp32 or bf16")
    blueprint = ParentBlueprintBatch.from_node_counts(
        clean.node_counts,
        dtype=clean.fractional_coordinates.dtype,
        device=clean.fractional_coordinates.device,
    )
    use_bf16 = precision == "bf16" and clean.lattice.device.type == "cuda"
    with torch.autocast(
        device_type=clean.lattice.device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        output = diffusion(
            clean.element_tokens,
            clean.fractional_coordinates,
            clean.lattice,
            clean.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            generator=generator,
        )
    if not torch.isfinite(output.loss):
        raise FloatingPointError("structure replay loss is non-finite")
    return output.loss
