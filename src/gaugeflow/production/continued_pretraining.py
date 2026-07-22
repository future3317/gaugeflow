"""Dataset-neutral structure batches for post-Stage-B continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .matpes_data import MatPESPhysicalBatch, MatPESPhysicalRecord
from .physical_pretraining import (
    FunctionalPhysicalNormalizer,
    PhysicalLossDenominators,
    PhysicalLossOutput,
)
from .physical_training import PhysicalTransferTrainer


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


@dataclass(frozen=True)
class ContinuedPretrainingWeights:
    lemat_structure: float
    matpes_physical: float
    alex_structure: float

    def validate(self) -> None:
        values = (self.lemat_structure, self.matpes_physical, self.alex_structure)
        if any(value <= 0.0 for value in values) or abs(sum(values) - 1.0) > 1.0e-12:
            raise ValueError("continued-pretraining weights must be positive and sum to one")


@dataclass(frozen=True)
class ContinuedPretrainingLosses:
    lemat_structure: torch.Tensor
    matpes_physical: PhysicalLossOutput
    alex_structure: torch.Tensor


def pack_structure_batch(
    element_tokens: torch.Tensor,
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
) -> StructureReplayBatch:
    """Validate and pack already-vectorized graph tensors."""

    graphs = lattice.shape[0]
    if (
        element_tokens.ndim != 1
        or element_tokens.dtype != torch.long
        or fractional_coordinates.shape != (element_tokens.numel(), 3)
        or lattice.shape != (graphs, 3, 3)
        or graphs < 1
        or batch.shape != element_tokens.shape
        or batch.dtype != torch.long
        or int(batch.min()) < 0
        or int(batch.max()) >= graphs
    ):
        raise ValueError("structure replay tensors are invalid")
    counts = torch.bincount(batch, minlength=graphs)
    if bool((counts < 1).any()):
        raise ValueError("structure replay graph indices must be contiguous and nonempty")
    return StructureReplayBatch(
        element_tokens=element_tokens,
        fractional_coordinates=fractional_coordinates,
        lattice=lattice,
        batch=batch,
        node_counts=counts,
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
    return pack_structure_batch(
        torch.cat([record.element_tokens for record in records]),
        torch.cat([record.fractional_coordinates for record in records]),
        torch.stack([record.lattice for record in records]),
        torch.repeat_interleave(torch.arange(len(records)), counts),
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


def accumulate_continued_pretraining_step(
    trainer: PhysicalTransferTrainer,
    lemat: StructureReplayBatch,
    matpes: MatPESPhysicalBatch,
    alex: StructureReplayBatch,
    normalizer: FunctionalPhysicalNormalizer,
    weights: ContinuedPretrainingWeights,
    *,
    lemat_rank_fraction: float,
    alex_rank_fraction: float,
    physical_denominators: PhysicalLossDenominators | None = None,
    generator: torch.Generator | None = None,
) -> ContinuedPretrainingLosses:
    """Accumulate one mathematically global three-stream objective.

    Physical losses already use global label-bearing denominators. Structure
    losses are local graph means, so each is multiplied by its exact rank
    fraction before gradients are summed across ranks.
    """

    weights.validate()
    if not 0.0 < lemat_rank_fraction <= 1.0 or not 0.0 < alex_rank_fraction <= 1.0:
        raise ValueError("continued-pretraining rank fractions must lie in (0,1]")
    trainer.begin_optimization_step()
    physical = trainer.accumulate_physical_step(
        matpes,
        normalizer,
        loss_weight=weights.matpes_physical,
        denominators=physical_denominators,
    )
    lemat_loss = structure_replay_loss(
        trainer.diffusion,
        lemat,
        generator=generator,
        precision=trainer.config.precision,
    )
    (weights.lemat_structure * lemat_rank_fraction * lemat_loss).backward()
    alex_loss = structure_replay_loss(
        trainer.diffusion,
        alex,
        generator=generator,
        precision=trainer.config.precision,
    )
    (weights.alex_structure * alex_rank_fraction * alex_loss).backward()
    return ContinuedPretrainingLosses(
        lemat_structure=lemat_loss.detach(),
        matpes_physical=physical,
        alex_structure=alex_loss.detach(),
    )
