"""One-owner Stage-B optimization with physical supervision and Alex replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .matpes_data import MatPESPhysicalBatch
from .physical_pretraining import (
    FunctionalPhysicalNormalizer,
    PhysicalLossOutput,
    PhysicalRepresentationModel,
    physical_multitask_loss,
)
from .training import ExponentialMovingAverage


@dataclass(frozen=True)
class PhysicalTransferTrainingConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    precision: str = "bf16"
    energy_weight: float = 1.0
    force_weight: float = 1.0
    stress_weight: float = 1.0
    feature_weight: float = 1.0

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("physical optimizer rates are invalid")
        if self.gradient_clip_norm <= 0.0 or not 0.0 < self.ema_decay < 1.0:
            raise ValueError("physical clipping or EMA is invalid")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("physical precision must be fp32 or bf16")
        weights = (
            self.energy_weight,
            self.force_weight,
            self.stress_weight,
            self.feature_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("physical task weights must be nonnegative and nonzero")


class PhysicalTransferTrainer:
    """Accumulate MatPES and replay losses before one shared optimizer step."""

    def __init__(
        self,
        model: PhysicalRepresentationModel,
        diffusion: TensorFreeHybridDiffusion,
        config: PhysicalTransferTrainingConfig,
    ) -> None:
        config.validate()
        if diffusion.denoiser is not model.backbone:
            raise ValueError("physical model and replay diffusion must share one backbone instance")
        self.model = model
        self.diffusion = diffusion
        self.config = config
        parameters = list(model.parameters())
        fused = bool(parameters) and all(parameter.device.type == "cuda" for parameter in parameters)
        if fused:
            torch.set_float32_matmul_precision("high")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=fused,
        )
        self.ema = ExponentialMovingAverage(model, config.ema_decay)
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def begin_optimization_step(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_physical_step(
        self,
        batch: MatPESPhysicalBatch,
        normalizer: FunctionalPhysicalNormalizer,
        *,
        loss_weight: float,
    ) -> PhysicalLossOutput:
        if not 0.0 < loss_weight <= 1.0:
            raise ValueError("physical microbatch weight must lie in (0,1]")
        targets = normalizer.normalize(batch.targets, batch.functional_index, batch.batch)
        use_bf16 = self.config.precision == "bf16" and batch.lattice.device.type == "cuda"
        with torch.autocast(
            device_type=batch.lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            prediction = self.model(
                batch.element_tokens,
                batch.fractional_coordinates,
                batch.lattice,
                batch.batch,
                batch.functional_index,
            )
            output = physical_multitask_loss(
                prediction,
                targets,
                batch.batch,
                energy_weight=self.config.energy_weight,
                force_weight=self.config.force_weight,
                stress_weight=self.config.stress_weight,
                feature_weight=self.config.feature_weight,
            )
        (loss_weight * output.loss).backward()
        return output

    def accumulate_alex_replay_step(
        self,
        clean_elements: torch.Tensor,
        clean_fractional_coordinates: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        loss_weight: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not 0.0 < loss_weight <= 1.0:
            raise ValueError("Alex replay weight must lie in (0,1]")
        use_bf16 = self.config.precision == "bf16" and clean_lattice.device.type == "cuda"
        with torch.autocast(
            device_type=clean_lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            output = self.diffusion(
                clean_elements,
                clean_fractional_coordinates,
                clean_lattice,
                batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                generator=generator,
            )
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Alex replay loss is non-finite")
        (loss_weight * output.loss).backward()
        return output.loss.detach()

    def finish_optimization_step(self) -> torch.Tensor:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("physical transfer gradient is non-finite")
        self.optimizer.step()
        self.ema.update(self.model)
        self._step += 1
        return gradient_norm.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        step = state.get("step")
        optimizer = state.get("optimizer")
        ema = state.get("ema")
        if not isinstance(step, int) or step < 0 or not isinstance(optimizer, dict) or not isinstance(ema, dict):
            raise ValueError("physical trainer checkpoint state is invalid")
        execution_backend = [
            (group.get("fused"), group.get("foreach"))
            for group in self.optimizer.param_groups
        ]
        self.optimizer.load_state_dict(optimizer)
        if len(execution_backend) != len(self.optimizer.param_groups):
            raise ValueError("physical optimizer group count changed during restore")
        for group, (fused, foreach) in zip(
            self.optimizer.param_groups,
            execution_backend,
            strict=True,
        ):
            group["fused"] = fused
            group["foreach"] = foreach
        self.ema.load_state_dict(ema)
        self._step = step
