"""Production tensor-free optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import HybridLossOutput, TensorFreeHybridDiffusion


@dataclass(frozen=True)
class ProductionTrainingConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-6
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    coordinate_sigma_min: float = 0.005
    coordinate_sigma_max: float = 0.5
    minimum_time: float = 1.0e-3
    maximum_time: float = 0.999
    precision: str = "bf16"

    def validate(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer rates must be nonnegative with positive learning rate")
        if self.gradient_clip_norm <= 0.0 or not 0.0 < self.ema_decay < 1.0:
            raise ValueError("gradient clipping and EMA decay are invalid")
        if not 0.0 < self.coordinate_sigma_min < self.coordinate_sigma_max:
            raise ValueError("fractional torus scales must satisfy 0 < min < max")
        if not 0.0 < self.minimum_time < self.maximum_time < 1.0:
            raise ValueError("training time interval is invalid")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("training precision must be fp32 or bf16")


class ExponentialMovingAverage:
    """EMA over the complete denoiser state, including floating buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state does not match model state")
        for name, value in state.items():
            target = self.shadow[name]
            if value.dtype.is_floating_point:
                target.lerp_(value.detach(), 1.0 - self.decay)
            else:
                target.copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        decay = float(state["decay"])
        shadow = state["shadow"]
        if not 0.0 < decay < 1.0 or not isinstance(shadow, dict):
            raise ValueError("invalid EMA checkpoint state")
        if shadow.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint does not match model")
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in shadow.items()}


class ProductionTrainer:
    """One-owner optimizer for the tensor-free hybrid objective."""

    def __init__(
        self,
        diffusion: TensorFreeHybridDiffusion,
        config: ProductionTrainingConfig,
    ) -> None:
        config.validate()
        self.diffusion = diffusion
        self.config = config
        self.optimizer = torch.optim.AdamW(
            diffusion.denoiser.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.ema = ExponentialMovingAverage(diffusion.denoiser, config.ema_decay)
        self.step = 0

    def train_step(
        self,
        clean_elements: torch.Tensor,
        clean_fractional_coordinates: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[HybridLossOutput, float]:
        self.diffusion.train()
        self.optimizer.zero_grad(set_to_none=True)
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
            raise FloatingPointError("hybrid training loss is non-finite")
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.diffusion.denoiser.parameters(), self.config.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("hybrid training gradient is non-finite")
        self.optimizer.step()
        self.ema.update(self.diffusion.denoiser)
        self.step = self.step + 1
        return output, float(gradient_norm.detach().cpu())
