"""Production tensor-free optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import nn

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import (
    HybridLossOutput,
    LatticeLossOutput,
    TensorFreeHybridDiffusion,
)

TrainingLossOutput: TypeAlias = HybridLossOutput | LatticeLossOutput


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
    objective: str = "joint"
    coordinate_clean_side_information: bool = False
    modality_time_mode: str = "shared"
    categorical_path: str = "absorbing_mask"
    composition_loss_weight: float = 0.0

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
        if self.categorical_path not in {"absorbing_mask", "uniform_replacement"}:
            raise ValueError("unknown categorical training path")
        if self.composition_loss_weight < 0.0:
            raise ValueError("composition loss weight must be nonnegative")
        if self.objective not in {"joint", "coordinate", "element", "lattice"}:
            raise ValueError("training objective must be joint, coordinate, element or lattice")
        if self.coordinate_clean_side_information and self.objective != "coordinate":
            raise ValueError("clean element/lattice side information is coordinate-only")
        if self.modality_time_mode not in {
            "shared",
            "independent_corner_mixture",
            "element_only",
            "lattice_only",
        }:
            raise ValueError("unknown modality-time training mode")
        if self.modality_time_mode == "independent_corner_mixture" and (
            self.objective != "coordinate" or self.coordinate_clean_side_information
        ):
            raise ValueError(
                "independent modality-time attribution requires coordinate training without clean-side override"
            )
        if (self.objective == "element") != (self.modality_time_mode == "element_only"):
            raise ValueError("element training requires the explicit element-only modality-time mode")
        if (self.objective == "lattice") != (self.modality_time_mode == "lattice_only"):
            raise ValueError("lattice training requires the explicit lattice-only modality-time mode")


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
    """One-owner optimizer for joint training or coordinate pretraining."""

    def __init__(
        self,
        diffusion: TensorFreeHybridDiffusion,
        config: ProductionTrainingConfig,
    ) -> None:
        config.validate()
        self.diffusion = diffusion
        self.config = config
        parameters = list(diffusion.denoiser.parameters())
        use_fused_adamw = bool(parameters) and all(parameter.device.type == "cuda" for parameter in parameters)
        if use_fused_adamw:
            # The RTX execution qualification compares this tensor-core path
            # against highest-precision FP32 matmuls on the same real batch.
            # Geometry/state tensors remain FP32; only eligible matmul kernels
            # use TF32 accumulation hardware.
            torch.set_float32_matmul_precision("high")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=use_fused_adamw,
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
    ) -> tuple[HybridLossOutput, torch.Tensor]:
        if self.config.objective == "lattice":
            raise ValueError("lattice objective must use train_lattice_step without coordinates")
        self.begin_optimization_step()
        output = self.accumulate_hybrid_step(
            clean_elements,
            clean_fractional_coordinates,
            clean_lattice,
            batch,
            blueprint,
            generator=generator,
        )
        gradient_norm = self.finish_optimization_step()
        return output, gradient_norm

    def begin_optimization_step(self) -> None:
        """Clear gradients before one optimizer step or microbatch group."""

        self.diffusion.train()
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_hybrid_step(
        self,
        clean_elements: torch.Tensor,
        clean_fractional_coordinates: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        loss_weight: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> HybridLossOutput:
        """Backpropagate one graph-weighted hybrid microbatch without stepping."""

        if self.config.objective == "lattice":
            raise ValueError("lattice objective must use train_lattice_step without coordinates")
        if not 0.0 < loss_weight <= 1.0:
            raise ValueError("microbatch loss weight must lie in (0,1]")
        self.diffusion.train()
        use_bf16 = self.config.precision == "bf16" and clean_lattice.device.type == "cuda"
        with torch.autocast(
            device_type=clean_lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            modality_arguments: dict[str, torch.Tensor] = {}
            if self.config.modality_time_mode == "independent_corner_mixture":
                graph_count = int(blueprint.node_counts.numel())
                modality_times = self.diffusion.sample_task_measure_times(
                    graph_count,
                    clean_fractional_coordinates,
                    generator=generator,
                )
                modality_arguments = {
                    "time": modality_times.coordinate,
                    "element_time": modality_times.element,
                    "lattice_time": modality_times.lattice,
                    "modality_regime": modality_times.regime,
                }
            elif self.config.modality_time_mode == "element_only":
                graph_count = int(blueprint.node_counts.numel())
                element_time = self.diffusion.sample_time(
                    graph_count,
                    clean_fractional_coordinates,
                    generator=generator,
                )
                clean_time = torch.zeros_like(element_time)
                modality_arguments = {
                    "time": clean_time,
                    "element_time": element_time,
                    "lattice_time": clean_time,
                }
            output = self.diffusion(
                clean_elements,
                clean_fractional_coordinates,
                clean_lattice,
                batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                generator=generator,
                clean_side_information=self.config.coordinate_clean_side_information,
                **modality_arguments,
            )
        optimization_loss = self.optimization_loss(output) * loss_weight
        if not torch.isfinite(optimization_loss):
            raise FloatingPointError("selected training loss is non-finite")
        optimization_loss.backward()
        return output

    def finish_optimization_step(self) -> torch.Tensor:
        """Clip accumulated gradients, update parameters and advance EMA once."""

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.diffusion.denoiser.parameters(), self.config.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("hybrid training gradient is non-finite")
        self.optimizer.step()
        self.ema.update(self.diffusion.denoiser)
        self.step = self.step + 1
        return gradient_norm.detach()

    def train_lattice_step(
        self,
        clean_elements: torch.Tensor,
        clean_lattice: torch.Tensor,
        batch: torch.Tensor,
        blueprint: ParentBlueprintBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[LatticeLossOutput, torch.Tensor]:
        """Optimize L1 without accepting target coordinates at the interface."""

        if self.config.objective != "lattice":
            raise ValueError("train_lattice_step requires the lattice objective")
        self.diffusion.train()
        self.optimizer.zero_grad(set_to_none=True)
        use_bf16 = self.config.precision == "bf16" and clean_lattice.device.type == "cuda"
        with torch.autocast(
            device_type=clean_lattice.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            graph_count = int(blueprint.node_counts.numel())
            lattice_time = self.diffusion.sample_time(
                graph_count,
                clean_lattice,
                generator=generator,
            )
            output = self.diffusion.forward_lattice(
                clean_elements,
                clean_lattice,
                batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                lattice_time=lattice_time,
                generator=generator,
            )
        gradient_norm = self._optimize(output)
        return output, gradient_norm

    def _optimize(self, output: TrainingLossOutput) -> torch.Tensor:
        optimization_loss = self.optimization_loss(output)
        if not torch.isfinite(optimization_loss):
            raise FloatingPointError("selected training loss is non-finite")
        optimization_loss.backward()
        # Keep the scalar on-device. Production logging materializes it only
        # at the existing synchronized log boundary.
        return self.finish_optimization_step()

    def optimization_loss(self, output: TrainingLossOutput) -> torch.Tensor:
        """Select the sole preregistered objective without mixing inactive heads."""

        if self.config.objective == "joint":
            if not isinstance(output, HybridLossOutput):
                raise TypeError("joint objective received a lattice loss output")
            return output.loss
        if self.config.objective == "coordinate":
            if not isinstance(output, HybridLossOutput):
                raise TypeError("coordinate objective received a lattice loss output")
            return output.coordinate_loss
        if self.config.objective == "lattice":
            if not isinstance(output, LatticeLossOutput):
                raise TypeError("lattice objective received a hybrid loss output")
            return output.loss
        if not isinstance(output, HybridLossOutput):
            raise TypeError("element objective received a lattice loss output")
        return output.element_loss + self.config.composition_loss_weight * output.composition_loss
