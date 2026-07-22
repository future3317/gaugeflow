"""One-owner Stage-B optimization with physical supervision and Alex replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .blueprint import ParentBlueprintBatch
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .matpes_data import MatPESPhysicalBatch
from .physical_pretraining import (
    FunctionalPhysicalNormalizer,
    PhysicalLossDenominators,
    PhysicalLossOutput,
    PhysicalRepresentationModel,
    physical_loss_denominators,
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
        *,
        optimizer_owner: bool = True,
    ) -> None:
        config.validate()
        if diffusion.denoiser is not model.backbone:
            raise ValueError("physical model and replay diffusion must share one backbone instance")
        self.model = model
        self.diffusion = diffusion
        self.config = config
        parameters = list(model.parameters())
        fused = bool(parameters) and all(parameter.device.type == "cuda" for parameter in parameters)
        self.optimizer = (
            torch.optim.AdamW(
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                fused=fused,
            )
            if optimizer_owner
            else None
        )
        self.ema = ExponentialMovingAverage(model, config.ema_decay) if optimizer_owner else None
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def begin_optimization_step(self) -> None:
        self.model.train()
        self.model.zero_grad(set_to_none=True)

    def accumulate_physical_step(
        self,
        batch: MatPESPhysicalBatch,
        normalizer: FunctionalPhysicalNormalizer,
        *,
        loss_weight: float,
        denominators: PhysicalLossDenominators | None = None,
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
                denominators=denominators,
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
        if self.optimizer is None or self.ema is None:
            raise RuntimeError("only the optimizer-owning rank can finish a local step")
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

    @staticmethod
    def distributed_local_fraction(local_count: int, *, device: torch.device) -> float:
        """Return the exact local/global example fraction without padded samples."""

        if local_count < 0 or not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed count reduction requires an initialized process group")
        count = torch.tensor(float(local_count), device=device, dtype=torch.float64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = float(count)
        if total <= 0.0:
            raise ValueError("distributed optimization received no examples")
        return local_count / total

    @staticmethod
    def distributed_physical_denominators(
        batch: MatPESPhysicalBatch,
    ) -> PhysicalLossDenominators:
        """All-reduce label-bearing graph counts for unbiased masked task means."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed physical normalization requires a process group")
        graph_count = batch.lattice.shape[0]
        local = physical_loss_denominators(batch.targets, batch.batch, graph_count)
        counts = torch.tensor(
            [local.energy, local.force, local.stress, local.feature],
            device=batch.lattice.device,
            dtype=torch.int64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return PhysicalLossDenominators(*(int(value) for value in counts))

    def finish_distributed_optimization_step(self, *, owner_rank: int = 0) -> torch.Tensor:
        """Sum globally weighted gradients, update once, then broadcast parameters."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed physical transfer requires a process group")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 2 or not 0 <= owner_rank < world_size:
            raise ValueError("distributed physical transfer needs a valid multi-rank owner")
        if (self.optimizer is not None) != (rank == owner_rank) or (
            self.ema is not None
        ) != (rank == owner_rank):
            raise RuntimeError("optimizer and EMA ownership disagree with distributed rank")
        parameters = tuple(self.model.parameters())
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        if rank == owner_rank:
            assert self.optimizer is not None and self.ema is not None
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("distributed physical transfer gradient is non-finite")
            self.optimizer.step()
            self.ema.update(self.model)
            self._step += 1
        else:
            gradient_norm = next(self.model.parameters()).new_zeros(())
        self._broadcast_model(owner_rank)
        step_and_norm = next(self.model.parameters()).new_tensor(
            [float(self._step), float(gradient_norm)], dtype=torch.float64
        )
        dist.broadcast(step_and_norm, src=owner_rank)
        self._step = int(step_and_norm[0])
        self.model.zero_grad(set_to_none=True)
        return step_and_norm[1].to(dtype=torch.float32)

    def finish_replicated_distributed_optimization_step(
        self,
        *,
        bucket_bytes: int = 25 * 1024 * 1024,
    ) -> torch.Tensor:
        """Reduce bucketed gradients and update identical optimizer replicas.

        Every rank starts from the same model, optimizer and EMA state.  A
        bounded flat buffer replaces thousands of scalar collectives, and
        every rank applies the same reduced gradient locally.  This removes
        the former full-parameter broadcast without changing the global loss.
        """

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("replicated physical transfer requires a process group")
        if dist.get_world_size() < 2 or self.optimizer is None or self.ema is None:
            raise RuntimeError("every distributed rank must own optimizer and EMA state")
        if bucket_bytes < 1024:
            raise ValueError("distributed gradient bucket is too small")
        parameters = tuple(self.model.parameters())
        buckets: list[list[torch.nn.Parameter]] = []
        current: list[torch.nn.Parameter] = []
        current_bytes = 0
        current_key: tuple[torch.device, torch.dtype] | None = None
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            key = (parameter.grad.device, parameter.grad.dtype)
            size = parameter.grad.numel() * parameter.grad.element_size()
            if current and (key != current_key or current_bytes + size > bucket_bytes):
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append(parameter)
            current_bytes += size
            current_key = key
        if current:
            buckets.append(current)
        for bucket in buckets:
            flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad = flat[offset : offset + count].view_as(parameter)
                offset += count
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.config.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("replicated physical transfer gradient is non-finite")
        self.optimizer.step()
        self.ema.update(self.model)
        self._step += 1
        self.model.zero_grad(set_to_none=True)
        return gradient_norm.detach()

    def _broadcast_model(self, owner_rank: int) -> None:
        for parameter in self.model.parameters():
            dist.broadcast(parameter.data, src=owner_rank)
        for buffer in self.model.buffers():
            dist.broadcast(buffer.data, src=owner_rank)

    def broadcast_distributed_state(self, *, owner_rank: int = 0) -> None:
        """Restore non-owner model/step replicas after a rank-0 checkpoint load."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed state broadcast requires a process group")
        rank = dist.get_rank()
        if (self.optimizer is not None) != (rank == owner_rank):
            raise RuntimeError("distributed state owner disagrees with optimizer ownership")
        self._broadcast_model(owner_rank)
        step = next(self.model.parameters()).new_tensor([self._step], dtype=torch.int64)
        dist.broadcast(step, src=owner_rank)
        self._step = int(step[0])
        self.model.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        if self.optimizer is None or self.ema is None:
            raise RuntimeError("only the optimizer-owning rank has trainer checkpoint state")
        return {
            "step": self._step,
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.optimizer is None or self.ema is None:
            raise RuntimeError("only the optimizer-owning rank can restore trainer state")
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
