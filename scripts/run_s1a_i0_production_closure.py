"""Run the frozen CUDA S1a-I0 trainer/reverse-sampler closure smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gaugeflow.production.blueprint import P1BlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.reverse_sampler import SamplingFailure, TensorFreeReverseSampler
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gates/s1a_tensor_free_production_v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_panel(device: torch.device) -> tuple[torch.Tensor, ...]:
    elements = torch.tensor([4, 6, 12, 15], dtype=torch.long, device=device)
    coordinates = torch.tensor(
        [[0.05, 0.10, 0.15], [0.35, 0.25, 0.70], [0.15, 0.75, 0.45], [0.72, 0.55, 0.20]],
        device=device,
    )
    lattice = torch.tensor(
        [[[3.2, 0.0, 0.0], [0.2, 3.8, 0.0], [0.1, 0.3, 4.2]]], device=device
    )
    blueprint = P1BlueprintBatch.from_counts(torch.tensor([4], device=device), device=device)
    return elements, coordinates, lattice, blueprint.batch, blueprint


def evaluate_fixed_loss(
    diffusion: TensorFreeHybridDiffusion,
    panel: tuple[torch.Tensor, ...],
    seed: int,
) -> float:
    elements, coordinates, lattice, batch, blueprint = panel
    generator = torch.Generator(device=elements.device).manual_seed(seed)
    with torch.no_grad():
        output = diffusion(
            elements,
            coordinates,
            lattice,
            batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=torch.tensor([0.5], device=elements.device),
            generator=generator,
        )
    return float(output.loss.cpu())


def gradient_norm(modules: tuple[torch.nn.Module, ...]) -> float:
    squared = torch.zeros((), device=next(modules[0].parameters()).device)
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().square().sum()
    return float(squared.sqrt().cpu())


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    gate = protocol["implementation_gate"]
    if not torch.cuda.is_available():
        raise RuntimeError("S1a-I0 is pinned to the WSL CUDA environment")
    device = torch.device("cuda")
    seed = int(protocol["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = {
        "hidden_dim": 32,
        "vector_dim": 8,
        "layers": 2,
        "radial_dim": 6,
        "atlas_residual_circle_samples": 8,
    }
    model = HybridCrystalDenoiser(**model_config).to(device)
    training = protocol["training"]
    training_config = ProductionTrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ema_decay=float(training["ema_decay"]),
        coordinate_sigma_max=float(training["coordinate_sigma_max_angstrom"]),
        minimum_time=float(training["minimum_time"]),
        maximum_time=float(training["maximum_time"]),
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        coordinate_sigma_max=training_config.coordinate_sigma_max,
        minimum_time=training_config.minimum_time,
        maximum_time=training_config.maximum_time,
    )
    trainer = ProductionTrainer(diffusion, training_config)
    panel = fixed_panel(device)
    initial_loss = evaluate_fixed_loss(diffusion, panel, seed + 100)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    started = time.perf_counter()
    final_gradient_norm = 0.0
    final_heads: dict[str, float] = {}
    for _ in range(int(gate["cuda_smoke_steps"])):
        output, final_gradient_norm = trainer.train_step(
            panel[0], panel[1], panel[2], panel[3], panel[4], generator=generator
        )
        final_heads = {
            "element": float(output.element_loss.detach().cpu()),
            "coordinate": float(output.coordinate_loss.detach().cpu()),
            "volume": float(output.volume_loss.detach().cpu()),
            "shape": float(output.shape_loss.detach().cpu()),
        }
    head_gradient_norms = {
        "element": gradient_norm((model.element_head,)),
        "coordinate": gradient_norm(
            (model.coordinate_vector_head, model.coordinate_time_gate, model.coordinate_edge_head)
        ),
        "volume": gradient_norm((model.volume_head,)),
        "shape": gradient_norm((model.shape_head,)),
    }
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final_loss = evaluate_fixed_loss(diffusion, panel, seed + 100)
    ema_model = HybridCrystalDenoiser(**model_config).to(device)
    trainer.ema.copy_to(ema_model)
    sampler = TensorFreeReverseSampler(
        ema_model,
        coordinate_sigma_max=training_config.coordinate_sigma_max,
        maximum_time=training_config.maximum_time,
    )
    failures = 0
    terminal_masks = -1
    failure_reason: str | None = None
    try:
        generated = sampler.sample(
            panel[4],
            steps=int(protocol["reverse_sampler"]["steps"]),
            generator=torch.Generator(device=device).manual_seed(seed + 2),
            time_grid=str(protocol["reverse_sampler"]["time_grid"]),
        )
        terminal_masks = int(generated.diagnostics.masked_count[-1])
    except SamplingFailure as error:
        failures = 1
        failure_reason = str(error)
    ratio = final_loss / max(initial_loss, 1.0e-12)
    passed = (
        ratio <= float(gate["fixed_evaluation_loss_ratio_max"])
        and failures == int(gate["sampling_failures"])
        and terminal_masks == int(gate["terminal_masks"])
        and final_gradient_norm > 0.0
        and all(torch.isfinite(torch.tensor(list(final_heads.values()))))
        and all(torch.isfinite(torch.tensor(list(head_gradient_norms.values()))))
    )
    result = {
        "protocol": protocol["protocol"],
        "subgate": "S1a-I0-production-closure",
        "decision": "passed_implementation_closure" if passed else "failed_implementation_closure",
        "scientific_s1a_status": "not_run",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "steps": trainer.step,
        "initial_fixed_loss": initial_loss,
        "final_fixed_loss": final_loss,
        "fixed_loss_ratio": ratio,
        "final_head_losses": final_heads,
        "final_gradient_norm": final_gradient_norm,
        "head_gradient_norms": head_gradient_norms,
        "sampling_failures": failures,
        "sampling_failure_reason": failure_reason,
        "terminal_masks": terminal_masks,
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
