"""Measure per-head gradient scale on the frozen H1a substrate sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("E:/DATA/T2C-Flow/processed/gaugeflow_h1a_v1/p1_structure_cache_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "reports/h1a_generator_substrate_audit_v1/post_normalization_gradients.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graphs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6113)
    arguments = parser.parse_args()
    if arguments.graphs < 1:
        raise ValueError("gradient audit needs at least one graph")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    dataset = PackedAlexP1Dataset(arguments.cache_root, "train")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(arguments.seed))[
        : arguments.graphs
    ]
    packed = Batch.from_data_list([dataset[int(index)] for index in indices]).to(device)
    counts = torch.bincount(packed.batch, minlength=packed.num_graphs)
    blueprint = ParentBlueprintBatch.from_node_counts(
        counts, dtype=packed.frac_coords.dtype, device=device
    )
    model = HybridCrystalDenoiser().to(device)
    standardizer = P1LatticeStandardizer.from_json(
        repo_root / "configs/statistics/h1a_p1_lattice_standardization.json"
    )
    diffusion = TensorFreeHybridDiffusion(model, standardizer)
    generator = torch.Generator(device=device).manual_seed(arguments.seed + 1)
    output = diffusion(
        packed.atom_types,
        packed.frac_coords,
        packed.lattice,
        packed.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        generator=generator,
    )
    shared = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "element_embedding",
                "degree_embedding",
                "time_embedding",
                "state_embedding",
                "blocks",
            )
        )
    ]
    losses = {
        "element": output.element_loss,
        "coordinate": output.coordinate_loss,
        "volume": output.volume_loss,
        "shape": output.shape_loss,
    }
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for name, loss in losses.items():
        gradients[name] = torch.autograd.grad(
            loss, shared, retain_graph=True, allow_unused=True
        )
    norms = {
        name: float(
            torch.sqrt(
                sum(
                    gradient.double().square().sum()
                    for gradient in values
                    if gradient is not None
                )
            ).cpu()
        )
        for name, values in gradients.items()
    }
    cosine: dict[str, float] = {}
    names = tuple(losses)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            dot = sum(
                (left_gradient.double() * right_gradient.double()).sum()
                for left_gradient, right_gradient in zip(
                    gradients[left], gradients[right], strict=True
                )
                if left_gradient is not None and right_gradient is not None
            )
            denominator = norms[left] * norms[right]
            cosine[f"{left}__{right}"] = (
                float(dot.cpu()) / denominator if denominator > 0.0 else 0.0
            )
    result = {
        "protocol": "h1a_generator_substrate_audit_v1",
        "stage": "after_multigraph_fractional_path_and_lattice_standardization",
        "seed": arguments.seed,
        "graphs": arguments.graphs,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "shared_gradient_norms": norms,
        "shared_gradient_cosines": cosine,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
