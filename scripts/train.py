"""Train standalone GaugeFlow on paired CIF/tensor CSV data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gaugeflow.data import PiezoCrystalDataset, collate_crystals
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import isotypic_slices, normalize_isotypic


def fit_isotypic_scales(csv_path: Path, condition_column: str = "piezo_irreps_raw") -> torch.Tensor:
    """Read tensor statistics without reparsing every CIF."""
    frame = pd.read_csv(csv_path)
    values = torch.tensor([json.loads(value) for value in frame[condition_column]], dtype=torch.float32)
    return torch.stack([values[:, block].square().mean().sqrt().clamp_min(1e-8) for block in isotypic_slices()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--orbit-frames", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Deterministic diagnostic order only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = PiezoCrystalDataset(args.train_csv)
    scales = fit_isotypic_scales(args.train_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle, collate_fn=collate_crystals)
    model = GaugeFlowVectorField(args.hidden_dim, args.layers, args.orbit_frames).to(args.device)
    matcher = RiemannianCrystalFlowMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(args.device)
        batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales.to(args.device))
        optimizer.zero_grad(set_to_none=True)
        terms = matcher.loss(model, batch)
        if not torch.isfinite(terms["loss"]):
            raise FloatingPointError(
                "Non-finite GaugeFlow loss. The run stops rather than silently applying a numerical fallback."
            )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print({key: float(value.detach().cpu()) for key, value in terms.items()})
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "isotypic_scales": scales,
            "format": "gaugeflow-standalone-v1",
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
