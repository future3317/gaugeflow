"""Sample standalone GaugeFlow from a tensor orbit; no target lattice is accepted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gaugeflow.checkpoints import load_safe_checkpoint
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import normalize_isotypic, piezo_to_irreps, piezo_voigt_to_cartesian
from torch_geometric.data import Batch, Data


def load_target(path: Path, scales: torch.Tensor) -> torch.Tensor:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"piezo_voigt", "voigt_order", "engineering_shear"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Missing target fields: {sorted(missing)}")
    if tuple(value["voigt_order"]) != ("xx", "yy", "zz", "yz", "xz", "xy"):
        raise ValueError("Expected Voigt order [xx, yy, zz, yz, xz, xy]")
    if value["engineering_shear"] is not True:
        raise ValueError("GaugeFlow requires engineering_shear=true")
    voigt = torch.tensor(value["piezo_voigt"], dtype=torch.float32)
    if voigt.shape != (3, 6) or not torch.isfinite(voigt).all():
        raise ValueError("piezo_voigt must be finite with shape [3,6]")
    return normalize_isotypic(piezo_to_irreps(piezo_voigt_to_cartesian(voigt)), scales)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-atoms", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload, metadata = load_safe_checkpoint(args.checkpoint, map_location=args.device)
    config = metadata["config"]
    model = GaugeFlowVectorField(
        config["hidden_dim"], config["layers"], config["orbit_frames"],
        conditioning_mode=config.get("conditioning_mode", "orbit_alignment"),
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    target = load_target(args.target, payload["isotypic_scales"]).to(args.device)
    samples = [
        Data(
            atom_types=torch.zeros(args.num_atoms, dtype=torch.long),
            frac_coords=torch.zeros((args.num_atoms, 3)),
            lattice=torch.eye(3).unsqueeze(0),
            piezo_irreps=target.cpu().unsqueeze(0),
            condition_present=torch.ones((1, 1), dtype=torch.bool),
            num_nodes=args.num_atoms,
        )
        for _ in range(args.num_samples)
    ]
    batch = Batch.from_data_list(samples).to(args.device)
    state = RiemannianCrystalFlowMatcher().sample(
        model, batch, steps=args.steps, guidance_scale=args.guidance_scale
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type_state": state.type_state.cpu(),
            "frac_coords": state.frac_coords.cpu(),
            "lattice_log": state.lattice_log.cpu(),
            "target_irreps": target.cpu(),
            "format": "gaugeflow-sample-tensors-v2",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
