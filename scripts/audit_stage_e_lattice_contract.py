"""Read-only Stage-E lattice convention and split-contract audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(lattice: torch.Tensor) -> dict[str, object]:
    lattice = lattice.double()
    metric = lattice @ lattice.transpose(-1, -2)
    eig = torch.linalg.eigvalsh(metric).clamp_min(torch.finfo(torch.float64).tiny)
    volume = torch.linalg.det(lattice)
    return {
        "shape": list(lattice.shape),
        "finite_fraction": float(torch.isfinite(lattice).all(dim=(-2, -1)).double().mean()),
        "positive_volume_fraction": float((volume > 0).double().mean()),
        "volume_quantiles": torch.quantile(
            volume, torch.tensor([0.0, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64)
        ).tolist(),
        "volume_per_atom_not_available": True,
        "minimum_width_quantiles": torch.quantile(
            eig[:, 0].sqrt(), torch.tensor([0.0, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64)
        ).tolist(),
        "metric_condition_quantiles": torch.quantile(
            eig[:, -1] / eig[:, 0],
            torch.tensor([0.0, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64),
        ).tolist(),
    }


def main() -> None:
    root = Path("/home/workspace/lrh/DATA/T2C-Flow")
    stage_d = root / "processed/stage_d_jarvis_multitask_v1"
    alex = root / "processed/gaugeflow_h1a_v1/p1_structure_cache_v1"
    lemat = root / "processed/lemat_training_index_v4"
    output = root / "evaluations/stage_e_lattice_contract_audit_v1.json"
    stage_d_payload = torch.load(stage_d / "data.pt", map_location="cpu", weights_only=False)
    alex_payload = torch.load(alex / "val.pt", map_location="cpu", weights_only=False)
    if not isinstance(stage_d_payload, dict) or not isinstance(alex_payload, dict):
        raise TypeError("unexpected cache payload")
    stage_d_lattice = stage_d_payload["lattice"]
    alex_lattice = alex_payload["lattice"]
    if not isinstance(stage_d_lattice, torch.Tensor) or not isinstance(alex_lattice, torch.Tensor):
        raise TypeError("lattice tensors missing")
    stage_d_manifest = json.loads((stage_d / "MANIFEST.json").read_text())
    alex_manifest = json.loads((alex / "manifest.json").read_text())
    lemat_manifest = json.loads((lemat / "manifest.json").read_text())
    # Stage-D train/val/test split identifiers are explicit in the payload.
    split_keys = {key: value for key, value in stage_d_payload.items() if "index" in key or "split" in key}
    split_shapes = {
        key: list(value.shape) if isinstance(value, torch.Tensor) else str(type(value))
        for key, value in split_keys.items()
    }
    source_index = stage_d_payload.get("source_index")
    split_index_tensor = stage_d_payload.get("split_index")
    split_source_overlaps: dict[str, int] = {}
    if isinstance(source_index, torch.Tensor) and isinstance(split_index_tensor, torch.Tensor):
        split_sources = {
            split_id: set(source_index[split_index_tensor == split_id].tolist())
            for split_id in (0, 1, 2)
        }
        split_source_overlaps = {
            "train_val": len(split_sources[0] & split_sources[1]),
            "train_test": len(split_sources[0] & split_sources[2]),
            "val_test": len(split_sources[1] & split_sources[2]),
        }
    # The Stage-E factorial panel is the deterministic Stage-D validation
    # selection after the six pure-noble-gas quarantine rows.
    validation_elements = stage_d_payload.get("element_tokens")
    node_offsets = stage_d_payload.get("node_offsets")
    selected_indices: list[int] = []
    if isinstance(validation_elements, torch.Tensor) and isinstance(node_offsets, torch.Tensor):
        noble = {1, 9, 17, 35, 53, 85, 117}
        validation_global = [
            index
            for index in range(int(stage_d_payload["lattice"].shape[0]))
            if int(stage_d_payload["split_index"][index].item()) == 1
        ]
        eligible = [
            local_index
            for local_index, index in enumerate(validation_global)
            if not (
                torch.unique(
                    validation_elements[node_offsets[index] : node_offsets[index + 1]]
                ).numel()
                == 1
                and int(validation_elements[node_offsets[index]].item()) in noble
            )
        ]
        generator = torch.Generator().manual_seed(20260723)
        selected_indices = torch.tensor(eligible)[torch.randperm(len(eligible), generator=generator)[:256]].tolist()
    result = {
        "schema": "gaugeflow.stage_e_lattice_contract_audit.v1",
        "source_files": {
            "stage_d_manifest_sha256": sha(stage_d / "MANIFEST.json"),
            "stage_d_data_sha256": sha(stage_d / "data.pt"),
            "alex_manifest_sha256": sha(alex / "manifest.json"),
            "alex_val_sha256": sha(alex / "val.pt"),
            "lemat_manifest_sha256": sha(lemat / "manifest.json"),
            "lemat_index_sha256": sha(lemat / "index.pt"),
        },
        "declared_conventions": {
            "stage_d": stage_d_manifest.get("conventions"),
            "alex_cache_protocol": alex_manifest.get("protocol"),
            "lemat_scope": lemat_manifest.get("scope"),
            "lattice_representation": (
                "3x3 rows are Cartesian lattice vectors; "
                "fractional row vectors multiply on the left"
            ),
            "volume": "det(L) in Angstrom^3; volume-per-atom is det(L)/N",
            "chart": "P1 trace-free six-coordinate chart projected to five active coordinates before standardization",
        },
        "cache_payload_keys": {
            "stage_d": sorted(stage_d_payload.keys()),
            "alex_val": sorted(alex_payload.keys()),
            "stage_d_split_like_shapes": split_shapes,
        },
        "lattice_statistics": {
            "stage_d_all_selected": summary(stage_d_lattice),
            "alex_validation": summary(alex_lattice),
        },
        "factorial_panel": {
            "seed": 20260723,
            "eligible_validation_count": len(eligible) if isinstance(validation_elements, torch.Tensor) else None,
            "selected_count": len(selected_indices),
            "selected_indices_sha256": hashlib.sha256(
                json.dumps(selected_indices, separators=(",", ":")).encode()
            ).hexdigest(),
            "selected_indices": selected_indices,
        },
        "split_contract": {
            "adapter_training_split": "stage_d train",
            "adapter_evaluation_split": "stage_d val/test only",
            "source_overlap_counts": split_source_overlaps,
            "validation_rows_excluded_by_quarantine": 6,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
