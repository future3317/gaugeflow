"""Run the fixed two-target Gate A3 early-branching screen and nothing else."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a3_early_branching_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_a3_early_branching_v1/two_target"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_two_target_only":
        raise ValueError("Gate A3 runner requires the frozen two-target protocol")
    if len(protocol["material_ids"]) != 2 or protocol["material_ids"] != protocol["selection"]["selected_material_ids"]:
        raise ValueError("Gate A3 must use exactly the verified two selected targets")
    if protocol["training"]["conditional_control"] != "original_injection":
        raise ValueError("A3 must not search the A2 residual/FiLM mechanism")
    if protocol["training"]["steps"] != 400 or protocol["training"]["checkpoint_steps"] != [200, 400]:
        raise ValueError("A3 step budget must remain the pre-registered 200/400 curve")
    selection_path = _resolve(protocol["selection"]["selection_manifest"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "verified_pre_registered_two_target_selection":
        raise ValueError("Run the A3 selection verification before training")
    if selection.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Selection manifest does not belong to this exact A3 protocol")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = protocol["training"]
    data = protocol["data"]
    manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": "dry_run" if args.dry_run else "running",
        "protocol_sha256": _sha256(protocol_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "scope": "two direct-irrep targets only; no S2, 4/8-target extension, full benchmark, relaxation, DFT, or DFPT",
        "variants": [],
    }
    for variant in protocol["variants"]:
        variant_dir = output_dir / variant["id"]
        final_checkpoint = variant_dir / "step_0400.pt"
        if final_checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite pre-registered checkpoint: {final_checkpoint}")
        command = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--train-csv", str(_resolve(data["train_csv"])),
            "--split-manifest", str(_resolve(data["split_manifest"])),
            "--split", data["split"],
            "--target-cache-dir", str(_resolve(data["target_cache_dir"])),
            "--preprocessed-cache", str(_resolve(data["preprocessed_cache"])),
            "--material-ids-file", str(protocol_path),
            "--checkpoint", str(final_checkpoint),
            "--checkpoint-at", "200",
            "--steps", "400",
            "--batch-size", str(train["batch_size"]),
            "--lr", str(train["learning_rate"]),
            "--hidden-dim", str(train["hidden_dim"]),
            "--layers", str(train["layers"]),
            "--orbit-frames", str(train["orbit_frames"]),
            "--conditioning-mode", train["conditioning_mode"],
            "--conditional-control", train["conditional_control"],
            "--condition-dropout", str(train["condition_dropout"]),
            "--counterfactual-weight", str(train["counterfactual_weight"]),
            "--uncertainty-weight", str(train["uncertainty_weight"]),
            "--identification-weight", str(variant["identification_weight"]),
            "--identification-temperature", str(variant["identification_temperature"]),
            "--identification-early-sigma", str(variant["identification_early_sigma"]),
            "--seed", str(train["seed"]),
            "--no-shuffle", "--no-condition-balanced-sampling", "--direct-irrep-random-frame",
            "--log-every", "100", "--device", args.device,
        ]
        manifest["variants"].append({
            "id": variant["id"], "command": command, "checkpoint_steps": train["checkpoint_steps"],
            "checkpoint": str(final_checkpoint),
        })
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        manifest["status"] = "completed_training_pending_evaluation"
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
