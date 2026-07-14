"""Run the pre-registered Gate A2 S1 direct-irrep mechanism screen only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/gate_a2_conditional_control_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate_a2_conditional_control_v1/s1_direct_irrep"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_s1_direct_irrep_only":
        raise ValueError("This runner only accepts the frozen A2 S1 direct-irrep protocol")
    if protocol["s1"]["methods"] != ["direct_irrep"]:
        raise ValueError("S1 must remain direct_irrep only")
    archive_path = resolve(protocol["parent_archive"])
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    if archive.get("status") != "frozen_negative_result_do_not_modify":
        raise ValueError("Gate A v1 archive must be frozen before S1")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = protocol["data"]
    train = protocol["s1"]["training"]
    run_manifest = {
        "schema": 1,
        "name": protocol["name"],
        "status": "running" if not args.dry_run else "dry_run",
        "protocol_sha256": sha256_file(protocol_path),
        "parent_archive_sha256": sha256_file(archive_path),
        "scope": "S1 direct_irrep only; no S2, full benchmark, relaxation, DFT, or DFPT",
        "variants": [],
    }
    for variant in protocol["s1"]["variants"]:
        variant_dir = output_dir / variant["id"]
        final_checkpoint = variant_dir / "step_0800.pt"
        if final_checkpoint.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite pre-registered checkpoint: {final_checkpoint}")
        command = [
            sys.executable, str(ROOT / "scripts" / "train.py"),
            "--train-csv", str(resolve(data["train_csv"])),
            "--split-manifest", str(resolve(data["split_manifest"])),
            "--split", "train",
            "--target-cache-dir", str(resolve(data["target_cache_dir"])),
            "--preprocessed-cache", str(resolve(data["preprocessed_cache"])),
            "--material-ids-file", str(protocol_path),
            "--checkpoint", str(final_checkpoint),
            "--checkpoint-at", *[str(step) for step in train["checkpoint_steps"] if step != train["steps"]],
            "--steps", str(train["steps"]),
            "--batch-size", str(train["batch_size"]),
            "--lr", str(train["learning_rate"]),
            "--hidden-dim", str(train["hidden_dim"]),
            "--layers", str(train["layers"]),
            "--orbit-frames", str(train["orbit_frames"]),
            "--conditioning-mode", "direct_irrep",
            "--conditional-control", variant["conditional_control"],
            "--residual-g-min", str(protocol["s1"]["conditional_residual_field"]["g_min"]),
            "--counterfactual-weight", str(variant["counterfactual_weight"]),
            "--counterfactual-margin", str(protocol["s1"]["counterfactual_loss"]["margin"]),
            "--condition-dropout", str(variant["condition_dropout"]),
            "--uncertainty-weight", str(train["uncertainty_weight"]),
            "--seed", str(protocol["s1"]["seed"]),
            "--no-shuffle", "--no-condition-balanced-sampling", "--direct-irrep-random-frame",
            "--log-every", "100", "--device", args.device,
        ]
        run_manifest["variants"].append({
            "id": variant["id"], "command": command, "final_checkpoint": str(final_checkpoint),
            "checkpoint_steps": train["checkpoint_steps"],
        })
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        run_manifest["status"] = "completed_training_pending_s1_evaluation"
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
