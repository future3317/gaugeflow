"""Build a tiny real generated-state replay cache from frozen GaugeFlow runtime.

This is a provenance smoke, not a training run and not a quality Gate.  It
uses real Alex structures for node counts, clean side states and source IDs,
then asks the current tensor-free product sampler to produce detached generated
assignment/lattice/coordinate carriers.  The output is written through the
fail-closed generated-state replay cache contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.alex_p1_data import PackedAlexP1Dataset
from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint, read_production_checkpoint_metadata
from gaugeflow.production.composition_runtime import load_qualified_composition_model
from gaugeflow.production.continued_checkpointing import build_continued_pretraining_objects
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.generated_state_replay import (
    GeneratedCarrierRole,
    GeneratedStateReplayEntry,
    GeneratedStateReplayKey,
    load_generated_state_replay_cache,
    write_generated_state_replay_cache,
)
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape, project_lattice_state
from gaugeflow.production.physical_checkpointing import (
    load_physical_ema_for_evaluation,
    read_physical_checkpoint_metadata,
)
from gaugeflow.production.reverse_sampler import TensorFreeReverseSampler
from gaugeflow.production.training import ExponentialMovingAverage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--composition-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help=(
            "If set, select a deterministic random permutation window instead "
            "of a contiguous source slice."
        ),
    )
    parser.add_argument(
        "--forbidden-source-ids",
        type=Path,
        default=None,
        help="Optional JSON list or newline-delimited source IDs that must not be selected.",
    )
    parser.add_argument("--reverse-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5705)
    parser.add_argument("--refresh-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler-commit", default=None)
    return parser.parse_args()


def _current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-git-commit"


def _read_forbidden_source_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        value: Any = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("forbidden source ID JSON must be a list of strings")
        return set(value)
    return {line.strip() for line in text.splitlines() if line.strip()}


def _select_source_indices(
    *,
    split_size: int,
    start_index: int,
    sample_count: int,
    selection_seed: int | None,
) -> list[int]:
    stop = start_index + sample_count
    if start_index < 0 or sample_count < 1 or stop > split_size:
        raise ValueError("requested source selection is outside the packed Alex split")
    if selection_seed is None:
        return list(range(start_index, stop))
    generator = torch.Generator(device="cpu").manual_seed(selection_seed)
    permutation = torch.randperm(split_size, generator=generator)
    return [int(index) for index in permutation[start_index:stop].tolist()]


def _source_ids_for_indices(material_ids: Sequence[str], indices: Sequence[int]) -> list[str]:
    return [str(material_ids[index]) for index in indices]


def _reject_forbidden_selection(source_ids: Sequence[str], forbidden_source_ids: set[str] | None) -> None:
    if forbidden_source_ids is None:
        return
    overlap = sorted(set(source_ids) & forbidden_source_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"selected replay sources overlap forbidden source ids: {preview}")


def _dense_counts(tokens: torch.Tensor, batch: torch.Tensor, graphs: int) -> torch.Tensor:
    flat = batch * 118 + tokens
    return torch.bincount(flat, minlength=graphs * 118).reshape(graphs, 118)


def _load_sampler(
    checkpoint: Path,
    composition_checkpoint: Path,
    composition_protocol: Path,
    *,
    device: torch.device,
) -> tuple[TensorFreeReverseSampler, dict[str, Any]]:
    denoiser: HybridCrystalDenoiser
    try:
        metadata = read_production_checkpoint_metadata(checkpoint)
    except ValueError:
        metadata = read_physical_checkpoint_metadata(checkpoint)
    model_config = metadata.get("model_config")
    training_config = metadata.get("training_config")
    standardization = metadata.get("lattice_standardization")
    if isinstance(model_config, dict) and isinstance(training_config, dict) and isinstance(standardization, dict):
        denoiser = HybridCrystalDenoiser(**model_config).to(device)
        ema = ExponentialMovingAverage(denoiser, float(training_config["ema_decay"]))
        load_production_checkpoint(checkpoint, model=denoiser, ema=ema, map_location=device)
        ema.copy_to(denoiser)
        denoiser.eval()
    else:
        stage_b_metadata = metadata.get("stage_b_metadata")
        if not isinstance(stage_b_metadata, dict):
            raise ValueError("base checkpoint metadata lacks model/training config")
        training_config = stage_b_metadata.get("a1_training_config")
        standardization = stage_b_metadata.get("lattice_standardization")
        physical_config = stage_b_metadata.get("physical_training_config")
        if not isinstance(training_config, dict) or not isinstance(standardization, dict):
            raise ValueError("Stage-B/C checkpoint metadata lacks A1 runtime config")
        if not isinstance(physical_config, dict):
            raise ValueError("Stage-B/C checkpoint metadata lacks physical EMA config")
        objects = build_continued_pretraining_objects(
            stage_b_metadata,
            device=device,
            optimizer_owner=False,
        )
        ema = ExponentialMovingAverage(objects.model, float(physical_config["ema_decay"]))
        load_physical_ema_for_evaluation(checkpoint, model=objects.model, ema=ema, map_location=device)
        objects.model.eval()
        denoiser = objects.model.backbone
    if training_config.get("categorical_path") != "orderless_reveal":
        raise ValueError("tiny replay writer requires an orderless exact-count product checkpoint")
    composition_model = load_qualified_composition_model(
        composition_checkpoint,
        composition_protocol,
        device=device,
    )
    sampler = TensorFreeReverseSampler(
        denoiser,
        P1LatticeStandardizer.from_mapping(standardization),
        coordinate_sigma_min=float(training_config["coordinate_sigma_min"]),
        coordinate_sigma_max=float(training_config["coordinate_sigma_max"]),
        maximum_time=float(training_config["maximum_time"]),
        categorical_path="orderless_reveal",
        composition_model=composition_model,
    )
    return sampler, metadata


def _clean_lattice_state(
    lattice: torch.Tensor,
    blueprint: ParentBlueprintBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = LatticeVolumeShape.from_lattice(lattice, blueprint.fractional_to_cartesian)
    return state.log_volume, project_lattice_state(state.log_shape, blueprint.shape_projector)


def _entry(
    *,
    role: GeneratedCarrierRole,
    source_structure_id: str,
    source_split: str,
    node_count: int,
    assignment_tokens: torch.Tensor,
    assignment_source: str,
    composition_counts: torch.Tensor,
    composition_source: str,
    lattice: torch.Tensor,
    lattice_source: str,
    log_volume: torch.Tensor,
    log_shape: torch.Tensor,
    fractional_coordinates: torch.Tensor,
    coordinate_source: str,
    base_checkpoint_sha256: str,
    sampler_commit: str,
    sampler_protocol_sha256: str,
    refresh_id: int,
    seed: int,
) -> GeneratedStateReplayEntry:
    return GeneratedStateReplayEntry(
        key=GeneratedStateReplayKey(
            source_structure_id=source_structure_id,
            role=role,
            base_checkpoint_sha256=base_checkpoint_sha256,
            sampler_commit=sampler_commit,
            sampler_protocol_sha256=sampler_protocol_sha256,
            refresh_id=refresh_id,
            seed=seed,
            coordinate_time=0.0,
            element_time=0.0,
            lattice_time=0.0,
        ),
        source_split=source_split,
        parent_or_flexible_carrier_id=f"p1_node_count_{node_count}",
        node_count=torch.tensor([node_count], dtype=torch.long),
        composition_counts=composition_counts.cpu(),
        composition_source=composition_source,  # type: ignore[arg-type]
        assignment_tokens=assignment_tokens.cpu(),
        assignment_source=assignment_source,  # type: ignore[arg-type]
        assignment_reveal_rank=torch.arange(node_count, dtype=torch.long),
        assignment_reveal_count=torch.tensor([node_count], dtype=torch.long),
        lattice=lattice.cpu(),
        lattice_source=lattice_source,  # type: ignore[arg-type]
        lattice_log_volume=log_volume.cpu(),
        lattice_log_shape=log_shape.cpu(),
        fractional_coordinates=fractional_coordinates.cpu(),
        coordinate_source=coordinate_source,  # type: ignore[arg-type]
    )


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.sample_count < 1 or args.reverse_steps < 1:
        raise ValueError("sample count and reverse steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    sampler, metadata = _load_sampler(
        args.base_checkpoint,
        args.composition_checkpoint,
        args.composition_protocol,
        device=device,
    )
    dataset = PackedAlexP1Dataset(args.cache_root, args.split, include_material_id=True)
    selected_indices = _select_source_indices(
        split_size=len(dataset),
        start_index=args.start_index,
        sample_count=args.sample_count,
        selection_seed=args.selection_seed,
    )
    source_ids = _source_ids_for_indices(dataset.material_ids_audit_only, selected_indices)
    forbidden_source_ids = _read_forbidden_source_ids(args.forbidden_source_ids)
    _reject_forbidden_selection(source_ids, forbidden_source_ids)
    indices = torch.tensor(selected_indices, dtype=torch.long)
    source = dataset.select_model_batch(indices, device=device)
    node_counts = torch.bincount(source.batch, minlength=args.sample_count)
    blueprint = ParentBlueprintBatch.from_node_counts(node_counts, dtype=source.lattice.dtype, device=device)
    clean_counts = _dense_counts(source.atom_types, source.batch, args.sample_count)

    initialization_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    categorical_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    continuous_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    generated_joint = sampler.sample(
        blueprint,
        steps=args.reverse_steps,
        initialization_generator=initialization_generator,
        categorical_generator=categorical_generator,
        continuous_generator=continuous_generator,
        continuous_mode="reverse_sde",
        time_grid="uniform_log_alpha",
    )

    lattice_initial = sampler.initialize_lattice_state(
        blueprint,
        generator=torch.Generator(device=device).manual_seed(args.seed + 4),
    )
    generated_clean_lattice = sampler.sample_lattice(
        source.atom_types,
        blueprint,
        steps=args.reverse_steps,
        initial_state=lattice_initial,
        continuous_generator=torch.Generator(device=device).manual_seed(args.seed + 5),
        continuous_mode="reverse_sde",
        time_grid="uniform_log_alpha",
    )

    base_sha = sha256_file(args.base_checkpoint)
    sampler_commit = args.sampler_commit or _current_git_commit()
    sampler_protocol = str(metadata.get("protocol_sha256", "unknown-protocol-sha"))
    clean_log_volume, clean_log_shape = _clean_lattice_state(source.lattice, blueprint)
    entries: list[GeneratedStateReplayEntry] = []
    rows: list[dict[str, Any]] = []
    for graph, source_id in enumerate(source_ids):
        selected = source.batch == graph
        node_count = int(node_counts[graph])
        clean_tokens = source.atom_types[selected]
        generated_tokens = generated_joint.element_tokens[selected]
        clean_fractional = source.fractional_coordinates[selected]
        generated_fractional = generated_joint.fractional_coordinates[selected]
        clean_count = clean_counts[graph : graph + 1]
        generated_count = generated_joint.composition_counts[graph : graph + 1]
        graph_seed = args.seed + graph
        common = {
            "source_structure_id": str(source_id),
            "source_split": args.split,
            "node_count": node_count,
            "base_checkpoint_sha256": base_sha,
            "sampler_commit": sampler_commit,
            "sampler_protocol_sha256": sampler_protocol,
            "refresh_id": args.refresh_id,
            "seed": graph_seed,
        }
        entries.extend(
            [
                _entry(
                    role="clean_clean",
                    assignment_tokens=clean_tokens,
                    assignment_source="clean",
                    composition_counts=clean_count,
                    composition_source="clean",
                    lattice=source.lattice[graph : graph + 1],
                    lattice_source="clean",
                    log_volume=clean_log_volume[graph : graph + 1],
                    log_shape=clean_log_shape[graph : graph + 1],
                    fractional_coordinates=clean_fractional,
                    coordinate_source="clean",
                    **common,
                ),
                _entry(
                    role="generated_assignment",
                    assignment_tokens=generated_tokens,
                    assignment_source="generated_assignment",
                    composition_counts=generated_count,
                    composition_source="sampled_composition",
                    lattice=source.lattice[graph : graph + 1],
                    lattice_source="clean",
                    log_volume=clean_log_volume[graph : graph + 1],
                    log_shape=clean_log_shape[graph : graph + 1],
                    fractional_coordinates=clean_fractional,
                    coordinate_source="clean",
                    **common,
                ),
                _entry(
                    role="generated_lattice",
                    assignment_tokens=clean_tokens,
                    assignment_source="clean",
                    composition_counts=clean_count,
                    composition_source="clean",
                    lattice=generated_clean_lattice.lattice[graph : graph + 1],
                    lattice_source="generated_lattice",
                    log_volume=generated_clean_lattice.log_volume[graph : graph + 1],
                    log_shape=generated_clean_lattice.log_shape[graph : graph + 1],
                    fractional_coordinates=clean_fractional,
                    coordinate_source="clean",
                    **common,
                ),
                _entry(
                    role="generated_joint",
                    assignment_tokens=generated_tokens,
                    assignment_source="generated_joint",
                    composition_counts=generated_count,
                    composition_source="sampled_composition",
                    lattice=generated_joint.lattice[graph : graph + 1],
                    lattice_source="generated_joint",
                    log_volume=generated_joint.log_volume[graph : graph + 1],
                    log_shape=generated_joint.log_shape[graph : graph + 1],
                    fractional_coordinates=generated_fractional,
                    coordinate_source="generated_joint",
                    **common,
                ),
            ]
        )
        rows.append(
            {
                "source_structure_id": str(source_id),
                "node_count": node_count,
                "clean_count_sum": int(clean_count.sum()),
                "generated_count_sum": int(generated_count.sum()),
                "generated_matches_clean_counts": bool(torch.equal(clean_count.cpu(), generated_count.cpu())),
            }
        )
    for entry in entries:
        entry.validate(
            expected_base_checkpoint_sha256=base_sha,
            expected_sampler_commit=sampler_commit,
            expected_sampler_protocol_sha256=sampler_protocol,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = write_generated_state_replay_cache(args.output_dir, entries)
    loaded_entries, loaded_manifest = load_generated_state_replay_cache(
        args.output_dir,
        expected_base_checkpoint_sha256=base_sha,
        expected_sampler_commit=sampler_commit,
        expected_sampler_protocol_sha256=sampler_protocol,
    )
    report = {
        "status": "passed",
        "split": args.split,
        "source_start_index": args.start_index,
        "source_sample_count": args.sample_count,
        "source_selection_mode": "permuted" if args.selection_seed is not None else "contiguous",
        "selection_seed": args.selection_seed,
        "source_indices": selected_indices,
        "entry_count": len(entries),
        "loaded_entry_count": len(loaded_entries),
        "reverse_steps": args.reverse_steps,
        "base_checkpoint_sha256": base_sha,
        "sampler_commit": sampler_commit,
        "sampler_protocol_sha256": sampler_protocol,
        "manifest_sha256": manifest_hash,
        "loaded_manifest_sha256": loaded_manifest.canonical_sha256(),
        "roles_per_source": 4,
        "forbidden_source_id_check": {
            "executed": forbidden_source_ids is not None,
            "count": 0 if forbidden_source_ids is None else len(forbidden_source_ids),
        },
        "source_rows": rows,
    }
    if report["loaded_manifest_sha256"] != manifest_hash or len(loaded_entries) != len(entries):
        raise RuntimeError("tiny generated-state replay cache failed round-trip")
    (args.output_dir / "tiny_generated_state_replay_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
