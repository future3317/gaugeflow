"""Train standalone GaugeFlow on paired CIF/tensor CSV data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from gaugeflow.conditioning import apply_condition_dropout, randomize_tensor_orbit_representative
from gaugeflow.data import RESPONSE_NORM_BOUNDS, PiezoCrystalDataset, collate_crystals
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.tensor import normalize_isotypic


def select_subset_indices(dataset: PiezoCrystalDataset, count: int, strategy: str) -> list[int]:
    """Return a deterministic real-data subset for diagnostic overfitting."""
    if count < 1:
        raise ValueError("max-examples must be positive")
    if count >= len(dataset):
        return list(range(len(dataset)))
    if strategy == "head":
        return list(range(count))
    # The paired CSV stores a cheap pre-cache norm.  It is sufficient to make
    # a diverse diagnostic subset and avoids opening thousands of cache files
    # across the WSL mount. The cached Reynolds-projected target remains the
    # actual condition used by the resulting training run.
    if "piezo_norm" in dataset.frame:
        norms = torch.tensor(dataset.frame["piezo_norm"].to_numpy(), dtype=torch.float32)
        bins = torch.full_like(norms, len(RESPONSE_NORM_BOUNDS), dtype=torch.long)
        bins[norms <= 1e-12] = 0
        for bin_index, upper in enumerate(RESPONSE_NORM_BOUNDS[1:], start=1):
            bins[(norms > 1e-12) & (norms < upper) & (bins == len(RESPONSE_NORM_BOUNDS))] = bin_index
    else:
        bins = dataset.condition_bins()
    groups = [torch.nonzero(bins == value, as_tuple=False).flatten().tolist() for value in range(int(bins.max()) + 1)]
    groups = [group for group in groups if group]
    selected: list[int] = []
    cursor = 0
    while len(selected) < count:
        group = groups[cursor % len(groups)]
        index = len(selected) // len(groups)
        if index < len(group):
            selected.append(group[index])
        cursor += 1
        if cursor >= len(groups) and all(len(selected) // len(groups) >= len(group) for group in groups):
            break
    if len(selected) < count:
        used = set(selected)
        selected.extend(index for index in range(len(dataset)) if index not in used)
    return sorted(selected[:count])


def select_material_id_indices(dataset: PiezoCrystalDataset, path: Path) -> list[int]:
    """Select a frozen diagnostic panel by material ID, preserving file order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    material_ids = payload.get("material_ids") if isinstance(payload, dict) else payload
    if not isinstance(material_ids, list) or not material_ids:
        raise ValueError(f"{path} must contain a non-empty material_ids list")
    requested = [str(value) for value in material_ids]
    if len(requested) != len(set(requested)):
        raise ValueError(f"{path} contains duplicate material IDs")
    lookup = {str(value): index for index, value in enumerate(dataset.frame.material_id)}
    missing = [value for value in requested if value not in lookup]
    if missing:
        raise ValueError(f"Material IDs are absent from the selected dataset split: {missing}")
    return [lookup[value] for value in requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="A paired CSV, or the original FlowMM piezo CSV directory")
    parser.add_argument("--split-manifest", type=Path,
                        help="Frozen TensorOrbit-JARVIS split manifest")
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="Protocol split to select from --split-manifest")
    parser.add_argument("--target-cache-dir", type=Path,
                        help="Frozen TensorOrbit-JARVIS Reynolds-target directory")
    parser.add_argument("--preprocessed-cache", type=Path,
                        help="Versioned CIF/Niggli/condition tensor cache")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--orbit-frames", type=int, default=24)
    parser.add_argument("--conditioning-mode", choices=("raw_tensor", "direct_irrep", "stabilizer_pooling", "orbit_alignment", "double_coset"), default="orbit_alignment",
                        help="Raw lab-frame control, Cartesian direct-irrep control, orbit pooling, inference-consistent alignment, or legacy alias")
    parser.add_argument("--max-examples", type=int,
                        help="Use a deterministic real-data diagnostic subset; intended for Gate A only")
    parser.add_argument("--material-ids-file", type=Path,
                        help="JSON file containing a frozen material_ids diagnostic panel")
    parser.add_argument("--subset-strategy", choices=("stratified", "head"), default="stratified",
                        help="How a Gate A diagnostic subset is selected")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--uncertainty-weight", type=float, default=0.1,
                        help="Weight of heteroscedastic tangent-velocity NLL after warmup")
    parser.add_argument("--uncertainty-warmup-steps", type=int, default=5_000,
                        help="Use deterministic flow matching before enabling UQ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Deterministic diagnostic order only")
    parser.add_argument("--condition-balanced-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Inverse-frequency sampling across TensorOrbit-JARVIS response-norm strata")
    parser.add_argument("--condition-sampling-power", type=float, default=0.5,
                        help="0 is uniform; 1 is full inverse-frequency reweighting")
    parser.add_argument("--condition-dropout", type=float, default=0.1,
                        help="Probability of replacing a present tensor condition with the learned CFG null token")
    parser.add_argument("--direct-irrep-random-frame", action=argparse.BooleanOptionalAction, default=True,
                        help="Randomize the tensor orbit representative for the direct-irrep control during training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_examples is not None and args.material_ids_file is not None:
        parser.error("--max-examples and --material-ids-file are mutually exclusive")

    torch.manual_seed(args.seed)
    full_dataset = PiezoCrystalDataset(
        args.train_csv,
        split_manifest=args.split_manifest,
        split=args.split,
        target_cache_dir=args.target_cache_dir,
        preprocessed_cache=args.preprocessed_cache,
    )
    if args.material_ids_file is not None:
        indices = select_material_id_indices(full_dataset, args.material_ids_file)
    elif args.max_examples is not None:
        indices = select_subset_indices(full_dataset, args.max_examples, args.subset_strategy)
    else:
        indices = list(range(len(full_dataset)))
    # Do not materialize every cache entry for a Gate A subset: that makes a
    # 32-example diagnostic pay the full 4,000-example label-I/O cost.
    condition_values = torch.stack([full_dataset._condition_for_index(index)[0] for index in indices])
    scales = torch.stack(
        [condition_values[:, block].square().mean().sqrt().clamp_min(1e-8) for block in (
            slice(0, 6), slice(6, 11), slice(11, 18)
        )]
    )
    diagnostic_subset = args.max_examples is not None or args.material_ids_file is not None
    # Gate A revisits the same tiny panel many times. Cache its parsed/Niggli-
    # reduced PyG records once so timing measures the model, not CIF parsing.
    dataset = [full_dataset[index] for index in indices] if diagnostic_subset else Subset(full_dataset, indices)
    sampler = None
    if not args.no_shuffle and args.condition_balanced_sampling:
        sampler = WeightedRandomSampler(
            full_dataset.condition_sampling_weights(args.condition_sampling_power)[indices],
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_crystals,
    )
    model = GaugeFlowVectorField(
        args.hidden_dim, args.layers, args.orbit_frames, conditioning_mode=args.conditioning_mode
    )
    records_are_normalized = False
    if diagnostic_subset:
        # The frozen panel revisits exactly the same conditions. Normalize and
        # build the fixed tensor orbit once; the state-derived posterior and
        # local bond queries remain dynamic inside the model.
        for record in dataset:
            record.piezo_irreps = normalize_isotypic(record.piezo_irreps, scales)
            if args.conditioning_mode in {"stabilizer_pooling", "orbit_alignment", "double_coset"}:
                with torch.no_grad():
                    record.condition_orbit = model.response.precompute_condition_orbit(
                        record.piezo_irreps
                    )
        records_are_normalized = True
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    resident_batch = None
    resident_piezo = None
    resident_present = None
    if diagnostic_subset and args.no_shuffle and sampler is None and len(dataset) <= args.batch_size:
        resident_batch = collate_crystals(dataset).to(args.device)
        resident_piezo = resident_batch.piezo_irreps
        resident_present = resident_batch.condition_present
    iterator = iter(loader) if resident_batch is None else None
    for step in range(1, args.steps + 1):
        if resident_batch is not None:
            batch = resident_batch
            batch.piezo_irreps = resident_piezo
            batch.condition_present = resident_present
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = batch.to(args.device)
        if not records_are_normalized:
            batch.piezo_irreps = normalize_isotypic(batch.piezo_irreps, scales.to(args.device))
        if args.conditioning_mode == "direct_irrep" and args.direct_irrep_random_frame:
            batch.piezo_irreps = randomize_tensor_orbit_representative(batch.piezo_irreps)
        batch.condition_present = apply_condition_dropout(
            batch.condition_present, args.condition_dropout
        )
        optimizer.zero_grad(set_to_none=True)
        uncertainty_weight = args.uncertainty_weight if step > args.uncertainty_warmup_steps else 0.0
        terms = RiemannianCrystalFlowMatcher(uncertainty_weight=uncertainty_weight).loss(model, batch)
        if not torch.isfinite(terms["loss"]):
            raise FloatingPointError(
                "Non-finite GaugeFlow loss. The run stops rather than silently applying a numerical fallback."
            )
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print({key: float(value.detach().cpu()) for key, value in terms.items()})
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "isotypic_scales": scales,
            "data_protocol": {
                "split_manifest": str(args.split_manifest) if args.split_manifest else None,
                "split": args.split,
                "target_cache_dir": str(args.target_cache_dir) if args.target_cache_dir else None,
                "preprocessed_cache": str(args.preprocessed_cache) if args.preprocessed_cache else None,
                "preprocessed_manifest": full_dataset.preprocessed_manifest,
                "condition_balanced_sampling": args.condition_balanced_sampling,
                "condition_sampling_power": args.condition_sampling_power,
                "condition_dropout": args.condition_dropout,
                "direct_irrep_random_frame": args.direct_irrep_random_frame,
                "max_examples": args.max_examples,
                "material_ids_file": str(args.material_ids_file) if args.material_ids_file else None,
                "subset_strategy": args.subset_strategy,
                "material_ids": [str(full_dataset.frame.iloc[index].material_id) for index in indices],
            },
            "format": "gaugeflow-standalone-v1",
        },
        args.checkpoint,
    )


if __name__ == "__main__":
    main()
