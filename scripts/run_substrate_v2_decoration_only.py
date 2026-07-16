"""Run the pre-registered substrate-v2 fixed-geometry decoration qualification.

This is deliberately a tiny generator-substrate test, not a tensor-conditioned
flow run.  It supplies exact graph composition only to define the finite
assignment support, while the scorer receives all-mask site tokens, periodic
geometry and an endpoint ID.  No target row index, species map, tensor,
stabilizer token, target metadata, relaxation or DFT is an input.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from gaugeflow.assignment import (
    exact_assignment_distribution_permutation_log_probability_error,
    exact_assignment_quotient_nll,
    residual_automorphism_permutations,
    sample_exact_assignment,
)
from gaugeflow.periodic_orbits import unlabeled_periodic_automorphisms
from gaugeflow.pymatgen_compat import enable_structure_matcher_numpy2_compatibility
from gaugeflow.substrate_v2 import GeometryAwareSiteScorer
from gaugeflow.vocabulary import MASK_TOKEN, atomic_numbers_to_tokens, tokens_to_atomic_numbers


ROOT = Path(__file__).resolve().parents[1]
PANEL = ("JVASP-1180", "JVASP-22673")
VARIANTS = (
    ("legacy_scalar_direction_baseline", False, False),
    ("rbf_metric_scorer", True, False),
    ("rbf_vector_invariant_scorer", True, True),
)


@dataclass(frozen=True)
class Endpoint:
    material_id: str
    endpoint_id: int
    lattice: torch.Tensor
    frac: torch.Tensor
    target: torch.Tensor
    counts: torch.Tensor
    proper: torch.Tensor
    full: torch.Tensor


def _load_records(data_dir: Path) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(data_dir / "piezo" / f"{name}.csv") for name in ("train", "val", "test")])
    frame["material_id"] = frame.material_id.astype(str)
    return frame.set_index("material_id", drop=False)


def _automorphism_tensor(structure: Structure, *, proper_only: bool) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    automorphisms = unlabeled_periodic_automorphisms(
        np.asarray(structure.lattice.matrix, dtype=float),
        np.asarray(structure.frac_coords, dtype=float),
        proper_only=proper_only,
    )
    permutations = torch.tensor(
        np.stack([operation["permutation"] for operation in automorphisms.operations]), dtype=torch.long
    )
    return permutations, automorphisms.lattice, automorphisms.frac_coords


def load_panel(data_dir: Path, device: torch.device) -> list[Endpoint]:
    frame = _load_records(data_dir)
    records: list[Endpoint] = []
    for endpoint_id, material_id in enumerate(PANEL):
        if material_id not in frame.index:
            raise ValueError(f"v2 source CSV does not contain required panel ID {material_id}")
        source = Structure.from_str(str(frame.loc[material_id].cif), fmt="cif")
        proper, lattice, frac = _automorphism_tensor(source, proper_only=True)
        full, full_lattice, full_frac = _automorphism_tensor(source, proper_only=False)
        if not np.allclose(lattice, full_lattice) or not np.allclose(frac, full_frac):
            raise RuntimeError("proper/full automorphism analyses disagree on canonical geometry")
        target = atomic_numbers_to_tokens(torch.tensor(source.atomic_numbers, dtype=torch.long))
        counts = torch.bincount(target, minlength=118)
        records.append(
            Endpoint(
                material_id=material_id,
                endpoint_id=endpoint_id,
                lattice=torch.tensor(lattice, dtype=torch.float32, device=device),
                frac=torch.tensor(frac, dtype=torch.float32, device=device),
                target=target.to(device),
                counts=counts.to(device),
                proper=proper.to(device),
                full=full.to(device),
            )
        )
    if any(record.target.numel() != 4 for record in records):
        raise ValueError("the v1 decoration protocol is pre-registered only for the four-site InN/BN panel")
    return records


def batched_inputs(panel: list[Endpoint]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.cat([torch.full_like(record.target, MASK_TOKEN) for record in panel])
    frac = torch.cat([record.frac for record in panel])
    lattice = torch.stack([record.lattice for record in panel])
    batch = torch.cat([torch.full_like(record.target, index) for index, record in enumerate(panel)])
    endpoint = torch.arange(len(panel), dtype=torch.long, device=frac.device)
    return tokens, frac, lattice, batch, endpoint


def scorer_loss(scores: torch.Tensor, panel: list[Endpoint]) -> tuple[torch.Tensor, list[Any]]:
    start = 0
    values = []
    for record in panel:
        stop = start + record.target.numel()
        mask = torch.full_like(record.target, MASK_TOKEN)
        values.append(
            exact_assignment_quotient_nll(
                scores[start:stop], record.counts, record.target, record.proper, mask
            )
        )
        start = stop
    return torch.stack([value.quotient_nll for value in values]).mean(), values


def _match_structure(record: Endpoint, assignment: torch.Tensor) -> bool:
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
    target = Structure(Lattice(record.lattice.detach().cpu().numpy()), tokens_to_atomic_numbers(record.target).cpu().tolist(), record.frac.detach().cpu().numpy())
    generated = Structure(Lattice(record.lattice.detach().cpu().numpy()), tokens_to_atomic_numbers(assignment).cpu().tolist(), record.frac.detach().cpu().numpy())
    return bool(matcher.fit(target, generated))


@torch.no_grad()
def evaluate(
    model: GeometryAwareSiteScorer,
    panel: list[Endpoint],
    *,
    seed: int,
    samples: int,
) -> list[dict[str, object]]:
    device = panel[0].target.device
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    # Both 2+2 endpoints have six unique assignments, so these Gumbels are
    # common random numbers for their endpoint-ID counterfactual samples.
    uniforms = torch.rand((samples, 6), generator=generator, device=device).clamp_(1e-7, 1 - 1e-7)
    gumbels = -torch.log(-torch.log(uniforms))
    output: list[dict[str, object]] = []
    for record in panel:
        mask = torch.full_like(record.target, MASK_TOKEN)
        single_batch = torch.zeros_like(record.target)
        single_lattice = record.lattice.unsqueeze(0)
        single_endpoint = torch.tensor([record.endpoint_id], dtype=torch.long, device=device)
        # Evaluate each graph independently so the subsequent relabeling check
        # contrasts exactly the same graph computation, not two batch layouts.
        site_scores = model(mask, record.frac, single_lattice, single_batch, single_endpoint)
        proper_result = exact_assignment_quotient_nll(
            site_scores, record.counts, record.target, record.proper, mask
        )
        full_result = exact_assignment_quotient_nll(
            site_scores, record.counts, record.target, record.full, mask
        )
        distribution = proper_result.distribution
        map_assignment = distribution.assignments[distribution.energies.argmax()]
        proper_targets = proper_result.unique_orbit_targets
        full_targets = full_result.unique_orbit_targets
        map_proper = bool((map_assignment == proper_targets).all(dim=-1).any())
        map_full = bool((map_assignment == full_targets).all(dim=-1).any())
        draws = [sample_exact_assignment(distribution, gumbel=gumbels[index])[0] for index in range(samples)]
        sampled_proper = np.mean([bool((draw == proper_targets).all(dim=-1).any()) for draw in draws])
        sampled_match = np.mean([_match_structure(record, draw) for draw in draws])
        node_permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long, device=device)
        relabelled_scores = model(
            mask[node_permutation], record.frac[node_permutation], single_lattice,
            single_batch[node_permutation], single_endpoint,
        )
        permutation_error = float(
            exact_assignment_distribution_permutation_log_probability_error(
                site_scores, relabelled_scores, record.counts, node_permutation
            )
        )
        score_error = float((site_scores[node_permutation] - relabelled_scores).abs().max())
        # Report the residual proper automorphism group as species are revealed.
        # These four explicit states avoid treating the initial full group as
        # valid after a partial chemical assignment has broken its symmetry.
        partial_states = [
            mask,
            torch.cat((record.target[:1], mask[1:])),
            torch.cat((record.target[:2], mask[2:])),
            record.target,
        ]
        residual_sizes = [int(residual_automorphism_permutations(state, record.proper).shape[0]) for state in partial_states]
        output.append(
            {
                "material_id": record.material_id,
                "target_quotient_probability": float(proper_result.target_log_probability.exp()),
                "quotient_nll": float(proper_result.quotient_nll),
                "proper_so3_map_quotient_accuracy": float(map_proper),
                "full_o3_map_quotient_accuracy": float(map_full),
                "fixed_cif_site_accuracy": float((map_assignment == record.target).float().mean()),
                "sampled_proper_quotient_accuracy": float(sampled_proper),
                "species_aware_periodic_match_rate": float(sampled_match),
                "assignment_entropy": float(-(distribution.log_probabilities.exp() * distribution.log_probabilities).sum()),
                "equivalent_label_probabilities": json.dumps(
                    [float(distribution.log_probabilities[(distribution.assignments == label).all(dim=-1)].exp().item()) for label in proper_targets]
                ),
                "residual_group_sizes_mask_to_terminal": json.dumps(residual_sizes),
                "exact_composition_count": int(all(torch.equal(torch.bincount(draw, minlength=118), record.counts) for draw in draws)),
                "terminal_masks": int(any(bool((draw == MASK_TOKEN).any()) for draw in draws)),
                "sampling_failures": 0,
                "node_relabel_log_probability_error": permutation_error,
                "node_relabel_score_max_abs_error": score_error,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/substrate_v2_decoration_only_v2.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tensororbit_jarvis_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/substrate_v2_decoration_only_v2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    enable_structure_matcher_numpy2_compatibility()
    protocol_path = ROOT / args.protocol if not args.protocol.is_absolute() else args.protocol
    data_dir = ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir
    output_dir = ROOT / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "pre_registered_ready_data_built_not_started":
        raise ValueError("only the unstarted fixed-budget substrate-v2 protocol may be run")
    settings = protocol["data_and_repetitions"]["optimization_budget"]
    if args.samples < 1:
        raise ValueError("samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    panel = load_panel(data_dir, device)
    tokens, frac, lattice, batch, endpoint = batched_inputs(panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for variant, use_rbf, use_vectors in VARIANTS:
        for seed in protocol["data_and_repetitions"]["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = GeometryAwareSiteScorer(
                hidden_dim=settings["hidden_dim"], layers=settings["layers"],
                vector_channels=settings["vector_channels"], rbf_dim=settings["rbf_dim"],
                cutoff=settings["cutoff_angstrom"], endpoint_classes=2,
                score_bound=settings.get("score_bound", 20.0),
                use_rbf=use_rbf, use_vector_invariants=use_vectors,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
            model.train()
            final_loss = float("nan")
            for _ in range(settings["steps"]):
                optimizer.zero_grad(set_to_none=True)
                scores = model(tokens, frac, lattice, batch, endpoint)
                loss, _ = scorer_loss(scores, panel)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
            model.eval()
            for result in evaluate(model, panel, seed=seed, samples=args.samples):
                rows.append({"variant": variant, "seed": seed, "final_quotient_nll": final_loss, **result})
    csv_path = output_dir / "decoration_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("variant", sort=True).mean(numeric_only=True)
    thresholds = protocol["primary_metrics"]
    primary_pass = bool(
        (frame.proper_so3_map_quotient_accuracy >= thresholds["proper_so3_quotient_assignment_accuracy_min"]).all()
        and (frame.species_aware_periodic_match_rate >= thresholds["species_aware_periodic_structure_match_min"]).all()
        and (frame.exact_composition_count == 1).all()
        and (frame.terminal_masks <= thresholds["terminal_masks_max"]).all()
        and (frame.sampling_failures <= thresholds["sampling_failures_max"]).all()
        and (frame.node_relabel_log_probability_error <= thresholds["node_relabeling_complete_assignment_log_probability_error_fp32_max"]).all()
    )
    manifest = {
        "schema": 1,
        "status": "passed_all_primary_metrics" if primary_pass else "not_passed_primary_metrics",
        "protocol": str(protocol_path),
        "protocol_sha256": __import__("hashlib").sha256(protocol_path.read_bytes()).hexdigest(),
        "runner_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_build_attestation": "artifacts/tensororbit_jarvis_v2_raw_build_v1/attestation.json",
        "device": str(device),
        "variants": [variant[0] for variant in VARIANTS],
        "seeds": protocol["data_and_repetitions"]["seeds"],
        "fixed_steps": settings["steps"],
        "results_csv": str(csv_path),
        "primary_pass": primary_pass,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "decoration_report.md").write_text(
        "# Substrate-v2 decoration-only qualification\n\n"
        f"Status: `{manifest['status']}`. This is fixed-geometry endpoint-ID qualification only; it does not restore tensor conditioning.\n\n"
        "## Mean metrics by variant\n\n"
        + grouped.to_markdown() + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
