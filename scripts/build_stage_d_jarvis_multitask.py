"""Build the immutable GaugeFlow Stage-D JARVIS multi-task tensor cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Structure

from gaugeflow.file_utils import sha256_file
from gaugeflow.production.response_targets import (
    canonical_gamma_spectrum,
    engineering_stiffness_to_kelvin,
    kelvin_stiffness_to_cartesian,
    scatter_internal_strain_blocks,
)
from gaugeflow.vocabulary import atomic_numbers_to_tokens

DFPT_BORN_CONVENTION = (
    "PiezoJet Z[j,i], coordinate/force row and polarization/field column"
)
DFPT_INTERNAL_STRAIN_CONVENTION = "PiezoJet Lambda=dF/deta; no additional sign change"
TENSOR_ORBIT_CONVENTION = (
    "canonical engineering Voigt [xx,yy,zz,yz,xz,xy] -> Cartesian ijk=ikj -> "
    "full-O(3) crystal-point-group Reynolds projection"
)
ELASTIC_TARGET_CONSTRUCTION = (
    "full Cartesian point-group Reynolds projection of the raw GPa stiffness; "
    "raw source and projection residual remain in audit rows"
)


@dataclass(frozen=True)
class AuditRow:
    material_id: str
    split: str
    atom_count: int
    reduced_composition: str
    tensor_target_sha256: str
    dfpt_sha256: str | None
    internal_strain_maximum_antisymmetric_residual: float
    internal_strain_source_symmetric_within_rounding: bool
    elastic_available: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensororbit-root", type=Path, required=True)
    parser.add_argument("--dfpt-root", type=Path, required=True)
    parser.add_argument("--strain-completion-root", type=Path, required=True)
    parser.add_argument("--elastic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-atoms", type=int, default=20)
    parser.add_argument("--gamma-eigenvalue-scale", type=float, default=1.0)
    return parser.parse_args()


def _material_key(material_id: str) -> str:
    return hashlib.sha256(material_id.encode("utf-8")).hexdigest()[:16]


def _reduced_composition(numbers: list[int]) -> str:
    counts = Counter(numbers)
    divisor = 0
    for value in counts.values():
        divisor = gcd(divisor, value)
    return ";".join(f"{number}:{counts[number] // divisor}" for number in sorted(counts))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_roots(
    tensororbit_root: Path,
    dfpt_root: Path,
    strain_completion_root: Path,
    elastic_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tensor_manifest = _load_json(tensororbit_root / "build_manifest.json")
    if (
        tensor_manifest.get("target_cache_schema") != 3
        or tensor_manifest.get("tensor_convention") != TENSOR_ORBIT_CONVENTION
        or tensor_manifest.get("split_counts") != {"train": 4000, "val": 499, "test": 499}
    ):
        raise ValueError("TensorOrbit artifact does not match the Stage-D source contract")
    dfpt_manifest = _load_json(dfpt_root / "manifest.json")
    if dfpt_manifest.get("schema") != 4 or dfpt_manifest.get("cached") != 4995:
        raise ValueError("JARVIS DFPT artifact does not match the Stage-D source contract")
    strain_manifest = _load_json(strain_completion_root / "manifest.json")
    if strain_manifest.get("schema") != 2 or strain_manifest.get("accepted") != 1638:
        raise ValueError("internal-strain completion does not match the Stage-D source contract")
    elastic_audit = _load_json(elastic_root / "audit.json")
    if (
        elastic_audit.get("schema") != 1
        or elastic_audit.get("source_field") != "elastic_total_kbar"
        or elastic_audit.get("target_unit") != "GPa"
        or elastic_audit.get("target_construction") != ELASTIC_TARGET_CONSTRUCTION
        or elastic_audit.get("accepted") != 3291
    ):
        raise ValueError("JARVIS elastic artifact does not match the Stage-D source contract")
    return tensor_manifest, dfpt_manifest, strain_manifest, elastic_audit


def _load_elastic_targets(root: Path, audit: dict[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(root / "accepted_targets_gpa.pt", map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != 1
        or payload.get("unit") != "GPa"
        or payload.get("source_field") != "elastic_total_kbar"
        or not isinstance(payload.get("targets"), dict)
    ):
        raise ValueError("JARVIS elastic target payload has the wrong contract")
    raw_targets = payload["targets"]
    accepted_ids = {
        str(row["jid"])
        for row in audit.get("rows", [])
        if isinstance(row, dict) and bool(row.get("accepted"))
    }
    if len(accepted_ids) != int(audit["accepted"]) or set(raw_targets) != accepted_ids:
        raise ValueError("JARVIS elastic target identities disagree with the audit")
    result: dict[str, torch.Tensor] = {}
    for material_id, value in raw_targets.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape != (6, 6) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{material_id}: elastic target is invalid")
        result[str(material_id)] = kelvin_stiffness_to_cartesian(
            engineering_stiffness_to_kelvin(tensor)
        )
    return result


def _load_split_rows(root: Path) -> list[tuple[str, dict[str, str]]]:
    result: list[tuple[str, dict[str, str]]] = []
    for split in ("train", "val", "test"):
        with (root / "piezo" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        result.extend((split, row) for row in rows)
    identifiers = [row["material_id"] for _, row in result]
    if len(identifiers) != 4998 or len(set(identifiers)) != len(identifiers):
        raise ValueError("TensorOrbit split does not contain 4,998 unique material IDs")
    return result


def _label_tensor(
    path: Path,
    *,
    expected_schema: int,
    field: str,
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"{path} has the wrong cache schema")
    value = payload.get(field)
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise ValueError(f"{path} lacks floating tensor field {field}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{path}:{field} contains non-finite values")
    return value.float()


def main() -> None:
    arguments = _arguments()
    if arguments.maximum_atoms < 1 or arguments.gamma_eigenvalue_scale <= 0.0:
        raise ValueError("Stage-D atom and Gamma-spectrum scales must be positive")
    tensor_manifest, dfpt_manifest, strain_manifest, elastic_audit = _validate_roots(
        arguments.tensororbit_root,
        arguments.dfpt_root,
        arguments.strain_completion_root,
        arguments.elastic_root,
    )
    elastic_targets = _load_elastic_targets(arguments.elastic_root, elastic_audit)
    rows = _load_split_rows(arguments.tensororbit_root)

    element_tokens: list[torch.Tensor] = []
    fractional_coordinates: list[torch.Tensor] = []
    lattices: list[torch.Tensor] = []
    node_offsets = [0]
    piezoelectric: list[torch.Tensor] = []
    dielectric: list[torch.Tensor] = []
    dielectric_mask: list[bool] = []
    elastic: list[torch.Tensor] = []
    elastic_mask: list[bool] = []
    born: list[torch.Tensor] = []
    born_mask: list[torch.Tensor] = []
    gamma_soft: list[torch.Tensor] = []
    gamma_magnitude: list[torch.Tensor] = []
    gamma_mask: list[torch.Tensor] = []
    internal_strain: list[torch.Tensor] = []
    internal_mask: list[torch.Tensor] = []
    split_index: list[int] = []
    audit_rows: list[AuditRow] = []
    excluded: list[dict[str, Any]] = []

    split_vocabulary = {"train": 0, "val": 1, "test": 2}
    for split, row in rows:
        material_id = row["material_id"]
        structure = Structure.from_str(row["cif"], fmt="cif")
        if not structure.is_ordered:
            excluded.append({"material_id": material_id, "reason": "disordered_structure"})
            continue
        atom_count = len(structure)
        if atom_count > arguments.maximum_atoms:
            excluded.append(
                {
                    "material_id": material_id,
                    "reason": "atom_count_outside_registered_domain",
                    "atom_count": atom_count,
                }
            )
            continue
        numbers = [int(site.specie.Z) for site in structure]
        key = _material_key(material_id)
        target_path = arguments.tensororbit_root / "reynolds_projected_targets" / f"{key}.pt"
        target = _label_tensor(target_path, expected_schema=3, field="target")
        if target.shape != (3, 3, 3):
            raise ValueError(f"{material_id}: piezoelectric target has the wrong shape")
        has_elastic = material_id in elastic_targets
        elastic_value = (
            elastic_targets[material_id]
            if has_elastic
            else torch.zeros(3, 3, 3, 3, dtype=torch.float32)
        )

        dfpt_path = arguments.dfpt_root / f"{key}.pt"
        has_dfpt = dfpt_path.is_file()
        if has_dfpt:
            dfpt = torch.load(dfpt_path, map_location="cpu", weights_only=True)
            if not isinstance(dfpt, dict) or dfpt.get("schema") != 4 or dfpt.get("jid") != material_id:
                raise ValueError(f"{material_id}: DFPT identity/schema mismatch")
            conventions = dfpt.get("conventions")
            if not isinstance(conventions, dict) or (
                conventions.get("born_charges_internal") != DFPT_BORN_CONVENTION
                or conventions.get("internal_strain_internal")
                != DFPT_INTERNAL_STRAIN_CONVENTION
            ):
                raise ValueError(f"{material_id}: DFPT tensor convention mismatch")
            born_value = torch.as_tensor(dfpt["born_charges"], dtype=torch.float32)
            if born_value.shape != (atom_count, 3, 3) or not bool(torch.isfinite(born_value).all()):
                raise ValueError(f"{material_id}: Born-charge/structure atom mismatch")
            epsilon = dfpt.get("epsilon")
            if not isinstance(epsilon, dict):
                raise ValueError(f"{material_id}: missing dielectric branches")
            electronic = torch.as_tensor(epsilon.get("epsilon"), dtype=torch.float32)
            ionic = torch.as_tensor(epsilon.get("epsilon_ion"), dtype=torch.float32)
            if electronic.shape != (3, 3) or ionic.shape != (3, 3):
                raise ValueError(f"{material_id}: dielectric branch shape mismatch")
            dielectric_value = 0.5 * (
                electronic + ionic + electronic.T + ionic.T
            )
            if not bool(torch.isfinite(dielectric_value).all()):
                raise ValueError(f"{material_id}: dielectric target is non-finite")
            gamma = canonical_gamma_spectrum(
                torch.as_tensor(dfpt["dynamical_eigenvalues"], dtype=torch.float32),
                maximum_atoms=arguments.maximum_atoms,
                eigenvalue_scale=arguments.gamma_eigenvalue_scale,
            )
            try:
                observed_strain = scatter_internal_strain_blocks(
                    torch.as_tensor(dfpt["internal_strain_tensors"], dtype=torch.float32),
                    torch.as_tensor(dfpt["internal_strain_ions"], dtype=torch.long),
                    torch.as_tensor(dfpt["internal_strain_directions"], dtype=torch.long),
                    atom_count=atom_count,
                    rounding_halfwidth=torch.as_tensor(
                        dfpt["internal_strain_rounding_halfwidth"], dtype=torch.float32
                    ),
                )
            except ValueError as error:
                raise ValueError(f"{material_id}: {error}") from error
            completion_path = arguments.strain_completion_root / f"{material_id}.pt"
            if completion_path.is_file():
                completion = torch.load(
                    completion_path, map_location="cpu", weights_only=True
                )
                if (
                    not isinstance(completion, dict)
                    or completion.get("schema") != 2
                    or completion.get("jid") != material_id
                ):
                    raise ValueError(f"{material_id}: internal-strain completion identity mismatch")
                completed_strain = torch.as_tensor(
                    completion.get("internal_strain_full"), dtype=torch.float32
                )
                if completed_strain.shape != (atom_count, 3, 3, 3) or not bool(
                    torch.isfinite(completed_strain).all()
                ):
                    raise ValueError(f"{material_id}: completed internal strain has invalid shape")
                strain_residual = (
                    completed_strain - completed_strain.transpose(-1, -2)
                ).abs().amax()
                if float(strain_residual) > 1e-5 * float(
                    completed_strain.abs().amax().clamp_min(1.0)
                ):
                    raise ValueError(f"{material_id}: completed internal strain is not symmetric")
                strain_value = 0.5 * (
                    completed_strain + completed_strain.transpose(-1, -2)
                )
                strain_mask_value = torch.ones_like(strain_value, dtype=torch.bool)
            else:
                strain_value = torch.zeros(atom_count, 3, 3, 3)
                strain_mask_value = torch.zeros_like(strain_value, dtype=torch.bool)
        else:
            born_value = torch.zeros(atom_count, 3, 3)
            dielectric_value = torch.zeros(3, 3)
            gamma = canonical_gamma_spectrum(
                torch.empty(0),
                maximum_atoms=arguments.maximum_atoms,
                eigenvalue_scale=arguments.gamma_eigenvalue_scale,
            )
            observed_strain = scatter_internal_strain_blocks(
                torch.empty(0, 3, 3),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                atom_count=atom_count,
            )
            strain_value = observed_strain.value
            strain_mask_value = torch.zeros_like(strain_value, dtype=torch.bool)

        element_tokens.append(
            atomic_numbers_to_tokens(torch.tensor(numbers, dtype=torch.long))
        )
        fractional_coordinates.append(
            torch.tensor(structure.frac_coords, dtype=torch.float32).remainder(1.0)
        )
        lattices.append(torch.tensor(structure.lattice.matrix, dtype=torch.float32))
        node_offsets.append(node_offsets[-1] + atom_count)
        piezoelectric.append(target)
        dielectric.append(dielectric_value)
        dielectric_mask.append(has_dfpt)
        elastic.append(elastic_value)
        elastic_mask.append(has_elastic)
        born.append(born_value)
        born_mask.append(torch.full((atom_count,), has_dfpt, dtype=torch.bool))
        gamma_soft.append(gamma.soft)
        gamma_magnitude.append(gamma.log_magnitude)
        gamma_mask.append(gamma.mask if has_dfpt else torch.zeros_like(gamma.mask))
        internal_strain.append(strain_value)
        internal_mask.append(strain_mask_value)
        split_index.append(split_vocabulary[split])
        audit_rows.append(
            AuditRow(
                material_id=material_id,
                split=split,
                atom_count=atom_count,
                reduced_composition=_reduced_composition(numbers),
                tensor_target_sha256=sha256_file(target_path),
                dfpt_sha256=sha256_file(dfpt_path) if has_dfpt else None,
                internal_strain_maximum_antisymmetric_residual=(
                    observed_strain.maximum_antisymmetric_residual
                ),
                internal_strain_source_symmetric_within_rounding=(
                    observed_strain.source_symmetric_within_rounding
                ),
                elastic_available=has_elastic,
            )
        )

    split_compositions = {
        split: {
            row.reduced_composition for row in audit_rows if row.split == split
        }
        for split in split_vocabulary
    }
    overlap = {
        "train_val": len(split_compositions["train"] & split_compositions["val"]),
        "train_test": len(split_compositions["train"] & split_compositions["test"]),
        "val_test": len(split_compositions["val"] & split_compositions["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"reduced-composition split leakage detected: {overlap}")

    arguments.output.mkdir(parents=True, exist_ok=False)
    cache_path = arguments.output / "data.pt"
    torch.save(
        {
            "schema": 1,
            "maximum_atoms": arguments.maximum_atoms,
            "gamma_eigenvalue_scale": arguments.gamma_eigenvalue_scale,
            "element_tokens": torch.cat(element_tokens),
            "fractional_coordinates": torch.cat(fractional_coordinates),
            "lattice": torch.stack(lattices),
            "node_offsets": torch.tensor(node_offsets, dtype=torch.long),
            "split_index": torch.tensor(split_index, dtype=torch.uint8),
            "piezoelectric": torch.stack(piezoelectric),
            "piezoelectric_mask": torch.ones(len(audit_rows), dtype=torch.bool),
            "dielectric": torch.stack(dielectric),
            "dielectric_mask": torch.tensor(dielectric_mask, dtype=torch.bool),
            "elastic": torch.stack(elastic),
            "elastic_mask": torch.tensor(elastic_mask, dtype=torch.bool),
            "born_effective_charge": torch.cat(born),
            "born_mask": torch.cat(born_mask),
            "gamma_soft": torch.stack(gamma_soft),
            "gamma_log_magnitude": torch.stack(gamma_magnitude),
            "gamma_mask": torch.stack(gamma_mask),
            "internal_strain": torch.cat(internal_strain),
            "internal_strain_mask": torch.cat(internal_mask),
            "source_index": torch.zeros(len(audit_rows), dtype=torch.long),
        },
        cache_path,
    )
    audit_path = arguments.output / "audit_rows.jsonl"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        for audit_row in audit_rows:
            handle.write(json.dumps(asdict(audit_row), sort_keys=True) + "\n")
    exclusions_path = arguments.output / "exclusions.json"
    exclusions_path.write_text(
        json.dumps(excluded, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    label_counts = {
        "piezoelectric": len(audit_rows),
        "dielectric": sum(dielectric_mask),
        "born_graphs": sum(dielectric_mask),
        "gamma_graphs": sum(dielectric_mask),
        "internal_strain_graphs": sum(
            bool(value.any()) for value in internal_mask
        ),
        "internal_strain_components": sum(int(value.sum()) for value in internal_mask),
        "internal_strain_source_symmetric_within_rounding": sum(
            row.internal_strain_source_symmetric_within_rounding for row in audit_rows
        ),
        "elastic": sum(elastic_mask),
    }
    split_counts = Counter(row.split for row in audit_rows)
    manifest = {
        "schema": "gaugeflow.stage_d_jarvis_multitask.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualified": True,
        "maximum_atoms": arguments.maximum_atoms,
        "source_counts": {
            "tensororbit": 4998,
            "jarvis_dfpt": int(dfpt_manifest["cached"]),
            "strict_internal_strain": int(strain_manifest["accepted"]),
            "jarvis_elastic": int(elastic_audit["accepted"]),
        },
        "selected_split_counts": dict(split_counts),
        "excluded_count": len(excluded),
        "label_counts": label_counts,
        "reduced_composition_overlap": overlap,
        "tensororbit_manifest_sha256": sha256_file(
            arguments.tensororbit_root / "build_manifest.json"
        ),
        "tensororbit_split_sha256": tensor_manifest["split_sha256"],
        "dfpt_manifest_sha256": sha256_file(arguments.dfpt_root / "manifest.json"),
        "strain_completion_manifest_sha256": sha256_file(
            arguments.strain_completion_root / "manifest.json"
        ),
        "elastic_audit_sha256": sha256_file(arguments.elastic_root / "audit.json"),
        "elastic_targets_sha256": sha256_file(
            arguments.elastic_root / "accepted_targets_gpa.pt"
        ),
        "cache_sha256": sha256_file(cache_path),
        "audit_rows_sha256": sha256_file(audit_path),
        "exclusions_sha256": sha256_file(exclusions_path),
        "conventions": {
            "piezoelectric": TENSOR_ORBIT_CONVENTION,
            "dielectric": "symmetric static epsilon = electronic epsilon + ionic epsilon_ion",
            "born_effective_charge": DFPT_BORN_CONVENTION,
            "gamma": "sorted VASP dynamical-matrix eigenvalues; sign plus log1p absolute magnitude",
            "internal_strain": (
                DFPT_INTERNAL_STRAIN_CONVENTION
                + "; supervision only from schema-2 strict space-group/acoustic-null "
                "completion; pre-completion observed-block antisymmetry retained in audit_rows"
            ),
            "elastic": (
                "JARVIS elastic_total_kbar converted exactly once to GPa; "
                "full Cartesian point-group Reynolds target; engineering Voigt "
                "[xx,yy,zz,yz,xz,xy] -> orthonormal Kelvin -> C_ijkl"
            ),
            "missingness": "explicit masks; physical zero remains labelled",
        },
    }
    (arguments.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
