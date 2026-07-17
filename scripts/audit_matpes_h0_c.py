"""Qualify two frozen, architecture-distinct MatPES-PBE teachers for H0-C."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_matpes_h0_c import file_sha256


def _runtime_identity(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Compare the live CUDA environment with the frozen runtime contract."""
    import torch

    distribution_names = {
        "torch": "torch",
        "matgl": "matgl",
        "ase": "ase",
        "huggingface_hub": "huggingface-hub",
        "torch_geometric": "torch-geometric",
        "pymatgen_core": "pymatgen-core",
        "numpy": "numpy",
    }
    observed = {
        "python_version": platform.python_version(),
        **{key: version(name) for key, name in distribution_names.items()},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    required = {
        key: config[key]
        for key in ("python_version", *distribution_names, "device", "gpu")
    }
    return observed == required, {"required": required, "observed": observed}


def _checkpoint_files_match(teacher_root: Path, manifest: dict[str, Any]) -> bool:
    """Rehash every frozen checkpoint file immediately before inference."""
    for role, teacher in manifest.get("teachers", {}).items():
        local_dir = teacher_root / str(teacher.get("local_dir", role))
        for record in teacher.get("files", []):
            path = local_dir / str(record["path"])
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                return False
    return bool(manifest.get("teachers"))


def voigt_kbar_to_full_gpa(stress: list[float] | np.ndarray) -> np.ndarray:
    """Convert MatPES [xx,yy,zz,yz,xz,xy] kbar to tensile-positive GPa."""
    xx, yy, zz, yz, xz, xy = np.asarray(stress, dtype=np.float64)
    return -0.1 * np.asarray(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64
    )


def full_to_voigt(stress: np.ndarray) -> np.ndarray:
    tensor = np.asarray(stress, dtype=np.float64)
    if tensor.shape != (3, 3):
        raise ValueError("stress tensor must have shape [3,3]")
    return tensor[(0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)]


def axis_angle_rotation(axis: list[float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) or np.linalg.det(
        rotation
    ) < 0.0:
        raise ValueError("configured rotation is not in SO(3)")
    return rotation


def select_held_out_rows(
    path: Path, *, protocol: str, sample_size: int
) -> tuple[list[dict[str, Any]], int]:
    """Select a prediction-independent deterministic hash sample from JSONL."""
    heap: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            material_id = str(row["matpes_id"])
            if material_id in seen:
                raise ValueError(f"duplicate MatPES ID in held-out split: {material_id}")
            seen.add(material_id)
            score = int(hashlib.sha256(f"{protocol}:{material_id}".encode()).hexdigest(), 16)
            item = (-score, material_id, row)
            if len(heap) < sample_size:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
            count += 1
    selected = [item[2] for item in heap]
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{protocol}:{row['matpes_id']}".encode()
        ).hexdigest()
    )
    return selected, count


def _predict(calculator: Any, atoms: Any) -> dict[str, np.ndarray | float]:
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    energy = float(calculator.results["energy"])
    forces = np.asarray(calculator.results["forces"], dtype=np.float64)
    stress = np.asarray(calculator.results["stress"], dtype=np.float64)
    if stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        stress = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    if forces.ndim != 2 or forces.shape[1] != 3 or stress.shape != (3, 3):
        raise ValueError("teacher returned an invalid force or stress shape")
    return {"energy": energy, "forces": forces, "stress": stress}


def _atoms_from_row(row: dict[str, Any]) -> Any:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(Structure.from_dict(row["structure"]))


def _load_calculator(model_dir: Path, device: str) -> tuple[Any, Any]:
    import matgl
    from matgl.ext.ase import PESCalculator

    potential = matgl.load_model(model_dir).to(device)
    potential.eval()
    return potential, PESCalculator(potential, stress_unit="GPa", use_voigt=False)


def _evaluate_teacher(
    role: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    *,
    device: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    import torch

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()
    potential, calculator = _load_calculator(model_dir, device)
    predictions: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    feature_contract: dict[str, Any] = {
        "available": False,
        "source": None,
        "shape": None,
    }
    for row_index, row in enumerate(rows, start=1):
        material_id = str(row["matpes_id"])
        try:
            atoms = _atoms_from_row(row)
            prediction = _predict(calculator, atoms)
            arrays = [
                np.asarray(prediction["energy"]),
                np.asarray(prediction["forces"]),
                np.asarray(prediction["stress"]),
            ]
            if not all(np.isfinite(value).all() for value in arrays):
                raise ValueError("non-finite prediction")
            predictions[material_id] = prediction
            if role == "primary" and not feature_contract["available"]:
                features = potential.model.feature_dict.get("readout")
                if features is None:
                    raise ValueError("TensorNet did not expose feature_dict['readout']")
                feature_contract = {
                    "available": True,
                    "source": "Potential.model.feature_dict['readout']",
                    "shape": list(features.shape),
                    "per_atom": int(features.shape[0]) == len(atoms),
                    "finite": bool(torch.isfinite(features).all().item()),
                }
        except Exception as error:
            failures.append(
                {
                    "role": role,
                    "matpes_id": material_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if row_index % 64 == 0 or row_index == len(rows):
            print(
                json.dumps(
                    {
                        "stage": "held_out_inference",
                        "role": role,
                        "completed": row_index,
                        "total": len(rows),
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    metadata = {
        "model_class": type(potential.model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in potential.parameters()),
        "cutoff_angstrom": float(potential.model.cutoff),
        "element_types": list(potential.model.element_types),
        "feature_contract": feature_contract,
        "peak_cuda_memory_bytes": peak_memory,
    }
    del calculator, potential
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, metadata, failures


def _metric_summary(
    rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    energy_errors: list[float] = []
    force_errors: list[np.ndarray] = []
    stress_errors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        material_id = str(row["matpes_id"])
        if material_id not in predictions:
            continue
        prediction = predictions[material_id]
        n_atoms = int(row["nsites"])
        energy_error = abs(float(prediction["energy"]) - float(row["energy"])) / n_atoms
        force_error = np.abs(
            np.asarray(prediction["forces"]) - np.asarray(row["forces"], dtype=np.float64)
        )
        target_stress = voigt_kbar_to_full_gpa(row["stress"])
        stress_error = np.abs(
            full_to_voigt(np.asarray(prediction["stress"])) - full_to_voigt(target_stress)
        )
        energy_errors.append(energy_error)
        force_errors.append(force_error.reshape(-1))
        stress_errors.append(stress_error)
        records.append(
            {
                "matpes_id": material_id,
                "n_atoms": n_atoms,
                "energy_abs_error_ev_per_atom": energy_error,
                "force_component_mae_ev_per_angstrom": float(force_error.mean()),
                "stress_voigt_mae_gpa": float(stress_error.mean()),
            }
        )
    return (
        {
            "energy_mae_ev_per_atom": float(np.mean(energy_errors)),
            "force_mae_ev_per_angstrom": float(np.mean(np.concatenate(force_errors))),
            "stress_mae_gpa": float(np.mean(np.concatenate(stress_errors))),
        },
        records,
    )


def _transformed_atoms(atoms: Any, kind: str, config: dict[str, Any]) -> tuple[Any, Any]:
    transformed = atoms.copy()
    if kind == "translation":
        transformed.positions += np.asarray(config["translation_cartesian_angstrom"])
        return transformed, None
    if kind == "rotation":
        rotation = axis_angle_rotation(
            config["proper_rotation_axis"], float(config["proper_rotation_angle_radians"])
        )
        transformed.set_cell(np.asarray(atoms.cell) @ rotation.T, scale_atoms=False)
        transformed.set_positions(np.asarray(atoms.positions) @ rotation.T)
        return transformed, rotation
    if kind == "permutation":
        permutation = np.arange(len(atoms) - 1, -1, -1)
        return atoms[permutation], permutation
    if kind == "cell_basis":
        basis = np.asarray(config["unimodular_cell_basis"], dtype=np.int64)
        determinant = round(float(np.linalg.det(basis)))
        if abs(determinant) != 1:
            raise ValueError("cell-basis transform is not unimodular")
        transformed.set_cell(basis @ np.asarray(atoms.cell), scale_atoms=False)
        transformed.wrap()
        return transformed, basis
    raise ValueError(f"unknown invariance transform: {kind}")


def _invariance_audit(
    model_dir: Path,
    rows: list[dict[str, Any]],
    base: dict[str, dict[str, Any]],
    *,
    device: str,
    transforms: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]], list[dict[str, str]]]:
    import torch

    potential, calculator = _load_calculator(model_dir, device)
    observed = {
        "rigid_energy_error_ev_per_atom": 0.0,
        "rigid_force_covariance_rmse_ev_per_angstrom": 0.0,
        "rigid_stress_covariance_rmse_gpa": 0.0,
        "cell_basis_energy_error_ev_per_atom": 0.0,
        "cell_basis_force_rmse_ev_per_angstrom": 0.0,
        "cell_basis_stress_rmse_gpa": 0.0,
    }
    details: dict[str, dict[str, Any]] = {
        kind: {
            "energy_error_ev_per_atom": 0.0,
            "energy_max_id": None,
            "force_rmse_ev_per_angstrom": 0.0,
            "force_max_id": None,
            "stress_rmse_gpa": 0.0,
            "stress_max_id": None,
        }
        for kind in ("translation", "rotation", "permutation", "cell_basis")
    }
    failures: list[dict[str, str]] = []
    for row in rows:
        material_id = str(row["matpes_id"])
        if material_id not in base:
            continue
        atoms = _atoms_from_row(row)
        reference = base[material_id]
        for kind in ("translation", "rotation", "permutation", "cell_basis"):
            try:
                transformed, action = _transformed_atoms(atoms, kind, transforms)
                prediction = _predict(calculator, transformed)
                expected_forces = np.asarray(reference["forces"])
                expected_stress = np.asarray(reference["stress"])
                if kind == "rotation":
                    expected_forces = expected_forces @ action.T
                    expected_stress = action @ expected_stress @ action.T
                elif kind == "permutation":
                    expected_forces = expected_forces[action]
                energy_error = abs(
                    float(prediction["energy"]) - float(reference["energy"])
                ) / len(atoms)
                force_error = float(
                    np.sqrt(np.mean((np.asarray(prediction["forces"]) - expected_forces) ** 2))
                )
                stress_error = float(
                    np.sqrt(np.mean((np.asarray(prediction["stress"]) - expected_stress) ** 2))
                )
                for metric_name, identifier_name, metric_value in (
                    ("energy_error_ev_per_atom", "energy_max_id", energy_error),
                    ("force_rmse_ev_per_angstrom", "force_max_id", force_error),
                    ("stress_rmse_gpa", "stress_max_id", stress_error),
                ):
                    if metric_value > details[kind][metric_name]:
                        details[kind][metric_name] = metric_value
                        details[kind][identifier_name] = material_id
                prefix = "cell_basis" if kind == "cell_basis" else "rigid"
                observed[f"{prefix}_energy_error_ev_per_atom"] = max(
                    observed[f"{prefix}_energy_error_ev_per_atom"], energy_error
                )
                force_key = (
                    "cell_basis_force_rmse_ev_per_angstrom"
                    if kind == "cell_basis"
                    else "rigid_force_covariance_rmse_ev_per_angstrom"
                )
                stress_key = (
                    "cell_basis_stress_rmse_gpa"
                    if kind == "cell_basis"
                    else "rigid_stress_covariance_rmse_gpa"
                )
                observed[force_key] = max(observed[force_key], force_error)
                observed[stress_key] = max(observed[stress_key], stress_error)
            except Exception as error:
                failures.append(
                    {
                        "matpes_id": material_id,
                        "transform": kind,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    del calculator, potential
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return observed, details, failures


def _disagreement_metrics(
    rows: list[dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    disagreement: dict[str, dict[str, Any]],
) -> dict[str, float]:
    energy: list[float] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    for row in rows:
        material_id = str(row["matpes_id"])
        if material_id not in primary or material_id not in disagreement:
            continue
        left, right = primary[material_id], disagreement[material_id]
        energy.append(abs(float(left["energy"]) - float(right["energy"])) / int(row["nsites"]))
        forces.append(
            np.abs(np.asarray(left["forces"]) - np.asarray(right["forces"])).reshape(-1)
        )
        stresses.append(
            np.abs(
                full_to_voigt(np.asarray(left["stress"]))
                - full_to_voigt(np.asarray(right["stress"]))
            )
        )
    return {
        "energy_mae_ev_per_atom": float(np.mean(energy)),
        "force_mae_ev_per_angstrom": float(np.mean(np.concatenate(forces))),
        "stress_mae_gpa": float(np.mean(np.concatenate(stresses))),
    }


def render_report(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    lines = [
        "# H0-C MatPES teacher qualification",
        "",
        "## Decision",
        "",
        f"`{'H0-C qualified' if manifest['qualified'] else 'H0-C failed'}` under "
        f"`{manifest['protocol']}`. H1 remains unauthorized because H0-D/E are not qualified.",
        "",
        "## Held-out metrics",
        "",
        "| Teacher | Energy MAE (eV/atom) | Force MAE (eV/A) | Stress MAE (GPa) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role in ("primary", "disagreement"):
        item = metrics[role]
        lines.append(
            f"| {role} | {item['energy_mae_ev_per_atom']:.6g} | "
            f"{item['force_mae_ev_per_angstrom']:.6g} | {item['stress_mae_gpa']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Scientific contract",
            "",
            "The primary TensorNet and architecture-distinct "
            f"{manifest['teacher_metadata']['disagreement']['model_class']} checkpoints are pinned to "
            "exact Hugging Face commits and evaluated only on a deterministic held-out test "
            "sample. Source stress is converted from compressive-positive kbar Voigt form to "
            "tensile-positive GPa. Teachers are authorized only for offline labels, filtering, "
            "representation extraction and auxiliary PES losses; they are not reverse-sampling "
            "guidance and are not independent DFT validation.",
            "",
            "## Checks",
            "",
            *[f"- `{name}`: `{value}`" for name, value in manifest["checks"].items()],
            "",
        ]
    )
    return "\n".join(lines)


def audit(config_path: Path, data_root: Path, teacher_root: Path, output_root: Path) -> dict[str, Any]:
    import torch

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = data_root / config["dataset"]["path"]
    if file_sha256(dataset_path) != config["dataset"]["sha256"]:
        raise ValueError("held-out MatPES test split hash does not match the frozen protocol")
    rows, total_rows = select_held_out_rows(
        dataset_path,
        protocol=config["selection"]["selection_seed_string"],
        sample_size=int(config["selection"]["sample_size"]),
    )
    if total_rows != config["dataset"]["rows"]:
        raise ValueError(f"expected {config['dataset']['rows']} test rows, found {total_rows}")
    checkpoint_manifest_path = teacher_root / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    runtime_matches, runtime_identity = _runtime_identity(config["runtime"])
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    metric_records: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    invariance: dict[str, Any] = {}
    invariance_details: dict[str, Any] = {}
    invariance_rows = rows[: int(config["selection"]["invariance_sample_size"])]
    for role in ("primary", "disagreement"):
        role_predictions, role_metadata, role_failures = _evaluate_teacher(
            role, teacher_root / role, rows, device=config["runtime"]["device"]
        )
        predictions[role] = role_predictions
        metadata[role] = role_metadata
        failures.extend(role_failures)
        role_metrics, role_records = _metric_summary(rows, role_predictions)
        metrics[role] = role_metrics
        for record in role_records:
            metric_records.append({"teacher": role, **record})
        role_invariance, role_invariance_details, invariance_failures = _invariance_audit(
            teacher_root / role,
            invariance_rows,
            role_predictions,
            device=config["runtime"]["device"],
            transforms=config["invariance_transforms"],
        )
        invariance[role] = role_invariance
        invariance_details[role] = role_invariance_details
        failures.extend({"role": role, **record} for record in invariance_failures)
    metrics["teacher_disagreement"] = _disagreement_metrics(
        rows, predictions["primary"], predictions["disagreement"]
    )
    thresholds = config["thresholds"]
    primary = metrics["primary"]
    other = metrics["disagreement"]
    rigid_keys = (
        "rigid_energy_error_ev_per_atom",
        "rigid_force_covariance_rmse_ev_per_angstrom",
        "rigid_stress_covariance_rmse_gpa",
        "cell_basis_energy_error_ev_per_atom",
        "cell_basis_force_rmse_ev_per_angstrom",
        "cell_basis_stress_rmse_gpa",
    )
    checks = {
        "snapshot_identity": checkpoint_manifest.get("qualified_snapshot_identity") is True,
        "snapshot_protocol": checkpoint_manifest.get("protocol") == config["protocol"],
        "snapshot_config": checkpoint_manifest.get("config_sha256")
        == file_sha256(config_path),
        "snapshot_files": _checkpoint_files_match(teacher_root, checkpoint_manifest),
        "runtime_identity": runtime_matches,
        "dataset_count": total_rows == config["dataset"]["rows"],
        "sample_complete": all(len(predictions[role]) == len(rows) for role in predictions),
        "sampling_failures": len(failures) == thresholds["sampling_failures"],
        "nonfinite_predictions": all(
            all(
                np.isfinite(np.asarray(value)).all()
                for prediction in predictions[role].values()
                for value in prediction.values()
            )
            for role in predictions
        ),
        "primary_energy": primary["energy_mae_ev_per_atom"]
        <= thresholds["primary_energy_mae_ev_per_atom"],
        "primary_force": primary["force_mae_ev_per_angstrom"]
        <= thresholds["primary_force_mae_ev_per_angstrom"],
        "primary_stress": primary["stress_mae_gpa"] <= thresholds["primary_stress_mae_gpa"],
        "disagreement_energy": other["energy_mae_ev_per_atom"]
        <= thresholds["disagreement_energy_mae_ev_per_atom"],
        "disagreement_force": other["force_mae_ev_per_angstrom"]
        <= thresholds["disagreement_force_mae_ev_per_angstrom"],
        "disagreement_stress": other["stress_mae_gpa"]
        <= thresholds["disagreement_stress_mae_gpa"],
        "energy_disagreement_nonzero": metrics["teacher_disagreement"][
            "energy_mae_ev_per_atom"
        ]
        >= thresholds["minimum_energy_disagreement_mae_ev_per_atom"],
        "force_disagreement_nonzero": metrics["teacher_disagreement"][
            "force_mae_ev_per_angstrom"
        ]
        >= thresholds["minimum_force_disagreement_mae_ev_per_angstrom"],
        "architectures_distinct": metadata["primary"]["model_class"]
        != metadata["disagreement"]["model_class"],
        "architectures_declared": all(
            metadata[role]["model_class"] == config["teachers"][role]["architecture"]
            for role in metadata
        ),
        "atom_feature_layer": all(
            (
                metadata["primary"]["feature_contract"].get("available") is True,
                metadata["primary"]["feature_contract"].get("per_atom") is True,
                metadata["primary"]["feature_contract"].get("finite") is True,
            )
        ),
        "invariance": all(
            invariance[role][key] <= thresholds[key]
            for role in invariance
            for key in rigid_keys
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "matpes_teacher_metrics.parquet"
    pq.write_table(pa.Table.from_pylist(metric_records), metrics_path, compression="zstd")
    manifest = {
        "protocol": config["protocol"],
        "qualified": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "counts": {
            "test_rows": total_rows,
            "selected": len(rows),
            "invariance_selected": len(invariance_rows),
            "failures": len(failures),
        },
        "selection": {
            "method": config["selection"]["method"],
            "selection_seed_string": config["selection"]["selection_seed_string"],
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(str(row["matpes_id"]) for row in rows).encode()
            ).hexdigest(),
        },
        "metrics": metrics,
        "invariance": invariance,
        "invariance_details": invariance_details,
        "teacher_metadata": metadata,
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest_path),
        "dataset_sha256": file_sha256(dataset_path),
        "metrics_path": metrics_path.name,
        "metrics_sha256": file_sha256(metrics_path),
        "config_sha256": file_sha256(config_path),
        "auditor_sha256": file_sha256(Path(__file__)),
        "runtime": {
            "identity": runtime_identity,
            "python_packages": {
                name: version(name)
                for name in (
                    "torch",
                    "matgl",
                    "ase",
                    "huggingface-hub",
                    "torch-geometric",
                    "numpy",
                    "pymatgen-core",
                )
            },
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "failures": failures[:50],
        "usage_policy": (
            "Frozen offline PES labels, uncertainty filtering, representation extraction and "
            "auxiliary losses only. Never reverse-sampling guidance and never independent DFT validation."
        ),
    }
    manifest_path = output_root / "matpes_teacher_qualification.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "matpes_teacher_qualification.md").write_text(
        render_report(manifest), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = audit(args.config, args.data_root, args.teacher_root, args.output_root)
    print(
        json.dumps(
            {
                "qualified": manifest["qualified"],
                "counts": manifest["counts"],
                "metrics": manifest["metrics"],
            }
        )
    )
    if not manifest["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
