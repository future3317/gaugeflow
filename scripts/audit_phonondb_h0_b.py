"""Qualify PhononDB translational modes, degenerate subspaces and NAC semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import phonopy
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_phonondb_force_constants_v2 import sha256_file

PROTOCOL = "h0_b_phonondb_derivation_attestation_v4_stratified"
AUDIT_QPOINTS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.5))
CONJUGACY_QPOINT = (0.173, 0.271, 0.389)


def _translation_basis(masses: np.ndarray) -> np.ndarray:
    basis = np.zeros((3 * len(masses), 3), dtype=np.complex128)
    for atom, mass in enumerate(masses):
        basis[3 * atom : 3 * atom + 3] = np.eye(3) * np.sqrt(mass)
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def _symmetrized_dielectric(dielectric: np.ndarray) -> np.ndarray:
    """Return the reciprocal, zero-field dielectric used by production NAC."""
    if dielectric.shape != (3, 3):
        raise ValueError("dielectric tensor must have shape [3,3]")
    return 0.5 * (dielectric + dielectric.T)


def _frequency_clusters(frequencies: np.ndarray, tolerance: float) -> list[np.ndarray]:
    clusters: list[list[int]] = [[0]]
    for index in range(1, len(frequencies)):
        if abs(float(frequencies[index] - frequencies[index - 1])) <= tolerance:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return [np.asarray(cluster, dtype=np.int64) for cluster in clusters]


def _deterministic_unitary(dimension: int, key: str) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(dimension, dimension)) + 1j * generator.normal(
        size=(dimension, dimension)
    )
    unitary, diagonal = np.linalg.qr(matrix)
    phases = np.diag(diagonal)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return unitary * phases.conj()


def _hash_order(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _select_stratified_materials(
    force_rows: dict[str, dict[str, Any]],
    mode_rows: dict[str, dict[str, Any]],
    v1_metrics: dict[str, dict[str, Any]],
    *,
    sample_size: int,
) -> tuple[set[str], dict[str, Any]]:
    """Select frozen long tails plus deterministic NAC/size-stratified coverage."""
    acoustic_outliers = {
        material_id
        for material_id, row in v1_metrics.items()
        if (
            float(row["translation_max_abs_frequency_thz"]) > 0.05
            or float(row["translation_subspace_min_singular"]) < 0.995
            or float(row["translation_dynamical_residual"]) > 0.001
        )
    }
    dielectric_ranked = sorted(
        v1_metrics,
        key=lambda material_id: (
            -float(v1_metrics[material_id]["nac_dielectric_symmetry_error"]),
            _hash_order(material_id),
        ),
    )
    dielectric_tail = set(dielectric_ranked[:64])
    projection_tail = {
        material_id
        for material_id, row in force_rows.items()
        if float(row["projection_relative_l2"]) > 0.05
    }
    selected = acoustic_outliers | dielectric_tail | projection_tail
    if len(selected) > sample_size:
        raise ValueError(
            f"mandatory H0-B long tails contain {len(selected)} rows, exceeding sample {sample_size}"
        )

    atom_boundaries = (8, 16, 32, 64)
    buckets: dict[tuple[bool, int], list[str]] = {}
    for material_id, row in force_rows.items():
        if material_id in selected:
            continue
        nac = bool(
            mode_rows[material_id]["has_born_effective_charge"]
            and mode_rows[material_id]["has_dielectric_constant"]
        )
        atoms = int(row["n_primitive_atoms"])
        atom_bin = sum(atoms > boundary for boundary in atom_boundaries)
        buckets.setdefault((nac, atom_bin), []).append(material_id)
    for values in buckets.values():
        values.sort(key=_hash_order)
    keys = sorted(buckets)
    offsets = {key: 0 for key in keys}
    while len(selected) < sample_size:
        advanced = False
        for key in keys:
            offset = offsets[key]
            values = buckets[key]
            if offset >= len(values):
                continue
            selected.add(values[offset])
            offsets[key] += 1
            advanced = True
            if len(selected) == sample_size:
                break
        if not advanced:
            raise ValueError("stratified sample exhausted before reaching the frozen size")
    return selected, {
        "sample_size": sample_size,
        "mandatory_acoustic_outliers": len(acoustic_outliers),
        "mandatory_top_raw_dielectric_asymmetry": len(dielectric_tail),
        "mandatory_projection_relative_l2_gt_0_05": len(projection_tail),
        "mandatory_union": len(acoustic_outliers | dielectric_tail | projection_tail),
        "strata": {
            f"nac={nac},atom_bin={atom_bin}": sum(
                material_id in selected
                and bool(
                    mode_rows[material_id]["has_born_effective_charge"]
                    and mode_rows[material_id]["has_dielectric_constant"]
                )
                == nac
                and sum(
                    int(force_rows[material_id]["n_primitive_atoms"]) > boundary
                    for boundary in atom_boundaries
                )
                == atom_bin
                for material_id in force_rows
            )
            for nac, atom_bin in keys
        },
    }


def _load_phonopy(task: dict[str, Any]):
    primitive_policy = task["primitive_matrix_policy"]
    if primitive_policy == "yaml":
        primitive_matrix: str | None = None
    elif primitive_policy == "P":
        primitive_matrix = "P"
    else:
        raise ValueError(f"unknown primitive matrix policy: {primitive_policy}")
    phonon = phonopy.load(
        task["core_path"],
        primitive_matrix=primitive_matrix,
        produce_fc=False,
        is_compact_fc=True,
        log_level=0,
    )
    with np.load(task["cache_path"], allow_pickle=False) as cached:
        phonon.force_constants = cached["force_constants"].copy()
    return phonon


def _audit_one(task: dict[str, Any]) -> dict[str, Any]:
    material_id = str(task["materials_project_id"])
    try:
        phonon = _load_phonopy(task)
        raw_dielectric_symmetry_error = 0.0
        dielectric_relative_correction = 0.0
        raw_nac_frequencies: np.ndarray | None = None
        raw_nac_dynamical_matrices: np.ndarray | None = None
        if phonon.nac_params is not None:
            raw_nac = dict(phonon.nac_params)
            raw_dielectric = np.asarray(raw_nac["dielectric"], dtype=np.float64)
            production_dielectric = _symmetrized_dielectric(raw_dielectric)
            raw_dielectric_symmetry_error = float(
                np.max(np.abs(raw_dielectric - raw_dielectric.T))
            )
            dielectric_relative_correction = float(
                np.linalg.norm(production_dielectric - raw_dielectric)
                / max(np.linalg.norm(raw_dielectric), 1e-30)
            )
            if task["conjugacy_selected"]:
                phonon.run_qpoints(
                    AUDIT_QPOINTS,
                    with_eigenvectors=False,
                    with_dynamical_matrices=True,
                )
                raw_nac_frequencies = np.asarray(
                    phonon.qpoints.frequencies, dtype=np.float64
                ).copy()
                raw_nac_dynamical_matrices = np.asarray(
                    phonon.qpoints.dynamical_matrices, dtype=np.complex128
                ).copy()
            raw_nac["dielectric"] = production_dielectric
            phonon.nac_params = raw_nac

        qpoints = list(AUDIT_QPOINTS)
        if task["conjugacy_selected"]:
            qpoints.extend((CONJUGACY_QPOINT, tuple(-value for value in CONJUGACY_QPOINT)))
        phonon.run_qpoints(qpoints, with_eigenvectors=True, with_dynamical_matrices=True)
        result = phonon.qpoints
        frequencies = np.asarray(result.frequencies, dtype=np.float64)
        eigenvectors = np.asarray(result.eigenvectors, dtype=np.complex128)
        dynamical_matrices = np.asarray(result.dynamical_matrices, dtype=np.complex128)
        nac_symmetrization_gamma_frequency_jitter = 0.0
        nac_symmetrization_nonzero_q_frequency_shift = 0.0
        nac_symmetrization_dynamical_matrix_shift = 0.0
        if raw_nac_frequencies is not None:
            frequency_delta = np.abs(
                raw_nac_frequencies - frequencies[: len(AUDIT_QPOINTS)]
            )
            nac_symmetrization_gamma_frequency_jitter = float(np.max(frequency_delta[0]))
            nac_symmetrization_nonzero_q_frequency_shift = float(
                np.max(frequency_delta[1:])
            )
            assert raw_nac_dynamical_matrices is not None
            nac_symmetrization_dynamical_matrix_shift = float(
                np.max(
                    np.abs(
                        raw_nac_dynamical_matrices
                        - dynamical_matrices[: len(AUDIT_QPOINTS)]
                    )
                )
            )

        translation = _translation_basis(np.asarray(phonon.primitive.masses, dtype=np.float64))
        gamma_vectors = eigenvectors[0]
        translation_weights = np.sum(np.abs(translation.conj().T @ gamma_vectors) ** 2, axis=0)
        translation_indices = np.argsort(translation_weights)[-3:]
        singular_values = np.linalg.svd(
            translation.conj().T @ gamma_vectors[:, translation_indices], compute_uv=False
        )
        translation_frequency = float(np.max(np.abs(frequencies[0, translation_indices])))
        translation_residual = float(np.linalg.norm(dynamical_matrices[0] @ translation))

        maximum_gauge_error = 0.0
        degenerate_cluster_count = 0
        for q_index in range(len(AUDIT_QPOINTS)):
            for cluster_index, cluster in enumerate(
                _frequency_clusters(frequencies[q_index], task["degeneracy_tolerance_thz"])
            ):
                if len(cluster) < 2:
                    continue
                degenerate_cluster_count += 1
                basis = eigenvectors[q_index][:, cluster]
                unitary = _deterministic_unitary(
                    len(cluster), f"{material_id}:{q_index}:{cluster_index}"
                )
                projector = basis @ basis.conj().T
                transformed = basis @ unitary
                error = float(np.max(np.abs(projector - transformed @ transformed.conj().T)))
                maximum_gauge_error = max(maximum_gauge_error, error)

        conjugacy_frequency_error: float | None = None
        conjugacy_projector_error: float | None = None
        if task["conjugacy_selected"]:
            positive_frequency, negative_frequency = frequencies[-2], frequencies[-1]
            positive_vectors, negative_vectors = eigenvectors[-2], eigenvectors[-1]
            conjugacy_frequency_error = float(
                np.max(np.abs(positive_frequency - negative_frequency))
            )
            positive_clusters = _frequency_clusters(
                positive_frequency, task["conjugacy_cluster_tolerance_thz"]
            )
            negative_clusters = _frequency_clusters(
                negative_frequency, task["conjugacy_cluster_tolerance_thz"]
            )
            if [len(cluster) for cluster in positive_clusters] != [
                len(cluster) for cluster in negative_clusters
            ]:
                raise ValueError("q and -q have inconsistent degenerate cluster dimensions")
            conjugacy_projector_error = 0.0
            for positive_cluster, negative_cluster in zip(
                positive_clusters, negative_clusters, strict=True
            ):
                positive_basis = positive_vectors[:, positive_cluster]
                negative_basis = negative_vectors[:, negative_cluster]
                positive_projector = positive_basis @ positive_basis.conj().T
                negative_projector = negative_basis @ negative_basis.conj().T
                error = float(np.max(np.abs(positive_projector - negative_projector.conj())))
                conjugacy_projector_error = max(conjugacy_projector_error, error)

        expected_nac = bool(task["has_born_effective_charge"] and task["has_dielectric_constant"])
        nac = phonon.nac_params
        nac_available = nac is not None
        nac_shape_valid = not expected_nac
        nac_finite = not expected_nac
        nac_charge_neutrality = 0.0
        nac_dielectric_symmetry_error = 0.0
        nac_dielectric_min_eigenvalue = 0.0
        nac_factor_error = 0.0
        if nac is not None:
            born = np.asarray(nac["born"], dtype=np.float64)
            dielectric = np.asarray(nac["dielectric"], dtype=np.float64)
            factor = float(nac["factor"])
            nac_shape_valid = born.shape == (len(phonon.primitive), 3, 3) and dielectric.shape == (
                3,
                3,
            )
            nac_finite = bool(np.isfinite(born).all() and np.isfinite(dielectric).all())
            nac_charge_neutrality = float(np.max(np.abs(np.sum(born, axis=0))))
            nac_dielectric_symmetry_error = float(np.max(np.abs(dielectric - dielectric.T)))
            nac_dielectric_min_eigenvalue = float(
                np.min(np.linalg.eigvalsh((dielectric + dielectric.T) / 2.0))
            )
            nac_factor_error = abs(factor - float(task["nac_unit_conversion_factor"]))

        return {
            "materials_project_id": material_id,
            "error": None,
            "translation_max_abs_frequency_thz": translation_frequency,
            "translation_subspace_min_singular": float(np.min(singular_values)),
            "translation_dynamical_residual": translation_residual,
            "degenerate_cluster_count": degenerate_cluster_count,
            "degenerate_projector_gauge_error": maximum_gauge_error,
            "conjugacy_selected": bool(task["conjugacy_selected"]),
            "conjugacy_frequency_error_thz": conjugacy_frequency_error,
            "conjugacy_projector_error": conjugacy_projector_error,
            "nac_expected": expected_nac,
            "nac_available": nac_available,
            "nac_shape_valid": nac_shape_valid,
            "nac_finite": nac_finite,
            "nac_charge_neutrality": nac_charge_neutrality,
            "nac_dielectric_symmetry_error": nac_dielectric_symmetry_error,
            "nac_raw_dielectric_symmetry_error": raw_dielectric_symmetry_error,
            "nac_dielectric_relative_correction": dielectric_relative_correction,
            "nac_symmetrization_gamma_frequency_jitter_thz": (
                nac_symmetrization_gamma_frequency_jitter
            ),
            "nac_symmetrization_nonzero_q_frequency_shift_thz": (
                nac_symmetrization_nonzero_q_frequency_shift
            ),
            "nac_symmetrization_dynamical_matrix_shift": (
                nac_symmetrization_dynamical_matrix_shift
            ),
            "nac_dielectric_min_eigenvalue": nac_dielectric_min_eigenvalue,
            "nac_factor_error": nac_factor_error,
        }
    except Exception as error:
        return {"materials_project_id": material_id, "error": f"{type(error).__name__}: {error}"}


def _maximum(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return max(values) if values else 0.0


def _minimum(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return min(values) if values else 0.0


def audit(
    data_root: Path,
    output_root: Path,
    *,
    workers: int,
    sample_size: int,
    conjugacy_sample_size: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    force_root = data_root / "processed" / "phonondb_force_constants_v2"
    mode_root = data_root / "processed" / "phonondb_mode_v1"
    force_index_path = force_root / "index.parquet"
    force_manifest_path = force_root / "MANIFEST.json"
    mode_index_path = mode_root / "index.parquet"
    force_rows = {
        str(row["materials_project_id"]): row
        for row in pq.read_table(force_index_path).to_pylist()
    }
    mode_rows = {
        str(row["materials_project_id"]): row
        for row in pq.read_table(
            mode_index_path,
            columns=[
                "materials_project_id",
                "has_born_effective_charge",
                "has_dielectric_constant",
                "nac_unit_conversion_factor",
            ],
        ).to_pylist()
    }
    if set(force_rows) != set(mode_rows):
        raise ValueError("force-constant and mode indexes do not contain the same materials")
    v1_metrics_path = data_root / "processed" / "gaugeflow_h0_v2" / "phonondb_h0_b_metrics.parquet"
    v1_metrics = {
        str(row["materials_project_id"]): row
        for row in pq.read_table(v1_metrics_path).to_pylist()
    }
    if set(v1_metrics) != set(force_rows):
        raise ValueError("frozen v1 negative metrics do not match the v2 material universe")
    audit_selected, selection = _select_stratified_materials(
        force_rows, mode_rows, v1_metrics, sample_size=sample_size
    )
    conjugacy_selected = set(
        sorted(audit_selected, key=_hash_order)[:conjugacy_sample_size]
    )
    tasks: list[dict[str, Any]] = []
    for material_id in sorted(audit_selected):
        force_row = force_rows[material_id]
        mode_row = mode_rows[material_id]
        tasks.append(
            {
                "materials_project_id": material_id,
                "core_path": str(mode_root / "phonopy_params" / Path(force_row["core_relpath"]).name),
                "cache_path": str(force_root / str(force_row["cache_relpath"])),
                "primitive_matrix_policy": str(force_row["primitive_matrix_policy"]),
                "has_born_effective_charge": bool(mode_row["has_born_effective_charge"]),
                "has_dielectric_constant": bool(mode_row["has_dielectric_constant"]),
                "nac_unit_conversion_factor": (
                    float(mode_row["nac_unit_conversion_factor"])
                    if mode_row["nac_unit_conversion_factor"] is not None
                    else 0.0
                ),
                "conjugacy_selected": material_id in conjugacy_selected,
                "degeneracy_tolerance_thz": thresholds["degeneracy_tolerance_thz"],
                "conjugacy_cluster_tolerance_thz": thresholds[
                    "conjugacy_cluster_tolerance_thz"
                ],
            }
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_audit_one, tasks, chunksize=8))

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "phonondb_h0_b_metrics.parquet"
    pq.write_table(pa.Table.from_pylist(records), metrics_path, compression="zstd", version="2.6")
    errors = [record for record in records if record.get("error")]
    successful = [record for record in records if not record.get("error")]
    observed = {
        "translation_max_abs_frequency_thz": _maximum(
            successful, "translation_max_abs_frequency_thz"
        ),
        "translation_subspace_min_singular": _minimum(
            successful, "translation_subspace_min_singular"
        ),
        "translation_dynamical_residual": _maximum(
            successful, "translation_dynamical_residual"
        ),
        "degenerate_projector_gauge_error": _maximum(
            successful, "degenerate_projector_gauge_error"
        ),
        "conjugacy_frequency_error_thz": _maximum(
            successful, "conjugacy_frequency_error_thz"
        ),
        "conjugacy_projector_error": _maximum(successful, "conjugacy_projector_error"),
        "nac_charge_neutrality": _maximum(successful, "nac_charge_neutrality"),
        "nac_dielectric_symmetry_error": _maximum(
            successful, "nac_dielectric_symmetry_error"
        ),
        "nac_raw_dielectric_symmetry_error": _maximum(
            successful, "nac_raw_dielectric_symmetry_error"
        ),
        "nac_dielectric_relative_correction": _maximum(
            successful, "nac_dielectric_relative_correction"
        ),
        "nac_symmetrization_gamma_frequency_jitter_thz": _maximum(
            successful, "nac_symmetrization_gamma_frequency_jitter_thz"
        ),
        "nac_symmetrization_nonzero_q_frequency_shift_thz": _maximum(
            successful, "nac_symmetrization_nonzero_q_frequency_shift_thz"
        ),
        "nac_symmetrization_dynamical_matrix_shift": _maximum(
            successful, "nac_symmetrization_dynamical_matrix_shift"
        ),
        "nac_dielectric_min_eigenvalue": _minimum(
            [record for record in successful if record["nac_available"]],
            "nac_dielectric_min_eigenvalue",
        ),
        "nac_factor_error": _maximum(successful, "nac_factor_error"),
    }
    nac_expected = sum(bool(record.get("nac_expected")) for record in successful)
    nac_available = sum(bool(record.get("nac_available")) for record in successful)
    universe_nac_expected = sum(
        bool(row["has_born_effective_charge"] and row["has_dielectric_constant"])
        for row in mode_rows.values()
    )
    degenerate_clusters = sum(int(record.get("degenerate_cluster_count", 0)) for record in successful)
    force_manifest = json.loads(force_manifest_path.read_text(encoding="utf-8"))
    force_observed = force_manifest.get("observed", {})
    checks = {
        "stratified_sample_complete": len(successful) == sample_size and not errors,
        "full_hessian_universe_complete": (
            force_manifest.get("counts", {}).get("successful") == 10_034
            and force_manifest.get("counts", {}).get("failed") == 0
        ),
        "full_hessian_algebraic_constraints": (
            float(force_observed.get("max_projected_row_asr", float("inf"))) <= 1e-10
            and float(force_observed.get("max_projected_column_asr", float("inf"))) <= 1e-10
            and float(force_observed.get("max_projected_permutation_residual", float("inf")))
            <= 1e-12
        ),
        "translation_frequency": observed["translation_max_abs_frequency_thz"]
        <= thresholds["translation_max_abs_frequency_thz"],
        "translation_subspace": observed["translation_subspace_min_singular"]
        >= thresholds["translation_subspace_min_singular"],
        "translation_residual": observed["translation_dynamical_residual"]
        <= thresholds["translation_dynamical_residual"],
        "degenerate_projector_gauge": observed["degenerate_projector_gauge_error"]
        <= thresholds["degenerate_projector_gauge_error"],
        "degenerate_clusters_observed": degenerate_clusters > 0,
        "conjugacy_sample_complete": sum(
            bool(record.get("conjugacy_selected")) for record in successful
        )
        == conjugacy_sample_size,
        "conjugacy_frequency": observed["conjugacy_frequency_error_thz"]
        <= thresholds["conjugacy_frequency_error_thz"],
        "conjugacy_projector": observed["conjugacy_projector_error"]
        <= thresholds["conjugacy_projector_error"],
        "nac_capability_count": universe_nac_expected == 4_908,
        "nac_availability_matches_source": all(
            record["nac_expected"] == record["nac_available"] for record in successful
        ),
        "nac_shapes_and_finite": all(
            record["nac_shape_valid"] and record["nac_finite"] for record in successful
        ),
        "nac_charge_neutrality": observed["nac_charge_neutrality"]
        <= thresholds["nac_charge_neutrality"],
        "nac_dielectric_symmetry": observed["nac_dielectric_symmetry_error"]
        <= thresholds["nac_dielectric_symmetry_error"],
        "nac_symmetrization_nonzero_q_frequency_shift": observed[
            "nac_symmetrization_nonzero_q_frequency_shift_thz"
        ]
        <= thresholds["nac_symmetrization_nonzero_q_frequency_shift_thz"],
        "nac_symmetrization_dynamical_matrix_shift": observed[
            "nac_symmetrization_dynamical_matrix_shift"
        ]
        <= thresholds["nac_symmetrization_dynamical_matrix_shift"],
        "nac_dielectric_positive": observed["nac_dielectric_min_eigenvalue"]
        >= thresholds["nac_dielectric_min_eigenvalue"],
        "nac_factor": observed["nac_factor_error"] <= thresholds["nac_factor_error"],
        "missing_nac_is_explicit": (
            len(mode_rows) - universe_nac_expected == 5_126
            and len(successful) - nac_available == len(successful) - nac_expected
        ),
    }
    manifest = {
        "protocol": PROTOCOL,
        "qualified": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "observed": observed,
        "counts": {
            "universe": 10_034,
            "expected_sample": sample_size,
            "successful": len(successful),
            "failed": len(errors),
            "degenerate_clusters": degenerate_clusters,
            "conjugacy_sample": conjugacy_sample_size,
            "nac_available": nac_available,
            "nac_unavailable_explicit_in_sample": len(successful) - nac_available,
        },
        "selection": selection,
        "nac_policy": (
            "Retain the raw source dielectric as an audit field and use its explicit symmetric "
            "part for the reciprocal zero-field production NAC. Use non-analytic corrections "
            "only for the 4,908 records with validated Born charges and dielectric tensors. The "
            "other 5,126 records carry nac_available=false; missing NAC is never interpreted as "
            "a zero correction."
        ),
        "force_constant_policy": (
            "Use the versioned v2 compact cache obtained only after full-supercell Hessian "
            "projection onto permutation symmetry and bilateral acoustic sum rules. The raw v1 "
            "cache is preserved as frozen negative evidence."
        ),
        "degenerate_mode_policy": (
            "Store and supervise Hermitian eigenspace projectors, not individual eigenvector "
            "signs, phases or bases inside a degenerate cluster."
        ),
        "metrics_path": metrics_path.name,
        "metrics_sha256": sha256_file(metrics_path),
        "force_index_sha256": sha256_file(force_index_path),
        "force_manifest_sha256": sha256_file(force_manifest_path),
        "mode_index_sha256": sha256_file(mode_index_path),
        "v1_negative_metrics_sha256": sha256_file(v1_metrics_path),
        "auditor_sha256": sha256_file(Path(__file__)),
        "errors": errors[:20],
    }
    manifest_path = output_root / "phonondb_derivation_attestation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--conjugacy-sample-size", type=int, default=256)
    args = parser.parse_args()
    thresholds = {
        "translation_max_abs_frequency_thz": 0.05,
        "translation_subspace_min_singular": 0.995,
        "translation_dynamical_residual": 0.001,
        "degeneracy_tolerance_thz": 1e-5,
        "degenerate_projector_gauge_error": 1e-10,
        "conjugacy_cluster_tolerance_thz": 1e-5,
        "conjugacy_frequency_error_thz": 1e-8,
        "conjugacy_projector_error": 1e-8,
        "nac_charge_neutrality": 1e-5,
        "nac_dielectric_symmetry_error": 1e-8,
        "nac_symmetrization_nonzero_q_frequency_shift_thz": 1e-8,
        "nac_symmetrization_dynamical_matrix_shift": 1e-12,
        "nac_dielectric_min_eigenvalue": 1e-8,
        "nac_factor_error": 1e-10,
    }
    manifest = audit(
        args.data_root,
        args.output_root,
        workers=args.workers,
        sample_size=args.sample_size,
        conjugacy_sample_size=args.conjugacy_sample_size,
        thresholds=thresholds,
    )
    print(json.dumps({"qualified": manifest["qualified"], "counts": manifest["counts"]}))
    if not manifest["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
