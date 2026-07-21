"""Memory-mapped, index-aligned per-atom teacher features for Stage-B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from gaugeflow.file_utils import load_json_object, sha256_file


TEACHER_FEATURE_CACHE_SCHEMA = 1


class MatPESTeacherFeatureCache:
    """Read a feature cache without exposing material identifiers to batches."""

    def __init__(
        self,
        root: str | Path,
        *,
        index_manifest: str | Path,
        verify_hashes: bool = True,
        require_qualified: bool = True,
    ) -> None:
        self.root = Path(root)
        manifest = load_json_object(self.root / "manifest.json")
        if manifest.get("schema") != TEACHER_FEATURE_CACHE_SCHEMA:
            raise ValueError("teacher feature cache schema mismatch")
        if require_qualified and not bool(manifest.get("qualified")):
            raise ValueError("teacher feature cache is not qualified")
        index_manifest_path = Path(index_manifest)
        if sha256_file(index_manifest_path) != manifest.get("index_manifest_sha256"):
            raise ValueError("teacher feature cache belongs to a different MatPES index")
        self.feature_dim = int(manifest.get("feature_dim", 0))
        self.row_count = int(manifest.get("row_count", 0))
        if self.feature_dim < 1 or self.row_count < 1:
            raise ValueError("teacher feature cache dimensions are invalid")
        offsets_path = self.root / str(manifest.get("offsets_file", ""))
        feature_path = self.root / str(manifest.get("features_file", ""))
        if verify_hashes and (
            sha256_file(offsets_path) != manifest.get("offsets_sha256")
            or sha256_file(feature_path) != manifest.get("features_sha256")
        ):
            raise ValueError("teacher feature cache file hash mismatch")
        payload = torch.load(offsets_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or payload.get("schema") != TEACHER_FEATURE_CACHE_SCHEMA:
            raise ValueError("teacher feature offset payload is invalid")
        self.offsets = payload.get("node_offsets")
        if (
            not isinstance(self.offsets, torch.Tensor)
            or self.offsets.dtype != torch.int64
            or self.offsets.shape != (self.row_count + 1,)
            or int(self.offsets[0]) != 0
            or bool((self.offsets[1:] < self.offsets[:-1]).any())
        ):
            raise ValueError("teacher feature offsets are invalid")
        feature_values = int(self.offsets[-1]) * self.feature_dim
        expected_bytes = feature_values * torch.tensor([], dtype=torch.float16).element_size()
        if feature_path.stat().st_size != expected_bytes:
            raise ValueError("teacher feature binary size disagrees with offsets")
        self._feature_path = feature_path
        self._feature_values = feature_values
        self._flat = torch.from_file(
            str(self._feature_path),
            shared=False,
            size=self._feature_values,
            dtype=torch.float16,
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_flat", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._flat = torch.from_file(
            str(self._feature_path),
            shared=False,
            size=self._feature_values,
            dtype=torch.float16,
        )

    def get(self, row: int, expected_nodes: int) -> torch.Tensor | None:
        if not 0 <= row < self.row_count or expected_nodes < 1:
            raise IndexError("teacher feature row or node count is invalid")
        start = int(self.offsets[row])
        stop = int(self.offsets[row + 1])
        if start == stop:
            return None
        if stop - start != expected_nodes:
            raise ValueError("teacher feature node count disagrees with MatPES index")
        return self._flat[start * self.feature_dim : stop * self.feature_dim].reshape(
            expected_nodes, self.feature_dim
        ).float()


def write_matpes_teacher_feature_cache(
    output: str | Path,
    rows: Iterable[tuple[int, torch.Tensor | None]],
    *,
    row_count: int,
    feature_dim: int,
    index_manifest: str | Path,
    teacher_manifest: str | Path,
    functional_scope: tuple[str, ...],
    expected_feature_rows: int,
    bounded_smoke: bool,
) -> dict[str, object]:
    """Write ordered per-row features directly to disk with bounded memory."""

    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite teacher feature cache {root}")
    if (
        row_count < 1
        or feature_dim < 1
        or not functional_scope
        or not 0 < expected_feature_rows <= row_count
    ):
        raise ValueError("teacher feature cache contract is empty")
    root.mkdir(parents=True, exist_ok=True)
    feature_path = root / "features.float16.bin"
    offsets = torch.zeros(row_count + 1, dtype=torch.int64)
    feature_rows = 0
    next_row = 0
    with feature_path.open("wb") as stream:
        for row, feature in rows:
            if row != next_row:
                raise ValueError("teacher feature rows must be complete and strictly ordered")
            if feature is not None:
                value = feature.detach().cpu().float()
                if value.ndim != 2 or value.shape[1] != feature_dim or not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError("teacher feature row has invalid shape or values")
                np.asarray(value.to(torch.float16).numpy()).tofile(stream)
                offsets[row + 1] = offsets[row] + value.shape[0]
                feature_rows += 1
            else:
                offsets[row + 1] = offsets[row]
            next_row += 1
    if next_row != row_count:
        raise ValueError("teacher feature cache did not receive every indexed row")
    offsets_path = root / "offsets.pt"
    torch.save(
        {"schema": TEACHER_FEATURE_CACHE_SCHEMA, "node_offsets": offsets},
        offsets_path,
    )
    index_manifest_path = Path(index_manifest)
    teacher_manifest_path = Path(teacher_manifest)
    if feature_rows != expected_feature_rows:
        raise ValueError("teacher feature cache coverage disagrees with its frozen scope")
    qualified = not bounded_smoke
    manifest: dict[str, object] = {
        "schema": TEACHER_FEATURE_CACHE_SCHEMA,
        "qualified": qualified,
        "scope": "type-matched per-atom TensorNet readout features; offline supervision only",
        "row_count": row_count,
        "feature_rows": feature_rows,
        "expected_feature_rows": expected_feature_rows,
        "feature_dim": feature_dim,
        "storage_dtype": "float16",
        "loss_contract": "per-atom cosine distance in float32",
        "functional_scope": list(functional_scope),
        "index_manifest": str(index_manifest_path.resolve()),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "teacher_manifest": str(teacher_manifest_path.resolve()),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "offsets_file": offsets_path.name,
        "offsets_sha256": sha256_file(offsets_path),
        "features_file": feature_path.name,
        "features_sha256": sha256_file(feature_path),
        "bounded_smoke": bounded_smoke,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
