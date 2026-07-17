"""Train-split standard coordinates for the H1a P1 lattice process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class P1LatticeStandardizer(nn.Module):
    """Invertible standardization of volume and trace-free log shape.

    Volume is residualized against ``log(N)`` before scaling. Shape uses the
    five active principal directions of the P1 trace-free Kelvin chart. The
    statistics are fitted on the training split only and travel in checkpoint
    metadata; validation/test structures never affect them.
    """

    def __init__(
        self,
        *,
        volume_residual_mean: float,
        volume_residual_std: float,
        shape_mean: torch.Tensor,
        shape_basis_columns: torch.Tensor,
        shape_scales: torch.Tensor,
    ) -> None:
        super().__init__()
        if volume_residual_std <= 0.0:
            raise ValueError("volume residual standard deviation must be positive")
        if shape_mean.shape != (6,) or shape_basis_columns.shape != (6, 5):
            raise ValueError("P1 shape standardization requires mean [6] and basis [6,5]")
        if shape_scales.shape != (5,) or bool((shape_scales <= 0.0).any()):
            raise ValueError("P1 shape scales must be five positive values")
        if not all(
            torch.isfinite(value).all()
            for value in (shape_mean, shape_basis_columns, shape_scales)
        ):
            raise ValueError("lattice standardization contains nonfinite values")
        gram = shape_basis_columns.T @ shape_basis_columns
        if not torch.allclose(
            gram, torch.eye(5, dtype=gram.dtype, device=gram.device), atol=2e-6, rtol=2e-6
        ):
            raise ValueError("shape whitening basis is not orthonormal")
        self.volume_residual_mean = float(volume_residual_mean)
        self.volume_residual_std = float(volume_residual_std)
        self.register_buffer("shape_mean", shape_mean.detach().to(torch.float64))
        self.register_buffer(
            "shape_basis_columns", shape_basis_columns.detach().to(torch.float64)
        )
        self.register_buffer("shape_scales", shape_scales.detach().to(torch.float64))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "P1LatticeStandardizer":
        if value.get("protocol") != "h1a_p1_lattice_standardization":
            raise ValueError("unexpected lattice-standardization protocol")
        return cls(
            volume_residual_mean=float(value["volume_residual_mean"]),
            volume_residual_std=float(value["volume_residual_std"]),
            shape_mean=torch.tensor(value["shape_mean"], dtype=torch.float64),
            shape_basis_columns=torch.tensor(
                value["shape_basis_columns"], dtype=torch.float64
            ),
            shape_scales=torch.tensor(value["shape_scales"], dtype=torch.float64),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "P1LatticeStandardizer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("lattice-standardization file must contain one object")
        return cls.from_mapping(value)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "protocol": "h1a_p1_lattice_standardization",
            "volume_residual_mean": self.volume_residual_mean,
            "volume_residual_std": self.volume_residual_std,
            "shape_mean": self.shape_mean.tolist(),
            "shape_basis_columns": self.shape_basis_columns.tolist(),
            "shape_scales": self.shape_scales.tolist(),
        }

    def encode_volume(self, log_volume: torch.Tensor, node_counts: torch.Tensor) -> torch.Tensor:
        counts = node_counts.to(log_volume)
        if log_volume.shape != counts.shape or bool((counts < 1).any()):
            raise ValueError("volume standardization requires one positive count per graph")
        return (
            log_volume - counts.log() - self.volume_residual_mean
        ) / self.volume_residual_std

    def decode_volume(self, latent: torch.Tensor, node_counts: torch.Tensor) -> torch.Tensor:
        counts = node_counts.to(latent)
        if latent.shape != counts.shape or bool((counts < 1).any()):
            raise ValueError("volume decoding requires one positive count per graph")
        return (
            latent * self.volume_residual_std
            + counts.log()
            + self.volume_residual_mean
        )

    def encode_shape(self, log_shape: torch.Tensor) -> torch.Tensor:
        if log_shape.ndim != 2 or log_shape.shape[-1] != 6:
            raise ValueError("shape standardization requires [graphs,6]")
        centered = log_shape - self.shape_mean.to(log_shape)
        return (centered @ self.shape_basis_columns.to(log_shape)) / self.shape_scales.to(
            log_shape
        )

    def decode_shape(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != 5:
            raise ValueError("shape decoding requires [graphs,5]")
        return self.shape_mean.to(latent) + (
            latent * self.shape_scales.to(latent)
        ) @ self.shape_basis_columns.to(latent).T
