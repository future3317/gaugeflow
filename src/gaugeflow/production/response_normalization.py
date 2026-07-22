"""Train-only covariant normalization for Stage-D response targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .response_multitask import ResponsePredictions, ResponseTargets


def _radial_asinh(value: torch.Tensor) -> torch.Tensor:
    flattened = value.flatten(1)
    radius = (flattened.square().mean(dim=-1) + value.new_tensor(1e-12)).sqrt()
    scale = torch.asinh(radius) / radius
    return value * scale.reshape((-1,) + (1,) * (value.ndim - 1))


def _radial_sinh(value: torch.Tensor) -> torch.Tensor:
    flattened = value.flatten(1)
    radius = (flattened.square().mean(dim=-1) + value.new_tensor(1e-12)).sqrt()
    scale = torch.sinh(radius) / radius
    return value * scale.reshape((-1,) + (1,) * (value.ndim - 1))


@dataclass(frozen=True)
class ResponseNormalizer:
    piezoelectric_scale: torch.Tensor
    dielectric_isotropic_location: torch.Tensor
    dielectric_scale: torch.Tensor
    elastic_scale: torch.Tensor
    born_isotropic_location: torch.Tensor
    born_scale: torch.Tensor
    gamma_log_location: torch.Tensor
    gamma_log_scale: torch.Tensor
    internal_strain_scale: torch.Tensor

    @property
    def source_count(self) -> int:
        return int(self.piezoelectric_scale.numel())

    def to(self, device: torch.device | str) -> ResponseNormalizer:
        return ResponseNormalizer(
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
            }
        )

    def normalize(
        self,
        target: ResponseTargets,
        source_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> ResponseTargets:
        if source_index.ndim != 1 or source_index.dtype != torch.long:
            raise ValueError("response source index must be a one-dimensional integer tensor")
        if batch.ndim != 1 or batch.dtype != torch.long:
            raise ValueError("response node batch must be a one-dimensional integer tensor")
        if int(source_index.min()) < 0 or int(source_index.max()) >= self.source_count:
            raise ValueError("response source index lies outside the normalizer vocabulary")
        graph_source = source_index
        node_source = source_index[batch]
        identity = torch.eye(3, dtype=target.dielectric.dtype, device=target.dielectric.device)
        dielectric = _radial_asinh((
            target.dielectric
            - self.dielectric_isotropic_location[graph_source, None, None] * identity
        ) / self.dielectric_scale[graph_source, None, None])
        born = _radial_asinh((
            target.born_effective_charge
            - self.born_isotropic_location[node_source, None, None] * identity
        ) / self.born_scale[node_source, None, None])
        return ResponseTargets(
            piezoelectric=_radial_asinh(
                target.piezoelectric
                / self.piezoelectric_scale[graph_source, None, None, None]
            ),
            dielectric=dielectric,
            elastic=_radial_asinh(
                target.elastic / self.elastic_scale[graph_source, None, None, None, None]
            ),
            born_effective_charge=born,
            gamma_soft=target.gamma_soft,
            gamma_log_magnitude=(
                target.gamma_log_magnitude
                - self.gamma_log_location[graph_source, None]
            )
            / self.gamma_log_scale[graph_source, None],
            internal_strain=_radial_asinh(
                target.internal_strain
                / self.internal_strain_scale[node_source, None, None, None]
            ),
            piezoelectric_mask=target.piezoelectric_mask,
            dielectric_mask=target.dielectric_mask,
            elastic_mask=target.elastic_mask,
            born_mask=target.born_mask,
            gamma_mask=target.gamma_mask,
            internal_strain_mask=target.internal_strain_mask,
        )

    def denormalize_predictions(
        self,
        prediction: ResponsePredictions,
        source_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> ResponsePredictions:
        graph_source = source_index
        node_source = source_index[batch]
        identity = torch.eye(
            3,
            dtype=prediction.dielectric.dtype,
            device=prediction.dielectric.device,
        )
        return ResponsePredictions(
            piezoelectric=_radial_sinh(prediction.piezoelectric)
            * self.piezoelectric_scale[graph_source, None, None, None],
            dielectric=_radial_sinh(prediction.dielectric)
            * self.dielectric_scale[graph_source, None, None]
            + self.dielectric_isotropic_location[graph_source, None, None] * identity,
            elastic=_radial_sinh(prediction.elastic)
            * self.elastic_scale[graph_source, None, None, None, None],
            born_effective_charge=_radial_sinh(prediction.born_effective_charge)
            * self.born_scale[node_source, None, None]
            + self.born_isotropic_location[node_source, None, None] * identity,
            gamma_soft_logits=prediction.gamma_soft_logits,
            gamma_log_magnitude=prediction.gamma_log_magnitude
            * self.gamma_log_scale[graph_source, None]
            + self.gamma_log_location[graph_source, None],
            internal_strain=_radial_sinh(prediction.internal_strain)
            * self.internal_strain_scale[node_source, None, None, None],
        )


def load_response_normalizer(
    path: Path,
    *,
    expected_cache_sha256: str,
) -> ResponseNormalizer:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "gaugeflow.stage_d_response_normalizer.v1"
        or not bool(payload.get("qualified"))
        or payload.get("cache_sha256") != expected_cache_sha256
    ):
        raise ValueError("Stage-D response normalizer provenance is invalid")
    source_count = payload.get("source_count")
    fields = payload.get("fields")
    if not isinstance(source_count, int) or source_count < 1 or not isinstance(fields, dict):
        raise ValueError("Stage-D response normalizer metadata are invalid")
    values: dict[str, torch.Tensor] = {}
    for name in ResponseNormalizer.__dataclass_fields__:
        value = fields.get(name)
        if not isinstance(value, list) or len(value) != source_count:
            raise ValueError(f"Stage-D response normalizer field {name!r} is invalid")
        tensor = torch.tensor(value, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Stage-D response normalizer field {name!r} is non-finite")
        if name.endswith("scale") and bool((tensor <= 0.0).any()):
            raise ValueError(f"Stage-D response normalizer scale {name!r} is nonpositive")
        values[name] = tensor
    return ResponseNormalizer(**values)


def _robust_tensor_scale(value: torch.Tensor, *, minimum: float = 1e-6) -> torch.Tensor:
    if value.numel() == 0:
        return torch.ones((), dtype=torch.float64)
    object_rms = value.double().square().flatten(1).mean(dim=-1).sqrt()
    nonzero = object_rms[object_rms > minimum]
    if not nonzero.numel():
        return torch.ones((), dtype=torch.float64)
    return nonzero.median().clamp_min(minimum)


def _robust_scalar_location_scale(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not value.numel():
        one = torch.ones((), dtype=torch.float64)
        return one.new_zeros(()), one
    value = value.double()
    location = value.median()
    median_absolute_deviation = (value - location).abs().median()
    scale = (1.4826 * median_absolute_deviation).clamp_min(1e-6)
    return location, scale


def fit_response_normalizer(
    target: ResponseTargets,
    source_index: torch.Tensor,
    batch: torch.Tensor,
    *,
    source_count: int,
) -> ResponseNormalizer:
    """Fit scalar O(3)-compatible statistics without dropping physical zeros."""

    if source_count < 1 or source_index.shape != target.piezoelectric_mask.shape:
        raise ValueError("response normalizer graph metadata are invalid")
    if batch.shape != target.born_mask.shape:
        raise ValueError("response normalizer node metadata are invalid")
    node_source = source_index[batch]
    identity = torch.eye(3, dtype=torch.float64)
    statistics: dict[str, list[torch.Tensor]] = {
        name: [] for name in ResponseNormalizer.__dataclass_fields__
    }
    for source in range(source_count):
        graph = source_index == source
        node = node_source == source

        piezo_mask = graph & target.piezoelectric_mask
        statistics["piezoelectric_scale"].append(
            _robust_tensor_scale(target.piezoelectric[piezo_mask])
        )

        dielectric_mask = graph & target.dielectric_mask
        dielectric_value = target.dielectric[dielectric_mask].double()
        if dielectric_value.numel():
            dielectric_trace = torch.einsum("gii->g", dielectric_value) / 3.0
            dielectric_location = dielectric_trace.median()
            dielectric_centered = dielectric_value - dielectric_location * identity
        else:
            dielectric_location = torch.zeros((), dtype=torch.float64)
            dielectric_centered = dielectric_value
        statistics["dielectric_isotropic_location"].append(dielectric_location)
        statistics["dielectric_scale"].append(
            _robust_tensor_scale(dielectric_centered)
        )

        elastic_mask = graph & target.elastic_mask
        statistics["elastic_scale"].append(
            _robust_tensor_scale(target.elastic[elastic_mask])
        )

        born_mask = node & target.born_mask
        born_value = target.born_effective_charge[born_mask].double()
        if born_value.numel():
            born_location = (torch.einsum("nii->n", born_value) / 3.0).median()
            born_centered = born_value - born_location * identity
        else:
            born_location = torch.zeros((), dtype=torch.float64)
            born_centered = born_value
        statistics["born_isotropic_location"].append(born_location)
        statistics["born_scale"].append(_robust_tensor_scale(born_centered))

        gamma_graph_mask = graph[:, None] & target.gamma_mask
        gamma_value = target.gamma_log_magnitude[gamma_graph_mask].double()
        gamma_location, gamma_scale = _robust_scalar_location_scale(gamma_value)
        statistics["gamma_log_location"].append(gamma_location)
        statistics["gamma_log_scale"].append(gamma_scale)

        internal_node_mask = node & target.internal_strain_mask.flatten(1).all(dim=-1)
        statistics["internal_strain_scale"].append(
            _robust_tensor_scale(target.internal_strain[internal_node_mask])
        )
    return ResponseNormalizer(
        **{
            name: torch.stack(values).to(torch.float32)
            for name, values in statistics.items()
        }
    )
