from __future__ import annotations

import torch

from gaugeflow.production.response_multitask import ResponsePredictions, ResponseTargets
from gaugeflow.production.response_normalization import fit_response_normalizer


def _targets() -> ResponseTargets:
    return ResponseTargets(
        piezoelectric=torch.stack((torch.zeros(3, 3, 3), torch.ones(3, 3, 3))),
        dielectric=torch.stack((2.0 * torch.eye(3), 4.0 * torch.eye(3))),
        elastic=torch.zeros(2, 3, 3, 3, 3),
        born_effective_charge=torch.stack((3.0 * torch.eye(3), 5.0 * torch.eye(3))),
        gamma_soft=torch.zeros(2, 6),
        gamma_log_magnitude=torch.tensor([[1.0] * 6, [3.0] * 6]),
        internal_strain=torch.stack((torch.zeros(3, 3, 3), 2.0 * torch.ones(3, 3, 3))),
        piezoelectric_mask=torch.ones(2, dtype=torch.bool),
        dielectric_mask=torch.ones(2, dtype=torch.bool),
        elastic_mask=torch.zeros(2, dtype=torch.bool),
        born_mask=torch.ones(2, dtype=torch.bool),
        gamma_mask=torch.ones(2, 6, dtype=torch.bool),
        internal_strain_mask=torch.ones(2, 3, 3, 3, dtype=torch.bool),
    )


def test_response_normalizer_is_source_local_and_preserves_masks_and_zeros():
    target = _targets()
    source = torch.tensor([0, 1])
    batch = torch.tensor([0, 1])
    normalizer = fit_response_normalizer(target, source, batch, source_count=2)
    normalized = normalizer.normalize(target, source, batch)
    assert torch.equal(normalized.piezoelectric_mask, target.piezoelectric_mask)
    assert torch.equal(normalized.gamma_mask, target.gamma_mask)
    assert torch.equal(normalized.piezoelectric[0], torch.zeros(3, 3, 3))
    assert torch.allclose(normalized.dielectric, torch.zeros_like(normalized.dielectric))
    assert torch.allclose(normalized.born_effective_charge, torch.zeros_like(normalized.born_effective_charge))
    assert normalizer.elastic_scale.tolist() == [1.0, 1.0]


def test_response_normalization_commutes_with_cartesian_rotation():
    target = _targets()
    source = torch.tensor([0, 0])
    batch = torch.tensor([0, 1])
    normalizer = fit_response_normalizer(target, source, batch, source_count=1)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(4)))

    def rank2(value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ia,...ab,jb->...ij", rotation, value, rotation)

    def rank3(value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ia,jb,kc,...abc->...ijk", rotation, rotation, rotation, value)

    rotated = ResponseTargets(
        piezoelectric=rank3(target.piezoelectric),
        dielectric=rank2(target.dielectric),
        elastic=target.elastic,
        born_effective_charge=rank2(target.born_effective_charge),
        gamma_soft=target.gamma_soft,
        gamma_log_magnitude=target.gamma_log_magnitude,
        internal_strain=rank3(target.internal_strain),
        piezoelectric_mask=target.piezoelectric_mask,
        dielectric_mask=target.dielectric_mask,
        elastic_mask=target.elastic_mask,
        born_mask=target.born_mask,
        gamma_mask=target.gamma_mask,
        internal_strain_mask=target.internal_strain_mask,
    )
    left = normalizer.normalize(rotated, source, batch)
    right = normalizer.normalize(target, source, batch)
    assert torch.allclose(left.piezoelectric, rank3(right.piezoelectric), atol=1e-5)
    assert torch.allclose(left.dielectric, rank2(right.dielectric), atol=1e-5)
    assert torch.allclose(left.born_effective_charge, rank2(right.born_effective_charge), atol=1e-5)


def test_radial_response_normalization_round_trips_predictions():
    target = _targets()
    source = torch.tensor([0, 0])
    batch = torch.tensor([0, 1])
    normalizer = fit_response_normalizer(target, source, batch, source_count=1)
    normalized = normalizer.normalize(target, source, batch)
    prediction = ResponsePredictions(
        piezoelectric=normalized.piezoelectric,
        dielectric=normalized.dielectric,
        elastic=normalized.elastic,
        born_effective_charge=normalized.born_effective_charge,
        gamma_soft_logits=normalized.gamma_soft,
        gamma_log_magnitude=normalized.gamma_log_magnitude,
        internal_strain=normalized.internal_strain,
    )
    restored = normalizer.denormalize_predictions(prediction, source, batch)
    assert torch.allclose(restored.piezoelectric, target.piezoelectric, atol=2e-5)
    assert torch.allclose(restored.dielectric, target.dielectric, atol=2e-5)
    assert torch.allclose(restored.born_effective_charge, target.born_effective_charge, atol=2e-5)
    assert torch.allclose(restored.internal_strain, target.internal_strain, atol=2e-5)
