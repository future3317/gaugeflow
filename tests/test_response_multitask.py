from __future__ import annotations

import torch

from gaugeflow.production.response_multitask import (
    CartesianResponseHeads,
    ResponseTargets,
    piezoelectric_response_probe_loss,
    response_multitask_loss,
)


def _rotation(dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(91)
    matrix = torch.randn(3, 3, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _rotate_rank2(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ia,...ab,jb->...ij", rotation, value, rotation)


def _rotate_rank3(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ia,jb,kc,...abc->...ijk", rotation, rotation, rotation, value)


def _rotate_rank4(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "ia,jb,kc,ld,...abcd->...ijkl", rotation, rotation, rotation, rotation, value
    )


def test_cartesian_response_heads_are_o3_covariant_and_permutation_invariant():
    torch.manual_seed(7)
    heads = CartesianResponseHeads(12, 5, covariant_rank=3, maximum_atoms=4).to(torch.float64)
    scalar = torch.randn(7, 12, dtype=torch.float64)
    vectors = torch.randn(7, 5, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    rotation = _rotation(torch.float64)

    original = heads(scalar, vectors, batch, 2)
    rotated_vectors = torch.einsum("ij,nvj->nvi", rotation, vectors)
    rotated = heads(scalar, rotated_vectors, batch, 2)
    assert torch.allclose(rotated.dielectric, _rotate_rank2(original.dielectric, rotation), atol=2e-9)
    assert torch.allclose(rotated.piezoelectric, _rotate_rank3(original.piezoelectric, rotation), atol=2e-9)
    assert torch.allclose(rotated.elastic, _rotate_rank4(original.elastic, rotation), atol=2e-9)
    assert torch.allclose(
        rotated.born_effective_charge,
        _rotate_rank2(original.born_effective_charge, rotation),
        atol=2e-9,
    )
    assert torch.allclose(rotated.internal_strain, _rotate_rank3(original.internal_strain, rotation), atol=2e-9)
    assert torch.equal(rotated.gamma_soft_logits, original.gamma_soft_logits)
    assert torch.equal(rotated.gamma_log_magnitude, original.gamma_log_magnitude)

    permutation = torch.tensor([2, 0, 1, 6, 3, 5, 4])
    permuted = heads(scalar[permutation], vectors[permutation], batch, 2)
    assert torch.allclose(permuted.piezoelectric, original.piezoelectric, atol=2e-9)
    assert torch.allclose(permuted.dielectric, original.dielectric, atol=2e-9)
    assert torch.allclose(permuted.elastic, original.elastic, atol=2e-9)


def test_response_loss_masks_missing_labels_but_trains_physical_zero():
    torch.manual_seed(8)
    heads = CartesianResponseHeads(10, 4, covariant_rank=2, maximum_atoms=3)
    scalar = torch.randn(5, 10, requires_grad=True)
    vectors = torch.randn(5, 4, 3, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1, 1])
    prediction = heads(scalar, vectors, batch, 2)
    targets = ResponseTargets(
        piezoelectric=torch.zeros_like(prediction.piezoelectric),
        dielectric=torch.zeros_like(prediction.dielectric),
        elastic=torch.zeros_like(prediction.elastic),
        born_effective_charge=torch.zeros_like(prediction.born_effective_charge),
        gamma_soft=torch.zeros_like(prediction.gamma_soft_logits),
        gamma_log_magnitude=torch.zeros_like(prediction.gamma_log_magnitude),
        internal_strain=torch.zeros_like(prediction.internal_strain),
        piezoelectric_mask=torch.tensor([True, False]),
        dielectric_mask=torch.zeros(2, dtype=torch.bool),
        elastic_mask=torch.zeros(2, dtype=torch.bool),
        born_mask=torch.tensor([False, False, True, True, True]),
        gamma_mask=torch.tensor(
            [[True] * 6 + [False] * 3, [False] * 9], dtype=torch.bool
        ),
        internal_strain_mask=torch.zeros_like(prediction.internal_strain, dtype=torch.bool),
    )
    targets.internal_strain_mask[0, 0, 0, 0] = True
    output = response_multitask_loss(prediction, targets, batch, 2)
    assert output.active_tasks == 4
    assert torch.isfinite(output.loss) and output.loss > 0.0
    output.loss.backward()
    assert scalar.grad is not None and torch.isfinite(scalar.grad).all()
    assert vectors.grad is not None and torch.isfinite(vectors.grad).all()


def test_response_loss_with_no_labels_is_differentiable_zero():
    heads = CartesianResponseHeads(8, 3, covariant_rank=2, maximum_atoms=2)
    scalar = torch.randn(2, 8, requires_grad=True)
    vectors = torch.randn(2, 3, 3, requires_grad=True)
    batch = torch.tensor([0, 0])
    prediction = heads(scalar, vectors, batch, 1)
    targets = ResponseTargets(
        piezoelectric=torch.zeros_like(prediction.piezoelectric),
        dielectric=torch.zeros_like(prediction.dielectric),
        elastic=torch.zeros_like(prediction.elastic),
        born_effective_charge=torch.zeros_like(prediction.born_effective_charge),
        gamma_soft=torch.zeros_like(prediction.gamma_soft_logits),
        gamma_log_magnitude=torch.zeros_like(prediction.gamma_log_magnitude),
        internal_strain=torch.zeros_like(prediction.internal_strain),
        piezoelectric_mask=torch.zeros(1, dtype=torch.bool),
        dielectric_mask=torch.zeros(1, dtype=torch.bool),
        elastic_mask=torch.zeros(1, dtype=torch.bool),
        born_mask=torch.zeros(2, dtype=torch.bool),
        gamma_mask=torch.zeros_like(prediction.gamma_soft_logits, dtype=torch.bool),
        internal_strain_mask=torch.zeros_like(prediction.internal_strain, dtype=torch.bool),
    )
    output = response_multitask_loss(prediction, targets, batch, 1)
    assert output.active_tasks == 0
    assert output.loss == 0.0
    output.loss.backward()
    assert scalar.grad is not None and torch.isfinite(scalar.grad).all()


def test_icosahedral_piezoelectric_probe_is_rotation_invariant():
    generator = torch.Generator().manual_seed(44)
    prediction = torch.randn(5, 3, 3, 3, generator=generator, dtype=torch.float64)
    target = torch.randn(5, 3, 3, 3, generator=generator, dtype=torch.float64)
    rotation = _rotation(torch.float64)
    mask = torch.tensor([True, True, False, True, True])
    original, active = piezoelectric_response_probe_loss(prediction, target, mask)
    rotated, rotated_active = piezoelectric_response_probe_loss(
        _rotate_rank3(prediction, rotation),
        _rotate_rank3(target, rotation),
        mask,
    )
    assert active and rotated_active
    assert torch.allclose(original, rotated, atol=2e-12)
