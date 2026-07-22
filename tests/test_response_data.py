from __future__ import annotations

import torch

from gaugeflow.production.response_data import (
    ResponseRecord,
    augment_equivalent_response_batch,
    collate_response_records,
)
from gaugeflow.production.response_multitask import ResponseTargets


def _record(atom_count: int, offset: float) -> ResponseRecord:
    generator = torch.Generator().manual_seed(atom_count)
    return ResponseRecord(
        element_tokens=torch.arange(atom_count),
        fractional_coordinates=torch.rand(atom_count, 3, generator=generator),
        lattice=torch.diag(torch.tensor([2.0 + offset, 3.0, 4.0])),
        source_index=torch.tensor([0]),
        targets=ResponseTargets(
            piezoelectric=torch.randn(3, 3, 3, generator=generator),
            dielectric=torch.randn(3, 3, generator=generator),
            elastic=torch.randn(3, 3, 3, 3, generator=generator),
            born_effective_charge=torch.randn(atom_count, 3, 3, generator=generator),
            gamma_soft=torch.zeros(12),
            gamma_log_magnitude=torch.arange(12, dtype=torch.float32),
            internal_strain=torch.randn(atom_count, 3, 3, 3, generator=generator),
            piezoelectric_mask=torch.tensor(True),
            dielectric_mask=torch.tensor(True),
            elastic_mask=torch.tensor(False),
            born_mask=torch.ones(atom_count, dtype=torch.bool),
            gamma_mask=torch.ones(12, dtype=torch.bool),
            internal_strain_mask=torch.ones(atom_count, 3, 3, 3, dtype=torch.bool),
        ),
    )


def _minimum_image_spectrum(fractional: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    shifts = torch.cartesian_prod(*(torch.arange(-2, 3),) * 3).to(fractional)
    pair = fractional[:, None] - fractional[None, :]
    images = torch.einsum("ijsc,cd->ijsd", pair[:, :, None] + shifts, lattice)
    distances = images.square().sum(dim=-1).amin(dim=-1).sqrt()
    upper = torch.triu_indices(fractional.shape[0], fractional.shape[0], offset=1)
    return torch.sort(distances[upper[0], upper[1]]).values


def test_response_collation_and_equivalent_view_preserve_physical_geometry():
    value = collate_response_records([_record(3, 0.0), _record(2, 0.5)])
    transformed = augment_equivalent_response_batch(
        value, generator=torch.Generator().manual_seed(17)
    )
    assert transformed.node_counts.tolist() == [3, 2]
    assert transformed.targets.piezoelectric_mask.shape == (2,)
    assert torch.equal(
        torch.sort(transformed.element_tokens[:3]).values,
        torch.sort(value.element_tokens[:3]).values,
    )
    assert torch.equal(
        torch.sort(transformed.element_tokens[3:]).values,
        torch.sort(value.element_tokens[3:]).values,
    )
    assert torch.allclose(
        torch.linalg.det(transformed.lattice).abs(),
        torch.linalg.det(value.lattice).abs(),
        atol=2e-5,
    )
    assert bool((torch.linalg.det(transformed.lattice) > 0.0).all())
    # Periodic pair distances are unchanged up to origin shifts, a certified
    # lattice basis change, atom relabelling and a common orthogonal action.
    for graph in range(2):
        old_distances = _minimum_image_spectrum(
            value.fractional_coordinates[value.batch == graph], value.lattice[graph]
        )
        new_distances = _minimum_image_spectrum(
            transformed.fractional_coordinates[transformed.batch == graph],
            transformed.lattice[graph],
        )
        assert torch.allclose(old_distances, new_distances, atol=2e-5)
    assert torch.allclose(
        transformed.targets.piezoelectric.square().sum(dim=(1, 2, 3)),
        value.targets.piezoelectric.square().sum(dim=(1, 2, 3)),
        atol=2e-5,
    )


def test_equivalent_view_is_deterministic_for_explicit_generator_state():
    value = collate_response_records([_record(4, 0.0)])
    left = augment_equivalent_response_batch(
        value, generator=torch.Generator().manual_seed(29)
    )
    right = augment_equivalent_response_batch(
        value, generator=torch.Generator().manual_seed(29)
    )
    for name in (
        "element_tokens",
        "fractional_coordinates",
        "lattice",
        "batch",
        "node_counts",
        "source_index",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name))
    for name in ResponseTargets.__dataclass_fields__:
        assert torch.equal(getattr(left.targets, name), getattr(right.targets, name))
