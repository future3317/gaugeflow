from __future__ import annotations

import itertools

import torch

from gaugeflow.geometry import periodic_radius_multigraph


def _edge_keys(edges) -> set[tuple[int, int, int, int, int]]:
    return {
        (int(source), int(target), *map(int, shift))
        for source, target, shift in zip(
            edges.source.tolist(),
            edges.target.tolist(),
            edges.image_shift.tolist(),
            strict=True,
        )
    }


def _brute_force_keys(
    fractional: torch.Tensor, lattice: torch.Tensor, cutoff: float, shell: int = 5
) -> set[tuple[int, int, int, int, int]]:
    output: set[tuple[int, int, int, int, int]] = set()
    for source in range(fractional.shape[0]):
        for target in range(fractional.shape[0]):
            for shift in itertools.product(range(-shell, shell + 1), repeat=3):
                if source == target and shift == (0, 0, 0):
                    continue
                delta = fractional[target] - fractional[source] + torch.tensor(
                    shift, dtype=fractional.dtype
                )
                if float(torch.linalg.vector_norm(delta @ lattice)) < cutoff:
                    output.add((source, target, *shift))
    return output


def test_single_site_retains_all_periodic_self_images():
    fractional = torch.tensor([[0.17, 0.31, 0.43]])
    lattice = torch.eye(3).unsqueeze(0)
    edges = periodic_radius_multigraph(
        fractional, lattice, torch.zeros(1, dtype=torch.long), cutoff=1.01
    )
    assert _edge_keys(edges) == _brute_force_keys(
        fractional, lattice[0], 1.01
    )
    assert edges.source.numel() == 6


def test_skew_cell_matches_complete_brute_force_multigraph():
    fractional = torch.tensor(
        [[0.05, 0.11, 0.91], [0.62, 0.49, 0.21], [0.28, 0.77, 0.44]],
        dtype=torch.float64,
    )
    lattice = torch.tensor(
        [[1.7, 0.0, 0.0], [1.2, 1.4, 0.0], [0.8, 0.6, 1.3]],
        dtype=torch.float64,
    )
    cutoff = 2.4
    edges = periodic_radius_multigraph(
        fractional,
        lattice.unsqueeze(0),
        torch.zeros(3, dtype=torch.long),
        cutoff=cutoff,
    )
    assert _edge_keys(edges) == _brute_force_keys(
        fractional, lattice, cutoff
    )
    assert bool((edges.distance < cutoff).all())


def test_multigraph_is_translation_periodic_and_has_finite_gradients():
    fractional = torch.tensor(
        [[0.1, 0.2, 0.3], [0.8, 0.6, 0.4]], requires_grad=True
    )
    lattice = torch.tensor(
        [[[2.1, 0.0, 0.0], [0.4, 2.3, 0.0], [0.2, 0.3, 2.5]]],
        requires_grad=True,
    )
    batch = torch.zeros(2, dtype=torch.long)
    first = periodic_radius_multigraph(fractional, lattice, batch, cutoff=2.6)
    second = periodic_radius_multigraph(
        fractional + torch.tensor([3.0, -2.0, 5.0]), lattice, batch, cutoff=2.6
    )
    assert _edge_keys(first) == _edge_keys(second)
    assert torch.allclose(
        torch.sort(first.distance).values,
        torch.sort(second.distance).values,
        atol=1e-6,
        rtol=1e-6,
    )
    first.distance.square().sum().backward()
    assert fractional.grad is not None and torch.isfinite(fractional.grad).all()
    assert lattice.grad is not None and torch.isfinite(lattice.grad).all()
