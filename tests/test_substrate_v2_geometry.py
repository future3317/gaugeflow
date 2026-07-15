import torch

from gaugeflow.geometry import GaussianRadialBasis, periodic_closest_image_edges
from gaugeflow.substrate_v2 import GeometryAwareSiteScorer
from gaugeflow.vocabulary import MASK_TOKEN


def _two_graph_geometry():
    frac = torch.tensor(
        [[0.00, 0.00, 0.00], [0.25, 0.25, 0.00], [0.50, 0.00, 0.25], [0.75, 0.25, 0.50],
         [0.00, 0.00, 0.00], [0.25, 0.25, 0.00], [0.50, 0.00, 0.25], [0.75, 0.25, 0.50]],
        dtype=torch.float32,
    )
    lattice = torch.tensor(
        [[[3.1, 0.2, 0.1], [0.7, 4.0, -0.3], [0.2, 0.5, 5.3]],
         [[3.1, 0.2, 0.1], [0.7, 4.0, -0.3], [0.2, 0.5, 5.3]]],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    endpoint = torch.tensor([0, 1])
    tokens = torch.full((8,), MASK_TOKEN, dtype=torch.long)
    return tokens, frac, lattice, batch, endpoint


def test_periodic_edges_carry_metric_displacement_distance_and_image_shift():
    _, frac, lattice, batch, _ = _two_graph_geometry()
    edges = periodic_closest_image_edges(frac, lattice, batch)
    assert edges.source.numel() == 24
    assert edges.displacement.shape == edges.direction.shape == edges.image_shift.shape == (24, 3)
    assert torch.allclose(torch.linalg.vector_norm(edges.displacement, dim=-1), edges.distance)
    assert torch.allclose(torch.linalg.vector_norm(edges.direction, dim=-1), torch.ones_like(edges.distance))
    assert torch.equal(edges.image_shift, edges.image_shift.round())


def test_rbf_is_metric_sensitive_and_smooth_inside_cutoff():
    rbf = GaussianRadialBasis(count=8, cutoff=5.0)
    distances = torch.tensor([1.0, 1.1], requires_grad=True)
    encoded = rbf(distances)
    assert not torch.allclose(encoded[0], encoded[1])
    encoded.sum().backward()
    assert torch.isfinite(distances.grad).all()


def test_geometry_scorer_is_node_permutation_equivariant_and_uses_dense_vocabulary():
    torch.manual_seed(9)
    scorer = GeometryAwareSiteScorer(hidden_dim=24, layers=2, vector_channels=5, rbf_dim=8)
    tokens, frac, lattice, batch, endpoint = _two_graph_geometry()
    scores = scorer(tokens, frac, lattice, batch, endpoint)
    assert scores.shape == (8, 118)
    assert torch.isfinite(scores).all()

    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    relabelled = scorer(tokens[permutation], frac[permutation], lattice, batch[permutation], endpoint)
    assert torch.allclose(relabelled, scores[permutation], atol=2e-6, rtol=2e-6)
