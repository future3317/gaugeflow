import torch
import numpy as np
import runpy
from pathlib import Path

from gaugeflow.geometry import GaussianRadialBasis, periodic_closest_image_edges
from gaugeflow.assignment import exact_assignment_distribution_permutation_log_probability_error
from gaugeflow.substrate_v2 import GeometryAwareSiteScorer
from gaugeflow.pymatgen_compat import enable_structure_matcher_numpy2_compatibility
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


def test_geometry_scorer_keeps_complete_assignment_law_stable_when_scores_are_sharp():
    torch.manual_seed(23)
    scorer = GeometryAwareSiteScorer(hidden_dim=24, layers=3, vector_channels=5, rbf_dim=8)
    # Mimic a saturated exact-assignment training state without making the
    # test depend on an optimizer trajectory.
    with torch.no_grad():
        scorer.score_head[-1].weight.mul_(1.0e5)
        scorer.score_head[-1].bias.mul_(1.0e5)
    tokens, frac, lattice, batch, endpoint = _two_graph_geometry()
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    original = scorer(tokens, frac, lattice, batch, endpoint)
    relabelled = scorer(tokens[permutation], frac[permutation], lattice, batch[permutation], endpoint)
    for start in (0, 4):
        local = permutation[start : start + 4] - start
        error = exact_assignment_distribution_permutation_log_probability_error(
            original[start : start + 4],
            relabelled[start : start + 4],
            torch.bincount(torch.tensor([4, 4, 6, 6]), minlength=118),
            local,
        )
        assert error <= 2e-6


def test_scalar_and_metric_ablations_have_declared_feature_paths():
    tokens, frac, lattice, batch, endpoint = _two_graph_geometry()
    legacy = GeometryAwareSiteScorer(
        hidden_dim=16, layers=2, vector_channels=4, rbf_dim=8,
        use_rbf=False, use_vector_invariants=False,
    )
    metric = GeometryAwareSiteScorer(
        hidden_dim=16, layers=2, vector_channels=4, rbf_dim=8,
        use_rbf=True, use_vector_invariants=False,
    )
    assert legacy(tokens, frac, lattice, batch, endpoint).shape == (8, 118)
    assert metric(tokens, frac, lattice, batch, endpoint).shape == (8, 118)
    assert legacy.score_head[0].in_features == 16
    assert metric.score_head[0].in_features == 16


def test_structure_matcher_numpy2_compatibility_exposes_removed_aliases(monkeypatch):
    monkeypatch.delitem(np.__dict__, "bool", raising=False)
    monkeypatch.delitem(np.__dict__, "int", raising=False)
    enable_structure_matcher_numpy2_compatibility()
    assert np.bool is np.bool_
    assert np.int is int


def test_decoration_runner_evaluates_a_complete_assignment_law(monkeypatch):
    runner = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_substrate_v2_decoration_only.py")
    )
    Endpoint = runner["Endpoint"]
    target = torch.tensor([4, 4, 6, 6], dtype=torch.long)
    record = Endpoint(
        material_id="synthetic",
        endpoint_id=0,
        lattice=torch.eye(3),
        frac=torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.4, 0.0, 0.2], [0.7, 0.3, 0.4]]
        ),
        target=target,
        counts=torch.bincount(target, minlength=118),
        proper=torch.arange(4).unsqueeze(0),
        full=torch.arange(4).unsqueeze(0),
    )
    monkeypatch.setitem(runner["evaluate"].__globals__, "_match_structure", lambda *_: True)
    model = GeometryAwareSiteScorer(hidden_dim=16, layers=2, vector_channels=4, rbf_dim=8)
    rows = runner["evaluate"](model, [record], seed=11, samples=2)
    assert len(rows) == 1
    assert rows[0]["exact_composition_count"] == 1
    assert rows[0]["sampling_failures"] == 0
    assert rows[0]["node_relabel_log_probability_error"] <= 2e-6
