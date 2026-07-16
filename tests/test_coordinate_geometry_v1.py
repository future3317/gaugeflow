"""Regression coverage for the versioned quotient coordinate-flow substrate."""

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from gaugeflow.coupling import remove_graphwise_translation, translation_aligned_torus_rms
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.geometry import GaussianRadialBasis, periodic_closest_image_edges
from gaugeflow.manifold import lattice_to_log_vector, torus_logmap, wrap01
from gaugeflow.model import GaugeFlowVectorField


def _coordinate_batch() -> Batch:
    return Batch.from_data_list([
        Data(
            atom_types=torch.tensor([5, 7, 14]),
            frac_coords=torch.tensor(
                [[0.07, 0.11, 0.19], [0.34, 0.22, 0.31], [0.72, 0.48, 0.41]]
            ),
            lattice=torch.tensor(
                [[3.9, 0.2, 0.1], [0.3, 4.3, 0.4], [0.1, 0.4, 5.1]]
            ).unsqueeze(0),
            num_nodes=3,
        )
    ])


def test_prepared_d0_3_contract_cannot_silently_reuse_the_legacy_path():
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "gate_p5_d0_3_translation_quotient_metric_v1.json")
        .read_text(encoding="utf-8")
    )
    assert protocol["status"] == "prepared_not_started"
    assert protocol["state"]["coordinate_gauge"] == "translation_quotient_no_drift"
    assert protocol["backbone"]["periodic_geometry"] == "closest_image_cartesian_distance_and_direction"
    assert protocol["backbone"]["legacy_implementations_removed"] == [
        "absolute_coordinate_gauge", "directions_only_backbone"
    ]


def test_metric_edge_geometry_preserves_legacy_direction_wrapper_and_exposes_distance():
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    short_lattice = torch.diag(torch.tensor([4.0, 5.0, 6.0])).unsqueeze(0)
    long_lattice = torch.diag(torch.tensor([8.0, 5.0, 6.0])).unsqueeze(0)

    short = periodic_closest_image_edges(frac, short_lattice, batch)
    long = periodic_closest_image_edges(frac, long_lattice, batch)
    short_direction, short_distance = short.direction, short.distance
    long_direction, long_distance = long.direction, long.distance
    assert torch.allclose(short_direction, long_direction)
    assert torch.allclose(short_distance, torch.ones_like(short_distance))
    assert torch.allclose(long_distance, torch.full_like(long_distance, 2.0))

    rbf = GaussianRadialBasis(count=8, cutoff=5.0)
    assert not torch.allclose(rbf(short_distance), rbf(long_distance))


def test_adaptive_cvp_beats_fixed_shell_counterexample():
    lattice = torch.tensor(
        [[[1.0, 0.0, 0.0], [14.554425, 0.061259, 0.0], [5.614603, -0.193389, 0.164521]]],
        dtype=torch.float64,
    )
    delta = torch.tensor([0.825511, 0.213272, 0.458993], dtype=torch.float64)
    frac = torch.stack((torch.zeros(3, dtype=torch.float64), delta))
    edges = periodic_closest_image_edges(frac, lattice, torch.zeros(2, dtype=torch.long))
    forward = torch.nonzero((edges.source == 0) & (edges.target == 1), as_tuple=False).flatten()
    assert forward.numel() == 1
    index = int(forward[0])
    # The review's [-8,8]^3 witness [8,-1,0] is better than [-2,2]^3 but is
    # itself not globally optimal. Exact CVP finds an even shorter image.
    assert torch.equal(edges.image_shift[index], torch.tensor([-21.0, 1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(
        edges.distance[index].square(), torch.tensor(0.0096380871, dtype=torch.float64), atol=1.0e-10
    )
    assert edges.distance[index].square() < 0.0267454


def test_metric_rbf_coordinate_field_is_translation_invariant_and_backpropagates():
    torch.manual_seed(1401)
    batch = _coordinate_batch()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    state = matcher.random_state(batch)
    model = GaugeFlowVectorField(
        hidden_dim=32,
        layers=2,
        conditioning_mode="unconditional",
        coordinate_rbf_dim=8,
        coordinate_rbf_cutoff=8.0,
    )
    time = torch.tensor([0.37])
    original = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)
    translated = model(
        state.type_state,
        wrap01(state.frac_coords + torch.tensor([0.13, 0.27, 0.41])),
        state.lattice_log,
        batch.batch,
        time,
    )
    for left, right in zip(original[:3], translated[:3]):
        assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)

    loss = original[1].square().mean()
    loss.backward()
    radial_gate_gradient = model.layers[0].vector_gates[0].weight.grad[:, -8:]
    assert torch.isfinite(radial_gate_gradient).all()
    assert radial_gate_gradient.abs().sum() > 0
    assert model.coordinate_edge_out[-1].weight.grad.abs().sum() > 0


def test_no_drift_path_closes_modulo_global_translation_with_the_production_sampler():
    class ConstantCoordinateField(torch.nn.Module):
        conditioning_mode = "unconditional"

        def __init__(self, coord_velocity: torch.Tensor, node_count: int):
            super().__init__()
            self.coord_velocity = coord_velocity
            self.node_count = node_count

        def forward(self, type_state, frac_coords, lattice_log, batch, time, **kwargs):
            del frac_coords, batch, time, kwargs
            return (
                torch.zeros_like(type_state),
                self.coord_velocity,
                torch.zeros_like(lattice_log),
                torch.ones((1, 1), dtype=type_state.dtype, device=type_state.device),
            )

    batch = _coordinate_batch()
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    target = matcher.target_state(batch)
    source = CrystalFlowState(
        type_state=target.type_state,
        frac_coords=wrap01(target.frac_coords + torch.tensor([[0.31, -0.22, 0.17], [-0.14, 0.08, 0.26], [0.21, 0.19, -0.35]])),
        lattice_log=lattice_to_log_vector(batch.lattice),
    )
    raw_velocity = torus_logmap(source.frac_coords, target.frac_coords)
    quotient_velocity = remove_graphwise_translation(raw_velocity, batch.batch, batch.num_graphs)
    assert torch.allclose(quotient_velocity.mean(dim=0), torch.zeros(3), atol=1e-7)
    sampled = matcher.sample(
        ConstantCoordinateField(raw_velocity, source.frac_coords.shape[0]),
        batch,
        steps=32,
        initial_state=source,
    )
    assert translation_aligned_torus_rms(sampled.frac_coords, target.frac_coords) < 2e-6
    assert torch.sqrt(torus_logmap(sampled.frac_coords, target.frac_coords).square().mean()) > 1e-3


def test_metric_geometry_is_the_only_production_coordinate_backbone():
    model = GaugeFlowVectorField(hidden_dim=16, layers=1, conditioning_mode="unconditional")
    assert model.coordinate_rbf is not None
    assert model.coordinate_rbf_dim == 16
    with pytest.raises(ValueError, match="at least two"):
        GaugeFlowVectorField(
            hidden_dim=16,
            layers=1,
            conditioning_mode="unconditional",
            coordinate_rbf_dim=1,
        )
