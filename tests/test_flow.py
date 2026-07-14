import copy

import torch

from gaugeflow.conditioning import apply_condition_dropout, randomize_tensor_orbit_representative
from gaugeflow.flow import CrystalFlowState, RiemannianCrystalFlowMatcher
from gaugeflow.manifold import torus_logmap
from gaugeflow.model import (
    GaugeFlowVectorField,
    OrbitResponseFieldEncoder,
    ResponseMessageLayer,
    direct_irrep_cartesian_products,
    periodic_complete_edges,
)
from gaugeflow.tensor import fixed_so3_frames, piezo_from_irreps, piezo_voigt_to_cartesian, rotate_rank3
from torch_geometric.data import Batch, Data


def test_standalone_flow_has_finite_loss_and_distinguishes_null_from_zero_tensor():
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.zeros(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.zeros(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(hidden_dim=32, layers=2, orbit_frames=4)
    terms = RiemannianCrystalFlowMatcher().loss(model, batch)
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    model.zero_grad(set_to_none=True)
    state = RiemannianCrystalFlowMatcher().target_state(batch)
    time = torch.full((2,), 0.5)
    present = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, batch.condition_present)[0]
    null = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time, batch.piezo_irreps, torch.zeros_like(batch.condition_present))[0]
    assert not torch.allclose(present, null)


def test_response_message_layer_is_so3_equivariant():
    torch.manual_seed(7)
    layer = ResponseMessageLayer(hidden_dim=8, vector_dim=3).eval()
    rotation = fixed_so3_frames(2)[1]
    nodes, vectors = torch.randn(3, 8), torch.randn(3, 3, 3)
    source, target = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])
    directions, response, auxiliary, condition = (
        torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 8)
    )
    with torch.no_grad():
        original = layer(nodes, vectors, source, target, directions, response, auxiliary, condition)
        rotated = layer(
            nodes,
            vectors @ rotation.T,
            source,
            target,
            directions @ rotation.T,
            response @ rotation.T,
            auxiliary @ rotation.T,
            condition,
        )
    assert torch.allclose(original[0], rotated[0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(original[1] @ rotation.T, rotated[1], atol=2e-5, rtol=2e-5)


def test_flow_ignores_target_only_stabilizers_to_match_sampling_information():
    c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool),
             stabilizer_rotations=torch.eye(3).unsqueeze(0),
             stabilizer_count=torch.tensor([1]), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool),
             stabilizer_rotations=torch.stack((torch.eye(3), c2z)),
             stabilizer_count=torch.tensor([2]), num_nodes=2),
    ])
    plain = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=batch.frac_coords[:2],
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=batch.piezo_irreps[:1],
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=batch.frac_coords[2:],
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=batch.piezo_irreps[1:],
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3)
    matcher = RiemannianCrystalFlowMatcher()
    torch.manual_seed(19)
    with_target_metadata = matcher.loss(model, batch)["loss"]
    torch.manual_seed(19)
    without_target_metadata = matcher.loss(model, plain)["loss"]
    assert torch.allclose(with_target_metadata, without_target_metadata)


def test_periodic_bond_directions_survive_a_unimodular_cell_change():
    frac = torch.tensor([[0.10, 0.20, 0.30], [0.85, 0.20, 0.30], [0.40, 0.65, 0.30]])
    lattice = torch.eye(3).unsqueeze(0)
    batch = torch.zeros(3, dtype=torch.long)
    _, _, original = periodic_complete_edges(frac, lattice, batch)
    transform = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transformed_frac = frac @ torch.linalg.inv(transform)
    transformed_lattice = (transform @ lattice[0]).unsqueeze(0)
    _, _, transformed = periodic_complete_edges(transformed_frac, transformed_lattice, batch)
    assert torch.allclose(original, transformed, atol=1e-6, rtol=1e-6)


def test_direct_irrep_baseline_has_a_finite_objective():
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3, conditioning_mode="direct_irrep")
    terms = RiemannianCrystalFlowMatcher().loss(model, batch)
    assert torch.isfinite(terms["loss"])


def test_gate_a_conditioning_controls_have_finite_objectives():
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    matcher = RiemannianCrystalFlowMatcher()
    for mode in ("raw_tensor", "direct_irrep", "stabilizer_pooling", "orbit_alignment"):
        model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3, conditioning_mode=mode)
        assert torch.isfinite(matcher.loss(model, batch)["loss"]), mode


def test_residual_conditional_field_keeps_base_condition_free_and_records_three_heads():
    torch.manual_seed(101)
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(
        hidden_dim=32, layers=2, orbit_frames=3, conditioning_mode="direct_irrep",
        conditional_control="residual_field", residual_g_min=0.25,
    ).eval()
    matcher = RiemannianCrystalFlowMatcher()
    state = matcher.target_state(batch)
    time = torch.tensor([0.0, 0.5])
    output = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, batch.condition_present, return_velocity_components=True,
    )
    components = output[4]
    assert torch.allclose(components["gate"], torch.tensor([0.25, 1.0]))
    assert components["type_base"].shape == output[0].shape
    assert components["coordinate_conditional_residual"].shape == output[1].shape
    assert components["lattice_conditional_residual"].shape == output[2].shape
    changed = batch.piezo_irreps.flip(0)
    alternate = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        changed, batch.condition_present, return_velocity_components=True,
    )[4]
    assert torch.allclose(components["type_base"], alternate["type_base"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(components["coordinate_base"], alternate["coordinate_base"], atol=1e-7, rtol=1e-7)


def test_counterfactual_tangent_ranking_is_finite_and_preserves_zero_null_separation():
    torch.manual_seed(103)
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.zeros(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(
        hidden_dim=32, layers=1, orbit_frames=3, conditioning_mode="direct_irrep",
        conditional_control="residual_field",
    )
    matcher = RiemannianCrystalFlowMatcher()
    terms = matcher.loss(model, batch, counterfactual_weight=0.25, counterfactual_margin=0.1)
    assert torch.isfinite(terms["loss"])
    assert torch.isfinite(terms["counterfactual"])
    assert terms["counterfactual"] > 0
    terms["loss"].backward()
    assert model.delta_type_out.weight.grad is not None
    state = matcher.target_state(batch)
    time = torch.full((batch.num_graphs,), 0.5)
    present = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, batch.condition_present,
    )[0]
    null = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, torch.zeros_like(batch.condition_present),
    )[0]
    assert not torch.allclose(present, null)


def test_endpoint_id_control_uses_the_existing_backbone_without_a_tensor_condition():
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[1.0, 0.0]]),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([49, 7]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[0.0, 1.0]]),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(
        hidden_dim=32, layers=1, orbit_frames=3, conditioning_mode="endpoint_id"
    )
    terms = RiemannianCrystalFlowMatcher().loss(model, batch)
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    assert model.response.embedding[0].weight.grad is not None


def test_sampler_accepts_a_fixed_initial_state_and_preserves_inactive_subspaces():
    class ConstantVelocity(torch.nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity

        def forward(self, *args, **kwargs):
            del args, kwargs
            return (*self.velocity, torch.ones(1, 1))

    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.tensor([[0.9, 0.1, 0.2], [0.2, 0.3, 0.4]]),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    matcher = RiemannianCrystalFlowMatcher(active_heads=("type",))
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    velocity = (
        target.type_state - base.type_state,
        torch.zeros_like(target.frac_coords),
        torch.zeros_like(target.lattice_log),
    )
    sampled = matcher.sample(ConstantVelocity(velocity), batch, steps=8, initial_state=base)
    assert torch.allclose(sampled.type_state, target.type_state, atol=2e-6, rtol=2e-6)
    assert torch.allclose(sampled.frac_coords, target.frac_coords)
    assert torch.allclose(sampled.lattice_log, target.lattice_log)


def test_exact_production_path_velocity_closes_type_torus_and_spd_endpoints():
    class ConstantVelocity(torch.nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity

        def forward(self, *args, **kwargs):
            del args, kwargs
            return (*self.velocity, torch.ones(1, 1))

    torch.manual_seed(809)
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.tensor([[0.95, 0.1, 0.2], [0.05, 0.3, 0.4]]),
             lattice=torch.tensor([[3.0, 0.0, 0.0], [0.1, 3.5, 0.0], [0.2, 0.3, 4.0]]).unsqueeze(0),
             piezo_irreps=torch.randn(1, 18), condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    matcher = RiemannianCrystalFlowMatcher()
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    velocity = (
        target.type_state - base.type_state,
        torus_logmap(base.frac_coords, target.frac_coords),
        target.lattice_log - base.lattice_log,
    )
    sampled = matcher.sample(ConstantVelocity(velocity), batch, steps=16, initial_state=base)
    assert torch.allclose(sampled.type_state, target.type_state, atol=3e-6, rtol=3e-6)
    assert torch.allclose(torus_logmap(sampled.frac_coords, target.frac_coords), torch.zeros_like(target.frac_coords), atol=3e-6, rtol=3e-6)
    assert torch.allclose(sampled.lattice_log, target.lattice_log, atol=3e-6, rtol=3e-6)


def test_simplex_type_path_preserves_probabilities_and_decodes_the_endpoint():
    class ConstantVelocity(torch.nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity

        def forward(self, *args, **kwargs):
            del args, kwargs
            return (*self.velocity, torch.ones(1, 1))

    torch.manual_seed(811)
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    matcher = RiemannianCrystalFlowMatcher(active_heads=("type",), type_path="simplex_probability")
    target = matcher.target_state(batch)
    base = matcher.random_state(batch)
    assert torch.all(base.type_state >= 0)
    assert torch.allclose(base.type_state.sum(dim=-1), torch.ones(base.type_state.shape[0]))
    sampled = matcher.sample(
        ConstantVelocity((target.type_state - base.type_state, torch.zeros_like(target.frac_coords), torch.zeros_like(target.lattice_log))),
        batch, steps=12, initial_state=base,
    )
    assert torch.all(sampled.type_state >= 0)
    assert torch.allclose(sampled.type_state.sum(dim=-1), torch.ones(sampled.type_state.shape[0]), atol=1e-6)
    assert torch.equal(sampled.type_state.argmax(-1), batch.atom_types)


def test_all_negative_early_identification_is_finite_and_requires_present_conditions():
    torch.manual_seed(107)
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
        Data(atom_types=torch.tensor([13, 7]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.randn(1, 18),
             condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2),
    ])
    model = GaugeFlowVectorField(
        hidden_dim=32, layers=1, orbit_frames=3, conditioning_mode="direct_irrep"
    )
    matcher = RiemannianCrystalFlowMatcher()
    terms = matcher.loss(
        model, batch, identification_weight=0.5,
        identification_temperature=0.25, identification_early_sigma=0.25,
    )
    assert torch.isfinite(terms["loss"])
    assert torch.isfinite(terms["identification"])
    assert 0.0 <= terms["identification_retrieval"] <= 1.0
    terms["loss"].backward()
    assert model.response.candidate[0].weight.grad is not None
    batch.condition_present[1] = False
    try:
        matcher.loss(
            model, batch, identification_weight=0.5,
            identification_temperature=0.25, identification_early_sigma=0.25,
        )
    except ValueError as error:
        assert "present physical tensor conditions" in str(error)
    else:
        raise AssertionError("Null CFG conditions must be rejected by all-negative identification")


def test_direct_irrep_cartesian_products_are_so3_equivariant_without_harmonics():
    torch.manual_seed(11)
    tensor = piezo_voigt_to_cartesian(torch.randn(3, 6)).unsqueeze(0)
    directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    edge_graph = torch.zeros(5, dtype=torch.long)
    rotation = fixed_so3_frames(2)[1]
    primary, auxiliary = direct_irrep_cartesian_products(tensor, directions, edge_graph)
    rotated_primary, rotated_auxiliary = direct_irrep_cartesian_products(
        rotate_rank3(tensor, rotation), directions @ rotation.T, edge_graph
    )
    assert torch.allclose(primary @ rotation.T, rotated_primary, atol=2e-5, rtol=2e-5)
    assert torch.allclose(auxiliary @ rotation.T, rotated_auxiliary, atol=2e-5, rtol=2e-5)


def test_cfg_dropout_trains_a_null_condition_without_conflating_a_zero_tensor():
    present = torch.ones(3, 1, dtype=torch.bool)
    assert apply_condition_dropout(present, 0.0).all()
    dropped = apply_condition_dropout(present, 1.0)
    assert not dropped.any()
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([14, 8]), frac_coords=torch.rand(2, 3),
             lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.zeros(1, 18),
             condition_present=present[:1], num_nodes=2),
    ])
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3)
    batch.condition_present = apply_condition_dropout(batch.condition_present, 1.0)
    terms = RiemannianCrystalFlowMatcher().loss(model, batch)
    terms["loss"].backward()
    assert model.response.null_condition.grad is not None
    assert model.response.null_condition.grad.abs().sum() > 0


def test_direct_irrep_random_frame_preserves_the_complete_tensor_norm():
    torch.manual_seed(7)
    condition = torch.randn(3, 18)
    randomized = randomize_tensor_orbit_representative(
        condition, generator=torch.Generator().manual_seed(13)
    )
    assert torch.allclose(
        piezo_from_irreps(condition).square().sum(dim=(-1, -2, -3)),
        piezo_from_irreps(randomized).square().sum(dim=(-1, -2, -3)),
        atol=2e-5,
        rtol=2e-5,
    )


def test_stabilizer_pooling_condition_path_is_state_independent():
    torch.manual_seed(31)
    encoder = OrbitResponseFieldEncoder(16, orbit_frames=4, mode="stabilizer_pooling").eval()
    condition = torch.randn(1, 18)
    present = torch.ones(1, 1, dtype=torch.bool)
    query = torch.randn(1, 16)
    directions = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    edge_graph = torch.zeros(2, dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    first = encoder(
        condition, present, query, directions, edge_graph,
        torch.rand(2, 3), torch.eye(3).unsqueeze(0), batch, torch.randn(2, 5),
    )
    second = encoder(
        condition, present, query, directions, edge_graph,
        torch.rand(2, 3), (3.0 * torch.eye(3)).unsqueeze(0), batch, torch.randn(2, 5),
    )
    for left, right in zip(first, second):
        assert torch.allclose(left, right, atol=1e-7, rtol=1e-7)


def test_cached_condition_orbit_matches_forward_gradients_loss_and_optimizer_update():
    torch.manual_seed(41)
    batch = Batch.from_data_list([
        Data(
            atom_types=torch.tensor([14, 8]),
            frac_coords=torch.rand(2, 3),
            lattice=torch.tensor([[3.1, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 4.2]]).unsqueeze(0),
            piezo_irreps=torch.randn(1, 18),
            condition_present=torch.ones(1, 1, dtype=torch.bool),
            num_nodes=2,
        )
    ])
    uncached_model = GaugeFlowVectorField(
        hidden_dim=16, layers=1, orbit_frames=4, conditioning_mode="stabilizer_pooling"
    )
    cached_model = copy.deepcopy(uncached_model)
    cached_batch = batch.clone()
    with torch.no_grad():
        cached_batch.condition_orbit = cached_model.response.precompute_condition_orbit(
            cached_batch.piezo_irreps
        )

    matcher = RiemannianCrystalFlowMatcher()
    torch.manual_seed(53)
    uncached_terms = matcher.loss(uncached_model, batch)
    torch.manual_seed(53)
    cached_terms = matcher.loss(cached_model, cached_batch)
    assert torch.allclose(uncached_terms["loss"], cached_terms["loss"], atol=1e-7, rtol=1e-7)

    fixed_state = matcher.target_state(batch)
    fixed_time = torch.full((batch.num_graphs,), 0.5)
    uncached_velocity = uncached_model(
        fixed_state.type_state,
        fixed_state.frac_coords,
        fixed_state.lattice_log,
        batch.batch,
        fixed_time,
        batch.piezo_irreps,
        batch.condition_present,
    )
    cached_velocity = cached_model(
        fixed_state.type_state,
        fixed_state.frac_coords,
        fixed_state.lattice_log,
        cached_batch.batch,
        fixed_time,
        cached_batch.piezo_irreps,
        cached_batch.condition_present,
        cached_batch.condition_orbit,
    )
    for left, right in zip(uncached_velocity, cached_velocity):
        assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)
    uncached_terms["loss"].backward()
    cached_terms["loss"].backward()
    for left, right in zip(uncached_model.parameters(), cached_model.parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            assert torch.allclose(left.grad, right.grad, atol=2e-6, rtol=2e-6)
    left_optimizer = torch.optim.SGD(uncached_model.parameters(), lr=1e-3)
    right_optimizer = torch.optim.SGD(cached_model.parameters(), lr=1e-3)
    left_optimizer.step()
    right_optimizer.step()
    for left, right in zip(uncached_model.parameters(), cached_model.parameters()):
        assert torch.allclose(left, right, atol=2e-7, rtol=2e-7)

    first_condition = batch.piezo_irreps.detach().clone().requires_grad_()
    second_condition = batch.piezo_irreps.detach().clone().requires_grad_()
    encoder = uncached_model.response.eval()
    graph_query = torch.randn(1, 16)
    directions = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    edge_graph = torch.zeros(2, dtype=torch.long)
    graph_batch = torch.zeros(2, dtype=torch.long)
    common = (
        batch.condition_present,
        graph_query,
        directions,
        edge_graph,
        batch.frac_coords,
        batch.lattice[0].unsqueeze(0),
        graph_batch,
        torch.randn(2, 5),
    )
    uncached = encoder(first_condition, *common)
    cached_orbit = encoder.precompute_condition_orbit(second_condition)
    cached = encoder(second_condition, *common, cached_orbit)
    assert torch.allclose(uncached[0], cached[0], atol=1e-7, rtol=1e-7)
    assert torch.allclose(uncached[1], cached[1], atol=1e-7, rtol=1e-7)
    first_gradient = torch.autograd.grad(uncached[0].sum() + uncached[1].sum(), first_condition)[0]
    second_gradient = torch.autograd.grad(cached[0].sum() + cached[1].sum(), second_condition)[0]
    assert torch.allclose(first_gradient, second_gradient, atol=2e-6, rtol=2e-6)


def test_cached_orbit_matches_uncached_for_random_so3_representatives():
    torch.manual_seed(67)
    encoder = OrbitResponseFieldEncoder(16, orbit_frames=8, mode="stabilizer_pooling").eval()
    condition = torch.randn(2, 18)
    present = torch.ones(2, 1, dtype=torch.bool)
    query = torch.randn(2, 16)
    directions = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    edge_graph = torch.tensor([0, 0, 1, 1])
    frac = torch.rand(4, 3)
    lattices = torch.stack((torch.eye(3), 1.3 * torch.eye(3)))
    batch = edge_graph.clone()
    types = torch.randn(4, 5)
    for repeat in range(4):
        representative = randomize_tensor_orbit_representative(
            condition, generator=torch.Generator().manual_seed(100 + repeat)
        )
        uncached = encoder(
            representative, present, query, directions, edge_graph,
            frac, lattices, batch, types,
        )
        cached_orbit = encoder.precompute_condition_orbit(representative)
        cached = encoder(
            representative, present, query, directions, edge_graph,
            frac, lattices, batch, types, cached_orbit,
        )
        for left, right in zip(uncached, cached):
            assert torch.allclose(left, right, atol=2e-6, rtol=2e-6)
