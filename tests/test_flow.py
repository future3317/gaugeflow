import torch

from gaugeflow.conditioning import apply_condition_dropout, randomize_tensor_orbit_representative
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import (
    GaugeFlowVectorField,
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
