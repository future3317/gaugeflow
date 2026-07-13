import torch

from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField, ResponseMessageLayer
from gaugeflow.tensor import fixed_so3_frames
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
    directions, response, condition = torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 8)
    with torch.no_grad():
        original = layer(nodes, vectors, source, target, directions, response, condition)
        rotated = layer(
            nodes,
            vectors @ rotation.T,
            source,
            target,
            directions @ rotation.T,
            response @ rotation.T,
            condition,
        )
    assert torch.allclose(original[0], rotated[0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(original[1] @ rotation.T, rotated[1], atol=2e-5, rtol=2e-5)


def test_flow_accepts_variable_proper_stabilizers():
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
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, orbit_frames=3)
    terms = RiemannianCrystalFlowMatcher().loss(model, batch)
    assert torch.isfinite(terms["loss"])
