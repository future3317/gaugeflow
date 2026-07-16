import torch
from torch_geometric.data import Batch, Data

from gaugeflow.manifold import lattice_to_log_vector, wrap01
from gaugeflow.model import FourierIntervalEmbedding, QuotientRolloutFlowMap
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT


def test_interval_embedding_rejects_degenerate_intervals_and_flow_map_is_translation_invariant():
    embedding = FourierIntervalEmbedding(hidden_dim=16, frequencies=4)
    with torch.no_grad():
        try:
            embedding(torch.tensor([0.5]), torch.tensor([0.5]))
        except ValueError:
            pass
        else:
            raise AssertionError("D0.6 must not expose a zero-interval time fallback")
    batch = Batch.from_data_list([
        Data(atom_types=torch.tensor([5, 7, 14]), frac_coords=torch.tensor([[0.07, 0.11, 0.19], [0.34, 0.22, 0.31], [0.72, 0.48, 0.41]]), lattice=torch.eye(3).unsqueeze(0), num_nodes=3)
    ])
    model = QuotientRolloutFlowMap(hidden_dim=32, layers=2, coordinate_rbf_dim=8)
    type_state = torch.nn.functional.one_hot(batch.atom_types, CHEMICAL_ELEMENT_COUNT).float()
    lattice_log = lattice_to_log_vector(batch.lattice)
    start, end = torch.tensor([0.2]), torch.tensor([0.7])
    original = model(type_state, batch.frac_coords, lattice_log, batch.batch, start, end)
    translated = model(type_state, wrap01(batch.frac_coords + torch.tensor([0.17, 0.29, 0.41])), lattice_log, batch.batch, start, end)
    assert torch.allclose(original, translated, atol=2e-6, rtol=2e-6)
    original.square().mean().backward()
    assert model.blocks[0].film.weight.grad.abs().sum() > 0
    assert model.coordinate_edge_out[-1].weight.grad.abs().sum() > 0
