import torch
from torch_geometric.data import Batch, Data

from gaugeflow.discrete import AbsorbingDiscreteTypeFlowMatcher, DiscreteSamplingNoise
from gaugeflow.model import GaugeFlowVectorField


def _batch():
    return Batch.from_data_list([
        Data(
            atom_types=torch.tensor([5, 7]), frac_coords=torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.2, 0.3]]),
            lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[1.0, 0.0]]),
            condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2,
        ),
        Data(
            atom_types=torch.tensor([49, 7]), frac_coords=torch.tensor([[0.2, 0.3, 0.4], [0.8, 0.3, 0.4]]),
            lattice=torch.eye(3).unsqueeze(0), piezo_irreps=torch.tensor([[0.0, 1.0]]),
            condition_present=torch.ones(1, 1, dtype=torch.bool), num_nodes=2,
        ),
    ])


class _OraclePosterior(torch.nn.Module):
    def __init__(self, targets: torch.Tensor, state_dim: int):
        super().__init__()
        self.targets = targets
        self.state_dim = state_dim

    def forward(self, type_state, frac_coords, lattice_log, batch, time, *args, **kwargs):
        del type_state, time, args, kwargs
        logits = torch.full((self.targets.numel(), self.state_dim), -30.0, device=self.targets.device)
        logits.scatter_(1, self.targets.unsqueeze(-1), 30.0)
        return logits, torch.zeros_like(frac_coords), torch.zeros_like(lattice_log), torch.ones((lattice_log.shape[0], 1), device=logits.device)


def test_absorbing_discrete_sampler_has_exact_oracle_endpoint_closure():
    batch = _batch()
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    model = _OraclePosterior(batch.atom_types, matcher.state_dim)
    steps = 5
    noise = DiscreteSamplingNoise(
        reveal_uniform=torch.rand(steps, batch.atom_types.numel()),
        categorical_uniform=torch.rand(steps, batch.atom_types.numel(), matcher.atom_types),
    )
    state, trajectory = matcher.sample(model, batch, steps=steps, noise=noise, return_trajectory=True)
    assert len(trajectory) == steps + 1
    assert torch.equal(state.type_state.argmax(-1), batch.atom_types)
    assert not torch.any(state.type_state.argmax(-1) == matcher.mask_index)


def test_discrete_endpoint_posterior_loss_and_sampling_are_finite():
    batch = _batch()
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, atom_types=matcher.state_dim, conditioning_mode="endpoint_id")
    terms = matcher.loss(model, batch)
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    state = matcher.sample(model, batch, steps=3)
    assert state.type_state.shape[-1] == matcher.state_dim
    assert not torch.any(state.type_state.argmax(-1) == matcher.mask_index)


def test_graph_count_constraint_preserves_the_sampled_chemical_multiset():
    batch = _batch()
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    model = _OraclePosterior(batch.atom_types, matcher.state_dim)
    counts = matcher.composition_count_targets(batch)
    state = matcher.sample(model, batch, steps=4, graph_counts=counts)
    tokens = state.type_state.argmax(-1)
    for graph in range(batch.num_graphs):
        nodes = batch.batch == graph
        assert torch.equal(torch.bincount(tokens[nodes], minlength=matcher.atom_types), counts[graph])


def test_composition_count_head_has_a_finite_loss_and_constrained_map():
    batch = _batch()
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    model = GaugeFlowVectorField(
        hidden_dim=32, layers=1, atom_types=matcher.state_dim, conditioning_mode="endpoint_id",
        composition_max_atoms=2, composition_atom_types=matcher.atom_types,
    )
    state = matcher.mask_state(batch)
    time = torch.zeros(batch.num_graphs)
    logits = model(
        state.type_state, state.frac_coords, state.lattice_log, batch.batch, time,
        batch.piezo_irreps, batch.condition_present, return_composition_counts=True,
    )[-1]
    loss, target = matcher.composition_count_loss(logits, batch)
    assert torch.isfinite(loss)
    loss.backward()
    for graph in range(batch.num_graphs):
        counts = matcher.map_composition_counts(logits[graph].detach(), atom_count=2)
        assert int(counts.sum()) == int(target[graph].sum())


def test_original_injection_endpoint_id_field_is_explicitly_time_conditioned():
    batch = _batch()
    matcher = AbsorbingDiscreteTypeFlowMatcher()
    model = GaugeFlowVectorField(hidden_dim=32, layers=1, atom_types=matcher.state_dim, conditioning_mode="endpoint_id")
    model.eval()
    state = matcher.mask_state(batch)
    with torch.no_grad():
        early = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch,
            torch.zeros(batch.num_graphs), batch.piezo_irreps, batch.condition_present,
        )[0]
        late = model(
            state.type_state, state.frac_coords, state.lattice_log, batch.batch,
            torch.ones(batch.num_graphs), batch.piezo_irreps, batch.condition_present,
        )[0]
    assert not torch.allclose(early, late)


def test_source_biased_discrete_time_weighting_increases_low_time_coverage():
    torch.manual_seed(7)
    uniform = AbsorbingDiscreteTypeFlowMatcher(training_time_distribution="uniform").training_time(20000, torch.device("cpu"))
    torch.manual_seed(7)
    source_biased = AbsorbingDiscreteTypeFlowMatcher(training_time_distribution="beta_half_source").training_time(20000, torch.device("cpu"))
    assert source_biased.mean() < uniform.mean()
    assert (1.0 - source_biased).pow(4).mean() > (1.0 - uniform).pow(4).mean()
