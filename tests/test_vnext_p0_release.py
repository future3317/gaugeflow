"""P0 regressions required before the versioned Q1 protocol can run."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from gaugeflow.checkpoints import load_safe_checkpoint, save_safe_checkpoint
from gaugeflow.flow import RiemannianCrystalFlowMatcher
from gaugeflow.model import GaugeFlowVectorField
from gaugeflow.stabilizer import proper_unimodular_candidates
from gaugeflow.vnext.experiments.p0_release_audit import _verify_run_manifest
from gaugeflow.vocabulary import CHEMICAL_ELEMENT_COUNT, tokens_to_atomic_numbers


def _batch() -> Batch:
    return Batch.from_data_list(
        [
            Data(
                atom_types=torch.tensor([5, 7, 13]),
                frac_coords=torch.tensor([[0.07, 0.11, 0.19], [0.34, 0.22, 0.31], [0.72, 0.48, 0.41]]),
                lattice=torch.tensor([[3.9, 0.2, 0.1], [0.3, 4.3, 0.4], [0.1, 0.4, 5.1]]).unsqueeze(0),
                piezo_irreps=torch.randn(1, 18),
                condition_present=torch.ones((1, 1), dtype=torch.bool),
                num_nodes=3,
            )
        ]
    )


def _outputs_at_times(model: GaugeFlowVectorField, batch: Batch) -> tuple[tuple[torch.Tensor, ...], ...]:
    state = RiemannianCrystalFlowMatcher().target_state(batch)
    outputs = []
    for value in (0.17, 0.83):
        time = torch.tensor([value])
        if model.conditioning_mode == "unconditional":
            output = model(state.type_state, state.frac_coords, state.lattice_log, batch.batch, time)
        elif model.conditioning_mode == "endpoint_id":
            endpoint_id = torch.tensor([[1.0, 0.0]])
            output = model(
                state.type_state,
                state.frac_coords,
                state.lattice_log,
                batch.batch,
                time,
                endpoint_id,
                batch.condition_present,
            )
        else:
            output = model(
                state.type_state,
                state.frac_coords,
                state.lattice_log,
                batch.batch,
                time,
                batch.piezo_irreps,
                batch.condition_present,
            )
        outputs.append(output)
    return tuple(outputs)


def test_time_reaches_every_conditioning_mode():
    torch.manual_seed(7201)
    batch = _batch()
    modes = (
        "unconditional",
        "endpoint_id",
        "raw_tensor",
        "direct_irrep",
        "direct_irrep_complete_v1",
        "invariant_only_v1",
        "stabilizer_pooling",
        "orbit_alignment",
        "harmonic_alignment_v1",
    )
    for mode in modes:
        frames = 12 if mode == "harmonic_alignment_v1" else 3
        model = GaugeFlowVectorField(hidden_dim=16, layers=2, orbit_frames=frames, conditioning_mode=mode).eval()
        with torch.no_grad():
            left, right = _outputs_at_times(model, batch)
        for head in range(3):
            difference = (left[head] - right[head]).abs().max()
            assert difference > 1.0e-7, (mode, head, float(difference))


def test_time_reaches_every_message_block():
    torch.manual_seed(7202)
    batch = _batch()
    model = GaugeFlowVectorField(
        hidden_dim=16,
        layers=3,
        orbit_frames=3,
        conditioning_mode="direct_irrep",
        conditional_control="residual_field",
    ).eval()
    calls = [0] * 6
    handles = []
    modules = [layer.time_film for layer in model.layers]
    modules += [layer.message.time_film for layer in model.conditional_layers]

    def hook_for(index):
        def record(_module, _inputs, _output):
            calls[index] += 1

        return record

    for index, module in enumerate(modules):
        handles.append(module.register_forward_hook(hook_for(index)))
    with torch.no_grad():
        _outputs_at_times(model, batch)
    for handle in handles:
        handle.remove()
    assert calls == [2] * 6


def test_coordinate_velocity_has_zero_graph_mean():
    matcher = RiemannianCrystalFlowMatcher(active_heads=("coord",))
    batch_index = torch.tensor([0, 0, 0, 1, 1])
    raw = torch.randn(5, 3)
    horizontal = matcher._coordinate_velocity(raw, type("Batch", (), {"batch": batch_index, "num_graphs": 2})())
    for graph in range(2):
        assert torch.allclose(horizontal[batch_index == graph].mean(dim=0), torch.zeros(3), atol=1.0e-7)


def test_all_792_actions_or_declared_topk_contract():
    assert proper_unimodular_candidates().shape == (792, 3, 3)


def test_atom_type_decode_never_emits_invalid_atomic_number():
    logits = torch.randn(256, CHEMICAL_ELEMENT_COUNT)
    tokens = logits.argmax(dim=-1)
    atomic_numbers = tokens_to_atomic_numbers(tokens)
    assert int(atomic_numbers.min()) >= 1
    assert int(atomic_numbers.max()) <= 118


def test_safe_checkpoint_loading_contract(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "weights.pt"
    sidecar = save_safe_checkpoint(
        path,
        model_state=model.state_dict(),
        isotypic_scales=torch.ones(3),
        training_step=17,
        metadata={"config": {"hidden_dim": 8}, "source_hash": "abc"},
    )
    assert sidecar == tmp_path / "weights.pt.json"
    payload, metadata = load_safe_checkpoint(path, map_location="cpu")
    assert payload["training_step"] == 17
    assert metadata["source_hash"] == "abc"
    with path.open("ab") as handle:
        handle.write(b"tamper")
    try:
        load_safe_checkpoint(path, map_location="cpu")
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered checkpoint must be rejected before torch.load")


def test_run_manifest_reproduces_all_published_csv_hashes():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs" / "Q0" / "20260715T182701Z_7af3ca57bff6"
    assert _verify_run_manifest(run)
