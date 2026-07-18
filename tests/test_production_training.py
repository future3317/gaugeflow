from pathlib import Path

import torch

from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.checkpointing import load_production_checkpoint, save_production_checkpoint
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.reverse_sampler import (
    TensorFreeReverseSampler,
    quotient_coordinate_reverse_step,
)
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig


def _small_clean_batch():
    counts = torch.tensor([2, 3])
    blueprint = ParentBlueprintBatch.from_node_counts(counts)
    elements = torch.tensor([4, 6, 12, 15, 6], dtype=torch.long)
    coordinates = torch.tensor(
        [
            [0.05, 0.10, 0.15],
            [0.35, 0.25, 0.70],
            [0.15, 0.75, 0.45],
            [0.72, 0.55, 0.20],
            [0.42, 0.31, 0.82],
        ]
    )
    lattice = torch.stack((3.0 * torch.eye(3), torch.diag(torch.tensor([3.5, 4.0, 4.5]))))
    return elements, coordinates, lattice, blueprint


def _small_model() -> HybridCrystalDenoiser:
    return HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
    )


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1]
        / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def test_tensor_free_loss_is_finite_and_bypasses_cartesian_candidates():
    torch.manual_seed(101)
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(
        _small_model(), _standardizer(), coordinate_sigma_min=0.005, coordinate_sigma_max=0.5
    )
    output = diffusion(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=torch.tensor([0.4, 0.7]),
        generator=torch.Generator().manual_seed(102),
    )
    assert torch.isfinite(output.loss)
    assert output.loss > 0
    assert torch.equal(
        output.prediction.gauge_atlas.effective_frame_count,
        torch.zeros(2, dtype=torch.long),
    )
    for graph in range(2):
        selected = output.noisy.coordinate_scaled_score_target[blueprint.batch == graph]
        assert torch.allclose(selected.mean(0), torch.zeros(3), atol=2e-6)
    output.loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in diffusion.denoiser.parameters()
    )


def test_tensor_free_path_skips_geometry_query_encoder():
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = _small_model()
    calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.geometry_query_encoder.register_forward_hook(count_calls)
    diffusion = TensorFreeHybridDiffusion(model, _standardizer())
    diffusion(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=torch.tensor([0.3, 0.6]),
        generator=torch.Generator().manual_seed(106),
    )
    handle.remove()
    assert calls == 0


def test_randomized_stratified_times_preserve_uniform_batch_coverage():
    diffusion = TensorFreeHybridDiffusion(_small_model(), _standardizer())
    graph_count = 16
    sampled = diffusion.sample_time(
        graph_count,
        torch.zeros(1),
        generator=torch.Generator().manual_seed(107),
    )
    unit = (sampled - diffusion.minimum_time) / (
        diffusion.maximum_time - diffusion.minimum_time
    )
    observed_strata = torch.floor(unit.sort().values * graph_count).long()
    assert torch.equal(observed_strata, torch.arange(graph_count))


def test_production_trainer_updates_ema_and_all_heads():
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    config = ProductionTrainingConfig(learning_rate=1.0e-3, ema_decay=0.9)
    diffusion = TensorFreeHybridDiffusion(
        _small_model(), _standardizer(), coordinate_sigma_min=0.005, coordinate_sigma_max=0.5
    )
    trainer = ProductionTrainer(diffusion, config)
    before = {name: value.clone() for name, value in diffusion.denoiser.state_dict().items()}
    output, gradient_norm = trainer.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(103),
    )
    assert trainer.step == 1 and gradient_norm > 0
    assert all(
        torch.isfinite(value)
        for value in (
            output.element_loss,
            output.coordinate_loss,
            output.volume_loss,
            output.shape_loss,
        )
    )
    assert any(not torch.equal(value, before[name]) for name, value in diffusion.denoiser.state_dict().items())


def test_joint_reverse_sampler_reveals_elements_and_projects_state():
    torch.manual_seed(104)
    model = _small_model()
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2, 3]))
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        coordinate_sigma_min=0.005,
        coordinate_sigma_max=0.5,
        maximum_time=0.8,
    )
    generated = sampler.sample(
        blueprint,
        steps=4,
        stochastic=False,
        generator=torch.Generator().manual_seed(105),
    )
    assert generated.atomic_numbers.min() >= 1 and generated.atomic_numbers.max() <= 118
    assert generated.fractional_coordinates.shape == (5, 3)
    assert generated.lattice.shape == (2, 3, 3)
    assert torch.isfinite(generated.lattice).all()
    assert generated.diagnostics.masked_count[-1] == 0
    for graph in range(2):
        assert torch.allclose(
            generated.log_shape[graph].dot(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])),
            torch.tensor(0.0),
            atol=2e-5,
        )


def test_quotient_coordinate_reverse_step_matches_deterministic_score_drift():
    coordinates = torch.tensor(
        [[-0.2, 0.1, 0.1], [0.2, -0.1, -0.1], [0.0, 0.3, -0.3]]
    )
    score = torch.tensor(
        [[0.4, 0.2, -0.2], [-0.4, -0.2, 0.2], [0.0, 0.0, 0.0]]
    )
    batch = torch.tensor([0, 0, 1])
    observed = quotient_coordinate_reverse_step(
        coordinates,
        score,
        torch.tensor([0.25, 0.25]),
        torch.tensor([0.16, 0.16]),
        batch,
        2,
        generator=None,
        stochastic=False,
    )
    expected = coordinates + 0.09 * score / 0.5
    expected[:2] -= expected[:2].mean(0)
    expected[2] = 0.0
    assert torch.allclose(observed, expected)


def test_production_checkpoint_restores_model_optimizer_ema_rng_and_count_prior(tmp_path: Path):
    model = _small_model()
    diffusion = TensorFreeHybridDiffusion(model, _standardizer())
    trainer = ProductionTrainer(diffusion, ProductionTrainingConfig())
    prior = EmpiricalNodeCountPrior.fit(torch.tensor([2, 2, 3, 4]))
    path = tmp_path / "production.pt"
    save_production_checkpoint(
        path,
        model=model,
        ema=trainer.ema,
        optimizer=trainer.optimizer,
        training_step=17,
        node_count_prior=prior,
        metadata={"model": {"hidden_dim": 16}, "protocol": "s1a_tensor_free_v1"},
    )
    restored_model = _small_model()
    restored_diffusion = TensorFreeHybridDiffusion(restored_model, _standardizer())
    restored_trainer = ProductionTrainer(restored_diffusion, ProductionTrainingConfig())
    step, restored_prior, metadata = load_production_checkpoint(
        path,
        model=restored_model,
        ema=restored_trainer.ema,
        optimizer=restored_trainer.optimizer,
    )
    assert step == 17 and metadata["protocol"] == "s1a_tensor_free_v1"
    assert torch.equal(restored_prior.support, prior.support)
    assert torch.equal(restored_prior.probabilities, prior.probabilities)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
