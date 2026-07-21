import copy
from pathlib import Path

import torch

from gaugeflow.production.blueprint import EmpiricalNodeCountPrior, ParentBlueprintBatch
from gaugeflow.production.checkpointing import (
    load_production_checkpoint,
    load_production_runtime_state,
    save_production_checkpoint,
)
from gaugeflow.production.composition_state import IntegerPartitionCatalogue, StoichiometryFirstCompositionModel
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.reverse_sampler import (
    ContinuousReverseInitialState,
    TensorFreeReverseSampler,
    quotient_coordinate_reverse_step,
    vp_reverse_step,
)
from gaugeflow.production.schedules import CosineNoiseSchedule
from gaugeflow.production.state_projection import fractional_tangent_to_cartesian
from gaugeflow.production.training import ProductionTrainer, ProductionTrainingConfig
from scripts.train_production import (
    _GRADIENT_GROUPS,
    _clipped_module_gradient_norms,
    _gradient_group,
    _validate_data_exposure,
)


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
        Path(__file__).parents[1] / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def test_coordinate_exposure_contract_accepts_complete_passes_and_explicit_prefix() -> None:
    _validate_data_exposure(
        {"data_passes": 2.0, "graph_presentations": 1_080_328},
        dataset_size=540_164,
        steps=16_882,
        batch_size=64,
    )
    _validate_data_exposure(
        {
            "exposure_mode": "prefix_screen",
            "data_passes": 135_104 / 540_164,
            "graph_presentations": 135_104,
        },
        dataset_size=540_164,
        steps=2111,
        batch_size=64,
    )
    for invalid in (
        {"data_passes": True, "graph_presentations": 540_164},
        {"data_passes": "1.0", "graph_presentations": 540_164},
        {"data_passes": 1.0, "graph_presentations": 540_164.0},
        {"data_passes": 1.5, "graph_presentations": 810_246},
        {"data_passes": 2.0, "graph_presentations": 1_080_327},
        {
            "exposure_mode": "prefix_screen",
            "data_passes": 0.25,
            "graph_presentations": 135_104,
        },
        {
            "exposure_mode": "prefix_screen",
            "data_passes": 135_104 / 540_164,
            "graph_presentations": 135_103,
        },
    ):
        try:
            _validate_data_exposure(invalid, dataset_size=540_164, steps=16_882, batch_size=64)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid exposure contract was accepted")


def test_training_gradient_partition_is_complete_and_reports_clipped_norms() -> None:
    model = _small_model()
    for name, parameter in model.named_parameters():
        assert _gradient_group(name) in _GRADIENT_GROUPS
        parameter.grad = torch.ones_like(parameter)
    norms = _clipped_module_gradient_norms(model)
    assert set(norms) == set(_GRADIENT_GROUPS)
    assert all(value > 0.0 for value in norms.values())


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
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in diffusion.denoiser.parameters()
    )


def _small_product_model() -> HybridCrystalDenoiser:
    return HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )


def _small_composition_model() -> StoichiometryFirstCompositionModel:
    catalogue = IntegerPartitionCatalogue.build(maximum_atoms=20, maximum_species=7)
    log_prior = torch.full((catalogue.size,), -torch.inf, dtype=torch.float64)
    for count in range(1, 21):
        support = catalogue.node_count == count
        log_prior[support] = -torch.log(support.sum().to(torch.float64))
    return StoichiometryFirstCompositionModel(
        context_dim=1,
        hidden_dim=8,
        partition_log_prior=log_prior,
        maximum_atoms=20,
        maximum_species=7,
    ).float()


def test_orderless_product_path_conditions_on_exact_composition_and_has_finite_gradients():
    torch.manual_seed(110)
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(
        _small_model(),
        _standardizer(),
        coordinate_sigma_min=0.005,
        coordinate_sigma_max=0.5,
        categorical_path="orderless_reveal",
        composition_conditioning=True,
    )
    output = diffusion(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=torch.tensor([0.35, 0.65]),
        generator=torch.Generator().manual_seed(111),
    )
    state = output.noisy.orderless_occupation
    assert state is not None
    assert output.noisy.composition_counts is not None
    assert torch.equal(state.composition_counts, output.noisy.composition_counts)
    assert torch.equal(state.remaining_counts.sum(dim=1), torch.bincount(
        blueprint.batch, minlength=2
    ) - state.reveal_count)
    expected_log_composition = output.noisy.composition_counts.to(torch.float32)
    expected_log_composition = expected_log_composition / expected_log_composition.sum(dim=1, keepdim=True)
    assert torch.allclose(
        output.prediction.clean_composition_logits.float(),
        expected_log_composition.clamp_min(1.0e-8).log(),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in diffusion.denoiser.parameters()
    )


def test_orderless_product_path_requires_joint_composition_conditioning() -> None:
    try:
        ProductionTrainingConfig(objective="joint", categorical_path="orderless_reveal").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("orderless product path accepted without sampled composition")
    try:
        ProductionTrainingConfig(objective="coordinate", composition_conditioning=True).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("coordinate-only objective accepted product composition conditioning")


def test_coordinate_loss_uses_volume_normalized_cartesian_chart() -> None:
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
        time=torch.tensor([0.2, 0.2]),
        generator=torch.Generator().manual_seed(108),
    )
    noisy_lattice = LatticeVolumeShape(output.noisy.log_volume, output.noisy.log_shape).lattice(
        blueprint.fractional_to_cartesian
    )
    target = fractional_tangent_to_cartesian(
        output.noisy.coordinate_scaled_score_target,
        noisy_lattice,
        blueprint.batch,
    )
    scale = torch.exp(output.noisy.log_volume / 3.0)[blueprint.batch, None]
    error = (output.prediction.coordinate_cartesian_scaled_score - target) / scale
    graph_loss = torch.stack(
        [error[blueprint.batch == graph].square().sum(-1).mean() / 3.0 for graph in range(2)]
    ).mean()
    torch.testing.assert_close(output.coordinate_loss, graph_loss)


def test_coordinate_clean_side_information_noises_only_coordinates() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(
        _small_model(), _standardizer(), coordinate_sigma_min=0.005, coordinate_sigma_max=0.5
    )
    noisy = diffusion.noise_clean_batch(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=torch.tensor([0.4, 0.7]),
        generator=torch.Generator().manual_seed(109),
        clean_side_information=True,
    )
    clean_lattice_state = LatticeVolumeShape.from_lattice(lattice, blueprint.fractional_to_cartesian)
    assert torch.equal(noisy.element_tokens, elements)
    assert not bool(noisy.element_was_masked.any())
    torch.testing.assert_close(noisy.log_volume, clean_lattice_state.log_volume)
    torch.testing.assert_close(noisy.log_shape, clean_lattice_state.log_shape)
    assert not torch.allclose(noisy.fractional_coordinates, coordinates)

    # The observed-side-information branch must not consume categorical or
    # lattice random draws.  Coordinate noise is therefore reproducible from
    # the first draw of an otherwise identical generator.
    repeated = diffusion.noise_clean_batch(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=torch.tensor([0.4, 0.7]),
        generator=torch.Generator().manual_seed(109),
        clean_side_information=True,
    )
    torch.testing.assert_close(noisy.fractional_coordinates, repeated.fractional_coordinates)


def test_element_only_corruption_keeps_geometry_exact_and_zeroes_inactive_score() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(
        HybridCrystalDenoiser(
            hidden_dim=16,
            vector_dim=4,
            layers=1,
            radial_dim=4,
            atlas_residual_circle_samples=8,
            modality_time_conditioning="separate",
        ),
        _standardizer(),
    )
    clean_time = torch.zeros(2)
    noisy = diffusion.noise_clean_batch(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=clean_time,
        element_time=torch.tensor([0.4, 0.8]),
        lattice_time=clean_time,
        generator=torch.Generator().manual_seed(116),
    )
    clean_coordinates = coordinates.clone()
    for graph in range(2):
        selected = blueprint.batch == graph
        clean_coordinates[selected] -= clean_coordinates[selected].mean(dim=0)
    clean_lattice_state = LatticeVolumeShape.from_lattice(
        lattice,
        blueprint.fractional_to_cartesian,
    )
    torch.testing.assert_close(noisy.fractional_coordinates, clean_coordinates)
    torch.testing.assert_close(noisy.log_volume, clean_lattice_state.log_volume)
    torch.testing.assert_close(noisy.log_shape, clean_lattice_state.log_shape)
    assert torch.equal(noisy.time, clean_time)
    assert torch.equal(noisy.lattice_time, clean_time)
    assert torch.count_nonzero(noisy.coordinate_scaled_score_target) == 0
    assert torch.isfinite(noisy.coordinate_scaled_score_target).all()


def test_five_regime_task_measure_has_frozen_balanced_counts() -> None:
    diffusion = TensorFreeHybridDiffusion(_small_model(), _standardizer())
    times = diffusion.sample_task_measure_times(
        64,
        torch.zeros(1),
        generator=torch.Generator().manual_seed(112),
    )
    coordinate = times.coordinate
    element = times.element
    lattice = times.lattice
    regime = times.regime
    assert torch.equal(torch.bincount(regime, minlength=5), torch.tensor([13, 13, 13, 13, 12]))
    torch.testing.assert_close(times.as_afl(), torch.stack((element, coordinate, lattice), dim=-1))
    assert torch.equal(element[regime == 0], torch.zeros(13))
    assert torch.equal(lattice[regime == 0], torch.zeros(13))
    assert torch.equal(element[regime == 1], coordinate[regime == 1])
    assert torch.equal(lattice[regime == 1], torch.zeros(13))
    assert torch.equal(element[regime == 2], torch.zeros(13))
    assert torch.equal(lattice[regime == 2], coordinate[regime == 2])
    assert torch.equal(element[regime == 3], coordinate[regime == 3])
    assert torch.equal(lattice[regime == 3], coordinate[regime == 3])
    assert bool((element[regime == 4] > 0).all())
    assert bool((lattice[regime == 4] > 0).all())


def test_explicit_diagonal_times_reproduce_shared_corruption_exactly() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(_small_model(), _standardizer())
    time = torch.tensor([0.4, 0.7])
    default = diffusion.noise_clean_batch(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time,
        generator=torch.Generator().manual_seed(113),
    )
    explicit = diffusion.noise_clean_batch(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=time,
        element_time=time,
        lattice_time=time,
        generator=torch.Generator().manual_seed(113),
    )
    for name in (
        "element_tokens",
        "fractional_coordinates",
        "log_volume",
        "log_shape",
        "coordinate_scaled_score_target",
    ):
        torch.testing.assert_close(getattr(default, name), getattr(explicit, name))


def test_independent_modality_times_are_non_degenerate_and_receive_gradients() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        independent_modality_times=True,
    )
    diffusion = TensorFreeHybridDiffusion(model, _standardizer())
    coordinate_time = torch.tensor([0.4, 0.7])
    output = diffusion(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        time=coordinate_time,
        element_time=torch.tensor([0.0, 0.7]),
        lattice_time=torch.tensor([0.4, 0.0]),
        generator=torch.Generator().manual_seed(114),
    )
    output.coordinate_loss.backward()
    for prefix in (
        "time_embedding.",
        "element_time_embedding.",
        "lattice_time_embedding.",
        "modality_time_fusion.",
    ):
        gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(float(value.square().sum()) for value in gradients if value is not None) > 0.0


def test_parameter_matched_clock_controls_have_equal_size_and_distinct_inputs() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    modes = ("matched_single", "side_mean", "separate")
    models = {
        mode: HybridCrystalDenoiser(
            hidden_dim=16,
            vector_dim=4,
            layers=1,
            radial_dim=4,
            atlas_residual_circle_samples=8,
            modality_time_conditioning=mode,
        )
        for mode in modes
    }
    counts = {sum(parameter.numel() for parameter in model.parameters()) for model in models.values()}
    assert len(counts) == 1
    graphs = blueprint.node_counts.numel()
    common = {
        "element_tokens": elements,
        "frac_coords": coordinates,
        "log_volume": torch.zeros(graphs),
        "log_shape": torch.zeros(graphs, 6),
        "batch": blueprint.batch,
        "time": torch.full((graphs,), 0.5),
        "tensor_condition": torch.zeros(graphs, 18),
        "condition_present": torch.zeros(graphs, 1, dtype=torch.bool),
        "shape_projector": blueprint.shape_projector,
        "fractional_to_cartesian": blueprint.fractional_to_cartesian,
    }
    single = models["matched_single"](**common)
    single.coordinate_cartesian_scaled_score.square().sum().backward()
    for name, parameter in models["matched_single"].named_parameters():
        if name.startswith(("element_time_embedding.", "lattice_time_embedding.", "modality_time_fusion.")):
            assert parameter.grad is None

    for mode in ("side_mean", "separate"):
        try:
            models[mode](**common)
        except ValueError as error:
            assert "requires explicit" in str(error)
        else:
            raise AssertionError(f"{mode} accepted missing side times")


def test_parameter_matched_clock_controls_accept_same_corrupted_side_states() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    for mode in ("matched_single", "side_mean", "separate"):
        model = HybridCrystalDenoiser(
            hidden_dim=16,
            vector_dim=4,
            layers=1,
            radial_dim=4,
            atlas_residual_circle_samples=8,
            modality_time_conditioning=mode,
        )
        diffusion = TensorFreeHybridDiffusion(model, _standardizer())
        output = diffusion(
            elements,
            coordinates,
            lattice,
            blueprint.batch,
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            time=torch.tensor([0.4, 0.7]),
            element_time=torch.tensor([0.0, 0.7]),
            lattice_time=torch.tensor([0.4, 0.0]),
            generator=torch.Generator().manual_seed(115),
        )
        output.coordinate_loss.backward()
        assert torch.isfinite(output.coordinate_loss)


def test_shared_time_model_rejects_silent_modality_mismatch() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = _small_model()
    graphs = blueprint.node_counts.numel()
    try:
        model(
            elements,
            coordinates,
            torch.zeros(graphs),
            torch.zeros(graphs, 6),
            blueprint.batch,
            torch.full((graphs,), 0.5),
            torch.zeros(graphs, 18),
            torch.zeros(graphs, 1, dtype=torch.bool),
            blueprint.shape_projector,
            blueprint.fractional_to_cartesian,
            element_time=torch.zeros(graphs),
            lattice_time=torch.zeros(graphs),
        )
    except ValueError as error:
        assert "cannot silently consume" in str(error)
    else:
        raise AssertionError("shared-time model accepted distinct modality times")


def test_coordinate_trainer_uses_clean_noncoordinate_side_information() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    diffusion = TensorFreeHybridDiffusion(_small_model(), _standardizer())
    trainer = ProductionTrainer(
        diffusion,
        ProductionTrainingConfig(objective="coordinate", coordinate_clean_side_information=True),
    )
    output, _ = trainer.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(110),
    )
    assert torch.equal(output.noisy.element_tokens, elements)
    assert not bool(output.noisy.element_was_masked.any())


def test_element_only_trainer_updates_element_path_but_not_continuous_heads() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    trainer = ProductionTrainer(
        TensorFreeHybridDiffusion(model, _standardizer()),
        ProductionTrainingConfig(
            precision="fp32",
            objective="element",
            modality_time_mode="element_only",
            ema_decay=0.9,
        ),
    )
    inactive_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith(
            (
                "coordinate_control_gate.",
                "coordinate_edge_encoder.",
                "coordinate_carrier.",
                "coordinate_carrier_mixer.",
                "volume_head.",
                "shape_head.",
            )
        )
    }
    element_before = {
        name: value.detach().clone() for name, value in model.named_parameters() if name.startswith("element_head.")
    }
    output, gradient_norm = trainer.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(117),
    )
    assert torch.isfinite(output.element_loss) and gradient_norm > 0.0
    assert torch.equal(output.noisy.time, torch.zeros(2))
    assert torch.equal(output.noisy.lattice_time, torch.zeros(2))
    assert bool((output.noisy.element_time > 0.0).all())
    current = dict(model.named_parameters())
    assert all(torch.equal(value, current[name]) for name, value in inactive_before.items())
    assert any(not torch.equal(value, current[name]) for name, value in element_before.items())


def test_lattice_forward_is_coordinate_free_and_composition_permutation_invariant() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    ).eval()
    state = LatticeVolumeShape.from_lattice(
        lattice,
        blueprint.fractional_to_cartesian,
    )
    lattice_time = torch.tensor([0.3, 0.7])
    block_calls = 0

    def count_block(_module, _inputs, _output):
        nonlocal block_calls
        block_calls += 1

    handle = model.blocks[0].register_forward_hook(count_block)
    first = model.forward_lattice(
        elements,
        state.log_volume,
        state.log_shape,
        blueprint.batch,
        lattice_time,
        blueprint.shape_projector,
    )
    handle.remove()
    permutation = torch.tensor([1, 0, 4, 2, 3])
    second = model.forward_lattice(
        elements[permutation],
        state.log_volume,
        state.log_shape,
        blueprint.batch,
        lattice_time,
        blueprint.shape_projector,
    )
    assert block_calls == 0
    torch.testing.assert_close(first.clean_volume_latent, second.clean_volume_latent)
    torch.testing.assert_close(first.clean_shape_latent, second.clean_shape_latent)

    condition = torch.zeros(2, 18)
    present = torch.zeros(2, 1, dtype=torch.bool)
    clean_time = torch.zeros(2)
    full_a = model(
        elements,
        coordinates,
        state.log_volume,
        state.log_shape,
        blueprint.batch,
        clean_time,
        condition,
        present,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        element_time=clean_time,
        lattice_time=lattice_time,
    )
    full_b = model(
        elements,
        coordinates.roll(1, dims=0),
        state.log_volume,
        state.log_shape,
        blueprint.batch,
        clean_time,
        condition,
        present,
        blueprint.shape_projector,
        blueprint.fractional_to_cartesian,
        element_time=clean_time,
        lattice_time=lattice_time,
    )
    torch.testing.assert_close(full_a.clean_volume_latent, full_b.clean_volume_latent)
    torch.testing.assert_close(full_a.clean_shape_latent, full_b.clean_shape_latent)


def test_lattice_only_trainer_updates_no_coordinate_or_element_readout() -> None:
    elements, _, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    trainer = ProductionTrainer(
        TensorFreeHybridDiffusion(model, _standardizer()),
        ProductionTrainingConfig(
            precision="fp32",
            objective="lattice",
            modality_time_mode="lattice_only",
            ema_decay=0.9,
        ),
    )
    inactive_prefixes = (
        "blocks.",
        "degree_embedding.",
        "element_head.",
        "composition_head.",
        "coordinate_control_gate.",
        "coordinate_edge_encoder.",
        "coordinate_carrier.",
        "coordinate_carrier_mixer.",
    )
    inactive_before = {
        name: value.detach().clone() for name, value in model.named_parameters() if name.startswith(inactive_prefixes)
    }
    lattice_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith(("volume_head.", "shape_head."))
    }
    output, gradient_norm = trainer.train_lattice_step(
        elements,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(121),
    )
    assert gradient_norm > 0.0 and torch.isfinite(output.loss)
    current = dict(model.named_parameters())
    assert all(torch.equal(value, current[name]) for name, value in inactive_before.items())
    assert any(not torch.equal(value, current[name]) for name, value in lattice_before.items())


def test_lattice_reverse_sampler_never_calls_full_geometry_forward() -> None:
    elements, _, _, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    full_forward_calls = 0

    def count_full_forward(_module, _inputs, _output):
        nonlocal full_forward_calls
        full_forward_calls += 1

    handle = model.register_forward_hook(count_full_forward)
    generated = TensorFreeReverseSampler(
        model,
        _standardizer(),
        maximum_time=0.8,
    ).sample_lattice(
        elements,
        blueprint,
        steps=3,
        initialization_generator=torch.Generator().manual_seed(122),
        continuous_generator=torch.Generator().manual_seed(123),
        continuous_mode="probability_flow",
    )
    handle.remove()
    assert full_forward_calls == 0
    assert generated.lattice.shape == (2, 3, 3)
    assert torch.isfinite(generated.lattice).all()
    assert bool((torch.linalg.det(generated.lattice) > 0.0).all())


def test_coordinate_reverse_sampler_holds_side_states_and_replays_common_noise() -> None:
    elements, _, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        maximum_time=0.8,
    )
    initial = sampler.initialize_coordinate_state(
        blueprint,
        generator=torch.Generator().manual_seed(124),
    )
    outputs = [
        sampler.sample_coordinates(
            elements,
            lattice,
            blueprint,
            steps=3,
            initial_state=initial,
            continuous_generator=torch.Generator().manual_seed(125),
        )
        for _ in range(2)
    ]
    torch.testing.assert_close(outputs[0].fractional_coordinates, outputs[1].fractional_coordinates)
    torch.testing.assert_close(outputs[0].lattice, lattice)
    torch.testing.assert_close(outputs[0].element_tokens, elements)
    assert outputs[0].diagnostics.coordinate_step_rms.shape == (3,)
    assert torch.isfinite(outputs[0].fractional_coordinates).all()
    assert bool(
        ((outputs[0].fractional_coordinates >= 0.0) & (outputs[0].fractional_coordinates < 1.0)).all()
    )


def test_graph_weighted_accumulation_matches_one_full_batch_update() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    reference_model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    accumulated_model = copy.deepcopy(reference_model)
    config = ProductionTrainingConfig(
        precision="fp32",
        objective="coordinate",
        coordinate_clean_side_information=True,
        ema_decay=0.9,
    )
    reference = ProductionTrainer(
        TensorFreeHybridDiffusion(reference_model, _standardizer()),
        config,
    )
    accumulated = ProductionTrainer(
        TensorFreeHybridDiffusion(accumulated_model, _standardizer()),
        config,
    )
    reference.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(126),
    )
    accumulated.begin_optimization_step()
    for _ in range(2):
        accumulated.accumulate_hybrid_step(
            elements,
            coordinates,
            lattice,
            blueprint.batch,
            blueprint,
            loss_weight=0.5,
            generator=torch.Generator().manual_seed(126),
        )
    accumulated.finish_optimization_step()

    for reference_value, accumulated_value in zip(
        reference_model.parameters(),
        accumulated_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(reference_value, accumulated_value, atol=1e-7, rtol=1e-6)


def test_uniform_element_training_is_self_correcting_and_uses_composition_loss() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    diffusion = TensorFreeHybridDiffusion(
        model,
        _standardizer(),
        categorical_path="uniform_replacement",
    )
    trainer = ProductionTrainer(
        diffusion,
        ProductionTrainingConfig(
            precision="fp32",
            objective="element",
            modality_time_mode="element_only",
            categorical_path="uniform_replacement",
            composition_loss_weight=1.0,
            ema_decay=0.9,
        ),
    )
    output, gradient_norm = trainer.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(119),
    )
    assert gradient_norm > 0.0
    assert torch.isfinite(output.element_loss)
    assert torch.isfinite(output.composition_loss)
    torch.testing.assert_close(
        trainer.optimization_loss(output),
        output.element_loss + output.composition_loss,
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
    unit = (sampled - diffusion.minimum_time) / (diffusion.maximum_time - diffusion.minimum_time)
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
    model = _small_product_model()
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2, 3]))
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        coordinate_sigma_min=0.005,
        coordinate_sigma_max=0.5,
        maximum_time=0.8,
        categorical_path="orderless_reveal",
        composition_model=_small_composition_model(),
    )
    generated = sampler.sample(
        blueprint,
        steps=4,
        initialization_generator=torch.Generator().manual_seed(105),
        categorical_generator=torch.Generator().manual_seed(106),
        continuous_mode="probability_flow",
    )
    assert generated.atomic_numbers.min() >= 1 and generated.atomic_numbers.max() <= 118
    assert generated.fractional_coordinates.shape == (5, 3)
    assert generated.lattice.shape == (2, 3, 3)
    assert torch.isfinite(generated.lattice).all()
    assert generated.diagnostics.masked_count[-1] == 0
    assert bool((generated.diagnostics.composition_closure_error == 0).all())
    assert torch.equal(generated.diagnostics.remaining_atom_count[-1], torch.zeros(2, dtype=torch.long))
    observed_counts = torch.bincount(
        blueprint.batch * 118 + generated.element_tokens,
        minlength=2 * 118,
    ).reshape(2, 118)
    assert torch.equal(observed_counts, generated.composition_counts)
    for graph in range(2):
        assert torch.allclose(
            generated.log_shape[graph].dot(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])),
            torch.tensor(0.0),
            atol=2e-5,
        )


def test_element_reverse_sampler_uses_observed_geometry_and_finishes_without_masks() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    del elements
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    seen_times: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def record_times(_module, args, kwargs):
        seen_times.append(
            (
                args[5].detach().clone(),
                kwargs["element_time"].detach().clone(),
                kwargs["lattice_time"].detach().clone(),
            )
        )

    handle = model.register_forward_pre_hook(record_times, with_kwargs=True)
    sampler = TensorFreeReverseSampler(model, _standardizer(), maximum_time=0.8)
    coordinate_copy = coordinates.clone()
    lattice_copy = lattice.clone()
    try:
        generated = sampler.sample_elements(
            blueprint,
            coordinates,
            lattice,
            steps=4,
            categorical_generator=torch.Generator().manual_seed(118),
        )
    finally:
        handle.remove()
    assert torch.equal(coordinates, coordinate_copy)
    assert torch.equal(lattice, lattice_copy)
    assert generated.element_tokens.shape == (5,)
    assert generated.atomic_numbers.min() >= 1 and generated.atomic_numbers.max() <= 118
    assert generated.diagnostics.masked_count[-1] == 0
    assert len(seen_times) == 4
    for coordinate_time, element_time, lattice_time in seen_times:
        assert torch.equal(coordinate_time, torch.zeros(2))
        assert torch.equal(lattice_time, torch.zeros(2))
        assert bool((element_time > 0.0).all())


def test_uniform_element_reverse_projects_model_predicted_integer_composition() -> None:
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    del elements
    model = HybridCrystalDenoiser(
        hidden_dim=16,
        vector_dim=4,
        layers=1,
        radial_dim=4,
        atlas_residual_circle_samples=8,
        modality_time_conditioning="separate",
    )
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        maximum_time=0.8,
        categorical_path="uniform_replacement",
    )
    generated = sampler.sample_elements(
        blueprint,
        coordinates,
        lattice,
        steps=4,
        categorical_generator=torch.Generator().manual_seed(120),
    )
    assert generated.element_tokens.min() >= 0 and generated.element_tokens.max() < 118
    assert torch.equal(
        generated.predicted_composition_counts.sum(dim=-1),
        blueprint.node_counts,
    )
    assert generated.diagnostics.masked_count[-1] == 0


def test_quotient_coordinate_reverse_modes_use_full_and_half_score_drift():
    coordinates = torch.tensor([[-0.2, 0.1, 0.1], [0.2, -0.1, -0.1], [0.0, 0.3, -0.3]])
    score = torch.tensor([[0.4, 0.2, -0.2], [-0.4, -0.2, 0.2], [0.0, 0.0, 0.0]])
    batch = torch.tensor([0, 0, 1])
    reverse_sde = quotient_coordinate_reverse_step(
        coordinates,
        score,
        torch.tensor([0.25, 0.25]),
        torch.tensor([0.0, 0.0]),
        batch,
        2,
        generator=None,
        mode="reverse_sde",
    )
    probability_flow = quotient_coordinate_reverse_step(
        coordinates,
        score,
        torch.tensor([0.25, 0.25]),
        torch.tensor([0.0, 0.0]),
        batch,
        2,
        generator=None,
        mode="probability_flow",
    )
    full_drift = coordinates + 0.25 * score / 0.5
    full_drift[:2] -= full_drift[:2].mean(0)
    full_drift[2] = 0.0
    half_drift = coordinates + 0.5 * 0.25 * score / 0.5
    half_drift[:2] -= half_drift[:2].mean(0)
    half_drift[2] = 0.0
    assert torch.allclose(reverse_sde, full_drift)
    assert torch.allclose(probability_flow, half_drift)


def test_quotient_coordinate_reverse_step_accepts_prescribed_horizontal_noise():
    coordinates = torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    scaled_score = torch.zeros_like(coordinates)
    prescribed = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    observed = quotient_coordinate_reverse_step(
        coordinates,
        scaled_score,
        torch.tensor([0.25]),
        torch.tensor([0.16]),
        torch.tensor([0, 0]),
        1,
        generator=None,
        mode="reverse_sde",
        standard_noise=prescribed,
    )
    bridge_scale = (0.16 * 0.09 / 0.25) ** 0.5
    assert torch.allclose(observed, coordinates + bridge_scale * prescribed)


def test_vp_probability_flow_step_matches_ddim_algebra():
    schedule = CosineNoiseSchedule()
    clean = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    noise = torch.tensor([[-0.3, 0.8], [0.5, -0.2]])
    time_from = torch.tensor([[0.8], [0.6]])
    time_to = torch.tensor([[0.3], [0.0]])
    state = schedule.alpha(time_from) * clean + schedule.sigma(time_from) * noise
    observed = vp_reverse_step(
        schedule,
        state,
        clean,
        time_from,
        time_to,
        generator=None,
        mode="probability_flow",
    )
    expected = schedule.alpha(time_to) * clean + schedule.sigma(time_to) * noise
    assert torch.allclose(observed, expected, atol=1.0e-6)
    assert torch.allclose(observed[1], clean[1], atol=1.0e-6)


def test_joint_reverse_modes_accept_common_initial_state_and_finish_cleanly():
    torch.manual_seed(107)
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2, 3]))
    sampler = TensorFreeReverseSampler(
        _small_product_model(),
        _standardizer(),
        maximum_time=0.8,
        categorical_path="orderless_reveal",
        composition_model=_small_composition_model(),
    )
    initial = sampler.initialize_continuous_state(blueprint, generator=torch.Generator().manual_seed(108))
    outputs = []
    for mode in ("reverse_sde", "probability_flow"):
        outputs.append(
            sampler.sample(
                blueprint,
                steps=3,
                initial_state=initial,
                categorical_generator=torch.Generator().manual_seed(109),
                continuous_generator=torch.Generator().manual_seed(110),
                continuous_mode=mode,
            )
        )
    for generated in outputs:
        assert generated.fractional_coordinates.shape == (5, 3)
        assert generated.lattice.shape == (2, 3, 3)
        assert torch.isfinite(generated.fractional_coordinates).all()
        assert torch.isfinite(generated.lattice).all()
        assert generated.diagnostics.masked_count[-1] == 0


def test_reverse_sampler_keeps_universal_cover_until_terminal_decode():
    model = _small_product_model()
    seen_coordinates: list[torch.Tensor] = []

    def record_coordinates(_module, inputs):
        seen_coordinates.append(inputs[1].detach().clone())

    handle = model.register_forward_pre_hook(record_coordinates)
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([2]))
    sampler = TensorFreeReverseSampler(
        model,
        _standardizer(),
        maximum_time=0.8,
        categorical_path="orderless_reveal",
        composition_model=_small_composition_model(),
    )
    initial = ContinuousReverseInitialState(
        fractional_coordinates=torch.tensor([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        volume_latent=torch.zeros(1),
        shape_latent=torch.zeros(1, 5),
    )
    try:
        generated = sampler.sample(
            blueprint,
            steps=1,
            initial_state=initial,
            categorical_generator=torch.Generator().manual_seed(111),
            continuous_mode="probability_flow",
        )
    finally:
        handle.remove()
    # Product sampling may make several categorical reveal queries before the
    # continuous update, but every query must stay on the unwrapped state.
    assert len(seen_coordinates) >= 1
    assert all(torch.equal(value, initial.fractional_coordinates) for value in seen_coordinates)
    assert bool(((generated.fractional_coordinates >= 0.0) & (generated.fractional_coordinates < 1.0)).all())


def test_coordinate_pretraining_updates_coordinate_path_without_other_heads():
    elements, coordinates, lattice, blueprint = _small_clean_batch()
    model = _small_model()
    diffusion = TensorFreeHybridDiffusion(model, _standardizer())
    trainer = ProductionTrainer(
        diffusion,
        ProductionTrainingConfig(precision="fp32", objective="coordinate", ema_decay=0.9),
    )
    inactive_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith(("element_head.", "volume_head.", "shape_head."))
    }
    coordinate_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.startswith(
            (
                "coordinate_control_gate.",
                "coordinate_edge_encoder.",
                "coordinate_carrier.",
                "coordinate_carrier_mixer.",
            )
        )
    }
    output, gradient_norm = trainer.train_step(
        elements,
        coordinates,
        lattice,
        blueprint.batch,
        blueprint,
        generator=torch.Generator().manual_seed(91),
    )
    assert torch.isfinite(output.coordinate_loss)
    assert gradient_norm > 0.0
    current = dict(model.named_parameters())
    assert all(torch.equal(value, current[name]) for name, value in inactive_before.items())
    assert any(not torch.equal(value, current[name]) for name, value in coordinate_before.items())


def test_production_checkpoint_restores_model_optimizer_ema_rng_and_count_prior(tmp_path: Path):
    model = _small_model()
    diffusion = TensorFreeHybridDiffusion(model, _standardizer())
    trainer = ProductionTrainer(diffusion, ProductionTrainingConfig())
    prior = EmpiricalNodeCountPrior.fit(torch.tensor([2, 2, 3, 4]))
    path = tmp_path / "production.pt"
    loader_generator = torch.Generator().manual_seed(701)
    device_generator = torch.Generator().manual_seed(702)
    runtime_state = {
        "epoch_loader_generator_state": loader_generator.get_state(),
        "batches_consumed_in_epoch": 7,
        "device_generator_state": device_generator.get_state(),
    }
    save_production_checkpoint(
        path,
        model=model,
        ema=trainer.ema,
        optimizer=trainer.optimizer,
        training_step=17,
        node_count_prior=prior,
        metadata={"model": {"hidden_dim": 16}, "protocol": "s1a_tensor_free_v1"},
        runtime_state=runtime_state,
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
    restored_runtime = load_production_runtime_state(path)
    assert restored_runtime["batches_consumed_in_epoch"] == 7
    assert torch.equal(
        restored_runtime["epoch_loader_generator_state"],
        runtime_state["epoch_loader_generator_state"],
    )
    assert torch.equal(restored_runtime["device_generator_state"], runtime_state["device_generator_state"])
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
