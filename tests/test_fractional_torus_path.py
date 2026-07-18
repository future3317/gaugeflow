import math
from pathlib import Path

import torch

from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.schedules import (
    ExponentialTorusNoiseSchedule,
    wrapped_normal_score,
)


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1]
        / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def _small_model() -> HybridCrystalDenoiser:
    return HybridCrystalDenoiser(
        hidden_dim=12, vector_dim=3, layers=1, radial_dim=3
    )


def test_exponential_torus_schedule_has_clean_origin_and_log_uniform_positive_scales():
    schedule = ExponentialTorusNoiseSchedule(sigma_min=0.005, sigma_max=0.5)
    time = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    sigma = schedule.sigma(time)
    assert sigma[0] == 0.0
    assert torch.allclose(sigma[-1], torch.tensor(0.5, dtype=torch.float64))
    log_steps = sigma[1:].log().diff()
    assert torch.allclose(log_steps, log_steps[0].expand_as(log_steps), atol=1e-12)


def test_terminal_fractional_heat_kernel_is_close_to_uniform_torus_prior():
    schedule = ExponentialTorusNoiseSchedule(sigma_min=0.005, sigma_max=0.5)
    variance = float(schedule.variance(torch.tensor(1.0)))
    first_fourier_residual = math.exp(-2.0 * math.pi**2 * variance)
    assert first_fourier_residual < 1.0e-2


def test_exponential_torus_reverse_variance_increment_is_nonnegative():
    schedule = ExponentialTorusNoiseSchedule(sigma_min=0.005, sigma_max=0.5)
    time_from = torch.linspace(1.0, 0.1, 11)
    time_to = (time_from - 0.1).clamp_min(0.0)
    increment = schedule.increment(time_from, time_to)
    assert torch.isfinite(increment).all()
    assert torch.all(increment >= 0.0)
    assert increment[-1] > 0.0


def test_fractional_coordinate_forward_kernel_is_independent_of_cell_metric():
    elements = torch.tensor([4, 6, 8], dtype=torch.long)
    coordinates = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 0.4, 0.9], [0.3, 0.8, 0.5]]
    )
    blueprint = ParentBlueprintBatch.from_node_counts(torch.tensor([3]))
    diffusion = TensorFreeHybridDiffusion(_small_model(), _standardizer())
    lattices = (
        (3.0 * torch.eye(3)).unsqueeze(0),
        torch.tensor([[[7.0, 0.0, 0.0], [1.1, 2.5, 0.0], [0.7, 0.3, 5.0]]]),
    )
    noisy = []
    for lattice in lattices:
        noisy.append(
            diffusion.noise_clean_batch(
                elements,
                coordinates,
                lattice,
                blueprint.batch,
                blueprint.shape_projector,
                blueprint.fractional_to_cartesian,
                time=torch.tensor([0.61]),
                generator=torch.Generator().manual_seed(7001),
            )
        )
    assert torch.equal(noisy[0].fractional_coordinates, noisy[1].fractional_coordinates)
    assert torch.equal(
        noisy[0].coordinate_scaled_score_target,
        noisy[1].coordinate_scaled_score_target,
    )


def test_wrapped_normal_score_matches_autograd_image_sum_and_is_periodic():
    displacement = torch.tensor(
        [-1.49, -0.47, -0.13, 0.0, 0.29, 0.51, 1.82],
        dtype=torch.float64,
        requires_grad=True,
    )
    sigma = torch.tensor([0.04, 0.11, 0.25, 0.31, 0.5, 0.73, 1.0], dtype=torch.float64)
    images = torch.arange(-16, 17, dtype=torch.float64)
    centered = torch.remainder(displacement + 0.5, 1.0) - 0.5
    log_density = torch.logsumexp(
        -0.5 * ((centered.unsqueeze(-1) + images) / sigma.unsqueeze(-1)).square(),
        dim=-1,
    ).sum()
    reference = torch.autograd.grad(log_density, displacement)[0]
    observed = wrapped_normal_score(displacement.detach(), sigma)
    shifted = wrapped_normal_score(displacement.detach() + 3.0, sigma)
    assert torch.allclose(observed, reference, atol=2e-10, rtol=2e-10)
    assert torch.allclose(shifted, observed, atol=2e-10, rtol=2e-10)


def test_wrapped_normal_score_vanishes_at_the_uniform_terminal_scale():
    displacement = torch.linspace(-0.5, 0.5, 101, dtype=torch.float64)
    score = wrapped_normal_score(displacement, torch.ones_like(displacement))
    assert float(score.abs().max()) < 4e-8


def test_lattice_state_is_visible_without_periodic_edges():
    torch.manual_seed(7003)
    model = _small_model().eval()
    common = dict(
        element_tokens=torch.tensor([5]),
        frac_coords=torch.tensor([[0.2, 0.3, 0.4]]),
        batch=torch.tensor([0]),
        time=torch.tensor([0.5]),
        tensor_condition=torch.zeros((1, 18)),
        condition_present=torch.zeros((1, 1), dtype=torch.bool),
        shape_projector=torch.eye(6).unsqueeze(0),
        fractional_to_cartesian=torch.eye(3).unsqueeze(0),
    )
    first = model(
        log_volume=torch.tensor([math.log(1000.0)]),
        log_shape=torch.zeros((1, 6)),
        **common,
    )
    second = model(
        log_volume=torch.tensor([math.log(8000.0)]),
        log_shape=torch.zeros((1, 6)),
        **common,
    )
    assert not torch.allclose(first.clean_volume_latent, second.clean_volume_latent)
    assert not torch.allclose(first.clean_element_logits, second.clean_element_logits)
