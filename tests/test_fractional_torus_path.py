import math
from pathlib import Path

import torch

from gaugeflow.production.blueprint import ParentBlueprintBatch
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.hybrid_diffusion import TensorFreeHybridDiffusion
from gaugeflow.production.lattice_standardization import P1LatticeStandardizer
from gaugeflow.production.schedules import FractionalTorusVarianceSchedule
from gaugeflow.production.state_projection import project_translation_state


def _standardizer() -> P1LatticeStandardizer:
    return P1LatticeStandardizer.from_json(
        Path(__file__).parents[1]
        / "configs/statistics/h1a_p1_lattice_standardization.json"
    )


def _small_model() -> HybridCrystalDenoiser:
    return HybridCrystalDenoiser(
        hidden_dim=12, vector_dim=3, layers=1, radial_dim=3
    )


def test_terminal_fractional_heat_kernel_matches_uniform_torus_prior():
    schedule = FractionalTorusVarianceSchedule(sigma_max=1.0)
    variance = float(schedule.variance(torch.tensor(0.999)))
    first_fourier_residual = math.exp(-2.0 * math.pi**2 * variance)
    assert first_fourier_residual < 1.0e-8


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
    assert torch.equal(noisy[0].coordinate_score_target, noisy[1].coordinate_score_target)


def test_oracle_fractional_score_recovers_endpoint_in_one_variance_step():
    generator = torch.Generator().manual_seed(7002)
    counts = torch.tensor([4, 7, 2])
    batch = torch.repeat_interleave(torch.arange(3), counts)
    endpoint = project_translation_state(
        torch.rand((int(counts.sum()), 3), generator=generator, dtype=torch.float64),
        batch,
        3,
    )
    variance = torch.tensor([0.17, 0.63, 0.91], dtype=torch.float64)
    noise = project_translation_state(
        torch.randn(endpoint.shape, generator=generator, dtype=torch.float64),
        batch,
        3,
    )
    state = endpoint + variance[batch].sqrt().unsqueeze(-1) * noise
    oracle_score = -(state - endpoint) / variance[batch].unsqueeze(-1)
    recovered = state + variance[batch].unsqueeze(-1) * oracle_score
    rms = (recovered - endpoint).square().mean().sqrt()
    assert rms <= 1.0e-12


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
