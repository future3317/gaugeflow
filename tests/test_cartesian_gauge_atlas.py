import inspect
import types

import torch
from torch import nn

from gaugeflow.geometry import GaussianRadialBasis
from gaugeflow.production.cartesian_gauge_atlas import (
    CartesianGeometryQueries,
    CartesianSTFGeometryQueryEncoder,
    StratifiedCartesianGaugeAtlas,
    _FrameData,
    cartesian_stf_moments,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3


def _rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    left, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(left) < 0:
        left[:, 0] = -left[:, 0]
    return left


def _rotate_candidates(tensor: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    return rotate_rank3(tensor.expand((rotations.shape[0], 3, 3, 3)), rotations)


def _geometry(positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    sites = positions.shape[0]
    source = torch.tensor([left for left in range(sites) for right in range(sites) if left != right])
    target = torch.tensor([right for left in range(sites) for right in range(sites) if left != right])
    displacement = positions[target] - positions[source]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction = displacement / distance.unsqueeze(-1)
    radial = GaussianRadialBasis(5, 8.0)(distance)
    generator = torch.Generator().manual_seed(113)
    nodes = torch.randn((sites, 16), generator=generator)
    return nodes, torch.zeros_like(nodes), source, target, direction, radial, torch.zeros(sites, dtype=torch.long)


def _encoder() -> CartesianSTFGeometryQueryEncoder:
    torch.manual_seed(114)
    return CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3).eval()


def test_cartesian_stf_moments_are_trace_free_and_covariant():
    direction = torch.nn.functional.normalize(torch.randn((9, 3), dtype=torch.float64), dim=-1)
    rotation = _rotation(115)
    first, second, third = cartesian_stf_moments(direction)
    rotated_first, rotated_second, rotated_third = cartesian_stf_moments(direction @ rotation.T)
    assert torch.allclose(torch.diagonal(second, dim1=-2, dim2=-1).sum(-1), torch.zeros(9, dtype=torch.float64))
    assert torch.allclose(torch.einsum("ei,ij->ej", first, rotation.T), rotated_first, atol=2e-12, rtol=2e-12)
    assert torch.allclose(
        torch.einsum("eij,ai,bj->eab", second, rotation, rotation), rotated_second, atol=2e-12, rtol=2e-12
    )
    assert torch.allclose(
        torch.einsum("eijk,ai,bj,ck->eabc", third, rotation, rotation, rotation), rotated_third, atol=3e-12, rtol=3e-12
    )


def test_cartesian_query_accepts_no_tensor_condition_and_is_covariant():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.3, 0.4], [0.3, 0.2, 1.7]])
    values = _geometry(positions)
    encoder = _encoder()
    query = encoder(*values, graph_count=1)
    parameters = set(inspect.signature(CartesianSTFGeometryQueryEncoder.forward).parameters)
    assert parameters.isdisjoint({"tensor", "tensor_condition", "condition", "piezo_irreps"})
    rotation = _rotation(116).float()
    rotated_values = list(values)
    rotated_values[4] = values[4] @ rotation.T
    rotated = encoder(*rotated_values, graph_count=1)
    assert torch.allclose(query.first @ rotation.T, rotated.first, atol=4e-6, rtol=4e-6)
    assert torch.allclose(
        torch.einsum("bcij,ai,dj->bcad", query.second, rotation, rotation), rotated.second, atol=5e-6, rtol=5e-6
    )
    assert torch.allclose(
        torch.einsum("bcijk,ai,dj,ek->bcade", query.third, rotation, rotation, rotation),
        rotated.third,
        atol=6e-6,
        rtol=6e-6,
    )
    assert torch.allclose(
        torch.einsum("bcijk,ai,dj,ek->bcade", query.rank_three, rotation, rotation, rotation),
        rotated.rank_three,
        atol=8e-6,
        rtol=8e-6,
    )


def test_atlas_is_representative_covariant_without_a_grid():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]])
    encoder = _encoder()
    values = _geometry(positions)
    query = encoder(*values, graph_count=1)
    rotation_state, rotation_condition = _rotation(117).float(), _rotation(118).float()
    rotated_values = list(values)
    rotated_values[4] = values[4] @ rotation_state.T
    rotated_query = encoder(*rotated_values, graph_count=1)
    torch.manual_seed(119)
    condition = torch.randn((1, 18))
    rotated_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), rotation_condition).contiguous())
    assert torch.allclose(
        piezo_from_irreps(rotated_condition),
        rotate_rank3(piezo_from_irreps(condition), rotation_condition),
        atol=3e-5,
        rtol=3e-5,
    )
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=6, generic_chart_samples=1).eval()
    original_candidates = atlas._raw_candidate_measure(
        atlas._frame_data(query.frame_tensor[0]), atlas._frame_data(atlas.invariant(piezo_from_irreps(condition))[1][0])
    )[0]
    transformed_candidates = atlas._raw_candidate_measure(
        atlas._frame_data(rotated_query.frame_tensor[0]),
        atlas._frame_data(atlas.invariant(piezo_from_irreps(rotated_condition))[1][0]),
    )[0]
    expected_candidates = rotation_state @ original_candidates @ rotation_condition.T
    distances = torch.linalg.matrix_norm(
        expected_candidates[:, None] - transformed_candidates[None], dim=(-2, -1)
    ).amin(dim=-1)
    assert distances.max() < 2e-4
    matching = torch.linalg.matrix_norm(
        expected_candidates[:, None] - transformed_candidates[None], dim=(-2, -1)
    ).argmin(dim=-1)
    original_rotated = _rotate_candidates(piezo_from_irreps(condition)[0], original_candidates)
    transformed_rotated = _rotate_candidates(
        piezo_from_irreps(rotated_condition)[0], transformed_candidates
    )
    expected_rotated = rotate_rank3(piezo_from_irreps(condition).expand_as(original_rotated), original_candidates)
    assert torch.allclose(original_rotated, expected_rotated, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        transformed_rotated[matching], rotate_rank3(original_rotated, rotation_state), atol=3e-4, rtol=3e-4
    )
    kwargs = (
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        torch.tensor([0.4]),
    )
    original = atlas(condition, kwargs[0], kwargs[1], kwargs[2], query, kwargs[3])
    transformed = atlas(rotated_condition, kwargs[0], rotated_values[4], kwargs[2], rotated_query, kwargs[3])
    expected = rotate_rank3(original.aligned_tensor, rotation_state)
    assert original.effective_frame_count.item() == 24 * 24
    assert transformed.effective_frame_count.item() == 24 * 24
    assert torch.allclose(transformed.aligned_tensor, expected, atol=3e-4, rtol=3e-4)
    assert torch.allclose(transformed.graph_condition, original.graph_condition, atol=3e-4, rtol=3e-4)


def test_isotropic_condition_disables_alignment_and_axial_stratum_expands_residual_group():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.1, 0.4], [0.3, 0.2, 1.5]])
    values = _geometry(positions)
    query = _encoder()(*values, graph_count=1)
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=5, generic_chart_samples=1).eval()
    zero = atlas(
        torch.zeros((1, 18)),
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        query,
        torch.tensor([0.3]),
    )
    assert zero.effective_frame_count.item() == 0
    assert zero.gate.item() == 0.0
    tensor = torch.zeros((1, 3, 3, 3))
    tensor[:, 0, 0, 0] = 1.0
    axial = atlas(
        piezo_to_irreps(tensor),
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        query,
        torch.tensor([0.3]),
    )
    assert axial.effective_frame_count.item() == 5 * 24 * 24
    assert axial.residual_kind.item() == StratifiedCartesianGaugeAtlas.AXIAL
    assert torch.isfinite(axial.graph_condition).all()


def test_atlas_has_finite_backward_gradients():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.2], [0.2, 1.2, 0.4], [0.4, 0.3, 1.6]])
    values = _geometry(positions)
    query = _encoder()(*values, graph_count=1)
    condition = torch.randn((1, 18), requires_grad=True)
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4)
    output = atlas(
        condition,
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        query,
        torch.tensor([0.5]),
    )
    output.graph_condition.square().sum().backward()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()


def test_candidate_measure_deduplication_is_order_and_duplicate_expansion_invariant():
    atlas = StratifiedCartesianGaugeAtlas(16, generic_chart_samples=1).double().eval()
    first, second = _rotation(130), _rotation(131)
    base = torch.stack((first, second, first))
    measure = atlas._deduplicate_measure(base)
    permuted = atlas._deduplicate_measure(base[torch.tensor([2, 0, 1])])
    expanded = atlas._deduplicate_measure(base.repeat_interleave(3, dim=0))

    tensor = torch.randn((3, 3, 3), dtype=torch.float64)
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)

    def pooled(candidate_measure):
        rotated = _rotate_candidates(tensor, candidate_measure.rotations)
        score = torch.einsum("fabc,qabc,q->f", rotated, query, atlas.score_channel.to(query))
        posterior = torch.softmax(score + candidate_measure.prior.log(), dim=0)
        return torch.einsum("f,fijk->ijk", posterior, rotated)

    assert measure.rotations.shape[0] == 2
    assert torch.allclose(measure.prior.sort().values, torch.tensor([1 / 3, 2 / 3], dtype=torch.float64))
    assert torch.allclose(pooled(measure), pooled(permuted), atol=1e-12, rtol=1e-12)
    assert torch.allclose(pooled(measure), pooled(expanded), atol=1e-12, rtol=1e-12)


def test_s0_4_generic_measure_has_preregistered_raw_count_and_reports_unique_count():
    atlas = StratifiedCartesianGaugeAtlas(16, generic_chart_samples=7).double().eval()
    covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    frame = atlas._frame_data(covariance, directional=True)
    measure = atlas._candidate_measure(frame, frame)
    assert measure.raw_count == 24 * 7 * 24
    assert 0 < measure.rotations.shape[0] <= measure.raw_count
    assert torch.allclose(measure.prior.sum(), torch.ones((), dtype=torch.float64))
    print("generic raw/unique", measure.raw_count, measure.rotations.shape[0])


def test_directional_frame_with_empty_partition_fails_closed():
    atlas = StratifiedCartesianGaugeAtlas(16, generic_chart_samples=1).double().eval()
    frame = _FrameData(
        basis=torch.eye(3, dtype=torch.float64),
        weights=torch.zeros(4, dtype=torch.float64),
        directional=True,
    )
    try:
        atlas._raw_candidate_measure(frame, frame)
    except RuntimeError as error:
        assert "activate at least one partition component" in str(error)
    else:
        raise AssertionError("invalid directional partition must not become an empty runtime fallback")


def test_nonzero_descriptor_isotropic_tensor_uses_cartesian_cubature_not_invariant_only():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.1, 1.1, 0.4], [0.3, 0.2, 1.5]])
    values = _geometry(positions)
    query = _encoder()(*values, graph_count=1)
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4, generic_chart_samples=1).eval()
    # The fully symmetric xyz cubic has isotropic quadratic contractions but
    # nonzero rank-three directional information.
    tensor = torch.zeros((1, 3, 3, 3))
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        tensor[(0,) + permutation] = 1.0
    covariance = atlas.invariant.frame_covariants(tensor)
    descriptor = covariance[0] + 0.61803398875 * covariance[1]
    assert torch.allclose(descriptor, torch.eye(3) * descriptor[0, 0, 0], atol=1e-6, rtol=1e-6)
    output = atlas(
        piezo_to_irreps(tensor),
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        query,
        torch.tensor([0.3]),
    )
    assert output.residual_kind.item() == StratifiedCartesianGaugeAtlas.ISOTROPIC
    assert output.effective_frame_count.item() > 0
    assert torch.isfinite(output.aligned_tensor).all()


def test_soft_stratum_partition_is_continuous_and_has_finite_gradient():
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4, generic_chart_samples=1).double().eval()
    tensor = torch.randn((3, 3, 3), dtype=torch.float64)
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)
    condition = atlas._frame_data(torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)), directional=True)

    def pooled(gap: torch.Tensor) -> torch.Tensor:
        covariance = torch.diag(torch.stack((gap.new_zeros(()), gap, gap.new_ones(()))))
        measure = atlas._candidate_measure(atlas._frame_data(covariance, directional=True), condition)
        rotated = _rotate_candidates(tensor, measure.rotations)
        score = torch.einsum("fabc,qabc,q->f", rotated, query, atlas.score_channel.to(query))
        posterior = torch.softmax(score + measure.prior.log(), dim=0)
        return torch.einsum("f,fijk->ijk", posterior, rotated)

    threshold = torch.tensor(atlas.relative_eigen_gap, dtype=torch.float64, requires_grad=True)
    center = pooled(threshold)
    center.square().sum().backward()
    assert threshold.grad is not None and torch.isfinite(threshold.grad)
    left = pooled(threshold.detach() - 1e-7)
    right = pooled(threshold.detach() + 1e-7)
    jump = torch.linalg.vector_norm(right - left) / torch.linalg.vector_norm(tensor)
    assert jump < 2e-3


def test_full_conditioner_is_invariant_to_uniform_raw_duplicate_expansion():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]])
    values = _geometry(positions)
    query = _encoder()(*values, graph_count=1)
    torch.manual_seed(132)
    condition = torch.randn((1, 18))
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4, generic_chart_samples=1).eval()
    expanded = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4, generic_chart_samples=1).eval()
    expanded.load_state_dict(atlas.state_dict())
    def duplicate_measure(self, geometry, tensor_frame):
        rotations, prior = self._raw_candidate_measure(geometry, tensor_frame)
        order = torch.arange(rotations.shape[0] - 1, -1, -1, device=rotations.device)
        rotations = rotations[order].repeat_interleave(2, dim=0)
        prior = (prior[order] / 2.0).repeat_interleave(2, dim=0)
        return self._deduplicate_measure(rotations, prior)

    expanded._candidate_measure = types.MethodType(duplicate_measure, expanded)
    arguments = (
        condition,
        torch.ones((1, 1), dtype=torch.bool),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long),
        query,
        torch.tensor([0.4]),
    )
    reference = atlas(*arguments)
    duplicated = expanded(*arguments)
    assert duplicated.raw_candidate_count.item() == 2 * reference.raw_candidate_count.item()
    assert torch.allclose(duplicated.aligned_tensor, reference.aligned_tensor, atol=2e-6, rtol=2e-6)
    assert torch.allclose(duplicated.graph_condition, reference.graph_condition, atol=2e-6, rtol=2e-6)


def test_full_denoiser_has_no_finite_jump_at_generic_axial_candidate_switch():
    class PrescribedQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gap = 1e-3
            generator = torch.Generator().manual_seed(133)
            self.register_buffer("rank_three", torch.randn((1, 2, 3, 3, 3), generator=generator))

        def forward(self, *args):
            graph_count = int(args[-1])
            zero_first = self.rank_three.new_zeros((graph_count, 2, 3))
            zero_second = self.rank_three.new_zeros((graph_count, 2, 3, 3))
            zero_third = self.rank_three.new_zeros((graph_count, 2, 3, 3, 3))
            frame = torch.diag(self.rank_three.new_tensor((0.0, self.gap, 1.0))).unsqueeze(0)
            return CartesianGeometryQueries(
                first=zero_first,
                second=zero_second,
                third=zero_third,
                rank_three=self.rank_three.expand(graph_count, -1, -1, -1, -1),
                frame_tensor=frame.expand(graph_count, -1, -1),
            )

    torch.manual_seed(134)
    model = HybridCrystalDenoiser(hidden_dim=16, vector_dim=4, layers=1, radial_dim=4).eval()
    model.gauge_atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=4, generic_chart_samples=1).eval()
    query = PrescribedQuery()
    model.geometry_query_encoder = query
    arguments = dict(
        element_tokens=torch.tensor([4, 6, 12, 14]),
        frac_coords=torch.tensor([[0.05, 0.07, 0.11], [0.31, 0.13, 0.19], [0.17, 0.41, 0.23], [0.29, 0.37, 0.47]]),
        log_volume=torch.zeros(1),
        log_shape=torch.zeros((1, 6)),
        batch=torch.zeros(4, dtype=torch.long),
        time=torch.tensor([0.4]),
        tensor_condition=torch.randn((1, 18)),
        condition_present=torch.ones((1, 1), dtype=torch.bool),
        shape_projector=torch.eye(6).unsqueeze(0),
        fractional_to_cartesian=torch.eye(3).unsqueeze(0),
    )
    query.gap = model.gauge_atlas.relative_eigen_gap - 1e-7
    left = model(**arguments)
    query.gap = model.gauge_atlas.relative_eigen_gap + 1e-7
    right = model(**arguments)
    compared = (
        (left.clean_element_logits, right.clean_element_logits),
        (left.clean_composition_logits, right.clean_composition_logits),
        (
            left.coordinate_fractional_scaled_score,
            right.coordinate_fractional_scaled_score,
        ),
        (left.clean_volume_latent, right.clean_volume_latent),
        (left.clean_shape_latent, right.clean_shape_latent),
        (left.gauge_atlas.graph_condition, right.gauge_atlas.graph_condition),
    )
    for first, second in compared:
        normalized_jump = torch.linalg.vector_norm(second - first) / torch.linalg.vector_norm(first).clamp_min(1e-6)
        assert normalized_jump < 2e-3
