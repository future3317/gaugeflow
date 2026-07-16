"""Run the pre-registered no-training S0.4 Cartesian-atlas-prior audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from gaugeflow.production.cartesian_gauge_atlas import StratifiedCartesianGaugeAtlas
from gaugeflow.tensor import piezo_from_irreps, piezo_to_irreps, rotate_rank3

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paper_s0_4_cartesian_atlas_prior_v1.json"
OUT = ROOT / "reports" / "paper_s0_4_cartesian_atlas_prior_v1"


def _load_s0_3_helpers():
    path = Path(__file__).with_name("run_paper_s0_3_cartesian_atlas_audit.py")
    spec = importlib.util.spec_from_file_location("s0_3_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen S0.3 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manual_pool(
    atlas: StratifiedCartesianGaugeAtlas,
    tensor: torch.Tensor,
    query: torch.Tensor,
    geometry_covariance: torch.Tensor,
    condition_covariance: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    measure = atlas._candidate_measure(
        atlas._frame_data(geometry_covariance, directional=True),
        atlas._frame_data(condition_covariance, directional=True),
    )
    if measure.rotations.shape[0] == 0:
        return torch.zeros_like(tensor), 0, 0
    rotated = atlas._rotate_rank_three(tensor, measure.rotations)
    score = torch.einsum("fijk,cijk,c->f", rotated, query, atlas.score_channel.to(query))
    posterior = torch.softmax(score + measure.prior.log(), dim=0)
    pooled = torch.einsum("f,fijk->ijk", posterior, rotated)
    return pooled, measure.raw_count, int(measure.rotations.shape[0])


def _posterior_relabel_metrics(
    original_measure,
    transformed_measure,
    original_posterior: torch.Tensor,
    transformed_posterior: torch.Tensor,
    state_rotation: torch.Tensor,
    condition_rotation: torch.Tensor,
) -> tuple[float, float]:
    """Compare posterior atoms after the required left/right pushforward."""
    expected = state_rotation @ original_measure.rotations @ condition_rotation.T
    candidates = transformed_measure.rotations
    if expected.shape != candidates.shape:
        return float("inf"), float("inf")
    matching_parts = []
    distance_parts = []
    # Chunking avoids materializing the full [4032,4032,3,3] tensor.
    for start in range(0, expected.shape[0], 128):
        distance = torch.linalg.matrix_norm(expected[start : start + 128, None] - candidates[None], dim=(-2, -1))
        minimum, matching = distance.min(dim=-1)
        matching_parts.append(matching)
        distance_parts.append(minimum)
    matching = torch.cat(matching_parts)
    minimum = torch.cat(distance_parts)
    if torch.unique(matching).numel() != matching.numel():
        return float("inf"), float(minimum.max())
    posterior_l1 = (original_posterior - transformed_posterior[matching]).abs().sum()
    return float(posterior_l1), float(minimum.max())


def representative_covariance_metrics(
    helpers, device: torch.device, dtype: torch.dtype, *, seed_offset: int
) -> dict[str, float | int]:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]],
        dtype=dtype,
        device=device,
    )
    values = helpers.geometry(positions, hidden=16, radial_dim=5)
    torch.manual_seed(9302 + seed_offset)
    encoder = (
        helpers.CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3)
        .to(device=device, dtype=dtype)
        .eval()
    )
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=8).to(device=device, dtype=dtype).eval()
    query = encoder(*values, graph_count=1)
    state = helpers.rotation(9303 + seed_offset, dtype=dtype, device=device)
    condition_rotation = helpers.rotation(9304 + seed_offset, dtype=dtype, device=device)
    rotated_values = list(values)
    rotated_values[4] = values[4] @ state.T
    rotated_query = encoder(*rotated_values, graph_count=1)
    torch.manual_seed(9305 + seed_offset)
    condition = torch.randn((1, 18), dtype=dtype, device=device)
    rotated_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), condition_rotation).contiguous())
    present = torch.ones((1, 1), dtype=torch.bool, device=device)
    edge_graph = torch.zeros(values[4].shape[0], dtype=torch.long, device=device)
    time = torch.tensor([0.4], dtype=dtype, device=device)
    original = atlas(condition, present, values[4], edge_graph, query, time)
    transformed = atlas(rotated_condition, present, rotated_values[4], edge_graph, rotated_query, time)

    tensor = piezo_from_irreps(condition)
    transformed_tensor = piezo_from_irreps(rotated_condition)
    _, condition_covariance = atlas.invariant(tensor)
    _, transformed_condition_covariance = atlas.invariant(transformed_tensor)
    original_measure = atlas._candidate_measure(
        atlas._frame_data(query.frame_tensor[0], directional=True),
        atlas._frame_data(condition_covariance[0], directional=True),
    )
    transformed_measure = atlas._candidate_measure(
        atlas._frame_data(rotated_query.frame_tensor[0], directional=True),
        atlas._frame_data(transformed_condition_covariance[0], directional=True),
    )
    count = int(original.effective_frame_count.item())
    posterior_l1, candidate_pushforward_error = _posterior_relabel_metrics(
        original_measure,
        transformed_measure,
        original.posterior[0, :count],
        transformed.posterior[0, :count],
        state,
        condition_rotation,
    )
    expected_tensor = rotate_rank3(original.aligned_tensor, state)
    tensor_error = torch.linalg.vector_norm(transformed.aligned_tensor - expected_tensor)
    expected_response = original.edge_response @ state.T
    response_error = torch.linalg.vector_norm(transformed.edge_response - expected_response)
    response_scale = torch.linalg.vector_norm(expected_response).clamp_min(1e-12)
    token_error = torch.linalg.vector_norm(transformed.graph_condition - original.graph_condition)
    return {
        "representative_posterior_l1_error": posterior_l1,
        "candidate_pushforward_max_error": candidate_pushforward_error,
        "representative_tensor_error": float(tensor_error),
        "representative_response_relative_error": float(response_error / response_scale),
        "representative_token_error": float(token_error),
        "generic_raw_candidate_count": int(original.raw_candidate_count.item()),
        "generic_unique_candidate_count": count,
    }


def mixed_precision_reference_audit(helpers) -> dict[str, object]:
    """Compare actual CUDA AMP paths with an identical-weight FP64 model."""
    if not torch.cuda.is_available():
        return {"status": "no_cuda", "finite": False}
    device = torch.device("cuda")
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]], device=device)
    values32 = helpers.geometry(positions, hidden=16, radial_dim=5)
    torch.manual_seed(10406)
    encoder32 = helpers.CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3).to(device).eval()
    atlas32 = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=8).to(device).eval()
    torch.manual_seed(10407)
    condition32 = torch.randn((1, 18), device=device)
    present = torch.ones((1, 1), dtype=torch.bool, device=device)
    edge_graph = torch.zeros(values32[4].shape[0], dtype=torch.long, device=device)
    time32 = torch.tensor([0.4], device=device)

    query32 = encoder32(*values32, graph_count=1)
    output32 = atlas32(condition32, present, values32[4], edge_graph, query32, time32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        query_bf16 = encoder32(*values32, graph_count=1)
        output_bf16 = atlas32(condition32, present, values32[4], edge_graph, query_bf16, time32)

    encoder64 = (
        helpers.CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3)
        .to(device=device, dtype=torch.float64)
        .eval()
    )
    atlas64 = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=8).to(device=device, dtype=torch.float64).eval()
    encoder64.load_state_dict(encoder32.state_dict())
    atlas64.load_state_dict(atlas32.state_dict())
    values64 = tuple(value.double() if value.is_floating_point() else value for value in values32)
    query64 = encoder64(*values64, graph_count=1)
    output64 = atlas64(condition32.double(), present, values64[4], edge_graph, query64, time32.double())

    count = int(output64.effective_frame_count.item())

    def relative(value: torch.Tensor, reference: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(value.double() - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        )

    def comparison(output) -> dict[str, float | int]:
        posterior = output.posterior[0, :count].double().sort().values
        reference_posterior = output64.posterior[0, :count].sort().values
        return {
            "aligned_relative_error": relative(output.aligned_tensor, output64.aligned_tensor),
            "response_relative_error": relative(output.edge_response, output64.edge_response),
            "token_relative_error": relative(output.graph_condition, output64.graph_condition),
            "sorted_posterior_l1_error": float((posterior - reference_posterior).abs().sum()),
            "unique_candidate_count": int(output.effective_frame_count.item()),
        }

    floating_outputs = (
        output_bf16.aligned_tensor,
        output_bf16.edge_response,
        output_bf16.graph_condition,
        output_bf16.posterior,
    )
    return {
        "status": torch.cuda.get_device_name(device),
        "fp32_vs_fp64": comparison(output32),
        "bf16_autocast_vs_fp64": comparison(output_bf16),
        "finite": all(bool(torch.isfinite(value).all()) for value in floating_outputs),
    }


def candidate_measure_audit() -> dict[str, float | int | list[int]]:
    """Audit the discrete prior rather than treating enumeration as a set."""
    torch.manual_seed(10400)
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    generic_covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    axial_covariance = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    generic = atlas._frame_data(generic_covariance, directional=True)
    axial = atlas._frame_data(axial_covariance, directional=True)
    generic_measure = atlas._candidate_measure(generic, generic)
    axial_measure = atlas._candidate_measure(axial, generic)
    worst_axial_measure = atlas._candidate_measure(axial, axial)

    tensor = torch.randn((3, 3, 3), dtype=torch.float64)
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)

    def pooled(measure):
        rotated = atlas._rotate_rank_three(tensor, measure.rotations)
        score = torch.einsum("fabc,qabc,q->f", rotated, query, atlas.score_channel.to(query))
        posterior = torch.softmax(score + measure.prior.log(), dim=0)
        return posterior, torch.einsum("f,fijk->ijk", posterior, rotated)

    posterior, aligned = pooled(generic_measure)
    order = torch.arange(generic_measure.raw_count - 1, -1, -1)
    raw_rotations, raw_prior = atlas._raw_candidate_measure(generic, generic)
    permuted = atlas._deduplicate_measure(raw_rotations[order], raw_prior[order])
    expanded = atlas._deduplicate_measure(
        raw_rotations.repeat_interleave(3, dim=0),
        (raw_prior / 3.0).repeat_interleave(3, dim=0),
    )
    permuted_posterior, permuted_aligned = pooled(permuted)
    expanded_posterior, expanded_aligned = pooled(expanded)
    # Posterior entries are in the sorted deduplicated rotation order, so they
    # can be compared directly after arbitrary raw enumeration order.
    multiplicities = torch.round(generic_measure.prior * generic_measure.raw_count).to(torch.long)
    return {
        "generic_raw_count": generic_measure.raw_count,
        "generic_unique_count": int(generic_measure.rotations.shape[0]),
        "generic_multiplicity_min_max": [int(multiplicities.min()), int(multiplicities.max())],
        "generic_effective_prior_rank": int((generic_measure.prior > 0).sum()),
        "one_sided_axial_raw_count": axial_measure.raw_count,
        "one_sided_axial_unique_count": int(axial_measure.rotations.shape[0]),
        "worst_axial_raw_count": worst_axial_measure.raw_count,
        "worst_axial_unique_count": int(worst_axial_measure.rotations.shape[0]),
        "enumeration_order_posterior_difference": float(torch.linalg.vector_norm(permuted_posterior - posterior)),
        "enumeration_order_aligned_difference": float(torch.linalg.vector_norm(permuted_aligned - aligned)),
        "duplicate_expansion_posterior_difference": float(torch.linalg.vector_norm(expanded_posterior - posterior)),
        "duplicate_expansion_aligned_difference": float(torch.linalg.vector_norm(expanded_aligned - aligned)),
    }


def stratum_boundary_audit() -> dict[str, object]:
    torch.manual_seed(10401)
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))[0]
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)
    condition_covariance = torch.diag(torch.tensor([0.0, 0.23, 1.0], dtype=torch.float64))
    multipliers = (0.20, 0.49, 0.50, 0.80, 0.999, 1.001, 1.20, 1.99, 2.00, 2.01, 3.0)
    pooled, raw_counts, unique_counts = [], [], []
    for multiplier in multipliers:
        gap = atlas.relative_eigen_gap * multiplier
        covariance = torch.diag(torch.tensor([0.0, gap, 1.0], dtype=torch.float64))
        value, raw_count, unique_count = _manual_pool(atlas, tensor, query, covariance, condition_covariance)
        pooled.append(value)
        raw_counts.append(raw_count)
        unique_counts.append(unique_count)
    scale = torch.linalg.vector_norm(tensor).clamp_min(1e-12)
    jumps = [float(torch.linalg.vector_norm(right - left) / scale) for left, right in zip(pooled, pooled[1:])]
    gradient_finite = []
    infinitesimal_jumps = []
    for center in (0.5, 1.0, 2.0):
        gap = torch.tensor(atlas.relative_eigen_gap * center, dtype=torch.float64, requires_grad=True)
        covariance = torch.diag(torch.stack((gap.new_zeros(()), gap, gap.new_ones(()))))
        value, _, _ = _manual_pool(atlas, tensor, query, covariance, condition_covariance)
        value.square().sum().backward()
        gradient_finite.append(bool(gap.grad is not None and torch.isfinite(gap.grad)))
        epsilon = atlas.relative_eigen_gap * 1e-4
        left, _, _ = _manual_pool(
            atlas,
            tensor,
            query,
            torch.diag(torch.tensor([0.0, float(gap.detach()) - epsilon, 1.0], dtype=torch.float64)),
            condition_covariance,
        )
        right, _, _ = _manual_pool(
            atlas,
            tensor,
            query,
            torch.diag(torch.tensor([0.0, float(gap.detach()) + epsilon, 1.0], dtype=torch.float64)),
            condition_covariance,
        )
        infinitesimal_jumps.append(float(torch.linalg.vector_norm(right - left) / scale))
    return {
        "gap_multipliers": list(multipliers),
        "raw_candidate_counts": raw_counts,
        "unique_candidate_counts": unique_counts,
        "normalized_jumps": jumps,
        "maximum_normalized_jump": max(jumps),
        "infinitesimal_normalized_jumps": infinitesimal_jumps,
        "maximum_infinitesimal_normalized_jump": max(infinitesimal_jumps),
        "all_backward_gradients_finite": all(gradient_finite),
    }


def axial_refinement_audit() -> dict[str, object]:
    """One-sided axial SO(2) refinement without changing the base cubature."""
    torch.manual_seed(10403)
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))[0]
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)
    axial = torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    generic = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    samples = (8, 16, 32, 64)
    pooled, raw_counts, unique_counts = [], [], []
    for count in samples:
        atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=count).double().eval()
        value, raw_count, unique_count = _manual_pool(atlas, tensor, query, axial, generic)
        pooled.append(value)
        raw_counts.append(raw_count)
        unique_counts.append(unique_count)
    scale = torch.linalg.vector_norm(tensor).clamp_min(1e-12)
    differences = [float(torch.linalg.vector_norm(right - left) / scale) for left, right in zip(pooled, pooled[1:])]
    monotone = all(right <= left + 1e-12 for left, right in zip(differences, differences[1:]))
    return {
        "circle_samples": list(samples),
        "raw_candidate_counts": raw_counts,
        "unique_candidate_counts": unique_counts,
        "successive_normalized_differences": differences,
        "successive_differences_monotone": monotone,
    }


def synthetic_coverage_audit() -> dict[str, object]:
    """Measure finite-prior coverage on independent known relative rotations."""
    torch.manual_seed(10404)
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    frame = atlas._frame_data(covariance, directional=True)
    measure = atlas._candidate_measure(frame, frame)
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))[0]
    generator = torch.Generator().manual_seed(10405)
    rows = []
    for _ in range(8):
        matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        target, _ = torch.linalg.qr(matrix)
        if torch.linalg.det(target) < 0:
            target[:, 0] = -target[:, 0]
        relative = measure.rotations.transpose(-1, -2) @ target
        cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        geodesic = torch.acos(cosine)
        nearest = int(torch.argmin(geodesic))
        target_tensor = atlas._rotate_rank_three(tensor, target.unsqueeze(0))[0]
        rotated = atlas._rotate_rank_three(tensor, measure.rotations)
        score = torch.einsum("fijk,ijk->f", rotated, target_tensor)
        posterior = torch.softmax(score + measure.prior.log(), dim=0)
        maximum = int(torch.argmax(posterior))
        rows.append(
            {
                "nearest_geodesic": float(geodesic[nearest]),
                "posterior_mode_geodesic": float(geodesic[maximum]),
                "nearest_candidate_is_posterior_mode": nearest == maximum,
                "nearest_candidate_posterior_mass": float(posterior[nearest]),
            }
        )
    return {
        "panel": rows,
        "maximum_nearest_geodesic": max(row["nearest_geodesic"] for row in rows),
        "mean_posterior_mode_geodesic": sum(row["posterior_mode_geodesic"] for row in rows) / len(rows),
        "posterior_mode_retrieval_rate": sum(row["nearest_candidate_is_posterior_mode"] for row in rows) / len(rows),
    }


def parity_and_zero_null_audit(helpers) -> dict[str, float | int]:
    torch.manual_seed(10402)
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    generic = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    candidates = atlas._raw_candidate_measure(atlas._frame_data(generic), atlas._frame_data(generic))[0]
    determinant_error = float((torch.linalg.det(candidates) - 1.0).abs().max())
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))
    inversion = -torch.eye(3, dtype=torch.float64)
    parity_error = float(
        torch.linalg.vector_norm(rotate_rank3(tensor, inversion) + tensor)
        / torch.linalg.vector_norm(tensor).clamp_min(1e-12)
    )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]],
        dtype=torch.float64,
    )
    values = helpers.geometry(positions, hidden=16, radial_dim=5)
    encoder = helpers.CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3).double().eval()
    query = encoder(*values, graph_count=1)
    edge_graph = torch.zeros(values[4].shape[0], dtype=torch.long)
    time = torch.tensor([0.4], dtype=torch.float64)
    zero_condition = torch.zeros((1, 18), dtype=torch.float64)
    present = atlas(zero_condition, torch.ones((1, 1), dtype=torch.bool), values[4], edge_graph, query, time)
    null = atlas(zero_condition, torch.zeros((1, 1), dtype=torch.bool), values[4], edge_graph, query, time)
    token_distance = float(torch.linalg.vector_norm(present.graph_condition - null.graph_condition))
    descriptor_isotropic_tensor = torch.zeros((1, 3, 3, 3), dtype=torch.float64)
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        descriptor_isotropic_tensor[(0,) + permutation] = 1.0
    _, descriptor_covariance = atlas.invariant(descriptor_isotropic_tensor)
    descriptor_frame = atlas._frame_data(descriptor_covariance[0], directional=True)
    generic_frame = atlas._frame_data(generic, directional=True)
    descriptor_measure = atlas._candidate_measure(generic_frame, descriptor_frame)
    return {
        "candidate_count": int(candidates.shape[0]),
        "proper_rotation_determinant_error": determinant_error,
        "polar_rank3_parity_error": parity_error,
        "physical_zero_null_token_distance": token_distance,
        "nonzero_descriptor_isotropic_unique_candidates": int(descriptor_measure.rotations.shape[0]),
    }


def _maximum(panel: list[dict[str, float | int]], key: str) -> float:
    return max(float(row[key]) for row in panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official",
        action="store_true",
        help="write an official S0.4 result after explicit protocol authorization",
    )
    arguments = parser.parse_args()
    if not arguments.official:
        raise SystemExit(
            "S0.4-v1 is pre-registered but officially unrun. "
            "Use --official only after explicit authorization; unit tests call diagnostics directly."
        )
    protocol = json.loads(CONFIG.read_text())
    if protocol["status"] != "preregistered_not_run":
        raise SystemExit(
            f"S0.4-v1 is frozen with status {protocol['status']}; "
            "the official runner may not overwrite a completed result."
        )
    thresholds = protocol["frozen_thresholds"]
    helpers = _load_s0_3_helpers()
    offsets = protocol["fixed_parameters"]["unseen_seed_offsets"]
    cpu_panel = [
        representative_covariance_metrics(helpers, torch.device("cpu"), torch.float64, seed_offset=value)
        for value in offsets
    ]
    cuda_panel = (
        [
            representative_covariance_metrics(helpers, torch.device("cuda"), torch.float32, seed_offset=value)
            for value in offsets
        ]
        if torch.cuda.is_available()
        else []
    )
    stratum = stratum_boundary_audit()
    candidate_measure = candidate_measure_audit()
    axial_refinement = axial_refinement_audit()
    synthetic_coverage = synthetic_coverage_audit()
    parity = parity_and_zero_null_audit(helpers)
    precision = mixed_precision_reference_audit(helpers)
    fp32_precision = precision.get("fp32_vs_fp64", {})
    bf16_precision = precision.get("bf16_autocast_vs_fp64", {})
    if not isinstance(fp32_precision, dict) or not isinstance(bf16_precision, dict):
        raise RuntimeError("mixed-precision audit returned an invalid result schema")
    denoiser = helpers.denoiser_metrics()
    benchmark = helpers.cuda_benchmark()

    checks = {
        "cpu_representative_tensor": _maximum(cpu_panel, "representative_tensor_error")
        <= thresholds["cpu_representative_tensor_error_max"],
        "cpu_representative_token": _maximum(cpu_panel, "representative_token_error")
        <= thresholds["cpu_representative_token_error_max"],
        "cpu_representative_posterior": _maximum(cpu_panel, "representative_posterior_l1_error")
        <= thresholds["cpu_representative_posterior_l1_error_max"],
        "cpu_representative_response": _maximum(cpu_panel, "representative_response_relative_error")
        <= thresholds["cpu_representative_response_relative_error_max"],
        "cuda_representative_tensor": bool(cuda_panel)
        and _maximum(cuda_panel, "representative_tensor_error") <= thresholds["cuda_representative_tensor_error_max"],
        "cuda_representative_token": bool(cuda_panel)
        and _maximum(cuda_panel, "representative_token_error") <= thresholds["cuda_representative_token_error_max"],
        "cuda_representative_posterior": bool(cuda_panel)
        and _maximum(cuda_panel, "representative_posterior_l1_error")
        <= thresholds["cuda_representative_posterior_l1_error_max"],
        "cuda_representative_response": bool(cuda_panel)
        and _maximum(cuda_panel, "representative_response_relative_error")
        <= thresholds["cuda_representative_response_relative_error_max"],
        "proper_rotation": parity["proper_rotation_determinant_error"]
        <= thresholds["proper_rotation_determinant_error_max"],
        "polar_parity": parity["polar_rank3_parity_error"] <= thresholds["polar_rank3_parity_error_max"],
        "stratum_boundary": stratum["maximum_normalized_jump"] <= thresholds["stratum_boundary_normalized_jump_max"],
        "stratum_infinitesimal_continuity": stratum["maximum_infinitesimal_normalized_jump"]
        <= thresholds["stratum_infinitesimal_token_jump_max"],
        "stratum_backward_gradients": stratum["all_backward_gradients_finite"],
        "candidate_enumeration_order": candidate_measure["enumeration_order_aligned_difference"]
        <= thresholds["candidate_measure_invariance_max"],
        "candidate_enumeration_order_posterior": candidate_measure["enumeration_order_posterior_difference"]
        <= thresholds["candidate_measure_invariance_max"],
        "candidate_duplicate_expansion": candidate_measure["duplicate_expansion_aligned_difference"]
        <= thresholds["candidate_measure_invariance_max"],
        "candidate_duplicate_expansion_posterior": candidate_measure["duplicate_expansion_posterior_difference"]
        <= thresholds["candidate_measure_invariance_max"],
        "generic_candidate_count": candidate_measure["generic_raw_count"]
        == protocol["fixed_parameters"]["generic_two_sided_candidate_count"]
        and candidate_measure["generic_unique_count"]
        == protocol["fixed_parameters"]["generic_two_sided_candidate_count"],
        "nonzero_descriptor_isotropic": parity["nonzero_descriptor_isotropic_unique_candidates"] > 0,
        "axial_refinement": axial_refinement["successive_differences_monotone"],
        "synthetic_coverage": synthetic_coverage["maximum_nearest_geodesic"]
        <= thresholds["synthetic_nearest_candidate_geodesic_max"],
        "fp32_aligned_reference": fp32_precision.get("aligned_relative_error", float("inf"))
        <= thresholds["fp32_vs_fp64_aligned_relative_error_max"],
        "fp32_response_reference": fp32_precision.get("response_relative_error", float("inf"))
        <= thresholds["fp32_vs_fp64_response_relative_error_max"],
        "fp32_token_reference": fp32_precision.get("token_relative_error", float("inf"))
        <= thresholds["fp32_vs_fp64_token_relative_error_max"],
        "fp32_posterior_reference": fp32_precision.get("sorted_posterior_l1_error", float("inf"))
        <= thresholds["fp32_vs_fp64_sorted_posterior_l1_error_max"],
        "bf16_aligned_reference": bf16_precision.get("aligned_relative_error", float("inf"))
        <= thresholds["bf16_vs_fp64_aligned_relative_error_max"],
        "bf16_response_reference": bf16_precision.get("response_relative_error", float("inf"))
        <= thresholds["bf16_vs_fp64_response_relative_error_max"],
        "bf16_token_reference": bf16_precision.get("token_relative_error", float("inf"))
        <= thresholds["bf16_vs_fp64_token_relative_error_max"],
        "bf16_posterior_reference": bf16_precision.get("sorted_posterior_l1_error", float("inf"))
        <= thresholds["bf16_vs_fp64_sorted_posterior_l1_error_max"],
        "bf16_finite": bool(precision.get("finite", False)) == thresholds["bf16_all_outputs_finite"],
        "translation": denoiser["translation_max_error"] <= thresholds["denoiser_translation_error_max"],
        "unimodular": denoiser["unimodular_max_error"] <= thresholds["denoiser_unimodular_error_max"],
        "zero_null": parity["physical_zero_null_token_distance"] >= thresholds["physical_zero_null_token_distance_min"],
        "cuda_latency": benchmark.get("atlas_ms_per_forward", float("inf")) <= thresholds["cuda_atlas_latency_ms_max"],
        "cuda_memory": benchmark.get("atlas_peak_memory_mb", float("inf"))
        <= thresholds["cuda_atlas_peak_memory_mb_max"],
        "finite": bool(benchmark.get("finite", False)),
    }
    passed = all(checks.values())
    result = {
        "protocol_id": protocol["protocol_id"],
        "decision": "passed_operator_prior_qualification" if passed else "failed_no_advance",
        "checks": checks,
        "cpu_unseen_panel": cpu_panel,
        "cuda_unseen_panel": cuda_panel,
        "parity_and_zero_null": parity,
        "candidate_measure": candidate_measure,
        "stratum_boundary": stratum,
        "axial_refinement": axial_refinement,
        "synthetic_coverage": synthetic_coverage,
        "mixed_precision_reference": precision,
        "denoiser": denoiser,
        "cuda_benchmark": benchmark,
        "hopf_comparison_role": "diagnostic_only_not_an_acceptance_check",
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s0_4_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    failed = [name for name, value in checks.items() if not value]
    report = (
        "# S0.4 Cartesian Atlas Prior qualification\n\n"
        f"Decision: **{result['decision']}**.\n\n"
        "This protocol qualifies the atlas as a replacement prior and quotient interface. "
        "It does not require reproduction of the archived Hopf posterior.\n\n"
        f"Failed primary checks: {failed if failed else 'none'}.\n\n"
        "```json\n" + json.dumps(result, indent=2) + "\n```\n"
    )
    (OUT / "s0_4_report.md").write_text(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
