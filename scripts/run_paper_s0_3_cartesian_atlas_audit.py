"""Frozen no-training numerical qualification for the Cartesian gauge atlas.

This script intentionally runs no sampler or optimizer.  It compares the
active Cartesian atlas with the archived Hopf rule only through a shared,
analytic Cartesian rank-three score; learned tokens from independently
parameterized conditioners are not scientifically comparable.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from gaugeflow.geometry import GaussianRadialBasis
from gaugeflow.production.archive_harmonic.harmonic_gaugeflow import (
    ConditionFreeGeometryQueryEncoder,
    HarmonicGaugeFlowConditioner,
)
from gaugeflow.production.cartesian_gauge_atlas import (
    CartesianSTFGeometryQueryEncoder,
    StratifiedCartesianGaugeAtlas,
)
from gaugeflow.production.equivariant_denoiser import HybridCrystalDenoiser
from gaugeflow.production.lattice_volume_shape import LatticeVolumeShape
from gaugeflow.production.so3_quadrature import nested_hopf_so3_grid
from gaugeflow.tensor import (
    fixed_lossless_response_probes,
    piezo_from_irreps,
    piezo_to_irreps,
    response_field,
    rotate_rank3,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paper_s0_3_cartesian_atlas_v1.json"
OUT = ROOT / "reports" / "paper_s0_3_cartesian_atlas_v1"


def rotation(seed: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((3, 3), generator=generator, dtype=dtype).to(device)
    frame, _ = torch.linalg.qr(value)
    if torch.linalg.det(frame) < 0:
        frame[:, 0] = -frame[:, 0]
    return frame


def geometry(
    positions: torch.Tensor, *, hidden: int, radial_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sites = positions.shape[0]
    source = torch.tensor([i for i in range(sites) for j in range(sites) if i != j], device=positions.device)
    target = torch.tensor([j for i in range(sites) for j in range(sites) if i != j], device=positions.device)
    displacement = positions[target] - positions[source]
    direction = torch.nn.functional.normalize(displacement, dim=-1)
    radial = GaussianRadialBasis(radial_dim, 8.0).to(positions)(torch.linalg.vector_norm(displacement, dim=-1))
    generator = torch.Generator(device="cpu").manual_seed(9301)
    nodes = torch.randn((sites, hidden), generator=generator, dtype=positions.dtype).to(positions.device)
    return (
        nodes,
        torch.zeros_like(nodes),
        source,
        target,
        direction,
        radial,
        torch.zeros(sites, dtype=torch.long, device=positions.device),
    )


def shared_score_reference(
    atlas: StratifiedCartesianGaugeAtlas,
    condition: torch.Tensor,
    query: torch.Tensor,
    frame_tensor: torch.Tensor,
) -> dict[str, float]:
    """Compare finite atlas and Hopf K=3840 with precisely the same score."""
    tensor = piezo_from_irreps(condition)[0]
    candidates = atlas._raw_candidate_measure(
        atlas._frame_data(frame_tensor[0]), atlas._frame_data(atlas.invariant(tensor.unsqueeze(0))[1][0])
    )[0]
    rotation_grid = nested_hopf_so3_grid(3840).to(tensor)
    channel = atlas.score_channel.to(query)
    atlas_rotated = atlas._rotate_rank_three(tensor, candidates)
    ref_rotated = rotate_rank3(tensor.expand(rotation_grid.shape[0], -1, -1, -1), rotation_grid)
    score_atlas = torch.einsum("fijk,cijk,c->f", atlas_rotated, query[0], channel)
    score_ref = torch.einsum("fijk,cijk,c->f", ref_rotated, query[0], channel)
    aligned_atlas = torch.einsum("f,fijk->ijk", torch.softmax(score_atlas, dim=0), atlas_rotated)
    aligned_ref = torch.einsum("f,fijk->ijk", torch.softmax(score_ref, dim=0), ref_rotated)
    probes = fixed_lossless_response_probes().to(tensor)
    response_atlas = response_field(aligned_atlas.unsqueeze(0), probes).squeeze(0)
    response_ref = response_field(aligned_ref.unsqueeze(0), probes).squeeze(0)
    return {
        "generic_frame_count": int(candidates.shape[0]),
        "aligned_relative_error": float(
            torch.linalg.vector_norm(aligned_atlas - aligned_ref)
            / torch.linalg.vector_norm(aligned_ref).clamp_min(1e-12)
        ),
        "response_relative_error": float(
            torch.linalg.vector_norm(response_atlas - response_ref)
            / torch.linalg.vector_norm(response_ref).clamp_min(1e-12)
        ),
    }


def covariance_metrics(
    device: torch.device, dtype: torch.dtype, *, seed_offset: int = 0
) -> dict[str, float | int]:
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]], dtype=dtype, device=device
    )
    values = geometry(positions, hidden=16, radial_dim=5)
    torch.manual_seed(9302 + seed_offset)
    encoder = CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3).to(device=device, dtype=dtype).eval()
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=8).to(device=device, dtype=dtype).eval()
    query = encoder(*values, graph_count=1)
    state, condition_rotation = rotation(
        9303 + seed_offset, dtype=dtype, device=device
    ), rotation(9304 + seed_offset, dtype=dtype, device=device)
    rotated_values = list(values)
    rotated_values[4] = values[4] @ state.T
    rotated_query = encoder(*rotated_values, graph_count=1)
    torch.manual_seed(9305 + seed_offset)
    condition = torch.randn((1, 18), dtype=dtype, device=device)
    rotated_condition = piezo_to_irreps(rotate_rank3(piezo_from_irreps(condition), condition_rotation).contiguous())
    arguments = (
        torch.ones((1, 1), dtype=torch.bool, device=device),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long, device=device),
        query,
        torch.tensor([0.4], dtype=dtype, device=device),
    )
    original = atlas(condition, *arguments)
    transformed = atlas(rotated_condition, arguments[0], rotated_values[4], arguments[2], rotated_query, arguments[4])
    tensor_error = torch.linalg.vector_norm(transformed.aligned_tensor - rotate_rank3(original.aligned_tensor, state))
    token_error = torch.linalg.vector_norm(transformed.graph_condition - original.graph_condition)
    zero = atlas(torch.zeros_like(condition), arguments[0], values[4], arguments[2], query, arguments[4])
    axial_tensor = torch.zeros((1, 3, 3, 3), dtype=dtype, device=device)
    axial_tensor[:, 0, 0, 0] = 1.0
    axial = atlas(piezo_to_irreps(axial_tensor), *arguments)
    return {
        "representative_tensor_error": float(tensor_error),
        "representative_token_error": float(token_error),
        "generic_frame_count": int(original.effective_frame_count.item()),
        "zero_frame_count": int(zero.effective_frame_count.item()),
        "zero_gate": float(zero.gate.item()),
        "axial_frame_count": int(axial.effective_frame_count.item()),
        "axial_kind": int(axial.residual_kind.item()),
    }


def denoiser_metrics() -> dict[str, float]:
    torch.manual_seed(9306)
    model = HybridCrystalDenoiser(hidden_dim=24, vector_dim=6, layers=2, radial_dim=5).eval()
    tokens = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    fractional = torch.tensor([[0.05, 0.10, 0.15], [0.35, 0.25, 0.68], [0.15, 0.73, 0.45], [0.72, 0.55, 0.20]])
    volume, shape, batch, time = (
        torch.tensor([math.log(27.0)]),
        torch.zeros((1, 6)),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([0.45]),
    )
    trace = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    projector = (torch.eye(6) - torch.outer(trace, trace) / 3.0).unsqueeze(0)
    chart = torch.eye(3).unsqueeze(0)
    values = (
        tokens,
        fractional,
        volume,
        shape,
        batch,
        time,
        torch.randn((1, 18)),
        torch.ones((1, 1), dtype=torch.bool),
        projector,
        chart,
    )
    original = model(*values)
    shifted_values = list(values)
    shifted_values[1] = fractional + torch.tensor([0.31, -0.27, 1.19])
    shifted = model(*shifted_values)
    translation = max(
        float((getattr(original, name) - getattr(shifted, name)).abs().max())
        for name in (
            "clean_element_logits",
            "coordinate_cartesian_score",
            "coordinate_fractional_score",
            "clean_log_volume",
            "clean_log_shape",
        )
    )
    basis = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transformed_values = list(values)
    transformed_values[1] = fractional @ torch.linalg.inv(basis)
    transformed_values[9] = chart @ basis.T
    transformed = model(*transformed_values)
    lattice = LatticeVolumeShape(volume, shape).lattice(chart)[0]
    lattice_t = LatticeVolumeShape(volume, shape).lattice(transformed_values[9])[0]
    rotation_matrix = torch.linalg.inv(lattice) @ torch.linalg.inv(basis) @ lattice_t
    basis_error = max(
        float((transformed.clean_element_logits - original.clean_element_logits).abs().max()),
        float(
            (transformed.coordinate_cartesian_score - original.coordinate_cartesian_score @ rotation_matrix).abs().max()
        ),
        float((transformed.coordinate_fractional_score - original.coordinate_fractional_score @ basis.T).abs().max()),
    )
    return {"translation_max_error": translation, "unimodular_max_error": basis_error}


def cuda_benchmark() -> dict[str, float | str]:
    if not torch.cuda.is_available():
        return {"status": "no_cuda"}
    device = torch.device("cuda")
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]], device=device)
    values = geometry(positions, hidden=64, radial_dim=8)
    torch.manual_seed(9307)
    encoder = CartesianSTFGeometryQueryEncoder(64, 8, query_channels=2, layers=3).to(device).eval()
    atlas = StratifiedCartesianGaugeAtlas(64, residual_circle_samples=8).to(device).eval()
    query = encoder(*values, graph_count=1)
    condition = torch.randn((1, 18), device=device)
    args = (
        condition,
        torch.ones((1, 1), dtype=torch.bool, device=device),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long, device=device),
        query,
        torch.tensor([0.4], device=device),
    )
    archived_query_encoder = ConditionFreeGeometryQueryEncoder(64, 8, query_channels=2, layers=3).to(device).eval()
    archived = HarmonicGaugeFlowConditioner(64, grid_size=3840, query_channels=2).to(device).eval()
    archived_query = archived_query_encoder(*values, graph_count=1)
    archived_args = (
        condition,
        torch.ones((1, 1), dtype=torch.bool, device=device),
        values[4],
        torch.zeros(values[4].shape[0], dtype=torch.long, device=device),
        archived_query,
        torch.tensor([0.4], device=device),
    )
    for _ in range(10):
        atlas(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(100):
        output = atlas(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0 / 100.0
    atlas_memory = torch.cuda.max_memory_allocated(device) / 2**20
    for _ in range(2):
        archived(*archived_args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(10):
        archived_output = archived(*archived_args)
    torch.cuda.synchronize()
    archived_elapsed = (time.perf_counter() - start) * 1000.0 / 10.0
    return {
        "status": torch.cuda.get_device_name(device),
        "atlas_ms_per_forward": elapsed,
        "atlas_peak_memory_mb": atlas_memory,
        "archived_hopf_k3840_ms_per_forward": archived_elapsed,
        "archived_hopf_k3840_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "finite": bool(
            torch.isfinite(output.graph_condition).all() and torch.isfinite(archived_output.graph_condition).all()
        ),
    }


def main() -> None:
    protocol = json.loads(CONFIG.read_text())
    torch.set_num_threads(1)
    cpu = covariance_metrics(torch.device("cpu"), torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.0, 0.35], [0.4, 0.1, 1.3]], dtype=torch.float64)
    values = geometry(positions, hidden=16, radial_dim=5)
    torch.manual_seed(9302)
    encoder = CartesianSTFGeometryQueryEncoder(16, 5, query_channels=2, layers=3).double().eval()
    atlas = StratifiedCartesianGaugeAtlas(16, residual_circle_samples=8).double().eval()
    query = encoder(*values, graph_count=1)
    torch.manual_seed(9305)
    reference_panel = []
    for seed in (9310, 9311, 9312, 9313):
        torch.manual_seed(seed)
        reference_panel.append(
            shared_score_reference(
                atlas, torch.randn((1, 18), dtype=torch.float64), query.rank_three, query.frame_tensor
            )
        )
    reference = {
        "panel_size": len(reference_panel),
        "generic_frame_count": reference_panel[0]["generic_frame_count"],
        "aligned_relative_error_mean": sum(item["aligned_relative_error"] for item in reference_panel)
        / len(reference_panel),
        "aligned_relative_error_max": max(item["aligned_relative_error"] for item in reference_panel),
        "response_relative_error_mean": sum(item["response_relative_error"] for item in reference_panel)
        / len(reference_panel),
        "response_relative_error_max": max(item["response_relative_error"] for item in reference_panel),
    }
    cuda = (
        covariance_metrics(torch.device("cuda"), torch.float32) if torch.cuda.is_available() else {"status": "no_cuda"}
    )
    result = {
        "protocol": protocol["protocol_id"],
        "scope": protocol["scope"],
        "cpu_float64": cpu,
        "shared_score_k3840_reference": reference,
        "denoiser": denoiser_metrics(),
        "cuda_float32": cuda,
        "cuda_benchmark": cuda_benchmark(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s0_3_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "s0_3_report.md").write_text(
        "# S0.3 Cartesian-atlas numerical qualification\n\n"
        + "```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n\n"
        + (
            "The K=3840 comparison uses a shared analytic Cartesian contraction score; "
            "it is not a learned-token accuracy or generation metric. The result is an "
            "audit measurement, not a training or generation result.\n"
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
