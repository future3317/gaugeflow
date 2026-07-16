import json
import runpy
from pathlib import Path

import pytest
import torch

from gaugeflow.production.cartesian_gauge_atlas import StratifiedCartesianGaugeAtlas
from gaugeflow.tensor import piezo_from_irreps

ROOT = Path(__file__).resolve().parents[1]


def test_s0_4_protocol_defines_replacement_prior_without_rewriting_s0_3():
    protocol = json.loads((ROOT / "configs" / "paper_s0_4_cartesian_atlas_prior_v1.json").read_text())
    assert protocol["predecessor"] == "paper_s0_3_cartesian_atlas_v1"
    assert protocol["status"] == "completed_failed_no_advance"
    assert protocol["official_run"]["failed_checks"] == ["cuda_latency"]
    assert "not an estimator required to reproduce" in protocol["scientific_definition"]
    assert "shared-score difference from archived K=3840 Hopf posterior" in protocol["diagnostic_only"]
    checks = " ".join(protocol["primary_checks"])
    assert "duplicate-expansion" in checks
    assert "descriptor-frame ambiguity" in checks
    assert "K=8,16,32,64" in checks


def test_s0_4_1_runtime_successor_is_frozen_without_reclassifying_s0_4():
    predecessor = json.loads((ROOT / "configs" / "paper_s0_4_cartesian_atlas_prior_v1.json").read_text())
    successor = json.loads((ROOT / "configs" / "paper_s0_4_1_cartesian_atlas_runtime_v1.json").read_text())
    metrics = json.loads(
        (ROOT / "reports" / "paper_s0_4_1_cartesian_atlas_runtime_v1" / "s0_4_1_metrics.json").read_text()
    )
    assert predecessor["status"] == "completed_failed_no_advance"
    assert predecessor["official_run"]["failed_checks"] == ["cuda_latency"]
    assert successor["status"] == "completed_passed_runtime_qualification"
    assert successor["official_run"]["decision"] == "passed_runtime_qualification"
    assert successor["scientific_semantics_frozen"]["generic_unique_candidates"] == 4032
    assert metrics["decision"] == "passed_runtime_qualification"
    assert all(metrics["checks"].values())
    assert metrics["cuda_benchmark"]["atlas_ms_per_forward"] <= 20.0


def test_generic_measure_and_soft_boundary_diagnostics_are_prepared():
    runner = runpy.run_path(str(ROOT / "scripts" / "run_paper_s0_4_cartesian_atlas_prior_v1.py"))
    candidate = runner["candidate_measure_audit"]()
    result = runner["stratum_boundary_audit"]()
    assert candidate["generic_raw_count"] == 4032
    assert candidate["generic_unique_count"] == 4032
    assert candidate["duplicate_expansion_aligned_difference"] <= 1e-10
    assert result["maximum_normalized_jump"] <= 0.10
    assert result["all_backward_gradients_finite"]


def test_s0_4_1_generic_fast_path_preserves_the_deduplicated_measure():
    atlas = StratifiedCartesianGaugeAtlas(16).double().eval()
    covariance = torch.diag(torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64))
    frame = atlas._frame_data(covariance, directional=True)
    raw_rotations, raw_prior = atlas._raw_candidate_measure(frame, frame)
    reference = atlas._deduplicate_measure(raw_rotations, raw_prior)
    optimized = atlas._candidate_measure(frame, frame)
    assert reference.raw_count == optimized.raw_count == 4032
    assert reference.rotations.shape[0] == optimized.rotations.shape[0] == 4032

    torch.manual_seed(10411)
    tensor = piezo_from_irreps(torch.randn((1, 18), dtype=torch.float64))[0]
    query = torch.randn((2, 3, 3, 3), dtype=torch.float64)

    def evaluate(measure):
        rotated = atlas._rotate_rank_three(tensor, measure.rotations)
        score = torch.einsum("fijk,cijk,c->f", rotated, query, atlas.score_channel)
        posterior = torch.softmax(score + measure.prior.log(), dim=0)
        aligned = torch.einsum("f,fijk->ijk", posterior, rotated)
        return posterior, aligned

    reference_posterior, reference_aligned = evaluate(reference)
    optimized_posterior, optimized_aligned = evaluate(optimized)
    assert torch.allclose(reference_aligned, optimized_aligned, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        reference_posterior.sort().values,
        optimized_posterior.sort().values,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        reference.prior.sort().values,
        optimized.prior.sort().values,
        atol=1e-15,
        rtol=1e-15,
    )


def test_axial_refinement_and_synthetic_coverage_diagnostics_are_prepared():
    runner = runpy.run_path(str(ROOT / "scripts" / "run_paper_s0_4_cartesian_atlas_prior_v1.py"))
    axial = runner["axial_refinement_audit"]()
    coverage = runner["synthetic_coverage_audit"]()
    assert axial["circle_samples"] == [8, 16, 32, 64]
    assert axial["successive_differences_monotone"], axial
    assert coverage["maximum_nearest_geodesic"] <= 0.40, coverage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 production path requires CUDA")
def test_bf16_autocast_production_path_is_finite_and_uses_4032_candidates():
    runner = runpy.run_path(str(ROOT / "scripts" / "run_paper_s0_4_cartesian_atlas_prior_v1.py"))
    helpers = runner["_load_s0_3_helpers"]()
    result = runner["mixed_precision_reference_audit"](helpers)
    assert result["finite"]
    assert result["fp32_vs_fp64"]["unique_candidate_count"] == 4032
    assert result["bf16_autocast_vs_fp64"]["unique_candidate_count"] == 4032
