import json

import pytest
import torch

from gaugeflow.vnext.diagnostics import (
    adaptive_rk4,
    analytic_endpoint_jacobians,
    audit_representation_collisions,
    euler_integrate,
    exact_equivalence_risk,
    knn_local_target_dispersion,
    reduced_vector_jacobian,
    rk4_integrate,
    variational_flow_jacobian,
)
from gaugeflow.vnext.experiments import GateBlockedError, require_gate_status
from gaugeflow.vnext.experiments.q0_c0_audit import _git_commit_from_metadata


def test_knn_dispersion_is_not_reported_as_conditional_variance():
    target = torch.tensor([[0.0], [0.0], [2.0], [2.0]], dtype=torch.float64)
    separated = torch.tensor([[0.0], [0.01], [1.0], [1.01]], dtype=torch.float64)
    collapsed = torch.zeros_like(separated)
    separated_result = knn_local_target_dispersion(separated, target, neighbors=1)
    collapsed_result = knn_local_target_dispersion(collapsed, target, neighbors=3)
    assert separated_result.trace_dispersion == 0.0
    assert torch.allclose(collapsed_result.normalized_trace_dispersion, torch.tensor(1.0, dtype=torch.float64))


def test_exact_full_state_conditional_variance_is_zero_before_endpoint():
    representation = torch.tensor([[0.0], [0.3], [0.8], [1.4]], dtype=torch.float64)
    target = torch.tensor([[2.0], [-1.0], [4.0], [0.5]], dtype=torch.float64)
    risk = exact_equivalence_risk(representation, target, absolute_tolerance=1.0e-12)
    assert risk.exact_collision_count == 0
    assert risk.trace_risk == 0.0


def test_endpoint_alias_variance_equals_global_target_variance():
    representation = torch.zeros((4, 2), dtype=torch.float64)
    target = torch.tensor([[0.0], [1.0], [3.0], [8.0]], dtype=torch.float64)
    risk = exact_equivalence_risk(representation, target, absolute_tolerance=1.0e-12)
    assert risk.exact_collision_count == 6
    assert torch.allclose(risk.trace_risk, risk.target_trace_variance)
    assert torch.allclose(risk.normalized_trace_risk, torch.ones((), dtype=torch.float64))


def test_representation_collision_uses_dimensionless_target_jump():
    representation = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    target = torch.tensor([[0.0], [4.0], [4.1]], dtype=torch.float64)
    audit, witnesses = audit_representation_collisions(
        representation,
        target,
        exact_absolute_tolerance=1.0e-12,
        near_quantile=0.34,
        alias_target_distance_min=1.0,
        distance_floor=1.0e-8,
    )
    assert audit.near_pair_count >= 1
    assert audit.exact_collision_count == 1
    assert audit.alias_collision_count >= 1
    assert audit.max_local_target_ratio > 1.0e6
    assert witnesses[0].exact_representation_collision


def test_reduced_and_analytic_jacobians_match_exact_linear_flow():
    matrix = torch.tensor([[1.0, 2.0], [-0.5, 3.0]], dtype=torch.float64)

    def field(value, _time):
        return matrix @ value

    state = torch.tensor([0.2, -0.7], dtype=torch.float64)
    assert torch.allclose(reduced_vector_jacobian(field, state, torch.tensor(0.3)), matrix)
    result = analytic_endpoint_jacobians(2, torch.tensor(0.75, dtype=torch.float64))
    assert torch.allclose(result.vector_jacobian, -4.0 * torch.eye(2, dtype=torch.float64))
    assert torch.allclose(result.flow_jacobian, 0.25 * torch.eye(2, dtype=torch.float64))
    terminal = analytic_endpoint_jacobians(2, torch.tensor(1.0, dtype=torch.float64))
    assert terminal.vector_jacobian is None
    assert torch.isneginf(terminal.log_abs_det)


def test_variational_rk4_and_state_solvers_converge_on_regular_field():
    end = 0.7
    flow_jacobian = variational_flow_jacobian(
        lambda _time: -torch.eye(2, dtype=torch.float64),
        dimension=2,
        end_time=end,
        steps=64,
    )
    expected_jacobian = torch.exp(torch.tensor(-end, dtype=torch.float64)) * torch.eye(2, dtype=torch.float64)
    assert torch.allclose(flow_jacobian, expected_jacobian, atol=1.0e-10)
    initial = torch.tensor([1.0, -2.0], dtype=torch.float64)

    def field(value, _time):
        return -value

    expected = torch.exp(torch.tensor(-end, dtype=torch.float64)) * initial
    euler = euler_integrate(field, initial, start=0.0, end=end, steps=1024)
    rk4 = rk4_integrate(field, initial, start=0.0, end=end, steps=32)
    adaptive = adaptive_rk4(field, initial, start=0.0, end=end, rtol=1.0e-9, atol=1.0e-11)
    assert torch.allclose(euler.state, expected, atol=5.0e-4)
    assert torch.allclose(rk4.state, expected, atol=1.0e-9)
    assert torch.allclose(adaptive.state, expected, atol=1.0e-8)


def test_gate_status_blocks_missing_wrong_and_failed_predecessors(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(GateBlockedError, match="missing"):
        require_gate_status(missing, gate="Q0", accepted=frozenset({"complete"}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"gate": "Q0", "status": "blocked"}), encoding="utf-8")
    with pytest.raises(GateBlockedError, match="blocked"):
        require_gate_status(status, gate="Q0", accepted=frozenset({"complete"}))
    status.write_text(json.dumps({"gate": "Q0", "status": "complete"}), encoding="utf-8")
    payload = require_gate_status(status, gate="Q0", accepted=frozenset({"complete"}))
    assert payload["status"] == "complete"


def test_worktree_git_commit_resolves_loose_common_reference(tmp_path):
    common = tmp_path / "main" / ".git"
    worktree = common / "worktrees" / "audit"
    worktree.mkdir(parents=True)
    (tmp_path / ".git").write_text(f"gitdir: {worktree}\n", encoding="utf-8")
    (worktree / "HEAD").write_text("ref: refs/heads/audit\n", encoding="utf-8")
    (worktree / "commondir").write_text("../..\n", encoding="utf-8")
    reference = common / "refs" / "heads" / "audit"
    reference.parent.mkdir(parents=True)
    expected = "a" * 40
    reference.write_text(expected + "\n", encoding="utf-8")
    assert _git_commit_from_metadata(tmp_path) == expected
