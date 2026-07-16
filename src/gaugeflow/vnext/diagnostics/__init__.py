"""Reusable diagnostics for the strictly gated vNext experiments."""

from .conditional_variance import (
    ExactEquivalenceRisk,
    LocalTargetDispersion,
    exact_equivalence_risk,
    knn_local_target_dispersion,
)
from .reduced_jacobian import (
    analytic_endpoint_jacobians,
    reduced_vector_jacobian,
    variational_flow_jacobian,
)
from .representation_collision import CollisionAudit, CollisionWitness, audit_representation_collisions
from .solver_convergence import SolverResult, adaptive_rk4, euler_integrate, rk4_integrate

__all__ = [
    "CollisionAudit",
    "CollisionWitness",
    "ExactEquivalenceRisk",
    "LocalTargetDispersion",
    "SolverResult",
    "adaptive_rk4",
    "analytic_endpoint_jacobians",
    "audit_representation_collisions",
    "euler_integrate",
    "exact_equivalence_risk",
    "knn_local_target_dispersion",
    "reduced_vector_jacobian",
    "rk4_integrate",
    "variational_flow_jacobian",
]
