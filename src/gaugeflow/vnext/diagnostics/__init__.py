"""Reusable diagnostics for the strictly gated vNext experiments."""

from .conditional_variance import ConditionalVarianceEstimate, knn_conditional_variance
from .reduced_jacobian import (
    analytic_endpoint_jacobians,
    reduced_vector_jacobian,
    variational_flow_jacobian,
)
from .representation_collision import CollisionAudit, audit_representation_collisions
from .solver_convergence import SolverResult, adaptive_rk4, euler_integrate, rk4_integrate

__all__ = [
    "CollisionAudit",
    "ConditionalVarianceEstimate",
    "SolverResult",
    "adaptive_rk4",
    "analytic_endpoint_jacobians",
    "audit_representation_collisions",
    "euler_integrate",
    "knn_conditional_variance",
    "reduced_vector_jacobian",
    "rk4_integrate",
    "variational_flow_jacobian",
]
