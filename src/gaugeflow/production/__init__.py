"""Production GaugeFlow-Piezo primitives defined by the revised manuscript.

This package is intentionally separate from the archived continuous-logit ODE
prototype.  Importing it never selects a legacy probability-path fallback.
"""

from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser
from .harmonic_gaugeflow import ConditionFreeGeometryQueryEncoder, HarmonicGaugeFlowConditioner
from .lattice_volume_shape import (
    LatticeVolumeShape,
    PointGroupMetricChart,
    SymmetryShapeBasis,
    project_lattice_state,
)
from .schedules import CosineNoiseSchedule
from .space_group_router import SpaceGroupCompatibilityRouter
from .state_projection import project_hybrid_reverse_state, project_translation_state
from .wrapped_coordinates import AdaptiveWrappedQuotient, ScalableWrappedQuotient

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CosineNoiseSchedule",
    "ConditionFreeGeometryQueryEncoder",
    "HarmonicGaugeFlowConditioner",
    "HybridCrystalDenoiser",
    "LatticeVolumeShape",
    "PointGroupMetricChart",
    "ScalableWrappedQuotient",
    "SpaceGroupCompatibilityRouter",
    "SymmetryShapeBasis",
    "project_lattice_state",
    "project_hybrid_reverse_state",
    "project_translation_state",
]
