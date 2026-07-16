"""Production GaugeFlow-Piezo primitives defined by the revised manuscript.

This package is intentionally separate from the archived continuous-logit ODE
prototype.  Importing it never selects a legacy probability-path fallback.
"""

from .categorical_mask import AbsorbingMaskDiffusion
from .equivariant_denoiser import HybridCrystalDenoiser
from .harmonic_gaugeflow import HarmonicGaugeFlowConditioner
from .lattice_volume_shape import LatticeVolumeShape, SymmetryShapeBasis
from .schedules import CosineNoiseSchedule
from .space_group_router import SpaceGroupCompatibilityRouter
from .wrapped_coordinates import AdaptiveWrappedQuotient

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CosineNoiseSchedule",
    "HarmonicGaugeFlowConditioner",
    "HybridCrystalDenoiser",
    "LatticeVolumeShape",
    "SpaceGroupCompatibilityRouter",
    "SymmetryShapeBasis",
]
