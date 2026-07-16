"""GaugeFlow revised-production package.

The retired continuous-logit flow prototype is available only from the
``archive/pre-production-cleanup-20260716`` Git tag.  The active package never
selects it as a compatibility fallback.
"""

from .production import (
    AbsorbingMaskDiffusion,
    AdaptiveWrappedQuotient,
    CartesianSTFGeometryQueryEncoder,
    ChildReconstructor,
    CosineNoiseSchedule,
    DistortionBlueprint,
    HybridCrystalDenoiser,
    LatticeVolumeShape,
    ModeCatalog,
    ModeDiffusionState,
    ParentBlueprint,
    PointGroupMetricChart,
    ReachableChildCompatibilityRouter,
    ScalableWrappedQuotient,
    StratifiedCartesianGaugeAtlas,
    SymmetryShapeBasis,
    TerminalGroupCompatibilityRouter,
)

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CartesianSTFGeometryQueryEncoder",
    "ChildReconstructor",
    "CosineNoiseSchedule",
    "DistortionBlueprint",
    "HybridCrystalDenoiser",
    "LatticeVolumeShape",
    "ModeCatalog",
    "ModeDiffusionState",
    "ParentBlueprint",
    "PointGroupMetricChart",
    "ScalableWrappedQuotient",
    "ReachableChildCompatibilityRouter",
    "StratifiedCartesianGaugeAtlas",
    "SymmetryShapeBasis",
    "TerminalGroupCompatibilityRouter",
]
