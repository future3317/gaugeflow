"""GaugeFlow revised-production package.

The retired continuous-logit flow prototype is available only from the
``archive/pre-production-cleanup-20260716`` Git tag.  The active package never
selects it as a compatibility fallback.
"""

from .production import (
    AbsorbingMaskDiffusion,
    AdaptiveWrappedQuotient,
    CartesianSTFGeometryQueryEncoder,
    CosineNoiseSchedule,
    HybridCrystalDenoiser,
    LatticeVolumeShape,
    PointGroupMetricChart,
    ScalableWrappedQuotient,
    SpaceGroupCompatibilityRouter,
    StratifiedCartesianGaugeAtlas,
    SymmetryShapeBasis,
)

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CartesianSTFGeometryQueryEncoder",
    "CosineNoiseSchedule",
    "HybridCrystalDenoiser",
    "LatticeVolumeShape",
    "PointGroupMetricChart",
    "ScalableWrappedQuotient",
    "SpaceGroupCompatibilityRouter",
    "StratifiedCartesianGaugeAtlas",
    "SymmetryShapeBasis",
]
