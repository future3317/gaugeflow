"""Production GaugeFlow-Piezo primitives defined by the revised manuscript.

This package is intentionally separate from the archived continuous-logit ODE
prototype.  Importing it never selects a legacy probability-path fallback.
"""

from .blueprint import EmpiricalNodeCountPrior, P1BlueprintBatch
from .cartesian_gauge_atlas import CartesianSTFGeometryQueryEncoder, StratifiedCartesianGaugeAtlas
from .categorical_mask import AbsorbingMaskDiffusion
from .checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
    save_production_checkpoint,
)
from .equivariant_denoiser import HybridCrystalDenoiser
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .lattice_volume_shape import (
    LatticeVolumeShape,
    PointGroupMetricChart,
    SymmetryShapeBasis,
    project_lattice_state,
)
from .reverse_sampler import GeneratedHybridBatch, SamplingFailure, TensorFreeReverseSampler
from .schedules import CosineNoiseSchedule
from .space_group_router import SpaceGroupCompatibilityRouter
from .state_projection import project_hybrid_reverse_state, project_translation_state
from .training import ExponentialMovingAverage, ProductionTrainer, ProductionTrainingConfig
from .wrapped_coordinates import AdaptiveWrappedQuotient, ScalableWrappedQuotient

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CartesianSTFGeometryQueryEncoder",
    "CosineNoiseSchedule",
    "EmpiricalNodeCountPrior",
    "ExponentialMovingAverage",
    "GeneratedHybridBatch",
    "HybridCrystalDenoiser",
    "LatticeVolumeShape",
    "P1BlueprintBatch",
    "PointGroupMetricChart",
    "ProductionTrainer",
    "ProductionTrainingConfig",
    "SamplingFailure",
    "ScalableWrappedQuotient",
    "SpaceGroupCompatibilityRouter",
    "StratifiedCartesianGaugeAtlas",
    "SymmetryShapeBasis",
    "TensorFreeHybridDiffusion",
    "TensorFreeReverseSampler",
    "load_production_checkpoint",
    "read_production_checkpoint_metadata",
    "project_hybrid_reverse_state",
    "project_lattice_state",
    "project_translation_state",
    "save_production_checkpoint",
]
