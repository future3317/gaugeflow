"""Production GaugeFlow-Piezo primitives defined by the revised manuscript.

This package is intentionally separate from the archived continuous-logit ODE
prototype.  Importing it never selects a legacy probability-path fallback.
"""

from .alex_p1_data import PackedAlexP1Dataset
from .blueprint import (
    DistortionBlueprint,
    EmpiricalNodeCountPrior,
    ModeCatalog,
    ModeCatalogEntry,
    ModeDiffusionState,
    OccupationalPattern,
    OPDBranch,
    ParentBlueprint,
    ParentBlueprintBatch,
    SelectedMode,
)
from .cartesian_coordinate_carrier import (
    CompactCartesianKrylovCarrier,
    StateAdaptiveCartesianCarrierMixer,
)
from .cartesian_gauge_atlas import CartesianSTFGeometryQueryEncoder, StratifiedCartesianGaugeAtlas
from .categorical_mask import AbsorbingMaskDiffusion
from .checkpointing import (
    load_production_checkpoint,
    read_production_checkpoint_metadata,
    save_production_checkpoint,
)
from .child_reconstruction import (
    ChildReconstructor,
    HierarchicalSample,
    ParentGeometryCarrier,
)
from .equivariant_denoiser import HybridCrystalDenoiser
from .hybrid_diffusion import TensorFreeHybridDiffusion
from .lattice_volume_shape import (
    LatticeVolumeShape,
    PointGroupMetricChart,
    SymmetryShapeBasis,
    project_lattice_state,
)
from .reverse_sampler import (
    ContinuousReverseInitialState,
    ContinuousReverseMode,
    GeneratedHybridBatch,
    SamplingFailure,
    TensorFreeReverseSampler,
    reverse_time_grid,
)
from .schedules import CosineNoiseSchedule
from .space_group_router import (
    ReachableChildCompatibilityRouter,
    ReachableChildPath,
    TerminalGroupCompatibilityRouter,
)
from .state_projection import project_hybrid_reverse_state, project_translation_state
from .terminal_symmetry_audit import (
    DetectedPointGroup,
    TerminalSymmetryAudit,
    audit_terminal_symmetry,
    detect_cartesian_point_group,
)
from .training import ExponentialMovingAverage, ProductionTrainer, ProductionTrainingConfig
from .wrapped_coordinates import AdaptiveWrappedQuotient, ScalableWrappedQuotient

__all__ = [
    "AbsorbingMaskDiffusion",
    "AdaptiveWrappedQuotient",
    "CartesianSTFGeometryQueryEncoder",
    "CompactCartesianKrylovCarrier",
    "ContinuousReverseInitialState",
    "ContinuousReverseMode",
    "ChildReconstructor",
    "CosineNoiseSchedule",
    "DistortionBlueprint",
    "DetectedPointGroup",
    "EmpiricalNodeCountPrior",
    "ExponentialMovingAverage",
    "GeneratedHybridBatch",
    "HybridCrystalDenoiser",
    "HierarchicalSample",
    "LatticeVolumeShape",
    "ModeCatalog",
    "ModeCatalogEntry",
    "ModeDiffusionState",
    "OPDBranch",
    "OccupationalPattern",
    "ParentBlueprint",
    "ParentBlueprintBatch",
    "ParentGeometryCarrier",
    "PackedAlexP1Dataset",
    "PointGroupMetricChart",
    "ProductionTrainer",
    "ProductionTrainingConfig",
    "SamplingFailure",
    "ScalableWrappedQuotient",
    "ReachableChildCompatibilityRouter",
    "ReachableChildPath",
    "SelectedMode",
    "StratifiedCartesianGaugeAtlas",
    "StateAdaptiveCartesianCarrierMixer",
    "SymmetryShapeBasis",
    "TensorFreeHybridDiffusion",
    "TensorFreeReverseSampler",
    "TerminalGroupCompatibilityRouter",
    "TerminalSymmetryAudit",
    "audit_terminal_symmetry",
    "detect_cartesian_point_group",
    "load_production_checkpoint",
    "read_production_checkpoint_metadata",
    "reverse_time_grid",
    "project_hybrid_reverse_state",
    "project_lattice_state",
    "project_translation_state",
    "save_production_checkpoint",
]
