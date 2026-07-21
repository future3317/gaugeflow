"""Production GaugeFlow-Piezo primitives defined by the revised manuscript.

This package is intentionally separate from the archived continuous-logit ODE
prototype.  Importing it never selects a legacy probability-path fallback.
"""

from .alex_p1_data import PackedAlexModelBatch, PackedAlexP1Dataset
from .assignment_data import (
    AssignmentCarrierExample,
    pack_assignment_carriers,
    prepare_assignment_carrier_example,
)
from .assignment_pretraining import (
    MaskedAssignmentCompilation,
    compile_masked_assignment_batch,
    complete_pair_indices,
    ddp_global_mean_loss,
    exact_periodic_pair_distances,
    rank_shard_of_global_batch,
    sample_rank_sharded_reveal_ranks,
)
from .assignment_scorer import (
    OrbitAwareAssignmentScorer,
    faithful_parent_action,
    parent_action_site_features,
    parent_carrier_graph_features,
)
from .assignment_training import (
    AssignmentCarrierBatch,
    OrderlessAssignmentObjective,
    OrderlessAssignmentTrainingModule,
    orderless_assignment_objective,
    sample_uniform_reveal_ranks,
)
from .autoregressive_assignment import (
    GeometryAwareRemainingCountScorer,
    RemainingCountAssignmentLaw,
    complete_pair_context_features,
    complete_pair_rbf,
)
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
from .cartesian_gauge_atlas import (
    CartesianSTFGeometryQueryEncoder,
    StratifiedCartesianGaugeAtlas,
)
from .categorical_mask import AbsorbingMaskDiffusion
from .categorical_uniform import UniformCategoricalDiffusion
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
from .composition_assignment import (
    AssignmentLogProbability,
    AssignmentSample,
    CountConstrainedAssignmentLaw,
    composition_counts_from_tokens,
    count_constrained_assignment,
    count_projected_assignment,
    occupation_block_composition_feasible,
    rounded_graph_composition,
)
from .composition_state import (
    IntegerPartitionCatalogue,
    SparseCompositionLogProbability,
    SparseCompositionSample,
    SparseCompositionState,
    StoichiometryFirstCompositionModel,
    fit_integer_partition_log_prior,
)
from .equivariant_denoiser import HybridCrystalDenoiser, LatticeDenoiserOutput
from .generation_law import (
    CarrierSelectionSample,
    CrystalGenerationState,
    FactorizedGenerationLogProbability,
    ParentDeltaNodeCountLaw,
    SupportedCarrierSelectionLaw,
)
from .hybrid_diffusion import LatticeLossOutput, LatticeNoisyBatch, TensorFreeHybridDiffusion
from .lattice_volume_shape import (
    LatticeVolumeShape,
    PointGroupMetricChart,
    SymmetryShapeBasis,
    project_lattice_state,
)
from .reverse_sampler import (
    ContinuousReverseInitialState,
    ContinuousReverseMode,
    ElementReverseDiagnostics,
    GeneratedElementBatch,
    GeneratedHybridBatch,
    GeneratedLatticeBatch,
    LatticeReverseDiagnostics,
    LatticeReverseInitialState,
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
from .split_contract import (
    EvaluationRole,
    MultiAxisSplitSchema,
    SplitAxis,
    SplitAxisContract,
)
from .state_projection import (
    project_hybrid_reverse_state,
    project_translation_state,
)
from .terminal_symmetry_audit import (
    DetectedPointGroup,
    TerminalSymmetryAudit,
    audit_terminal_symmetry,
    detect_cartesian_point_group,
)
from .training import (
    ExponentialMovingAverage,
    ProductionTrainer,
    ProductionTrainingConfig,
)
from .wrapped_coordinates import AdaptiveWrappedQuotient, ScalableWrappedQuotient

__all__ = [
    "AbsorbingMaskDiffusion",
    "OrbitAwareAssignmentScorer",
    "GeometryAwareRemainingCountScorer",
    "AssignmentCarrierBatch",
    "AssignmentCarrierExample",
    "MaskedAssignmentCompilation",
    "OrderlessAssignmentObjective",
    "OrderlessAssignmentTrainingModule",
    "orderless_assignment_objective",
    "pack_assignment_carriers",
    "prepare_assignment_carrier_example",
    "sample_uniform_reveal_ranks",
    "RemainingCountAssignmentLaw",
    "AssignmentLogProbability",
    "AssignmentSample",
    "CarrierSelectionSample",
    "UniformCategoricalDiffusion",
    "AdaptiveWrappedQuotient",
    "CartesianSTFGeometryQueryEncoder",
    "CompactCartesianKrylovCarrier",
    "CountConstrainedAssignmentLaw",
    "composition_counts_from_tokens",
    "complete_pair_context_features",
    "complete_pair_indices",
    "complete_pair_rbf",
    "compile_masked_assignment_batch",
    "ddp_global_mean_loss",
    "exact_periodic_pair_distances",
    "rank_shard_of_global_batch",
    "sample_rank_sharded_reveal_ranks",
    "count_constrained_assignment",
    "count_projected_assignment",
    "occupation_block_composition_feasible",
    "ContinuousReverseInitialState",
    "ContinuousReverseMode",
    "ChildReconstructor",
    "CosineNoiseSchedule",
    "CrystalGenerationState",
    "DistortionBlueprint",
    "DetectedPointGroup",
    "EmpiricalNodeCountPrior",
    "ExponentialMovingAverage",
    "EvaluationRole",
    "ElementReverseDiagnostics",
    "GeneratedElementBatch",
    "GeneratedHybridBatch",
    "GeneratedLatticeBatch",
    "HybridCrystalDenoiser",
    "LatticeDenoiserOutput",
    "LatticeLossOutput",
    "LatticeNoisyBatch",
    "LatticeReverseDiagnostics",
    "LatticeReverseInitialState",
    "HierarchicalSample",
    "IntegerPartitionCatalogue",
    "FactorizedGenerationLogProbability",
    "LatticeVolumeShape",
    "ModeCatalog",
    "ModeCatalogEntry",
    "ModeDiffusionState",
    "MultiAxisSplitSchema",
    "OPDBranch",
    "OccupationalPattern",
    "ParentBlueprint",
    "ParentBlueprintBatch",
    "ParentDeltaNodeCountLaw",
    "ParentGeometryCarrier",
    "PackedAlexP1Dataset",
    "PackedAlexModelBatch",
    "PointGroupMetricChart",
    "ProductionTrainer",
    "ProductionTrainingConfig",
    "SamplingFailure",
    "ScalableWrappedQuotient",
    "ReachableChildCompatibilityRouter",
    "ReachableChildPath",
    "rounded_graph_composition",
    "SparseCompositionLogProbability",
    "SparseCompositionSample",
    "SparseCompositionState",
    "SplitAxis",
    "SplitAxisContract",
    "StoichiometryFirstCompositionModel",
    "SupportedCarrierSelectionLaw",
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
    "fit_integer_partition_log_prior",
    "faithful_parent_action",
    "load_production_checkpoint",
    "read_production_checkpoint_metadata",
    "reverse_time_grid",
    "project_hybrid_reverse_state",
    "parent_action_site_features",
    "parent_carrier_graph_features",
    "project_lattice_state",
    "project_translation_state",
    "save_production_checkpoint",
]
