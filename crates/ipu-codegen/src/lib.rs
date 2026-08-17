//! Whole-device graph compilation and IPU package construction.
//!
//! [`build_package`] is the primary entry point: it plans a [`ComputeGraph`],
//! lowers it to tile work, compiles and links the required kernels, assigns
//! memory, emits tile programs, and returns a loadable [`CompiledPackage`].

mod package;
pub use package::{
    CompiledPackage, CompiledTensor, CompiledTensorShard, DiagnosticCheckpoint, PackageBuildError,
    PackageBuildResult, PackageConfig, TileProgramData, build_diagnostic_package, build_package,
    build_tile_program_package,
};

mod config;
mod cost;
mod estimate;
pub mod exchange;
pub mod graph;
mod host;
mod ir;
pub mod kernel;
mod layout;
pub mod low;
pub mod memory;
mod metrics;
pub mod mid;
mod operator;
pub mod place;
pub mod storage;
pub mod tile;
pub use config::{
    AttentionStrategy, ConversionStreamingPolicy, HardwareMemoryConstraints, HardwareTarget,
    OperatorClass, PipelineConfig, PlannerSearchDomain, ProfilingConfig,
};
pub use exchange::{
    EXCHANGE_SCHEDULE_SNAPSHOT_VERSION, ExchangeActivity, ExchangeActivityDiagnostic,
    ExchangeActivityKind, ExchangeLoweringError, ExchangeLoweringOptions, ExchangeScheduleProblem,
    ExchangeScheduleRun, ExchangeScheduleSnapshot, ExchangeTileDiagnostic, LoweredExchanges,
    PhysicalExchangePhase, PhysicalTransfer, TransferEndpoint, TransferWidth,
    diagnose_exchange_tile, inactive_exchange_program, lower_exchanges, schedule_exchange_problem,
    validate_exchange_schedule,
};
pub use graph::{
    AddOptions, AttentionOptions, AttentionScale, BroadcastMode, ComputeGraph, GemmOptions,
    GraphError, GraphInput, GraphInputKind, GraphResult, Operation, OperationId, OperationKind,
    Region, RegionBuilder, Repeat, RepeatArguments, SplitHeadsOptions, TensorShape, ValueId,
    ValueSequence, ValueSequenceId,
};
pub use ir::{
    MidGraph, MidInput, MidOperation, MidOperationKind, MidRegion, MidRepeat, MidValue, MidValueId,
};
pub use kernel::{
    KernelAbi, KernelAbiError, KernelAvailability, KernelBuildPlan, KernelCompilation,
    KernelMaterializationError, KernelSymbols, PlannedKernelCall, ScalarArgument,
    materialize_kernel_run, tile_kernel_abi, validate_kernel_run,
};
pub use layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_OUTPUT_COLUMN_BLOCK, AmpOrder, AxisTiling,
    BlockMajorOrder, ElementOrder, Layout, LayoutError, MemoryClass, Padding, ShardExtent,
    TensorAxis, TensorFormat, TensorRegion, TensorTiling, TensorType,
};
pub use low::{
    ExchangeOrder, ExchangePhase, ExchangePhaseId, KernelOperand, KernelRequirements, KernelRun,
    KernelRunId, KernelRunMetadata, LocalCopy, LocalCopyId, LocalCopyPattern, LogicalExchange,
    LowInput, LowLoweringError, LowLoweringResult, LowProgram, LowShard, LowShardId, LowValue,
    RepeatCarried, RepeatInvariant, RepeatIterated, RepeatRun, RepeatRunId, ShardDefinition,
    ShardView, TileWork, TileWorkList, TileWorkRef, WorkProvenance, WorkReason, lower_to_tiles,
};
pub use memory::{
    IPU21_DATA_BASE, IPU21_INTERLEAVED_REGION_BYTES, IPU21_PLANNED_DATA_BYTES,
    IPU21_STANDARD_FIXED_BYTES,
};
pub use metrics::{
    MemoryEstimate, MemoryPeaks, MemoryUsage, OperationMetrics, PlanMetrics, RegionMetrics,
};
pub use mid::{
    CostEstimate, CostModel, IPU21_TARGET_COSTS, Ipu21CostModel, Ipu21TargetCosts, LoweringError,
    LoweringResult, lower,
};
pub use operator::{
    AccumulationPrecision, AllocationRequirements, AttentionBlocking, AttentionKernelFamily,
    AttentionPadding, AttentionPlan, BlockedGemmPlan, ConversionPlan, ConversionStrategy,
    GemmBlockShape, GemmDistribution, GemmGeometry, GemmGrid, GemmKernelFamily, GemmKernelMode,
    GemmOrientation, GemmPlanConstraint, GemmResultGrid, GemmWeightLoad, GridOrder,
    LocalOperandStaging, MemoryElementRequirement, MemoryOperand, MemorySpaceRequirements,
    MidOperator, OperandMaterialization, OperandRequirement, OperatorDispatch, OperatorPlan,
    OperatorPlanError, OperatorRequirements, OutputAliasing, ParallelReductionPlan,
    PointwiseInputMapping, Precision, ReductionStaging, TileKernelSpec,
};
pub use place::{Placement, PlacementError, place};
pub use storage::{
    ByteSpan, StorageError, StorageResult, amp_matrix_coordinates, block_major_matrix_coordinates,
    logical_view_byte_spans, shard_storage_bytes, view_byte_spans,
};
pub use tile::{TileLoweringError, TileProgramLowering, compact_exchange_row_address};
