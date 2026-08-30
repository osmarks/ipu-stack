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
mod conversion;
mod cost;
pub mod exchange;
pub mod graph;
mod host;
pub mod kernel;
mod layout;
pub mod low;
pub mod memory;
mod metrics;
pub mod mid;
mod operator;
pub mod place;
mod schedule;
pub mod storage;
pub mod tile;
pub use config::{
    AttentionStrategy, ConversionStreamingPolicy, OperatorClass, PipelineConfig,
    PlannerSearchDomain, ProfilingConfig,
};
pub use conversion::{
    ConversionGeometryError, ConversionMapping, ConversionStrategy, CopyGeometry, DeferredTransform,
};
pub(crate) use exchange::lower_exchanges;
pub use exchange::{
    ExchangeActivity, ExchangeLoweringError, PhysicalExchangePhase, inactive_exchange_program,
};
pub use graph::{
    AddOptions, AttentionOptions, AttentionScale, BroadcastMode, ComputeGraph, GemmOptions,
    GraphError, GraphInput, GraphInputKind, GraphResult, Operation, OperationId, OperationKind,
    Region, RegionBuilder, Repeat, RepeatArguments, SplitHeadsOptions, TensorShape, ValueId,
    ValueSequence, ValueSequenceId,
};
pub use kernel::{
    KernelAbi, KernelAbiError, KernelAvailability, KernelBuildPlan, KernelCompilation,
    KernelMaterializationError, KernelOptimization, KernelSymbols, PlannedKernelCall,
    ScalarArgument, TileKernelSpec, materialize_kernel_run, tile_kernel_abi, validate_kernel_run,
};
pub use layout::{
    AMP_COLUMN_MICRO, AMP_INNER_BLOCK, AMP_OUTPUT_COLUMN_BLOCK, AxisTiling, BlockedOrder, Layout,
    LayoutError, MemoryClass, NativeKernelOrder, Padding, ShardExtent, StorageOrder, TensorAxis,
    TensorFormat, TensorRegion, TensorTiling, TensorType,
};
pub use low::{
    ExchangeOrder, ExchangePhase, ExchangePhaseId, KernelOperand, KernelRequirements, KernelRun,
    KernelRunId, KernelRunMetadata, LocalCopy, LocalCopyId, LocalCopyPattern, LogicalExchange,
    LowInput, LowLoweringError, LowLoweringResult, LowProgram, LowShard, LowShardId, LowValue,
    RepeatCarried, RepeatInvariant, RepeatIterated, RepeatRun, RepeatRunId, ShardDefinition,
    ShardView, TileWork, TileWorkList, TileWorkRef, WorkProvenance, WorkReason, lower_to_tiles,
};
pub use metrics::{
    CostEstimate, ExchangeFootprint, MemoryEstimate, MemoryPeaks, MemoryUsage, OperationMetrics,
    PlanMetrics, RegionMetrics,
};
pub use mid::{
    CostModel, Ipu21CostModel, LoweringError, LoweringResult, MidGraph, MidInput, MidOperation,
    MidOperationKind, MidRegion, MidRepeat, MidValue, MidValueId, lower,
};
pub use operator::{
    AccumulationPrecision, AllocationRequirements, AttentionBlocking, AttentionPadding,
    AttentionPlan, BlockedGemmPlan, GemmBlockShape, GemmGeometry, GemmGrid, GemmKernelFamily,
    GemmKernelMode, GemmOrientation, GemmPlanConstraint, GemmResultGrid, GemmWeightLoad, GridOrder,
    LocalOperandStaging, MemoryElementRequirement, MemoryOperand, MemorySpaceRequirements,
    MidOperator, OperandMaterialization, OperandRequirement, OperatorPlan, OperatorPlanError,
    OperatorRequirements, OutputAliasing, Precision, ReductionStaging,
};
pub use place::{Placement, PlacementError, place};
pub use schedule::{
    KernelMap, OperatorSchedule, ScheduleAccess, ScheduleDomain, ScheduleStep, ScheduleValue,
};
pub use storage::{
    ByteSpan, StorageError, StorageResult, amp_matrix_coordinates, block_major_matrix_coordinates,
    logical_view_byte_spans, shard_storage_bytes, view_byte_spans,
};
pub use tile::{TileLoweringError, TileProgramLowering, compact_exchange_row_address};
