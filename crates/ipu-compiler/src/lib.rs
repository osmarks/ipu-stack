use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

use ipu_exchange::{
    MulticastPlan, PlanProgramBuilder, PlanRow, RETURN_M10_INSTRUCTION, SANS_INACTIVE_INSTRUCTION,
    SYNC_ANS_INSTRUCTION, Topology, finalize_point_receiver, patch_receiver_address,
    patch_sender_address,
};

mod attention;
mod mlp;
mod placement;
mod rowwise;
pub use attention::{
    AttentionKeyValuePlacement, AttentionTaskPlacement, FlashAttentionConfig, FlashAttentionPlan,
    append_flash_attention_from_a16_qkv, append_flash_attention_from_a16_qkv_in_arenas,
    append_flash_attention_to_a16_row_shards, append_flash_attention_to_a16_row_shards_in_arenas,
    plan_flash_attention,
};
pub use mlp::{BlockedMlpConfig, BlockedMlpPlan, plan_blocked_mlp};
use placement::{WindowRequirement, partition_address_window};
pub use rowwise::{
    AffineLayerNormConfig, AffineLayerNormPlan, AppendAffineLayerNormConfig,
    AppendedAffineLayerNorm, RowShardPlacement, RowShardTransitionConfig,
    append_a16_to_a16_row_shards_reblocked_in_arenas,
    append_add_affine_layer_norm_f16_with_memory_policy, append_add_f16_row_shards_in_place,
    append_affine_layer_norm_f16, append_affine_layer_norm_f16_in_arenas,
    append_affine_layer_norm_f16_with_memory_policy, append_c16_to_a16_blocks_gelu_f16,
    append_c16_to_a16_blocks_gelu_f16_in_arenas, append_c16_to_a16_row_shards,
    append_c16_to_a16_row_shards_gelu_f16, append_c16_to_a16_row_shards_reblocked_in_arenas,
    choose_row_shard_rows_for_copies_in_arenas, end_tensor_lifetimes, make_tensors_resident,
    make_tensors_resident_since, plan_affine_layer_norm_f16,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_TILE_COUNT: u16 = 64;
// The AMP worker pipeline uses a repeat count of `worker_rows - 1`. Keep at
// least two rows on each of the six workers so that repeat never sees zero.
const GEMM_MINIMUM_ROW_SHARD: u16 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorKind {
    Input,
    Weight,
    Intermediate,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub kind: TensorKind,
    pub producer: Option<OpId>,
}

impl Tensor {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    MatMul,
    Add,
    Mul,
    Reshape { shape: Vec<usize> },
    Transpose { permutation: Vec<usize> },
    LayerNorm { epsilon: f32 },
    Softmax { axis: usize },
    Gelu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Op {
    pub id: OpId,
    pub name: String,
    pub kind: OpKind,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("invalid graph: {0}")]
    Graph(String),
    #[error("SRAM allocation failed: {0}")]
    Memory(String),
    #[error("exchange lowering failed: {0}")]
    Exchange(#[from] ipu_exchange::ExchangeError),
}

impl Graph {
    pub fn input(&mut self, name: &str, shape: &[usize]) -> TensorId {
        self.tensor(name, shape, TensorKind::Input, None)
    }

    pub fn weight(&mut self, name: &str, shape: &[usize]) -> TensorId {
        self.tensor(name, shape, TensorKind::Weight, None)
    }

    pub fn mark_output(&mut self, tensor: TensorId) {
        self.tensors[tensor.0].kind = TensorKind::Output;
        self.outputs.push(tensor);
    }

    pub fn matmul(
        &mut self,
        name: &str,
        left: TensorId,
        right: TensorId,
    ) -> Result<TensorId, CompileError> {
        let a = &self.tensors[left.0].shape;
        let b = &self.tensors[right.0].shape;
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(CompileError::Graph(format!(
                "{name}: incompatible matmul {a:?} x {b:?}"
            )));
        }
        self.op(name, OpKind::MatMul, &[left, right], &[a[0], b[1]])
    }

    pub fn add(
        &mut self,
        name: &str,
        left: TensorId,
        right: TensorId,
    ) -> Result<TensorId, CompileError> {
        self.binary(name, OpKind::Add, left, right)
    }

    pub fn mul(
        &mut self,
        name: &str,
        left: TensorId,
        right: TensorId,
    ) -> Result<TensorId, CompileError> {
        self.binary(name, OpKind::Mul, left, right)
    }

    pub fn reshape(
        &mut self,
        name: &str,
        input: TensorId,
        shape: &[usize],
    ) -> Result<TensorId, CompileError> {
        if self.tensors[input.0].elements() != shape.iter().product() {
            return Err(CompileError::Graph(format!(
                "{name}: reshape changes element count"
            )));
        }
        self.op(
            name,
            OpKind::Reshape {
                shape: shape.to_vec(),
            },
            &[input],
            shape,
        )
    }

    pub fn transpose(
        &mut self,
        name: &str,
        input: TensorId,
        permutation: &[usize],
    ) -> Result<TensorId, CompileError> {
        let source = &self.tensors[input.0].shape;
        let axes: HashSet<_> = permutation.iter().copied().collect();
        if permutation.len() != source.len()
            || axes.len() != source.len()
            || permutation.iter().any(|axis| *axis >= source.len())
        {
            return Err(CompileError::Graph(format!(
                "{name}: invalid transpose permutation"
            )));
        }
        let shape: Vec<_> = permutation.iter().map(|axis| source[*axis]).collect();
        self.op(
            name,
            OpKind::Transpose {
                permutation: permutation.to_vec(),
            },
            &[input],
            &shape,
        )
    }

    pub fn layer_norm(
        &mut self,
        name: &str,
        input: TensorId,
        epsilon: f32,
    ) -> Result<TensorId, CompileError> {
        let shape = self.tensors[input.0].shape.clone();
        self.op(name, OpKind::LayerNorm { epsilon }, &[input], &shape)
    }

    pub fn softmax(
        &mut self,
        name: &str,
        input: TensorId,
        axis: usize,
    ) -> Result<TensorId, CompileError> {
        let shape = self.tensors[input.0].shape.clone();
        if axis >= shape.len() {
            return Err(CompileError::Graph(format!(
                "{name}: softmax axis out of range"
            )));
        }
        self.op(name, OpKind::Softmax { axis }, &[input], &shape)
    }

    pub fn gelu(&mut self, name: &str, input: TensorId) -> Result<TensorId, CompileError> {
        let shape = self.tensors[input.0].shape.clone();
        self.op(name, OpKind::Gelu, &[input], &shape)
    }

    pub fn validate(&self) -> Result<(), CompileError> {
        for (index, tensor) in self.tensors.iter().enumerate() {
            if tensor.id.0 != index || tensor.shape.is_empty() || tensor.shape.contains(&0) {
                return Err(CompileError::Graph(format!(
                    "invalid tensor {}",
                    tensor.name
                )));
            }
        }
        for (index, op) in self.ops.iter().enumerate() {
            if op.id.0 != index
                || op.inputs.iter().any(|input| input.0 >= self.tensors.len())
                || op.output.0 >= self.tensors.len()
                || self.tensors[op.output.0].producer != Some(op.id)
            {
                return Err(CompileError::Graph(format!("invalid op {}", op.name)));
            }
        }
        if self.outputs.is_empty() {
            return Err(CompileError::Graph("graph has no outputs".into()));
        }
        Ok(())
    }

    fn tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        kind: TensorKind,
        producer: Option<OpId>,
    ) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(Tensor {
            id,
            name: name.into(),
            dtype: DType::F32,
            shape: shape.to_vec(),
            kind,
            producer,
        });
        id
    }

    fn op(
        &mut self,
        name: &str,
        kind: OpKind,
        inputs: &[TensorId],
        shape: &[usize],
    ) -> Result<TensorId, CompileError> {
        if inputs.iter().any(|tensor| tensor.0 >= self.tensors.len()) {
            return Err(CompileError::Graph(format!("{name}: unknown input")));
        }
        let id = OpId(self.ops.len());
        let output = self.tensor(name, shape, TensorKind::Intermediate, Some(id));
        self.ops.push(Op {
            id,
            name: name.into(),
            kind,
            inputs: inputs.to_vec(),
            output,
        });
        Ok(output)
    }

    fn binary(
        &mut self,
        name: &str,
        kind: OpKind,
        left: TensorId,
        right: TensorId,
    ) -> Result<TensorId, CompileError> {
        let left_shape = self.tensors[left.0].shape.clone();
        if left_shape != self.tensors[right.0].shape {
            return Err(CompileError::Graph(format!("{name}: binary shapes differ")));
        }
        self.op(name, kind, &[left, right], &left_shape)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sharding {
    Replicated,
    Rows,
    Columns,
    Heads,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub tensor: TensorId,
    pub tiles: Vec<u16>,
    pub sharding: Sharding,
    pub alignment: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpecializationKey {
    pub operation: Cow<'static, str>,
    pub shape: Vec<usize>,
    pub worker_count: u8,
    pub role: Cow<'static, str>,
    pub alignment: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCommand {
    pub tile: u16,
    pub output: TensorId,
    pub inputs: Vec<TensorId>,
    pub arguments: Vec<u32>,
    pub specialization: SpecializationKey,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transfer {
    pub source_tile: u16,
    pub destination_tile: u16,
    pub tensor: TensorId,
    pub bytes: u32,
    /// Resolved destination staging address. Legacy graph compilation may
    /// leave this unset and provide an `ExchangeStaging` allocation instead.
    #[serde(default)]
    pub staging_address: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Exchange {
        transfers: Vec<Transfer>,
    },
    Compute {
        op: OpId,
        commands: Vec<Arc<KernelCommand>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatedRegion {
    pub name: String,
    pub phase_instances: Vec<Range<usize>>,
    shape: Vec<RepeatedPhaseShape>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum RepeatedPhaseShape {
    Exchange,
    Compute(Vec<RepeatedTileComputeShape>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RepeatedTileComputeShape {
    tile: u16,
    commands: Vec<RepeatedCommandShape>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RepeatedCommandShape {
    phase_tile_command_index: usize,
    operation: String,
    input_count: usize,
    argument_count: usize,
}

impl RepeatedRegion {
    pub fn new(
        name: impl Into<String>,
        schedule: &Schedule,
        phases: Range<usize>,
    ) -> Result<Self, CompileError> {
        let shape = repeated_region_shape(schedule, phases.clone())?;
        Ok(Self {
            name: name.into(),
            phase_instances: vec![phases],
            shape,
        })
    }

    pub fn push_instance(
        &mut self,
        schedule: &Schedule,
        phases: Range<usize>,
    ) -> Result<(), CompileError> {
        let shape = repeated_region_shape(schedule, phases.clone())?;
        if !repeated_shapes_compatible(&self.shape, &shape) {
            return Err(CompileError::Graph(format!(
                "instance {} of repeated region {} has a different phase or command structure",
                self.phase_instances.len(),
                self.name
            )));
        }
        if self
            .phase_instances
            .last()
            .is_some_and(|previous| previous.end > phases.start)
        {
            return Err(CompileError::Graph(format!(
                "instances of repeated region {} overlap or are out of order",
                self.name
            )));
        }
        merge_repeated_shapes(&mut self.shape, shape);
        self.phase_instances.push(phases);
        Ok(())
    }

    pub fn is_compatible(&self, schedule: &Schedule, phases: Range<usize>) -> bool {
        repeated_region_shape(schedule, phases)
            .is_ok_and(|shape| repeated_shapes_compatible(&self.shape, &shape))
    }
}

fn repeated_region_shape(
    schedule: &Schedule,
    phases: Range<usize>,
) -> Result<Vec<RepeatedPhaseShape>, CompileError> {
    let selected = schedule.phases.get(phases.clone()).ok_or_else(|| {
        CompileError::Graph(format!(
            "repeated region phase range {}..{} exceeds {} phases",
            phases.start,
            phases.end,
            schedule.phases.len()
        ))
    })?;
    if selected.is_empty() {
        return Err(CompileError::Graph(
            "repeated region must contain at least one phase".into(),
        ));
    }
    selected
        .iter()
        .map(|phase| match phase {
            Phase::Exchange { .. } => Ok(RepeatedPhaseShape::Exchange),
            Phase::Compute { commands, .. } => {
                let mut by_tile = BTreeMap::<u16, Vec<RepeatedCommandShape>>::new();
                for command in commands {
                    let tile_commands = by_tile.entry(command.tile).or_default();
                    tile_commands.push(RepeatedCommandShape {
                        phase_tile_command_index: tile_commands.len(),
                        operation: command.specialization.operation.to_string(),
                        input_count: command.inputs.len(),
                        argument_count: command.arguments.len(),
                    });
                }
                Ok(RepeatedPhaseShape::Compute(
                    by_tile
                        .into_iter()
                        .map(|(tile, commands)| RepeatedTileComputeShape { tile, commands })
                        .collect(),
                ))
            }
        })
        .collect()
}

fn repeated_shapes_compatible(left: &[RepeatedPhaseShape], right: &[RepeatedPhaseShape]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| match (left, right) {
                (RepeatedPhaseShape::Exchange, RepeatedPhaseShape::Exchange) => true,
                (RepeatedPhaseShape::Compute(_), RepeatedPhaseShape::Compute(_)) => true,
                _ => false,
            })
}

fn merge_repeated_command_shapes(
    left: &[RepeatedCommandShape],
    right: &[RepeatedCommandShape],
) -> Vec<RepeatedCommandShape> {
    let columns = right.len() + 1;
    let mut common = vec![0usize; (left.len() + 1) * columns];
    for left_index in (0..left.len()).rev() {
        for right_index in (0..right.len()).rev() {
            common[left_index * columns + right_index] = if left[left_index] == right[right_index] {
                1 + common[(left_index + 1) * columns + right_index + 1]
            } else {
                common[(left_index + 1) * columns + right_index]
                    .max(common[left_index * columns + right_index + 1])
            };
        }
    }
    let mut merged = Vec::with_capacity(left.len() + right.len());
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        if left[left_index] == right[right_index] {
            merged.push(left[left_index].clone());
            left_index += 1;
            right_index += 1;
        } else if common[(left_index + 1) * columns + right_index]
            >= common[left_index * columns + right_index + 1]
        {
            merged.push(left[left_index].clone());
            left_index += 1;
        } else {
            merged.push(right[right_index].clone());
            right_index += 1;
        }
    }
    merged.extend_from_slice(&left[left_index..]);
    merged.extend_from_slice(&right[right_index..]);
    merged
}

fn merge_repeated_shapes(target: &mut [RepeatedPhaseShape], instance: Vec<RepeatedPhaseShape>) {
    for (target, instance) in target.iter_mut().zip(instance) {
        let (RepeatedPhaseShape::Compute(target), RepeatedPhaseShape::Compute(instance)) =
            (target, instance)
        else {
            continue;
        };
        for instance_tile in instance {
            match target.binary_search_by_key(&instance_tile.tile, |tile| tile.tile) {
                Ok(index) => {
                    target[index].commands = merge_repeated_command_shapes(
                        &target[index].commands,
                        &instance_tile.commands,
                    );
                }
                Err(index) => target.insert(index, instance_tile),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub tensor: TensorId,
    pub tile: u16,
    pub address: u32,
    pub size: u32,
    pub live_from: usize,
    pub live_until: usize,
    pub kind: AllocationKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationKind {
    Home,
    HomeAlias { source: TensorId },
    ExchangeStaging { phase: usize },
}

impl AllocationKind {
    fn has_home_address(&self) -> bool {
        matches!(self, Self::Home | Self::HomeAlias { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPlacement {
    Low,
    High,
}

impl Default for MemoryPlacement {
    fn default() -> Self {
        Self::Low
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryArena {
    /// Inclusive first address available for ordinary tensors.
    pub base: u32,
    /// Exclusive end address. Allocations never span arena boundaries.
    pub limit: u32,
    /// Direction in which objects are packed within this arena.
    #[serde(default)]
    pub placement: MemoryPlacement,
}

impl MemoryArena {
    pub const fn low(base: u32, limit: u32) -> Self {
        Self {
            base,
            limit,
            placement: MemoryPlacement::Low,
        }
    }

    pub const fn high(base: u32, limit: u32) -> Self {
        Self {
            base,
            limit,
            placement: MemoryPlacement::High,
        }
    }

    pub const fn with_placement(self, placement: MemoryPlacement) -> Self {
        Self { placement, ..self }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPolicy {
    /// Ordered arenas for long-lived tensors. Earlier arenas are preferred.
    pub resident: Vec<MemoryArena>,
    /// Ordered arenas for short-lived activations and ordinary kernel scratch.
    pub transient: Vec<MemoryArena>,
    /// Controls the allocation window used to balance resident tensors across
    /// tiles, or retains the child planner's tile assignment unchanged.
    #[serde(default)]
    pub resident_tile_assignment: ResidentTileAssignment,
    #[serde(skip)]
    allocation_occupancy: AllocationOccupancyCache,
}

#[derive(Clone, Debug, Default)]
struct AllocationOccupancyCache(Arc<Mutex<Option<CachedAllocationOccupancy>>>);

impl PartialEq for AllocationOccupancyCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for AllocationOccupancyCache {}

#[derive(Debug)]
struct CachedAllocationOccupancy {
    schedule: usize,
    allocation_count: usize,
    tile_count: u16,
    base: u32,
    limit: u32,
    occupied: Vec<Vec<(u32, u32)>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidentTileAssignment {
    #[default]
    Balanced,
    Fixed,
}

/// Allocatable IPU21 SRAM regions, excluding executable and exchange storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ipu21MemoryRegion {
    OrdinaryLow,
    Interleaved,
    OrdinaryHigh,
}

impl Ipu21MemoryRegion {
    pub fn arena(
        self,
        ordinary_low_base: u32,
        data_limit: u32,
        placement: MemoryPlacement,
    ) -> MemoryArena {
        match self {
            Self::OrdinaryLow => MemoryArena {
                base: ordinary_low_base,
                limit: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                placement: MemoryPlacement::High,
            },
            Self::Interleaved => MemoryArena {
                base: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
                limit: ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
                placement,
            },
            Self::OrdinaryHigh => MemoryArena {
                base: ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT,
                limit: data_limit,
                placement,
            },
        }
    }
}

impl MemoryPolicy {
    pub fn contiguous(base: u32, limit: u32) -> Self {
        let resident = MemoryArena {
            base,
            limit,
            placement: MemoryPlacement::High,
        };
        let transient = MemoryArena {
            base,
            limit,
            placement: MemoryPlacement::Low,
        };
        Self {
            resident: vec![resident],
            transient: vec![transient],
            resident_tile_assignment: ResidentTileAssignment::Balanced,
            allocation_occupancy: AllocationOccupancyCache::default(),
        }
    }

    /// Builds a policy from the physical IPU21 SRAM regions.
    ///
    /// `ordinary_low_base` is the first address left free by runtime code and
    /// fixed runtime data. `data_limit` is the end of tile SRAM. Ordering is a
    /// placement preference; allocations may spill to later regions.
    pub fn ipu21(
        ordinary_low_base: u32,
        data_limit: u32,
        resident_order: &[Ipu21MemoryRegion],
        transient_order: &[Ipu21MemoryRegion],
    ) -> Result<Self, CompileError> {
        if ordinary_low_base >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
            || data_limit <= ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
            || data_limit > ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE
            || !unique_ipu21_regions(resident_order)
            || !unique_ipu21_regions(transient_order)
        {
            return Err(CompileError::Memory(
                "invalid allocatable IPU21 SRAM bounds".into(),
            ));
        }
        let expand = |order: &[Ipu21MemoryRegion], placement| {
            order
                .iter()
                .copied()
                .map(|region| region.arena(ordinary_low_base, data_limit, placement))
                .collect::<Vec<_>>()
        };
        let policy = Self {
            resident: expand(resident_order, MemoryPlacement::High),
            transient: expand(transient_order, MemoryPlacement::Low),
            resident_tile_assignment: ResidentTileAssignment::Balanced,
            allocation_occupancy: AllocationOccupancyCache::default(),
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), CompileError> {
        if self.resident.is_empty()
            || self.transient.is_empty()
            || self
                .resident
                .iter()
                .chain(&self.transient)
                .any(|arena| arena.base >= arena.limit)
        {
            return Err(CompileError::Memory("invalid tile-memory policy".into()));
        }
        Ok(())
    }

    fn occupied_all(&self, schedule: &Schedule, base: u32, limit: u32) -> Vec<Vec<(u32, u32)>> {
        let schedule_id = schedule as *const Schedule as usize;
        let mut cache = self.allocation_occupancy.0.lock().unwrap();
        let reset = cache.as_ref().is_none_or(|cache| {
            cache.schedule != schedule_id
                || cache.tile_count != schedule.tile_count
                || cache.base != base
                || cache.limit != limit
                || cache.allocation_count > schedule.allocations.len()
        });
        if reset {
            *cache = Some(CachedAllocationOccupancy {
                schedule: schedule_id,
                allocation_count: 0,
                tile_count: schedule.tile_count,
                base,
                limit,
                occupied: vec![Vec::new(); usize::from(schedule.tile_count)],
            });
        }
        let cache = cache.as_mut().unwrap();
        for allocation in &schedule.allocations[cache.allocation_count..] {
            if allocation.live_until == 0 || allocation.live_from == usize::MAX {
                continue;
            }
            let start = allocation.address.max(base);
            let end = allocation
                .address
                .saturating_add(allocation.size)
                .min(limit);
            if start < end {
                cache.occupied[usize::from(allocation.tile)].push((start, end));
            }
        }
        cache.allocation_count = schedule.allocations.len();
        merge_occupied_intervals(&mut cache.occupied);
        cache.occupied.clone()
    }
}

fn unique_ipu21_regions(regions: &[Ipu21MemoryRegion]) -> bool {
    !regions.is_empty()
        && regions
            .iter()
            .enumerate()
            .all(|(index, region)| !regions[..index].iter().any(|previous| previous == region))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryConstraint {
    /// Inclusive first byte available for allocation.
    pub base: u32,
    /// Exclusive end of the allocation window.
    pub limit: u32,
    pub alignment: u32,
    pub placement: MemoryPlacement,
}

/// Finds an address satisfying an address window and a half-open phase lifetime.
/// Allocations on different tiles or with disjoint lifetimes may share an address.
pub fn find_free_region(
    allocations: &[Allocation],
    tile: u16,
    size: u32,
    live_from: usize,
    live_until: usize,
    constraint: MemoryConstraint,
) -> Result<u32, CompileError> {
    if size == 0
        || live_from >= live_until
        || !constraint.alignment.is_power_of_two()
        || constraint.base >= constraint.limit
        || size > constraint.limit - constraint.base
    {
        return Err(CompileError::Memory("invalid allocation constraint".into()));
    }
    let mut occupied = allocations
        .iter()
        .filter(|allocation| {
            allocation.tile == tile
                && live_from < allocation.live_until
                && allocation.live_from < live_until
        })
        .filter_map(|allocation| {
            let start = allocation.address.max(constraint.base);
            let end = allocation
                .address
                .saturating_add(allocation.size)
                .min(constraint.limit);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();
    occupied.sort_unstable();
    let mut merged = Vec::<(u32, u32)>::new();
    for (start, end) in occupied {
        if let Some(previous) = merged.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    let mut gaps = Vec::with_capacity(merged.len() + 1);
    let mut cursor = constraint.base;
    for (start, end) in merged {
        if cursor < start {
            gaps.push((cursor, start));
        }
        cursor = cursor.max(end);
    }
    if cursor < constraint.limit {
        gaps.push((cursor, constraint.limit));
    }
    let occupied_bytes = (constraint.limit - constraint.base)
        .saturating_sub(gaps.iter().map(|(start, end)| end - start).sum::<u32>());
    let largest_gap = gaps
        .iter()
        .map(|(start, end)| end - start)
        .max()
        .unwrap_or(0);
    let alignment = constraint.alignment;
    let candidate = match constraint.placement {
        MemoryPlacement::Low => gaps.iter().copied().find_map(|(start, end)| {
            let address = align_u32(start, alignment);
            address
                .checked_add(size)
                .filter(|&candidate_end| candidate_end <= end)
                .map(|_| address)
        }),
        MemoryPlacement::High => gaps.iter().copied().rev().find_map(|(start, end)| {
            let address = end.checked_sub(size)? & !(alignment - 1);
            (address >= start).then_some(address)
        }),
    };
    if let Some(address) = candidate {
        return Ok(address);
    }
    Err(CompileError::Memory(format!(
        "no {size}-byte region for tile {tile} in 0x{:x}..0x{:x}: {occupied_bytes} live bytes, {largest_gap}-byte largest gap",
        constraint.base, constraint.limit,
    )))
}

pub fn find_free_region_in_arenas(
    allocations: &[Allocation],
    tile: u16,
    size: u32,
    live_from: usize,
    live_until: usize,
    arenas: &[MemoryArena],
    alignment: u32,
) -> Result<u32, CompileError> {
    let base = arenas
        .iter()
        .map(|arena| arena.base)
        .min()
        .ok_or_else(|| CompileError::Memory("allocation requires an SRAM arena".into()))?;
    let limit = arenas.iter().map(|arena| arena.limit).max().unwrap();
    let mut intervals = allocations
        .iter()
        .filter(|allocation| {
            allocation.tile == tile
                && live_from < allocation.live_until
                && allocation.live_from < live_until
        })
        .filter_map(|allocation| {
            let start = allocation.address.max(base);
            let end = allocation
                .address
                .saturating_add(allocation.size)
                .min(limit);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable();
    let mut occupied = Vec::<(u32, u32)>::with_capacity(intervals.len());
    for (start, end) in intervals {
        if let Some(previous) = occupied.last_mut()
            && start <= previous.1
        {
            previous.1 = previous.1.max(end);
        } else {
            occupied.push((start, end));
        }
    }
    allocate_from_occupied_arenas(&mut occupied, size, arenas, alignment)
}

pub fn allocate_from_occupied(
    occupied: &mut Vec<(u32, u32)>,
    size: u32,
    constraint: MemoryConstraint,
) -> Result<u32, CompileError> {
    let alignment = constraint.alignment;
    let address = match constraint.placement {
        MemoryPlacement::Low => {
            let mut cursor = constraint.base;
            let mut found = None;
            for &(start, end) in occupied.iter() {
                if cursor < start {
                    let candidate = align_u32(cursor, alignment);
                    if candidate
                        .checked_add(size)
                        .is_some_and(|candidate_end| candidate_end <= start)
                    {
                        found = Some(candidate);
                        break;
                    }
                }
                cursor = cursor.max(end);
            }
            found.or_else(|| {
                let candidate = align_u32(cursor, alignment);
                candidate
                    .checked_add(size)
                    .filter(|&end| end <= constraint.limit)
                    .map(|_| candidate)
            })
        }
        MemoryPlacement::High => {
            let mut cursor = constraint.limit;
            let mut found = None;
            for &(start, end) in occupied.iter().rev() {
                if end < cursor {
                    let candidate = cursor
                        .checked_sub(size)
                        .map(|value| value & !(alignment - 1));
                    if candidate.is_some_and(|candidate| candidate >= end) {
                        found = candidate;
                        break;
                    }
                }
                cursor = cursor.min(start);
            }
            found.or_else(|| {
                cursor
                    .checked_sub(size)
                    .map(|value| value & !(alignment - 1))
                    .filter(|&candidate| candidate >= constraint.base)
            })
        }
    }
    .ok_or_else(|| CompileError::Memory(format!("no {size}-byte region in SRAM arena")))?;
    let end = address + size;
    let insertion = occupied.partition_point(|&(start, _)| start < address);
    occupied.insert(insertion, (address, end));
    Ok(address)
}

pub fn allocate_from_occupied_arenas(
    occupied: &mut Vec<(u32, u32)>,
    size: u32,
    arenas: &[MemoryArena],
    alignment: u32,
) -> Result<u32, CompileError> {
    if size == 0
        || arenas.is_empty()
        || !alignment.is_power_of_two()
        || arenas.iter().any(|arena| arena.base >= arena.limit)
    {
        return Err(CompileError::Memory("invalid SRAM arena allocation".into()));
    }
    for arena in arenas {
        let mut arena_occupied = occupied
            .iter()
            .filter_map(|&(start, end)| {
                let start = start.max(arena.base);
                let end = end.min(arena.limit);
                (start < end).then_some((start, end))
            })
            .collect::<Vec<_>>();
        let address = allocate_from_occupied(
            &mut arena_occupied,
            size,
            MemoryConstraint {
                base: arena.base,
                limit: arena.limit,
                alignment,
                placement: arena.placement,
            },
        );
        let Ok(address) = address else {
            continue;
        };
        let end = address
            .checked_add(size)
            .ok_or_else(|| CompileError::Memory("SRAM arena allocation overflow".into()))?;
        let insertion = occupied.partition_point(|&(start, _)| start < address);
        occupied.insert(insertion, (address, end));
        return Ok(address);
    }
    Err(CompileError::Memory(format!(
        "no arena can hold a {size}-byte SRAM allocation"
    )))
}

pub fn occupied_intervals_by_tile(
    allocations: &[Allocation],
    tile_count: u16,
    live_from: usize,
    live_until: usize,
    base: u32,
    limit: u32,
) -> Vec<Vec<(u32, u32)>> {
    let mut occupied = vec![Vec::<(u32, u32)>::new(); usize::from(tile_count)];
    for allocation in allocations {
        if live_from >= allocation.live_until || allocation.live_from >= live_until {
            continue;
        }
        let start = allocation.address.max(base);
        let end = allocation
            .address
            .saturating_add(allocation.size)
            .min(limit);
        if start < end {
            occupied[usize::from(allocation.tile)].push((start, end));
        }
    }
    for intervals in &mut occupied {
        intervals.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::with_capacity(intervals.len());
        for &(start, end) in intervals.iter() {
            if let Some(previous) = merged.last_mut()
                && start <= previous.1
            {
                previous.1 = previous.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        *intervals = merged;
    }
    occupied
}

fn merge_occupied_intervals(occupied: &mut [Vec<(u32, u32)>]) {
    for intervals in occupied {
        intervals.sort_unstable();
        let mut merged = Vec::<(u32, u32)>::with_capacity(intervals.len());
        for &(start, end) in intervals.iter() {
            if let Some(previous) = merged.last_mut()
                && start <= previous.1
            {
                previous.1 = previous.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        *intervals = merged;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    pub layouts: Vec<Layout>,
    pub phases: Vec<Phase>,
    pub allocations: Vec<Allocation>,
    pub tile_count: u16,
    pub peak_sram: BTreeMap<u16, u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlacement {
    pub tensor: TensorId,
    pub tile: u16,
    pub address: u32,
    pub block_row: u16,
    pub block_column: u16,
    pub row_start: u16,
    pub rows: u16,
    pub column_start: u16,
    pub columns: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockedGemmConfig {
    pub rows: u16,
    pub inner_dimension: u16,
    pub columns: u16,
    pub block_dimension: u16,
    pub inner_block_dimension: u16,
    pub row_block_dimension: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
    pub data_type: GemmDataType,
    pub retain_profile_metadata: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmDataType {
    F16,
    F16F8Weights { scale: i8 },
    F8F143 { input_scale: i8, weight_scale: i8 },
    F32,
}

impl GemmDataType {
    pub const fn input_element_bytes(self) -> u32 {
        match self {
            Self::F16 | Self::F16F8Weights { .. } => 2,
            Self::F8F143 { .. } => 1,
            Self::F32 => 4,
        }
    }

    pub const fn output_element_bytes(self) -> u32 {
        match self {
            Self::F16 | Self::F16F8Weights { .. } | Self::F8F143 { .. } => 2,
            Self::F32 => 4,
        }
    }

    pub const fn element_bytes(self) -> u32 {
        self.output_element_bytes()
    }

    pub const fn weight_element_bytes(self) -> u32 {
        match self {
            Self::F16F8Weights { .. } | Self::F8F143 { .. } => 1,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    const fn expands_weights(self) -> bool {
        matches!(self, Self::F16F8Weights { .. })
    }

    const fn weight_scale(self) -> i8 {
        match self {
            Self::F16F8Weights { scale } => scale,
            Self::F8F143 { weight_scale, .. } => weight_scale,
            Self::F16 | Self::F32 => 0,
        }
    }

    const fn product_scale(self) -> Option<i16> {
        match self {
            Self::F8F143 {
                input_scale,
                weight_scale,
            } => Some(input_scale as i16 + weight_scale as i16),
            _ => None,
        }
    }

    const fn input_scale(self) -> Option<i8> {
        match self {
            Self::F8F143 { input_scale, .. } => Some(input_scale),
            _ => None,
        }
    }

    const fn kernel_operation(self, initialize: bool, small_rows: bool) -> &'static str {
        match (self, initialize, small_rows) {
            (Self::F16, true, true) => "gemm_f16_init_small_rows",
            (Self::F16, true, false) => "gemm_f16_init_large_rows",
            (Self::F16, false, true) => "gemm_f16_accumulate_small_rows",
            (Self::F16, false, false) => "gemm_f16_accumulate_large_rows",
            (Self::F16F8Weights { .. }, true, true) => "gemm_f16_f8w_init_small_rows",
            (Self::F16F8Weights { .. }, true, false) => "gemm_f16_f8w_init_large_rows",
            (Self::F16F8Weights { .. }, false, true) => "gemm_f16_f8w_accumulate_small_rows",
            (Self::F16F8Weights { .. }, false, false) => "gemm_f16_f8w_accumulate_large_rows",
            (Self::F8F143 { .. }, true, true) => "gemm_f8_init_small_rows",
            (Self::F8F143 { .. }, true, false) => "gemm_f8_init_large_rows",
            (Self::F8F143 { .. }, false, true) => "gemm_f8_accumulate_small_rows",
            (Self::F8F143 { .. }, false, false) => "gemm_f8_accumulate_large_rows",
            (Self::F32, true, true) => "gemm_f32_init_small_rows",
            (Self::F32, true, false) => "gemm_f32_init_large_rows",
            (Self::F32, false, true) => "gemm_f32_accumulate_small_rows",
            (Self::F32, false, false) => "gemm_f32_accumulate_large_rows",
        }
    }
}

pub fn set_f8_weight_block_scales(
    schedule: &mut Schedule,
    blocks: &[BlockPlacement],
    scales: &[i8],
) -> Result<(), CompileError> {
    set_f8_weight_block_scales_in_phases(schedule, 0..schedule.phases.len(), blocks, scales)
}

pub fn set_f8_weight_block_scales_in_phases(
    schedule: &mut Schedule,
    phases: std::ops::Range<usize>,
    blocks: &[BlockPlacement],
    scales: &[i8],
) -> Result<(), CompileError> {
    if blocks.len() != scales.len() {
        return Err(CompileError::Graph(
            "FP8 weight block and scale counts differ".into(),
        ));
    }
    let mut by_tensor = BTreeMap::new();
    for (block, &scale) in blocks.iter().zip(scales) {
        if by_tensor.insert(block.tensor.0, scale).is_some() {
            return Err(CompileError::Graph(
                "FP8 weight block tensors are not unique".into(),
            ));
        }
    }
    let mut applied = BTreeSet::new();
    let selected = schedule.phases.get_mut(phases).ok_or_else(|| {
        CompileError::Graph("FP8 scale patch phase range is outside the schedule".into())
    })?;
    for phase in selected {
        let Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            if command.specialization.operation != "expand_f8_f143_to_f16" {
                continue;
            }
            let Some((&tensor, &scale)) = command
                .inputs
                .first()
                .and_then(|tensor| by_tensor.get_key_value(&tensor.0))
            else {
                continue;
            };
            let argument = command.arguments.get_mut(1).ok_or_else(|| {
                CompileError::Graph("FP8 expansion command has no scale argument".into())
            })?;
            *argument = u32::from(scale as u8);
            if !command.metadata.is_empty() {
                command
                    .metadata
                    .insert("f143_scale".into(), scale.to_string());
            }
            applied.insert(tensor);
        }
    }
    if applied.len() != by_tensor.len() {
        return Err(CompileError::Graph(
            "some FP8 weight blocks have no expansion command".into(),
        ));
    }
    Ok(())
}

pub fn set_native_f8_weight_block_scales_in_phases(
    schedule: &mut Schedule,
    phases: std::ops::Range<usize>,
    input_scale: i8,
    blocks: &[BlockPlacement],
    scales: &[i8],
) -> Result<(), CompileError> {
    if blocks.len() != scales.len() {
        return Err(CompileError::Graph(
            "native FP8 weight block and scale counts differ".into(),
        ));
    }
    let by_tensor = blocks
        .iter()
        .zip(scales)
        .map(|(block, &scale)| (block.tensor.0, scale))
        .collect::<BTreeMap<_, _>>();
    if by_tensor.len() != blocks.len() {
        return Err(CompileError::Graph(
            "native FP8 weight block tensors are not unique".into(),
        ));
    }
    let mut applied = BTreeSet::new();
    let selected = schedule.phases.get_mut(phases).ok_or_else(|| {
        CompileError::Graph("native FP8 scale patch phase range is outside the schedule".into())
    })?;
    for phase in selected {
        let Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            if !command.specialization.operation.starts_with("gemm_f8_") {
                continue;
            }
            let Some((&tensor, &weight_scale)) = command
                .inputs
                .get(1)
                .and_then(|tensor| by_tensor.get_key_value(&tensor.0))
            else {
                continue;
            };
            let product_scale = i16::from(input_scale) + i16::from(weight_scale);
            if !(-32..=31).contains(&product_scale) {
                return Err(CompileError::Graph(format!(
                    "native FP8 product scale {product_scale} is outside the hardware range"
                )));
            }
            let argument = command.arguments.first_mut().ok_or_else(|| {
                CompileError::Graph("native FP8 GEMM command has no scale argument".into())
            })?;
            *argument = u32::from((product_scale as u8) & 0x3f);
            if !command.metadata.is_empty() {
                command
                    .metadata
                    .insert("f143_input_scale".into(), input_scale.to_string());
                command
                    .metadata
                    .insert("f143_weight_scale".into(), weight_scale.to_string());
            }
            applied.insert(tensor);
        }
    }
    if applied.len() != by_tensor.len() {
        return Err(CompileError::Graph(
            "some native FP8 weight blocks have no GEMM command".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedGemmPlan {
    pub schedule: Schedule,
    pub left: Vec<BlockPlacement>,
    pub right: Vec<BlockPlacement>,
    pub output: Vec<BlockPlacement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppendedBlockedGemm {
    pub right: Vec<BlockPlacement>,
    pub output: Vec<BlockPlacement>,
}

pub fn append_bias_f16_c16(
    schedule: &mut Schedule,
    output: &[BlockPlacement],
    data_base: u32,
    data_limit: u32,
) -> Result<Vec<BlockPlacement>, CompileError> {
    append_bias_f16_c16_in_arenas(schedule, output, &[MemoryArena::low(data_base, data_limit)])
}

pub fn append_bias_f16_c16_in_arenas(
    schedule: &mut Schedule,
    output: &[BlockPlacement],
    arenas: &[MemoryArena],
) -> Result<Vec<BlockPlacement>, CompileError> {
    let data_base = arenas.iter().map(|arena| arena.base).min().unwrap_or(0);
    let data_limit = arenas.iter().map(|arena| arena.limit).max().unwrap_or(0);
    if output.is_empty()
        || data_base & 7 != 0
        || data_base >= data_limit
        || output
            .iter()
            .any(|block| block.columns == 0 || !block.columns.is_multiple_of(16))
    {
        return Err(CompileError::Graph(
            "C16 bias add requires 16-column-aligned output blocks and aligned SRAM".into(),
        ));
    }
    let exchange_phase = schedule.phases.len();
    let compute_phase = exchange_phase + 1;
    let mut next_tensor = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    let mut occupied = occupied_intervals_by_tile(
        &schedule.allocations,
        schedule.tile_count,
        0,
        usize::MAX,
        data_base,
        data_limit,
    );
    let mut biases = Vec::new();
    for block_column in 0..=output.iter().map(|block| block.block_column).max().unwrap() {
        let owner = output
            .iter()
            .find(|block| block.block_column == block_column)
            .ok_or_else(|| CompileError::Graph("C16 output has a missing column block".into()))?;
        let size = u32::from(owner.columns) * 2;
        let address =
            allocate_from_occupied_arenas(&mut occupied[usize::from(owner.tile)], size, arenas, 8)?;
        let bias = BlockPlacement {
            tensor: TensorId(next_tensor),
            address,
            rows: 1,
            row_start: 0,
            ..*owner
        };
        next_tensor += 1;
        schedule.allocations.push(Allocation {
            tensor: bias.tensor,
            tile: bias.tile,
            address,
            size,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        biases.push(bias);
    }
    let mut transfers = Vec::new();
    let mut commands = Vec::with_capacity(output.len());
    let mut staging_cursors =
        vec![ipu_exchange::EXCHANGE_WINDOW_BASE; usize::from(schedule.tile_count)];
    for output in output {
        let bias = biases
            .iter()
            .find(|bias| bias.block_column == output.block_column)
            .unwrap();
        let bytes = u32::from(output.columns) * 2;
        if bias.tile != output.tile {
            let cursor = &mut staging_cursors[usize::from(output.tile)];
            let staging_address = *cursor;
            *cursor = align_u32(
                cursor.checked_add(bytes).ok_or_else(|| {
                    CompileError::Memory("C16 bias staging address overflow".into())
                })?,
                32,
            );
            if *cursor > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES {
                return Err(CompileError::Memory(format!(
                    "C16 bias staging exhausts tile {} exchange window",
                    output.tile
                )));
            }
            transfers.push(Transfer {
                source_tile: bias.tile,
                destination_tile: output.tile,
                tensor: bias.tensor,
                bytes,
                staging_address: Some(staging_address),
            });
        }
        commands.push(KernelCommand {
            tile: output.tile,
            output: output.tensor,
            inputs: vec![output.tensor, bias.tensor],
            arguments: vec![
                u32::from(output.rows),
                u32::from(output.rows / 6) | (u32::from(output.columns / 16) << 16),
                u32::from(output.rows % 6),
            ],
            specialization: SpecializationKey {
                operation: "add_bias_f16_c16".into(),
                shape: vec![usize::from(output.rows), usize::from(output.columns)],
                worker_count: 6,
                role: "blocked-bias".into(),
                alignment: 8,
            },
            metadata: BTreeMap::from([
                ("label".into(), "blocked bias add".into()),
                ("row_start".into(), output.row_start.to_string()),
                ("column_start".into(), output.column_start.to_string()),
            ]),
        });
    }
    schedule.phases.push(Phase::Exchange { transfers });
    schedule.phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands: commands.into_iter().map(Arc::new).collect(),
    });
    Ok(biases)
}

pub fn append_blocked_gemm_f16_with_a16_input(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: BlockedGemmConfig,
) -> Result<AppendedBlockedGemm, CompileError> {
    let arenas = [MemoryArena::low(config.data_base, config.data_limit)];
    append_blocked_gemm_f16_with_a16_input_in_arenas(schedule, input, config, &arenas)
}

pub fn append_blocked_gemm_f16_with_a16_input_in_arenas(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: BlockedGemmConfig,
    arenas: &[MemoryArena],
) -> Result<AppendedBlockedGemm, CompileError> {
    append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        input,
        config,
        &MemoryPolicy {
            resident: arenas
                .iter()
                .map(|arena| arena.with_placement(MemoryPlacement::High))
                .collect(),
            transient: arenas.to_vec(),
            resident_tile_assignment: ResidentTileAssignment::Balanced,
            allocation_occupancy: AllocationOccupancyCache::default(),
        },
    )
}

pub fn append_blocked_gemm_f16_with_a16_input_with_memory_policy(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    config: BlockedGemmConfig,
    memory: &MemoryPolicy,
) -> Result<AppendedBlockedGemm, CompileError> {
    if !matches!(
        config.data_type,
        GemmDataType::F16 | GemmDataType::F16F8Weights { .. } | GemmDataType::F8F143 { .. }
    ) {
        return Err(CompileError::Graph(
            "A16 composition requires an FP16 GEMM".into(),
        ));
    }
    memory.validate()?;
    let mut plan = plan_appended_blocked_gemm_with_memory_policy(schedule, config, memory)?;
    let tensor_base = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    remap_gemm_tensors(&mut plan, tensor_base)?;
    let mut next_tensor = plan
        .schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(tensor_base)
        + 1;
    if input
        .iter()
        .any(|shard| shard.columns != config.inner_dimension)
    {
        return Err(CompileError::Graph(format!(
            "A16 input columns do not all match {}",
            config.inner_dimension
        )));
    }
    type InputFragment<'a> = (&'a RowShardPlacement, u16, u16, u16);
    let first_compute_phase = schedule.phases.len() + 1;
    let placements = plan
        .left
        .iter()
        .map(|block| {
            let row_end = block.row_start + block.rows;
            let fragments = input
                .iter()
                .filter_map(|shard| {
                    let overlap_start = block.row_start.max(shard.row_start);
                    let overlap_end = row_end.min(shard.row_start + shard.rows);
                    (overlap_start < overlap_end).then(|| {
                        (
                            shard,
                            overlap_start - shard.row_start,
                            overlap_start - block.row_start,
                            overlap_end - overlap_start,
                        )
                    })
                })
                .collect::<Vec<InputFragment<'_>>>();
            let covered = fragments
                .iter()
                .map(|(_, _, _, copy_rows)| *copy_rows)
                .sum::<u16>();
            if covered != block.rows {
                return Err(CompileError::Graph(format!(
                    "GEMM row block {}..{} has only {covered} A16 source rows",
                    block.row_start, row_end
                )));
            }
            let exact = fragments.len() == 1
                && fragments[0].1 == 0
                && fragments[0].2 == 0
                && fragments[0].3 == block.rows
                && fragments[0].0.rows == block.rows;
            Ok((fragments, exact))
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let requirements = plan
        .left
        .iter()
        .zip(&placements)
        .map(|(block, (fragments, _))| WindowRequirement {
            tile: block.tile,
            regions: fragments
                .iter()
                .filter(|(shard, ..)| shard.tile != block.tile)
                .map(|(shard, _, _, copy_rows)| {
                    let panel_count = block.columns / 16;
                    let panel_stride = u32::from(shard.rows) * 32;
                    u32::from(panel_count - 1) * panel_stride + u32::from(*copy_rows) * 32
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let placement_passes = partition_address_window(
        &requirements,
        config.tile_count,
        ipu_exchange::EXCHANGE_WINDOW_BASE,
        ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        32,
    )?;
    for selected in placement_passes {
        let exchange_phase = schedule.phases.len();
        let compute_phase = exchange_phase + 1;
        let mut transfers = Vec::with_capacity(selected.len());
        let mut commands = Vec::with_capacity(selected.len());
        let mut staging_cursors =
            vec![ipu_exchange::EXCHANGE_WINDOW_BASE; usize::from(config.tile_count)];
        for block_index in selected {
            let block = &plan.left[block_index];
            let (fragments, exact) = &placements[block_index];
            let exact = *exact;
            for (shard, source_row_start, destination_row_start, copy_rows) in
                fragments.iter().copied()
            {
                let panel_count = block.columns / 16;
                let panel_stride = u32::from(shard.rows) * 32;
                let source_bytes =
                    u32::from(panel_count - 1) * panel_stride + u32::from(copy_rows) * 32;
                let alias = TensorId(next_tensor);
                next_tensor += 1;
                let address = shard.address
                    + u32::from(block.column_start) * u32::from(shard.rows) * 2
                    + u32::from(source_row_start) * 32;
                schedule.allocations.push(Allocation {
                    tensor: alias,
                    tile: shard.tile,
                    address,
                    size: source_bytes,
                    live_from: exchange_phase,
                    live_until: compute_phase + 1,
                    kind: AllocationKind::HomeAlias {
                        source: shard.tensor,
                    },
                });
                if shard.tile != block.tile {
                    let cursor = &mut staging_cursors[usize::from(block.tile)];
                    let staging_address = *cursor;
                    *cursor = align_u32(
                        cursor.checked_add(source_bytes).ok_or_else(|| {
                            CompileError::Memory("A16 staging address overflow".into())
                        })?,
                        32,
                    );
                    if *cursor
                        > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES
                    {
                        return Err(CompileError::Memory(format!(
                            "A16 GEMM input staging exhausts tile {} exchange window",
                            block.tile
                        )));
                    }
                    transfers.push(Transfer {
                        source_tile: shard.tile,
                        destination_tile: block.tile,
                        tensor: alias,
                        bytes: source_bytes,
                        staging_address: Some(staging_address),
                    });
                }
                let units = source_bytes / 8;
                let input_scale = config.data_type.input_scale();
                commands.push(KernelCommand {
                    tile: block.tile,
                    output: block.tensor,
                    inputs: vec![alias, alias],
                    arguments: match (exact, input_scale) {
                        (true, Some(scale)) => vec![
                            u32::from(block.rows),
                            u32::from(block.columns),
                            u32::from(scale as u8),
                        ],
                        (true, None) => vec![units, units / 6, units % 6],
                        (false, Some(scale)) => vec![
                            pack_reblock_row_pair(shard.rows, block.rows)?,
                            pack_reblock_row_pair(0, destination_row_start)?,
                            pack_a16_reblock_count(copy_rows, block.columns, Some(scale))?,
                        ],
                        (false, None) => vec![
                            pack_reblock_row_pair(shard.rows, block.rows)?,
                            pack_reblock_row_pair(0, destination_row_start)?,
                            pack_a16_reblock_count(copy_rows, block.columns, None)?,
                        ],
                    },
                    specialization: SpecializationKey {
                        operation: match (exact, input_scale.is_some()) {
                            (true, true) => "quantize_a16_to_a32_f143",
                            (true, false) => "copy_u64",
                            (false, true) => "quantize_reblock_a16_to_a32_f143",
                            (false, false) => "reblock_f16_a16_to_a16",
                        }
                        .into(),
                        shape: if exact {
                            vec![usize::from(block.rows), usize::from(block.columns)]
                        } else {
                            vec![
                                usize::from(shard.rows),
                                usize::from(block.rows),
                                usize::from(copy_rows),
                            ]
                        },
                        worker_count: 6,
                        role: if input_scale.is_some() {
                            "row-sharded A16 to native FP8 GEMM input placement"
                        } else {
                            "row-sharded A16 GEMM input placement"
                        }
                        .into(),
                        alignment: if input_scale.is_some() { 16 } else { 8 },
                    },
                    metadata: BTreeMap::from([
                        ("label".into(), "place row-sharded GEMM input".into()),
                        ("row_start".into(), block.row_start.to_string()),
                        ("column_start".into(), block.column_start.to_string()),
                        ("copy_rows".into(), copy_rows.to_string()),
                    ]),
                });
            }
        }
        schedule.phases.push(Phase::Exchange { transfers });
        schedule.phases.push(Phase::Compute {
            op: OpId(compute_phase),
            commands: commands.into_iter().map(Arc::new).collect(),
        });
    }
    prepare_appended_gemm_lifetimes(&mut plan);
    let left_tensors = plan
        .left
        .iter()
        .map(|block| block.tensor)
        .collect::<HashSet<_>>();
    let allocation_base = schedule.allocations.len();
    append_child_schedule(schedule, &mut plan.schedule)?;
    set_appended_gemm_left_start(
        &mut schedule.allocations[allocation_base..],
        &left_tensors,
        first_compute_phase,
    );
    Ok(AppendedBlockedGemm {
        right: plan.right,
        output: plan.output,
    })
}

pub fn append_blocked_gemm_f16_with_a16_blocks(
    schedule: &mut Schedule,
    input: &[BlockPlacement],
    config: BlockedGemmConfig,
) -> Result<AppendedBlockedGemm, CompileError> {
    let arenas = [MemoryArena::low(config.data_base, config.data_limit)];
    append_blocked_gemm_f16_with_a16_blocks_in_arenas(schedule, input, config, &arenas)
}

pub fn append_blocked_gemm_f16_with_a16_blocks_in_arenas(
    schedule: &mut Schedule,
    input: &[BlockPlacement],
    config: BlockedGemmConfig,
    arenas: &[MemoryArena],
) -> Result<AppendedBlockedGemm, CompileError> {
    append_blocked_gemm_f16_with_a16_blocks_with_memory_policy(
        schedule,
        input,
        config,
        &MemoryPolicy {
            resident: arenas
                .iter()
                .map(|arena| arena.with_placement(MemoryPlacement::High))
                .collect(),
            transient: arenas.to_vec(),
            resident_tile_assignment: ResidentTileAssignment::Balanced,
            allocation_occupancy: AllocationOccupancyCache::default(),
        },
    )
}

pub fn append_blocked_gemm_f16_with_a16_blocks_with_memory_policy(
    schedule: &mut Schedule,
    input: &[BlockPlacement],
    config: BlockedGemmConfig,
    memory: &MemoryPolicy,
) -> Result<AppendedBlockedGemm, CompileError> {
    if !matches!(
        config.data_type,
        GemmDataType::F16 | GemmDataType::F16F8Weights { .. } | GemmDataType::F8F143 { .. }
    ) {
        return Err(CompileError::Graph(
            "A16 block composition requires an FP16 GEMM".into(),
        ));
    }
    memory.validate()?;
    let mut plan = plan_appended_blocked_gemm_with_memory_policy(schedule, config, memory)?;
    let tensor_base = schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(0)
        + 1;
    remap_gemm_tensors(&mut plan, tensor_base)?;
    let mut next_tensor = plan
        .schedule
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.0)
        .max()
        .unwrap_or(tensor_base)
        + 1;
    type InputFragment<'a> = (&'a BlockPlacement, u16, u16, u16);
    let first_compute_phase = schedule.phases.len() + 1;
    let placements = plan
        .left
        .iter()
        .map(|destination| {
            let row_end = destination.row_start + destination.rows;
            let mut fragments = input
                .iter()
                .filter_map(|source| {
                    if source.column_start != destination.column_start
                        || source.columns != destination.columns
                    {
                        return None;
                    }
                    let overlap_start = destination.row_start.max(source.row_start);
                    let overlap_end = row_end.min(source.row_start + source.rows);
                    (overlap_start < overlap_end).then(|| {
                        (
                            source,
                            overlap_start - source.row_start,
                            overlap_start - destination.row_start,
                            overlap_end - overlap_start,
                        )
                    })
                })
                .collect::<Vec<InputFragment<'_>>>();
            fragments.sort_unstable_by_key(|(_, _, destination_row, _)| *destination_row);
            let covered = fragments.iter().map(|(_, _, _, rows)| *rows).sum::<u16>();
            if covered != destination.rows {
                return Err(CompileError::Graph(format!(
                    "GEMM input block ({}, {}) has only {covered} source rows",
                    destination.block_row, destination.block_column
                )));
            }
            let exact = fragments.len() == 1
                && fragments[0].1 == 0
                && fragments[0].2 == 0
                && fragments[0].3 == destination.rows
                && fragments[0].0.rows == destination.rows;
            Ok((fragments, exact))
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let requirements = plan
        .left
        .iter()
        .zip(&placements)
        .map(|(destination, (fragments, _))| WindowRequirement {
            tile: destination.tile,
            regions: fragments
                .iter()
                .filter(|(source, ..)| source.tile != destination.tile)
                .map(|(source, _, _, copy_rows)| {
                    let panel_count = source.columns / 16;
                    let panel_stride = u32::from(source.rows) * 32;
                    u32::from(panel_count - 1) * panel_stride + u32::from(*copy_rows) * 32
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let placement_passes = partition_address_window(
        &requirements,
        config.tile_count,
        ipu_exchange::EXCHANGE_WINDOW_BASE,
        ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        32,
    )?;
    for selected in placement_passes {
        let exchange_phase = schedule.phases.len();
        let compute_phase = exchange_phase + 1;
        let mut transfers = Vec::with_capacity(selected.len());
        let mut commands = Vec::with_capacity(selected.len());
        let mut staging_cursors =
            vec![ipu_exchange::EXCHANGE_WINDOW_BASE; usize::from(config.tile_count)];
        for destination_index in selected {
            let destination = &plan.left[destination_index];
            let (fragments, exact) = &placements[destination_index];
            let exact = *exact;
            for (source, source_row_start, destination_row_start, copy_rows) in
                fragments.iter().copied()
            {
                let panel_count = source.columns / 16;
                let panel_stride = u32::from(source.rows) * 32;
                let source_bytes =
                    u32::from(panel_count - 1) * panel_stride + u32::from(copy_rows) * 32;
                let source_alias = TensorId(next_tensor);
                next_tensor += 1;
                schedule.allocations.push(Allocation {
                    tensor: source_alias,
                    tile: source.tile,
                    address: source.address + u32::from(source_row_start) * 32,
                    size: source_bytes,
                    live_from: exchange_phase,
                    live_until: compute_phase + 1,
                    kind: AllocationKind::HomeAlias {
                        source: source.tensor,
                    },
                });
                if source.tile != destination.tile {
                    let cursor = &mut staging_cursors[usize::from(destination.tile)];
                    let staging_address = *cursor;
                    *cursor = align_u32(
                        cursor.checked_add(source_bytes).ok_or_else(|| {
                            CompileError::Memory("A16 staging address overflow".into())
                        })?,
                        32,
                    );
                    if *cursor
                        > ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES
                    {
                        return Err(CompileError::Memory(format!(
                            "distributed A16 GEMM input staging exhausts tile {} exchange window",
                            destination.tile
                        )));
                    }
                    transfers.push(Transfer {
                        source_tile: source.tile,
                        destination_tile: destination.tile,
                        tensor: source_alias,
                        bytes: source_bytes,
                        staging_address: Some(staging_address),
                    });
                }
                let input_scale = config.data_type.input_scale();
                let units = source_bytes / 8;
                commands.push(KernelCommand {
                    tile: destination.tile,
                    output: destination.tensor,
                    inputs: vec![source_alias, source_alias],
                    arguments: match (exact, input_scale) {
                        (true, Some(scale)) => vec![
                            u32::from(destination.rows),
                            u32::from(destination.columns),
                            u32::from(scale as u8),
                        ],
                        (true, None) => vec![units, units / 6, units % 6],
                        (false, Some(scale)) => vec![
                            pack_reblock_row_pair(source.rows, destination.rows)?,
                            pack_reblock_row_pair(0, destination_row_start)?,
                            pack_a16_reblock_count(copy_rows, destination.columns, Some(scale))?,
                        ],
                        (false, None) => vec![
                            pack_reblock_row_pair(source.rows, destination.rows)?,
                            pack_reblock_row_pair(0, destination_row_start)?,
                            pack_a16_reblock_count(copy_rows, destination.columns, None)?,
                        ],
                    },
                    specialization: SpecializationKey {
                        operation: match (exact, input_scale.is_some()) {
                            (true, true) => "quantize_a16_to_a32_f143",
                            (true, false) => "copy_u64",
                            (false, true) => "quantize_reblock_a16_to_a32_f143",
                            (false, false) => "reblock_f16_a16_to_a16",
                        }
                        .into(),
                        shape: vec![
                            usize::from(source.rows),
                            usize::from(destination.rows),
                            usize::from(copy_rows),
                            usize::from(destination.columns),
                        ],
                        worker_count: 6,
                        role: if input_scale.is_some() {
                            "distributed A16 to native FP8 GEMM input placement"
                        } else {
                            "distributed A16 GEMM input placement"
                        }
                        .into(),
                        alignment: if input_scale.is_some() { 16 } else { 8 },
                    },
                    metadata: BTreeMap::from([
                        ("label".into(), "place distributed GEMM input".into()),
                        ("row_start".into(), destination.row_start.to_string()),
                        ("column_start".into(), destination.column_start.to_string()),
                        ("copy_rows".into(), copy_rows.to_string()),
                    ]),
                });
            }
        }
        schedule.phases.push(Phase::Exchange { transfers });
        schedule.phases.push(Phase::Compute {
            op: OpId(compute_phase),
            commands: commands.into_iter().map(Arc::new).collect(),
        });
    }
    prepare_appended_gemm_lifetimes(&mut plan);
    let left_tensors = plan
        .left
        .iter()
        .map(|block| block.tensor)
        .collect::<HashSet<_>>();
    let allocation_base = schedule.allocations.len();
    append_child_schedule(schedule, &mut plan.schedule)?;
    set_appended_gemm_left_start(
        &mut schedule.allocations[allocation_base..],
        &left_tensors,
        first_compute_phase,
    );
    Ok(AppendedBlockedGemm {
        right: plan.right,
        output: plan.output,
    })
}

fn prepare_appended_gemm_lifetimes(plan: &mut BlockedGemmPlan) {
    let completion = plan.schedule.phases.len();
    let left = plan
        .left
        .iter()
        .map(|block| block.tensor)
        .collect::<HashSet<_>>();
    for allocation in &mut plan.schedule.allocations {
        if allocation.kind == AllocationKind::Home && left.contains(&allocation.tensor) {
            allocation.live_until = completion;
        }
    }
    let output = plan
        .output
        .iter()
        .map(|block| block.tensor)
        .collect::<HashSet<_>>();
    for allocation in &mut plan.schedule.allocations {
        if allocation.kind == AllocationKind::Home
            && output.contains(&allocation.tensor)
            && allocation.live_from == 0
        {
            allocation.live_from = 1;
        }
    }
}

fn set_appended_gemm_left_start(
    allocations: &mut [Allocation],
    left_tensors: &HashSet<TensorId>,
    live_from: usize,
) {
    for allocation in allocations {
        if allocation.kind == AllocationKind::Home && left_tensors.contains(&allocation.tensor) {
            allocation.live_from = live_from;
        }
    }
}

fn plan_appended_blocked_gemm_with_memory_policy(
    parent: &Schedule,
    config: BlockedGemmConfig,
    memory: &MemoryPolicy,
) -> Result<BlockedGemmPlan, CompileError> {
    let mut plan = plan_blocked_gemm(config)?;
    let tile_rotation = match memory.resident_tile_assignment {
        ResidentTileAssignment::Balanced => choose_resident_tile_rotation_in_arenas(
            parent,
            &plan.right,
            config.data_type.weight_element_bytes(),
            memory,
        ),
        ResidentTileAssignment::Fixed => 0,
    };
    rotate_gemm_plan_tiles(&mut plan, tile_rotation)?;
    let mut regions = plan
        .left
        .iter()
        .chain(&plan.output)
        .map(|placement| {
            (
                placement.tensor,
                placement.tile,
                placement.address,
                u32::from(placement.rows) * u32::from(placement.columns) * 2,
                false,
            )
        })
        .collect::<Vec<_>>();
    regions.extend(plan.right.iter().map(|placement| {
        (
            placement.tensor,
            placement.tile,
            placement.address,
            u32::from(placement.rows)
                * u32::from(placement.columns)
                * config.data_type.weight_element_bytes(),
            true,
        )
    }));
    regions.sort_unstable_by_key(|&(_, _, _, size, _)| std::cmp::Reverse(size));
    let arena_base = memory
        .resident
        .iter()
        .chain(&memory.transient)
        .map(|arena| arena.base)
        .min()
        .ok_or_else(|| {
            CompileError::Memory("appended GEMM requires at least one SRAM arena".into())
        })?;
    let arena_limit = memory
        .resident
        .iter()
        .chain(&memory.transient)
        .map(|arena| arena.limit)
        .max()
        .unwrap();
    let mut occupied_current = occupied_intervals_by_tile(
        &parent.allocations,
        parent.tile_count,
        parent.phases.len(),
        usize::MAX,
        arena_base,
        arena_limit,
    );
    let mut occupied_all = memory.occupied_all(parent, arena_base, arena_limit);
    let movable = regions
        .iter()
        .map(|&(tensor, ..)| tensor)
        .collect::<HashSet<_>>();
    for allocation in &plan.schedule.allocations {
        if allocation.kind != AllocationKind::Home || movable.contains(&allocation.tensor) {
            continue;
        }
        let start = allocation.address.max(arena_base);
        let end = allocation
            .address
            .saturating_add(allocation.size)
            .min(arena_limit);
        if start < end {
            occupied_current[usize::from(allocation.tile)].push((start, end));
            occupied_all[usize::from(allocation.tile)].push((start, end));
        }
    }
    merge_occupied_intervals(&mut occupied_current);
    merge_occupied_intervals(&mut occupied_all);
    let mut relocated = HashMap::<TensorId, u32>::default();
    for &(tensor, tile, _old_address, size, resident) in &regions {
        let allocation_arenas = if resident {
            &memory.resident
        } else {
            &memory.transient
        };
        let occupied = if resident {
            &mut occupied_all
        } else {
            &mut occupied_current
        };
        let address = allocate_from_occupied_arenas(
            &mut occupied[usize::from(tile)],
            size,
            allocation_arenas,
            32,
        )
        .map_err(|error| {
            let lifetime = if resident { "resident" } else { "transient" };
            CompileError::Memory(format!(
                "cannot place {size}-byte {lifetime} GEMM tensor {} on tile {tile}: {error}",
                tensor.0
            ))
        })?;
        let end = address.checked_add(size).ok_or_else(|| {
            CompileError::Memory("appended GEMM allocation address overflow".into())
        })?;
        let other = if resident {
            &mut occupied_current
        } else {
            &mut occupied_all
        };
        let intervals = &mut other[usize::from(tile)];
        let insertion = intervals.partition_point(|&(start, _)| start < address);
        intervals.insert(insertion, (address, end));
        relocated.insert(tensor, address);
    }
    for placement in plan
        .left
        .iter_mut()
        .chain(&mut plan.right)
        .chain(&mut plan.output)
    {
        placement.address = relocated[&placement.tensor];
    }
    for allocation in &mut plan.schedule.allocations {
        if let AllocationKind::HomeAlias { source } = allocation.kind {
            let owner = regions
                .iter()
                .find(|region| region.0 == source && region.1 == allocation.tile)
                .ok_or_else(|| {
                    CompileError::Memory(format!(
                        "GEMM alias tensor {} has no movable source tensor {} on tile {}",
                        allocation.tensor.0, source.0, allocation.tile
                    ))
                })?;
            allocation.address = relocated[&source]
                .checked_add(allocation.address.checked_sub(owner.2).ok_or_else(|| {
                    CompileError::Memory("GEMM alias precedes its source allocation".into())
                })?)
                .ok_or_else(|| CompileError::Memory("GEMM alias relocation overflow".into()))?;
            continue;
        }
        if allocation.kind != AllocationKind::Home || allocation.address < config.data_base {
            continue;
        }
        if let Some(&address) = relocated.get(&allocation.tensor) {
            allocation.address = address;
            continue;
        }
        if let Some(owner) = regions.iter().find(|region| {
            region.1 == allocation.tile
                && allocation.address >= region.2
                && allocation.address.saturating_add(allocation.size) <= region.2 + region.3
        }) {
            allocation.address = relocated[&owner.0] + allocation.address - owner.2;
        }
    }
    Ok(plan)
}

fn choose_resident_tile_rotation_in_arenas(
    parent: &Schedule,
    child_resident: &[BlockPlacement],
    element_bytes: u32,
    memory: &MemoryPolicy,
) -> u16 {
    let tile_count = usize::from(parent.tile_count);
    if tile_count == 0 {
        return 0;
    }
    let arena_base = memory
        .resident
        .iter()
        .map(|arena| arena.base)
        .min()
        .unwrap_or(0);
    let arena_limit = memory
        .resident
        .iter()
        .map(|arena| arena.limit)
        .max()
        .unwrap_or(0);
    let occupied = memory.occupied_all(parent, arena_base, arena_limit);
    let parent_bytes = occupied
        .iter()
        .map(|intervals| {
            intervals
                .iter()
                .flat_map(|&(start, end)| {
                    memory.resident.iter().map(move |arena| {
                        u64::from(end.min(arena.limit).saturating_sub(start.max(arena.base)))
                    })
                })
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let mut child_bytes = vec![0u64; tile_count];
    for block in child_resident {
        child_bytes[usize::from(block.tile)] +=
            u64::from(block.rows) * u64::from(block.columns) * u64::from(element_bytes);
    }
    (0..tile_count)
        .min_by_key(|&rotation| {
            let loads = (0..tile_count).map(|tile| {
                parent_bytes[tile] + child_bytes[(tile + tile_count - rotation) % tile_count]
            });
            let maximum = loads.clone().max().unwrap_or(0);
            let squared = loads
                .map(|load| u128::from(load) * u128::from(load))
                .sum::<u128>();
            (maximum, squared, rotation)
        })
        .unwrap_or(0) as u16
}

fn rotate_gemm_plan_tiles(plan: &mut BlockedGemmPlan, rotation: u16) -> Result<(), CompileError> {
    if rotation == 0 {
        return Ok(());
    }
    let tile_count = plan.schedule.tile_count;
    let rotate = |tile: &mut u16| {
        *tile = ((*tile as u32 + rotation as u32) % u32::from(tile_count)) as u16;
    };
    for placement in plan
        .left
        .iter_mut()
        .chain(&mut plan.right)
        .chain(&mut plan.output)
    {
        rotate(&mut placement.tile);
    }
    for allocation in &mut plan.schedule.allocations {
        rotate(&mut allocation.tile);
    }
    for phase in &mut plan.schedule.phases {
        match phase {
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    rotate(&mut transfer.source_tile);
                    rotate(&mut transfer.destination_tile);
                }
            }
            Phase::Compute { commands, .. } => {
                for command in commands {
                    let command = Arc::make_mut(command);
                    rotate(&mut command.tile);
                }
            }
        }
    }
    let mut peak_sram = BTreeMap::new();
    for (mut tile, bytes) in std::mem::take(&mut plan.schedule.peak_sram) {
        rotate(&mut tile);
        if peak_sram.insert(tile, bytes).is_some() {
            return Err(CompileError::Graph(
                "GEMM tile rotation produced duplicate peak SRAM entries".into(),
            ));
        }
    }
    plan.schedule.peak_sram = peak_sram;
    Ok(())
}

fn remap_gemm_tensors(plan: &mut BlockedGemmPlan, base: usize) -> Result<(), CompileError> {
    let remap = |tensor: &mut TensorId| -> Result<(), CompileError> {
        tensor.0 = tensor
            .0
            .checked_add(base)
            .ok_or_else(|| CompileError::Graph("GEMM tensor ID overflow".into()))?;
        Ok(())
    };
    for placement in plan
        .left
        .iter_mut()
        .chain(&mut plan.right)
        .chain(&mut plan.output)
    {
        remap(&mut placement.tensor)?;
    }
    for allocation in &mut plan.schedule.allocations {
        remap(&mut allocation.tensor)?;
        if let AllocationKind::HomeAlias { source } = &mut allocation.kind {
            remap(source)?;
        }
    }
    for phase in &mut plan.schedule.phases {
        match phase {
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    remap(&mut transfer.tensor)?;
                }
            }
            Phase::Compute { commands, .. } => {
                for command in commands {
                    let command = Arc::make_mut(command);
                    remap(&mut command.output)?;
                    for input in &mut command.inputs {
                        remap(input)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_child_schedule(parent: &mut Schedule, child: &mut Schedule) -> Result<(), CompileError> {
    if parent.tile_count != child.tile_count {
        return Err(CompileError::Graph(
            "composed schedules have different tile counts".into(),
        ));
    }
    let phase_base = parent.phases.len();
    for allocation in &mut child.allocations {
        // A child allocation spanning its complete schedule is resident and was
        // intentionally available before the composed program starts. Every
        // finite lifetime, including one beginning at local phase zero, is
        // relative to the child schedule.
        if allocation.live_from != 0 || allocation.live_until != usize::MAX {
            allocation.live_from = allocation
                .live_from
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("allocation phase overflow".into()))?;
        }
        if allocation.live_until != usize::MAX {
            allocation.live_until = allocation
                .live_until
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("allocation phase overflow".into()))?;
        }
        if let AllocationKind::ExchangeStaging { phase } = &mut allocation.kind {
            *phase = phase
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("staging phase overflow".into()))?;
        }
    }
    for phase in &mut child.phases {
        if let Phase::Compute { op, .. } = phase {
            op.0 =
                op.0.checked_add(phase_base)
                    .ok_or_else(|| CompileError::Graph("operation ID overflow".into()))?;
        }
    }
    parent.allocations.append(&mut child.allocations);
    parent.phases.append(&mut child.phases);
    for (&tile, &peak) in &child.peak_sram {
        parent
            .peak_sram
            .entry(tile)
            .and_modify(|current| *current = (*current).max(peak))
            .or_insert(peak);
    }
    Ok(())
}

pub fn choose_gemm_row_block(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Option<u16> {
    choose_gemm_row_block_for(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        GemmDataType::F32,
    )
}

pub fn choose_gemm_row_block_for(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
) -> Option<u16> {
    choose_gemm_row_block_for_max_rows(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        data_type,
        u16::MAX,
    )
}

pub fn choose_gemm_row_block_for_shape(
    rows: u16,
    inner_dimension: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
) -> Option<u16> {
    choose_gemm_row_block_for_shape_max_rows(
        rows,
        inner_dimension,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        data_type,
        u16::MAX,
    )
}

pub fn choose_gemm_row_block_for_max_rows(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
    maximum_rows: u16,
) -> Option<u16> {
    choose_gemm_row_block_with_inner_jobs(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        data_type,
        0,
        maximum_rows,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn choose_gemm_row_block_for_shape_max_rows(
    rows: u16,
    inner_dimension: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
    maximum_rows: u16,
) -> Option<u16> {
    if !inner_dimension.is_multiple_of(inner_block_dimension) {
        return None;
    }
    let inner_work_units = inner_dimension.div_ceil(GEMM_COST_INNER_MICRO_COLUMNS);
    choose_gemm_row_block_with_inner_jobs(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        data_type,
        inner_work_units,
        maximum_rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn choose_gemm_row_block_with_inner_jobs(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
    inner_work_units: u16,
    maximum_rows: u16,
) -> Option<u16> {
    let candidates = gemm_row_block_candidates_for(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        data_type,
    )
    .into_iter()
    .filter(|&candidate| candidate <= maximum_rows);
    candidates.min_by_key(|&target| {
        gemm_row_block_cost_components(
            rows,
            target,
            inner_work_units,
            columns,
            block_dimension,
            tile_count,
        )
        .expect("GEMM row-block candidates have valid dimensions")
    })
}

const GEMM_COST_INNER_MICRO_COLUMNS: u16 = 64;
const GEMM_COST_SETUP_WEIGHT_NUMERATOR: usize = 4;
const GEMM_COST_SETUP_WEIGHT_DENOMINATOR: usize = 5;

pub fn gemm_row_block_cost(
    rows: u16,
    target_rows: u16,
    inner_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Option<(usize, usize, usize)> {
    gemm_row_block_cost_components(
        rows,
        target_rows,
        inner_dimension.div_ceil(GEMM_COST_INNER_MICRO_COLUMNS),
        columns,
        block_dimension,
        tile_count,
    )
}

fn gemm_row_block_cost_components(
    rows: u16,
    target_rows: u16,
    inner_work_units: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Option<(usize, usize, usize)> {
    let row_shards = rows.div_ceil(target_rows);
    let maximum_rows = rows.div_ceil(row_shards);
    let column_blocks = columns.checked_div(block_dimension)?;
    let output_blocks = usize::from(row_shards) * usize::from(column_blocks);
    let waves = output_blocks.div_ceil(usize::from(tile_count));
    let unused_tiles = waves * usize::from(tile_count) - output_blocks;
    // Row work determines the critical path while each output block repeats
    // coefficient setup for every 64-column K microblock. The setup weight is
    // calibrated from cycle profiles of rectangular FP16 GEMMs.
    let weighted_inner_jobs =
        output_blocks * usize::from(inner_work_units) * GEMM_COST_SETUP_WEIGHT_NUMERATOR;
    let average_inner_work =
        weighted_inner_jobs.div_ceil(usize::from(tile_count) * GEMM_COST_SETUP_WEIGHT_DENOMINATOR);
    Some((
        waves * usize::from(maximum_rows) + average_inner_work,
        waves,
        unused_tiles,
    ))
}

pub fn gemm_row_block_candidates(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Vec<u16> {
    gemm_row_block_candidates_for(
        rows,
        inner_block_dimension,
        columns,
        block_dimension,
        tile_count,
        GemmDataType::F32,
    )
}

pub fn gemm_row_block_candidates_for(
    rows: u16,
    inner_block_dimension: u16,
    columns: u16,
    block_dimension: u16,
    tile_count: u16,
    data_type: GemmDataType,
) -> Vec<u16> {
    if rows == 0
        || columns == 0
        || block_dimension == 0
        || inner_block_dimension == 0
        || tile_count == 0
        || !columns.is_multiple_of(block_dimension)
    {
        return Vec::new();
    }
    let mut row_shard_counts = BTreeSet::new();
    let maximum_rows = (ipu_exchange::MAX_TRANSFER_WORDS * 4
        / (u32::from(inner_block_dimension) * data_type.input_element_bytes()))
    .min(
        (ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT - ipu_package::IPU21_INTERLEAVED_MEMORY_BASE)
            / (u32::from(block_dimension) * data_type.output_element_bytes()),
    ) as u16;
    (GEMM_MINIMUM_ROW_SHARD..=maximum_rows)
        .filter(|&target| {
            let row_shards = rows.div_ceil(target);
            rows / row_shards >= GEMM_MINIMUM_ROW_SHARD && row_shard_counts.insert(row_shards)
        })
        .collect()
}

pub fn plan_blocked_gemm(config: BlockedGemmConfig) -> Result<BlockedGemmPlan, CompileError> {
    let inner_micro_dimension = match config.data_type {
        GemmDataType::F16 | GemmDataType::F16F8Weights { .. } => 16,
        GemmDataType::F8F143 { .. } => 32,
        GemmDataType::F32 => 8,
    };
    if config.rows == 0
        || config.inner_dimension == 0
        || config.columns == 0
        || !matches!(
            (config.data_type, config.block_dimension),
            (
                GemmDataType::F16 | GemmDataType::F16F8Weights { .. } | GemmDataType::F8F143 { .. },
                32 | 64 | 128
            ) | (GemmDataType::F32, 64)
        )
        || !config
            .inner_block_dimension
            .is_multiple_of(inner_micro_dimension)
        || !config.columns.is_multiple_of(config.block_dimension)
        || !config
            .inner_dimension
            .is_multiple_of(config.inner_block_dimension)
        || config.data_base >= config.data_limit
    {
        return Err(CompileError::Graph(format!(
            "blocked GEMM requires a supported column block and inner blocks divisible by {inner_micro_dimension}"
        )));
    }
    if config
        .data_type
        .product_scale()
        .is_some_and(|scale| !(-32..=31).contains(&scale))
    {
        return Err(CompileError::Graph(
            "native FP8 GEMM product scale is outside the hardware's signed six-bit range".into(),
        ));
    }
    let column_grid = config.columns / config.block_dimension;
    let inner_grid = config.inner_dimension / config.inner_block_dimension;
    let row_grid = config.rows.div_ceil(config.row_block_dimension);
    let base_rows = config.rows / row_grid;
    let larger_row_shards = config.rows % row_grid;
    if base_rows < GEMM_MINIMUM_ROW_SHARD {
        return Err(CompileError::Graph(format!(
            "blocked GEMM requires at least {GEMM_MINIMUM_ROW_SHARD} rows in every balanced row shard"
        )));
    }
    let output_block_count = usize::from(row_grid) * usize::from(column_grid);
    let input_element_bytes = config.data_type.input_element_bytes();
    let output_element_bytes = config.data_type.output_element_bytes();
    let right_block_bytes = u32::from(config.inner_block_dimension)
        .checked_mul(u32::from(config.block_dimension))
        .and_then(|elements| elements.checked_mul(config.data_type.weight_element_bytes()))
        .ok_or_else(|| CompileError::Memory("GEMM block size overflow".into()))?;
    let expanded_right_block_bytes = u32::from(config.inner_block_dimension)
        .checked_mul(u32::from(config.block_dimension))
        .and_then(|elements| elements.checked_mul(output_element_bytes))
        .ok_or_else(|| CompileError::Memory("expanded GEMM block size overflow".into()))?;
    if right_block_bytes > ipu_exchange::MAX_TRANSFER_WORDS * 4 {
        return Err(CompileError::Graph(format!(
            "{right_block_bytes}-byte GEMM blocks exceed one exchange transfer"
        )));
    }
    let maximum_rows = base_rows + u16::from(larger_row_shards != 0);
    let max_left_bytes = u32::from(maximum_rows)
        .checked_mul(u32::from(config.inner_block_dimension))
        .and_then(|elements| elements.checked_mul(input_element_bytes))
        .ok_or_else(|| CompileError::Memory("GEMM left block size overflow".into()))?;
    let max_output_bytes = u32::from(maximum_rows)
        .checked_mul(u32::from(config.block_dimension))
        .and_then(|elements| elements.checked_mul(output_element_bytes))
        .ok_or_else(|| CompileError::Memory("GEMM output block size overflow".into()))?;
    let exchange_slot_bytes = align_u32(
        max_left_bytes
            .checked_add(right_block_bytes)
            .ok_or_else(|| CompileError::Memory("GEMM exchange slot overflow".into()))?,
        32,
    );
    let direct_inner_batch_size = u16::try_from(
        (ipu_exchange::EXCHANGE_WINDOW_BYTES / exchange_slot_bytes).min(u32::from(inner_grid)),
    )
    .map_err(|_| CompileError::Graph("GEMM inner batch size overflow".into()))?;
    let tile_data_end = config
        .data_base
        .checked_add(max_left_bytes)
        .and_then(|end| end.checked_add(right_block_bytes))
        .ok_or_else(|| CompileError::Memory("GEMM per-tile data address overflow".into()))?;
    let output_address = ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
    let output_end = output_address
        .checked_add(max_output_bytes)
        .ok_or_else(|| CompileError::Memory("GEMM output address overflow".into()))?;
    let expanded_right_address = align_u32(output_end, 32);
    let inner_batch_size = direct_inner_batch_size;
    if inner_batch_size == 0 {
        return Err(CompileError::Memory(
            "one GEMM operand pair and its scratch do not fit on a tile".into(),
        ));
    }
    let interleaved_scratch_end = if config.data_type.expands_weights() {
        expanded_right_address
            .checked_add(expanded_right_block_bytes)
            .ok_or_else(|| CompileError::Memory("expanded GEMM scratch overflow".into()))?
    } else {
        output_end
    };
    if config.data_base & 31 != 0
        || tile_data_end > config.data_limit
        || interleaved_scratch_end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
        || (config.data_base < output_end && output_address < tile_data_end)
    {
        return Err(CompileError::Memory(
            "GEMM operands do not fit their required IPU21 memory elements".into(),
        ));
    }
    let left_exchange_address = ipu_exchange::EXCHANGE_WINDOW_BASE;
    let right_exchange_address = left_exchange_address + max_left_bytes;
    let left_count = usize::from(row_grid) * usize::from(inner_grid);
    let right_count = usize::from(inner_grid) * usize::from(column_grid);
    let output_tensor_base = left_count + right_count;
    let scratch_tensor_base = output_tensor_base + output_block_count;
    let expanded_right_tensor_base = scratch_tensor_base + output_block_count;
    let evacuation_tensor_base = expanded_right_tensor_base + output_block_count;
    let mut allocations = Vec::new();
    let mut left = Vec::with_capacity(left_count);
    let mut right = Vec::with_capacity(right_count);
    let mut output = Vec::with_capacity(output_block_count);
    let mut data_cursors = vec![config.data_base; usize::from(config.tile_count)];

    let source_tile = |preferred: usize, consumers: &BTreeSet<u16>, cursors: &[u32]| {
        let preferred = preferred % cursors.len();
        (0..usize::from(config.tile_count))
            .map(|offset| (preferred + offset) % usize::from(config.tile_count))
            .filter(|candidate| !consumers.contains(&(*candidate as u16)))
            .min_by_key(|candidate| {
                (
                    cursors[*candidate],
                    (*candidate + cursors.len() - preferred) % cursors.len(),
                )
            })
            .or_else(|| (0..cursors.len()).min_by_key(|candidate| cursors[*candidate]))
            .and_then(|tile| u16::try_from(tile).ok())
            .ok_or_else(|| CompileError::Graph("GEMM has no operand storage tile".into()))
    };

    for block_row in 0..row_grid {
        let rows = base_rows + u16::from(block_row < larger_row_shards);
        let row_start = block_row * base_rows + block_row.min(larger_row_shards);
        let size = u32::from(rows) * u32::from(config.inner_block_dimension) * input_element_bytes;
        for block_column in 0..inner_grid {
            let index =
                usize::from(block_row) * usize::from(inner_grid) + usize::from(block_column);
            let consumers = (0..column_grid)
                .map(|output_column| {
                    let output_index = usize::from(block_row) * usize::from(column_grid)
                        + usize::from(output_column);
                    u16::try_from(output_index % usize::from(config.tile_count)).unwrap()
                })
                .collect::<BTreeSet<_>>();
            let tile = source_tile(index, &consumers, &data_cursors)?;
            let address = data_cursors[usize::from(tile)];
            data_cursors[usize::from(tile)] = address
                .checked_add(size)
                .ok_or_else(|| CompileError::Memory("GEMM data address overflow".into()))?;
            let tensor = TensorId(index);
            allocations.push(Allocation {
                tensor,
                tile,
                address,
                size,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
            left.push(BlockPlacement {
                tensor,
                tile,
                address,
                block_row,
                block_column,
                row_start,
                rows,
                column_start: block_column * config.inner_block_dimension,
                columns: config.inner_block_dimension,
            });
        }
    }
    for block_row in 0..inner_grid {
        for block_column in 0..column_grid {
            let index =
                usize::from(block_row) * usize::from(column_grid) + usize::from(block_column);
            let consumers = (0..row_grid)
                .map(|output_row| {
                    let output_index = usize::from(output_row) * usize::from(column_grid)
                        + usize::from(block_column);
                    u16::try_from(output_index % usize::from(config.tile_count)).unwrap()
                })
                .collect::<BTreeSet<_>>();
            let tile = source_tile(left_count + index, &consumers, &data_cursors)?;
            let tensor = TensorId(left_count + index);
            let address = data_cursors[usize::from(tile)];
            data_cursors[usize::from(tile)] = address
                .checked_add(right_block_bytes)
                .ok_or_else(|| CompileError::Memory("GEMM data address overflow".into()))?;
            allocations.push(Allocation {
                tensor,
                tile,
                address,
                size: right_block_bytes,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
            right.push(BlockPlacement {
                tensor,
                tile,
                address,
                block_row,
                block_column,
                row_start: block_row * config.inner_block_dimension,
                rows: config.inner_block_dimension,
                column_start: block_column * config.block_dimension,
                columns: config.block_dimension,
            });
        }
    }

    for block_row in 0..row_grid {
        let rows = base_rows + u16::from(block_row < larger_row_shards);
        let row_start = block_row * base_rows + block_row.min(larger_row_shards);
        let size = u32::from(rows) * u32::from(config.block_dimension) * output_element_bytes;
        for block_column in 0..column_grid {
            let index =
                usize::from(block_row) * usize::from(column_grid) + usize::from(block_column);
            let tile = u16::try_from((index + 1) % usize::from(config.tile_count))
                .map_err(|_| CompileError::Graph("GEMM tile index overflow".into()))?;
            let address = data_cursors[usize::from(tile)];
            data_cursors[usize::from(tile)] = address
                .checked_add(size)
                .ok_or_else(|| CompileError::Memory("GEMM data address overflow".into()))?;
            output.push(BlockPlacement {
                tensor: TensorId(output_tensor_base + index),
                tile,
                address,
                block_row,
                block_column,
                row_start,
                rows,
                column_start: block_column * config.block_dimension,
                columns: config.block_dimension,
            });
            allocations.push(Allocation {
                tensor: TensorId(output_tensor_base + index),
                tile,
                address,
                size,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
        }
    }
    if let Some((tile, end)) = data_cursors
        .iter()
        .copied()
        .enumerate()
        .find(|(_, end)| *end > config.data_limit)
    {
        return Err(CompileError::Memory(format!(
            "GEMM resident data exhausts tile {tile}: 0x{end:x} exceeds 0x{:x}",
            config.data_limit
        )));
    }

    let output_waves = output.len().div_ceil(usize::from(config.tile_count));
    let inner_batches = inner_grid.div_ceil(inner_batch_size);
    let phases_per_batch = 2;
    let mut phases =
        Vec::with_capacity(output_waves * (usize::from(inner_batches) * phases_per_batch + 2));
    for (wave, wave_outputs) in output.chunks(usize::from(config.tile_count)).enumerate() {
        let wave_start = phases.len();
        for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
            let output_index = wave * usize::from(config.tile_count) + wave_tile;
            allocations.push(Allocation {
                tensor: TensorId(scratch_tensor_base + output_index),
                tile: u16::try_from(wave_tile)
                    .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?,
                address: output_address,
                size: u32::from(output_block.rows)
                    * u32::from(config.block_dimension)
                    * output_element_bytes,
                live_from: wave_start,
                live_until: wave_start + usize::from(inner_batches) * phases_per_batch + 1,
                kind: AllocationKind::Home,
            });
        }
        for inner_batch_start in (0..inner_grid).step_by(usize::from(inner_batch_size)) {
            let inner_batch_end = (inner_batch_start + inner_batch_size).min(inner_grid);
            let mut transfers = Vec::new();
            for inner_block in inner_batch_start..inner_batch_end {
                let slot_offset = u32::from(inner_block - inner_batch_start) * exchange_slot_bytes;
                for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
                    let destination_tile = u16::try_from(wave_tile)
                        .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?;
                    let source_index = usize::from(output_block.block_row)
                        * usize::from(inner_grid)
                        + usize::from(inner_block);
                    let source = left[source_index];
                    if source.tile != destination_tile {
                        let bytes = u32::from(source.rows)
                            * u32::from(config.inner_block_dimension)
                            * input_element_bytes;
                        transfers.push(Transfer {
                            source_tile: source.tile,
                            destination_tile,
                            tensor: source.tensor,
                            bytes,
                            staging_address: Some(left_exchange_address + slot_offset),
                        });
                    }
                    let source_index = usize::from(inner_block) * usize::from(column_grid)
                        + usize::from(output_block.block_column);
                    let source = right[source_index];
                    if source.tile != destination_tile {
                        transfers.push(Transfer {
                            source_tile: source.tile,
                            destination_tile,
                            tensor: source.tensor,
                            bytes: right_block_bytes,
                            staging_address: Some(right_exchange_address + slot_offset),
                        });
                    }
                }
            }
            phases.push(Phase::Exchange { transfers });

            let gemm_phase = phases.len();
            let mut gemm_commands = Vec::with_capacity(
                wave_outputs.len()
                    * usize::from(inner_batch_end - inner_batch_start)
                    * (1 + usize::from(config.data_type.expands_weights())),
            );
            if config.data_type.expands_weights() {
                for (wave_tile, _) in wave_outputs.iter().enumerate() {
                    let output_index = wave * usize::from(config.tile_count) + wave_tile;
                    allocations.push(Allocation {
                        tensor: TensorId(expanded_right_tensor_base + output_index),
                        tile: u16::try_from(wave_tile)
                            .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?,
                        address: expanded_right_address,
                        size: expanded_right_block_bytes,
                        live_from: gemm_phase,
                        live_until: gemm_phase,
                        kind: AllocationKind::Home,
                    });
                }
            }
            for inner_block in inner_batch_start..inner_batch_end {
                for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
                    let output_index = wave * usize::from(config.tile_count) + wave_tile;
                    let tile = u16::try_from(wave_tile)
                        .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?;
                    let left_tensor = left[usize::from(output_block.block_row)
                        * usize::from(inner_grid)
                        + usize::from(inner_block)]
                    .tensor;
                    let source_right_tensor = right[usize::from(inner_block)
                        * usize::from(column_grid)
                        + usize::from(output_block.block_column)]
                    .tensor;
                    let right_tensor = if config.data_type.expands_weights() {
                        let expanded_tensor = TensorId(expanded_right_tensor_base + output_index);
                        gemm_commands.push(KernelCommand {
                            tile,
                            output: expanded_tensor,
                            inputs: vec![source_right_tensor, source_right_tensor],
                            arguments: vec![
                                u32::from(config.inner_block_dimension)
                                    * u32::from(config.block_dimension),
                                u32::from(config.data_type.weight_scale() as u8),
                            ],
                            specialization: SpecializationKey {
                                operation: "expand_f8_f143_to_f16".into(),
                                shape: if config.retain_profile_metadata {
                                    vec![
                                        usize::from(config.inner_block_dimension),
                                        usize::from(config.block_dimension),
                                    ]
                                } else {
                                    Vec::new()
                                },
                                worker_count: 6,
                                role: "weight-expansion".into(),
                                alignment: 4,
                            },
                            metadata: if config.retain_profile_metadata {
                                BTreeMap::from([
                                    ("label".into(), "expand FP8 weights to FP16".into()),
                                    ("wave".into(), wave.to_string()),
                                    ("inner_block".into(), inner_block.to_string()),
                                    ("bytes".into(), expanded_right_block_bytes.to_string()),
                                ])
                            } else {
                                BTreeMap::new()
                            },
                        });
                        expanded_tensor
                    } else {
                        source_right_tensor
                    };
                    gemm_commands.push(KernelCommand {
                        tile,
                        output: TensorId(scratch_tensor_base + output_index),
                        inputs: vec![left_tensor, right_tensor],
                        arguments: config
                            .data_type
                            .product_scale()
                            .map(|scale| vec![u32::from((scale as u8) & 0x3f)])
                            .unwrap_or_default(),
                        specialization: SpecializationKey {
                            operation: config
                                .data_type
                                .kernel_operation(inner_block == 0, output_block.rows == base_rows)
                                .into(),
                            shape: vec![
                                usize::from(output_block.rows),
                                usize::from(config.inner_block_dimension),
                                usize::from(config.block_dimension),
                            ],
                            worker_count: 6,
                            role: "blocked-gemm".into(),
                            alignment: 32,
                        },
                        metadata: if config.retain_profile_metadata {
                            BTreeMap::from([
                                (
                                    "label".into(),
                                    format!(
                                        "GEMM block ({}, {}) inner block {}",
                                        output_block.block_row,
                                        output_block.block_column,
                                        inner_block
                                    ),
                                ),
                                ("wave".into(), wave.to_string()),
                                (
                                    "output_block_row".into(),
                                    output_block.block_row.to_string(),
                                ),
                                (
                                    "output_block_column".into(),
                                    output_block.block_column.to_string(),
                                ),
                                ("inner_block".into(), inner_block.to_string()),
                                ("row_start".into(), output_block.row_start.to_string()),
                                ("rows".into(), output_block.rows.to_string()),
                                (
                                    "output_bytes".into(),
                                    (u32::from(output_block.rows)
                                        * u32::from(config.block_dimension)
                                        * output_element_bytes)
                                        .to_string(),
                                ),
                                ("block_dimension".into(), config.block_dimension.to_string()),
                                (
                                    "inner_block_dimension".into(),
                                    config.inner_block_dimension.to_string(),
                                ),
                            ])
                        } else {
                            BTreeMap::new()
                        },
                    });
                }
            }
            phases.push(Phase::Compute {
                op: OpId(gemm_phase),
                commands: gemm_commands.into_iter().map(Arc::new).collect(),
            });
        }
        let evacuation_phase = phases.len();
        let max_transfer_rows = u16::try_from(
            ipu_exchange::MAX_TRANSFER_WORDS * 4
                / (u32::from(config.block_dimension) * output_element_bytes),
        )
        .map_err(|_| CompileError::Graph("GEMM evacuation row limit overflow".into()))?;
        let mut transfers = Vec::new();
        let mut evacuation_chunks = Vec::new();
        for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
            let output_index = wave * usize::from(config.tile_count) + wave_tile;
            let source_tile = u16::try_from(wave_tile)
                .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?;
            let mut row_offset = 0u16;
            for chunk_index in 0..output_block.rows.div_ceil(max_transfer_rows) {
                let rows = (output_block.rows - row_offset).min(max_transfer_rows);
                let byte_offset = u32::from(row_offset)
                    * u32::from(config.block_dimension)
                    * output_element_bytes;
                let bytes =
                    u32::from(rows) * u32::from(config.block_dimension) * output_element_bytes;
                let tensor =
                    TensorId(evacuation_tensor_base + output_index * 2 + usize::from(chunk_index));
                allocations.push(Allocation {
                    tensor,
                    tile: source_tile,
                    address: output_address + byte_offset,
                    size: bytes,
                    live_from: evacuation_phase,
                    live_until: evacuation_phase,
                    kind: AllocationKind::Home,
                });
                allocations.push(Allocation {
                    tensor,
                    tile: output_block.tile,
                    address: output_block.address + byte_offset,
                    size: bytes,
                    live_from: evacuation_phase + 1,
                    live_until: evacuation_phase + 2,
                    kind: AllocationKind::HomeAlias {
                        source: output_block.tensor,
                    },
                });
                transfers.push(Transfer {
                    source_tile,
                    destination_tile: output_block.tile,
                    tensor,
                    bytes,
                    staging_address: Some(left_exchange_address + byte_offset),
                });
                evacuation_chunks.push((output_block, tensor, row_offset, rows, bytes));
                row_offset += rows;
            }
        }
        phases.push(Phase::Exchange { transfers });
        let copy_phase = phases.len();
        let mut copy_commands = Vec::with_capacity(evacuation_chunks.len());
        for (output_block, tensor, row_offset, rows, bytes) in evacuation_chunks {
            let units = bytes / 8;
            copy_commands.push(KernelCommand {
                tile: output_block.tile,
                output: tensor,
                inputs: vec![tensor, tensor],
                arguments: vec![units, units / 6, units % 6],
                specialization: SpecializationKey {
                    operation: "copy_u64".into(),
                    shape: if config.retain_profile_metadata {
                        vec![usize::try_from(units).map_err(|_| {
                            CompileError::Graph("GEMM output copy size overflow".into())
                        })?]
                    } else {
                        Vec::new()
                    },
                    worker_count: 6,
                    role: if config.retain_profile_metadata {
                        format!(
                            "output-wave-{wave}-rows-{row_offset}..{}",
                            row_offset + rows
                        )
                        .into()
                    } else {
                        "gemm-output".into()
                    },
                    alignment: 8,
                },
                metadata: if config.retain_profile_metadata {
                    BTreeMap::from([
                        (
                            "label".into(),
                            format!(
                                "store GEMM block ({}, {}) rows {}..{}",
                                output_block.block_row,
                                output_block.block_column,
                                row_offset,
                                row_offset + rows
                            ),
                        ),
                        ("wave".into(), wave.to_string()),
                        (
                            "output_block_row".into(),
                            output_block.block_row.to_string(),
                        ),
                        (
                            "output_block_column".into(),
                            output_block.block_column.to_string(),
                        ),
                        (
                            "row_start".into(),
                            (output_block.row_start + row_offset).to_string(),
                        ),
                        ("rows".into(), rows.to_string()),
                        ("bytes".into(), bytes.to_string()),
                    ])
                } else {
                    BTreeMap::new()
                },
            });
        }
        phases.push(Phase::Compute {
            op: OpId(copy_phase),
            commands: copy_commands.into_iter().map(Arc::new).collect(),
        });
    }

    Ok(BlockedGemmPlan {
        schedule: Schedule {
            layouts: Vec::new(),
            phases,
            allocations,
            tile_count: config.tile_count,
            peak_sram: BTreeMap::new(),
        },
        left,
        right,
        output,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredExchangeGroup {
    pub source_tile: u16,
    pub destination_tiles: Vec<u16>,
    pub tensor: TensorId,
    pub bytes: u32,
    pub addressing: ExchangeAddressing,
    pub sender: PlanRow,
    pub receivers: Vec<PlanRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeAddressing {
    Relative,
    Absolute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeCost {
    pub launches: u32,
    pub estimated_cycles: u64,
    pub payload_words: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredExchangeEpoch {
    pub groups: Vec<LoweredExchangeGroup>,
    pub tile_rows: BTreeMap<u16, Vec<u32>>,
    pub cost: ExchangeCost,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredExchangePhase {
    pub phase: usize,
    pub epochs: Vec<LoweredExchangeEpoch>,
    pub cost: ExchangeCost,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredComputeCommand {
    pub op: OpId,
    pub phase: usize,
    pub phase_tile_command_index: usize,
    pub command: Arc<KernelCommand>,
    pub output_address: u32,
    pub input_addresses: SmallVec<[u32; 4]>,
}

impl std::ops::Deref for LoweredComputeCommand {
    type Target = KernelCommand;

    fn deref(&self) -> &Self::Target {
        &self.command
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoweredTileStep {
    Exchange {
        phase: usize,
        epoch: usize,
        row: Arc<[u32]>,
    },
    Compute(LoweredComputeCommand),
    IdleCompute {
        op: OpId,
        phase: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredTileProgram {
    pub tile: u16,
    pub steps: Vec<LoweredTileStep>,
}

impl LoweredExchangeEpoch {
    pub fn row_for(&self, tile: u16) -> Vec<u32> {
        self.tile_rows.get(&tile).cloned().unwrap_or_else(|| {
            let mut row = vec![0; ipu_exchange::PLAN_WORDS];
            // The runtime performs the all-tile epoch barrier. Inactive tiles
            // then use the SDK's local non-participation sequence.
            row[0] = SANS_INACTIVE_INSTRUCTION;
            row[1] = SYNC_ANS_INSTRUCTION;
            row[2] = RETURN_M10_INSTRUCTION;
            row
        })
    }
}

impl Schedule {
    /// Verifies that tensors which are live at the same time do not occupy
    /// overlapping SRAM on the same tile.
    pub fn validate_allocations(&self) -> Result<(), CompileError> {
        let mut by_tile = vec![Vec::new(); usize::from(self.tile_count)];
        let mut home_by_location = HashMap::<(TensorId, u16), Vec<usize>>::default();
        for (index, allocation) in self.allocations.iter().enumerate() {
            if allocation.kind.has_home_address() {
                home_by_location
                    .entry((allocation.tensor, allocation.tile))
                    .or_default()
                    .push(index);
            }
        }
        for (index, allocation) in self.allocations.iter().enumerate() {
            let allocations = by_tile
                .get_mut(usize::from(allocation.tile))
                .ok_or_else(|| {
                    CompileError::Memory(format!(
                        "tensor {} is allocated on tile {}, outside the {}-tile schedule",
                        allocation.tensor.0, allocation.tile, self.tile_count
                    ))
                })?;
            allocation
                .address
                .checked_add(allocation.size)
                .ok_or_else(|| {
                    CompileError::Memory(format!(
                        "tensor {} allocation address overflows on tile {}",
                        allocation.tensor.0, allocation.tile
                    ))
                })?;
            if allocation.size != 0 && allocation.live_from < allocation.live_until {
                if let AllocationKind::HomeAlias { source } = allocation.kind {
                    let end = allocation.address + allocation.size;
                    let backed = home_by_location
                        .get(&(source, allocation.tile))
                        .into_iter()
                        .flatten()
                        .map(|&index| &self.allocations[index])
                        .any(|candidate| {
                            candidate.address <= allocation.address
                                && candidate.address.saturating_add(candidate.size) >= end
                                && candidate.live_from <= allocation.live_from
                                && candidate.live_until >= allocation.live_until
                        });
                    if !backed {
                        let candidates = home_by_location
                            .get(&(source, allocation.tile))
                            .into_iter()
                            .flatten()
                            .map(|&index| {
                                let candidate = &self.allocations[index];
                                format!(
                                    "0x{:x}..0x{:x}@{}..{} {:?}",
                                    candidate.address,
                                    candidate.address.saturating_add(candidate.size),
                                    candidate.live_from,
                                    candidate.live_until,
                                    candidate.kind
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        let global_candidates = self
                            .allocations
                            .iter()
                            .filter(|candidate| candidate.tensor == source)
                            .map(|candidate| {
                                format!(
                                    "tile{}:0x{:x}..0x{:x}@{}..{} {:?}",
                                    candidate.tile,
                                    candidate.address,
                                    candidate.address.saturating_add(candidate.size),
                                    candidate.live_from,
                                    candidate.live_until,
                                    candidate.kind
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        return Err(CompileError::Memory(format!(
                            "tensor {} alias 0x{:x}..0x{:x}@{}..{} on tile {} is not backed by source tensor {} (local: {candidates}; global: {global_candidates})",
                            allocation.tensor.0,
                            allocation.address,
                            end,
                            allocation.live_from,
                            allocation.live_until,
                            allocation.tile,
                            source.0
                        )));
                    }
                    continue;
                }
                allocations.push(index);
            }
        }
        for allocations in by_tile {
            let mut events = Vec::with_capacity(allocations.len() * 2);
            for index in allocations {
                let allocation = &self.allocations[index];
                events.push((allocation.live_from, true, index));
                if allocation.live_until != usize::MAX {
                    events.push((allocation.live_until, false, index));
                }
            }
            events.sort_unstable_by_key(|&(phase, starts, _)| (phase, starts));
            let mut active = BTreeMap::<(u32, usize), usize>::new();
            for (_, starts, index) in events {
                let allocation = &self.allocations[index];
                let key = (allocation.address, index);
                if !starts {
                    active.remove(&key);
                    continue;
                }
                let end = allocation.address + allocation.size;
                let previous = active
                    .range(..key)
                    .next_back()
                    .map(|(_, &index)| index)
                    .filter(|&index| {
                        let previous = &self.allocations[index];
                        previous.address + previous.size > allocation.address
                    });
                let next = active
                    .range(key..)
                    .next()
                    .map(|(_, &index)| index)
                    .filter(|&index| self.allocations[index].address < end);
                if let Some(other_index) = previous.or(next) {
                    let other = &self.allocations[other_index];
                    let usage = |tensor| {
                        self.phases
                            .iter()
                            .enumerate()
                            .flat_map(|(phase, entry)| match entry {
                                Phase::Compute { commands, .. } => commands
                                    .iter()
                                    .filter(move |command| {
                                        command.output == tensor || command.inputs.contains(&tensor)
                                    })
                                    .map(move |command| {
                                        format!(
                                            "{}:{}:{}",
                                            phase,
                                            command.specialization.operation,
                                            command.specialization.role
                                        )
                                    })
                                    .take(3)
                                    .collect::<Vec<_>>(),
                                Phase::Exchange { .. } => Vec::new(),
                            })
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    return Err(CompileError::Memory(format!(
                        "live tensors {} ({:?}; {}) and {} ({:?}; {}) overlap on tile {}: 0x{:x}..0x{:x} at phases {}..{} and 0x{:x}..0x{:x} at phases {}..{}",
                        allocation.tensor.0,
                        allocation.kind,
                        usage(allocation.tensor),
                        other.tensor.0,
                        other.kind,
                        usage(other.tensor),
                        allocation.tile,
                        allocation.address,
                        end,
                        allocation.live_from,
                        allocation.live_until,
                        other.address,
                        other.address + other.size,
                        other.live_from,
                        other.live_until,
                    )));
                }
                active.insert(key, index);
            }
        }
        Ok(())
    }

    /// Releases command annotations used only to construct semantic profiles.
    /// Kernel operation identity and invocation operands remain intact.
    pub fn discard_profile_metadata(&mut self, phases: Range<usize>) {
        for phase in &mut self.phases[phases] {
            let Phase::Compute { commands, .. } = phase else {
                continue;
            };
            for command in commands {
                let command = Arc::make_mut(command);
                command.specialization.shape = Vec::new();
                command.specialization.role = Cow::Borrowed("");
                command.metadata = BTreeMap::new();
            }
        }
    }

    pub fn lower_exchanges(
        &self,
        topology: &Topology,
    ) -> Result<Vec<LoweredExchangePhase>, CompileError> {
        let allocation_index = AllocationIndex::new(&self.allocations);
        self.lower_exchanges_with_index(topology, &allocation_index)
    }

    fn lower_exchanges_with_index(
        &self,
        topology: &Topology,
        allocation_index: &AllocationIndex<'_>,
    ) -> Result<Vec<LoweredExchangePhase>, CompileError> {
        if topology.tile_count() < usize::from(self.tile_count) {
            return Err(CompileError::Graph(
                "exchange topology has too few tiles".into(),
            ));
        }
        #[derive(Clone)]
        struct PendingGroup {
            source: u16,
            tensor: TensorId,
            bytes: u32,
            destinations: Vec<(u16, Option<u32>)>,
        }

        let mut staging_additions = vec![Vec::new(); self.phases.len() + 1];
        let mut staging_removals = vec![Vec::new(); self.phases.len() + 1];
        for allocation in &self.allocations {
            let AllocationKind::ExchangeStaging { phase } = allocation.kind else {
                continue;
            };
            let available_from = allocation.live_from.max(phase.saturating_add(1));
            if available_from >= allocation.live_until || available_from >= self.phases.len() {
                continue;
            }
            let location = (allocation.tensor, allocation.tile);
            staging_additions[available_from].push(location);
            if allocation.live_until < staging_removals.len() {
                staging_removals[allocation.live_until].push(location);
            }
        }
        let mut available_staging = HashSet::<(TensorId, u16)>::default();
        let mut available_staging_counts = HashMap::<(TensorId, u16), usize>::default();

        let mut lowered_phases = Vec::new();
        for (phase_index, phase) in self.phases.iter().enumerate() {
            for location in &staging_removals[phase_index] {
                let count = available_staging_counts.get_mut(location).ok_or_else(|| {
                    CompileError::Graph("staging lifetime removal underflow".into())
                })?;
                *count -= 1;
                if *count == 0 {
                    available_staging_counts.remove(location);
                    available_staging.remove(location);
                }
            }
            for &location in &staging_additions[phase_index] {
                *available_staging_counts.entry(location).or_default() += 1;
                available_staging.insert(location);
            }
            let Phase::Exchange { transfers } = phase else {
                continue;
            };
            validate_transfers(transfers)?;
            let direct_staging = transfers
                .iter()
                .filter_map(|transfer| {
                    transfer
                        .staging_address
                        .map(|address| ((transfer.tensor, transfer.destination_tile), address))
                })
                .collect::<HashMap<_, _>>();
            let mut groups: Vec<PendingGroup> = Vec::new();
            let mut group_indices = HashMap::<(u16, TensorId, u32), usize>::default();
            for transfer in transfers {
                let key = (transfer.source_tile, transfer.tensor, transfer.bytes);
                if let Some(&index) = group_indices.get(&key) {
                    groups[index]
                        .destinations
                        .push((transfer.destination_tile, transfer.staging_address));
                } else {
                    group_indices.insert(key, groups.len());
                    groups.push(PendingGroup {
                        source: transfer.source_tile,
                        tensor: transfer.tensor,
                        bytes: transfer.bytes,
                        destinations: vec![(transfer.destination_tile, transfer.staging_address)],
                    });
                }
            }
            for group in &mut groups {
                group.destinations.sort_unstable_by_key(|&(tile, _)| tile);
                group.destinations.dedup_by(|right, left| {
                    if right.0 != left.0 {
                        return false;
                    }
                    debug_assert_eq!(right.1, left.1);
                    true
                });
            }

            // A tile can execute one exchange role at a time. Incremental
            // DSATUR preserves the compact static schedule without rescanning
            // every uncolored group after each assignment.
            let mut groups_by_tile = vec![Vec::new(); topology.tile_count()];
            for (group_index, group) in groups.iter().enumerate() {
                groups_by_tile[usize::from(group.source)].push(group_index);
                for &(destination, _) in &group.destinations {
                    groups_by_tile[usize::from(destination)].push(group_index);
                }
            }
            let mut adjacency = vec![HashSet::default(); groups.len()];
            for tile_groups in groups_by_tile {
                for (offset, &left) in tile_groups.iter().enumerate() {
                    for &right in &tile_groups[offset + 1..] {
                        adjacency[left].insert(right);
                        adjacency[right].insert(left);
                    }
                }
            }
            let mut colors = vec![None; groups.len()];
            let mut saturation = vec![HashSet::default(); groups.len()];
            let key = |index: usize, saturation: &[HashSet<usize>]| {
                (
                    saturation[index].len(),
                    adjacency[index].len(),
                    std::cmp::Reverse(groups[index].source),
                    std::cmp::Reverse(groups[index].tensor.0),
                    index,
                )
            };
            let mut queue = (0..groups.len())
                .map(|index| key(index, &saturation))
                .collect::<BinaryHeap<_>>();
            for _ in 0..groups.len() {
                let index = loop {
                    let candidate = queue
                        .pop()
                        .ok_or_else(|| CompileError::Graph("exchange coloring failed".into()))?;
                    let index = candidate.4;
                    if colors[index].is_none() && candidate == key(index, &saturation) {
                        break index;
                    }
                };
                let color = (0..)
                    .find(|color| !saturation[index].contains(color))
                    .ok_or_else(|| CompileError::Graph("exchange color overflow".into()))?;
                colors[index] = Some(color);
                for &neighbor in &adjacency[index] {
                    if colors[neighbor].is_none() && saturation[neighbor].insert(color) {
                        queue.push(key(neighbor, &saturation));
                    }
                }
            }
            let color_count = colors
                .iter()
                .filter_map(|color| *color)
                .max()
                .map_or(0, |color| color + 1);
            let mut colored_groups = vec![Vec::new(); color_count];
            for (group, color) in groups.into_iter().zip(colors) {
                let color =
                    color.ok_or_else(|| CompileError::Graph("uncolored exchange group".into()))?;
                colored_groups[color].push(group);
            }
            let mut available = available_staging.clone();
            let available_before_phase = available.clone();
            let location_available = |available: &HashSet<_>, tensor, tile| {
                available.contains(&(tensor, tile))
                    || allocation_index
                        .at(tensor, tile)
                        .any(|allocation| allocation.kind.has_home_address())
            };
            let mut epoch_groups = Vec::with_capacity(colored_groups.len());
            while !colored_groups.is_empty() {
                let ready = colored_groups
                    .iter()
                    .position(|slot| {
                        slot.iter().all(|group| {
                            location_available(&available, group.tensor, group.source)
                        })
                    })
                    .ok_or_else(|| {
                        let blocked = colored_groups
                            .iter()
                            .flat_map(|slot| slot.iter())
                            .filter(|group| {
                                !location_available(&available, group.tensor, group.source)
                            })
                            .map(|group| {
                                format!("tensor {} on tile {}", group.tensor.0, group.source)
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        CompileError::Graph(format!(
                            "exchange phase {phase_index} staging dependencies contain a cycle or missing source: {blocked}"
                        ))
                    })?;
                let slot = colored_groups.remove(ready);
                for group in &slot {
                    available.extend(
                        group
                            .destinations
                            .iter()
                            .map(|&(destination, _)| (group.tensor, destination)),
                    );
                }
                epoch_groups.push(slot);
            }
            let mut lowered_groups = Vec::new();
            let mut builders = BTreeMap::<u16, PlanProgramBuilder>::new();
            let mut cost = ExchangeCost {
                launches: u32::from(!epoch_groups.is_empty()),
                ..ExchangeCost::default()
            };
            let mut horizon = 0u32;
            for pending in epoch_groups {
                let schedule_offset = if horizon == 0 { 0 } else { horizon + 1 };
                let mut slot_horizon = horizon;
                for PendingGroup {
                    source,
                    tensor,
                    bytes,
                    destinations: destination_entries,
                } in pending
                {
                    let destinations = destination_entries
                        .iter()
                        .map(|&(tile, _)| tile)
                        .collect::<Vec<_>>();
                    if bytes == 0 || bytes & 3 != 0 {
                        return Err(CompileError::Graph(format!(
                            "tensor {} exchange size is not whole words",
                            tensor.0
                        )));
                    }
                    let words = bytes / 4;
                    let candidates = allocation_index.at(tensor, source);
                    let same_phase_staging = || {
                        candidates.clone().find(|allocation| {
                            allocation.kind
                                == AllocationKind::ExchangeStaging { phase: phase_index }
                        })
                    };
                    let earlier_staging = || {
                        candidates.clone().find(|allocation| {
                            matches!(
                                allocation.kind,
                                AllocationKind::ExchangeStaging { phase }
                                    if phase < phase_index
                            ) && allocation.live_from <= phase_index
                                && allocation.live_until > phase_index
                        })
                    };
                    let home = || {
                        candidates
                            .clone()
                            .find(|allocation| allocation.kind.has_home_address())
                    };
                    let direct_same_phase = direct_staging.get(&(tensor, source)).copied();
                    let source_address =
                        if location_available(&available_before_phase, tensor, source) {
                            earlier_staging()
                                .or_else(home)
                                .or_else(same_phase_staging)
                                .map(|allocation| allocation.address)
                                .or(direct_same_phase)
                        } else {
                            direct_same_phase.or_else(|| {
                                same_phase_staging()
                                    .or_else(earlier_staging)
                                    .or_else(home)
                                    .map(|allocation| allocation.address)
                            })
                        }
                        .ok_or_else(|| {
                            CompileError::Memory(format!(
                                "missing source allocation for tensor {} on tile {source}",
                                tensor.0
                            ))
                        })?;
                    let destination_addresses = destination_entries
                        .iter()
                        .map(|&(destination, direct_address)| {
                            direct_address
                                .or_else(|| {
                                    allocation_index
                                        .at(tensor, destination)
                                        .find(|allocation| {
                                            allocation.kind
                                                == AllocationKind::ExchangeStaging {
                                                    phase: phase_index,
                                                }
                                        })
                                        .map(|allocation| allocation.address)
                                })
                                .ok_or_else(|| {
                                    CompileError::Memory(format!(
                                        "missing staging address for tensor {} on tile {destination}",
                                        tensor.0
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut plan: MulticastPlan = if destinations.len() == 1 && schedule_offset == 0
                    {
                        let point = topology.point_to_point(source, destinations[0], words)?;
                        MulticastPlan {
                            sender: point.sender,
                            receivers: vec![finalize_point_receiver(
                                &point.receiver,
                                topology.physical(source)?,
                            )?],
                        }
                    } else {
                        let mut plan = topology.multicast(source, &destinations, words, 0)?;
                        ipu_exchange::offset_plan(&mut plan.sender, schedule_offset)?;
                        for receiver in &mut plan.receivers {
                            ipu_exchange::offset_plan(receiver, schedule_offset)?;
                        }
                        plan
                    };
                    patch_sender_address(&mut plan.sender, source_address)?;
                    for (receiver, address) in plan
                        .receivers
                        .iter_mut()
                        .zip(destination_addresses.iter().copied())
                    {
                        patch_receiver_address(receiver, address)?;
                    }
                    let sender = plan.sender;
                    let receivers = plan.receivers;
                    debug!(
                        source,
                        destinations = ?destinations,
                        destination_addresses = ?destination_addresses,
                        sender = ?sender,
                        receivers = ?receivers,
                        "lowered multicast rows"
                    );
                    builders
                        .entry(source)
                        .or_default()
                        .append_scheduled_row(&sender)?;
                    for (destination, receiver) in
                        destinations.iter().copied().zip(receivers.iter().copied())
                    {
                        builders
                            .entry(destination)
                            .or_default()
                            .append_scheduled_row(&receiver)?;
                    }
                    let group_cycles = std::iter::once(&sender)
                        .chain(receivers.iter())
                        .map(|row| ipu_exchange::plan_event_cycles(row))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .max()
                        .unwrap_or(0);
                    slot_horizon = slot_horizon.max(group_cycles);
                    cost.payload_words += u64::from(words);
                    lowered_groups.push(LoweredExchangeGroup {
                        source_tile: source,
                        destination_tiles: destinations,
                        tensor,
                        bytes,
                        addressing: ExchangeAddressing::Absolute,
                        sender,
                        receivers,
                    });
                }
                horizon = slot_horizon;
            }
            cost.estimated_cycles = u64::from(horizon);
            let tile_rows = builders
                .into_iter()
                .map(|(tile, builder)| Ok((tile, builder.finish(horizon)?)))
                .collect::<Result<BTreeMap<_, _>, CompileError>>()?;
            debug!(horizon, tile_rows = ?tile_rows, "composed exchange programs");
            let epochs = if lowered_groups.is_empty() {
                Vec::new()
            } else {
                vec![LoweredExchangeEpoch {
                    groups: lowered_groups,
                    tile_rows,
                    cost,
                }]
            };
            let phase_cost = cost;
            lowered_phases.push(LoweredExchangePhase {
                phase: phase_index,
                epochs,
                cost: phase_cost,
            });
        }
        let launches: u32 = lowered_phases.iter().map(|phase| phase.cost.launches).sum();
        info!(
            phases = lowered_phases.len(),
            launches, "lowered exchange schedule"
        );
        Ok(lowered_phases)
    }

    pub fn lower_tile_programs(
        &self,
        topology: &Topology,
    ) -> Result<Vec<LoweredTileProgram>, CompileError> {
        self.lower_tile_programs_impl(topology, true)
    }

    /// Lowers executable tile programs without materializing inactive compute
    /// phases. Those phases emit no device instructions and are only needed by
    /// the profiling representation.
    pub fn lower_tile_programs_for_codegen(
        &self,
        topology: &Topology,
    ) -> Result<Vec<LoweredTileProgram>, CompileError> {
        self.lower_tile_programs_impl(topology, false)
    }

    fn lower_tile_programs_impl(
        &self,
        topology: &Topology,
        include_idle_compute: bool,
    ) -> Result<Vec<LoweredTileProgram>, CompileError> {
        let lowering = self.prepare_tile_program_lowering(topology)?;
        (0..self.tile_count)
            .into_par_iter()
            .map(|tile| lowering.lower(tile, include_idle_compute))
            .collect()
    }

    pub fn prepare_tile_program_lowering(
        &self,
        topology: &Topology,
    ) -> Result<TileProgramLowering<'_>, CompileError> {
        let allocation_index = AllocationIndex::new(&self.allocations);
        let exchanges = self.lower_exchanges_with_index(topology, &allocation_index)?;
        let mut exchange_by_phase = vec![None; self.phases.len()];
        for (index, exchange) in exchanges.iter().enumerate() {
            exchange_by_phase[exchange.phase] = Some(index);
        }
        let mut commands_by_tile = vec![Vec::new(); usize::from(self.tile_count)];
        for (phase, scheduled) in self.phases.iter().enumerate() {
            let Phase::Compute { commands, .. } = scheduled else {
                continue;
            };
            let mut tile_command_counts = vec![0usize; usize::from(self.tile_count)];
            for command in commands {
                let tile = usize::from(command.tile);
                let phase_tile_command_index = tile_command_counts[tile];
                tile_command_counts[tile] += 1;
                commands_by_tile[usize::from(command.tile)].push((
                    phase,
                    phase_tile_command_index,
                    command.clone(),
                ));
            }
        }
        let mut inactive_row = vec![0; ipu_exchange::PLAN_WORDS];
        inactive_row[0] = SANS_INACTIVE_INSTRUCTION;
        inactive_row[1] = SYNC_ANS_INSTRUCTION;
        inactive_row[2] = RETURN_M10_INSTRUCTION;
        Ok(TileProgramLowering {
            schedule: self,
            allocation_index,
            exchanges,
            exchange_by_phase,
            commands_by_tile,
            inactive_row: inactive_row.into(),
        })
    }
}

pub struct TileProgramLowering<'a> {
    schedule: &'a Schedule,
    allocation_index: AllocationIndex<'a>,
    exchanges: Vec<LoweredExchangePhase>,
    exchange_by_phase: Vec<Option<usize>>,
    commands_by_tile: Vec<Vec<(usize, usize, Arc<KernelCommand>)>>,
    inactive_row: Arc<[u32]>,
}

impl TileProgramLowering<'_> {
    pub fn lower(
        &self,
        tile: u16,
        include_idle_compute: bool,
    ) -> Result<LoweredTileProgram, CompileError> {
        if tile >= self.schedule.tile_count {
            return Err(CompileError::Graph(format!(
                "tile {tile} exceeds tile count {}",
                self.schedule.tile_count
            )));
        }
        let mut program = LoweredTileProgram {
            tile,
            steps: Vec::new(),
        };
        let tile_commands = &self.commands_by_tile[usize::from(tile)];
        let mut command_cursor = 0usize;
        let mut direct_staging = HashMap::<(TensorId, u16), u32>::default();
        for (phase_index, phase) in self.schedule.phases.iter().enumerate() {
            match phase {
                Phase::Exchange { transfers } => {
                    direct_staging.clear();
                    direct_staging.extend(transfers.iter().filter_map(|transfer| {
                        transfer
                            .staging_address
                            .map(|address| ((transfer.tensor, transfer.destination_tile), address))
                    }));
                    let exchange = self
                        .exchange_by_phase
                        .get(phase_index)
                        .and_then(|index| index.map(|index| &self.exchanges[index]))
                        .ok_or_else(|| {
                            CompileError::Graph(format!(
                                "missing lowered exchange phase {phase_index}"
                            ))
                        })?;
                    for (epoch, lowered) in exchange.epochs.iter().enumerate() {
                        program.steps.push(LoweredTileStep::Exchange {
                            phase: phase_index,
                            epoch,
                            row: lowered
                                .tile_rows
                                .get(&tile)
                                .map(|row| Arc::<[u32]>::from(row.as_slice()))
                                .unwrap_or_else(|| self.inactive_row.clone()),
                        });
                    }
                }
                Phase::Compute { op, .. } => {
                    let mut active = false;
                    while command_cursor < tile_commands.len()
                        && tile_commands[command_cursor].0 == phase_index
                    {
                        let phase_tile_command_index = tile_commands[command_cursor].1;
                        let command = &tile_commands[command_cursor].2;
                        command_cursor += 1;
                        active = true;
                        let output_address =
                            self.allocation_index.home_address(command.output, tile)?;
                        let input_addresses = command
                            .inputs
                            .iter()
                            .map(|input| {
                                direct_staging
                                    .get(&(*input, tile))
                                    .copied()
                                    .map(Ok)
                                    .unwrap_or_else(|| {
                                        self.allocation_index.compute_input_address(
                                            *input,
                                            tile,
                                            phase_index,
                                        )
                                    })
                            })
                            .collect::<Result<_, _>>()?;
                        program
                            .steps
                            .push(LoweredTileStep::Compute(LoweredComputeCommand {
                                op: *op,
                                phase: phase_index,
                                phase_tile_command_index,
                                command: command.clone(),
                                output_address,
                                input_addresses,
                            }));
                    }
                    if include_idle_compute && !active {
                        program.steps.push(LoweredTileStep::IdleCompute {
                            op: *op,
                            phase: phase_index,
                        });
                    }
                    direct_staging.clear();
                }
            }
        }
        Ok(program)
    }
}

struct AllocationIndex<'a> {
    allocations: &'a [Allocation],
    by_location: HashMap<(TensorId, u16), usize>,
    next: Vec<usize>,
}

impl<'a> AllocationIndex<'a> {
    fn new(allocations: &'a [Allocation]) -> Self {
        let mut by_location = HashMap::default();
        let mut next = vec![usize::MAX; allocations.len()];
        // Build backwards so iteration retains the original allocation order.
        for (index, allocation) in allocations.iter().enumerate().rev() {
            if let Some(successor) = by_location.insert((allocation.tensor, allocation.tile), index)
            {
                next[index] = successor;
            }
        }
        Self {
            allocations,
            by_location,
            next,
        }
    }

    fn at(&self, tensor: TensorId, tile: u16) -> AllocationCandidates<'_> {
        AllocationCandidates {
            allocations: self.allocations,
            next: &self.next,
            current: self
                .by_location
                .get(&(tensor, tile))
                .copied()
                .unwrap_or(usize::MAX),
        }
    }

    fn home_address(&self, tensor: TensorId, tile: u16) -> Result<u32, CompileError> {
        self.at(tensor, tile)
            .find(|allocation| allocation.kind.has_home_address())
            .map(|allocation| allocation.address)
            .ok_or_else(|| {
                CompileError::Memory(format!(
                    "missing home allocation for tensor {} on tile {tile}",
                    tensor.0
                ))
            })
    }

    fn compute_input_address(
        &self,
        tensor: TensorId,
        tile: u16,
        compute_phase: usize,
    ) -> Result<u32, CompileError> {
        if let Some(staging) = self.at(tensor, tile).find(|allocation| {
            allocation.live_from < compute_phase
                && allocation.live_until >= compute_phase
                && matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
        }) {
            return Ok(staging.address);
        }
        self.home_address(tensor, tile)
    }
}

#[derive(Clone)]
struct AllocationCandidates<'a> {
    allocations: &'a [Allocation],
    next: &'a [usize],
    current: usize,
}

impl<'a> Iterator for AllocationCandidates<'a> {
    type Item = &'a Allocation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == usize::MAX {
            return None;
        }
        let index = self.current;
        self.current = self.next[index];
        Some(&self.allocations[index])
    }
}

#[derive(Clone, Debug)]
pub struct CompilerOptions {
    pub tile_count: u16,
    pub exchange_base: u32,
    pub exchange_limit: u32,
    pub data_arenas: Vec<MemoryArena>,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            tile_count: DEFAULT_TILE_COUNT,
            exchange_base: 0x50000,
            exchange_limit: 0x58000,
            data_arenas: vec![
                MemoryArena::high(0x58000, ipu_package::IPU21_INTERLEAVED_MEMORY_BASE),
                MemoryArena::low(ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT, 0xe0000),
            ],
        }
    }
}

pub fn compile(graph: &Graph, options: &CompilerOptions) -> Result<Schedule, CompileError> {
    info!(
        operations = graph.ops.len(),
        tensors = graph.tensors.len(),
        tiles = options.tile_count,
        "compiling graph"
    );
    graph.validate()?;
    if options.tile_count == 0
        || options.exchange_base < ipu_exchange::EXCHANGE_WINDOW_BASE
        || options.exchange_limit > ipu_exchange::EXCHANGE_WINDOW_BASE + 0x8000
        || options.exchange_base >= options.exchange_limit
        || options.data_arenas.is_empty()
        || options
            .data_arenas
            .iter()
            .any(|arena| arena.base >= arena.limit)
        || options
            .data_arenas
            .windows(2)
            .any(|arenas| arenas[0].base >= arenas[1].base || arenas[0].limit > arenas[1].base)
        || options
            .data_arenas
            .iter()
            .any(|arena| arena.base < options.exchange_limit)
    {
        return Err(CompileError::Graph("invalid compiler options".into()));
    }
    let layouts: Vec<_> = graph
        .tensors
        .iter()
        .map(|tensor| choose_layout(tensor, options.tile_count))
        .collect();
    let mut phases = Vec::new();
    for op in &graph.ops {
        let output_layout = &layouts[op.output.0];
        let mut transfers = Vec::new();
        for input in &op.inputs {
            let input_layout = &layouts[input.0];
            if input_layout.tiles != output_layout.tiles
                || input_layout.sharding != output_layout.sharding
            {
                for (index, destination) in output_layout.tiles.iter().enumerate() {
                    let source = input_layout.tiles[index % input_layout.tiles.len()];
                    if source != *destination {
                        transfers.push(Transfer {
                            source_tile: source,
                            destination_tile: *destination,
                            tensor: *input,
                            bytes: local_bytes(&graph.tensors[input.0], input_layout) as u32,
                            staging_address: None,
                        });
                    }
                }
            }
        }
        if !transfers.is_empty() {
            validate_transfers(&transfers)?;
            phases.push(Phase::Exchange { transfers });
        }
        let operation = operation_name(&op.kind);
        let commands = output_layout
            .tiles
            .iter()
            .map(|tile| {
                Arc::new(KernelCommand {
                    tile: *tile,
                    output: op.output,
                    inputs: op.inputs.clone(),
                    arguments: Vec::new(),
                    specialization: SpecializationKey {
                        operation: operation.into(),
                        shape: graph.tensors[op.output.0].shape.clone(),
                        worker_count: worker_count(&graph.tensors[op.output.0]),
                        role: if output_layout.tiles.last() == Some(tile) {
                            "tail".into()
                        } else {
                            "body".into()
                        },
                        alignment: output_layout.alignment,
                    },
                    metadata: BTreeMap::new(),
                })
            })
            .collect();
        phases.push(Phase::Compute {
            op: op.id,
            commands,
        });
    }
    let allocations = plan_memory(graph, &layouts, &phases, options)?;
    let mut peak_sram: BTreeMap<u16, u32> = BTreeMap::new();
    for allocation in &allocations {
        let memory_base = options
            .data_arenas
            .first()
            .map_or(options.exchange_base, |arena| arena.base)
            .min(options.exchange_base);
        peak_sram
            .entry(allocation.tile)
            .and_modify(|peak| {
                *peak = (*peak).max(allocation.address + allocation.size - memory_base)
            })
            .or_insert(allocation.address + allocation.size - memory_base);
    }
    let schedule = Schedule {
        layouts,
        phases,
        allocations,
        tile_count: options.tile_count,
        peak_sram,
    };
    let peak_bytes = schedule.peak_sram.values().copied().max().unwrap_or(0);
    debug!(peak_bytes, "planned tile SRAM");
    info!(
        phases = schedule.phases.len(),
        allocations = schedule.allocations.len(),
        peak_bytes,
        "graph compilation completed"
    );
    Ok(schedule)
}

fn choose_layout(tensor: &Tensor, tile_count: u16) -> Layout {
    let useful =
        ((tensor.elements() * tensor.dtype.size()).div_ceil(1024) as u16).clamp(1, tile_count);
    let tiles = (0..useful).collect();
    let sharding = match tensor.kind {
        TensorKind::Weight if tensor.shape.len() == 2 => Sharding::Columns,
        _ if tensor.shape.len() >= 3 => Sharding::Heads,
        _ if tensor.shape.len() == 2 => Sharding::Rows,
        _ => Sharding::Replicated,
    };
    Layout {
        tensor: tensor.id,
        tiles,
        sharding,
        alignment: 16,
    }
}

fn local_bytes(tensor: &Tensor, layout: &Layout) -> usize {
    (tensor.elements() * tensor.dtype.size()).div_ceil(layout.tiles.len())
}

fn operation_name(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::MatMul => "matmul",
        OpKind::Add => "add",
        OpKind::Mul => "mul",
        OpKind::Reshape { .. } => "reshape",
        OpKind::Transpose { .. } => "transpose",
        OpKind::LayerNorm { .. } => "layer_norm",
        OpKind::Softmax { .. } => "softmax",
        OpKind::Gelu => "gelu",
    }
}

fn worker_count(tensor: &Tensor) -> u8 {
    if tensor.elements() >= 96 { 6 } else { 1 }
}

fn validate_transfers(transfers: &[Transfer]) -> Result<(), CompileError> {
    let mut destinations = HashSet::default();
    let mut direct_ranges = Vec::new();
    for transfer in transfers {
        if transfer.source_tile == transfer.destination_tile || transfer.bytes == 0 {
            return Err(CompileError::Graph("invalid exchange transfer".into()));
        }
        if transfer
            .staging_address
            .is_some_and(|address| address & 3 != 0)
        {
            return Err(CompileError::Graph(
                "exchange staging address is not word aligned".into(),
            ));
        }
        if let Some(address) = transfer.staging_address {
            let end = address
                .checked_add(transfer.bytes)
                .ok_or_else(|| CompileError::Memory("exchange staging address overflow".into()))?;
            direct_ranges.push((transfer.destination_tile, address, end));
        }
        if !destinations.insert((transfer.destination_tile, transfer.tensor)) {
            return Err(CompileError::Graph(
                "multiple sends target one tensor region in an epoch".into(),
            ));
        }
    }
    direct_ranges.sort_unstable();
    if direct_ranges
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[0].2 > pair[1].1)
    {
        return Err(CompileError::Memory(
            "direct exchange staging regions overlap on one tile".into(),
        ));
    }
    Ok(())
}

fn plan_memory(
    graph: &Graph,
    layouts: &[Layout],
    phases: &[Phase],
    options: &CompilerOptions,
) -> Result<Vec<Allocation>, CompileError> {
    let mut produced = vec![0usize; graph.tensors.len()];
    let mut consumed = vec![0usize; graph.tensors.len()];
    for (phase_index, phase) in phases.iter().enumerate() {
        match phase {
            Phase::Compute { op, .. } => {
                let operation = &graph.ops[op.0];
                produced[operation.output.0] = phase_index;
                for input in &operation.inputs {
                    consumed[input.0] = consumed[input.0].max(phase_index + 1);
                }
            }
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    consumed[transfer.tensor.0] = consumed[transfer.tensor.0].max(phase_index + 1);
                }
            }
        }
    }
    for output in &graph.outputs {
        consumed[output.0] = phases.len();
    }

    let mut allocations = Vec::new();
    let mut by_tile: HashMap<u16, Vec<Allocation>> = HashMap::default();
    for tensor in &graph.tensors {
        let layout = &layouts[tensor.id.0];
        let size = align_u32(local_bytes(tensor, layout) as u32, layout.alignment);
        for tile in &layout.tiles {
            allocate_region(
                &mut allocations,
                &mut by_tile,
                tensor.id,
                *tile,
                size,
                produced[tensor.id.0],
                consumed[tensor.id.0],
                layout.alignment,
                AllocationKind::Home,
                &options.data_arenas,
                &tensor.name,
            )?;
        }
    }
    for (phase_index, phase) in phases.iter().enumerate() {
        let Phase::Exchange { transfers } = phase else {
            continue;
        };
        for transfer in transfers {
            if allocations.iter().any(|allocation| {
                allocation.tensor == transfer.tensor
                    && allocation.tile == transfer.destination_tile
                    && allocation.kind == AllocationKind::ExchangeStaging { phase: phase_index }
            }) {
                continue;
            }
            allocate_region(
                &mut allocations,
                &mut by_tile,
                transfer.tensor,
                transfer.destination_tile,
                align_u32(transfer.bytes, 16),
                phase_index,
                phase_index + 1,
                16,
                AllocationKind::ExchangeStaging { phase: phase_index },
                &[MemoryArena::low(
                    options.exchange_base,
                    options.exchange_limit,
                )],
                &format!("tensor {} exchange staging", transfer.tensor.0),
            )?;
        }
    }
    Ok(allocations)
}

#[allow(clippy::too_many_arguments)]
fn allocate_region(
    allocations: &mut Vec<Allocation>,
    by_tile: &mut HashMap<u16, Vec<Allocation>>,
    tensor: TensorId,
    tile: u16,
    size: u32,
    live_from: usize,
    live_until: usize,
    alignment: u32,
    kind: AllocationKind,
    arenas: &[MemoryArena],
    label: &str,
) -> Result<(), CompileError> {
    let existing = by_tile.entry(tile).or_default();
    for arena in arenas {
        let mut address = align_u32(arena.base, alignment);
        while address < arena.limit {
            let end = address
                .checked_add(size)
                .ok_or_else(|| CompileError::Memory("address overflow".into()))?;
            if end > arena.limit {
                break;
            }
            let conflict = existing
                .iter()
                .filter(|allocation| {
                    lifetimes_overlap(
                        live_from,
                        live_until,
                        allocation.live_from,
                        allocation.live_until,
                    )
                })
                .find(|allocation| {
                    address < allocation.address + allocation.size && allocation.address < end
                });
            if let Some(conflict) = conflict {
                address = align_u32(conflict.address + conflict.size, alignment);
                continue;
            }
            let allocation = Allocation {
                tensor,
                tile,
                address,
                size,
                live_from,
                live_until,
                kind,
            };
            existing.push(allocation.clone());
            allocations.push(allocation);
            return Ok(());
        }
    }
    Err(CompileError::Memory(format!(
        "tile {tile} has no arena for {size} bytes allocating {label}"
    )))
}

fn lifetimes_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn align_u32(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

const REBLOCK_ROW_BITS: u32 = 10;
const REBLOCK_ROW_LIMIT: u16 = 1 << REBLOCK_ROW_BITS;

pub(crate) fn pack_reblock_row_pair(first: u16, second: u16) -> Result<u32, CompileError> {
    if first >= REBLOCK_ROW_LIMIT || second >= REBLOCK_ROW_LIMIT {
        return Err(CompileError::Graph(format!(
            "row reblocking supports per-shard dimensions and offsets below {REBLOCK_ROW_LIMIT}"
        )));
    }
    Ok(u32::from(first) | (u32::from(second) << REBLOCK_ROW_BITS))
}

fn pack_a16_reblock_count(
    copy_rows: u16,
    columns: u16,
    input_scale: Option<i8>,
) -> Result<u32, CompileError> {
    if copy_rows >= REBLOCK_ROW_LIMIT {
        return Err(CompileError::Graph(format!(
            "row reblocking supports copy counts below {REBLOCK_ROW_LIMIT}"
        )));
    }
    let column_group = if input_scale.is_some() { 32 } else { 16 };
    if columns == 0 || !columns.is_multiple_of(column_group) {
        return Err(CompileError::Graph(format!(
            "A16 reblocking requires columns divisible by {column_group}"
        )));
    }
    let groups = columns / column_group;
    Ok(u32::from(copy_rows)
        | (u32::from(input_scale.map_or(0, |scale| (scale as u8) & 0x3f)) << 10)
        | (u32::from(groups) << 16))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderConfig {
    pub sequence: usize,
    pub hidden: usize,
    pub heads: usize,
    pub feed_forward: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            sequence: 32,
            hidden: 64,
            heads: 4,
            feed_forward: 256,
        }
    }
}

pub fn encoder_graph(config: EncoderConfig) -> Result<Graph, CompileError> {
    if config.hidden == 0
        || config.sequence == 0
        || config.heads == 0
        || !config.hidden.is_multiple_of(config.heads)
    {
        return Err(CompileError::Graph("invalid encoder dimensions".into()));
    }
    let mut graph = Graph::default();
    let input = graph.input("input", &[config.sequence, config.hidden]);
    let wq = graph.weight("wq", &[config.hidden, config.hidden]);
    let wk = graph.weight("wk", &[config.hidden, config.hidden]);
    let wv = graph.weight("wv", &[config.hidden, config.hidden]);
    let wo = graph.weight("wo", &[config.hidden, config.hidden]);
    let w1 = graph.weight("w1", &[config.hidden, config.feed_forward]);
    let w2 = graph.weight("w2", &[config.feed_forward, config.hidden]);
    let norm1 = graph.layer_norm("norm1", input, 1e-5)?;
    let q = graph.matmul("q", norm1, wq)?;
    let k = graph.matmul("k", norm1, wk)?;
    let v = graph.matmul("v", norm1, wv)?;
    let kt = graph.transpose("k_transpose", k, &[1, 0])?;
    let scores = graph.matmul("attention_scores", q, kt)?;
    let probabilities = graph.softmax("attention_softmax", scores, 1)?;
    let context = graph.matmul("attention_context", probabilities, v)?;
    let projected = graph.matmul("attention_projection", context, wo)?;
    let residual1 = graph.add("attention_residual", input, projected)?;
    let norm2 = graph.layer_norm("norm2", residual1, 1e-5)?;
    let hidden = graph.matmul("ffn_up", norm2, w1)?;
    let activated = graph.gelu("ffn_gelu", hidden)?;
    let down = graph.matmul("ffn_down", activated, w2)?;
    let output = graph.add("ffn_residual", residual1, down)?;
    graph.mark_output(output);
    Ok(graph)
}

#[derive(Clone, Debug)]
pub struct EncoderWeights {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
}

impl EncoderWeights {
    pub fn deterministic(config: EncoderConfig) -> Self {
        let matrix = |rows: usize, columns: usize, salt: usize| {
            (0..rows * columns)
                .map(|index| {
                    (((index * 17 + salt * 31) % 101) as f32 - 50.0) / (50.0 * rows as f32).sqrt()
                })
                .collect()
        };
        Self {
            wq: matrix(config.hidden, config.hidden, 1),
            wk: matrix(config.hidden, config.hidden, 2),
            wv: matrix(config.hidden, config.hidden, 3),
            wo: matrix(config.hidden, config.hidden, 4),
            w1: matrix(config.hidden, config.feed_forward, 5),
            w2: matrix(config.feed_forward, config.hidden, 6),
        }
    }
}

pub fn encoder_reference(
    config: EncoderConfig,
    input: &[f32],
    weights: &EncoderWeights,
) -> Result<Vec<f32>, CompileError> {
    if input.len() != config.sequence * config.hidden {
        return Err(CompileError::Graph("encoder input size mismatch".into()));
    }
    let norm1 = layer_norm(input, config.sequence, config.hidden, 1e-5);
    let q = matmul(
        &norm1,
        &weights.wq,
        config.sequence,
        config.hidden,
        config.hidden,
    );
    let k = matmul(
        &norm1,
        &weights.wk,
        config.sequence,
        config.hidden,
        config.hidden,
    );
    let v = matmul(
        &norm1,
        &weights.wv,
        config.sequence,
        config.hidden,
        config.hidden,
    );
    let mut context = vec![0.0; input.len()];
    let head_size = config.hidden / config.heads;
    for head in 0..config.heads {
        let mut scores = vec![0.0; config.sequence * config.sequence];
        for row in 0..config.sequence {
            for column in 0..config.sequence {
                let mut sum = 0.0;
                for inner in 0..head_size {
                    let index = head * head_size + inner;
                    sum += q[row * config.hidden + index] * k[column * config.hidden + index];
                }
                scores[row * config.sequence + column] = sum / (head_size as f32).sqrt();
            }
        }
        softmax_rows(&mut scores, config.sequence, config.sequence);
        for row in 0..config.sequence {
            for inner in 0..head_size {
                let mut sum = 0.0;
                for column in 0..config.sequence {
                    sum += scores[row * config.sequence + column]
                        * v[column * config.hidden + head * head_size + inner];
                }
                context[row * config.hidden + head * head_size + inner] = sum;
            }
        }
    }
    let projected = matmul(
        &context,
        &weights.wo,
        config.sequence,
        config.hidden,
        config.hidden,
    );
    let residual1: Vec<_> = input
        .iter()
        .zip(projected)
        .map(|(left, right)| left + right)
        .collect();
    let norm2 = layer_norm(&residual1, config.sequence, config.hidden, 1e-5);
    let mut hidden = matmul(
        &norm2,
        &weights.w1,
        config.sequence,
        config.hidden,
        config.feed_forward,
    );
    for value in &mut hidden {
        *value = gelu(*value);
    }
    let down = matmul(
        &hidden,
        &weights.w2,
        config.sequence,
        config.feed_forward,
        config.hidden,
    );
    Ok(residual1
        .into_iter()
        .zip(down)
        .map(|(left, right)| left + right)
        .collect())
}

fn matmul(left: &[f32], right: &[f32], rows: usize, inner: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += left[row * inner + k] * right[k * columns + column];
            }
            output[row * columns + column] = sum;
        }
    }
    output
}

fn layer_norm(input: &[f32], rows: usize, columns: usize, epsilon: f32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * columns..(row + 1) * columns];
        let mean = values.iter().sum::<f32>() / columns as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / columns as f32;
        let scale = (variance + epsilon).sqrt().recip();
        for column in 0..columns {
            output[row * columns + column] = (values[column] - mean) * scale;
        }
    }
    output
}

fn softmax_rows(values: &mut [f32], rows: usize, columns: usize) {
    for row in 0..rows {
        let values = &mut values[row * columns..(row + 1) * columns];
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for value in values.iter_mut() {
            *value = (*value - maximum).exp();
            total += *value;
        }
        for value in values {
            *value /= total;
        }
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044715 * value.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn composing_a_schedule_offsets_every_finite_lifetime() {
        let allocation = |tensor, live_until, kind| Allocation {
            tensor: TensorId(tensor),
            tile: 0,
            address: 0x60000 + tensor as u32 * 0x100,
            size: 0x100,
            live_from: 0,
            live_until,
            kind,
        };
        let mut parent = Schedule {
            layouts: Vec::new(),
            phases: vec![Phase::Exchange {
                transfers: Vec::new(),
            }],
            allocations: Vec::new(),
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };
        let mut child = Schedule {
            layouts: Vec::new(),
            phases: vec![Phase::Exchange {
                transfers: Vec::new(),
            }],
            allocations: vec![
                allocation(0, usize::MAX, AllocationKind::Home),
                allocation(1, 1, AllocationKind::Home),
                allocation(2, 1, AllocationKind::ExchangeStaging { phase: 0 }),
            ],
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };

        append_child_schedule(&mut parent, &mut child).unwrap();

        assert_eq!(parent.allocations[0].live_from, 0);
        assert_eq!(parent.allocations[0].live_until, usize::MAX);
        for allocation in &parent.allocations[1..] {
            assert_eq!((allocation.live_from, allocation.live_until), (1, 2));
        }
        assert_eq!(
            parent.allocations[2].kind,
            AllocationKind::ExchangeStaging { phase: 1 }
        );
    }

    #[test]
    fn distributed_a16_blocks_feed_gemm_without_a_full_row_shard() {
        let source = BlockPlacement {
            tensor: TensorId(0),
            tile: 1,
            address: 0xa0000,
            block_row: 0,
            block_column: 0,
            row_start: 0,
            column_start: 0,
            rows: 12,
            columns: 64,
        };
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: vec![Allocation {
                tensor: source.tensor,
                tile: source.tile,
                address: source.address,
                size: 12 * 64 * 2,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            }],
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };

        let appended = append_blocked_gemm_f16_with_a16_blocks(
            &mut schedule,
            &[source],
            BlockedGemmConfig {
                rows: 12,
                inner_dimension: 64,
                columns: 64,
                block_dimension: 64,
                inner_block_dimension: 64,
                row_block_dimension: 12,
                tile_count: 2,
                data_base: 0xb0000,
                data_limit: 0xe8000,
                data_type: GemmDataType::F16,
                retain_profile_metadata: true,
            },
        )
        .unwrap();

        assert_eq!(appended.output.len(), 1);
        assert!(matches!(&schedule.phases[0], Phase::Exchange { .. }));
        assert!(matches!(
            &schedule.phases[1],
            Phase::Compute { commands, .. } if commands.len() == 1
        ));
        let placed_input = match &schedule.phases[1] {
            Phase::Compute { commands, .. } => commands[0].output,
            _ => unreachable!(),
        };
        assert!(
            schedule
                .allocations
                .iter()
                .filter(|allocation| {
                    allocation.tensor == placed_input && allocation.kind == AllocationKind::Home
                })
                .all(|allocation| {
                    allocation.live_from > 0 && allocation.live_until < usize::MAX
                })
        );
        end_tensor_lifetimes(
            &mut schedule,
            appended.output.iter().map(|block| block.tensor),
        )
        .unwrap();
        for output in &appended.output {
            let output_end =
                output.address + u32::from(output.rows) * u32::from(output.columns) * 2;
            assert!(
                schedule
                    .allocations
                    .iter()
                    .filter(|allocation| {
                        allocation.kind == AllocationKind::Home
                            && allocation.tile == output.tile
                            && allocation.address >= output.address
                            && allocation.address + allocation.size <= output_end
                    })
                    .all(|allocation| allocation.live_until < usize::MAX)
            );
        }
    }

    #[test]
    fn distributed_native_fp8_input_reblocks_mismatched_row_fragments() {
        let mut input = Vec::new();
        let mut allocations = Vec::new();
        for block_row in 0..3u16 {
            let tensor = TensorId(usize::from(block_row));
            let tile = block_row;
            let address = 0xa0000;
            input.push(BlockPlacement {
                tensor,
                tile,
                address,
                block_row,
                block_column: 0,
                row_start: block_row * 16,
                column_start: 0,
                rows: 16,
                columns: 64,
            });
            allocations.push(Allocation {
                tensor,
                tile,
                address,
                size: 16 * 64 * 2,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
        }
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 8,
            peak_sram: BTreeMap::new(),
        };

        append_blocked_gemm_f16_with_a16_blocks(
            &mut schedule,
            &input,
            BlockedGemmConfig {
                rows: 48,
                inner_dimension: 64,
                columns: 64,
                block_dimension: 64,
                inner_block_dimension: 64,
                row_block_dimension: 12,
                tile_count: 8,
                data_base: 0xb0000,
                data_limit: 0xe8000,
                data_type: GemmDataType::F8F143 {
                    input_scale: -4,
                    weight_scale: -7,
                },
                retain_profile_metadata: true,
            },
        )
        .unwrap();

        let placement_commands = match &schedule.phases[1] {
            Phase::Compute { commands, .. } => commands,
            _ => unreachable!(),
        };
        assert!(placement_commands.len() > 4);
        assert!(placement_commands.iter().all(|command| {
            command.specialization.operation == "quantize_reblock_a16_to_a32_f143"
                && command.metadata.contains_key("copy_rows")
        }));
        let copied_rows = placement_commands
            .iter()
            .map(|command| command.metadata["copy_rows"].parse::<u16>().unwrap())
            .sum::<u16>();
        assert_eq!(
            copied_rows,
            input.iter().map(|block| block.rows).sum::<u16>()
        );
        schedule.validate_allocations().unwrap();
    }

    #[test]
    fn distributed_a16_gemm_reblocks_mismatched_row_shards() {
        let input = (0..3u16)
            .map(|index| RowShardPlacement {
                tensor: TensorId(usize::from(index)),
                tile: index,
                address: 0xa0000,
                row_start: index * 10,
                rows: 10,
                columns: 64,
            })
            .collect::<Vec<_>>();
        for (data_type, expected_operation) in [
            (GemmDataType::F16, "reblock_f16_a16_to_a16"),
            (
                GemmDataType::F8F143 {
                    input_scale: -4,
                    weight_scale: -7,
                },
                "quantize_reblock_a16_to_a32_f143",
            ),
        ] {
            let mut schedule = Schedule {
                layouts: Vec::new(),
                phases: Vec::new(),
                allocations: input
                    .iter()
                    .map(|block| Allocation {
                        tensor: block.tensor,
                        tile: block.tile,
                        address: block.address,
                        size: u32::from(block.rows) * u32::from(block.columns) * 2,
                        live_from: 0,
                        live_until: usize::MAX,
                        kind: AllocationKind::Home,
                    })
                    .collect(),
                tile_count: 4,
                peak_sram: BTreeMap::new(),
            };

            let appended = append_blocked_gemm_f16_with_a16_input(
                &mut schedule,
                &input,
                BlockedGemmConfig {
                    rows: 30,
                    inner_dimension: 64,
                    columns: 64,
                    block_dimension: 64,
                    inner_block_dimension: 64,
                    row_block_dimension: 16,
                    tile_count: 4,
                    data_base: 0xb0000,
                    data_limit: 0xe8000,
                    data_type,
                    retain_profile_metadata: true,
                },
            )
            .unwrap();

            assert_eq!(
                appended.output.iter().map(|block| block.rows).sum::<u16>(),
                30
            );
            let reblocks = schedule
                .phases
                .iter()
                .filter_map(|phase| match phase {
                    Phase::Compute { commands, .. } => Some(commands),
                    Phase::Exchange { .. } => None,
                })
                .flatten()
                .filter(|command| command.specialization.operation == expected_operation)
                .collect::<Vec<_>>();
            assert!(!reblocks.is_empty());
            assert!(reblocks.iter().all(|command| command.arguments.len() == 3));
            assert_eq!(
                reblocks
                    .iter()
                    .map(|command| command.metadata["copy_rows"].parse::<u16>().unwrap())
                    .sum::<u16>(),
                30
            );
            schedule.validate_allocations().unwrap();
            schedule
                .lower_tile_programs(&ipu_exchange::Topology::c600())
                .unwrap();
        }
    }

    #[test]
    fn distributed_a16_gemm_partitions_input_placement_across_exchange_windows() {
        let rows = 128u16;
        let columns = 512u16;
        let row_block_rows = 16u16;
        let input = (0..rows / row_block_rows)
            .map(|block_row| RowShardPlacement {
                tensor: TensorId(usize::from(block_row)),
                tile: block_row % 2,
                address: 0xa0000 + u32::from(block_row / 2) * 0x4000,
                row_start: block_row * row_block_rows,
                rows: row_block_rows,
                columns,
            })
            .collect::<Vec<_>>();
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: input
                .iter()
                .map(|shard| Allocation {
                    tensor: shard.tensor,
                    tile: shard.tile,
                    address: shard.address,
                    size: u32::from(shard.rows) * u32::from(shard.columns) * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                })
                .collect(),
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };

        append_blocked_gemm_f16_with_a16_input(
            &mut schedule,
            &input,
            BlockedGemmConfig {
                rows,
                inner_dimension: columns,
                columns: 64,
                block_dimension: 64,
                inner_block_dimension: 64,
                row_block_dimension: row_block_rows,
                tile_count: 2,
                data_base: 0xb0000,
                data_limit: 0xe8000,
                data_type: GemmDataType::F16,
                retain_profile_metadata: true,
            },
        )
        .unwrap();

        let first_gemm_phase = schedule
            .phases
            .iter()
            .position(|phase| {
                matches!(phase, Phase::Compute { commands, .. } if commands.iter().any(|command| {
                    command.specialization.operation.starts_with("gemm_f16_")
                }))
            })
            .unwrap();
        let placement_phases = &schedule.phases[..first_gemm_phase];
        assert!(placement_phases.len() > 2);
        assert!(placement_phases.chunks_exact(2).all(|pair| {
            matches!(pair[0], Phase::Exchange { .. }) && matches!(pair[1], Phase::Compute { .. })
        }));
        let placed_blocks = placement_phases
            .iter()
            .filter_map(|phase| match phase {
                Phase::Compute { commands, .. } => Some(commands.len()),
                Phase::Exchange { .. } => None,
            })
            .sum::<usize>();
        assert_eq!(placed_blocks, usize::from(rows / row_block_rows) * 8);
        for phase in 0..first_gemm_phase {
            let staging = schedule
                .allocations
                .iter()
                .filter(|allocation| {
                    matches!(allocation.kind, AllocationKind::ExchangeStaging { phase: owner } if owner == phase)
                })
                .fold(BTreeMap::<u16, u32>::new(), |mut ends, allocation| {
                    ends.entry(allocation.tile)
                        .and_modify(|end| *end = (*end).max(allocation.address + allocation.size))
                        .or_insert(allocation.address + allocation.size);
                    ends
                });
            assert!(staging.values().all(|&end| {
                end <= ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES
            }));
        }
        schedule.validate_allocations().unwrap();
        schedule
            .lower_tile_programs(&ipu_exchange::Topology::c600())
            .unwrap();
    }

    #[test]
    fn distributed_a16_prelude_assigns_disjoint_staging_ranges() {
        let mut tile_offsets = [0u32; 2];
        let mut input = Vec::new();
        let mut allocations = Vec::new();
        for block_row in 0..2u16 {
            for block_column in 0..3u16 {
                let index = usize::from(block_row * 3 + block_column);
                let destination_tile = index as u16 % 2;
                let tile = 1 - destination_tile;
                let address = 0xa0000 + tile_offsets[usize::from(tile)];
                tile_offsets[usize::from(tile)] += 12 * 64 * 2;
                let tensor = TensorId(index);
                input.push(BlockPlacement {
                    tensor,
                    tile,
                    address,
                    block_row,
                    block_column,
                    row_start: block_row * 12,
                    column_start: block_column * 64,
                    rows: 12,
                    columns: 64,
                });
                allocations.push(Allocation {
                    tensor,
                    tile,
                    address,
                    size: 12 * 64 * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
            }
        }
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 2,
            peak_sram: BTreeMap::new(),
        };

        append_blocked_gemm_f16_with_a16_blocks(
            &mut schedule,
            &input,
            BlockedGemmConfig {
                rows: 24,
                inner_dimension: 192,
                columns: 64,
                block_dimension: 64,
                inner_block_dimension: 64,
                row_block_dimension: 12,
                tile_count: 2,
                data_base: 0xb0000,
                data_limit: 0xe8000,
                data_type: GemmDataType::F16,
                retain_profile_metadata: true,
            },
        )
        .unwrap();

        for tile in 0..2 {
            let mut ranges = schedule
                .allocations
                .iter()
                .filter(|allocation| {
                    allocation.tile == tile
                        && matches!(
                            allocation.kind,
                            AllocationKind::ExchangeStaging { phase: 0 }
                        )
                })
                .map(|allocation| (allocation.address, allocation.address + allocation.size))
                .collect::<Vec<_>>();
            ranges.sort_unstable();
            assert!(ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        }
    }

    #[test]
    fn c16_bias_is_stored_once_per_column_block_and_shared_across_rows() {
        let mut output = Vec::new();
        let mut allocations = Vec::new();
        for block_row in 0..2u16 {
            for block_column in 0..2u16 {
                let index = usize::from(block_row * 2 + block_column);
                let block = BlockPlacement {
                    tensor: TensorId(index),
                    tile: index as u16,
                    address: 0xa0000,
                    block_row,
                    block_column,
                    row_start: block_row * 12,
                    column_start: block_column * 64,
                    rows: 12,
                    columns: 64,
                };
                output.push(block);
                allocations.push(Allocation {
                    tensor: block.tensor,
                    tile: block.tile,
                    address: block.address,
                    size: 12 * 64 * 2,
                    live_from: 0,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
            }
        }
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 4,
            peak_sram: BTreeMap::new(),
        };

        let biases = append_bias_f16_c16(&mut schedule, &output, 0xa0000, 0xe8000).unwrap();

        assert_eq!(biases.len(), 2);
        assert!(matches!(
            &schedule.phases[0],
            Phase::Exchange { transfers } if transfers.len() == 2
        ));
        assert!(matches!(
            &schedule.phases[1],
            Phase::Compute { commands, .. } if commands.len() == output.len()
        ));
    }

    #[test]
    fn appended_gemm_rotation_avoids_resident_tile_pressure() {
        let allocation = |tensor, tile, size| Allocation {
            tensor: TensorId(tensor),
            tile,
            address: 0xa0000,
            size,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        };
        let schedule = |allocations| Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: 4,
            peak_sram: BTreeMap::new(),
        };
        let parent = schedule(vec![allocation(0, 0, 128)]);
        let child = vec![
            BlockPlacement {
                tensor: TensorId(1),
                tile: 0,
                address: 0xa0000,
                block_row: 0,
                block_column: 0,
                row_start: 0,
                rows: 1,
                column_start: 0,
                columns: 32,
            },
            BlockPlacement {
                tensor: TensorId(2),
                tile: 1,
                address: 0xa0000,
                block_row: 0,
                block_column: 1,
                row_start: 0,
                rows: 1,
                column_start: 32,
                columns: 32,
            },
        ];

        let memory = MemoryPolicy::contiguous(0xa0000, 0xe8000);
        let rotation = choose_resident_tile_rotation_in_arenas(
            &parent,
            &child,
            GemmDataType::F16.element_bytes(),
            &memory,
        );

        assert_ne!(rotation, 0);
        assert_ne!(rotation, 3);
    }

    #[test]
    fn blocked_gemm_plan_preserves_block_ownership_and_phase_dependencies() {
        let plan = plan_blocked_gemm(BlockedGemmConfig {
            rows: 128,
            inner_dimension: 128,
            columns: 128,
            block_dimension: 64,
            inner_block_dimension: 32,
            row_block_dimension: 64,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F32,
            retain_profile_metadata: true,
        })
        .unwrap();
        assert!(!plan.left.is_empty());
        assert!(!plan.right.is_empty());
        assert!(!plan.output.is_empty());
        assert!(plan.output.iter().all(|block| {
            (0xa0000..0xe8000).contains(&block.address)
                && block.address != ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
        }));
        assert_eq!(
            plan.output
                .iter()
                .map(|block| block.tile)
                .collect::<BTreeSet<_>>()
                .len(),
            plan.output.len()
        );
        let rounds = &plan.schedule.phases[..plan.schedule.phases.len() - 2];
        assert!(matches!(
            plan.schedule.phases[plan.schedule.phases.len() - 2],
            Phase::Exchange { .. }
        ));
        assert!(matches!(
            plan.schedule.phases.last().unwrap(),
            Phase::Compute { .. }
        ));
        assert!(rounds.chunks_exact(2).all(|round| {
            matches!(round[0], Phase::Exchange { .. }) && matches!(round[1], Phase::Compute { .. })
        }));
        assert!(plan.schedule.phases.iter().all(|phase| {
            match phase {
                Phase::Exchange { transfers } => transfers.iter().all(|transfer| {
                    transfer.bytes <= ipu_exchange::MAX_TRANSFER_WORDS * 4
                        && transfer.staging_address.is_some()
                }),
                Phase::Compute { commands, .. } => commands.iter().all(|command| {
                    let units = u32::try_from(command.specialization.shape[0]).unwrap();
                    let operation = command.specialization.operation.as_ref();
                    let valid_arguments = if operation.starts_with("gemm_f32_") {
                        command.arguments.is_empty()
                    } else {
                        command.arguments == [units, units / 6, units % 6]
                    };
                    command.inputs.len() == 2
                        && valid_arguments
                        && (operation.starts_with("gemm_f32_") || operation == "copy_u64")
                        && command.metadata.contains_key("label")
                        && command.metadata.contains_key("wave")
                        && command.metadata.contains_key("output_block_row")
                        && command.metadata.contains_key("output_block_column")
                }),
            }
        }));
        assert!(plan.schedule.allocations.iter().all(|allocation| {
            !matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
        }));
    }

    #[test]
    fn fp8_weight_storage_expands_transiently_before_fp16_gemm() {
        let mut plan = plan_blocked_gemm(BlockedGemmConfig {
            // More output blocks than tiles force the same right-hand block to
            // be expanded in multiple waves.
            rows: 320,
            inner_dimension: 64,
            columns: 64,
            block_dimension: 64,
            inner_block_dimension: 64,
            row_block_dimension: 64,
            tile_count: 4,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F16F8Weights { scale: 0 },
            retain_profile_metadata: true,
        })
        .unwrap();
        let scales = vec![-7; plan.right.len()];
        set_f8_weight_block_scales(&mut plan.schedule, &plan.right, &scales).unwrap();

        for block in &plan.right {
            let allocation = plan
                .schedule
                .allocations
                .iter()
                .find(|allocation| {
                    allocation.tensor == block.tensor
                        && allocation.tile == block.tile
                        && allocation.kind == AllocationKind::Home
                })
                .unwrap();
            assert_eq!(
                allocation.size,
                u32::from(block.rows) * u32::from(block.columns)
            );
        }
        let expansion_commands = plan.schedule.phases.iter().flat_map(|phase| match phase {
            Phase::Compute { commands, .. } => commands
                .iter()
                .filter(|command| command.specialization.operation == "expand_f8_f143_to_f16")
                .collect::<Vec<_>>(),
            Phase::Exchange { .. } => Vec::new(),
        });
        assert!(
            expansion_commands
                .clone()
                .all(|command| command.arguments.get(1) == Some(&u32::from((-7i8) as u8)))
        );
        let expanded = plan
            .schedule
            .phases
            .iter()
            .flat_map(|phase| match phase {
                Phase::Compute { commands, .. } => commands
                    .iter()
                    .filter(|command| command.specialization.operation == "expand_f8_f143_to_f16")
                    .map(|command| command.output)
                    .collect::<Vec<_>>(),
                Phase::Exchange { .. } => Vec::new(),
            })
            .collect::<HashSet<_>>();
        assert!(!expanded.is_empty());
        assert!(plan.schedule.allocations.iter().any(|allocation| {
            expanded.contains(&allocation.tensor)
                && allocation.size
                    == u32::from(plan.right[0].rows) * u32::from(plan.right[0].columns) * 2
                && allocation.address >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                && allocation.address + allocation.size
                    <= ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
        }));
        assert!(plan.schedule.phases.iter().any(|phase| {
            matches!(phase, Phase::Compute { commands, .. } if commands.iter().any(|command| {
                command.specialization.operation.starts_with("gemm_f16_f8w_")
            }))
        }));
        plan.schedule
            .lower_tile_programs(&Topology::c600())
            .unwrap();
    }

    #[test]
    fn fp8_weight_expansion_batches_independent_inner_blocks() {
        let plan = plan_blocked_gemm(BlockedGemmConfig {
            rows: 64,
            inner_dimension: 256,
            columns: 128,
            block_dimension: 128,
            inner_block_dimension: 64,
            row_block_dimension: 64,
            tile_count: 64,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F16F8Weights { scale: 0 },
            retain_profile_metadata: true,
        })
        .unwrap();

        let expansion_batches = plan
            .schedule
            .phases
            .iter()
            .filter_map(|phase| match phase {
                Phase::Compute { commands, .. } => {
                    let count = commands
                        .iter()
                        .filter(|command| {
                            command.specialization.operation == "expand_f8_f143_to_f16"
                        })
                        .count();
                    (count != 0).then_some(count)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(expansion_batches.iter().sum::<usize>(), plan.right.len());
        assert!(expansion_batches.len() < plan.right.len());
        assert!(expansion_batches.iter().any(|&commands| commands > 1));
        assert!(plan.schedule.phases.iter().all(|phase| match phase {
            Phase::Compute { commands, .. }
                if commands.iter().any(|command| {
                    command.specialization.operation == "expand_f8_f143_to_f16"
                }) =>
                commands.chunks_exact(2).all(|pair| {
                    pair[0].specialization.operation == "expand_f8_f143_to_f16"
                        && pair[1]
                            .specialization
                            .operation
                            .starts_with("gemm_f16_f8w_")
                        && pair[0].tile == pair[1].tile
                }),
            _ => true,
        }));

        plan.schedule
            .lower_tile_programs(&Topology::c600())
            .unwrap();
    }

    #[test]
    fn native_fp8_gemm_keeps_fp8_operands_and_fp16_outputs() {
        let plan = plan_blocked_gemm(BlockedGemmConfig {
            rows: 128,
            inner_dimension: 128,
            columns: 128,
            block_dimension: 64,
            inner_block_dimension: 64,
            row_block_dimension: 64,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F8F143 {
                input_scale: -6,
                weight_scale: -9,
            },
            retain_profile_metadata: true,
        })
        .unwrap();

        let home_size = |block: &BlockPlacement| {
            plan.schedule
                .allocations
                .iter()
                .find(|allocation| {
                    allocation.tensor == block.tensor
                        && allocation.tile == block.tile
                        && allocation.kind == AllocationKind::Home
                })
                .unwrap()
                .size
        };
        assert!(
            plan.left.iter().all(|block| {
                home_size(block) == u32::from(block.rows) * u32::from(block.columns)
            })
        );
        assert!(
            plan.right.iter().all(|block| {
                home_size(block) == u32::from(block.rows) * u32::from(block.columns)
            })
        );
        assert!(plan.output.iter().all(|block| {
            home_size(block) == u32::from(block.rows) * u32::from(block.columns) * 2
        }));
        assert!(plan.schedule.phases.iter().all(|phase| match phase {
            Phase::Exchange { .. } => true,
            Phase::Compute { commands, .. } => commands.iter().all(|command| {
                command.specialization.operation == "copy_u64"
                    || (command.specialization.operation.starts_with("gemm_f8_")
                        && command.arguments == [u32::from((-15i8 as u8) & 0x3f)])
            }),
        }));
    }

    #[test]
    fn blocked_f16_gemm_uses_half_sized_storage_and_f16_kernels() {
        let config = BlockedGemmConfig {
            rows: 128,
            inner_dimension: 128,
            columns: 128,
            block_dimension: 64,
            inner_block_dimension: 64,
            row_block_dimension: 64,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F32,
            retain_profile_metadata: true,
        };
        let f32_plan = plan_blocked_gemm(config).unwrap();
        let f16_plan = plan_blocked_gemm(BlockedGemmConfig {
            data_type: GemmDataType::F16,
            ..config
        })
        .unwrap();

        assert_eq!(
            f16_plan.schedule.allocations.len(),
            f32_plan.schedule.allocations.len()
        );
        assert!(
            f16_plan
                .schedule
                .allocations
                .iter()
                .zip(&f32_plan.schedule.allocations)
                .all(|(f16_allocation, f32_allocation)| {
                    f16_allocation.tensor == f32_allocation.tensor
                        && f16_allocation.tile == f32_allocation.tile
                        && f16_allocation.size * 2 == f32_allocation.size
                })
        );
        assert!(f16_plan.schedule.phases.iter().all(|phase| match phase {
            Phase::Compute { commands, .. } => commands.iter().all(|command| {
                command.specialization.operation.starts_with("gemm_f16_")
                    || command.specialization.operation == "copy_u64"
            }),
            Phase::Exchange { .. } => true,
        }));
    }

    #[test]
    fn blocked_f16_gemm_supports_rectangular_output_blocks() {
        let plan = plan_blocked_gemm(BlockedGemmConfig {
            rows: 96,
            inner_dimension: 128,
            columns: 128,
            block_dimension: 32,
            inner_block_dimension: 64,
            row_block_dimension: 24,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F16,
            retain_profile_metadata: true,
        })
        .unwrap();

        assert_eq!(plan.output.len(), 16);
        assert!(
            plan.output
                .iter()
                .all(|block| block.rows == 24 && block.columns == 32)
        );
        assert!(plan.schedule.phases.iter().all(|phase| match phase {
            Phase::Exchange { .. } => true,
            Phase::Compute { commands, .. } => commands.iter().all(|command| {
                command.specialization.operation == "copy_u64"
                    || (command.specialization.operation.starts_with("gemm_f16_")
                        && command.specialization.shape == [24, 64, 32])
            }),
        }));
    }

    #[test]
    fn blocked_gemm_balances_rows_without_gaps_or_tile_overcommitment() {
        for dimension in [64, 128, 256, 1024, 1920, 2048, 2304, 4096] {
            let target = choose_gemm_row_block(dimension, 32, dimension, 64, 1472).unwrap();
            let plan = plan_blocked_gemm(BlockedGemmConfig {
                rows: dimension,
                inner_dimension: dimension,
                columns: dimension,
                block_dimension: 64,
                inner_block_dimension: 32,
                row_block_dimension: target,
                tile_count: 1472,
                data_base: 0xa0000,
                data_limit: 0xe8000,
                data_type: GemmDataType::F32,
                retain_profile_metadata: true,
            })
            .unwrap();
            let first_column: Vec<_> = plan
                .output
                .iter()
                .filter(|block| block.block_column == 0)
                .collect();

            assert_eq!(first_column.first().unwrap().row_start, 0);
            assert_eq!(
                first_column.last().unwrap().row_start + first_column.last().unwrap().rows,
                dimension
            );
            assert!(
                first_column
                    .windows(2)
                    .all(|blocks| { blocks[0].row_start + blocks[0].rows == blocks[1].row_start })
            );
            let minimum = first_column.iter().map(|block| block.rows).min().unwrap();
            let maximum = first_column.iter().map(|block| block.rows).max().unwrap();
            assert!(maximum - minimum <= 1);
            let exchange_slot_bytes =
                align_u32(u32::from(maximum) * 32 * 4 + u32::from(32u16) * 64 * 4, 32);
            let batch_capacity =
                usize::try_from(ipu_exchange::EXCHANGE_WINDOW_BYTES / exchange_slot_bytes).unwrap();
            assert!(plan.schedule.phases.iter().all(|phase| match phase {
                Phase::Compute { commands, .. } => {
                    let mut per_tile = BTreeMap::<u16, usize>::new();
                    for command in commands
                        .iter()
                        .filter(|command| command.specialization.operation.starts_with("gemm_f32_"))
                    {
                        *per_tile.entry(command.tile).or_default() += 1;
                    }
                    per_tile.values().all(|&count| count <= batch_capacity)
                }
                Phase::Exchange { .. } => true,
            }));
        }
    }

    #[test]
    fn gemm_tuning_candidates_are_feasible_and_layout_distinct() {
        for dimension in [64, 128, 256, 1024, 1920, 2048, 2304, 4096] {
            let candidates = gemm_row_block_candidates(dimension, 32, dimension, 64, 1472);
            assert!(!candidates.is_empty());
            assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
            let row_shards: BTreeSet<_> = candidates
                .iter()
                .map(|target| dimension.div_ceil(*target))
                .collect();
            assert_eq!(row_shards.len(), candidates.len());
            assert!(
                candidates
                    .contains(&choose_gemm_row_block(dimension, 32, dimension, 64, 1472).unwrap())
            );
        }
        assert!(choose_gemm_row_block(65, 32, 65, 64, 1472).is_none());
    }

    #[test]
    fn gemm_row_block_choice_respects_materialized_row_limit() {
        let unrestricted =
            choose_gemm_row_block_for(1458, 64, 3456, 64, 1472, GemmDataType::F16).unwrap();
        let constrained =
            choose_gemm_row_block_for_max_rows(1458, 64, 3456, 64, 1472, GemmDataType::F16, 35)
                .unwrap();

        assert!(unrestricted > 35);
        assert!(constrained <= 35);
        assert!(
            gemm_row_block_candidates_for(1458, 64, 3456, 64, 1472, GemmDataType::F16)
                .contains(&constrained)
        );
    }

    #[test]
    fn shape_aware_gemm_blocking_amortizes_deeper_inner_dimensions() {
        let shallow = choose_gemm_row_block_for_shape(
            729,
            1152,
            64,
            1152,
            64,
            1472,
            GemmDataType::F16F8Weights { scale: 0 },
        )
        .unwrap();
        let deep = choose_gemm_row_block_for_shape(
            729,
            4352,
            128,
            1152,
            64,
            1472,
            GemmDataType::F16F8Weights { scale: 0 },
        )
        .unwrap();

        assert!(deep >= shallow);
    }

    #[test]
    fn a16_reblock_count_encodes_full_column_span() {
        assert_eq!(pack_a16_reblock_count(17, 128, None).unwrap(), 8 << 16 | 17);
        assert_eq!(
            pack_a16_reblock_count(17, 128, Some(-4)).unwrap(),
            4 << 16 | 0x3c << 10 | 17
        );
        assert!(pack_a16_reblock_count(17, 80, Some(-4)).is_err());
    }

    fn exchange_schedule(transfers: Vec<Transfer>) -> Schedule {
        let mut allocations = Vec::new();
        for transfer in &transfers {
            if !allocations.iter().any(|allocation: &Allocation| {
                allocation.tensor == transfer.tensor
                    && allocation.tile == transfer.source_tile
                    && allocation.kind == AllocationKind::Home
            }) {
                allocations.push(Allocation {
                    tensor: transfer.tensor,
                    tile: transfer.source_tile,
                    address: 0x62000,
                    size: transfer.bytes,
                    live_from: 0,
                    live_until: 1,
                    kind: AllocationKind::Home,
                });
            }
            if transfer.staging_address.is_none()
                && !allocations.iter().any(|allocation| {
                    allocation.tensor == transfer.tensor
                        && allocation.tile == transfer.destination_tile
                        && allocation.kind == AllocationKind::ExchangeStaging { phase: 0 }
                })
            {
                allocations.push(Allocation {
                    tensor: transfer.tensor,
                    tile: transfer.destination_tile,
                    address: 0x52000,
                    size: transfer.bytes,
                    live_from: 0,
                    live_until: 1,
                    kind: AllocationKind::ExchangeStaging { phase: 0 },
                });
            }
        }
        Schedule {
            layouts: Vec::new(),
            phases: vec![Phase::Exchange { transfers }],
            allocations,
            tile_count: 16,
            peak_sram: BTreeMap::new(),
        }
    }

    #[test]
    fn repeated_region_aligns_changed_command_abis() {
        let command = |phase, inputs: usize| Phase::Compute {
            op: OpId(phase),
            commands: vec![Arc::new(KernelCommand {
                tile: 3,
                output: TensorId(phase),
                inputs: (0..inputs).map(TensorId).collect(),
                arguments: vec![16],
                specialization: SpecializationKey {
                    operation: "test_kernel".into(),
                    shape: vec![16],
                    worker_count: 6,
                    role: "compute".into(),
                    alignment: 8,
                },
                metadata: BTreeMap::new(),
            })],
        };
        let schedule = Schedule {
            layouts: Vec::new(),
            phases: vec![
                Phase::Exchange {
                    transfers: Vec::new(),
                },
                command(1, 2),
                Phase::Exchange {
                    transfers: Vec::new(),
                },
                command(3, 2),
                Phase::Exchange {
                    transfers: Vec::new(),
                },
                command(5, 1),
            ],
            allocations: Vec::new(),
            tile_count: 4,
            peak_sram: BTreeMap::new(),
        };
        let mut repeated = RepeatedRegion::new("block", &schedule, 0..2).unwrap();
        repeated.push_instance(&schedule, 2..4).unwrap();
        assert_eq!(repeated.phase_instances, vec![0..2, 2..4]);
        assert!(repeated.is_compatible(&schedule, 4..6));
        repeated.push_instance(&schedule, 4..6).unwrap();
        assert_eq!(repeated.phase_instances, vec![0..2, 2..4, 4..6]);
    }

    #[test]
    fn encoder_graph_compiles_deterministically() {
        let graph = encoder_graph(EncoderConfig::default()).unwrap();
        let first = compile(&graph, &CompilerOptions::default()).unwrap();
        let second = compile(&graph, &CompilerOptions::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(graph.ops.len(), 15);
        assert!(first.phases.len() >= graph.ops.len());
        assert!(first.peak_sram.values().all(|peak| *peak < 0x8e000));
        assert!(
            first.allocations.iter().any(|allocation| matches!(
                allocation.kind,
                AllocationKind::ExchangeStaging { .. }
            ))
        );
        let lowered = first.lower_exchanges(&Topology::c600()).unwrap();
        assert!(!lowered.is_empty());
        assert!(lowered.iter().all(|phase| phase.cost.launches > 0));
        let programs = first.lower_tile_programs(&Topology::c600()).unwrap();
        assert_eq!(programs.len(), usize::from(first.tile_count));
        assert!(programs.iter().all(|program| !program.steps.is_empty()));
    }

    #[test]
    fn live_allocations_do_not_overlap() {
        let graph = encoder_graph(EncoderConfig::default()).unwrap();
        let schedule = compile(&graph, &CompilerOptions::default()).unwrap();
        for (index, left) in schedule.allocations.iter().enumerate() {
            for right in &schedule.allocations[index + 1..] {
                if left.tile != right.tile
                    || !lifetimes_overlap(
                        left.live_from,
                        left.live_until,
                        right.live_from,
                        right.live_until,
                    )
                {
                    continue;
                }
                assert!(
                    left.address + left.size <= right.address
                        || right.address + right.size <= left.address
                );
            }
        }
    }

    #[test]
    fn lowering_coalesces_fanout_and_packs_disjoint_groups() {
        let schedule = exchange_schedule(vec![
            Transfer {
                source_tile: 0,
                destination_tile: 1,
                tensor: TensorId(0),
                bytes: 64,
                staging_address: None,
            },
            Transfer {
                source_tile: 0,
                destination_tile: 2,
                tensor: TensorId(0),
                bytes: 64,
                staging_address: None,
            },
            Transfer {
                source_tile: 3,
                destination_tile: 4,
                tensor: TensorId(1),
                bytes: 128,
                staging_address: None,
            },
        ]);
        let lowered = schedule.lower_exchanges(&Topology::c600()).unwrap();
        assert_eq!(lowered.len(), 1);
        assert_eq!(lowered[0].epochs.len(), 1);
        assert_eq!(lowered[0].epochs[0].groups.len(), 2);
        assert!(lowered[0].epochs[0].groups.iter().any(|group| {
            group.destination_tiles.len() == 2 && group.addressing == ExchangeAddressing::Absolute
        }));
        assert_eq!(lowered[0].cost.launches, 1);
        assert_eq!(lowered[0].cost.payload_words, 16 + 32);
        assert_eq!(lowered[0].epochs[0].tile_rows.len(), 5);
        let inactive = lowered[0].epochs[0].row_for(15);
        assert_eq!(inactive[0], SANS_INACTIVE_INSTRUCTION);
        assert_eq!(inactive[1], SYNC_ANS_INSTRUCTION);
    }

    #[test]
    fn lowering_uses_transfer_owned_staging_addresses() {
        let schedule = exchange_schedule(vec![Transfer {
            source_tile: 0,
            destination_tile: 1,
            tensor: TensorId(0),
            bytes: 64,
            staging_address: Some(0x53a40),
        }]);

        assert!(schedule.allocations.iter().all(|allocation| {
            !matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
        }));
        schedule.lower_exchanges(&Topology::c600()).unwrap();
        schedule.lower_tile_programs(&Topology::c600()).unwrap();
    }

    #[test]
    fn lowering_uses_point_schedule_for_an_independent_single_destination() {
        let topology = Topology::c600();
        let schedule = exchange_schedule(vec![Transfer {
            source_tile: 0,
            destination_tile: 1,
            tensor: TensorId(0),
            bytes: 64,
            staging_address: None,
        }]);
        let lowered = schedule.lower_exchanges(&topology).unwrap();
        let group = &lowered[0].epochs[0].groups[0];

        let point = topology.point_to_point(0, 1, 16).unwrap();
        let mut expected_sender = point.sender;
        let mut expected_receiver =
            finalize_point_receiver(&point.receiver, topology.physical(0).unwrap()).unwrap();
        patch_sender_address(&mut expected_sender, 0x62000).unwrap();
        patch_receiver_address(&mut expected_receiver, 0x52000).unwrap();

        assert_eq!(group.sender, expected_sender);
        assert_eq!(group.receivers, vec![expected_receiver]);
    }

    #[test]
    fn lowering_assigns_distinct_times_to_tile_role_conflicts() {
        let schedule = exchange_schedule(vec![
            Transfer {
                source_tile: 0,
                destination_tile: 1,
                tensor: TensorId(0),
                bytes: 64,
                staging_address: None,
            },
            Transfer {
                source_tile: 2,
                destination_tile: 3,
                tensor: TensorId(1),
                bytes: 64,
                staging_address: None,
            },
            Transfer {
                source_tile: 1,
                destination_tile: 2,
                tensor: TensorId(2),
                bytes: 64,
                staging_address: None,
            },
        ]);
        let lowered = schedule.lower_exchanges(&Topology::c600()).unwrap();
        assert_eq!(lowered[0].epochs.len(), 1);
        assert_eq!(lowered[0].cost.launches, 1);
        let row = lowered[0].epochs[0].row_for(1);
        assert_eq!(
            row.iter()
                .filter(|word| **word == ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION)
                .count(),
            1
        );
        assert_eq!(
            row.iter()
                .filter(|word| **word == RETURN_M10_INSTRUCTION)
                .count(),
            1
        );
        assert!(
            u64::from(ipu_exchange::plan_event_cycles(&row).unwrap())
                <= lowered[0].cost.estimated_cycles
        );
    }

    #[test]
    fn lowering_places_dependent_relay_rows_on_one_timeline() {
        let tensor = TensorId(0);
        let mut schedule = exchange_schedule(vec![
            Transfer {
                source_tile: 3,
                destination_tile: 1,
                tensor,
                bytes: 256,
                staging_address: None,
            },
            Transfer {
                source_tile: 1,
                destination_tile: 3,
                tensor,
                bytes: 256,
                staging_address: None,
            },
        ]);
        schedule
            .allocations
            .retain(|allocation| !(allocation.tensor == tensor && allocation.tile == 1));
        schedule.allocations.push(Allocation {
            tensor,
            tile: 1,
            address: ipu_exchange::EXCHANGE_WINDOW_BASE + 1024,
            size: 256,
            live_from: 0,
            live_until: 1,
            kind: AllocationKind::ExchangeStaging { phase: 0 },
        });
        let lowered = schedule.lower_exchanges(&Topology::c600()).unwrap();
        let epoch = &lowered[0].epochs[0];

        assert_eq!(lowered[0].cost.launches, 1);
        assert_eq!(epoch.groups.len(), 2);
        assert_eq!(
            epoch
                .groups
                .iter()
                .map(|group| group.source_tile)
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
        let relay = epoch.row_for(1);
        assert_eq!(
            relay
                .iter()
                .filter(|word| **word == ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION)
                .count(),
            1
        );
        assert_eq!(
            relay
                .iter()
                .filter(|word| **word == RETURN_M10_INSTRUCTION)
                .count(),
            1
        );
        assert!(
            u64::from(ipu_exchange::plan_event_cycles(&relay).unwrap())
                <= lowered[0].cost.estimated_cycles
        );
    }

    #[test]
    fn lowering_selects_live_staging_as_a_later_phase_source() {
        let tensor = TensorId(0);
        let schedule = Schedule {
            layouts: Vec::new(),
            phases: vec![
                Phase::Exchange {
                    transfers: vec![Transfer {
                        source_tile: 0,
                        destination_tile: 1,
                        tensor,
                        bytes: 64,
                        staging_address: None,
                    }],
                },
                Phase::Exchange {
                    transfers: vec![Transfer {
                        source_tile: 1,
                        destination_tile: 2,
                        tensor,
                        bytes: 64,
                        staging_address: None,
                    }],
                },
            ],
            allocations: vec![
                Allocation {
                    tensor,
                    tile: 0,
                    address: 0x62000,
                    size: 64,
                    live_from: 0,
                    live_until: 2,
                    kind: AllocationKind::Home,
                },
                Allocation {
                    tensor,
                    tile: 1,
                    address: 0x52000,
                    size: 64,
                    live_from: 0,
                    live_until: 2,
                    kind: AllocationKind::ExchangeStaging { phase: 0 },
                },
                Allocation {
                    tensor,
                    tile: 2,
                    address: 0x53000,
                    size: 64,
                    live_from: 1,
                    live_until: 2,
                    kind: AllocationKind::ExchangeStaging { phase: 1 },
                },
            ],
            tile_count: 16,
            peak_sram: BTreeMap::new(),
        };

        let lowered = schedule.lower_exchanges(&Topology::c600()).unwrap();
        assert_eq!(lowered.len(), schedule.phases.len());
        assert!(lowered.iter().all(|phase| phase.epochs.len() == 1));
        assert!(lowered[1].epochs[0].tile_rows.contains_key(&1));
    }

    #[test]
    fn constrained_allocator_obeys_bounds_and_reuses_dead_storage() {
        let constraint = MemoryConstraint {
            base: ipu_exchange::EXCHANGE_WINDOW_BASE,
            limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
            alignment: 32,
            placement: MemoryPlacement::High,
        };
        let first = find_free_region(&[], 0, 256, 0, 1, constraint).unwrap();
        let allocation = Allocation {
            tensor: TensorId(0),
            tile: 0,
            address: first,
            size: 256,
            live_from: 0,
            live_until: 1,
            kind: AllocationKind::Home,
        };
        let concurrent =
            find_free_region(std::slice::from_ref(&allocation), 0, 256, 0, 1, constraint).unwrap();
        let reused =
            find_free_region(std::slice::from_ref(&allocation), 0, 256, 1, 2, constraint).unwrap();
        let other_tile = find_free_region(&[allocation], 1, 256, 0, 1, constraint).unwrap();

        for address in [first, concurrent, reused, other_tile] {
            assert_eq!(address % constraint.alignment, 0);
            assert!(address >= constraint.base);
            assert!(address + 256 <= constraint.limit);
        }
        assert!(concurrent + 256 <= first || first + 256 <= concurrent);
        assert_eq!(reused, first);
        assert_eq!(other_tile, first);
    }

    #[test]
    fn constrained_allocator_reports_exhaustion() {
        let constraint = MemoryConstraint {
            base: 0x50000,
            limit: 0x50040,
            alignment: 32,
            placement: MemoryPlacement::Low,
        };
        let allocations = [Allocation {
            tensor: TensorId(0),
            tile: 7,
            address: constraint.base,
            size: constraint.limit - constraint.base,
            live_from: 3,
            live_until: 5,
            kind: AllocationKind::ExchangeStaging { phase: 3 },
        }];

        assert!(find_free_region(&allocations, 7, 32, 4, 6, constraint).is_err());
    }

    #[test]
    fn occupied_allocator_spills_without_crossing_arena_boundaries() {
        let arenas = [
            MemoryArena::low(0x1000, 0x1040),
            MemoryArena::low(0x2000, 0x2080),
        ];
        let mut occupied = vec![(arenas[0].base, arenas[0].limit)];
        let size = 32;
        let address = allocate_from_occupied_arenas(&mut occupied, size, &arenas, 16).unwrap();

        assert_eq!(address % 16, 0);
        assert!(
            arenas
                .iter()
                .any(|arena| address >= arena.base && address + size <= arena.limit)
        );
        assert!(occupied.windows(2).all(|pair| pair[0].1 <= pair[1].0));
    }

    #[test]
    fn ipu21_policy_maps_named_regions_without_overlap() {
        let ordinary_low_base =
            ipu_package::TILE_MEMORY_BASE + 7 * ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let data_limit = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
        let regions = [
            Ipu21MemoryRegion::OrdinaryHigh,
            Ipu21MemoryRegion::OrdinaryLow,
            Ipu21MemoryRegion::Interleaved,
        ];
        let policy =
            MemoryPolicy::ipu21(ordinary_low_base, data_limit, &regions, &regions).unwrap();

        assert_eq!(policy.resident.len(), regions.len());
        assert!(policy.resident.iter().all(|arena| arena.base < arena.limit));
        assert!(policy.resident.iter().enumerate().all(|(index, arena)| {
            policy.resident[..index]
                .iter()
                .all(|previous| previous.limit <= arena.base || arena.limit <= previous.base)
        }));
        assert!(
            MemoryPolicy::ipu21(
                ordinary_low_base,
                data_limit,
                &[Ipu21MemoryRegion::OrdinaryHigh; 2],
                &regions,
            )
            .is_err()
        );
    }

    #[test]
    fn resident_allocation_does_not_reuse_expired_transient_storage() {
        let arena = MemoryArena::low(0x1000, 0x1100);
        let expired = Allocation {
            tensor: TensorId(1),
            tile: 0,
            address: arena.base,
            size: 64,
            live_from: 0,
            live_until: 1,
            kind: AllocationKind::Home,
        };

        let transient =
            find_free_region_in_arenas(std::slice::from_ref(&expired), 0, 64, 1, 2, &[arena], 16)
                .unwrap();
        let resident =
            find_free_region_in_arenas(&[expired], 0, 64, 0, usize::MAX, &[arena], 16).unwrap();

        assert_eq!(transient, arena.base);
        assert_ne!(resident, transient);
    }

    #[test]
    fn segmented_allocator_never_spans_holes_and_reuses_lifetimes() {
        let arenas = [
            MemoryArena::low(0x58000, 0x58100),
            MemoryArena::low(0x60000, 0x60400),
        ];
        let mut allocations = Vec::new();
        let mut by_tile = HashMap::default();
        allocate_region(
            &mut allocations,
            &mut by_tile,
            TensorId(0),
            3,
            512,
            0,
            2,
            32,
            AllocationKind::Home,
            &arenas,
            "large tensor",
        )
        .unwrap();
        allocate_region(
            &mut allocations,
            &mut by_tile,
            TensorId(1),
            3,
            512,
            2,
            4,
            32,
            AllocationKind::Home,
            &arenas,
            "later tensor",
        )
        .unwrap();

        assert!(
            allocations
                .iter()
                .all(|allocation| arenas.iter().any(|arena| {
                    allocation.address >= arena.base
                        && allocation.address + allocation.size <= arena.limit
                }))
        );
        assert_eq!(allocations[0].address, allocations[1].address);
    }

    #[test]
    fn scheduler_is_deterministic_for_varied_transfer_graphs() {
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        for _ in 0..64 {
            let mut transfers = Vec::new();
            let mut destinations = HashSet::default();
            for tensor in 0..12 {
                let source = rng.u16(0..16);
                let mut destination = rng.u16(0..16);
                if destination == source {
                    destination = (destination + 1) % 16;
                }
                if !destinations.insert((destination, TensorId(tensor))) {
                    continue;
                }
                transfers.push(Transfer {
                    source_tile: source,
                    destination_tile: destination,
                    tensor: TensorId(tensor),
                    bytes: 4 * rng.u32(1..=64),
                    staging_address: None,
                });
            }
            let schedule = exchange_schedule(transfers.clone());
            let first = schedule.lower_exchanges(&Topology::c600()).unwrap();
            let second = schedule.lower_exchanges(&Topology::c600()).unwrap();
            assert_eq!(first, second);
            let represented: usize = first[0]
                .epochs
                .iter()
                .flat_map(|epoch| &epoch.groups)
                .map(|group| group.destination_tiles.len())
                .sum();
            assert_eq!(represented, transfers.len());
        }
    }

    #[test]
    fn randomized_exchange_encoding_preserves_static_invariants() {
        const WORD_COUNTS: [u32; 10] = [1, 2, 15, 16, 17, 63, 64, 65, 127, 256];
        let topology = Topology::c600();
        let mut rng = fastrand::Rng::with_seed(0xd1b5_4a32_d192_ed03);
        for case in 0..64 {
            let mut transfers = Vec::new();
            for tensor in 0..24 {
                let source = rng.u16(0..96);
                let words = WORD_COUNTS[rng.usize(0..WORD_COUNTS.len())];
                let fanout = rng.usize(1..=4);
                let mut destinations = Vec::new();
                while destinations.len() < fanout {
                    let destination = rng.u16(0..96);
                    if destination != source && !destinations.contains(&destination) {
                        destinations.push(destination);
                    }
                }
                transfers.extend(destinations.into_iter().map(|destination| Transfer {
                    source_tile: source,
                    destination_tile: destination,
                    tensor: TensorId(tensor),
                    bytes: words * 4,
                    staging_address: None,
                }));
            }
            let schedule = exchange_schedule(transfers.clone());
            let first = schedule.lower_exchanges(&topology).unwrap();
            let second = schedule.lower_exchanges(&topology).unwrap();
            assert_eq!(first, second, "case={case}");

            let epoch = &first[0].epochs[0];
            assert_eq!(first[0].epochs.len(), 1, "case={case}");
            assert_eq!(first[0].cost.launches, 1, "case={case}");
            assert_eq!(
                epoch
                    .groups
                    .iter()
                    .map(|group| group.destination_tiles.len())
                    .sum::<usize>(),
                transfers.len(),
                "case={case}"
            );
            for (tile, row) in &epoch.tile_rows {
                assert_eq!(
                    row.iter()
                        .filter(|word| { **word == ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION })
                        .count(),
                    1,
                    "case={case} tile={tile}"
                );
                assert_eq!(
                    row.iter()
                        .filter(|word| **word == RETURN_M10_INSTRUCTION)
                        .count(),
                    1,
                    "case={case} tile={tile}"
                );
                assert!(
                    u64::from(ipu_exchange::plan_event_cycles(row).unwrap())
                        <= first[0].cost.estimated_cycles,
                    "case={case} tile={tile}"
                );
            }
            assert_eq!(
                epoch
                    .tile_rows
                    .values()
                    .map(|row| u64::from(ipu_exchange::plan_event_cycles(row).unwrap()))
                    .max(),
                Some(first[0].cost.estimated_cycles),
                "case={case}"
            );
        }
    }

    #[test]
    fn lowering_encodes_an_all_tile_matching_in_one_epoch() {
        let transfers = (0..736)
            .map(|pair| Transfer {
                source_tile: pair * 2,
                destination_tile: pair * 2 + 1,
                tensor: TensorId(usize::from(pair)),
                bytes: 4,
                staging_address: None,
            })
            .collect();
        let mut schedule = exchange_schedule(transfers);
        schedule.tile_count = 1472;
        let lowered = schedule.lower_exchanges(&Topology::c600()).unwrap();
        assert_eq!(lowered[0].epochs.len(), 1);
        assert_eq!(lowered[0].epochs[0].groups.len(), 736);
        assert_eq!(lowered[0].cost.payload_words, 736);
    }

    #[test]
    fn lowering_encodes_ring_and_all_to_all_fanout_in_one_epoch() {
        let ring = (0..16)
            .map(|source| Transfer {
                source_tile: source,
                destination_tile: (source + 1) % 16,
                tensor: TensorId(usize::from(source)),
                bytes: 4,
                staging_address: None,
            })
            .collect();
        let lowered = exchange_schedule(ring)
            .lower_exchanges(&Topology::c600())
            .unwrap();
        assert_eq!(lowered[0].epochs.len(), 1);
        assert_eq!(lowered[0].cost.launches, 1);

        let all_to_all = (0..8)
            .flat_map(|source| {
                (0..8)
                    .filter(move |destination| *destination != source)
                    .map(move |destination| Transfer {
                        source_tile: source,
                        destination_tile: destination,
                        tensor: TensorId(usize::from(source)),
                        bytes: 4,
                        staging_address: None,
                    })
            })
            .collect();
        let lowered = exchange_schedule(all_to_all)
            .lower_exchanges(&Topology::c600())
            .unwrap();
        assert_eq!(lowered[0].epochs.len(), 1);
        assert_eq!(lowered[0].cost.launches, 1);
        assert_eq!(lowered[0].cost.payload_words, 8);
    }

    #[test]
    fn encoder_reference_is_finite_and_repeatable() {
        let config = EncoderConfig {
            sequence: 4,
            hidden: 8,
            heads: 2,
            feed_forward: 16,
        };
        let input: Vec<_> = (0..config.sequence * config.hidden)
            .map(|index| (index as f32 - 9.0) / 16.0)
            .collect();
        let weights = EncoderWeights::deterministic(config);
        let first = encoder_reference(config, &input, &weights).unwrap();
        let second = encoder_reference(config, &input, &weights).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
        let checksum = first.iter().sum::<f32>();
        assert!((checksum - 107.46423).abs() < 1e-4, "checksum={checksum}");
    }
}
