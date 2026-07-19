use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use tracing::{debug, info};

use ipu_exchange::{
    MulticastPlan, PlanProgramBuilder, PlanRow, RETURN_M10_INSTRUCTION, SANS_INACTIVE_INSTRUCTION,
    SYNC_ANS_INSTRUCTION, Topology, finalize_point_receiver, patch_receiver_address,
    patch_sender_address,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_TILE_COUNT: u16 = 64;

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
    pub operation: String,
    pub shape: Vec<usize>,
    pub worker_count: u8,
    pub role: String,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Exchange {
        transfers: Vec<Transfer>,
    },
    Compute {
        op: OpId,
        commands: Vec<KernelCommand>,
    },
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
    ExchangeStaging { phase: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPlacement {
    Low,
    High,
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
    let alignment = constraint.alignment;
    let start = match constraint.placement {
        MemoryPlacement::Low => align_u32(constraint.base, alignment),
        MemoryPlacement::High => (constraint.limit - size) & !(alignment - 1),
    };
    let fits = |address: u32| {
        let end = address + size;
        end <= constraint.limit
            && allocations.iter().all(|allocation| {
                allocation.tile != tile
                    || live_from >= allocation.live_until
                    || allocation.live_from >= live_until
                    || end <= allocation.address
                    || address >= allocation.address.saturating_add(allocation.size)
            })
    };

    match constraint.placement {
        MemoryPlacement::Low => {
            let mut address = start;
            while address <= constraint.limit - size {
                if fits(address) {
                    return Ok(address);
                }
                address = address
                    .checked_add(alignment)
                    .ok_or_else(|| CompileError::Memory("allocation address overflow".into()))?;
            }
        }
        MemoryPlacement::High => {
            let mut address = start;
            loop {
                if address >= constraint.base && fits(address) {
                    return Ok(address);
                }
                let Some(previous) = address.checked_sub(alignment) else {
                    break;
                };
                address = previous;
                if address < constraint.base {
                    break;
                }
            }
        }
    }
    Err(CompileError::Memory(format!(
        "no {size}-byte region for tile {tile} in 0x{:x}..0x{:x}",
        constraint.base, constraint.limit
    )))
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
    pub dimension: u16,
    pub block_dimension: u16,
    pub inner_block_dimension: u16,
    pub row_block_dimension: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedGemmPlan {
    pub schedule: Schedule,
    pub left: Vec<BlockPlacement>,
    pub right: Vec<BlockPlacement>,
    pub output: Vec<BlockPlacement>,
}

pub fn choose_gemm_row_block(
    dimension: u16,
    inner_block_dimension: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Option<u16> {
    let candidates = gemm_row_block_candidates(
        dimension,
        inner_block_dimension,
        block_dimension,
        tile_count,
    );
    let column_blocks = dimension.checked_div(block_dimension)?;
    candidates.into_iter().min_by_key(|&target| {
        let row_shards = dimension.div_ceil(target);
        let maximum_rows = dimension.div_ceil(row_shards);
        let output_blocks = usize::from(row_shards) * usize::from(column_blocks);
        let waves = output_blocks.div_ceil(usize::from(tile_count));
        let unused_tiles = waves * usize::from(tile_count) - output_blocks;

        // Each wave runs at the pace of its largest row shard. Prefer the
        // shortest aggregate critical path, then fewer barriers and fuller waves.
        (waves * usize::from(maximum_rows), waves, unused_tiles)
    })
}

pub fn gemm_row_block_candidates(
    dimension: u16,
    inner_block_dimension: u16,
    block_dimension: u16,
    tile_count: u16,
) -> Vec<u16> {
    if dimension == 0
        || block_dimension == 0
        || inner_block_dimension == 0
        || tile_count == 0
        || !dimension.is_multiple_of(block_dimension)
        || !dimension.is_multiple_of(inner_block_dimension)
    {
        return Vec::new();
    }
    let mut row_shard_counts = BTreeSet::new();
    let maximum_rows =
        (ipu_exchange::MAX_TRANSFER_WORDS * 4 / (u32::from(inner_block_dimension) * 4)).min(
            (ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
                - ipu_package::IPU21_INTERLEAVED_MEMORY_BASE)
                / (u32::from(block_dimension) * 4),
        ) as u16;
    (12..=maximum_rows)
        .filter(|&target| {
            let row_shards = dimension.div_ceil(target);
            row_shard_counts.insert(row_shards)
        })
        .collect()
}

pub fn plan_blocked_gemm(config: BlockedGemmConfig) -> Result<BlockedGemmPlan, CompileError> {
    if config.dimension == 0
        || config.block_dimension != 64
        || !matches!(config.inner_block_dimension, 32 | 64)
        || !config.dimension.is_multiple_of(config.block_dimension)
        || !config
            .dimension
            .is_multiple_of(config.inner_block_dimension)
        || config.data_base >= config.data_limit
    {
        return Err(CompileError::Graph(
            "blocked GEMM requires 64-column blocks and 32- or 64-wide inner blocks".into(),
        ));
    }
    let column_grid = config.dimension / config.block_dimension;
    let inner_grid = config.dimension / config.inner_block_dimension;
    let row_grid = config.dimension.div_ceil(config.row_block_dimension);
    let base_rows = config.dimension / row_grid;
    let larger_row_shards = config.dimension % row_grid;
    let output_block_count = usize::from(row_grid) * usize::from(column_grid);
    let block_bytes = u32::from(config.inner_block_dimension)
        .checked_mul(u32::from(config.block_dimension))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| CompileError::Memory("GEMM block size overflow".into()))?;
    if block_bytes > ipu_exchange::MAX_TRANSFER_WORDS * 4 {
        return Err(CompileError::Graph(format!(
            "{block_bytes}-byte GEMM blocks exceed one exchange transfer"
        )));
    }
    let maximum_rows = base_rows + u16::from(larger_row_shards != 0);
    let max_left_bytes = u32::from(maximum_rows)
        .checked_mul(u32::from(config.inner_block_dimension))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| CompileError::Memory("GEMM left block size overflow".into()))?;
    let max_output_bytes = u32::from(maximum_rows)
        .checked_mul(u32::from(config.block_dimension))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| CompileError::Memory("GEMM output block size overflow".into()))?;
    let tile_data_end = config
        .data_base
        .checked_add(max_left_bytes)
        .and_then(|end| end.checked_add(block_bytes))
        .ok_or_else(|| CompileError::Memory("GEMM per-tile data address overflow".into()))?;
    let output_address = ipu_package::IPU21_INTERLEAVED_MEMORY_BASE;
    let output_end = output_address
        .checked_add(max_output_bytes)
        .ok_or_else(|| CompileError::Memory("GEMM output address overflow".into()))?;
    if config.data_base & 31 != 0
        || tile_data_end > config.data_limit
        || output_end > ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT
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
    let evacuation_tensor_base = scratch_tensor_base + output_block_count;
    let mut allocations = Vec::new();
    let mut left = Vec::with_capacity(left_count);
    let mut right = Vec::with_capacity(right_count);
    let mut output = Vec::with_capacity(output_block_count);
    let mut data_cursors = vec![config.data_base; usize::from(config.tile_count)];

    for block_row in 0..row_grid {
        let rows = base_rows + u16::from(block_row < larger_row_shards);
        let row_start = block_row * base_rows + block_row.min(larger_row_shards);
        let size = u32::from(rows) * u32::from(config.inner_block_dimension) * 4;
        for block_column in 0..inner_grid {
            let index =
                usize::from(block_row) * usize::from(inner_grid) + usize::from(block_column);
            let tile = u16::try_from(index % usize::from(config.tile_count))
                .map_err(|_| CompileError::Graph("GEMM tile index overflow".into()))?;
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
            let tile = u16::try_from((left_count + index) % usize::from(config.tile_count))
                .map_err(|_| CompileError::Graph("GEMM tile index overflow".into()))?;
            let tensor = TensorId(left_count + index);
            let address = data_cursors[usize::from(tile)];
            data_cursors[usize::from(tile)] = address
                .checked_add(block_bytes)
                .ok_or_else(|| CompileError::Memory("GEMM data address overflow".into()))?;
            allocations.push(Allocation {
                tensor,
                tile,
                address,
                size: block_bytes,
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
        let size = u32::from(rows) * u32::from(config.block_dimension) * 4;
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
    let mut phases = Vec::with_capacity(output_waves * (usize::from(inner_grid) * 2 + 2));
    for (wave, wave_outputs) in output.chunks(usize::from(config.tile_count)).enumerate() {
        let wave_start = phases.len();
        for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
            let output_index = wave * usize::from(config.tile_count) + wave_tile;
            allocations.push(Allocation {
                tensor: TensorId(scratch_tensor_base + output_index),
                tile: u16::try_from(wave_tile)
                    .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?,
                address: output_address,
                size: u32::from(output_block.rows) * u32::from(config.block_dimension) * 4,
                live_from: wave_start,
                live_until: wave_start + usize::from(inner_grid) * 2 + 1,
                kind: AllocationKind::Home,
            });
        }
        for inner_block in 0..inner_grid {
            let exchange_phase = phases.len();
            let mut transfers = Vec::new();
            for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
                let destination_tile = u16::try_from(wave_tile)
                    .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?;
                let source_index = usize::from(output_block.block_row) * usize::from(inner_grid)
                    + usize::from(inner_block);
                let source = left[source_index];
                if source.tile != destination_tile {
                    transfers.push(Transfer {
                        source_tile: source.tile,
                        destination_tile,
                        tensor: source.tensor,
                        bytes: u32::from(source.rows) * u32::from(config.inner_block_dimension) * 4,
                    });
                    allocations.push(Allocation {
                        tensor: source.tensor,
                        tile: destination_tile,
                        address: left_exchange_address,
                        size: u32::from(source.rows) * u32::from(config.inner_block_dimension) * 4,
                        live_from: exchange_phase,
                        live_until: exchange_phase + 1,
                        kind: AllocationKind::ExchangeStaging {
                            phase: exchange_phase,
                        },
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
                        bytes: block_bytes,
                    });
                    allocations.push(Allocation {
                        tensor: source.tensor,
                        tile: destination_tile,
                        address: right_exchange_address,
                        size: block_bytes,
                        live_from: exchange_phase,
                        live_until: exchange_phase + 1,
                        kind: AllocationKind::ExchangeStaging {
                            phase: exchange_phase,
                        },
                    });
                }
            }
            phases.push(Phase::Exchange { transfers });

            let gemm_phase = phases.len();
            let mut gemm_commands = Vec::with_capacity(wave_outputs.len());
            for (wave_tile, output_block) in wave_outputs.iter().enumerate() {
                let output_index = wave * usize::from(config.tile_count) + wave_tile;
                let left_tensor = left[usize::from(output_block.block_row)
                    * usize::from(inner_grid)
                    + usize::from(inner_block)]
                .tensor;
                let right_tensor = right[usize::from(inner_block) * usize::from(column_grid)
                    + usize::from(output_block.block_column)]
                .tensor;
                gemm_commands.push(KernelCommand {
                    tile: u16::try_from(wave_tile)
                        .map_err(|_| CompileError::Graph("GEMM wave tile overflow".into()))?,
                    output: TensorId(scratch_tensor_base + output_index),
                    inputs: vec![left_tensor, right_tensor],
                    arguments: vec![
                        u32::from(output_block.rows),
                        u32::from(output_block.rows / 6),
                        u32::from(output_block.rows % 6),
                    ],
                    specialization: SpecializationKey {
                        operation: if inner_block == 0 {
                            "gemm_f32_init"
                        } else {
                            "gemm_f32_accumulate"
                        }
                        .into(),
                        shape: vec![
                            usize::from(output_block.rows),
                            usize::from(config.inner_block_dimension),
                            usize::from(config.block_dimension),
                        ],
                        worker_count: 6,
                        role: format!("inner-block-{inner_block}"),
                        alignment: 32,
                    },
                    metadata: BTreeMap::from([
                        (
                            "label".into(),
                            format!(
                                "GEMM block ({}, {}) inner block {}",
                                output_block.block_row, output_block.block_column, inner_block
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
                            (u32::from(output_block.rows) * u32::from(config.block_dimension) * 4)
                                .to_string(),
                        ),
                        ("block_dimension".into(), config.block_dimension.to_string()),
                        (
                            "inner_block_dimension".into(),
                            config.inner_block_dimension.to_string(),
                        ),
                    ]),
                });
            }
            phases.push(Phase::Compute {
                op: OpId(gemm_phase),
                commands: gemm_commands,
            });
        }
        let evacuation_phase = phases.len();
        let max_transfer_rows =
            u16::try_from(ipu_exchange::MAX_TRANSFER_WORDS / u32::from(config.block_dimension))
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
                let byte_offset = u32::from(row_offset) * u32::from(config.block_dimension) * 4;
                let bytes = u32::from(rows) * u32::from(config.block_dimension) * 4;
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
                    address: left_exchange_address + byte_offset,
                    size: bytes,
                    live_from: evacuation_phase,
                    live_until: evacuation_phase + 1,
                    kind: AllocationKind::ExchangeStaging {
                        phase: evacuation_phase,
                    },
                });
                allocations.push(Allocation {
                    tensor,
                    tile: output_block.tile,
                    address: output_block.address + byte_offset,
                    size: bytes,
                    live_from: evacuation_phase + 1,
                    live_until: usize::MAX,
                    kind: AllocationKind::Home,
                });
                transfers.push(Transfer {
                    source_tile,
                    destination_tile: output_block.tile,
                    tensor,
                    bytes,
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
                    shape: vec![usize::try_from(units).map_err(|_| {
                        CompileError::Graph("GEMM output copy size overflow".into())
                    })?],
                    worker_count: 6,
                    role: format!(
                        "output-wave-{wave}-rows-{row_offset}..{}",
                        row_offset + rows
                    ),
                    alignment: 8,
                },
                metadata: BTreeMap::from([
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
                ]),
            });
        }
        phases.push(Phase::Compute {
            op: OpId(copy_phase),
            commands: copy_commands,
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
    pub output: TensorId,
    pub inputs: Vec<TensorId>,
    pub output_address: u32,
    pub input_addresses: Vec<u32>,
    pub arguments: Vec<u32>,
    pub specialization: SpecializationKey,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoweredTileStep {
    Exchange {
        phase: usize,
        epoch: usize,
        row: Vec<u32>,
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
    pub fn lower_exchanges(
        &self,
        topology: &Topology,
    ) -> Result<Vec<LoweredExchangePhase>, CompileError> {
        if topology.tile_count() < usize::from(self.tile_count) {
            return Err(CompileError::Graph(
                "exchange topology has too few tiles".into(),
            ));
        }
        let allocation_index = AllocationIndex::new(&self.allocations);
        #[derive(Clone)]
        struct PendingGroup {
            source: u16,
            tensor: TensorId,
            bytes: u32,
            destinations: Vec<u16>,
        }

        let mut lowered_phases = Vec::new();
        for (phase_index, phase) in self.phases.iter().enumerate() {
            let Phase::Exchange { transfers } = phase else {
                continue;
            };
            validate_transfers(transfers)?;
            let mut groups: Vec<PendingGroup> = Vec::new();
            for transfer in transfers {
                if let Some(group) = groups.iter_mut().find(|group| {
                    group.source == transfer.source_tile
                        && group.tensor == transfer.tensor
                        && group.bytes == transfer.bytes
                }) {
                    group.destinations.push(transfer.destination_tile);
                } else {
                    groups.push(PendingGroup {
                        source: transfer.source_tile,
                        tensor: transfer.tensor,
                        bytes: transfer.bytes,
                        destinations: vec![transfer.destination_tile],
                    });
                }
            }
            for group in &mut groups {
                group.destinations.sort_unstable();
                group.destinations.dedup();
            }

            // A tile can execute one exchange role at a time. Color the
            // multicast-hyperedge conflict graph into timed slots with deterministic DSATUR.
            let adjacency: Vec<HashSet<usize>> = groups
                .iter()
                .enumerate()
                .map(|(left_index, left)| {
                    groups
                        .iter()
                        .enumerate()
                        .filter(|(right_index, right)| {
                            left_index != *right_index
                                && exchange_groups_conflict(
                                    left.source,
                                    &left.destinations,
                                    right.source,
                                    &right.destinations,
                                )
                        })
                        .map(|(index, _)| index)
                        .collect()
                })
                .collect();
            let mut colors = vec![None; groups.len()];
            for _ in 0..groups.len() {
                let index = (0..groups.len())
                    .filter(|index| colors[*index].is_none())
                    .max_by_key(|index| {
                        let saturation: HashSet<_> = adjacency[*index]
                            .iter()
                            .filter_map(|neighbor| colors[*neighbor])
                            .collect();
                        (
                            saturation.len(),
                            adjacency[*index].len(),
                            std::cmp::Reverse(groups[*index].source),
                            std::cmp::Reverse(groups[*index].tensor.0),
                        )
                    })
                    .ok_or_else(|| CompileError::Graph("exchange coloring failed".into()))?;
                let unavailable: HashSet<_> = adjacency[index]
                    .iter()
                    .filter_map(|neighbor| colors[*neighbor])
                    .collect();
                colors[index] = Some(
                    (0..)
                        .find(|color| !unavailable.contains(color))
                        .ok_or_else(|| CompileError::Graph("exchange color overflow".into()))?,
                );
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
            let mut available: HashSet<_> = self
                .allocations
                .iter()
                .filter(|allocation| {
                    allocation.kind == AllocationKind::Home
                        || matches!(
                            allocation.kind,
                            AllocationKind::ExchangeStaging { phase }
                                if phase < phase_index
                        ) && allocation.live_from <= phase_index
                            && allocation.live_until > phase_index
                })
                .map(|allocation| (allocation.tensor, allocation.tile))
                .collect();
            let available_before_phase = available.clone();
            let mut epoch_groups = Vec::with_capacity(colored_groups.len());
            while !colored_groups.is_empty() {
                let ready = colored_groups
                    .iter()
                    .position(|slot| {
                        slot.iter()
                            .all(|group| available.contains(&(group.tensor, group.source)))
                    })
                    .ok_or_else(|| {
                        CompileError::Graph("exchange staging dependencies contain a cycle".into())
                    })?;
                let slot = colored_groups.remove(ready);
                for group in &slot {
                    available.extend(
                        group
                            .destinations
                            .iter()
                            .map(|destination| (group.tensor, *destination)),
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
                    destinations,
                } in pending
                {
                    if bytes == 0 || bytes & 3 != 0 {
                        return Err(CompileError::Graph(format!(
                            "tensor {} exchange size is not whole words",
                            tensor.0
                        )));
                    }
                    let words = bytes / 4;
                    let candidates = allocation_index.at(tensor, source);
                    let same_phase_staging = || {
                        candidates.iter().copied().find(|allocation| {
                            allocation.kind
                                == AllocationKind::ExchangeStaging { phase: phase_index }
                        })
                    };
                    let earlier_staging = || {
                        candidates.iter().copied().find(|allocation| {
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
                            .iter()
                            .copied()
                            .find(|allocation| allocation.kind == AllocationKind::Home)
                    };
                    let source_address = if available_before_phase.contains(&(tensor, source)) {
                        earlier_staging().or_else(home).or_else(same_phase_staging)
                    } else {
                        same_phase_staging().or_else(earlier_staging).or_else(home)
                    }
                    .ok_or_else(|| {
                        CompileError::Memory(format!(
                            "missing source allocation for tensor {} on tile {source}",
                            tensor.0
                        ))
                    })?
                    .address;
                    let destination_addresses = destinations
                        .iter()
                        .map(|destination| {
                            allocation_index
                                .at(tensor, *destination)
                                .iter()
                                .copied()
                                .find(|allocation| {
                                    allocation.kind
                                        == AllocationKind::ExchangeStaging { phase: phase_index }
                                })
                                .map(|allocation| allocation.address)
                                .ok_or_else(|| {
                                    CompileError::Memory(format!(
                                        "missing staging allocation for tensor {} on tile {destination}",
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
        let exchanges = self.lower_exchanges(topology)?;
        let exchange_by_phase: HashMap<_, _> = exchanges
            .iter()
            .map(|exchange| (exchange.phase, exchange))
            .collect();
        let allocation_index = AllocationIndex::new(&self.allocations);
        let mut programs = Vec::with_capacity(usize::from(self.tile_count));
        for tile in 0..self.tile_count {
            let mut steps = Vec::new();
            for (phase_index, phase) in self.phases.iter().enumerate() {
                match phase {
                    Phase::Exchange { .. } => {
                        let exchange = exchange_by_phase.get(&phase_index).ok_or_else(|| {
                            CompileError::Graph(format!(
                                "missing lowered exchange phase {phase_index}"
                            ))
                        })?;
                        for (epoch, lowered) in exchange.epochs.iter().enumerate() {
                            steps.push(LoweredTileStep::Exchange {
                                phase: phase_index,
                                epoch,
                                row: lowered.row_for(tile),
                            });
                        }
                    }
                    Phase::Compute { op, commands } => {
                        let mut active = false;
                        for command in commands.iter().filter(|command| command.tile == tile) {
                            active = true;
                            let output_address =
                                allocation_index.home_address(command.output, tile)?;
                            let input_addresses = command
                                .inputs
                                .iter()
                                .map(|input| {
                                    allocation_index.compute_input_address(
                                        *input,
                                        tile,
                                        phase_index,
                                    )
                                })
                                .collect::<Result<_, _>>()?;
                            steps.push(LoweredTileStep::Compute(LoweredComputeCommand {
                                op: *op,
                                phase: phase_index,
                                output: command.output,
                                inputs: command.inputs.clone(),
                                output_address,
                                input_addresses,
                                arguments: command.arguments.clone(),
                                specialization: command.specialization.clone(),
                                metadata: command.metadata.clone(),
                            }));
                        }
                        if !active {
                            steps.push(LoweredTileStep::IdleCompute {
                                op: *op,
                                phase: phase_index,
                            });
                        }
                    }
                }
            }
            programs.push(LoweredTileProgram { tile, steps });
        }
        info!(tiles = programs.len(), "lowered per-tile programs");
        Ok(programs)
    }
}

struct AllocationIndex<'a> {
    by_location: HashMap<(TensorId, u16), Vec<&'a Allocation>>,
}

impl<'a> AllocationIndex<'a> {
    fn new(allocations: &'a [Allocation]) -> Self {
        let mut by_location = HashMap::<_, Vec<_>>::new();
        for allocation in allocations {
            by_location
                .entry((allocation.tensor, allocation.tile))
                .or_default()
                .push(allocation);
        }
        Self { by_location }
    }

    fn at(&self, tensor: TensorId, tile: u16) -> &[&'a Allocation] {
        self.by_location
            .get(&(tensor, tile))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn home_address(&self, tensor: TensorId, tile: u16) -> Result<u32, CompileError> {
        self.at(tensor, tile)
            .iter()
            .copied()
            .find(|allocation| allocation.kind == AllocationKind::Home)
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
        if let Some(staging) = self.at(tensor, tile).iter().copied().find(|allocation| {
            allocation.live_until == compute_phase
                && matches!(allocation.kind, AllocationKind::ExchangeStaging { .. })
        }) {
            return Ok(staging.address);
        }
        self.home_address(tensor, tile)
    }
}

fn exchange_groups_conflict(
    left_source: u16,
    left_destinations: &[u16],
    right_source: u16,
    right_destinations: &[u16],
) -> bool {
    left_source == right_source
        || left_destinations.contains(&right_source)
        || right_destinations.contains(&left_source)
        || left_destinations
            .iter()
            .any(|tile| right_destinations.contains(tile))
}

#[derive(Clone, Debug)]
pub struct CompilerOptions {
    pub tile_count: u16,
    pub exchange_base: u32,
    pub exchange_limit: u32,
    pub data_base: u32,
    pub data_limit: u32,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            tile_count: DEFAULT_TILE_COUNT,
            exchange_base: 0x50000,
            exchange_limit: 0x58000,
            data_base: 0x58000,
            data_limit: 0xe0000,
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
        || options.exchange_limit > options.data_base
        || options.data_base >= options.data_limit
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
            .map(|tile| KernelCommand {
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
            .collect();
        phases.push(Phase::Compute {
            op: op.id,
            commands,
        });
    }
    let allocations = plan_memory(graph, &layouts, &phases, options)?;
    let mut peak_sram: BTreeMap<u16, u32> = BTreeMap::new();
    for allocation in &allocations {
        let memory_base = options.exchange_base.min(options.data_base);
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
    let mut destinations = HashSet::new();
    for transfer in transfers {
        if transfer.source_tile == transfer.destination_tile || transfer.bytes == 0 {
            return Err(CompileError::Graph("invalid exchange transfer".into()));
        }
        if !destinations.insert((transfer.destination_tile, transfer.tensor)) {
            return Err(CompileError::Graph(
                "multiple sends target one tensor region in an epoch".into(),
            ));
        }
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
                    consumed[input.0] = consumed[input.0].max(phase_index);
                }
            }
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    consumed[transfer.tensor.0] = consumed[transfer.tensor.0].max(phase_index);
                }
            }
        }
    }
    for output in &graph.outputs {
        consumed[output.0] = phases.len();
    }

    let mut allocations = Vec::new();
    let mut by_tile: HashMap<u16, Vec<Allocation>> = HashMap::new();
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
                options.data_base,
                options.data_limit,
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
                options.exchange_base,
                options.exchange_limit,
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
    allocation_base: u32,
    allocation_limit: u32,
    label: &str,
) -> Result<(), CompileError> {
    let existing = by_tile.entry(tile).or_default();
    let mut address = allocation_base;
    loop {
        let end = address
            .checked_add(size)
            .ok_or_else(|| CompileError::Memory("address overflow".into()))?;
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
        if end > allocation_limit {
            return Err(CompileError::Memory(format!(
                "tile {tile} exceeds data limit allocating {label}"
            )));
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

fn lifetimes_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn align_u32(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
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
    fn blocked_gemm_plan_preserves_block_ownership_and_phase_dependencies() {
        let plan = plan_blocked_gemm(BlockedGemmConfig {
            dimension: 128,
            block_dimension: 64,
            inner_block_dimension: 32,
            row_block_dimension: 64,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
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
                Phase::Exchange { transfers } => transfers
                    .iter()
                    .all(|transfer| transfer.bytes <= ipu_exchange::MAX_TRANSFER_WORDS * 4),
                Phase::Compute { commands, .. } => commands.iter().all(|command| {
                    let units = u32::try_from(command.specialization.shape[0]).unwrap();
                    command.inputs.len() == 2
                        && command.arguments == [units, units / 6, units % 6]
                        && matches!(
                            command.specialization.operation.as_str(),
                            "gemm_f32_init" | "gemm_f32_accumulate" | "copy_u64"
                        )
                        && command.metadata.contains_key("label")
                        && command.metadata.contains_key("wave")
                        && command.metadata.contains_key("output_block_row")
                        && command.metadata.contains_key("output_block_column")
                }),
            }
        }));
    }

    #[test]
    fn blocked_gemm_balances_rows_without_gaps_or_tile_overcommitment() {
        for dimension in [64, 128, 256, 1024, 1920, 2048, 2304, 4096] {
            let target = choose_gemm_row_block(dimension, 32, 64, 1472).unwrap();
            let plan = plan_blocked_gemm(BlockedGemmConfig {
                dimension,
                block_dimension: 64,
                inner_block_dimension: 32,
                row_block_dimension: target,
                tile_count: 1472,
                data_base: 0xa0000,
                data_limit: 0xe8000,
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
            assert!(plan.schedule.phases.iter().all(|phase| match phase {
                Phase::Compute { commands, .. } => {
                    let mut per_tile = BTreeMap::<u16, usize>::new();
                    for command in commands {
                        *per_tile.entry(command.tile).or_default() += 1;
                    }
                    per_tile.values().all(|&count| count <= 2)
                }
                Phase::Exchange { .. } => true,
            }));
        }
    }

    #[test]
    fn gemm_tuning_candidates_are_feasible_and_layout_distinct() {
        for dimension in [64, 128, 256, 1024, 1920, 2048, 2304, 4096] {
            let candidates = gemm_row_block_candidates(dimension, 32, 64, 1472);
            assert!(!candidates.is_empty());
            assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
            let row_shards: BTreeSet<_> = candidates
                .iter()
                .map(|target| dimension.div_ceil(*target))
                .collect();
            assert_eq!(row_shards.len(), candidates.len());
            assert!(candidates.contains(&choose_gemm_row_block(dimension, 32, 64, 1472).unwrap()));
        }
        assert!(choose_gemm_row_block(65, 32, 64, 1472).is_none());
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
            if !allocations.iter().any(|allocation| {
                allocation.tensor == transfer.tensor
                    && allocation.tile == transfer.destination_tile
                    && allocation.kind == AllocationKind::ExchangeStaging { phase: 0 }
            }) {
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
            },
            Transfer {
                source_tile: 0,
                destination_tile: 2,
                tensor: TensorId(0),
                bytes: 64,
            },
            Transfer {
                source_tile: 3,
                destination_tile: 4,
                tensor: TensorId(1),
                bytes: 128,
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
    fn lowering_uses_point_schedule_for_an_independent_single_destination() {
        let topology = Topology::c600();
        let schedule = exchange_schedule(vec![Transfer {
            source_tile: 0,
            destination_tile: 1,
            tensor: TensorId(0),
            bytes: 64,
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
            },
            Transfer {
                source_tile: 2,
                destination_tile: 3,
                tensor: TensorId(1),
                bytes: 64,
            },
            Transfer {
                source_tile: 1,
                destination_tile: 2,
                tensor: TensorId(2),
                bytes: 64,
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
            },
            Transfer {
                source_tile: 1,
                destination_tile: 3,
                tensor,
                bytes: 256,
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
                    }],
                },
                Phase::Exchange {
                    transfers: vec![Transfer {
                        source_tile: 1,
                        destination_tile: 2,
                        tensor,
                        bytes: 64,
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
    fn scheduler_is_deterministic_for_varied_transfer_graphs() {
        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        for _ in 0..64 {
            let mut transfers = Vec::new();
            let mut destinations = HashSet::new();
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
