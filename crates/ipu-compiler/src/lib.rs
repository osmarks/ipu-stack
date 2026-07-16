use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use tracing::{debug, info};

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
    pub specialization: SpecializationKey,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    pub layouts: Vec<Layout>,
    pub phases: Vec<Phase>,
    pub allocations: Vec<Allocation>,
    pub tile_count: u16,
    pub peak_sram: BTreeMap<u16, u32>,
}

pub const TILE_COMMAND_WORDS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TileOpcode {
    Exchange = 1,
    Compute = 2,
    End = 0xff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedTileCommand(pub [u32; TILE_COMMAND_WORDS]);

impl EncodedTileCommand {
    pub fn to_le_bytes(self) -> [u8; TILE_COMMAND_WORDS * 4] {
        let mut bytes = [0; TILE_COMMAND_WORDS * 4];
        for (index, word) in self.0.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

impl Schedule {
    /// Encode the per-tile declarative command stream. Exchange records describe
    /// routing intent; a later lowering pass replaces them with executable plans.
    pub fn tile_commands(&self, tile: u16) -> Result<Vec<EncodedTileCommand>, CompileError> {
        if tile >= self.tile_count {
            return Err(CompileError::Graph(format!("tile {tile} is out of range")));
        }
        let mut encoded = Vec::new();
        for (phase_index, phase) in self.phases.iter().enumerate() {
            match phase {
                Phase::Exchange { transfers } => {
                    for transfer in transfers.iter().filter(|transfer| {
                        transfer.source_tile == tile || transfer.destination_tile == tile
                    }) {
                        encoded.push(EncodedTileCommand([
                            TileOpcode::Exchange as u32,
                            phase_index as u32,
                            u32::from(transfer.source_tile),
                            u32::from(transfer.destination_tile),
                            transfer.tensor.0 as u32,
                            transfer.bytes,
                            0,
                            0,
                        ]));
                    }
                }
                Phase::Compute { op, commands } => {
                    for command in commands.iter().filter(|command| command.tile == tile) {
                        let mut words = [0; TILE_COMMAND_WORDS];
                        words[0] = TileOpcode::Compute as u32;
                        words[1] = phase_index as u32;
                        words[2] = op.0 as u32;
                        words[3] = command.output.0 as u32;
                        words[4] = command.inputs.len() as u32;
                        for (index, input) in command.inputs.iter().take(3).enumerate() {
                            words[5 + index] = input.0 as u32;
                        }
                        encoded.push(EncodedTileCommand(words));
                    }
                }
            }
        }
        encoded.push(EncodedTileCommand([
            TileOpcode::End as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]));
        Ok(encoded)
    }
}

#[derive(Clone, Debug)]
pub struct CompilerOptions {
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            tile_count: DEFAULT_TILE_COUNT,
            data_base: 0x52000,
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
    if options.tile_count == 0 || options.data_base >= options.data_limit {
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
        peak_sram
            .entry(allocation.tile)
            .and_modify(|peak| {
                *peak = (*peak).max(allocation.address + allocation.size - options.data_base)
            })
            .or_insert(allocation.address + allocation.size - options.data_base);
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
            let existing = by_tile.entry(*tile).or_default();
            let mut address = options.data_base;
            loop {
                let end = address
                    .checked_add(size)
                    .ok_or_else(|| CompileError::Memory("address overflow".into()))?;
                let conflict = existing
                    .iter()
                    .filter(|allocation| {
                        lifetimes_overlap(
                            produced[tensor.id.0],
                            consumed[tensor.id.0],
                            allocation.live_from,
                            allocation.live_until,
                        )
                    })
                    .find(|allocation| {
                        address < allocation.address + allocation.size && allocation.address < end
                    });
                if let Some(conflict) = conflict {
                    address = align_u32(conflict.address + conflict.size, layout.alignment);
                    continue;
                }
                if end > options.data_limit {
                    return Err(CompileError::Memory(format!(
                        "tile {tile} exceeds data limit allocating {}",
                        tensor.name
                    )));
                }
                let allocation = Allocation {
                    tensor: tensor.id,
                    tile: *tile,
                    address,
                    size,
                    live_from: produced[tensor.id.0],
                    live_until: consumed[tensor.id.0],
                };
                existing.push(allocation.clone());
                allocations.push(allocation);
                break;
            }
        }
    }
    Ok(allocations)
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

    #[test]
    fn encoder_graph_compiles_deterministically() {
        let graph = encoder_graph(EncoderConfig::default()).unwrap();
        let first = compile(&graph, &CompilerOptions::default()).unwrap();
        let second = compile(&graph, &CompilerOptions::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(graph.ops.len(), 15);
        assert!(first.phases.len() >= graph.ops.len());
        assert!(first.peak_sram.values().all(|peak| *peak < 0x8e000));
        let commands = first.tile_commands(0).unwrap();
        assert_eq!(commands.last().unwrap().0[0], TileOpcode::End as u32);
        assert_eq!(commands.last().unwrap().to_le_bytes().len(), 32);
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
