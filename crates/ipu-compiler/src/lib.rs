use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use tracing::{debug, info};

use ipu_exchange::{
    MulticastPlan, PlanRow, Topology, finalize_point_receiver, patch_multicast_receiver_address,
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
    pub kind: AllocationKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationKind {
    Home,
    ExchangeStaging { phase: usize },
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
    pub tile_rows: BTreeMap<u16, PlanRow>,
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
    pub output_address: u32,
    pub input_addresses: Vec<u32>,
    pub specialization: SpecializationKey,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoweredTileStep {
    Exchange {
        phase: usize,
        epoch: usize,
        row: PlanRow,
    },
    Compute(LoweredComputeCommand),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoweredTileProgram {
    pub tile: u16,
    pub steps: Vec<LoweredTileStep>,
}

impl LoweredExchangeEpoch {
    pub fn row_for(&self, tile: u16) -> PlanRow {
        self.tile_rows.get(&tile).copied().unwrap_or_else(|| {
            let mut row = [0; ipu_exchange::PLAN_WORDS];
            // The runtime performs the all-tile epoch barrier. Inactive tiles
            // then use the SDK's local non-participation sequence.
            row[0] = 0x40c0_0000;
            row[1] = 0x4180_0001;
            row[2] = 0x43a0_0000;
            row
        })
    }
}

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

    pub fn lower_exchanges(
        &self,
        topology: &Topology,
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

            // A tile has one supervisor exchange role in an epoch. Color the
            // multicast-hyperedge conflict graph with deterministic DSATUR.
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
            let mut epoch_groups = vec![Vec::new(); color_count];
            for (group, color) in groups.into_iter().zip(colors) {
                let color =
                    color.ok_or_else(|| CompileError::Graph("uncolored exchange group".into()))?;
                epoch_groups[color].push(group);
            }
            let mut epochs = Vec::new();
            for pending in epoch_groups {
                let mut lowered_groups = Vec::new();
                let mut tile_rows = BTreeMap::new();
                let mut cost = ExchangeCost {
                    launches: 1,
                    ..ExchangeCost::default()
                };
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
                    let source_address = self
                        .allocations
                        .iter()
                        .find(|allocation| {
                            allocation.tensor == tensor
                                && allocation.tile == source
                                && allocation.kind == AllocationKind::Home
                        })
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
                            self.allocations
                                .iter()
                                .find(|allocation| {
                                    allocation.tensor == tensor
                                        && allocation.tile == *destination
                                        && allocation.kind
                                            == AllocationKind::ExchangeStaging {
                                                phase: phase_index,
                                            }
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
                    let (sender, receivers, addressing) = if destinations.len() == 1 {
                        let plan = topology.point_to_point(source, destinations[0], words)?;
                        let receiver =
                            finalize_point_receiver(&plan.receiver, topology.physical(source)?)?;
                        (plan.sender, vec![receiver], ExchangeAddressing::Relative)
                    } else {
                        let mut plan: MulticastPlan =
                            topology.multicast(source, &destinations, words, 0)?;
                        patch_sender_address(&mut plan.sender, source_address)?;
                        for (receiver, address) in plan
                            .receivers
                            .iter_mut()
                            .zip(destination_addresses.iter().copied())
                        {
                            patch_multicast_receiver_address(receiver, address)?;
                        }
                        (plan.sender, plan.receivers, ExchangeAddressing::Absolute)
                    };
                    if tile_rows.insert(source, sender).is_some() {
                        return Err(CompileError::Graph(
                            "sender scheduled twice in one epoch".into(),
                        ));
                    }
                    for (destination, receiver) in
                        destinations.iter().copied().zip(receivers.iter().copied())
                    {
                        if tile_rows.insert(destination, receiver).is_some() {
                            return Err(CompileError::Graph(
                                "receiver scheduled twice in one epoch".into(),
                            ));
                        }
                    }
                    cost.estimated_cycles = cost.estimated_cycles.max(u64::from(156 + words));
                    cost.payload_words += u64::from(words);
                    lowered_groups.push(LoweredExchangeGroup {
                        source_tile: source,
                        destination_tiles: destinations,
                        tensor,
                        bytes,
                        addressing,
                        sender,
                        receivers,
                    });
                }
                epochs.push(LoweredExchangeEpoch {
                    groups: lowered_groups,
                    tile_rows,
                    cost,
                });
            }
            let phase_cost = epochs
                .iter()
                .fold(ExchangeCost::default(), |mut total, epoch| {
                    total.launches += epoch.cost.launches;
                    total.estimated_cycles += epoch.cost.estimated_cycles;
                    total.payload_words += epoch.cost.payload_words;
                    total
                });
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
                        for command in commands.iter().filter(|command| command.tile == tile) {
                            let output_address = self.home_address(command.output, tile)?;
                            let input_addresses = command
                                .inputs
                                .iter()
                                .map(|input| self.compute_input_address(*input, tile, phase_index))
                                .collect::<Result<_, _>>()?;
                            steps.push(LoweredTileStep::Compute(LoweredComputeCommand {
                                op: *op,
                                output_address,
                                input_addresses,
                                specialization: command.specialization.clone(),
                            }));
                        }
                    }
                }
            }
            programs.push(LoweredTileProgram { tile, steps });
        }
        info!(tiles = programs.len(), "lowered per-tile programs");
        Ok(programs)
    }

    fn home_address(&self, tensor: TensorId, tile: u16) -> Result<u32, CompileError> {
        self.allocations
            .iter()
            .find(|allocation| {
                allocation.tensor == tensor
                    && allocation.tile == tile
                    && allocation.kind == AllocationKind::Home
            })
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
        if let Some(staging) = self.allocations.iter().find(|allocation| {
            allocation.tensor == tensor
                && allocation.tile == tile
                && allocation.live_until == compute_phase
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
    fn scheduler_coalesces_fanout_and_packs_disjoint_groups() {
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
        assert_eq!(inactive[0], 0x40c0_0000);
        assert_eq!(inactive[1], 0x4180_0001);
    }

    #[test]
    fn scheduler_splits_tile_role_conflicts() {
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
        assert_eq!(lowered[0].epochs.len(), 2);
        assert_eq!(lowered[0].cost.launches, 2);
        for epoch in &lowered[0].epochs {
            let mut active = HashSet::new();
            for group in &epoch.groups {
                assert!(active.insert(group.source_tile));
                for destination in &group.destination_tiles {
                    assert!(active.insert(*destination));
                }
            }
        }
    }

    #[test]
    fn scheduler_is_deterministic_for_varied_transfer_graphs() {
        let mut state = 0x1234_5678u32;
        for _ in 0..64 {
            let mut transfers = Vec::new();
            let mut destinations = HashSet::new();
            for tensor in 0..12 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let source = (state % 16) as u16;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let mut destination = (state % 16) as u16;
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
                    bytes: 4 * (1 + (state % 64)),
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
    fn scheduler_packs_an_all_tile_matching_into_one_epoch() {
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
    fn scheduler_colors_ring_and_all_to_all_fanout() {
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
        assert_eq!(lowered[0].epochs.len(), 2);

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
        assert_eq!(lowered[0].epochs.len(), 8);
        assert!(
            lowered[0]
                .epochs
                .iter()
                .all(|epoch| epoch.groups.len() == 1)
        );
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
