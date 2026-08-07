//! Mid-level, layout-aware representation.
//!
//! This is the boundary between semantic graph operations and scheduling. It
//! records tensor shapes, storage precision, AMP memory order, interleaving,
//! and coarse tile sharding, but deliberately does not assign tile addresses
//! or emit exchange rows. [`lower`] tries a small set of legal kernel formats,
//! prices them with a [`CostModel`], and inserts explicit precision casts and
//! layout rearrangements at format boundaries.

use crate::graph::{
    ComputeGraph, GraphInputKind, Operation, OperationId, OperationKind, Repeat, TensorShape,
    ValueId,
};
use std::collections::BTreeMap;

/// In-memory representation of one tensor element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    F8F143 { scale: i8 },
    F16,
    F32,
}

impl Precision {
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F8F143 { .. } => 1,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sharding {
    Replicated,
    Rows,
    Columns,
    Heads,
}

/// AMP operand order with configurable inner and column block dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AmpOrder {
    Left { inner: u16 },
    Right { inner: u16, columns: u16 },
    Output { columns: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementOrder {
    RowMajor,
    Amp(AmpOrder),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Storage {
    Contiguous,
    /// Adjacent grains are distributed between the tile's memory banks.
    Interleaved {
        grain_bytes: u16,
    },
}

/// Layout decisions which constrain kernels and exchange generation.
///
/// `tile_count` describes a logical tile group, not final physical placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layout {
    pub order: ElementOrder,
    pub sharding: Sharding,
    pub tile_count: u16,
    pub storage: Storage,
    pub alignment: u16,
}

impl Layout {
    pub const fn row_major(sharding: Sharding, tile_count: u16) -> Self {
        Self {
            order: ElementOrder::RowMajor,
            sharding,
            tile_count,
            storage: Storage::Contiguous,
            alignment: 8,
        }
    }

    pub const fn amp_left(inner: u16, tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Left { inner }),
            sharding: Sharding::Rows,
            tile_count,
            storage: Storage::Contiguous,
            alignment: 8,
        }
    }

    pub const fn amp_right(inner: u16, tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Right { inner, columns: 16 }),
            sharding: Sharding::Columns,
            tile_count,
            storage: Storage::Interleaved { grain_bytes: 8 },
            alignment: 8,
        }
    }

    pub const fn amp_output(tile_count: u16) -> Self {
        Self {
            order: ElementOrder::Amp(AmpOrder::Output { columns: 16 }),
            sharding: Sharding::Rows,
            tile_count,
            storage: Storage::Contiguous,
            alignment: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorFormat {
    pub precision: Precision,
    pub layout: Layout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    pub shape: TensorShape,
    pub format: TensorFormat,
}

impl TensorType {
    pub fn new(shape: impl IntoIterator<Item = u32>, precision: Precision, layout: Layout) -> Self {
        Self {
            shape: TensorShape::new(shape),
            format: TensorFormat { precision, layout },
        }
    }
}

/// Information supplied for semantic graph inputs and parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweringConfig {
    pub tile_count: u16,
    pub inputs: BTreeMap<ValueId, TensorFormat>,
    /// Candidate GEMM storage precisions, in tie-breaking order.
    pub gemm_precisions: Vec<Precision>,
}

impl LoweringConfig {
    pub fn new(tile_count: u16) -> Self {
        Self {
            tile_count,
            inputs: BTreeMap::new(),
            gemm_precisions: vec![Precision::F16, Precision::F32],
        }
    }

    pub fn with_input(mut self, value: ValueId, format: TensorFormat) -> Self {
        self.inputs.insert(value, format);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MidValueId(u32);

impl MidValueId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidValue {
    pub id: MidValueId,
    pub tensor_type: TensorType,
    /// Semantic value represented by this value; conversions retain the same
    /// origin. Region arguments also refer to their high-level argument ID.
    pub origin: ValueId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidKernel {
    Gemm {
        multiply: Precision,
        accumulate: AccumulationPrecision,
    },
    Gelu,
    Add,
    FlashAttention {
        accumulate: AccumulationPrecision,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidOperationKind {
    Kernel(MidKernel),
    CastPrecision { from: Precision, to: Precision },
    Rearrange { from: Layout, to: Layout },
    Repeat(MidRepeat),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidOperation {
    pub source: Option<OperationId>,
    pub inputs: Vec<MidValueId>,
    pub results: Vec<MidValueId>,
    pub kind: MidOperationKind,
    pub estimated_cost: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRegion {
    pub arguments: Vec<MidValueId>,
    pub operations: Vec<MidOperation>,
    pub yields: Vec<MidValueId>,
    pub estimated_cost: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRepeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    /// One normalized value list for each iterated body argument. Keeping the
    /// lists on the structured operation avoids unrolling layer parameters.
    pub iterated_inputs: Vec<Vec<MidValueId>>,
    pub body: MidRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MidGraph {
    pub inputs: Vec<MidInput>,
    pub values: Vec<MidValue>,
    pub operations: Vec<MidOperation>,
    pub outputs: Vec<MidValueId>,
    pub estimated_cost: u64,
}

/// Deliberately small and replaceable cost interface. Units are arbitrary but
/// must be comparable within one lowering run.
pub trait CostModel {
    fn kernel_cost(&self, kernel: MidKernel, inputs: &[TensorType], output: &TensorType) -> u64;
    fn cast_cost(&self, shape: &TensorShape, from: Precision, to: Precision) -> u64;
    fn rearrange_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: Layout,
        to: Layout,
    ) -> u64;
}

/// A transparent placeholder model: arithmetic is priced by rough vector
/// throughput and conversions by bytes moved. It is intended to be replaced
/// by measurements without changing the IR.
#[derive(Clone, Copy, Debug, Default)]
pub struct ToyCostModel;

impl CostModel for ToyCostModel {
    fn kernel_cost(&self, kernel: MidKernel, inputs: &[TensorType], output: &TensorType) -> u64 {
        let elements = output.shape.elements();
        match kernel {
            MidKernel::Gemm { multiply, .. } => {
                let k = inputs[0].shape.0.last().copied().unwrap_or(1) as u64;
                let throughput = match multiply {
                    Precision::F8F143 { .. } => 128,
                    Precision::F16 => 64,
                    Precision::F32 => 16,
                };
                output
                    .shape
                    .elements()
                    .saturating_mul(2)
                    .saturating_mul(k)
                    .div_ceil(throughput)
            }
            MidKernel::FlashAttention { .. } => elements.saturating_mul(8).div_ceil(32),
            MidKernel::Gelu => elements.saturating_mul(6).div_ceil(16),
            MidKernel::Add => elements.div_ceil(16),
        }
    }

    fn cast_cost(&self, shape: &TensorShape, from: Precision, to: Precision) -> u64 {
        shape
            .elements()
            .saturating_mul(from.bytes() + to.bytes())
            .div_ceil(16)
    }

    fn rearrange_cost(
        &self,
        shape: &TensorShape,
        precision: Precision,
        from: Layout,
        to: Layout,
    ) -> u64 {
        let bytes = shape
            .elements()
            .saturating_mul(precision.bytes())
            .saturating_mul(2);
        let exchange_penalty =
            u64::from(from.sharding != to.sharding || from.tile_count != to.tile_count) + 1;
        bytes.saturating_mul(exchange_penalty).div_ceil(16)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoweringError {
    #[error("mid-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("no tensor type was supplied for graph input {0:?}")]
    MissingInputType(ValueId),
    #[error("graph has no stored shape for value {0:?}")]
    MissingShape(ValueId),
    #[error("operation {0:?} has no legal format candidate")]
    NoCandidate(OperationId),
    #[error("internal lowering error: value {0:?} is unavailable")]
    UnknownValue(ValueId),
}

pub type LoweringResult<T> = std::result::Result<T, LoweringError>;

pub fn lower(
    graph: &ComputeGraph,
    config: &LoweringConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidGraph> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let mut state = LoweringState::default();
    let mut values = BTreeMap::new();
    let mut inputs = Vec::with_capacity(graph.inputs().len());
    for input in graph.inputs() {
        let format = config
            .inputs
            .get(&input.value)
            .copied()
            .ok_or(LoweringError::MissingInputType(input.value))?;
        let tensor_type = TensorType {
            shape: input.shape.clone(),
            format,
        };
        let value = state.value(input.value, tensor_type);
        values.insert(input.value, value);
        inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value,
        });
    }
    let operations = lower_operations(
        graph.operations(),
        &mut values,
        graph.value_shapes(),
        graph,
        config,
        costs,
        &mut state,
    )?;
    let outputs = graph
        .outputs()
        .iter()
        .map(|value| lookup(&values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let estimated_cost = operations
        .iter()
        .map(|operation| operation.estimated_cost)
        .sum();
    Ok(MidGraph {
        inputs,
        values: state.values,
        operations,
        outputs,
        estimated_cost,
    })
}

#[derive(Default)]
struct LoweringState {
    values: Vec<MidValue>,
}

impl LoweringState {
    fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            tensor_type,
            origin,
        });
        id
    }

    fn get(&self, id: MidValueId) -> &MidValue {
        &self.values[id.0 as usize]
    }
}

fn lower_operations(
    source: &[Operation],
    values: &mut BTreeMap<ValueId, MidValueId>,
    shapes: &BTreeMap<ValueId, TensorShape>,
    graph: &ComputeGraph,
    config: &LoweringConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
) -> LoweringResult<Vec<MidOperation>> {
    let mut operations = Vec::new();
    for operation in source {
        if let OperationKind::Repeat(repeat) = &operation.kind {
            lower_repeat(
                operation,
                repeat,
                values,
                graph,
                config,
                costs,
                state,
                &mut operations,
            )?;
            continue;
        }
        let input_ids = operation
            .inputs
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?;
        let input_types = input_ids
            .iter()
            .map(|value| state.get(*value).tensor_type.clone())
            .collect::<Vec<_>>();
        let output_shape = shapes
            .get(&operation.results[0])
            .cloned()
            .ok_or(LoweringError::MissingShape(operation.results[0]))?;
        let plans = plans(operation, &input_types, &output_shape, config);
        let plan = plans
            .into_iter()
            .map(|plan| {
                let conversion = input_types
                    .iter()
                    .zip(&plan.inputs)
                    .map(|(from, to)| conversion_cost(from, *to, costs))
                    .sum::<u64>();
                let output = TensorType {
                    shape: output_shape.clone(),
                    format: plan.output,
                };
                let planned_inputs = input_types
                    .iter()
                    .zip(&plan.inputs)
                    .map(|(input, format)| TensorType {
                        shape: input.shape.clone(),
                        format: *format,
                    })
                    .collect::<Vec<_>>();
                let cost = conversion + costs.kernel_cost(plan.kernel, &planned_inputs, &output);
                (cost, plan)
            })
            .min_by_key(|(cost, _)| *cost)
            .ok_or(LoweringError::NoCandidate(operation.id))?
            .1;
        let converted = input_ids
            .into_iter()
            .zip(plan.inputs)
            .map(|(value, format)| {
                ensure_format(value, format, operation.id, costs, state, &mut operations)
            })
            .collect::<Vec<_>>();
        let result = state.value(
            operation.results[0],
            TensorType {
                shape: output_shape,
                format: plan.output,
            },
        );
        let converted_types = converted
            .iter()
            .map(|value| state.get(*value).tensor_type.clone())
            .collect::<Vec<_>>();
        let kernel_cost = costs.kernel_cost(
            plan.kernel,
            &converted_types,
            &state.get(result).tensor_type,
        );
        operations.push(MidOperation {
            source: Some(operation.id),
            inputs: converted,
            results: vec![result],
            kind: MidOperationKind::Kernel(plan.kernel),
            estimated_cost: kernel_cost,
        });
        values.insert(operation.results[0], result);
    }
    Ok(operations)
}

#[derive(Clone)]
struct Plan {
    kernel: MidKernel,
    inputs: Vec<TensorFormat>,
    output: TensorFormat,
}

fn plans(
    operation: &Operation,
    inputs: &[TensorType],
    _output: &TensorShape,
    config: &LoweringConfig,
) -> Vec<Plan> {
    match operation.kind {
        OperationKind::Gemm => config
            .gemm_precisions
            .iter()
            .filter_map(|&precision| {
                let inner = match precision {
                    Precision::F16 => 16,
                    Precision::F32 => 8,
                    Precision::F8F143 { .. } => return None,
                };
                Some(Plan {
                    kernel: MidKernel::Gemm {
                        multiply: precision,
                        accumulate: AccumulationPrecision::F32,
                    },
                    inputs: vec![
                        TensorFormat {
                            precision,
                            layout: Layout::amp_left(inner, config.tile_count),
                        },
                        TensorFormat {
                            precision,
                            layout: Layout::amp_right(inner, config.tile_count),
                        },
                    ],
                    output: TensorFormat {
                        precision,
                        layout: Layout::amp_output(config.tile_count),
                    },
                })
            })
            .collect(),
        OperationKind::Add => inputs
            .iter()
            .map(|input| input.format)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|format| Plan {
                kernel: MidKernel::Add,
                inputs: vec![format, format],
                output: format,
            })
            .collect(),
        OperationKind::Gelu => vec![Plan {
            kernel: MidKernel::Gelu,
            inputs: vec![inputs[0].format],
            output: inputs[0].format,
        }],
        OperationKind::FlashAttention => [Precision::F16, Precision::F32]
            .into_iter()
            .map(|precision| {
                let format = TensorFormat {
                    precision,
                    layout: Layout::row_major(Sharding::Heads, config.tile_count),
                };
                Plan {
                    kernel: MidKernel::FlashAttention {
                        accumulate: AccumulationPrecision::F32,
                    },
                    inputs: vec![format; 3],
                    output: format,
                }
            })
            .collect(),
        OperationKind::Repeat(_) => unreachable!("repeat is lowered separately"),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_repeat(
    operation: &Operation,
    repeat: &Repeat,
    values: &mut BTreeMap<ValueId, MidValueId>,
    graph: &ComputeGraph,
    config: &LoweringConfig,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| lookup(values, *value))
        .collect::<LoweringResult<Vec<_>>>()?;
    let mut argument_types = inputs
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let mut iterated_inputs = Vec::with_capacity(repeat.iterated_inputs.len());
    for sequence_id in &repeat.iterated_inputs {
        let sequence = &graph.sequences()[sequence_id.index() as usize];
        let first = lookup(values, sequence.values[0])?;
        let first_type = state.get(first).tensor_type.clone();
        let normalized = sequence
            .values
            .iter()
            .map(|value| lookup(values, *value))
            .collect::<LoweringResult<Vec<_>>>()?
            .into_iter()
            .map(|value| {
                ensure_format(
                    value,
                    first_type.format,
                    operation.id,
                    costs,
                    state,
                    operations,
                )
            })
            .collect();
        iterated_inputs.push(normalized);
        argument_types.push(first_type);
    }
    let mut body_values = BTreeMap::new();
    let mut arguments = Vec::new();
    for (&origin, tensor_type) in repeat.body.arguments.iter().zip(argument_types) {
        let value = state.value(origin, tensor_type);
        body_values.insert(origin, value);
        arguments.push(value);
    }
    let mut body_operations = lower_operations(
        &repeat.body.operations,
        &mut body_values,
        &repeat.body.value_shapes,
        graph,
        config,
        costs,
        state,
    )?;
    let mut yields = Vec::new();
    for (index, high_yield) in repeat.body.yields.iter().enumerate() {
        let value = lookup(&body_values, *high_yield)?;
        let target = state.get(inputs[index]).tensor_type.format;
        yields.push(ensure_format(
            value,
            target,
            operation.id,
            costs,
            state,
            &mut body_operations,
        ));
    }
    let body_cost = body_operations
        .iter()
        .map(|operation| operation.estimated_cost)
        .sum();
    let mut results = Vec::new();
    for (origin, input) in operation.results.iter().zip(&inputs) {
        let result = state.value(*origin, state.get(*input).tensor_type.clone());
        values.insert(*origin, result);
        results.push(result);
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs,
        results,
        kind: MidOperationKind::Repeat(MidRepeat {
            count: repeat.count,
            carried_inputs: repeat.carried_inputs,
            invariant_inputs: repeat.invariant_inputs,
            iterated_inputs,
            body: MidRegion {
                arguments,
                operations: body_operations,
                yields,
                estimated_cost: body_cost,
            },
        }),
        estimated_cost: body_cost.saturating_mul(u64::from(repeat.count)),
    });
    Ok(())
}

fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    source: OperationId,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    let original = state.get(value).clone();
    if original.tensor_type.format.precision != target.precision {
        let mut tensor_type = original.tensor_type.clone();
        let from = tensor_type.format.precision;
        tensor_type.format.precision = target.precision;
        let result = state.value(original.origin, tensor_type.clone());
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::CastPrecision {
                from,
                to: target.precision,
            },
            estimated_cost: costs.cast_cost(&tensor_type.shape, from, target.precision),
        });
        value = result;
    }
    let current = state.get(value).clone();
    if current.tensor_type.format.layout != target.layout {
        let mut tensor_type = current.tensor_type.clone();
        let from = tensor_type.format.layout;
        tensor_type.format.layout = target.layout;
        let result = state.value(current.origin, tensor_type.clone());
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::Rearrange {
                from,
                to: target.layout,
            },
            estimated_cost: costs.rearrange_cost(
                &tensor_type.shape,
                tensor_type.format.precision,
                from,
                target.layout,
            ),
        });
        value = result;
    }
    value
}

fn conversion_cost(from: &TensorType, to: TensorFormat, costs: &impl CostModel) -> u64 {
    let cast = if from.format.precision != to.precision {
        costs.cast_cost(&from.shape, from.format.precision, to.precision)
    } else {
        0
    };
    let rearrange = if from.format.layout != to.layout {
        costs.rearrange_cost(&from.shape, to.precision, from.format.layout, to.layout)
    } else {
        0
    };
    cast.saturating_add(rearrange)
}

fn lookup(values: &BTreeMap<ValueId, MidValueId>, value: ValueId) -> LoweringResult<MidValueId> {
    values
        .get(&value)
        .copied()
        .ok_or(LoweringError::UnknownValue(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANDOM_CASES: usize = 128;

    fn dimension(random: &mut fastrand::Rng) -> u32 {
        random.u32(1..=128)
    }

    fn precision(random: &mut fastrand::Rng) -> Precision {
        if random.bool() {
            Precision::F16
        } else {
            Precision::F32
        }
    }

    fn format(precision: Precision, layout: Layout) -> TensorFormat {
        TensorFormat { precision, layout }
    }

    fn value(lowered: &MidGraph, id: MidValueId) -> &MidValue {
        &lowered.values[id.index() as usize]
    }

    fn assert_conversions_are_explicit(lowered: &MidGraph, operations: &[MidOperation]) {
        for operation in operations {
            let [input] = operation.inputs.as_slice() else {
                continue;
            };
            let [result] = operation.results.as_slice() else {
                continue;
            };
            let before = &value(lowered, *input).tensor_type;
            let after = &value(lowered, *result).tensor_type;
            match operation.kind {
                MidOperationKind::CastPrecision { from, to } => {
                    assert_eq!(from, before.format.precision);
                    assert_eq!(to, after.format.precision);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.layout, after.format.layout);
                }
                MidOperationKind::Rearrange { from, to } => {
                    assert_eq!(from, before.format.layout);
                    assert_eq!(to, after.format.layout);
                    assert_eq!(before.shape, after.shape);
                    assert_eq!(before.format.precision, after.format.precision);
                }
                MidOperationKind::Kernel(_) | MidOperationKind::Repeat(_) => {}
            }
        }
    }

    #[test]
    fn randomized_gemm_lowering_makes_every_format_boundary_explicit() {
        let mut random = fastrand::Rng::with_seed(0x6d69_6467);
        for case in 0..RANDOM_CASES {
            let (rows, inner, columns) = (
                dimension(&mut random),
                dimension(&mut random),
                dimension(&mut random),
            );
            let batches = (0..random.usize(0..=3))
                .map(|_| dimension(&mut random))
                .collect::<Vec<_>>();
            let tiles = random.u16(1..=64);
            let kernel_precision = precision(&mut random);
            let mut left_shape = batches.clone();
            left_shape.extend([rows, inner]);
            let mut right_shape = batches;
            right_shape.extend([inner, columns]);

            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", left_shape).unwrap();
            let right = graph.parameter("right", right_shape).unwrap();
            let product = graph.gemm(left, right).unwrap();
            graph.set_outputs([product]).unwrap();
            let linear = Layout::row_major(Sharding::Rows, tiles);
            let mut config = LoweringConfig::new(tiles)
                .with_input(left, format(precision(&mut random), linear))
                .with_input(right, format(precision(&mut random), linear));
            config.gemm_precisions = vec![kernel_precision];

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let kernel = lowered
                .operations
                .iter()
                .find(|operation| matches!(operation.kind, MidOperationKind::Kernel(_)))
                .unwrap();
            let MidOperationKind::Kernel(MidKernel::Gemm { multiply, .. }) = kernel.kind else {
                panic!("random case {case}: expected GEMM");
            };
            assert_eq!(multiply, kernel_precision, "random case {case}");
            let inner_block = if kernel_precision == Precision::F16 {
                16
            } else {
                8
            };
            assert_eq!(
                value(&lowered, kernel.inputs[0]).tensor_type.format,
                format(kernel_precision, Layout::amp_left(inner_block, tiles)),
                "random case {case}"
            );
            assert_eq!(
                value(&lowered, kernel.inputs[1]).tensor_type.format,
                format(kernel_precision, Layout::amp_right(inner_block, tiles)),
                "random case {case}"
            );
            let output = value(&lowered, lowered.outputs[0]);
            let expected_shape = graph.value_shape(product).unwrap().clone();
            assert_eq!(
                output.tensor_type.shape, expected_shape,
                "random case {case}"
            );
            assert_eq!(
                output.tensor_type.format,
                format(kernel_precision, Layout::amp_output(tiles)),
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_add_lowering_unifies_operand_formats() {
        let mut random = fastrand::Rng::with_seed(0x6164_642b);
        for case in 0..RANDOM_CASES {
            let shape = (0..random.usize(1..=5))
                .map(|_| dimension(&mut random))
                .collect::<Vec<_>>();
            let tiles = random.u16(1..=64);
            let layout = Layout::row_major(Sharding::Rows, tiles);
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", shape.clone()).unwrap();
            let right = graph.host_input("right", shape).unwrap();
            let sum = graph.add(left, right).unwrap();
            graph.set_outputs([sum]).unwrap();
            let config = LoweringConfig::new(tiles)
                .with_input(left, format(Precision::F16, layout))
                .with_input(right, format(Precision::F32, layout));

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let kernel = lowered.operations.last().unwrap();
            assert!(matches!(
                kernel.kind,
                MidOperationKind::Kernel(MidKernel::Add)
            ));
            let left_format = value(&lowered, kernel.inputs[0]).tensor_type.format;
            let right_format = value(&lowered, kernel.inputs[1]).tensor_type.format;
            assert_eq!(left_format, right_format, "random case {case}");
            assert_eq!(
                value(&lowered, kernel.results[0]).tensor_type.format,
                left_format,
                "random case {case}"
            );
            assert!(
                lowered.operations.iter().any(|operation| matches!(
                    operation.kind,
                    MidOperationKind::CastPrecision { .. }
                ))
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
        }
    }

    #[test]
    fn randomized_repeat_lowering_retains_sequences_without_unrolling() {
        let mut random = fastrand::Rng::with_seed(0x7265_7065);
        for case in 0..RANDOM_CASES {
            let size = dimension(&mut random);
            let count = random.u32(1..=12);
            let tiles = random.u16(1..=64);
            let layout = Layout::row_major(Sharding::Rows, tiles);
            let carried_format = format(precision(&mut random), layout);
            let mut graph = ComputeGraph::new();
            let carried = graph.host_input("state", [size, size]).unwrap();
            let weights = (0..count)
                .map(|index| graph.parameter(format!("weight.{index}"), [size, size]))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
            let output = graph
                .repeat(count, [carried], [], [sequence], |body, arguments| {
                    Ok(vec![
                        body.gemm(arguments.carried[0], arguments.iterated[0])?,
                    ])
                })
                .unwrap()[0];
            graph.set_outputs([output]).unwrap();
            let mut config = LoweringConfig::new(tiles).with_input(carried, carried_format);
            for weight in weights {
                config
                    .inputs
                    .insert(weight, format(precision(&mut random), layout));
            }

            let lowered = lower(&graph, &config, &ToyCostModel).unwrap();
            let repeat = lowered
                .operations
                .iter()
                .find_map(|operation| match &operation.kind {
                    MidOperationKind::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .unwrap();
            assert_eq!(repeat.count, count, "random case {case}");
            assert_eq!(repeat.iterated_inputs.len(), 1, "random case {case}");
            assert_eq!(
                repeat.iterated_inputs[0].len(),
                count as usize,
                "random case {case}"
            );
            let sequence_format = value(&lowered, repeat.iterated_inputs[0][0])
                .tensor_type
                .format;
            assert!(
                repeat.iterated_inputs[0]
                    .iter()
                    .all(
                        |value_id| value(&lowered, *value_id).tensor_type.format == sequence_format
                    )
            );
            assert_eq!(
                value(&lowered, repeat.body.yields[0]).tensor_type.format,
                carried_format,
                "random case {case}"
            );
            assert_eq!(
                value(&lowered, lowered.outputs[0]).tensor_type.format,
                carried_format,
                "random case {case}"
            );
            assert_conversions_are_explicit(&lowered, &lowered.operations);
            assert_conversions_are_explicit(&lowered, &repeat.body.operations);
        }
    }
}
