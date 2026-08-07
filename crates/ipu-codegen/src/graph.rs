//! Structured SSA-like input graph for package construction.
//!
//! A graph and each nested region contain an ordered list of operations, but
//! dataflow is expressed through globally unique [`ValueId`]s. This preserves
//! residual and branching connections without making the dataflow a tree.
//! Region-internal values cannot escape except through explicit yields.
//!
//! [`ComputeGraph::repeat`] separates loop-carried state, loop invariants, and
//! iterated value sequences. The latter allow one shared body to use different
//! layer parameters on each iteration without unrolling it.
//!
//! ```
//! use ipu_codegen::{ComputeGraph, GraphError};
//!
//! # fn build() -> Result<(), GraphError> {
//! let mut graph = ComputeGraph::new();
//! let input = graph.host_input("input", [1, 1024])?;
//! let layer_weights = (0..12)
//!     .map(|layer| graph.parameter(format!("layer.{layer}.weight"), [1024, 1024]))
//!     .collect::<Result<Vec<_>, _>>()?;
//! let weights = graph.value_sequence("layer weights", layer_weights)?;
//! let output = graph
//!     .repeat(12, [input], [], [weights], |body, arguments| {
//!         let update = body.gemm(arguments.carried[0], arguments.iterated[0])?;
//!         let residual = body.add(arguments.carried[0], update)?;
//!         Ok(vec![residual])
//!     })?
//!     .remove(0);
//! graph.set_outputs([output])?;
//! # Ok(())
//! # }
//! # build().unwrap();
//! ```

use std::collections::{BTreeMap, BTreeSet};

/// Logical tensor dimensions. Shapes are semantic graph information; storage
/// precision and physical layout are selected during mid-level lowering.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorShape(pub Vec<u32>);

impl TensorShape {
    pub fn new(dimensions: impl IntoIterator<Item = u32>) -> Self {
        Self(dimensions.into_iter().collect())
    }

    pub fn elements(&self) -> u64 {
        self.0.iter().copied().map(u64::from).product()
    }
}

/// Stable identity of an operation, used for diagnostics and transformations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationId(u32);

impl OperationId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Stable identity of one tensor result.
///
/// Consumers refer to values rather than operations because an operation may
/// have zero, one, or several results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(u32);

impl ValueId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Stable identity of a collection supplying one value per repeat iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueSequenceId(u32);

impl ValueSequenceId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphInputKind {
    Host,
    Parameter,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: ValueId,
    pub shape: TensorShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueSequence {
    pub id: ValueSequenceId,
    pub name: String,
    pub values: Vec<ValueId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    pub id: OperationId,
    pub inputs: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub kind: OperationKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationKind {
    Gemm(GemmOptions),
    /// Exact Gaussian error linear unit.
    Gelu,
    Add(AddOptions),
    FlashAttention(AttentionOptions),
    Repeat(Repeat),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GemmOptions {
    pub transpose_left: bool,
    pub transpose_right: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BroadcastMode {
    #[default]
    Numpy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddOptions {
    pub broadcasting: BroadcastMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionScale {
    #[default]
    InverseSqrtQueryWidth,
    ValueBits(u32),
}

impl AttentionScale {
    pub fn value(scale: f32) -> Self {
        Self::ValueBits(scale.to_bits())
    }

    pub fn as_value(self) -> Option<f32> {
        match self {
            Self::InverseSqrtQueryWidth => None,
            Self::ValueBits(bits) => Some(f32::from_bits(bits)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionOptions {
    pub causal: bool,
    pub scale: AttentionScale,
}

/// Structured repetition whose body is an ordered SSA operation list.
///
/// Body arguments are ordered as carried values, invariants, then iterated
/// values. Only `yields` escape the region. Each yield corresponds to one
/// carried input and becomes one result of the repeat operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    pub iterated_inputs: Vec<ValueSequenceId>,
    pub body: Region,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub arguments: Vec<ValueId>,
    pub operations: Vec<Operation>,
    pub yields: Vec<ValueId>,
    /// Shapes for arguments and every value defined inside this region.
    pub value_shapes: BTreeMap<ValueId, TensorShape>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatArguments {
    pub carried: Vec<ValueId>,
    pub invariants: Vec<ValueId>,
    pub iterated: Vec<ValueId>,
}

/// High-level graph accepted by the package pipeline.
///
/// Operations are stored in construction order while dependencies are
/// represented explicitly with `ValueId`. Nested regions provide structured
/// control flow without turning residual and branching dataflow into a tree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeGraph {
    inputs: Vec<GraphInput>,
    sequences: Vec<ValueSequence>,
    operations: Vec<Operation>,
    outputs: Vec<ValueId>,
    values: BTreeSet<ValueId>,
    shapes: BTreeMap<ValueId, TensorShape>,
    next_operation: u32,
    next_value: u32,
    next_sequence: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("graph name must not be empty")]
    EmptyName,
    #[error("value {0:?} is not available in this operation list")]
    UnknownValue(ValueId),
    #[error("value sequence {0:?} does not exist")]
    UnknownSequence(ValueSequenceId),
    #[error("repeat count must be nonzero")]
    EmptyRepeat,
    #[error("value sequence {sequence:?} has {available} entries for {required} iterations")]
    ShortSequence {
        sequence: ValueSequenceId,
        available: usize,
        required: u32,
    },
    #[error("repeat body yielded {actual} values for {expected} carried values")]
    YieldArity { expected: usize, actual: usize },
    #[error("operation has invalid shapes: {0}")]
    InvalidShape(String),
    #[error("value sequence entry {entry} has a different shape from its first entry")]
    SequenceShape { entry: usize },
    #[error("repeat yield {index} does not have the shape of carried value {index}")]
    YieldShape { index: usize },
}

pub type GraphResult<T> = std::result::Result<T, GraphError>;

impl ComputeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inputs(&self) -> &[GraphInput] {
        &self.inputs
    }

    pub fn sequences(&self) -> &[ValueSequence] {
        &self.sequences
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn value_shape(&self, value: ValueId) -> Option<&TensorShape> {
        self.shapes.get(&value)
    }

    pub fn value_shapes(&self) -> &BTreeMap<ValueId, TensorShape> {
        &self.shapes
    }

    pub fn host_input(
        &mut self,
        name: impl Into<String>,
        shape: impl IntoIterator<Item = u32>,
    ) -> GraphResult<ValueId> {
        self.input(name, GraphInputKind::Host, TensorShape::new(shape))
    }

    pub fn parameter(
        &mut self,
        name: impl Into<String>,
        shape: impl IntoIterator<Item = u32>,
    ) -> GraphResult<ValueId> {
        self.input(name, GraphInputKind::Parameter, TensorShape::new(shape))
    }

    pub fn value_sequence(
        &mut self,
        name: impl Into<String>,
        values: impl IntoIterator<Item = ValueId>,
    ) -> GraphResult<ValueSequenceId> {
        let name = nonempty(name)?;
        let values = values.into_iter().collect::<Vec<_>>();
        validate_inputs(&self.values, &values)?;
        if let Some(first) = values.first().and_then(|value| self.shapes.get(value))
            && let Some(entry) = values
                .iter()
                .position(|value| self.shapes.get(value) != Some(first))
        {
            return Err(GraphError::SequenceShape { entry });
        }
        let id = ValueSequenceId(self.next_sequence);
        self.next_sequence += 1;
        self.sequences.push(ValueSequence { id, name, values });
        Ok(id)
    }

    pub fn gemm(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
        self.gemm_with_options(left, right, GemmOptions::default())
    }

    pub fn gemm_with_options(
        &mut self,
        left: ValueId,
        right: ValueId,
        options: GemmOptions,
    ) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Gemm(options), [left, right])
    }

    pub fn gelu(&mut self, input: ValueId) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Gelu, [input])
    }

    pub fn add(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Add(AddOptions::default()), [left, right])
    }

    pub fn flash_attention(
        &mut self,
        query: ValueId,
        key: ValueId,
        value: ValueId,
    ) -> GraphResult<ValueId> {
        self.flash_attention_with_options(query, key, value, AttentionOptions::default())
    }

    pub fn flash_attention_with_options(
        &mut self,
        query: ValueId,
        key: ValueId,
        value: ValueId,
        options: AttentionOptions,
    ) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::FlashAttention(options), [query, key, value])
    }

    pub fn operation(
        &mut self,
        kind: OperationKind,
        inputs: impl IntoIterator<Item = ValueId>,
        result_shapes: impl IntoIterator<Item = TensorShape>,
    ) -> GraphResult<Vec<ValueId>> {
        append_operation(
            &mut self.operations,
            &mut self.values,
            &mut self.shapes,
            &mut self.next_operation,
            &mut self.next_value,
            kind,
            inputs,
            result_shapes,
        )
    }

    pub fn repeat<F>(
        &mut self,
        count: u32,
        carried: impl IntoIterator<Item = ValueId>,
        invariants: impl IntoIterator<Item = ValueId>,
        iterated: impl IntoIterator<Item = ValueSequenceId>,
        build: F,
    ) -> GraphResult<Vec<ValueId>>
    where
        F: FnOnce(&mut RegionBuilder<'_>, &RepeatArguments) -> GraphResult<Vec<ValueId>>,
    {
        if count == 0 {
            return Err(GraphError::EmptyRepeat);
        }
        let carried = carried.into_iter().collect::<Vec<_>>();
        let invariants = invariants.into_iter().collect::<Vec<_>>();
        let iterated = iterated.into_iter().collect::<Vec<_>>();
        validate_inputs(&self.values, &carried)?;
        validate_inputs(&self.values, &invariants)?;
        for &sequence in &iterated {
            let values = self
                .sequences
                .iter()
                .find(|candidate| candidate.id == sequence)
                .ok_or(GraphError::UnknownSequence(sequence))?;
            if values.values.len() < count as usize {
                return Err(GraphError::ShortSequence {
                    sequence,
                    available: values.values.len(),
                    required: count,
                });
            }
        }

        let argument_count = carried.len() + invariants.len() + iterated.len();
        let arguments = allocate_values(&mut self.next_value, argument_count);
        let mut argument_shapes = carried
            .iter()
            .chain(&invariants)
            .map(|value| self.shapes[value].clone())
            .collect::<Vec<_>>();
        for sequence in &iterated {
            let sequence = &self.sequences[sequence.index() as usize];
            argument_shapes.push(self.shapes[&sequence.values[0]].clone());
        }
        let split_invariant = carried.len();
        let split_iterated = split_invariant + invariants.len();
        let repeat_arguments = RepeatArguments {
            carried: arguments[..split_invariant].to_vec(),
            invariants: arguments[split_invariant..split_iterated].to_vec(),
            iterated: arguments[split_iterated..].to_vec(),
        };
        let mut body = RegionBuilder::new(
            arguments,
            argument_shapes,
            &mut self.next_operation,
            &mut self.next_value,
        );
        let yields = build(&mut body, &repeat_arguments)?;
        if yields.len() != carried.len() {
            return Err(GraphError::YieldArity {
                expected: carried.len(),
                actual: yields.len(),
            });
        }
        for (index, (yielded, carried)) in yields.iter().zip(&carried).enumerate() {
            if body.shapes.get(yielded) != self.shapes.get(carried) {
                return Err(GraphError::YieldShape { index });
            }
        }
        let body = body.finish(yields)?;
        let result_shapes = carried
            .iter()
            .map(|value| self.shapes[value].clone())
            .collect::<Vec<_>>();
        let mut inputs = carried;
        let carried_inputs = inputs.len();
        let invariant_inputs = invariants.len();
        inputs.extend(invariants);
        self.operation(
            OperationKind::Repeat(Repeat {
                count,
                carried_inputs,
                invariant_inputs,
                iterated_inputs: iterated,
                body,
            }),
            inputs,
            result_shapes,
        )
    }

    pub fn set_outputs(&mut self, outputs: impl IntoIterator<Item = ValueId>) -> GraphResult<()> {
        let outputs = outputs.into_iter().collect::<Vec<_>>();
        validate_inputs(&self.values, &outputs)?;
        self.outputs = outputs;
        Ok(())
    }

    fn input(
        &mut self,
        name: impl Into<String>,
        kind: GraphInputKind,
        shape: TensorShape,
    ) -> GraphResult<ValueId> {
        let name = nonempty(name)?;
        let value = ValueId(self.next_value);
        self.next_value += 1;
        self.values.insert(value);
        self.shapes.insert(value, shape.clone());
        self.inputs.push(GraphInput {
            name,
            kind,
            value,
            shape,
        });
        Ok(value)
    }

    fn inferred_result(
        &mut self,
        kind: OperationKind,
        inputs: impl IntoIterator<Item = ValueId>,
    ) -> GraphResult<ValueId> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        validate_inputs(&self.values, &inputs)?;
        let shape = infer_shape(&kind, &inputs, &self.shapes)?;
        Ok(self
            .operation(kind, inputs, [shape])?
            .into_iter()
            .next()
            .expect("one result was requested"))
    }
}

pub struct RegionBuilder<'a> {
    arguments: Vec<ValueId>,
    operations: Vec<Operation>,
    values: BTreeSet<ValueId>,
    shapes: BTreeMap<ValueId, TensorShape>,
    next_operation: &'a mut u32,
    next_value: &'a mut u32,
}

impl<'a> RegionBuilder<'a> {
    fn new(
        arguments: Vec<ValueId>,
        argument_shapes: Vec<TensorShape>,
        next_operation: &'a mut u32,
        next_value: &'a mut u32,
    ) -> Self {
        Self {
            values: arguments.iter().copied().collect(),
            shapes: arguments.iter().copied().zip(argument_shapes).collect(),
            arguments,
            operations: Vec::new(),
            next_operation,
            next_value,
        }
    }

    pub fn gemm(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
        self.gemm_with_options(left, right, GemmOptions::default())
    }

    pub fn gemm_with_options(
        &mut self,
        left: ValueId,
        right: ValueId,
        options: GemmOptions,
    ) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Gemm(options), [left, right])
    }

    pub fn gelu(&mut self, input: ValueId) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Gelu, [input])
    }

    pub fn add(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::Add(AddOptions::default()), [left, right])
    }

    pub fn flash_attention(
        &mut self,
        query: ValueId,
        key: ValueId,
        value: ValueId,
    ) -> GraphResult<ValueId> {
        self.flash_attention_with_options(query, key, value, AttentionOptions::default())
    }

    pub fn flash_attention_with_options(
        &mut self,
        query: ValueId,
        key: ValueId,
        value: ValueId,
        options: AttentionOptions,
    ) -> GraphResult<ValueId> {
        self.inferred_result(OperationKind::FlashAttention(options), [query, key, value])
    }

    pub fn operation(
        &mut self,
        kind: OperationKind,
        inputs: impl IntoIterator<Item = ValueId>,
        result_shapes: impl IntoIterator<Item = TensorShape>,
    ) -> GraphResult<Vec<ValueId>> {
        append_operation(
            &mut self.operations,
            &mut self.values,
            &mut self.shapes,
            self.next_operation,
            self.next_value,
            kind,
            inputs,
            result_shapes,
        )
    }

    fn inferred_result(
        &mut self,
        kind: OperationKind,
        inputs: impl IntoIterator<Item = ValueId>,
    ) -> GraphResult<ValueId> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        validate_inputs(&self.values, &inputs)?;
        let shape = infer_shape(&kind, &inputs, &self.shapes)?;
        Ok(self
            .operation(kind, inputs, [shape])?
            .into_iter()
            .next()
            .expect("one result was requested"))
    }

    fn finish(self, yields: Vec<ValueId>) -> GraphResult<Region> {
        validate_inputs(&self.values, &yields)?;
        Ok(Region {
            arguments: self.arguments,
            operations: self.operations,
            yields,
            value_shapes: self.shapes,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn append_operation(
    operations: &mut Vec<Operation>,
    values: &mut BTreeSet<ValueId>,
    shapes: &mut BTreeMap<ValueId, TensorShape>,
    next_operation: &mut u32,
    next_value: &mut u32,
    kind: OperationKind,
    inputs: impl IntoIterator<Item = ValueId>,
    result_shapes: impl IntoIterator<Item = TensorShape>,
) -> GraphResult<Vec<ValueId>> {
    let inputs = inputs.into_iter().collect::<Vec<_>>();
    validate_inputs(values, &inputs)?;
    let id = OperationId(*next_operation);
    *next_operation += 1;
    let result_shapes = result_shapes.into_iter().collect::<Vec<_>>();
    let results = allocate_values(next_value, result_shapes.len());
    values.extend(results.iter().copied());
    shapes.extend(results.iter().copied().zip(result_shapes));
    operations.push(Operation {
        id,
        inputs,
        results: results.clone(),
        kind,
    });
    Ok(results)
}

fn infer_shape(
    kind: &OperationKind,
    inputs: &[ValueId],
    shapes: &BTreeMap<ValueId, TensorShape>,
) -> GraphResult<TensorShape> {
    let input = |index: usize| {
        inputs
            .get(index)
            .and_then(|value| shapes.get(value))
            .ok_or_else(|| GraphError::InvalidShape("wrong input arity".into()))
    };
    match kind {
        OperationKind::Gemm(options) => {
            let (left, right) = (input(0)?, input(1)?);
            if left.0.len() < 2 || right.0.len() < 2 {
                return Err(GraphError::InvalidShape(
                    "GEMM inputs must have rank at least two".into(),
                ));
            }
            let left_rows = left.0[left.0.len() - 2 + usize::from(options.transpose_left)];
            let left_inner = left.0[left.0.len() - 1 - usize::from(options.transpose_left)];
            let right_inner = right.0[right.0.len() - 2 + usize::from(options.transpose_right)];
            let right_columns = right.0[right.0.len() - 1 - usize::from(options.transpose_right)];
            if left_inner != right_inner {
                return Err(GraphError::InvalidShape(
                    "GEMM inner dimensions do not match".into(),
                ));
            }
            let mut output = broadcast(&left.0[..left.0.len() - 2], &right.0[..right.0.len() - 2])?;
            output.push(left_rows);
            output.push(right_columns);
            Ok(TensorShape(output))
        }
        OperationKind::Gelu => Ok(input(0)?.clone()),
        OperationKind::Add(AddOptions {
            broadcasting: BroadcastMode::Numpy,
        }) => Ok(TensorShape(broadcast(&input(0)?.0, &input(1)?.0)?)),
        OperationKind::FlashAttention(options) => {
            let (query, key, value) = (input(0)?, input(1)?, input(2)?);
            if query.0.len() < 2 || key.0.len() < 2 || value.0.len() < 2 {
                return Err(GraphError::InvalidShape(
                    "FlashAttention inputs must have rank at least two".into(),
                ));
            }
            if query.0[query.0.len() - 1] != key.0[key.0.len() - 1]
                || key.0[key.0.len() - 2] != value.0[value.0.len() - 2]
            {
                return Err(GraphError::InvalidShape(
                    "FlashAttention head or sequence dimensions do not match".into(),
                ));
            }
            if let Some(scale) = options.scale.as_value()
                && (!scale.is_finite() || scale <= 0.0)
            {
                return Err(GraphError::InvalidShape(
                    "attention scale must be finite and positive".into(),
                ));
            }
            let q_batch = &query.0[..query.0.len() - 2];
            let kv_batch = broadcast(&key.0[..key.0.len() - 2], &value.0[..value.0.len() - 2])?;
            let mut output = broadcast(q_batch, &kv_batch)?;
            output.push(query.0[query.0.len() - 2]);
            output.push(value.0[value.0.len() - 1]);
            Ok(TensorShape(output))
        }
        OperationKind::Repeat(_) => Err(GraphError::InvalidShape(
            "repeat result shapes are defined by carried values".into(),
        )),
    }
}

fn broadcast(left: &[u32], right: &[u32]) -> GraphResult<Vec<u32>> {
    let rank = left.len().max(right.len());
    let mut output = Vec::with_capacity(rank);
    for index in 0..rank {
        let left = if index + left.len() >= rank {
            left[index + left.len() - rank]
        } else {
            1
        };
        let right = if index + right.len() >= rank {
            right[index + right.len() - rank]
        } else {
            1
        };
        if left != right && left != 1 && right != 1 {
            return Err(GraphError::InvalidShape(format!(
                "dimensions {left} and {right} cannot be broadcast"
            )));
        }
        output.push(left.max(right));
    }
    Ok(output)
}

fn allocate_values(next_value: &mut u32, count: usize) -> Vec<ValueId> {
    (0..count)
        .map(|_| {
            let value = ValueId(*next_value);
            *next_value += 1;
            value
        })
        .collect()
}

fn validate_inputs(values: &BTreeSet<ValueId>, inputs: &[ValueId]) -> GraphResult<()> {
    if let Some(value) = inputs.iter().find(|value| !values.contains(value)) {
        Err(GraphError::UnknownValue(*value))
    } else {
        Ok(())
    }
}

fn nonempty(name: impl Into<String>) -> GraphResult<String> {
    let name = name.into();
    if name.is_empty() {
        Err(GraphError::EmptyName)
    } else {
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANDOM_CASES: usize = 256;

    fn dimension(random: &mut fastrand::Rng) -> u32 {
        random.u32(1..=128)
    }

    fn broadcast_batch(random: &mut fastrand::Rng) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let rank = random.usize(0..=4);
        let mut left = Vec::with_capacity(rank);
        let mut right = Vec::with_capacity(rank);
        let mut output = Vec::with_capacity(rank);
        for _ in 0..rank {
            let extent = dimension(random);
            let (left_extent, right_extent) = match random.u8(0..3) {
                0 => (extent, extent),
                1 => (1, extent),
                _ => (extent, 1),
            };
            left.push(left_extent);
            right.push(right_extent);
            output.push(extent);
        }
        (left, right, output)
    }

    #[test]
    fn randomized_batched_gemm_preserves_matrix_axes_and_broadcasts_batch() {
        let mut random = fastrand::Rng::with_seed(0x6a65_6d6d);
        for case in 0..RANDOM_CASES {
            let (mut left_shape, mut right_shape, mut expected) = broadcast_batch(&mut random);
            let (rows, inner, columns) = (
                dimension(&mut random),
                dimension(&mut random),
                dimension(&mut random),
            );
            let options = GemmOptions {
                transpose_left: random.bool(),
                transpose_right: random.bool(),
            };
            left_shape.extend(if options.transpose_left {
                [inner, rows]
            } else {
                [rows, inner]
            });
            right_shape.extend(if options.transpose_right {
                [columns, inner]
            } else {
                [inner, columns]
            });
            expected.extend([rows, columns]);

            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", left_shape).unwrap();
            let right = graph.parameter("right", right_shape).unwrap();
            let output = graph.gemm_with_options(left, right, options).unwrap();

            assert_eq!(
                graph.value_shape(output),
                Some(&TensorShape(expected)),
                "random case {case}"
            );
        }
    }

    #[test]
    fn randomized_attention_preserves_query_and_value_axes() {
        let mut random = fastrand::Rng::with_seed(0x6174_746e);
        for case in 0..RANDOM_CASES {
            let (query_batch, key_batch, mut expected) = broadcast_batch(&mut random);
            let query_rows = dimension(&mut random);
            let key_rows = dimension(&mut random);
            let key_columns = dimension(&mut random);
            let value_columns = dimension(&mut random);
            let mut query_shape = query_batch;
            query_shape.extend([query_rows, key_columns]);
            let mut key_shape = key_batch.clone();
            key_shape.extend([key_rows, key_columns]);
            let mut value_shape = key_batch;
            value_shape.extend([key_rows, value_columns]);
            expected.extend([query_rows, value_columns]);

            let mut graph = ComputeGraph::new();
            let query = graph.host_input("q", query_shape).unwrap();
            let key = graph.host_input("k", key_shape).unwrap();
            let value = graph.host_input("v", value_shape).unwrap();
            let options = AttentionOptions {
                causal: random.bool(),
                scale: if random.bool() {
                    AttentionScale::InverseSqrtQueryWidth
                } else {
                    AttentionScale::value(random.f32() + 0.01)
                },
            };
            let output = graph
                .flash_attention_with_options(query, key, value, options)
                .unwrap();

            assert_eq!(
                graph.value_shape(output),
                Some(&TensorShape(expected)),
                "random case {case}"
            );
        }
    }

    #[test]
    fn randomized_attention_rejects_nonpositive_or_nonfinite_scales() {
        let mut random = fastrand::Rng::with_seed(0x7363_616c);
        for case in 0..RANDOM_CASES {
            let scale = match random.u8(0..4) {
                0 => 0.0,
                1 => -random.f32(),
                2 => f32::INFINITY,
                _ => f32::NAN,
            };
            let mut graph = ComputeGraph::new();
            let query = graph.host_input("q", [4, 8]).unwrap();
            let key = graph.host_input("k", [4, 8]).unwrap();
            let value = graph.host_input("v", [4, 8]).unwrap();
            let result = graph.flash_attention_with_options(
                query,
                key,
                value,
                AttentionOptions {
                    causal: random.bool(),
                    scale: AttentionScale::value(scale),
                },
            );
            assert!(result.is_err(), "random case {case}, scale {scale}");
        }
    }
}
