//! Allocation requirements and region liveness; no cycle pricing.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryUsage {
    pub standard: u64,
    pub interleaved: u64,
}

impl MemoryUsage {
    pub const fn total(self) -> u64 {
        self.standard.saturating_add(self.interleaved)
    }

    pub(crate) fn add_class(&mut self, class: MemoryClass, bytes: u64) {
        let target = match class {
            MemoryClass::Ipu21Standard => &mut self.standard,
            MemoryClass::Ipu21Interleaved => &mut self.interleaved,
        };
        *target = target.saturating_add(bytes);
    }

    pub(crate) fn saturating_add(self, other: Self) -> Self {
        Self {
            standard: self.standard.saturating_add(other.standard),
            interleaved: self.interleaved.saturating_add(other.interleaved),
        }
    }

    pub fn fits_ipu21(self) -> bool {
        self.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            && self.total() <= u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
    }
}

/// Independent class maxima and the maximum simultaneous total. The allocator
/// fixes the interleaved arena boundary for the whole program, so feasibility
/// uses the sum of the class maxima even when they occur in different phases.
/// `total` remains useful for ranking the actual peak live working set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryPeaks {
    pub standard: u64,
    pub interleaved: u64,
    pub total: u64,
    /// Persistent standard-memory estimate for generated exchange rows.
    pub exchange_rows: u64,
    pub maximum_standard_allocation: u64,
    /// Largest amount by which one standard-addressed allocation exceeded
    /// both contiguous ranges left around the interleaved region.
    pub standard_contiguous_overflow: u64,
}

impl MemoryPeaks {
    pub(crate) fn observe(&mut self, usage: MemoryUsage, maximum_standard_allocation: u64) {
        self.standard = self.standard.max(usage.standard);
        self.interleaved = self.interleaved.max(usage.interleaved);
        self.total = self.total.max(usage.total());
        self.maximum_standard_allocation = self
            .maximum_standard_allocation
            .max(maximum_standard_allocation);
        let interleaved_boundary = self
            .interleaved
            .div_ceil(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
            * u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE);
        let upper_standard = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            .saturating_sub(interleaved_boundary);
        let contiguous_capacity =
            u64::from(crate::memory::IPU21_STANDARD_FIXED_BYTES).max(upper_standard);
        self.standard_contiguous_overflow = self
            .maximum_standard_allocation
            .saturating_sub(contiguous_capacity);
    }

    pub fn fits_ipu21_with_budget(
        self,
        reserved_standard_bytes: u64,
        tile_memory_budget_bytes: u64,
    ) -> bool {
        let partitioned_bytes = self
            .standard
            .saturating_add(self.aligned_interleaved_bytes())
            .saturating_add(reserved_standard_bytes);
        self.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            && partitioned_bytes
                <= tile_memory_budget_bytes.min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
            && self.standard_contiguous_overflow_with_reservation(reserved_standard_bytes) == 0
    }

    fn aligned_interleaved_bytes(self) -> u64 {
        self.interleaved
            .div_ceil(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
            .saturating_mul(u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE))
    }

    pub fn standard_contiguous_overflow_with_reservation(
        self,
        reserved_standard_bytes: u64,
    ) -> u64 {
        let interleaved_boundary = self.aligned_interleaved_bytes();
        let upper_standard = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
            .saturating_sub(interleaved_boundary);
        let lower_standard = u64::from(crate::memory::IPU21_STANDARD_FIXED_BYTES)
            .saturating_sub(reserved_standard_bytes.saturating_add(self.exchange_rows));
        self.maximum_standard_allocation
            .saturating_sub(lower_standard.max(upper_standard))
    }

    pub(crate) fn conservative_tensor_usage(self) -> MemoryUsage {
        MemoryUsage {
            standard: self.standard.saturating_sub(self.exchange_rows),
            interleaved: self.interleaved,
        }
    }
}

/// Storage visible at an operator boundary plus phase-local scratch. Peak is
/// the simultaneous requirement used for candidate feasibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub live: MemoryUsage,
    pub temporary: MemoryUsage,
    pub peak: MemoryUsage,
    pub exchange_row_bytes: u64,
    /// Largest phase-local standard-addressed buffer which must fit in one
    /// contiguous standard-memory range.
    pub maximum_standard_temporary_allocation: u64,
}

pub(crate) fn tensor_memory(tensor: &TensorType) -> MemoryUsage {
    let mut usage = MemoryUsage::default();
    usage.add_class(
        tensor.format.layout.memory_class,
        maximum_shard_bytes(tensor),
    );
    usage
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocationRequirement {
    access_tail: u64,
    distinct_element: bool,
}

fn allocation_memory(tensor: &TensorType, requirement: AllocationRequirement) -> MemoryUsage {
    let mut bytes = maximum_shard_bytes(tensor).saturating_add(requirement.access_tail);
    if requirement.distinct_element {
        let element = match tensor.format.layout.memory_class {
            MemoryClass::Ipu21Standard => ipu_package::TILE_MEMORY_ELEMENT_SIZE,
            MemoryClass::Ipu21Interleaved => ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE,
        };
        bytes = bytes.div_ceil(u64::from(element)) * u64::from(element);
    }
    let mut usage = MemoryUsage::default();
    usage.add_class(tensor.format.layout.memory_class, bytes);
    usage
}

fn allocation_requirements(
    operations: &[MidOperation],
) -> BTreeMap<MidValueId, AllocationRequirement> {
    let mut requirements = BTreeMap::<MidValueId, AllocationRequirement>::new();
    for operation in operations {
        if let Some(plan) = operation.operator_plan() {
            for (&id, operand) in operation.inputs.iter().zip(&plan.requirements.inputs) {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(operand.access_tail_bytes));
            }
            if let Some(&id) = operation.results.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.requirements.output.access_tail_bytes));
            }
            for operands in &plan.requirements.distinct_elements {
                for operand in operands {
                    let id = match operand {
                        MemoryOperand::Output => operation.results.first().copied(),
                        MemoryOperand::Input(index) => {
                            operation.inputs.get(usize::from(*index)).copied()
                        }
                    };
                    if let Some(id) = id {
                        requirements.entry(id).or_default().distinct_element = true;
                    }
                }
            }
        }
        if let Some(plan) = operation.conversion_plan() {
            if let Some(&id) = operation.inputs.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.input.access_tail_bytes));
            }
            if let Some(&id) = operation.results.first() {
                let requirement = requirements.entry(id).or_default();
                requirement.access_tail = requirement
                    .access_tail
                    .max(u64::from(plan.output.access_tail_bytes));
            }
        }
    }
    requirements
}

fn value_allocation(
    id: MidValueId,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirement>,
) -> MemoryUsage {
    allocation_memory(
        &values[id.index() as usize].tensor_type,
        requirements.get(&id).copied().unwrap_or_default(),
    )
}

fn maximum_standard_allocation(
    ids: &BTreeSet<MidValueId>,
    values: &[MidValue],
    requirements: &BTreeMap<MidValueId, AllocationRequirement>,
) -> u64 {
    ids.iter()
        .map(|&id| value_allocation(id, values, requirements).standard)
        .max()
        .unwrap_or(0)
}

pub(crate) fn operator_memory_estimate(
    dispatch: &OperatorDispatch,
    requirements: &StorageRequirements,
    inputs: &[TensorType],
    output: &TensorType,
) -> MemoryEstimate {
    let live = inputs.iter().zip(&requirements.inputs).fold(
        tensor_memory(output),
        |usage, (input, requirement)| {
            if requirement.materialization == OperandMaterialization::DispatchSlices {
                usage
            } else {
                usage.saturating_add(tensor_memory(input))
            }
        },
    );
    let mut temporary = MemoryUsage::default();
    let mut maximum_standard_temporary_allocation = 0u64;
    if let (
        OperatorDispatch::BlockedGemm {
            orientation,
            distribution:
                GemmDistribution::ParallelReduction {
                    column_partitions,
                    inner_partitions,
                    result_row_partitions,
                    result_column_partitions,
                    reduction_staging,
                    ..
                },
            output_column_block,
            ..
        },
        Some(first),
        Some(second),
    ) = (dispatch, inputs.first(), inputs.get(1))
    {
        let (left, right, left_requirement) = match orientation {
            crate::GemmOrientation::Normal => (first, second, requirements.inputs.first()),
            crate::GemmOrientation::Swapped => (second, first, requirements.inputs.get(1)),
        };
        let right_rank = right.shape.0.len();
        let (right_inner_axis, right_column_axis) = orientation.matrix_axes(right_rank);
        let inner_blocks = right.shape.0[right_inner_axis].div_ceil(AMP_INNER_BLOCK);
        let column_blocks = right.shape.0[right_column_axis].div_ceil(*output_column_block);
        let right_staging = u64::from(inner_blocks.div_ceil(u32::from(*inner_partitions)))
            .saturating_mul(u64::from(AMP_INNER_BLOCK))
            .saturating_mul(u64::from(
                column_blocks.div_ceil(u32::from(*column_partitions)),
            ))
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
        let mut convolution = MemoryUsage::default();
        convolution.add_class(MemoryClass::Ipu21Interleaved, right_staging);
        if left_requirement.is_some_and(|requirement| {
            requirement.materialization == OperandMaterialization::DispatchSlices
        }) {
            let requirement = left_requirement.expect("checked requirement");
            let mut left_staging =
                maximum_shard_bytes(left).saturating_add(u64::from(requirement.access_tail_bytes));
            let left_must_be_distinct = requirements.distinct_elements.iter().any(|operands| {
                operands.contains(&MemoryOperand::Input(
                    orientation.operand_indices().0 as u16,
                ))
            });
            if left_must_be_distinct {
                left_staging = left_staging
                    .div_ceil(u64::from(ipu_package::TILE_MEMORY_ELEMENT_SIZE))
                    .saturating_mul(u64::from(ipu_package::TILE_MEMORY_ELEMENT_SIZE));
            }
            convolution.add_class(left.format.layout.memory_class, left_staging);
            if left.format.layout.memory_class == MemoryClass::Ipu21Standard {
                maximum_standard_temporary_allocation =
                    maximum_standard_temporary_allocation.max(left_staging);
            }
        }
        // Compute retains one local partial alongside operand staging. The
        // later reduction ping-pongs an accumulator and result while its
        // staging policy bounds the simultaneously resident remote partials.
        let partial_bytes = maximum_shard_bytes(&dispatch.gemm_partial_tensor(output));
        let reduction_partial_bytes =
            if (*result_row_partitions, *result_column_partitions) != (1, 1) {
                maximum_shard_bytes(output)
            } else {
                partial_bytes
            };
        convolution.interleaved = convolution.interleaved.saturating_add(partial_bytes);
        let staged_remote_partials = match reduction_staging {
            crate::ReductionStaging::Complete => inner_partitions.saturating_sub(1),
            crate::ReductionStaging::Streamed => 1,
        };
        let reduction = MemoryUsage {
            standard: reduction_partial_bytes
                .saturating_mul(u64::from(staged_remote_partials).saturating_add(2)),
            interleaved: partial_bytes,
        };
        temporary = MemoryUsage {
            standard: convolution.standard.max(reduction.standard),
            interleaved: convolution.interleaved.max(reduction.interleaved),
        };
    }
    if let (OperatorDispatch::BlockedGemm { inner_block, .. }, Some(left), Some(requirement)) =
        (dispatch, inputs.first(), requirements.inputs.first())
        && requirement.materialization == OperandMaterialization::DispatchSlices
        && !matches!(
            dispatch,
            OperatorDispatch::BlockedGemm {
                distribution: GemmDistribution::ParallelReduction { .. },
                ..
            }
        )
    {
        let inner = left.shape.0.last().copied().map_or(1, u64::from).max(1);
        let bytes = maximum_shard_bytes(left)
            .div_ceil(inner)
            .saturating_mul(u64::from(*inner_block))
            .saturating_add(u64::from(requirement.access_tail_bytes));
        temporary.add_class(left.format.layout.memory_class, bytes);
    }
    if let (
        OperatorDispatch::BlockedGemm {
            inner_block,
            output_column_block,
            ..
        },
        Some(right),
    ) = (dispatch, inputs.get(1))
        && right.format.precision == Precision::F16
        && !matches!(
            dispatch,
            OperatorDispatch::BlockedGemm {
                distribution: GemmDistribution::ParallelReduction { .. },
                ..
            }
        )
        && gemm_uses_panel_buffer(dispatch, right, output)
    {
        // Each local output-column panel has one final kernel buffer reused
        // across K phases. Remote bytes can be exchanged directly into it.
        let output_columns =
            maximum_axis_shard_extent(output, output.shape.0.len().saturating_sub(1));
        let panels = output_columns.div_ceil(u64::from(*output_column_block));
        temporary.interleaved = panels
            .saturating_mul(u64::from(*inner_block))
            .saturating_mul(u64::from(*output_column_block))
            .saturating_mul(right.format.precision.bytes());
    }
    if let OperatorDispatch::BlockedAttention {
        query_block_rows,
        key_block_rows,
        padded_query_dimension,
        padded_value_dimension,
        ..
    } = dispatch
    {
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(*key_block_rows).max(1));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(*key_block_rows)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        // Every attention tile retains the current K and V panels. Prepared
        // panels are spread over query-tile owners; each needs both its
        // row-major gather buffer and its packed source until consumption.
        temporary.standard = temporary.standard.saturating_add(
            u64::from(*key_block_rows)
                .saturating_mul(u64::from(
                    padded_query_dimension.saturating_add(*padded_value_dimension),
                ))
                .saturating_mul(element_bytes)
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(
                    u64::from(*query_block_rows)
                        .saturating_mul(u64::from(key_block_rows.saturating_add(16)))
                        .saturating_mul(element_bytes),
                ),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*key_block_rows)))
                .saturating_mul(Precision::F32.bytes()),
        );
    }
    if let OperatorDispatch::MaterializedAttention {
        query_block_rows,
        padded_key_rows,
        padded_query_dimension,
        padded_value_dimension,
        ..
    } = dispatch
    {
        let element_bytes = inputs.first().map_or(Precision::F16.bytes(), |input| {
            input.format.precision.bytes()
        });
        let key_rows = inputs
            .get(1)
            .and_then(|key| key.shape.0.get(key.shape.0.len().saturating_sub(2)))
            .copied()
            .map_or(1, u64::from);
        let blocks = key_rows.div_ceil(u64::from(AMP_INNER_BLOCK));
        let panels_per_block = u64::from(
            padded_query_dimension
                .div_ceil(AMP_COLUMN_MICRO)
                .saturating_add(padded_value_dimension.div_ceil(AMP_COLUMN_MICRO)),
        );
        let query_rows = output
            .shape
            .0
            .get(output.shape.0.len().saturating_sub(2))
            .copied()
            .map_or(1, u64::from);
        let query_partitions = query_rows
            .div_ceil(u64::from(*query_block_rows).max(1))
            .max(1);
        let prepared_panels_per_owner = blocks
            .saturating_mul(panels_per_block)
            .div_ceil(query_partitions);
        let panel_bytes = u64::from(AMP_INNER_BLOCK)
            .saturating_mul(u64::from(AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        let operand_staging = u64::from(*padded_key_rows)
            .saturating_mul(u64::from(
                (*padded_query_dimension).max(*padded_value_dimension),
            ))
            .saturating_mul(element_bytes);
        maximum_standard_temporary_allocation =
            maximum_standard_temporary_allocation.max(operand_staging);
        let probability_state = u64::from(*query_block_rows)
            .saturating_mul(u64::from(padded_key_rows + AMP_COLUMN_MICRO))
            .saturating_mul(element_bytes);
        temporary.standard = temporary.standard.saturating_add(
            operand_staging
                .saturating_add(
                    prepared_panels_per_owner
                        .saturating_mul(panel_bytes)
                        .saturating_mul(2),
                )
                .saturating_add(if probability_state > operand_staging {
                    probability_state
                } else {
                    0
                }),
        );
        temporary.interleaved = temporary.interleaved.saturating_add(
            u64::from(*query_block_rows)
                .saturating_mul(u64::from((*padded_value_dimension).max(*padded_key_rows)))
                .saturating_mul(element_bytes),
        );
    }
    MemoryEstimate {
        live,
        temporary,
        peak: live.saturating_add(temporary),
        exchange_row_bytes: 0,
        maximum_standard_temporary_allocation,
    }
}

pub(crate) fn conversion_memory_estimate(
    input: &TensorType,
    output: &TensorType,
) -> MemoryEstimate {
    let live = tensor_memory(input).saturating_add(tensor_memory(output));
    MemoryEstimate {
        live,
        temporary: MemoryUsage::default(),
        peak: live,
        exchange_row_bytes: 0,
        maximum_standard_temporary_allocation: 0,
    }
}

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(initial, operations, outputs, values, &BTreeMap::new())
}

pub(crate) fn region_peak_memory_with_multiplicity(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> MemoryPeaks {
    let requirements = allocation_requirements(operations);
    let streamed_aliases = operations
        .iter()
        .filter_map(|operation| {
            let plan = operation.conversion_plan()?;
            if plan.output.materialization != OperandMaterialization::DispatchSlices {
                return None;
            }
            Some((*operation.results.first()?, *operation.inputs.first()?))
        })
        .collect::<BTreeMap<_, _>>();
    let mut uses = BTreeMap::<MidValueId, u32>::new();
    for input in operations.iter().flat_map(operation_value_inputs) {
        *uses.entry(*input).or_default() += 1;
    }
    for output in outputs {
        *uses.entry(*output).or_default() += 1;
    }
    let mut live_values = BTreeSet::new();
    for id in initial {
        live_values.insert(*id);
    }
    let mut peaks = MemoryPeaks::default();
    let observe = |peaks: &mut MemoryPeaks,
                   ids: &BTreeSet<MidValueId>,
                   temporary: MemoryUsage,
                   maximum_temporary: u64| {
        let roots = ids
            .iter()
            .map(|id| allocation_root(*id, &streamed_aliases))
            .collect::<BTreeSet<_>>();
        let live = roots.iter().fold(MemoryUsage::default(), |usage, id| {
            let allocation = value_allocation(*id, values, &requirements);
            let copies = u64::from(allocation_multiplicity.get(id).copied().unwrap_or(1));
            usage.saturating_add(MemoryUsage {
                standard: allocation.standard.saturating_mul(copies),
                interleaved: allocation.interleaved.saturating_mul(copies),
            })
        });
        peaks.observe(
            live.saturating_add(temporary),
            maximum_standard_allocation(&roots, values, &requirements).max(maximum_temporary),
        );
    };
    observe(&mut peaks, &live_values, MemoryUsage::default(), 0);
    for operation in operations {
        let mut during_values = live_values.clone();
        for result in &operation.results {
            during_values.insert(*result);
        }
        observe(
            &mut peaks,
            &during_values,
            operation.memory.temporary,
            operation.memory.maximum_standard_temporary_allocation,
        );
        for input in operation_value_inputs(operation) {
            if let Some(remaining) = uses.get_mut(input) {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    live_values.remove(input);
                }
            }
        }
        for result in &operation.results {
            if uses.get(result).copied().unwrap_or(0) != 0 {
                live_values.insert(*result);
            }
        }
    }
    observe(&mut peaks, &live_values, MemoryUsage::default(), 0);
    let exchange_rows = operations
        .iter()
        .map(|operation| operation.memory.exchange_row_bytes)
        .fold(0u64, u64::saturating_add);
    peaks.exchange_rows = exchange_rows;
    peaks.standard = peaks.standard.saturating_add(exchange_rows);
    peaks.total = peaks.total.saturating_add(exchange_rows);
    peaks
}

fn operation_value_inputs(operation: &MidOperation) -> Vec<&MidValueId> {
    let mut inputs = operation.inputs.iter().collect::<Vec<_>>();
    if let MidOperationKind::Repeat(repeat) = &operation.kind {
        inputs.extend(repeat.iterated_inputs.iter().flatten());
    }
    inputs
}

fn allocation_root(mut id: MidValueId, aliases: &BTreeMap<MidValueId, MidValueId>) -> MidValueId {
    while let Some(source) = aliases.get(&id) {
        id = *source;
    }
    id
}
