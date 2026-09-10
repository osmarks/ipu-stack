//! Select ownership rotations before expanding the low schedule.

use super::*;

impl MidProgram {
    /// Offer independent reduction results on disjoint owner sets when their
    /// immediately grouped copies need local preparation. Keep this a candidate:
    /// moving the reduction roots can increase exchange or SRAM costs.
    pub(super) fn with_disjoint_copy_sources(&self, checkpoints: bool) -> Option<Self> {
        let storage_groups = self
            .values
            .iter()
            .map(|value| value.storage_group)
            .collect::<Vec<_>>();
        let mut uses = vec![0; self.values.len()];
        let mut sums = BTreeSet::new();
        for operation in &self.operations {
            for input in operation.read_values() {
                uses[storage_groups[input.index() as usize].index() as usize] += 1;
            }
            if matches!(
                operation.kind,
                MidOperationKind::Primitive(Primitive::Sum { .. })
            ) {
                sums.extend(operation.results.iter().copied());
            }
        }
        let mut result = self.clone();
        let mut changed = false;
        let mut index = 0;
        while index < self.operations.len() {
            let count =
                independent_copy_prefix(&self.operations[index..], checkpoints, &storage_groups);
            let sources = self.operations[index..index + count]
                .iter()
                .filter_map(|operation| {
                    let MidOperationKind::Primitive(Primitive::Copy { mapping, .. }) =
                        &operation.kind
                    else {
                        return None;
                    };
                    let input = *operation.inputs.first()?;
                    let value = &self.values[input.index() as usize];
                    (mapping.view.is_some()
                        && sums.contains(&input)
                        && uses[value.storage_group.index() as usize] == 1
                        && !self.outputs.iter().any(|output| {
                            self.values[output.index() as usize].storage_group
                                == value.storage_group
                        })
                        && value.tensor_type.format.layout.order
                            == ElementOrder::Amp(AmpOrder::TransposedLeft))
                    .then_some(input)
                })
                .collect::<Vec<_>>();
            let total = sources.iter().map(|&id| self.owner_count(id)).sum::<u32>();
            if sources.len() > 1 && total <= u32::from(self.tile_count) {
                changed |= result.rotate_owners(
                    &sources,
                    self.values[sources[0].index() as usize].tile_offset,
                );
            }
            index += count.max(1);
        }
        changed.then_some(result)
    }

    /// Delay independent sums until their producers have all run. Only move
    /// results consumed through explicit copies: direct compute operands retain
    /// the owner alignment selected by their implementation.
    pub(super) fn with_overlapped_reductions(&self, limit: usize) -> Option<Self> {
        let groups = self
            .values
            .iter()
            .map(|v| v.storage_group)
            .collect::<Vec<_>>();
        let accesses = |ids: &[MidValueId]| {
            ids.iter()
                .map(|id| groups[id.index() as usize])
                .collect::<BTreeSet<_>>()
        };
        let conflicts = |a: &MidOperation, b: &MidOperation| {
            let ar = accesses(&a.inputs);
            let aw = accesses(&a.results);
            let br = accesses(&b.inputs);
            let bw = accesses(&b.results);
            !aw.is_disjoint(&br) || !aw.is_disjoint(&bw) || !ar.is_disjoint(&bw)
        };
        let eligible = |op: &MidOperation| {
            let [output] = op.results.as_slice() else {
                return false;
            };
            if !matches!(op.kind, MidOperationKind::Primitive(Primitive::Sum { .. })) {
                return false;
            }
            let group = groups[output.index() as usize];
            if self
                .outputs
                .iter()
                .any(|id| groups[id.index() as usize] == group)
            {
                return false;
            }
            let mut used = false;
            for consumer in &self.operations {
                if consumer
                    .read_values()
                    .any(|id| groups[id.index() as usize] == group)
                {
                    used = true;
                    if !matches!(
                        consumer.kind,
                        MidOperationKind::Primitive(Primitive::Copy { .. })
                    ) {
                        return false;
                    }
                }
            }
            used
        };
        let mut result = self.clone();
        let mut changed = false;
        let mut start = 0;
        while start < result.operations.len() {
            if !eligible(&result.operations[start]) {
                start += 1;
                continue;
            }
            let mut selected = vec![start];
            let owners = |op: &MidOperation| self.owner_count(op.results[0]);
            let mut total = owners(&result.operations[start]);
            for next in start + 1..result.operations.len() {
                let operation = &result.operations[next];
                if selected.len() >= limit
                    || !(matches!(
                        operation.kind,
                        MidOperationKind::Primitive(Primitive::Copy { .. } | Primitive::Sum { .. })
                    ) || matches!(&operation.kind, MidOperationKind::Primitive(Primitive::Compute { output_aliases, product: Some(_), .. }) if output_aliases.is_empty()))
                    || selected
                        .iter()
                        .any(|&index| conflicts(&result.operations[index], operation))
                {
                    break;
                }
                if eligible(operation) {
                    total += owners(operation);
                    if total > u32::from(self.tile_count) {
                        break;
                    }
                    selected.push(next);
                }
            }
            if selected.len() < 2 {
                start += 1;
                continue;
            }
            let insertion = selected.last().copied().unwrap() + 1 - selected.len();
            let mut sums = Vec::new();
            for index in selected.into_iter().rev() {
                sums.push(result.operations.remove(index));
            }
            sums.reverse();
            result.rotate_owners(
                &sums.iter().map(|sum| sum.results[0]).collect::<Vec<_>>(),
                self.values[sums[0].results[0].index() as usize].tile_offset,
            );
            start = insertion + sums.len();
            result.operations.splice(insertion..insertion, sums);
            changed = true;
        }
        changed.then_some(result)
    }

    fn owner_count(&self, value: MidValueId) -> u32 {
        u32::from(
            self.values[value.index() as usize]
                .tensor_type
                .format
                .layout
                .tiling
                .tile_count,
        )
    }

    /// Rotate complete alias groups together; callers check the combined owner count.
    fn rotate_owners(&mut self, sources: &[MidValueId], mut offset: u16) -> bool {
        let mut changed = false;
        for &source in sources {
            let group = self.values[source.index() as usize].storage_group;
            for alias in &mut self.values {
                if alias.storage_group == group {
                    changed |= alias.tile_offset != offset;
                    alias.tile_offset = offset;
                }
            }
            offset = ((u32::from(offset) + self.owner_count(source)) % u32::from(self.tile_count))
                as u16;
        }
        changed
    }
}

/// Keep small broadcasts in transfer-sized chunks rather than eight-byte
/// shards, unless the resident sequence would exceed the allocation budget.
pub(super) fn compact_parameter_layout(
    tensor: &TensorType,
    copies: u32,
    config: &PipelineConfig,
) -> Option<Layout> {
    let layout = if tensor.format.layout.order == ElementOrder::RowMajor {
        let grain = tensor.format.layout.tiling.linear_grain()?;
        let limit = config
            .tile_memory_budget_bytes
            .min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
            .saturating_sub(config.standard_memory_reservation_bytes);
        let chunks_per_owner =
            limit / (u64::from(grain) * tensor.format.precision.bytes() * u64::from(copies));
        if chunks_per_owner == 0 {
            return None;
        }
        let required = tensor
            .shape
            .elements()
            .div_ceil(u64::from(grain))
            .div_ceil(chunks_per_owner);
        let owners = tensor
            .shape
            .elements()
            .saturating_mul(tensor.format.precision.bytes())
            .div_ceil(256)
            .min(u64::from(tensor.format.layout.tiling.tile_count))
            .max(required);
        if owners > u64::from(tensor.format.layout.tiling.tile_count) {
            return None;
        }
        Layout::logical_linear(owners as u16, grain)
    } else {
        compact_matrix_layout(tensor, config.tile_count)?
    };
    let bytes = layout
        .resolve(&tensor.shape)
        .ok()?
        .maximum_tile_elements()
        .saturating_mul(tensor.format.precision.bytes())
        .saturating_mul(u64::from(copies));
    (bytes.saturating_add(config.standard_memory_reservation_bytes)
        <= config
            .tile_memory_budget_bytes
            .min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)))
    .then_some(layout)
}

// Preserve micro-panel traversal, not the consumer's macro-panel dimensions.
// The existing panel exchange can regroup these fragments without permutation;
// retaining large macro panels needlessly restricts persistent ownership.
fn compact_matrix_layout(tensor: &TensorType, tiles: u16) -> Option<Layout> {
    use crate::{AmpOrder, AxisTiling, BlockMajorOrder, Padding, TensorAxis, TensorTiling};
    let dimensions = &tensor.shape.0;
    let (&rows, &columns) = (
        dimensions.get(dimensions.len().checked_sub(2)?)?,
        dimensions.last()?,
    );
    let micro = 32 / tensor.format.precision.bytes() as u32;
    let order = match tensor.format.layout.order {
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
            ElementOrder::Amp(AmpOrder::TransposedLeft)
        }
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
            ElementOrder::Amp(AmpOrder::Left)
        }
        order => order,
    };
    let (row_grain, column_grain) = match order {
        ElementOrder::RowMajor => return None,
        ElementOrder::Amp(AmpOrder::Left) => (1, micro),
        ElementOrder::Amp(AmpOrder::TransposedLeft) => (micro, 1),
        ElementOrder::Amp(AmpOrder::Output) => (1, 16),
        ElementOrder::Amp(AmpOrder::TransposedOutput) => (16, 1),
        ElementOrder::Amp(AmpOrder::TransposedRight) => (16, micro),
        ElementOrder::BlockMajor(_) => unreachable!(),
    };
    if row_grain == 0 || column_grain == 0 {
        return None;
    }
    let row_blocks = rows.div_ceil(row_grain);
    let column_blocks = columns.div_ceil(column_grain);
    let (_, _, row_parts, column_parts) = (1..=row_blocks.min(u32::from(tiles)))
        .filter_map(|row_parts| {
            let column_parts = column_blocks.min(u32::from(tiles) / row_parts);
            if column_parts == 0 {
                return None;
            }
            let shard = u64::from(row_blocks.div_ceil(row_parts))
                * u64::from(row_grain)
                * u64::from(column_blocks.div_ceil(column_parts))
                * u64::from(column_grain);
            Some((
                shard,
                shard * u64::from(row_parts * column_parts),
                row_parts,
                column_parts,
            ))
        })
        .min()?;
    Some(Layout {
        order,
        tiling: TensorTiling {
            tile_count: (row_parts * column_parts) as u16,
            replicas: 1,
            axes: vec![
                AxisTiling::new(
                    TensorAxis::FromEnd(2),
                    row_parts as u16,
                    row_grain,
                    Padding::Zero,
                ),
                AxisTiling::new(
                    TensorAxis::FromEnd(1),
                    column_parts as u16,
                    column_grain,
                    Padding::Zero,
                ),
            ],
        },
        memory_class: tensor.format.layout.memory_class,
    })
}

/// Select persistent homes before capacity screening. Sequence members share
/// one rotation; derived values follow that rotation but are not counted again.
pub(super) fn assign_parameter_tiles(
    values: &mut [MidValue],
    parameters: &[MidValueId],
    copies: &BTreeMap<MidValueId, u32>,
    tile_count: u16,
) -> LoweringResult<bool> {
    let mut groups = BTreeMap::<MidValueId, Vec<u64>>::new();
    for &id in parameters {
        let value = &values[id.index() as usize];
        let layout = &value.tensor_type.format.layout;
        layout.validate_tile_count(tile_count)?;
        let resolved = layout.resolve(&value.tensor_type.shape)?;
        let bytes = groups
            .entry(value.storage_group)
            .or_insert_with(|| vec![0; usize::from(tile_count)]);
        for owner in 0..layout.tiling.tile_count {
            bytes[usize::from(owner)] = bytes[usize::from(owner)].saturating_add(
                resolved
                    .tile_elements(owner)
                    .saturating_mul(value.tensor_type.format.precision.bytes())
                    .saturating_mul(u64::from(copies.get(&id).copied().unwrap_or(1))),
            );
        }
    }
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(id, bytes)| {
        (
            std::cmp::Reverse(bytes.iter().copied().max().unwrap_or(0)),
            *id,
        )
    });
    let mut loads = vec![0u64; usize::from(tile_count)];
    let mut offsets = BTreeMap::new();
    for (group, bytes) in groups {
        let offset = balanced_offset(&loads, &bytes);
        for (owner, bytes) in bytes.into_iter().enumerate() {
            let load = &mut loads[(owner + usize::from(offset)) % usize::from(tile_count)];
            *load = load.saturating_add(bytes);
        }
        offsets.insert(group, offset);
    }
    let mut changed = false;
    for value in values {
        if let Some(&offset) = offsets.get(&value.storage_group) {
            changed |= value.tile_offset != offset;
            value.tile_offset = offset;
        }
    }
    Ok(changed)
}

fn balanced_offset(loads: &[u64], bytes: &[u64]) -> u16 {
    if bytes.windows(2).all(|pair| pair[0] == pair[1]) && bytes.len() == loads.len() {
        return 0;
    }
    let peak = loads.iter().copied().max().unwrap_or(0);
    // Each shard affects a different tile. The old peak covers unchanged tiles,
    // so candidate scoring needs neither a cloned load array nor a full rescan.
    (0..loads.len())
        .step_by(loads.len().div_ceil(128))
        .min_by_key(|&offset| {
            let peak = bytes
                .iter()
                .enumerate()
                .fold(peak, |peak, (logical, &bytes)| {
                    peak.max(loads[(logical + offset) % loads.len()].saturating_add(bytes))
                });
            (peak, offset)
        })
        .unwrap_or(0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_fp8_homes_preserve_panels_and_avoid_byte_permutations() {
        let config = PipelineConfig::new(64);
        let precision = Precision::F8F143 { scale_exponent: -4 };
        for order in [
            ElementOrder::Amp(AmpOrder::TransposedLeft),
            ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            }),
        ] {
            let mut target = Layout::amp_transposed_left_parallel_grid(64, 8, 2, 2, 2);
            target.order = order;
            let native = TensorType::new([128, 256], precision, target.clone());
            let home = compact_parameter_layout(&native, 27, &config).unwrap();
            assert_eq!(
                home.order.micro_panel_order(),
                target.order.micro_panel_order()
            );
            assert_eq!(home.tiling.replicas, 1);
            assert!(home.tiling.tile_count > target.tiling.tile_count);
            for (layout, encodable) in [(Layout::logical_linear(64, 8), false), (home, true)] {
                let source = TensorType::new([128, 256], precision, layout);
                let output = TensorType::new([128, 256], precision, target.clone());
                let program = MidProgram {
                    tile_count: 64,
                    inputs: vec![MidInput {
                        name: "weight".into(),
                        kind: GraphInputKind::Parameter,
                        value: MidValueId(0),
                    }],
                    values: [source.clone(), output.clone()]
                        .into_iter()
                        .enumerate()
                        .map(|(id, tensor_type)| MidValue {
                            id: MidValueId(id as u32),
                            tile_offset: 0,
                            tensor_type,
                            origin: ValueId::from_index(0),
                            storage_group: MidValueId(id as u32),
                        })
                        .collect(),
                    operations: vec![MidOperation {
                        source: None,
                        inputs: vec![MidValueId(0)],
                        results: vec![MidValueId(1)],
                        kind: MidOperationKind::Convert(ConversionPlan {
                            strategy: layout_conversion_strategy(
                                &source.format.layout,
                                &output.format.layout,
                            ),
                            input: OperandRequirement::new(source.format, 8),
                            output: OperandRequirement::new(output.format, 8),
                        }),
                        estimated_cycles: 0,
                        estimated_exchange_cycles: 0,
                    }],
                    outputs: vec![MidValueId(1)],
                    ..Default::default()
                };
                let expanded = crate::expand_tiles(&program).unwrap();
                assert_eq!(
                    expanded
                        .local_copies
                        .iter()
                        .all(|copy| crate::tile::local_copy_call(copy).is_some()),
                    encodable
                );
            }
        }
    }

    #[test]
    fn compact_weight_homes_do_not_inherit_gemm_macro_panel_padding() {
        let config = PipelineConfig::new(1472);
        for shape in [[1152, 3456], [1152, 1152], [1152, 4304], [4304, 1152]] {
            let tensor = TensorType::new(
                shape,
                Precision::F8F143 { scale_exponent: -4 },
                Layout::block_major_matrix_storage(288, 16, 1, 1, 1, MemoryClass::Ipu21Standard),
            );
            let home = compact_parameter_layout(&tensor, 27, &config).unwrap();
            let shard = home.resolve(&tensor.shape).unwrap().maximum_tile_elements();
            let ideal = tensor
                .shape
                .elements()
                .div_ceil(u64::from(config.tile_count));
            assert!(
                shard * 100 <= ideal * 110,
                "{shape:?}: shard {shard}, ideal {ideal}"
            );
        }
    }

    #[test]
    fn independent_copy_roots_rotate_without_moving_partials_or_shared_results() {
        let mut program = MidProgram {
            tile_count: 16,
            ..MidProgram::default()
        };
        let mut layout = Layout::row_sharded(4);
        layout.order = ElementOrder::Amp(AmpOrder::TransposedLeft);
        for id in 0..6 {
            program.values.push(MidValue {
                id: MidValueId(id),
                tile_offset: 0,
                tensor_type: TensorType::new([1, 16, 16], Precision::F16, layout.clone()),
                origin: ValueId::from_index(id),
                storage_group: MidValueId(id),
            });
        }
        let operation = |input, output, primitive| MidOperation {
            source: None,
            inputs: vec![MidValueId(input)],
            results: vec![MidValueId(output)],
            kind: MidOperationKind::Primitive(primitive),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let copy = |input, output| {
            operation(
                input,
                output,
                Primitive::Copy {
                    mapping: CoordinateMapping {
                        offsets: vec![],
                        view: Some(AxisFactorView {
                            split_axis: 2,
                            merge_axis: 0,
                            factor: 1,
                            reversed: false,
                        }),
                    },
                    reuse_local: false,
                },
            )
        };
        program.operations = vec![
            operation(
                4,
                0,
                Primitive::Sum {
                    axis: 0,
                    staging: ReductionStaging::Complete,
                },
            ),
            operation(
                5,
                1,
                Primitive::Sum {
                    axis: 0,
                    staging: ReductionStaging::Complete,
                },
            ),
            copy(0, 2),
            copy(1, 3),
        ];
        let groups = (0..6).map(MidValueId).collect::<Vec<_>>();
        assert_eq!(
            independent_copy_prefix(&program.operations[2..], true, &groups),
            2
        );
        let rotated = program.with_disjoint_copy_sources(true).unwrap();
        assert_eq!(
            rotated
                .values
                .iter()
                .map(|v| v.tile_offset)
                .collect::<Vec<_>>(),
            [0, 4, 0, 0, 0, 0]
        );
        assert!(program.values.iter().all(|v| v.tile_offset == 0));
        let mut delayed = program.clone();
        delayed.operations.insert(1, copy(4, 5));
        let overlapped = delayed.with_overlapped_reductions(2).unwrap();
        assert_eq!(overlapped.operations[0].results, [MidValueId(5)]);
        assert_eq!(overlapped.operations[1].results, [MidValueId(0)]);
        assert_eq!(overlapped.operations[2].results, [MidValueId(1)]);
        assert_eq!(overlapped.values[1].tile_offset, 4);
        assert!(delayed.with_overlapped_reductions(1).is_none());
        delayed.operations[1].kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::Gelu,
            operands: Vec::new(),
            product: None,
            output_aliases: vec![(0, 0)],
        });
        assert!(delayed.with_overlapped_reductions(2).is_none());
        delayed.operations[1] = copy(5, 4); // Would overwrite a delayed partial.
        assert!(delayed.with_overlapped_reductions(2).is_none());

        program.outputs.push(MidValueId(1));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId(1);
        program.outputs.push(MidValueId(3));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId(3);
        program.operations.push(copy(1, 2));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), copy(2, 3)], false, &groups),
            1
        );
        let mut boundary = copy(1, 3);
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("x", [16]).unwrap();
        graph.gelu(input).unwrap();
        boundary.source = Some(graph.operations()[0].id);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], true, &groups),
            1
        );
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &groups),
            2
        );
        let mut aliases = groups.clone();
        aliases[1] = MidValueId(2);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &aliases),
            1
        );
        aliases = groups.clone();
        aliases[3] = MidValueId(0);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &aliases),
            1
        );
        program.operations.truncate(4);
        program.operations[3] = boundary;
        assert!(program.with_disjoint_copy_sources(true).is_none());
        assert!(program.with_disjoint_copy_sources(false).is_some());
    }

    #[test]
    fn parameter_homes_group_sequences_without_counting_temporary_derivatives() {
        let mut values = (0..5)
            .map(|i| MidValue {
                id: MidValueId(i),
                origin: ValueId::from_index(i),
                storage_group: MidValueId(if i == 1 { 0 } else { i }),
                tile_offset: 0,
                tensor_type: TensorType::new([64], Precision::F16, Layout::logical_linear(2, 4)),
            })
            .collect::<Vec<_>>();
        values[4].storage_group = MidValueId(2);
        values[4].tensor_type =
            TensorType::new([8192], Precision::F16, Layout::logical_linear(8, 4));
        let parameters = [MidValueId(0), MidValueId(1), MidValueId(2), MidValueId(3)];
        assign_parameter_tiles(&mut values, &parameters, &BTreeMap::new(), 8).unwrap();
        assert_eq!(values[0].tile_offset, values[1].tile_offset);
        assert_eq!(values[2].tile_offset, values[4].tile_offset);
        assert_ne!(values[0].tile_offset, values[2].tile_offset);
        let offsets = values.iter().map(|v| v.tile_offset).collect::<Vec<_>>();
        values[4].tensor_type.shape = TensorShape(vec![16384]);
        assign_parameter_tiles(&mut values, &parameters, &BTreeMap::new(), 8).unwrap();
        assert_eq!(
            offsets,
            values.iter().map(|v| v.tile_offset).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rotations_match_full_load_array_scoring() {
        let mut random = fastrand::Rng::with_seed(0x0074_696c_6573);
        for _ in 0..1000 {
            let tiles = random.usize(1..=64);
            let loads = (0..tiles).map(|_| random.u64(0..4096)).collect::<Vec<_>>();
            let bytes = (0..random.usize(0..=tiles))
                .map(|_| random.u64(0..4096))
                .collect::<Vec<_>>();
            let expected = (0..tiles)
                .min_by_key(|&offset| {
                    let mut candidate = loads.clone();
                    for (logical, &bytes) in bytes.iter().enumerate() {
                        let tile = (logical + offset) % tiles;
                        candidate[tile] = candidate[tile].saturating_add(bytes);
                    }
                    (candidate.into_iter().max().unwrap(), offset)
                })
                .unwrap() as u16;
            assert_eq!(balanced_offset(&loads, &bytes), expected);
        }
    }
}
