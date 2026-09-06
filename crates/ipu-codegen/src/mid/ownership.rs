//! Select ownership rotations before expanding the low schedule.

use super::*;
use crate::storage::{StorageError, TensorStorage, storage_bytes};

impl MidProgram {
    /// Offer independent reduction results on disjoint owner sets when their
    /// immediately grouped copies need local preparation. Keep this a candidate:
    /// moving the reduction roots can increase exchange or SRAM costs.
    pub(super) fn with_disjoint_copy_sources(&self) -> Option<Self> {
        let mut uses = vec![0; self.values.len()];
        let mut sums = BTreeSet::new();
        for operation in &self.operations {
            for input in &operation.inputs {
                uses[self.values[input.index() as usize].storage_group.index() as usize] += 1;
            }
            if let MidOperationKind::Repeat(repeat) = &operation.kind {
                for input in repeat.iterated_inputs.iter().flatten() {
                    uses[self.values[input.index() as usize].storage_group.index() as usize] += 1;
                }
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
            let count = independent_copy_prefix(&self.operations[index..]);
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
            let total = sources
                .iter()
                .map(|id| {
                    u32::from(
                        self.values[id.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .tiling
                            .tile_count,
                    )
                })
                .sum::<u32>();
            if sources.len() > 1 && total <= u32::from(self.tile_count) {
                let mut offset = self.values[sources[0].index() as usize].tile_offset;
                for source in sources {
                    let value = &self.values[source.index() as usize];
                    for alias in &mut result.values {
                        if alias.storage_group == value.storage_group {
                            changed |= alias.tile_offset != offset;
                            alias.tile_offset = offset;
                        }
                    }
                    offset = ((u32::from(offset)
                        + u32::from(value.tensor_type.format.layout.tiling.tile_count))
                        % u32::from(self.tile_count)) as u16;
                }
            }
            index += count.max(1);
        }
        changed.then_some(result)
    }

    pub(super) fn assign_parameter_tiles(&mut self) -> LoweringResult<()> {
        let parameter_origins = self
            .inputs
            .iter()
            .filter(|input| input.kind == GraphInputKind::Parameter)
            .map(|input| self.values[input.value.index() as usize].origin)
            .collect::<BTreeSet<_>>();
        let parameter_groups = self
            .values
            .iter()
            .filter(|value| parameter_origins.contains(&value.origin))
            .map(|value| value.storage_group)
            .collect::<BTreeSet<_>>();
        let mut loads = vec![0u64; usize::from(self.tile_count)];
        let mut offsets = BTreeMap::new();
        for value in &mut self.values {
            let layout = &value.tensor_type.format.layout;
            layout.validate_tile_count(self.tile_count)?;
            if !parameter_groups.contains(&value.storage_group) {
                continue;
            }
            let extents = layout.shard_extents(&value.tensor_type.shape)?;
            let bytes = extents
                .iter()
                .map(|(_, extents)| {
                    storage_bytes(TensorStorage {
                        format: &value.tensor_type.format,
                        extents,
                    })
                    .map(u64::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offset = *offsets
                .entry(value.storage_group)
                .or_insert_with(|| balanced_offset(&loads, &bytes));
            value.tile_offset = offset;
            for ((owner, _), bytes) in extents.iter().zip(bytes) {
                let tile = (usize::from(*owner) + usize::from(offset)) % loads.len();
                loads[tile] = loads[tile]
                    .checked_add(bytes)
                    .ok_or(StorageError::Overflow)?;
            }
        }
        Ok(())
    }
}

fn balanced_offset(loads: &[u64], bytes: &[u64]) -> u16 {
    let peak = loads.iter().copied().max().unwrap_or(0);
    // Each shard affects a different tile. The old peak covers unchanged tiles,
    // so candidate scoring needs neither a cloned load array nor a full rescan.
    (0..loads.len())
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
        assert_eq!(independent_copy_prefix(&program.operations[2..]), 2);
        let rotated = program.with_disjoint_copy_sources().unwrap();
        assert_eq!(
            rotated
                .values
                .iter()
                .map(|v| v.tile_offset)
                .collect::<Vec<_>>(),
            [0, 4, 0, 0, 0, 0]
        );
        assert!(program.values.iter().all(|v| v.tile_offset == 0));
        program.outputs.push(MidValueId(1));
        assert!(program.with_disjoint_copy_sources().is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId(1);
        program.outputs.push(MidValueId(3));
        assert!(program.with_disjoint_copy_sources().is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId(3);
        program.operations.push(copy(1, 2));
        assert!(program.with_disjoint_copy_sources().is_none());
        assert_eq!(independent_copy_prefix(&[copy(0, 2), copy(2, 3)]), 1);
        let mut boundary = copy(1, 3);
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("x", [16]).unwrap();
        graph.gelu(input).unwrap();
        boundary.source = Some(graph.operations()[0].id);
        assert_eq!(independent_copy_prefix(&[copy(0, 2), boundary]), 1);
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
