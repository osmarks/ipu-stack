//! Pack smaller panels on additional owners, then transfer packed storage.

use crate::kernel::TileKernelSpec;
use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
    OperandIndexing, ProgramError, WorkSite,
};
use crate::tensor::{
    AxisTiling, BlockMajorOrder, ElementOrder, Layout, OwnerMap, Padding, Precision, TensorAxis,
    TensorTiling, TensorType,
};

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::num::NonZeroU16;

/// Retile a copy through smaller panels on an explicit working domain. The
/// destination keeps its selected layout and home; workspace ownership is an
/// absolute assignment, independent of constructor result-base rotations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PanelPacking {
    pub rows: NonZeroU16,
    pub workspace: OwnerMap,
}

impl MidProgram {
    /// Report inexpensive choices before packing replaces the original copy.
    /// The planner supplies its row neighborhood; this module owns feasibility.
    pub(crate) fn packing_choices(
        &self,
        rows: &[u16],
        selected: &BTreeMap<WorkSite, PanelPacking>,
    ) -> BTreeMap<WorkSite, Vec<PanelPacking>> {
        self.walk_operations()
            .filter_map(|operation| {
                let target = packing_target(operation, &self.values)?;
                let site = operation.work_site()?;
                let workspace = selected
                    .get(&site)
                    .map_or(&target.owners, |choice| &choice.workspace);
                let choices = rows
                    .iter()
                    .copied()
                    .filter_map(NonZeroU16::new)
                    .filter_map(|rows| {
                        let layout = packing_layout(&target.tensor_type, self.tile_count, rows)?;
                        let workspace = if workspace
                            .tile(layout.tiling.tile_count - 1, self.tile_count)
                            .is_some()
                        {
                            workspace.clone()
                        } else {
                            OwnerMap::default()
                        };
                        Some(PanelPacking { rows, workspace })
                    })
                    .collect::<Vec<_>>();
                (!choices.is_empty()).then_some((site, choices))
            })
            .collect()
    }

    /// Resolve the old global preference once. It packed every eligible copy
    /// using the destination's map, and fell back to the original program if a
    /// workspace exceeded that domain. Subsequent recipes contain named choices.
    pub(crate) fn legacy_packing_choices(
        &self,
        rows: NonZeroU16,
    ) -> Result<BTreeMap<WorkSite, PanelPacking>, ProgramError> {
        let mut choices = BTreeMap::new();
        for operation in self.walk_operations() {
            let Some(target) = packing_target(operation, &self.values) else {
                continue;
            };
            let Some(layout) = packing_layout(&target.tensor_type, self.tile_count, rows) else {
                continue;
            };
            if target
                .owners
                .validate(layout.tiling.tile_count, self.tile_count)
                .is_err()
            {
                return Ok(BTreeMap::new());
            }
            let site = operation.work_site().ok_or_else(|| {
                ProgramError::Invalid("legacy packing request cannot name an anonymous copy".into())
            })?;
            choices.insert(
                site,
                PanelPacking {
                    rows,
                    workspace: target.owners.clone(),
                },
            );
        }
        Ok(choices)
    }

    pub(crate) fn apply_packing(
        &mut self,
        choices: &BTreeMap<WorkSite, PanelPacking>,
    ) -> Result<(), ProgramError> {
        if choices.is_empty() {
            return Ok(());
        }
        let mut layouts = BTreeMap::new();
        for operation in self.walk_operations() {
            let Some(site) = operation.work_site() else {
                continue;
            };
            let Some(choice) = choices.get(&site) else {
                continue;
            };
            let layout = packing_target(operation, &self.values)
                .and_then(|value| packing_layout(&value.tensor_type, self.tile_count, choice.rows))
                .ok_or_else(|| {
                    ProgramError::Invalid(format!("packing choice cannot apply at {site:?}"))
                })?;
            choice
                .workspace
                .validate(layout.tiling.tile_count, self.tile_count)?;
            layouts.insert(site, layout);
        }
        if let Some(site) = choices.keys().find(|site| !layouts.contains_key(site)) {
            return Err(ProgramError::Invalid(format!(
                "packing choice is unavailable at {site:?}"
            )));
        }
        // All requested geometry and workspaces are checked before modifying work.
        distribute_region(&mut self.operations, &mut self.values, choices, &layouts);
        Ok(())
    }
}

fn packing_target<'a>(operation: &MidOperation, values: &'a [MidValue]) -> Option<&'a MidValue> {
    if !matches!(operation.kind, MidOperationKind::Copy { .. }) {
        return None;
    }
    let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice()) else {
        return None;
    };
    let input = &values[input.index() as usize].tensor_type.format;
    let output = &values[output.index() as usize];
    (input.precision == Precision::F16
        && input.layout.order == ElementOrder::RowMajor
        && output.tensor_type.format.precision == Precision::F16)
        .then_some(output)
}

fn packing_layout(tensor: &TensorType, capacity: u16, block_rows: NonZeroU16) -> Option<Layout> {
    let block_rows = block_rows.get();
    let ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block,
        column_block,
    }) = tensor.format.layout.order
    else {
        return None;
    };
    if block_rows >= row_block || column_block != 16 || tensor.format.layout.tiling.replicas != 1 {
        return None;
    }
    let rank = tensor.shape.0.len();
    let row_axis = rank.checked_sub(2)?;
    let parts = u16::try_from(tensor.shape.0[row_axis].div_ceil(u32::from(block_rows))).ok()?;
    let original = &tensor.format.layout.tiling;
    let mut axes = Vec::new();
    let mut tiles = 1u16;
    for index in (0..rank).rev() {
        let axis = if index == row_axis {
            AxisTiling::new(
                TensorAxis::FromStart(index as u16),
                parts,
                u32::from(block_rows),
                Padding::Zero,
            )
        } else if let Some(axis) = original
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(rank) == Ok(index))
        {
            *axis
        } else {
            continue;
        };
        axes.push(axis.with_tile_stride(tiles));
        tiles = tiles.checked_mul(axis.partitions)?;
    }
    if tiles <= original.tile_count || tiles > capacity {
        return None;
    }
    let mut layout = Layout::row_major(TensorTiling {
        tile_count: tiles,
        replicas: 1,
        axes,
    });
    layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block: block_rows,
        column_block,
    });
    if layout.resolve(&tensor.shape).ok()?.has_empty_shards() {
        return None;
    }
    Some(layout)
}

fn distribute_region(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    choices: &BTreeMap<WorkSite, PanelPacking>,
    layouts: &BTreeMap<WorkSite, Layout>,
) {
    let mut result = Vec::new();
    for mut operation in std::mem::take(operations) {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            distribute_region(&mut repeat.body.operations, values, choices, layouts);
        }
        let Some(site) = operation
            .work_site()
            .filter(|site| layouts.contains_key(site))
        else {
            result.push(operation);
            continue;
        };
        let target = values[operation.results[0].index() as usize].clone();
        let layout = layouts[&site].clone();
        let choice = &choices[&site];
        let logical = MidValueId(values.len() as u32);
        let packed = MidValueId(logical.index() + 1);
        let mut tensor = target.tensor_type.clone();
        tensor.format.layout = layout.clone();
        let mut row_major = tensor.clone();
        row_major.format.layout.order = ElementOrder::RowMajor;
        for (id, tensor_type) in [(logical, row_major.clone()), (packed, tensor)] {
            values.push(MidValue {
                id,
                tensor_type,
                storage_group: id,
                owners: choice.workspace.clone(),
                ..target
            });
        }
        let pack = MidOperation {
            site: operation.site.as_ref().map(|site| site.child("pack")),
            inputs: vec![logical],
            results: vec![packed],
            source: operation.source,
            kind: MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::Rearrange {
                    from: row_major.format.layout,
                    to: layout,
                },
                operands: vec![OperandIndexing::Elementwise { result: 0 }],
                output_aliases: Vec::new(),
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let transfer = MidOperation {
            site: operation.site.as_ref().map(|site| site.child("distribute")),
            inputs: vec![packed],
            results: operation.results.clone(),
            source: operation.source,
            kind: MidOperationKind::Copy {
                policy: crate::CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
                mapping: CoordinateMapping::default(),
                reuse_local: true,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        operation.results = vec![logical];
        result.extend([operation, pack, transfer]);
    }
    *operations = result;
}

#[cfg(test)]
mod tests {
    use crate::graph::{GraphInputKind, ValueId};
    use crate::mid::MidInput;

    use crate::tensor::{AxisFactorView, MemoryClass};

    use super::*;
    fn packing_copy(view: bool) -> (crate::ComputeGraph, MidProgram) {
        let mut provenance = crate::ComputeGraph::new();
        let argument = provenance.host_input("x", [1, 16]).unwrap();
        provenance.gelu(argument).unwrap();
        let source = TensorType::new(
            if view {
                vec![1, 729, 144]
            } else {
                vec![2, 729, 72]
            },
            Precision::F16,
            Layout::row_sharded(64),
        );
        let target = TensorType::new(
            [2, 729, 72],
            Precision::F16,
            Layout {
                order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: 768,
                    column_block: 16,
                }),
                memory_class: MemoryClass::Ipu21Standard,
                tiling: TensorTiling {
                    tile_count: 10,
                    replicas: 1,
                    axes: vec![
                        AxisTiling::new(TensorAxis::FromEnd(1), 5, 16, Padding::Zero)
                            .with_tile_stride(1),
                        AxisTiling::new(TensorAxis::FromEnd(2), 1, 768, Padding::Zero)
                            .with_tile_stride(5),
                        AxisTiling::new(TensorAxis::FromEnd(3), 2, 1, Padding::Reject)
                            .with_tile_stride(5),
                    ],
                },
            },
        );
        let values = [source, target]
            .into_iter()
            .enumerate()
            .map(|(index, tensor_type)| {
                let id = MidValueId(index as u32);
                MidValue {
                    id,
                    tensor_type,
                    origin: ValueId::from_index(index as u32),
                    storage_group: id,
                    owners: crate::tensor::OwnerMap::default(),
                }
            })
            .collect();
        let program = MidProgram {
            tile_count: 64,
            values,
            inputs: vec![MidInput {
                name: "x".into(),
                kind: GraphInputKind::Host,
                value: MidValueId(0),
            }],
            outputs: vec![MidValueId(1)],
            operations: vec![MidOperation {
                site: Some("packing".into()),
                source: Some(provenance.operations()[0].id),
                inputs: vec![MidValueId(0)],
                results: vec![MidValueId(1)],
                kind: MidOperationKind::Copy {
                    policy: crate::CopyPolicy::Automatic,
                    packing: crate::PackingPolicy::Automatic,
                    mapping: CoordinateMapping {
                        offsets: vec![],
                        view: view.then_some(AxisFactorView::new(2, 0, 2)),
                    },
                    reuse_local: true,
                },
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            }],
            ..MidProgram::default()
        };
        (provenance, program)
    }

    #[test]
    fn packing_choices_preserve_other_copies_and_independent_result_homes() {
        let (graph, mut raw) = packing_copy(false);
        let mut other = raw.values[1].clone();
        other.id = MidValueId(2);
        other.storage_group = other.id;
        raw.values.push(other);
        let mut copy = raw.operations[0].clone();
        copy.site = Some("other".into());
        copy.results = vec![MidValueId(2)];
        raw.operations.push(copy.clone());
        raw.outputs.push(MidValueId(2));
        let site = raw.operations[0].work_site().unwrap();
        let small_home = OwnerMap::embedded((0..10).map(|i| 1 + 4 * i).collect::<Vec<_>>());
        let workspace = OwnerMap::embedded((0..64).rev().collect::<Vec<_>>());
        let mut recipe = crate::planner::Recipe::default();
        recipe.owners.results.insert(
            raw.operations[0].result_site(0).unwrap(),
            small_home.clone(),
        );
        recipe.packing.insert(
            site.clone(),
            PanelPacking {
                rows: NonZeroU16::new(128).unwrap(),
                workspace: workspace.clone(),
            },
        );
        let replay: crate::planner::Recipe =
            serde_json::from_slice(&serde_json::to_vec(&recipe).unwrap()).unwrap();
        assert!(recipe == replay);
        let mut bound = raw.clone();
        bound.apply_ownership(&recipe.owners).unwrap();
        let choices = bound.packing_choices(&[128], &BTreeMap::new());
        assert_eq!(
            choices[&site][0].workspace,
            OwnerMap::default(),
            "the small result domain cannot hold the packing workspace"
        );
        let before = bound.clone();
        let mut invalid = recipe.packing.clone();
        invalid.get_mut(&site).unwrap().workspace = OwnerMap::embedded(vec![0]);
        assert!(bound.apply_packing(&invalid).is_err());
        assert_eq!(bound, before);
        let mut missing = site.clone();
        missing.local = missing.local.child("missing");
        invalid = BTreeMap::from([(missing, recipe.packing[&site].clone())]);
        assert!(bound.apply_packing(&invalid).is_err());
        assert_eq!(bound, before);
        bound.apply_packing(&recipe.packing).unwrap();
        bound.refresh_estimates().unwrap();
        assert_eq!(bound.values[1].owners, small_home);
        assert_eq!(bound.operations.last().unwrap(), &copy);
        let packed = bound
            .operations
            .iter()
            .find(|op| matches!(op.kind, MidOperationKind::Compute(_)))
            .unwrap()
            .results[0];
        let low = crate::lower_to_tiles(
            &crate::low::expand::expand_tiles(&bound, false).unwrap(),
            false,
        );
        crate::place(&low).unwrap();
        let actual_tiles = low
            .value_views(packed)
            .iter()
            .map(|id| low.shards[id.shard.index() as usize].tile)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            actual_tiles,
            (0..60)
                .map(|owner| workspace.tile(owner, 64).unwrap())
                .collect()
        );

        let mapping = (0..64)
            .map(|tile| tile % 8 * 8 + tile / 8)
            .collect::<Vec<_>>();
        let mapped = recipe.remapped(&graph, &mapping, 64).unwrap();
        raw.apply_ownership(&mapped.owners).unwrap();
        raw.apply_packing(&mapped.packing).unwrap();
        bound.remap_tiles(&mapping).unwrap();
        assert_eq!(raw.values, bound.values);
        assert_eq!(raw.operations, bound.operations);
        let mut encoded = serde_json::to_value(recipe).unwrap();
        encoded["packing"][0][1]["rows"] = 0.into();
        assert!(serde_json::from_value::<crate::planner::Recipe>(encoded).is_err());
    }

    #[test]
    fn distributed_panels_retile_without_unpacking_the_packed_intermediate() {
        for view in [false, true] {
            let (_, program) = packing_copy(view);
            let candidates = [32, 64, 128, 256]
                .into_iter()
                .filter_map(|rows| {
                    let choices = program
                        .legacy_packing_choices(NonZeroU16::new(rows).unwrap())
                        .unwrap();
                    if choices.is_empty() {
                        return None;
                    }
                    let mut packed = program.clone();
                    packed.apply_packing(&choices).unwrap();
                    packed.refresh_estimates().unwrap();
                    Some(packed)
                })
                .collect::<Vec<_>>();
            assert!(!candidates.is_empty() && candidates.len() <= 4);
            for packed in candidates {
                let low = crate::lower_to_tiles(
                    &crate::low::expand::expand_tiles(&packed, false).unwrap(),
                    false,
                );
                let mut packs = 0;
                for run in &low.kernel_runs {
                    if let TileKernelSpec::Rearrange { from, to } = &run.kernel {
                        assert_eq!(from.order, ElementOrder::RowMajor);
                        assert!(matches!(
                            to.order,
                            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                                row_block: 32..=256,
                                ..
                            })
                        ));
                        run.call().unwrap();
                        packs += 1;
                    }
                }
                assert!(packs > 10 && packs <= 64);
            }
        }
    }
}
