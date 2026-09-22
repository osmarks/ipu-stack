//! A catalogue of implementations, independent of search histories. Inputs name
//! required operand representations; outputs name produced representations.
//! Search supplies boundary conversions and accounts for unrelated live values.

use super::{BoundaryLayouts, PlanningError, PlanningResult, elementwise};
use crate::config::PipelineConfig;
use crate::graph::{GraphInputKind, HighGraph, OperationKind, ValueId};
use crate::mid::{MidGraph, MidInput, MidOperationKind, MidValueId};
use crate::tensor::{OwnerMap, TensorType};
use std::collections::{BTreeMap, BTreeSet};

/// Storage boundaries favour dense elementwise access, either keeping complete
/// rows together or distributing aligned contiguous intervals across the device.
/// GEMM compute grids are deliberately absent: they are private implementations.
fn default_layouts(tensor: &TensorType, tiles: u16) -> BTreeSet<crate::Layout> {
    let elements = tensor.shape.elements();
    let grain = (8 / tensor.format.precision.bytes()) as u32;
    // Packed-to-row-major copies cannot yet split rows with sub-word tails.
    // Preserve complete matrices for these shapes rather than propose a default
    // which the physical copy implementation cannot realize.
    if tensor.shape.0.len() > 1
        && !(u64::from(*tensor.shape.0.last().unwrap()) * tensor.format.precision.bytes())
            .is_multiple_of(4)
    {
        return BTreeSet::from([crate::Layout::row_sharded(1)]);
    }
    let mut grains = BTreeSet::from([grain]);
    if let Some(&columns) = tensor.shape.0.last()
        && tensor.shape.0.len() > 1
        && columns.is_multiple_of(grain)
    {
        grains.insert(columns);
    }
    grains
        .into_iter()
        .map(|grain| {
            let blocks = elements.div_ceil(u64::from(grain));
            let width = blocks.div_ceil(u64::from(tiles));
            let owners = blocks.div_ceil(width.max(1)).max(1) as u16;
            if tensor.shape.0.len() > 1
                && tensor.shape.0[..tensor.shape.0.len() - 2]
                    .iter()
                    .all(|&n| n == 1)
                && grain == *tensor.shape.0.last().unwrap()
            {
                return crate::Layout::row_sharded(owners);
            }
            let mut layout = crate::Layout::logical_linear(owners, grain);
            layout.tiling.axes[0].padding = crate::Padding::Zero;
            layout
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_compact_complete_and_bounded_for_batched_shapes() {
        let mut rng = fastrand::Rng::with_seed(0xb0_0d);
        for _ in 0..128 {
            let shape = vec![rng.u32(1..=8), rng.u32(1..=1024), 8 * rng.u32(1..=256)];
            let tensor =
                TensorType::new(shape, crate::Precision::F16, crate::Layout::row_sharded(1));
            let tiles = rng.u16(1..=1472);
            let layouts = default_layouts(&tensor, tiles);
            assert!((1..=2).contains(&layouts.len()));
            for layout in layouts {
                assert_eq!(layout.tiling.replicas, 1);
                assert!(layout.tiling.tile_count <= tiles);
                let resolved = layout.resolve(&tensor.shape).unwrap();
                assert!(!resolved.has_empty_shards());
                assert_eq!(
                    (0..layout.tiling.tile_count)
                        .map(|tile| resolved.tile_elements(tile))
                        .sum::<u64>(),
                    tensor.shape.elements()
                );
            }
        }
    }
}

/// No allocation IDs from a particular history belong in a boundary key.
/// Cross-boundary aliases are deliberately unsupported until their backing
/// identity and access requirements can be included in this key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct BoundaryValue {
    pub tensor: TensorType,
    pub owners: OwnerMap,
}

pub(super) type LiveValues = BTreeMap<ValueId, BoundaryValue>;

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    /// First high operation not implemented by this fragment.
    pub end: usize,
    pub graph: MidGraph,
    /// High boundary identity -> fragment-local value. Private values have no entry.
    pub bindings: BTreeMap<ValueId, MidValueId>,
}

impl Candidate {
    /// Import exactly the supplied representations. Generators supply operands;
    /// search supplies its live boundary when connecting an implementation.
    pub fn inputs(high: &HighGraph, live: &LiveValues, tile_count: u16, end: usize) -> Self {
        let mut candidate = Self {
            end,
            graph: MidGraph {
                tile_count,
                ..MidGraph::default()
            },
            bindings: BTreeMap::new(),
        };
        for (&origin, boundary) in live {
            let id = super::construction::value(
                &mut candidate.graph,
                origin,
                boundary.tensor.clone(),
                boundary.owners.clone(),
            );
            let input = high.inputs().iter().find(|input| input.value == origin);
            candidate.graph.inputs.push(MidInput {
                name: input.map_or_else(|| format!("value.{}", origin.index()), |i| i.name.clone()),
                kind: input.map_or(GraphInputKind::Host, |i| i.kind),
                value: id,
            });
            candidate.bindings.insert(origin, id);
        }
        candidate
    }
}

pub(super) fn catalogue(
    high: &HighGraph,
    choices: &BoundaryLayouts,
    settings: &PipelineConfig,
) -> PlanningResult<Vec<Vec<Candidate>>> {
    let mut tensors = BTreeMap::new();
    let mut offers = BTreeMap::<ValueId, BTreeSet<crate::Layout>>::new();
    for input in high.inputs() {
        if input.kind == GraphInputKind::Host
            && !high.outputs().contains(&input.value)
            && !high
                .operations()
                .iter()
                .any(|op| high.operation_inputs(op).any(|id| id == input.value))
        {
            continue;
        }
        let format = super::parameters::initial_format(
            high,
            input,
            choices.get(&input.value).and_then(Option::as_ref),
            settings,
        )?;
        offers
            .entry(input.value)
            .or_default()
            .insert(format.layout.clone());
        tensors.insert(
            input.value,
            TensorType {
                shape: input.shape.clone(),
                format,
            },
        );
    }
    for op in high.operations() {
        let precision = if op.kind.is_elementwise() {
            tensors[&op.inputs[0]].format.precision
        } else if matches!(op.kind, OperationKind::Gemm(_)) {
            super::gemm::OUTPUT_PRECISION
        } else {
            return Err(PlanningError::Unimplemented("operator candidates"));
        };
        for &output in &op.results {
            let tensor = TensorType::new(
                high.value_shape(output).unwrap().0.clone(),
                precision,
                crate::Layout::row_sharded(1),
            );
            offers.insert(
                output,
                choices
                    .get(&output)
                    .and_then(Option::as_ref)
                    .map(|layout| BTreeSet::from([layout.clone()]))
                    .unwrap_or_else(|| default_layouts(&tensor, settings.tile_count)),
            );
            tensors.insert(output, tensor);
        }
    }
    // Propagate only through layout-preserving operations, before constructing
    // candidates. Internal GEMM packing must not expand this finite vocabulary.
    loop {
        let mut changed = false;
        for op in high.operations() {
            if !op.kind.is_elementwise() {
                continue;
            }
            let ports = op
                .inputs
                .iter()
                .chain(&op.results)
                .copied()
                .collect::<Vec<_>>();
            let layouts = ports
                .iter()
                .flat_map(|id| offers[id].iter().cloned())
                .collect::<BTreeSet<_>>();
            for id in ports {
                if choices.get(&id).and_then(Option::as_ref).is_none() {
                    let old = offers[&id].len();
                    offers.get_mut(&id).unwrap().extend(layouts.iter().cloned());
                    changed |= offers[&id].len() != old;
                }
            }
        }
        if !changed {
            break;
        }
    }
    high.operations()
        .iter()
        .enumerate()
        .map(|(position, op)| {
            let alternatives = match op.kind {
                OperationKind::Add => elementwise::generate(
                    high,
                    position,
                    &tensors,
                    choices,
                    settings,
                    MidOperationKind::Add,
                    &offers,
                )?,
                OperationKind::Gelu => elementwise::generate(
                    high,
                    position,
                    &tensors,
                    choices,
                    settings,
                    MidOperationKind::Gelu,
                    &offers,
                )?,
                OperationKind::Gemm(options) => {
                    super::gemm::generate(high, position, &tensors, settings, options, &offers)?
                }
                _ => return Err(PlanningError::Unimplemented("operator candidates")),
            };
            if alternatives.is_empty() {
                return Err(PlanningError::NoPlan(position + 1));
            }
            Ok(alternatives)
        })
        .collect()
}
