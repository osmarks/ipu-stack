//! Build explicit mid values and copies without high-graph bindings or search state.

use crate::graph::ValueId;
use crate::mid::MidOperationKind;
use crate::{MidGraph, MidValue, MidValueId, OwnerMap, TensorType};

/// Materialize a selected representation or slice; identical resident
/// operands need no copy. This does not change the high-value binding.
pub(super) fn copy(
    graph: &mut MidGraph,
    source: crate::OperationId,
    mut input: MidValueId,
    tensor: TensorType,
    offsets: Vec<u32>,
) -> MidValueId {
    // Native panels must be assembled before distribution when logical cuts
    // would transfer sub-word pieces. This is a boundary conversion property,
    // including conversions requested by elementwise consumers of GEMMs.
    let from = &graph.values[input.index() as usize].tensor_type;
    if tensor.format.layout.tiling.tile_count > 1
        && offsets.iter().all(|&n| n == 0)
        && from.shape == tensor.shape
        && from.format.layout.order != tensor.format.layout.order
        && tensor.format.layout.order.micro_panel_order().is_some()
        && (tensor.format.layout.order == crate::ElementOrder::Amp(crate::AmpOrder::TransposedLeft)
            || !(tensor.shape.0.last().unwrap() * tensor.format.precision.bytes() as u32)
                .is_multiple_of(4))
    {
        let mut packed = tensor.clone();
        packed.format.layout.tiling.tile_count = 1;
        packed.format.layout.tiling.replicas = 1;
        for axis in &mut packed.format.layout.tiling.axes {
            axis.partitions = 1;
            axis.tile_stride = Some(1);
        }
        input = copy(graph, source, input, packed, Vec::new());
    }
    let from = &graph.values[input.index() as usize];
    let owners = OwnerMap::default();
    if from.tensor_type == tensor && from.owners == owners && offsets.iter().all(|&n| n == 0) {
        return input;
    }
    let policy = crate::default_copy_policy(&from.tensor_type.format.layout, &tensor.format.layout);
    let packing = if from
        .tensor_type
        .format
        .layout
        .order
        .micro_panel_order()
        .is_some()
        && from.tensor_type.format.layout.order.micro_panel_order()
            == tensor.format.layout.order.micro_panel_order()
    {
        crate::PackingPolicy::Direct
    } else {
        crate::PackingPolicy::Staged
    };
    let output = value(graph, from.origin, tensor, owners);
    graph.operations.push(crate::MidOperation {
        source: Some(source),
        inputs: vec![input],
        results: vec![output],
        kind: MidOperationKind::Copy {
            mapping: crate::CoordinateMapping {
                offsets,
                view: None,
            },
            policy,
            packing,
        },
        operands: Vec::new(),
        output_aliases: Vec::new(),
        output_windows: Vec::new(),
    });
    output
}

pub(super) fn value(
    graph: &mut MidGraph,
    origin: ValueId,
    tensor_type: TensorType,
    owners: OwnerMap,
) -> MidValueId {
    let id = MidValueId::from_index(graph.values.len().try_into().expect("mid value count"));
    graph.values.push(MidValue {
        id,
        origin,
        tensor_type,
        owners,
        storage_group: id,
    });
    id
}
