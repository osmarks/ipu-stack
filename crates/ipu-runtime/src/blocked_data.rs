use ipu_compiler::BlockPlacement;
use ipu_package::{Binding, RegionSlice};

const INNER_MICRO_DIMENSION: u16 = 8;
const COLUMN_MICRO_DIMENSION: u16 = 16;

#[derive(Clone, Copy)]
pub enum BlockLayout {
    AmpA8,
    AmpB8x16,
    AmpC16,
}

pub fn block_binding(
    name: &str,
    rows: u16,
    columns: u16,
    placements: &[BlockPlacement],
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    Binding {
        name: name.into(),
        dtype: "f32".into(),
        shape: vec![u32::from(rows), u32::from(columns)],
        slices: placements
            .iter()
            .scan(0u64, |file_offset, placement| {
                let size = u64::from(placement.rows) * u64::from(placement.columns) * 4;
                let slice = RegionSlice {
                    tile: u32::from(topology.physical(placement.tile).unwrap()),
                    tile_address: placement.address,
                    file_offset: *file_offset,
                    size,
                };
                *file_offset += size;
                Some(slice)
            })
            .collect(),
    }
}

pub fn block_coordinates(layout: BlockLayout, rows: u16, _columns: u16, linear: u16) -> (u16, u16) {
    match layout {
        BlockLayout::AmpA8 => {
            let panel_elements = rows * INNER_MICRO_DIMENSION;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            (
                panel_offset / INNER_MICRO_DIMENSION,
                panel * INNER_MICRO_DIMENSION + panel_offset % INNER_MICRO_DIMENSION,
            )
        }
        BlockLayout::AmpB8x16 => {
            let panel_elements = INNER_MICRO_DIMENSION * COLUMN_MICRO_DIMENSION;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let inner_groups = rows / INNER_MICRO_DIMENSION;
            let column_group = panel / inner_groups;
            let inner_group = panel % inner_groups;
            let load_channel = panel_offset / INNER_MICRO_DIMENSION;
            let inner_in_group = panel_offset % INNER_MICRO_DIMENSION;
            let load_pair = load_channel / 2;
            let logical_pair = (load_pair % 2) * 4 + load_pair / 2;
            let column_in_group = logical_pair * 2 + load_channel % 2;
            (
                inner_group * INNER_MICRO_DIMENSION + inner_in_group,
                column_group * COLUMN_MICRO_DIMENSION + column_in_group,
            )
        }
        BlockLayout::AmpC16 => {
            let panel_elements = rows * COLUMN_MICRO_DIMENSION;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let physical_column = panel_offset % COLUMN_MICRO_DIMENSION;
            let physical_pair = physical_column / 2;
            let logical_pair = (physical_pair % 2) * 4 + physical_pair / 2;
            (
                panel_offset / COLUMN_MICRO_DIMENSION,
                panel * COLUMN_MICRO_DIMENSION + logical_pair * 2 + physical_column % 2,
            )
        }
    }
}

pub fn blocked_matrix(
    placements: &[BlockPlacement],
    layout: BlockLayout,
    value: impl Fn(u16, u16) -> f32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) =
                block_coordinates(layout, placement.rows, placement.columns, linear);
            bytes.extend_from_slice(
                &value(placement.row_start + row, placement.column_start + column).to_le_bytes(),
            );
        }
    }
    bytes
}
