use ipu_compiler::BlockPlacement;
use ipu_package::{Binding, RegionSlice};

const INNER_MICRO_DIMENSION: u16 = 8;
const COLUMN_MICRO_DIMENSION: u16 = 16;

#[derive(Clone, Copy)]
pub enum BlockLayout {
    AmpA8,
    AmpA16,
    AmpA32,
    AmpB8x16,
    AmpB16x16,
    AmpB32x16,
    AmpC16,
    AmpC16F16,
}

pub fn block_binding(
    name: &str,
    rows: u16,
    columns: u16,
    placements: &[BlockPlacement],
) -> Binding {
    block_binding_typed(name, rows, columns, placements, "f32", 4)
}

pub fn block_binding_typed(
    name: &str,
    rows: u16,
    columns: u16,
    placements: &[BlockPlacement],
    dtype: &str,
    element_bytes: u64,
) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    Binding {
        name: name.into(),
        dtype: dtype.into(),
        shape: vec![u32::from(rows), u32::from(columns)],
        slices: placements
            .iter()
            .scan(0u64, |file_offset, placement| {
                let size = u64::from(placement.rows) * u64::from(placement.columns) * element_bytes;
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
        BlockLayout::AmpA8 | BlockLayout::AmpA16 | BlockLayout::AmpA32 => {
            let inner_micro_dimension = match layout {
                BlockLayout::AmpA8 => INNER_MICRO_DIMENSION,
                BlockLayout::AmpA16 => 16,
                BlockLayout::AmpA32 => 32,
                _ => unreachable!(),
            };
            let panel_elements = rows * inner_micro_dimension;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            (
                panel_offset / inner_micro_dimension,
                panel * inner_micro_dimension + panel_offset % inner_micro_dimension,
            )
        }
        BlockLayout::AmpB8x16 | BlockLayout::AmpB16x16 | BlockLayout::AmpB32x16 => {
            let inner_micro_dimension = match layout {
                BlockLayout::AmpB8x16 => INNER_MICRO_DIMENSION,
                BlockLayout::AmpB16x16 => 16,
                BlockLayout::AmpB32x16 => 32,
                _ => unreachable!(),
            };
            let panel_elements = inner_micro_dimension * COLUMN_MICRO_DIMENSION;
            let panel = linear / panel_elements;
            let panel_offset = linear % panel_elements;
            let inner_groups = rows / inner_micro_dimension;
            let column_group = panel / inner_groups;
            let inner_group = panel % inner_groups;
            let load_channel = panel_offset / inner_micro_dimension;
            let inner_in_group = panel_offset % inner_micro_dimension;
            let load_pair = load_channel / 2;
            let logical_pair = (load_pair % 2) * 4 + load_pair / 2;
            let column_in_group = logical_pair * 2 + load_channel % 2;
            (
                inner_group * inner_micro_dimension + inner_in_group,
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
        BlockLayout::AmpC16F16 => {
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

pub fn blocked_matrix_f16(
    placements: &[BlockPlacement],
    layout: BlockLayout,
    value: impl Fn(u16, u16) -> f32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) =
                block_coordinates(layout, placement.rows, placement.columns, linear);
            let bits = half::f16::from_f32(value(
                placement.row_start + row,
                placement.column_start + column,
            ))
            .to_bits();
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
    }
    bytes
}

pub fn f143_scale(values: impl IntoIterator<Item = f32>) -> i8 {
    let maximum = values
        .into_iter()
        .filter(|value| value.is_finite())
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    if maximum == 0.0 {
        return 0;
    }
    (maximum / 240.0).log2().ceil().clamp(-32.0, 31.0) as i8
}

pub fn f143_from_f32(value: f32, scale: i8) -> u8 {
    let scale_multiplier = f32::from_bits(((127 - i32::from(scale)) as u32) << 23);
    f143_from_scaled_f32(value * scale_multiplier)
}

fn f143_from_scaled_f32(value: f32) -> u8 {
    let sign = u8::from(value.is_sign_negative()) << 7;
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return 0x80;
    }
    if magnitude == 0.0 {
        return 0;
    }
    if !magnitude.is_finite() || magnitude >= 240.0 {
        return sign | 0x7f;
    }
    if magnitude < 1.0 / 128.0 {
        let mantissa = (magnitude * 1024.0).round_ties_even() as u8;
        let mantissa = mantissa.min(8);
        return if mantissa == 0 { 0 } else { sign | mantissa };
    }

    let exponent = i32::try_from((magnitude.to_bits() >> 23) & 0xff).unwrap() - 127;
    let mut encoded_exponent = exponent + 8;
    let unit = f32::from_bits(((exponent + 127) as u32) << 23);
    let mut mantissa = ((magnitude / unit - 1.0) * 8.0).round_ties_even() as i32;
    if mantissa == 8 {
        mantissa = 0;
        encoded_exponent += 1;
    }
    if encoded_exponent > 15 {
        return sign | 0x7f;
    }
    sign | ((encoded_exponent as u8) << 3) | mantissa as u8
}

pub fn f143_to_f32(bits: u8, scale: i8) -> f32 {
    if bits == 0x80 {
        return f32::NAN;
    }
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0xf;
    let mantissa = bits & 7;
    let value = if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-10)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 8)
    };
    sign * value * 2.0f32.powi(i32::from(scale))
}

pub fn blocked_matrix_f8_f143(
    placements: &[BlockPlacement],
    layout: BlockLayout,
    scale: i8,
    value: impl Fn(u16, u16) -> f32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    for placement in placements {
        for linear in 0..placement.rows * placement.columns {
            let (row, column) =
                block_coordinates(layout, placement.rows, placement.columns, linear);
            bytes.push(f143_from_f32(
                value(placement.row_start + row, placement.column_start + column),
                scale,
            ));
        }
    }
    bytes
}

pub fn f143_block_scales(
    placements: &[BlockPlacement],
    value: &impl Fn(u16, u16) -> f32,
) -> Vec<i8> {
    placements
        .iter()
        .map(|placement| {
            f143_scale((0..placement.rows * placement.columns).map(|linear| {
                let row = linear / placement.columns;
                let column = linear % placement.columns;
                value(placement.row_start + row, placement.column_start + column)
            }))
        })
        .collect()
}

pub fn blocked_matrix_f8_f143_by_block(
    placements: &[BlockPlacement],
    layout: BlockLayout,
    scales: &[i8],
    value: impl Fn(u16, u16) -> f32,
) -> Vec<u8> {
    assert_eq!(placements.len(), scales.len());
    let mut bytes = Vec::with_capacity(
        placements
            .iter()
            .map(|placement| usize::from(placement.rows) * usize::from(placement.columns))
            .sum(),
    );
    for (placement, &scale) in placements.iter().zip(scales) {
        let scale_multiplier = f32::from_bits(((127 - i32::from(scale)) as u32) << 23);
        for linear in 0..placement.rows * placement.columns {
            let (row, column) =
                block_coordinates(layout, placement.rows, placement.columns, linear);
            bytes.push(f143_from_scaled_f32(
                value(placement.row_start + row, placement.column_start + column)
                    * scale_multiplier,
            ));
        }
    }
    bytes
}

pub fn normal_f16(elements: usize, seed: u64, standard_deviation: f32) -> Vec<half::f16> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut values = Vec::with_capacity(elements);
    while values.len() < elements {
        let radius = (-2.0 * (1.0 - rng.f64()).ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.f64();
        values.push(half::f16::from_f32(
            (radius * angle.cos()) as f32 * standard_deviation,
        ));
        if values.len() < elements {
            values.push(half::f16::from_f32(
                (radius * angle.sin()) as f32 * standard_deviation,
            ));
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f143_encoding_matches_documented_rounding_examples() {
        let examples = [
            (7.0, 0x56),
            (7.25, 0x56),
            (7.5, 0x57),
            (7.75, 0x58),
            (8.0, 0x58),
            (8.5, 0x58),
            (9.0, 0x59),
            (9.5, 0x5a),
        ];
        for (value, bits) in examples {
            assert_eq!(f143_from_f32(value, 0), bits);
        }
    }

    #[test]
    fn finite_f143_values_round_trip_for_every_scale() {
        for scale in -32..=31 {
            for bits in 0u8..=u8::MAX {
                if bits == 0x80 {
                    continue;
                }
                assert_eq!(f143_from_f32(f143_to_f32(bits, scale), scale), bits);
            }
        }
    }

    #[test]
    fn negative_underflow_encodes_zero_instead_of_nanoo() {
        assert_eq!(f143_from_f32(-0.0001, 0), 0);
    }

    #[test]
    fn selected_scale_uses_the_available_normal_range() {
        for maximum in [0.001, 0.1, 1.0, 100.0, 10_000.0] {
            let scale = f143_scale([maximum, -maximum]);
            let scaled = maximum * 2.0f32.powi(-i32::from(scale));
            assert!(scaled <= 240.0);
            assert!(scale == -32 || scaled > 120.0);
        }
    }
}
