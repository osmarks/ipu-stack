//! Physical access geometry for the shifted FP16-to-F143 cast. Its output
//! prefix and chunk boundaries keep stores out of the input's memory elements.
//! Whether the extra prefix saves storage belongs to the mid donation rewrite.

use crate::mid::{AmpOrder, ElementOrder};

/// Keep the output in earlier memory elements than each input chunk.
pub(crate) const CAST_PREFIX_BYTES: u32 = 32768;

pub(crate) struct CastChunks {
    pub axis: usize,
    pub ranges: Vec<(u32, u32)>,
}

impl CastChunks {
    pub fn new(order: ElementOrder, dimensions: &[u32]) -> Option<Self> {
        let (axis, grain, atom) = match (order, dimensions) {
            (ElementOrder::Amp(AmpOrder::Left), [rows, columns]) if columns.is_multiple_of(32) => {
                (1, 32, rows.checked_mul(64)?)
            }
            (ElementOrder::RowMajor, [elements]) if elements.is_multiple_of(8) => (0, 8, 16),
            (ElementOrder::RowMajor, [_, columns]) if columns.is_multiple_of(8) => {
                (0, 1, columns.checked_mul(2)?)
            }
            _ => return None,
        };
        if atom == 0 {
            return None;
        }
        let atoms = dimensions[axis] / grain;
        atoms.checked_mul(atom)?;
        if atoms == 0 || atom > CAST_PREFIX_BYTES {
            return None;
        }
        let mut ranges = Vec::new();
        let mut consumed = 0u32;
        while consumed < atoms {
            let input_start = CAST_PREFIX_BYTES.checked_add(consumed.checked_mul(atom)?)?;
            let output_start = consumed.checked_mul(atom)? / 2;
            // End output before the input's first 32 KiB group. Reads may
            // span several groups; none is touched by this chunk's stores.
            let output_limit = input_start / CAST_PREFIX_BYTES * CAST_PREFIX_BYTES;
            let count =
                ((output_limit - output_start).checked_mul(2)? / atom).min(atoms - consumed);
            if count == 0 {
                return None;
            }
            ranges.push((consumed * grain, (consumed + count) * grain));
            consumed += count;
        }
        Some(Self { axis, ranges })
    }
}
