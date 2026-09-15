//! Construct communication, staging values and local arithmetic for reductions.
use super::fragments::FragmentBuilder;
use crate::{
    MidOperationKind, MidValueId, OperandIndexing, OperandWindow, Precision, ShardExtent,
    TensorAxis, TensorType,
};
use serde::{Deserialize, Serialize};

#[derive(
    Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum ReductionStaging {
    #[default]
    Complete,
    Streamed,
    Batched(std::num::NonZeroU16),
}

impl FragmentBuilder {
    pub(crate) fn sum(
        &mut self,
        input: MidValueId,
        output: &TensorType,
        axis: usize,
        staging: ReductionStaging,
    ) -> Option<MidValueId> {
        let source = self.tensor(input).clone();
        let partials = *source.shape.0.get(axis)?;
        if axis + 2 >= source.shape.0.len()
            || partials == 0
            || source.format.precision != Precision::F16
            || output.format.precision != Precision::F16
            || source
                .shape
                .0
                .iter()
                .enumerate()
                .filter(|(a, _)| *a != axis)
                .map(|(_, d)| d)
                .ne(output.shape.0.iter())
        {
            return None;
        }
        let mut receive = source.clone();
        receive.format.layout = output.format.layout.clone();
        for dim in &mut receive.format.layout.tiling.axes {
            let d = dim.axis.resolve(output.shape.0.len()).ok()?;
            dim.axis = TensorAxis::FromStart((d + usize::from(d >= axis)) as u16);
        }
        let mut result = None;
        let per_stage = match staging {
            ReductionStaging::Complete => partials.saturating_sub(1).max(1),
            ReductionStaging::Streamed => 1,
            ReductionStaging::Batched(n) => u32::from(n.get()),
        };
        receive.shape.0[axis] = 1;
        // Keep the accumulator's singleton contributor axis until the final
        // write. Every stage then uses the same input windows and physical order.
        let accumulator = receive.clone();
        let extents = receive
            .format
            .layout
            .resolve(&receive.shape)
            .ok()?
            .axes()?
            .iter()
            .enumerate()
            .map(|(axis, a)| ShardExtent {
                axis: axis as u16,
                start: 0,
                logical_end: a.maximum_extent(),
                physical_end: a.maximum_extent(),
            })
            .collect::<Vec<_>>();
        let regions = crate::storage::contiguous_axis_blocks(
            crate::storage::TensorStorage {
                format: &receive.format,
                extents: &extents,
            },
            axis,
        )
        .ok()?
        .into_iter()
        .map(|extents| {
            let input_window = OperandWindow(
                extents
                    .iter()
                    .enumerate()
                    .filter(|(a, _)| *a != axis)
                    .map(|(a, e)| (a as u16, e.start, e.physical_end))
                    .collect(),
            );
            let output_window = OperandWindow(
                input_window
                    .0
                    .iter()
                    .map(|&(a, start, end)| (a - u16::from(usize::from(a) > axis), start, end))
                    .collect(),
            );
            (input_window, output_window)
        })
        .collect::<Vec<_>>();
        let seed = self.materialize(
            input,
            receive.clone(),
            vec![],
            self.can_borrow_dense(input, &receive),
        );
        if partials == 1 {
            return Some(self.kernel(
                vec![seed],
                output.clone(),
                MidOperationKind::ReductionSum { partials: 1 },
                None,
                vec![OperandIndexing::local()],
            ));
        }
        // Finish all regions of a stage before gathering the next contributors;
        // otherwise streamed reduction would keep every receive buffer live.
        for start in (1..partials).step_by(per_stage as usize) {
            let count = per_stage.min(partials - start);
            receive.shape.0[axis] = count;
            let mut offsets = vec![0; source.shape.0.len()];
            offsets[axis] = start;
            let remote = self.materialize(input, receive.clone(), offsets, false);
            let previous = result.unwrap_or(seed);
            let final_stage = start + count == partials;
            let destination = if final_stage { output } else { &accumulator };
            let mut stage_result = None;
            for (input_window, output_window) in &regions {
                let destination_window = if final_stage {
                    output_window
                } else {
                    input_window
                };
                stage_result = Some(
                    self.compute(
                        vec![previous, remote],
                        [(
                            destination.clone(),
                            stage_result,
                            destination_window.clone(),
                        )],
                        MidOperationKind::ReductionSum {
                            partials: u16::try_from(count + 1).ok()?,
                        },
                        vec![
                            OperandIndexing::Fragment(input_window.clone()),
                            OperandIndexing::Fragment(input_window.clone()),
                        ],
                    )[0],
                );
            }
            result = stage_result;
        }
        result
    }
}
