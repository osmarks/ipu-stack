//! Consume shrinking casts in bank-separated chunks over shifted shared storage.
use super::*;
use crate::mid::MidOperationKind;

impl TileGraphBuilder {
    pub(super) fn build_shifted_cast(
        &mut self,
        body: &mut BlockRegion,
        tile: u16,
        provenance: WorkProvenance,
        kernel: MidOperationKind,
        input: ShardView,
        output: ShardView,
    ) -> ExpansionResult<()> {
        let dimensions = input
            .extents
            .iter()
            .map(|e| e.physical_end - e.start)
            .collect::<Vec<_>>();
        let chunks = crate::kernel::cast::CastChunks::new(
            self.shards[output.shard.index() as usize]
                .tensor_type
                .format
                .layout
                .order,
            &dimensions,
        )
        .ok_or(ExpansionError::InvalidOperatorPlan)?;
        for (start, end) in chunks.ranges {
            let mut input = input.clone();
            let mut output = output.clone();
            for view in [&mut input, &mut output] {
                let extent = &mut view.extents[chunks.axis];
                let base = extent.start;
                extent.start = base + start;
                extent.physical_end = base + end;
                extent.logical_end = extent
                    .logical_end
                    .min(extent.physical_end)
                    .max(extent.start);
            }
            // Output is 32 KiB before input; chunk geometry guarantees
            // disjoint memory elements and no writes into unread input.
            let run = self.bind_kernel(provenance, kernel.clone(), vec![input], vec![output])?;
            self.append_kernel(body, tile, run)?;
        }
        Ok(())
    }
}
