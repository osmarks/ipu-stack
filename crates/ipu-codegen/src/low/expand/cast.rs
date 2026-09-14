//! Consume shrinking casts in bank-separated chunks over shifted shared storage.
use super::*;

impl TileGraphBuilder {
    pub(super) fn append_in_place_cast(
        &mut self,
        body: &mut BlockRegion,
        tile: u16,
        run: KernelRun,
    ) -> ExpansionResult<()> {
        let dimensions = run.inputs[0]
            .extents
            .iter()
            .map(|e| e.physical_end - e.start)
            .collect::<Vec<_>>();
        let chunks = crate::kernel::cast::CastChunks::new(
            run.requirements.outputs[0].format.layout.order,
            &dimensions,
        )
        .ok_or(ExpansionError::InvalidOperatorPlan)?;
        for (start, end) in chunks.ranges {
            let mut part = run.clone();
            for view in [&mut part.inputs[0], &mut part.outputs[0]] {
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
            self.append_kernel(body, tile, part)?;
        }
        Ok(())
    }
}
