//! Tile expansion and per-tile work, followed by placement and code generation.

mod copy;
pub(crate) mod expand;
mod graph;
mod passes;
pub(crate) mod storage;
pub use copy::*;
pub use expand::{ExpansionError, ExpansionResult};
pub use graph::*;
#[cfg(test)]
pub use storage::view_byte_spans;
#[cfg(test)]
pub(crate) use storage::view_byte_traversal;
pub use storage::{logical_view_byte_spans, shard_storage_bytes};

use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatRun {
    pub provenance: WorkProvenance,
    pub count: u32,
    pub binding: BlockRepeatBinding,
    pub body: TileWorkList,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileWorkList {
    pub tile: u16,
    /// Same operations as the device-wide graph; Repeat holds an index into
    /// LowProgram::repeat_runs. Exchanges remain present on idle tiles for sync.
    pub work: Vec<BlockOperation<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowProgram {
    pub program: Arc<TileGraph>,
    pub tiles: Vec<TileWorkList>,
    pub repeat_runs: Vec<RepeatRun>,
}

impl std::ops::Deref for LowProgram {
    type Target = TileGraph;
    fn deref(&self) -> &Self::Target {
        &self.program
    }
}

/// Derive per-tile indexes from the executable graph without changing its work.
/// Run low transformations before projection so costing and emission agree.
pub fn lower_to_tiles(program: &Arc<TileGraph>, diagnostic_checkpoints: bool) -> LowProgram {
    fn project(
        region: &BlockRegion,
        program: &TileGraph,
        repeats: &mut Vec<RepeatRun>,
        checkpoints: bool,
    ) -> Vec<TileWorkList> {
        let mut tiles = (0..program.tile_count)
            .map(|tile| TileWorkList {
                tile,
                work: Vec::new(),
            })
            .collect::<Vec<_>>();
        for operation in &region.operations {
            match operation {
                BlockOperation::Exchange(id) => {
                    for tile in &mut tiles {
                        tile.work.push(BlockOperation::Exchange(*id));
                    }
                }
                BlockOperation::Copy { tile, copy } => {
                    tiles[usize::from(*tile)].work.push(BlockOperation::Copy {
                        tile: *tile,
                        copy: *copy,
                    })
                }
                BlockOperation::Compute { tile, run } => {
                    tiles[usize::from(*tile)]
                        .work
                        .push(BlockOperation::Compute {
                            tile: *tile,
                            run: *run,
                        })
                }
                BlockOperation::Checkpoint(operation, breakpoint) if checkpoints => {
                    for tile in &mut tiles {
                        tile.work
                            .push(BlockOperation::Checkpoint(*operation, *breakpoint));
                    }
                }
                BlockOperation::Checkpoint(..) => {}
                BlockOperation::Repeat(repeat) => {
                    let body = project(&repeat.body, program, repeats, false);
                    for binding in &repeat.bindings {
                        let id = repeats.len();
                        repeats.push(RepeatRun {
                            provenance: repeat.provenance,
                            count: repeat.count,
                            binding: binding.clone(),
                            body: body[usize::from(binding.tile)].clone(),
                        });
                        tiles[usize::from(binding.tile)]
                            .work
                            .push(BlockOperation::Repeat(id));
                    }
                }
            }
        }
        tiles
    }
    let mut repeat_runs = Vec::new();
    let tiles = project(
        &program.body,
        program,
        &mut repeat_runs,
        diagnostic_checkpoints,
    );
    LowProgram {
        program: Arc::clone(program),
        tiles,
        repeat_runs,
    }
}
