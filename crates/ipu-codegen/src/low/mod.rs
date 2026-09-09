//! Tile expansion and per-tile work, followed by placement and code generation.

mod call;
mod copy;
pub(crate) mod expand;
mod graph;
mod initialization;
mod passes;
pub(crate) use call::*;
pub use copy::*;
#[cfg(test)]
pub use expand::view_byte_spans;
pub(crate) use expand::view_byte_traversal;
pub use expand::{ExpansionError, ExpansionResult, logical_view_byte_spans, shard_storage_bytes};
pub use graph::*;

use crate::graph::OperationId;
use crate::mid::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepeatRunId(u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatRun {
    pub provenance: WorkProvenance,
    pub count: u32,
    pub carried: Vec<RepeatCarried>,
    pub invariants: Vec<RepeatInvariant>,
    pub iterated: Vec<RepeatIterated>,
    pub body: Box<TileWorkList>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileWork {
    /// All tiles encounter a phase marker, including tiles without transfers.
    Exchange(ExchangePhaseId),
    LocalCopy(LocalCopyId),
    Kernel(KernelRunId),
    Repeat(RepeatRunId),
    Checkpoint(OperationId, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileWorkRef<'a> {
    Exchange(ExchangePhaseId),
    LocalCopy(&'a LocalCopy),
    Kernel(&'a KernelRun),
    Repeat(&'a RepeatRun),
    Checkpoint(OperationId, u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileWorkList {
    pub tile: u16,
    pub work: Vec<TileWork>,
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

impl LowProgram {
    /// Resolves compact schedule entries as they are consumed, without
    /// constructing a second per-tile work list.
    pub fn work<'a>(
        &'a self,
        tile: &'a TileWorkList,
    ) -> impl Iterator<Item = TileWorkRef<'a>> + 'a {
        tile.work.iter().map(|work| match *work {
            TileWork::Exchange(id) => TileWorkRef::Exchange(id),
            TileWork::LocalCopy(id) => TileWorkRef::LocalCopy(&self.local_copies[id.0 as usize]),
            TileWork::Kernel(id) => TileWorkRef::Kernel(&self.kernel_runs[id.0 as usize]),
            TileWork::Repeat(id) => TileWorkRef::Repeat(&self.repeat_runs[id.0 as usize]),
            TileWork::Checkpoint(operation, breakpoint) => {
                TileWorkRef::Checkpoint(operation, breakpoint)
            }
        })
    }
}

/// Project tile work and remove redundant finite-only initialization.
/// Numerical inputs/results must be finite; loading must first initialize SRAM.
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
                        tile.work.push(TileWork::Exchange(*id));
                    }
                }
                BlockOperation::Copy { tile, copy } => tiles[usize::from(*tile)]
                    .work
                    .push(TileWork::LocalCopy(*copy)),
                BlockOperation::Compute { tile, run } => {
                    tiles[usize::from(*tile)].work.push(TileWork::Kernel(*run))
                }
                BlockOperation::Checkpoint(operation, breakpoint) if checkpoints => {
                    for tile in &mut tiles {
                        tile.work
                            .push(TileWork::Checkpoint(*operation, *breakpoint));
                    }
                }
                BlockOperation::Checkpoint(..) => {}
                BlockOperation::Repeat(repeat) => {
                    let body = project(&repeat.body, program, repeats, false);
                    for binding in &repeat.bindings {
                        let id = RepeatRunId(
                            u32::try_from(repeats.len())
                                .expect("too many projected repeat instances"),
                        );
                        repeats.push(RepeatRun {
                            provenance: repeat.provenance,
                            count: repeat.count,
                            carried: binding.carried.clone(),
                            invariants: binding.invariants.clone(),
                            iterated: binding.iterated.clone(),
                            body: Box::new(body[usize::from(binding.tile)].clone()),
                        });
                        tiles[usize::from(binding.tile)]
                            .work
                            .push(TileWork::Repeat(id));
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
    let mut low = LowProgram {
        program: Arc::clone(program),
        tiles,
        repeat_runs,
    };
    initialization::reuse_finite_padding(&mut low);
    low
}
