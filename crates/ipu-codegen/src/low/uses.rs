//! Storage-use facts for transformations of the whole-device graph.
//! Positions are one-based body.walk() ordinals, not placement's per-tile events.
//! Recompute after a rewrite: roots and uses describe one immutable graph.

use super::storage::storage_root;
use super::*;

#[derive(Clone, Default)]
pub(crate) struct StorageUse {
    /// Enclosing structural lifetime: inputs start at zero; parameters and
    /// outputs end at usize::MAX. These endpoints identify external storage.
    pub first: Option<usize>,
    pub last: usize,
    pub last_write: Option<usize>,
    /// Writer operands in one structural traversal, not the dynamic iteration count.
    pub writes: usize,
    /// Shards sharing this root, including the root itself.
    pub aliases: usize,
    /// Bound by Repeat, or used both before and inside a loop.
    pub boundary: bool,
    /// Includes possible writes through invariant/carried/iterated bindings.
    pub read_only: bool,
    /// Copies, exchanges, checkpoints, outputs and opaque Repeat binding reads.
    pub non_kernel_read: bool,
}

pub(crate) struct StorageUses {
    pub roots: Vec<usize>,
    /// Only root entries contain facts; aliases index them through roots.
    pub allocations: Vec<StorageUse>,
}

impl StorageUses {
    pub(crate) fn analyze(program: &TileGraph) -> Self {
        let mut uses = Self {
            roots: program
                .shards
                .iter()
                .map(|s| storage_root(&program.shards, s.id).index() as usize)
                .collect(),
            allocations: vec![StorageUse::default(); program.shards.len()],
        };
        for &root in &uses.roots {
            uses.allocations[root].aliases += 1;
        }
        for input in &program.inputs {
            for view in program.value_views(input.value) {
                uses.touch(view.shard, 0, false, false);
                let allocation = &mut uses.allocations[uses.roots[view.shard.index() as usize]];
                if input.kind == crate::GraphInputKind::Parameter {
                    allocation.last = usize::MAX;
                }
            }
        }
        let mut bindings = Vec::new();
        uses.visit(program, &program.body, &mut 1, &mut bindings);
        for &output in &program.outputs {
            for view in program.value_views(output) {
                uses.touch(view.shard, usize::MAX, false, true);
            }
        }
        for allocation in &mut uses.allocations {
            allocation.read_only = allocation.writes == 0;
        }
        // A write through an argument can affect its input, and vice versa.
        // Nested bindings and iterated inputs require a fixed point.
        loop {
            let mut changed = false;
            for group in &bindings {
                if group.iter().any(|&root| !uses.allocations[root].read_only) {
                    for &root in group {
                        changed |= uses.allocations[root].read_only;
                        uses.allocations[root].read_only = false;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        uses
    }

    fn touch(&mut self, id: BlockValueId, position: usize, write: bool, non_kernel: bool) {
        let use_ = &mut self.allocations[self.roots[id.index() as usize]];
        use_.first.get_or_insert(position);
        use_.last = use_.last.max(position);
        if write {
            use_.writes += 1;
            use_.last_write = Some(position);
        } else {
            use_.non_kernel_read |= non_kernel;
        }
    }

    fn visit(
        &mut self,
        program: &TileGraph,
        region: &BlockRegion,
        next: &mut usize,
        bindings: &mut Vec<Vec<usize>>,
    ) {
        for operation in &region.operations {
            let position = *next;
            *next += 1;
            match operation {
                BlockOperation::Compute { run, .. } => {
                    let run = &program.kernel_runs[run.0 as usize];
                    for view in &run.inputs {
                        self.touch(view.shard, position, false, false);
                    }
                    for view in &run.outputs {
                        self.touch(view.shard, position, true, false);
                    }
                }
                BlockOperation::Copy { copy, .. } => {
                    let copy = program.local_copies[copy.0 as usize].movement();
                    self.touch(copy.source, position, false, true);
                    self.touch(copy.destination, position, true, false);
                }
                BlockOperation::Exchange(id) => {
                    for transfer in &program.exchange_phases[id.index() as usize].transfers {
                        self.touch(transfer.source.shard, position, false, true);
                        for view in &transfer.destinations {
                            self.touch(view.shard, position, true, false);
                        }
                    }
                }
                BlockOperation::Checkpoint(id, _) => {
                    for (_, values) in program
                        .checkpoints
                        .iter()
                        .filter(|(source, _)| source == id)
                    {
                        for &value in values {
                            for view in program.value_views(value) {
                                self.touch(view.shard, position, false, true);
                            }
                        }
                    }
                }
                BlockOperation::Repeat(repeat) => {
                    let first_binding = bindings.len();
                    for binding in &repeat.bindings {
                        for carried in &binding.carried {
                            bindings.push(
                                [
                                    carried.initial,
                                    carried.argument,
                                    carried.yielded,
                                    carried.result,
                                ]
                                .into_iter()
                                .map(|id| self.roots[id.index() as usize])
                                .collect(),
                            );
                        }
                        for invariant in &binding.invariants {
                            bindings.push(
                                [invariant.input, invariant.argument]
                                    .into_iter()
                                    .map(|id| self.roots[id.index() as usize])
                                    .collect(),
                            );
                        }
                        for iterated in &binding.iterated {
                            bindings.push(
                                std::iter::once(iterated.argument)
                                    .chain(iterated.inputs.iter().copied())
                                    .map(|id| self.roots[id.index() as usize])
                                    .collect(),
                            );
                        }
                    }
                    self.visit(program, &repeat.body, next, bindings);
                    let end = *next - 1;
                    // Conservatively protect storage already used outside this
                    // loop. Its last syntactic use is not necessarily its last
                    // runtime use. Fresh body-local scratch remains reusable.
                    for use_ in &mut self.allocations {
                        if use_.first.is_some_and(|first| first < position)
                            && (position..=end).contains(&use_.last)
                        {
                            use_.boundary = true;
                            use_.last = end;
                        }
                    }
                    for group in &bindings[first_binding..] {
                        for &root in group {
                            let use_ = &mut self.allocations[root];
                            use_.first =
                                Some(use_.first.map_or(position, |first| first.min(position)));
                            use_.last = use_.last.max(end);
                            use_.boundary = true;
                            use_.non_kernel_read = true;
                        }
                    }
                }
            }
        }
    }
}
