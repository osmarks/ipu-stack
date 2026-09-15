//! Transform the executable tile graph before deriving per-tile indexes.

mod cast;
pub(super) mod movement;
mod padding;
mod relay;

use super::{ExpansionResult, TileGraph};
use crate::low::storage::storage_root;
use crate::storage::GeometryCache;

pub(super) fn run(
    program: &mut TileGraph,
    geometry: &GeometryCache,
    reuse_cast_inputs: bool,
) -> ExpansionResult<()> {
    movement::eliminate_copies(program)?;
    let grouped = movement::group_exchanges(
        &mut program.body,
        &mut program.exchange_phases,
        &program.local_copies,
        &program.shards,
    )?;
    if grouped != 0 {
        tracing::debug!(grouped, "consolidated exchange boundaries");
    }
    let roots = program
        .shards
        .iter()
        .map(|value| storage_root(&program.shards, value.id))
        .collect::<Vec<_>>();
    let old = std::mem::take(&mut program.local_copies);
    let merged = movement::merge_copies(
        &mut program.body,
        &old,
        &mut program.local_copies,
        &roots,
        &program.shards,
    )?;
    if merged != 0 {
        tracing::debug!(merged, "merged adjacent low copies");
    }
    drop(old);
    drop(roots);
    // Relay selection must see the merged exchanges. Padding analysis must
    // then see every reader and scratch allocation introduced by that choice.
    relay::select(program, geometry)?;
    if reuse_cast_inputs {
        cast::donate(program)?;
    }
    padding::omit_unread_fp8_input_padding(program);
    padding::reuse_finite_padding(program);
    Ok(())
}
