//! Bind copy geometry to low storage identities, then select local launches.
use super::*;

pub(super) fn append_span_copies(
    cache: &crate::storage::GeometryCache,
    shards: &[BlockValue],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
    order: CopyOrder,
) -> ExpansionResult<()> {
    let a = source.bind(shards)?;
    let b = destination.bind(shards)?;
    let source_geometry = a.geometry(cache, order)?;
    let target_geometry = b.geometry(cache, order)?;
    let pair = cache.pair(&source_geometry, &target_geometry)?;
    copies.extend(
        LocalCopy::from_pair(
            source.shard,
            destination.shard,
            a.backing.0 == b.backing.0,
            &pair,
        )?
        .into_iter()
        .map(|copy| (tile, copy)),
    );
    Ok(())
}
