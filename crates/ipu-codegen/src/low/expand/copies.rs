//! Bind relative copy descriptors to the selected low storage identities.

use super::*;

pub(super) fn append_span_copies(
    cache: &ExpansionCache,
    shards: &[BlockValue],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
    order: CopyOrder,
) -> ExpansionResult<()> {
    copies.extend(
        cache
            .copy(shards, source, destination, order)?
            .iter()
            .map(|c| {
                (
                    tile,
                    LocalCopy {
                        source: source.shard,
                        destination: destination.shard,
                        source_offset: c.source_offset,
                        destination_offset: c.destination_offset,
                        bytes: c.bytes,
                        pattern: c.pattern,
                    },
                )
            }),
    );
    Ok(())
}
