use super::program::MAX_TRANSFER_WORDS;
use super::{ExchangeItemWidth, ExchangeLoweringError};
use crate::{BlockValueId, LogicalExchange, LowGraph, Placement, ShardDefinition};
use ipu_target::ipu21::memory::effective_memory_elements;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BTreeMap;

pub(super) fn repeat_source_bases(
    program: &LowGraph,
    placement: &Placement,
) -> Result<BTreeMap<BlockValueId, Vec<u32>>, ExchangeLoweringError> {
    let mut repeat_inputs = BTreeMap::<BlockValueId, Vec<BlockValueId>>::new();
    for repeat in &program.repeat_runs {
        for iterated in &repeat.binding.iterated {
            match repeat_inputs.entry(iterated.argument) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(iterated.inputs.clone());
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() != &iterated.inputs =>
                {
                    return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                        "argument has inconsistent sequence bindings",
                    ));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    if repeat_inputs.is_empty() {
        return Ok(BTreeMap::new());
    }
    program
        .shards
        .iter()
        .filter_map(|shard| {
            crate::low::storage::storage_chain(&program.shards, shard.id)
                .find_map(|(source, offset)| {
                    repeat_inputs.get(&source).map(|inputs| (inputs, offset))
                })
                .map(|(inputs, offset)| {
                    let addresses = inputs
                        .iter()
                        .map(|input| {
                            let address = placement
                                .shard_addresses
                                .get(input)
                                .ok_or(ExchangeLoweringError::UnplacedShard)?;
                            u32::try_from(i64::from(*address) + offset)
                                .map_err(|_| ExchangeLoweringError::Overflow)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((shard.id, addresses))
                })
        })
        .collect()
}

pub(super) fn prepare_phase(
    program: &LowGraph,
    placement: &Placement,
    phase: &crate::low::ExchangePhase,
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let mut pending: Vec<PendingTransfer> = phase
        .transfers
        .par_iter()
        .enumerate()
        .map(|(index, transfer)| {
            prepare_transfer(program, placement, transfer).inspect_err(|error| {
                tracing::error!(
                    phase = phase.id.index(),
                    transfer = index,
                    provenance = ?phase.provenance,
                    source = ?transfer.source,
                    source_storage = ?program.shards[transfer.source.shard.index() as usize],
                    order = ?transfer.order,
                    destinations = ?transfer.destinations,
                    ?error,
                    "failed to prepare logical exchange transfer"
                );
            })
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?
        .into_iter()
        .flatten()
        .collect();
    attach_repeat_source_addresses(&mut pending, repeat_inputs)?;
    Ok(coalesce_pending_transfers(pending))
}
fn prepare_transfer(
    program: &LowGraph,
    placement: &Placement,
    transfer: &LogicalExchange,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    let source = &program.shards[transfer.source.shard.index() as usize];
    let order = transfer.span_order(&program.shards);
    let source_base = placement
        .shard_addresses
        .get(&source.id)
        .copied()
        .ok_or(ExchangeLoweringError::UnplacedShard)?;
    let destinations = transfer
        .destinations
        .iter()
        .map(|view| {
            let shard = &program.shards[view.shard.index() as usize];
            if matches!(shard.definition, ShardDefinition::Alias(_)) {
                return Err(ExchangeLoweringError::InvalidDestination);
            }
            Ok((
                shard.tile,
                placement
                    .shard_addresses
                    .get(&view.shard)
                    .copied()
                    .ok_or(ExchangeLoweringError::UnplacedShard)?,
                view.bind(&program.shards)?.traversal(order)?,
            ))
        })
        .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
    if destinations.is_empty() {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let source_spans = transfer.source.bind(&program.shards)?.traversal(order)?;
    if destinations
        .iter()
        .any(|(_, _, spans)| spans.byte_len() != source_spans.byte_len())
    {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    let mut receivers = destinations
        .iter()
        .map(|(_, _, spans)| spans.spans().peekable())
        .collect::<Vec<_>>();
    let mut pending = Vec::new();
    for source_span in source_spans.spans() {
        if source_span.bytes == 0 || source_span.offset & 3 != 0 {
            return Err(ExchangeLoweringError::UnalignedPayload);
        }
        let mut offset = 0;
        while offset < source_span.bytes {
            let mut bytes = (source_span.bytes - offset).min(MAX_TRANSFER_WORDS * 4);
            for receiver in &mut receivers {
                let span = receiver.peek().ok_or(ExchangeLoweringError::SizeMismatch)?;
                if span.offset & 3 != 0 {
                    return Err(ExchangeLoweringError::UnalignedPayload);
                }
                bytes = bytes.min(span.bytes);
            }
            if bytes == 0 || bytes & 3 != 0 {
                return Err(ExchangeLoweringError::UnalignedPayload);
            }
            let source_offset = source_span
                .offset
                .checked_add(offset)
                .ok_or(ExchangeLoweringError::Overflow)?;
            let source_address = source_base
                .checked_add(source_offset)
                .ok_or(ExchangeLoweringError::Overflow)?;
            let destination_entries = destinations
                .iter()
                .zip(&mut receivers)
                .map(|((tile, base, _), receiver)| {
                    let span = receiver
                        .peek_mut()
                        .ok_or(ExchangeLoweringError::SizeMismatch)?;
                    let address = base
                        .checked_add(span.offset)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    span.offset = span
                        .offset
                        .checked_add(bytes)
                        .ok_or(ExchangeLoweringError::Overflow)?;
                    span.bytes -= bytes;
                    if span.bytes == 0 {
                        receiver.next();
                    }
                    Ok((*tile, address))
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            pending.push(PendingTransfer {
                source: source.tile,
                source_shard: source.id,
                source_offset,
                destinations: destination_entries,
                source_addresses: vec![source_address],
                source_elements: effective_memory_elements(source_address, bytes / 4),
                words: bytes / 4,
                width: ExchangeItemWidth::Word32,
                reserved_source: None,
            });
            offset += bytes;
        }
    }
    if receivers.iter_mut().any(|r| r.peek().is_some()) {
        return Err(ExchangeLoweringError::SizeMismatch);
    }
    Ok(pending)
}

#[derive(Clone)]
pub(super) struct PendingTransfer {
    pub(super) source: u16,
    pub(super) source_shard: BlockValueId,
    pub(super) source_offset: u32,
    pub(super) destinations: Vec<(u16, u32)>,
    pub(super) source_addresses: Vec<u32>,
    pub(super) source_elements: Vec<u32>,
    pub(super) words: u32,
    pub(super) width: ExchangeItemWidth,
    pub(super) reserved_source: Option<u16>,
}

impl PendingTransfer {
    pub(super) fn moving_source(&self) -> bool {
        self.source_addresses
            .iter()
            .any(|&a| a != self.source_address())
    }

    pub(super) fn tiles(&self) -> impl Iterator<Item = u16> + '_ {
        std::iter::once(self.source)
            .chain(self.reserved_source)
            .chain(self.destinations.iter().map(|entry| entry.0))
    }

    pub(super) fn pressure_resources(&self, directional: bool) -> impl Iterator<Item = usize> + '_ {
        let stride = if directional { 2 } else { 1 };
        std::iter::once(self.source)
            .chain(self.reserved_source)
            .map(move |tile| usize::from(tile) * stride)
            .chain(
                self.destinations
                    .iter()
                    .map(move |&(tile, _)| usize::from(tile) * stride + usize::from(directional)),
            )
    }

    pub(super) fn source_address(&self) -> u32 {
        self.source_addresses[0]
    }

    pub(super) fn item_count(&self) -> Result<u32, ExchangeLoweringError> {
        self.width.item_count(self.words)
    }

    /// Payload eligibility only; topology and route encoding are checked separately.
    pub(super) fn supports_paired_width(&self) -> bool {
        self.words >= 2
            && self.words.is_multiple_of(2)
            && self
                .source_addresses
                .iter()
                .copied()
                .chain(self.destinations.iter().map(|&(_, address)| address))
                .all(|address| address.is_multiple_of(8))
    }

    pub(super) fn refresh_source_elements(&mut self) {
        self.source_elements = self
            .source_addresses
            .iter()
            .flat_map(|&address| effective_memory_elements(address, self.words))
            .collect();
        self.source_elements.sort_unstable();
        self.source_elements.dedup();
    }
}

/// Use a common moving source as each tile's base, patching exceptions.
/// All relative addresses must remain representable for every Repeat binding.
pub(super) fn repeat_outgoing_bases(
    pending: &[PendingTransfer],
    patch_words: &[usize],
    addresses: &BTreeMap<BlockValueId, u32>,
    tile_count: u16,
) -> Vec<Option<(BlockValueId, u32)>> {
    let mut tiles = vec![Vec::new(); usize::from(tile_count)];
    for (transfer, &words) in pending.iter().zip(patch_words) {
        if transfer.moving_source() {
            tiles[usize::from(transfer.source)].push((transfer, words));
        }
    }
    tiles
        .into_iter()
        .map(|transfers| {
            let count = transfers
                .iter()
                .map(|(t, _)| t.source_addresses.len())
                .max()
                .unwrap_or(1);
            let mut patterns = BTreeMap::<Vec<u32>, (usize, &PendingTransfer)>::new();
            for &(transfer, words) in &transfers {
                let deltas = (0..count)
                    .map(|i| {
                        repeat_source_address(transfer, i).wrapping_sub(transfer.source_address())
                    })
                    .collect::<Vec<_>>();
                let entry = patterns.entry(deltas).or_insert((0, transfer));
                entry.0 += words;
                if transfer.source_address() < entry.1.source_address() {
                    entry.1 = transfer;
                }
            }
            let paired = transfers
                .iter()
                .any(|(t, _)| t.width == ExchangeItemWidth::Paired64);
            patterns
                .into_values()
                .filter(|&(_, base)| {
                    (!paired || base.source_address().is_multiple_of(8))
                        && transfers.iter().all(|(t, _)| {
                            (0..count).all(|i| {
                                repeat_source_address(t, i) >= repeat_source_address(base, i)
                            })
                        })
                })
                .max_by_key(|&(uses, base)| (uses, Reverse(base.source_address())))
                .and_then(|(_, base)| {
                    let offset = base
                        .source_address()
                        .checked_sub(*addresses.get(&base.source_shard)?)?;
                    Some((base.source_shard, offset))
                })
        })
        .collect()
}

pub(super) fn repeat_source_address(transfer: &PendingTransfer, iteration: usize) -> u32 {
    transfer
        .source_addresses
        .get(iteration)
        .copied()
        .unwrap_or(transfer.source_address())
}

fn attach_repeat_source_addresses(
    pending: &mut [PendingTransfer],
    repeat_inputs: &BTreeMap<BlockValueId, Vec<u32>>,
) -> Result<(), ExchangeLoweringError> {
    for transfer in pending {
        if let Some(inputs) = repeat_inputs.get(&transfer.source_shard) {
            let addresses = inputs
                .iter()
                .map(|address| {
                    address
                        .checked_add(transfer.source_offset)
                        .ok_or(ExchangeLoweringError::Overflow)
                })
                .collect::<Result<Vec<_>, ExchangeLoweringError>>()?;
            if addresses.first().copied() != Some(transfer.source_address()) {
                return Err(ExchangeLoweringError::IncompatibleRepeatRows(
                    "argument address differs from its first sequence member",
                ));
            }
            transfer.source_addresses = addresses;
        }
        transfer.refresh_source_elements();
    }
    Ok(())
}
/// Combine contiguous source and destination spans into one hardware message.
pub(super) fn coalesce_pending_transfers(transfers: Vec<PendingTransfer>) -> Vec<PendingTransfer> {
    let mut merged = Vec::<PendingTransfer>::with_capacity(transfers.len());
    for transfer in transfers {
        let Some(previous) = merged.last_mut() else {
            merged.push(transfer);
            continue;
        };
        let previous_bytes = previous.words * 4;
        let contiguous = previous.source == transfer.source
            && previous.width == transfer.width
            && previous.source_shard == transfer.source_shard
            && previous
                .source_offset
                .checked_add(previous_bytes)
                .is_some_and(|end| end == transfer.source_offset)
            && previous.destinations.len() == transfer.destinations.len()
            && previous.source_addresses.len() == transfer.source_addresses.len()
            && previous
                .source_addresses
                .iter()
                .zip(&transfer.source_addresses)
                .all(|(&left, &right)| {
                    left.checked_add(previous_bytes)
                        .is_some_and(|end| end == right)
                })
            && previous
                .destinations
                .iter()
                .zip(&transfer.destinations)
                .all(
                    |(&(left_tile, left_address), &(right_tile, right_address))| {
                        left_tile == right_tile
                            && left_address
                                .checked_add(previous_bytes)
                                .is_some_and(|end| end == right_address)
                    },
                );
        // A merged phase may contain receive-then-forward dependencies. Do
        // not absorb a local write into a SEND that also reads those bytes;
        // memory_dependencies must still see the original ordered transfers.
        let combined_words = previous
            .words
            .checked_add(transfer.words)
            .filter(|&words| contiguous && words <= MAX_TRANSFER_WORDS)
            .filter(|&words| {
                let bytes = u64::from(words) * 4;
                previous.destinations.iter().all(|&(tile, address)| {
                    tile != previous.source
                        || previous.source_addresses.iter().all(|&source| {
                            u64::from(address) + bytes <= u64::from(source)
                                || u64::from(source) + bytes <= u64::from(address)
                        })
                })
            });
        if let Some(words) = combined_words {
            previous.words = words;
            previous.refresh_source_elements();
        } else {
            merged.push(transfer);
        }
    }
    merged
}
