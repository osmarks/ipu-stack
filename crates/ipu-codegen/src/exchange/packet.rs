//! Packet boundaries must be encodable when a sender is also a receiver.
use super::*;

pub(super) fn split_self_receive_conflicts(
    topology: &Topology,
    pending: Vec<PendingTransfer>,
) -> Result<Vec<PendingTransfer>, ExchangeLoweringError> {
    if !pending.iter().any(|transfer| {
        transfer
            .destinations
            .iter()
            .any(|&(tile, _)| tile == transfer.source)
    }) {
        return Ok(pending);
    }
    let mut result = Vec::with_capacity(pending.len());
    let mut work = pending;
    work.reverse();
    while let Some(transfer) = work.pop() {
        let Some(receiver) = transfer
            .destinations
            .iter()
            .position(|&(tile, _)| tile == transfer.source)
        else {
            result.push(transfer);
            continue;
        };
        let tiles = transfer
            .destinations
            .iter()
            .map(|&(tile, _)| tile)
            .collect::<Vec<_>>();
        let count = transfer.item_count()?;
        let plan = match transfer.width {
            ExchangeItemWidth::Word32 => {
                ipu_exchange::multicast(&topology, transfer.source, &tiles, count, 0)?
            }
            ExchangeItemWidth::Paired64 => {
                ipu_exchange::paired_multicast(&topology, transfer.source, &tiles, count)?
            }
        };
        if !plan.prepare()?.receiver_conflicts_with_send_start(receiver) {
            result.push(transfer);
            continue;
        }
        if count == 1 {
            return Err(ExchangeLoweringError::Invariant(
                "single-item self-receive conflicts with send initialization".into(),
            ));
        }
        let mut first = transfer.clone();
        first.words = (count / 2) * transfer.width.item_words();
        first.refresh_source_elements();
        let mut rest = transfer;
        rest.words -= first.words;
        let bytes = first
            .words
            .checked_mul(4)
            .ok_or(ExchangeLoweringError::Overflow)?;
        for address in rest
            .source_addresses
            .iter_mut()
            .chain(rest.destinations.iter_mut().map(|(_, address)| address))
        {
            *address = address
                .checked_add(bytes)
                .ok_or(ExchangeLoweringError::Overflow)?;
        }
        rest.source_offset = rest
            .source_offset
            .checked_add(bytes)
            .ok_or(ExchangeLoweringError::Overflow)?;
        rest.refresh_source_elements();
        // Keep byte order, including the original dependency order and every
        // Repeat source address. Check both smaller packets by the same rule.
        work.push(rest);
        work.push(first);
    }
    Ok(result)
}
