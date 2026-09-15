//! Structural validation of caller-supplied supervisor programs.
use super::*;

impl ExchangePatchValues {
    fn valid_for_count(&self, count: u32) -> bool {
        match self {
            Self::Table(row) => row.words.len() == count as usize && row.address.is_multiple_of(4),
            Self::Arithmetic { .. } => count != 0,
        }
    }
}

pub(super) fn program(
    program: &TileProgram,
    host: &HostProgram,
    options: &CodegenOptions,
) -> Result<Vec<PlacedExchangeRow>> {
    if options.invocations == 0 {
        return Err(invalid("invocation count must be nonzero"));
    }
    if host
        .initialize
        .iter()
        .chain(&host.inputs)
        .chain(&host.outputs)
        .any(|phase| phase.active && phase.run_table.is_none())
    {
        return Err(invalid("active host phase has no run table"));
    }
    let mut rows = BTreeMap::new();
    validate_steps(&program.steps, None, None, &mut rows)?;
    Ok(rows
        .into_iter()
        .map(|(address, words)| PlacedExchangeRow { address, words })
        .collect())
}

fn validate_steps(
    steps: &[TileStep],
    repeat_pointer_count: Option<usize>,
    repeat_count: Option<u32>,
    rows: &mut BTreeMap<u32, Vec<u32>>,
) -> Result<()> {
    for step in steps {
        match step {
            TileStep::Exchange(exchange) => {
                validate_exchange_program(exchange)?;
                if let Some(base) = exchange.outgoing_base {
                    if exchange.preserve_base_registers {
                        return Err(invalid(
                            "exchange base relocation conflicts with preserved bases",
                        ));
                    }
                    validate_address(base, repeat_pointer_count)?;
                }
                if exchange.setup_patch.as_ref().is_some_and(|patch| {
                    patch.offsets.words.is_empty()
                        || patch.offsets.words.len() != patch.values.words.len()
                }) {
                    return Err(invalid("exchange setup patch has an invalid shape"));
                }
                let mut patched_words = std::collections::BTreeSet::new();
                for patch in &exchange.repeat_patches {
                    if !patched_words.insert(patch.word_offset)
                        || repeat_count.is_none_or(|count| !patch.values.valid_for_count(count))
                        || patch.word_offset as usize >= exchange.program.words.len()
                    {
                        return Err(invalid("exchange patch has invalid shape or address"));
                    }
                }
                let data = std::iter::once(&exchange.program)
                    .chain(
                        exchange
                            .setup_patch
                            .iter()
                            .flat_map(|patch| [&patch.offsets, &patch.values]),
                    )
                    .chain(exchange.repeat_patches.iter().filter_map(
                        |patch| match &patch.values {
                            ExchangePatchValues::Table(row) => Some(row),
                            ExchangePatchValues::Arithmetic { .. } => None,
                        },
                    ));
                for row in data {
                    if let Some(previous) = rows.get(&row.address) {
                        if previous != &row.words {
                            return Err(invalid("different exchange rows share an address"));
                        }
                    } else {
                        rows.insert(row.address, row.words.clone());
                    }
                }
            }
            TileStep::Compute(compute) => {
                if compute.symbol.is_empty() {
                    return Err(invalid("compute symbol is empty"));
                }
                let values = compute.input_addresses.len() + compute.arguments.len();
                let available = usize::from(LAST_VALUE_REGISTER - FIRST_INPUT_REGISTER + 1);
                if values == 0 || values > available {
                    return Err(invalid(format!(
                        "kernel {} needs {values} input/argument registers; 1..={available} are supported",
                        compute.symbol
                    )));
                }
                validate_address(compute.output_address, repeat_pointer_count)?;
                for &address in &compute.input_addresses {
                    validate_address(address, repeat_pointer_count)?;
                }
            }
            TileStep::Repeat(repeat) => {
                if repeat_pointer_count.is_some() {
                    return Err(invalid("nested finalized repeats are not yet supported"));
                }
                if repeat.count == 0 {
                    return Err(invalid("repeat count must be nonzero"));
                }
                if repeat.iterated_pointers.len() >= usize::from(u16::MAX) {
                    return Err(invalid("too many repeat pointers"));
                }
                validate_steps(
                    &repeat.body,
                    Some(repeat.iterated_pointers.len()),
                    Some(repeat.count),
                    rows,
                )?;
            }
            TileStep::Checkpoint(checkpoint) => {
                if checkpoint.breakpoint > 1 {
                    return Err(invalid("checkpoint breakpoint must be zero or one"));
                }
            }
        }
    }
    Ok(())
}

fn validate_exchange_program(exchange: &ExchangeStep) -> Result<()> {
    let embedded_sync = exchange
        .program
        .words
        .first()
        .is_some_and(|word| *word == SYNC_SUPERVISOR_INSTRUCTION);
    if exchange.program.address & 0b11 != 0
        || exchange.program.words.last()
            != Some(&ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION)
        || embedded_sync != exchange.sync_in_program
        || exchange
            .program
            .words
            .iter()
            .skip(usize::from(embedded_sync))
            .any(|word| {
                matches!(
                    *word,
                    SANS_INACTIVE_INSTRUCTION | SYNC_SUPERVISOR_INSTRUCTION
                )
            })
    {
        return Err(invalid(
            "exchange phase has an invalid boundary or timed program",
        ));
    }
    if exchange.active != (exchange.program.words.len() > 1 + usize::from(embedded_sync)) {
        return Err(invalid(
            "exchange participation does not match timed program",
        ));
    }
    Ok(())
}

fn validate_address(address: TileAddress, repeat_pointer_count: Option<usize>) -> Result<()> {
    if let TileAddress::RepeatPointer { index, .. } = address
        && repeat_pointer_count.is_none_or(|count| usize::from(index) >= count)
    {
        return Err(invalid(
            "compute address refers to an unavailable repeat pointer",
        ));
    }
    Ok(())
}
