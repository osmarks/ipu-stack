//! Decoding and validation for generated supervisor exchange programs.
//!
//! The decoder follows the exchange event timeline rather than the ordinary
//! supervisor instruction stream. In particular, `sendpicp` is one aligned
//! two-word supervisor instruction: the first word carries the send fields and
//! the second is inline PIC/XPIC payload, not an independently executed word.

use super::*;
use ipu_target::ipu21::instruction::SETZI_M_OPCODE;
use std::fmt::{self, Write};

/// Address-bearing instructions for each outgoing message, in execution
/// order. Each entry is `(word offset, byte offset from the message source)`.
/// SENDPICP restarts the outgoing source stream explicitly after its inline
/// control word, so repeat relocation must patch it as well as the first SEND.
pub fn sender_address_instruction_groups(
    row: &[u32],
) -> Result<Vec<Vec<(usize, u32)>>, ExchangeError> {
    let mut groups = Vec::<Vec<(usize, u32)>>::new();
    let mut source_address = None;
    let mut cursor = 0;
    while cursor < row.len() {
        let instruction = row[cursor];
        let address =
            || ((instruction & SEND_ADDRESS_MASK) >> 3) << if instruction & 4 != 0 { 3 } else { 2 };
        if instruction & LONG_OPCODE_MASK == SEND_OPCODE {
            groups.push(vec![(cursor, 0)]);
            source_address = Some(address());
        } else if is_send_control_pair(instruction) && instruction & 7 != 0 {
            let source = source_address.ok_or(ExchangeError::Schedule(
                "SENDPICP precedes initial outgoing SEND",
            ))?;
            // The restart contains its absolute source address. Read it directly;
            // recovering it from instruction durations duplicates stream logic.
            let offset = address()
                .checked_sub(source)
                .ok_or(ExchangeError::Schedule("SENDPICP precedes outgoing source"))?;
            groups
                .last_mut()
                .ok_or(ExchangeError::Schedule("SENDPICP outgoing group"))?
                .push((cursor, offset));
        }
        cursor += if is_send_control_pair(instruction) {
            2
        } else {
            1
        };
    }
    Ok(groups)
}

/// Removes tile-memory address fields while retaining exchange roles, routes,
/// transfer sizes, and event timing. Rows with the same result can share one
/// executable slot and restore their addresses before invocation.
pub fn normalized_exchange_address_words(row: &[u32]) -> Vec<u32> {
    let mut normalized = row.to_vec();
    let mut cursor = 0;
    while cursor < normalized.len() {
        let instruction = normalized[cursor];
        if is_send_control_pair(instruction) {
            if instruction & 7 != 0 {
                normalized[cursor] &= !SEND_ADDRESS_MASK;
            }
            if instruction & (1 << 27) == 0
                && let Some(payload) = normalized.get_mut(cursor + 1)
            {
                *payload &= !PIC_RECEIVE_ADDRESS_MASK;
            }
            cursor += 2;
            continue;
        }
        normalized[cursor] = if instruction & LONG_OPCODE_MASK == SEND_OPCODE {
            instruction & !SEND_ADDRESS_MASK
        } else if (is_send_control(instruction) && (instruction >> 18) & 3 == 2)
            || (instruction & OPCODE_MASK == DELAY_PIC_OPCODE && instruction & (1 << 18) == 0)
        {
            instruction & !PIC_RECEIVE_ADDRESS_MASK
        } else {
            instruction
        };
        cursor += 1;
    }
    normalized
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncomingControlStream {
    Pic,
    Xpic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncomingControl {
    pub stream: IncomingControlStream,
    /// Complete raw configuration value, including the stream's selector bit.
    pub value: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SendEncoding {
    Explicit,
    Offset,
    Pic,
    PicPair,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanOperation {
    Delay,
    SetImmediate {
        register: u8,
        value: u32,
    },
    WriteBase {
        incoming: bool,
        register: u8,
    },
    IncomingControl(IncomingControl),
    Send {
        encoding: SendEncoding,
        words: u32,
        /// Encoding-specific raw operand: an initial source word address, a
        /// continuation delta, or compact direction/control bits.
        raw_operand: u32,
        /// Explicit three-bit SCTL field. `None` denotes SENDPIC, which
        /// continues the currently active outgoing stream implicitly.
        send_control: Option<u8>,
        controls: Vec<IncomingControl>,
    },
    Sync(u8),
    Return,
    Unknown(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodedPlanInstruction {
    pub word_offset: u32,
    pub address: Option<u32>,
    pub start_cycle: u32,
    pub end_cycle: u32,
    pub operation: PlanOperation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanProgramDiagnostic {
    pub instructions: Vec<DecodedPlanInstruction>,
    pub event_cycles: u32,
    pub row_words: u32,
}

impl fmt::Display for DecodedPlanInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.address {
            Some(address) => write!(f, "0x{address:05x}"),
            None => write!(f, "word+{}", self.word_offset),
        }?;
        write!(
            f,
            " cycles={}..{} {:?}",
            self.start_cycle, self.end_cycle, self.operation
        )
    }
}

impl PlanProgramDiagnostic {
    pub fn render(&self) -> String {
        let mut output = String::new();
        for instruction in &self.instructions {
            writeln!(output, "{instruction}").unwrap();
        }
        output
    }

    pub fn render_around_address(&self, address: u32, radius: usize) -> String {
        let focus = self
            .instructions
            .iter()
            .position(|instruction| {
                instruction.address.is_some_and(|start| {
                    let width = if matches!(
                        instruction.operation,
                        PlanOperation::Send {
                            encoding: SendEncoding::PicPair,
                            ..
                        }
                    ) {
                        8
                    } else {
                        4
                    };
                    address
                        .checked_sub(start)
                        .is_some_and(|offset| offset < width)
                })
            })
            .unwrap_or_else(|| {
                self.instructions
                    .partition_point(|instruction| {
                        instruction.address.is_some_and(|pc| pc < address)
                    })
                    .min(self.instructions.len().saturating_sub(1))
            });
        let start = focus.saturating_sub(radius);
        let end = focus
            .saturating_add(radius)
            .saturating_add(1)
            .min(self.instructions.len());
        let mut output = String::new();
        for (index, instruction) in self.instructions[start..end].iter().enumerate() {
            let marker = if start + index == focus { ">" } else { " " };
            writeln!(output, "{marker} {instruction}").unwrap();
        }
        output
    }
}

/// Decodes one synchronization-free exchange row. `base_address` is optional
/// because provisional rows do not have placement yet.
pub fn diagnose_plan_program(
    words: &[u32],
    base_address: Option<u32>,
) -> Result<PlanProgramDiagnostic, ExchangeError> {
    let mut instructions = Vec::new();
    let mut cycle = 0u32;
    let mut offset = 0usize;
    while offset < words.len() {
        let start = cycle;
        let (operation, advance, width) = decode_operation(words, offset)?;
        cycle = cycle
            .checked_add(advance)
            .ok_or(ExchangeError::Schedule("diagnostic event horizon overflow"))?;
        let address = match base_address {
            Some(base) => Some(
                base.checked_add(offset as u32 * 4)
                    .ok_or(ExchangeError::Schedule("diagnostic row address overflow"))?,
            ),
            None => None,
        };
        instructions.push(DecodedPlanInstruction {
            word_offset: offset as u32,
            address,
            start_cycle: start,
            end_cycle: cycle,
            operation,
        });
        offset += width;
    }
    Ok(PlanProgramDiagnostic {
        instructions,
        event_cycles: cycle,
        row_words: words.len() as u32,
    })
}

fn decode_operation(
    words: &[u32],
    offset: usize,
) -> Result<(PlanOperation, u32, usize), ExchangeError> {
    let word = words[offset];
    if word == RETURN_M10_INSTRUCTION {
        return Ok((PlanOperation::Return, 0, 1));
    }
    if word & DELAY_OPCODE_MASK == DELAY_OPCODE {
        return Ok((PlanOperation::Delay, (word & 0x7_ffff) + 1, 1));
    }
    if word & 0xff00_0000 == SETZI_M_OPCODE {
        return Ok((
            PlanOperation::SetImmediate {
                register: ((word >> 20) & 15) as u8,
                value: word & 0xf_ffff,
            },
            1,
            1,
        ));
    }
    if word & 0xff0f_ffff == PUT_SPECIAL_M_OPCODE | u32::from(INCOMING_BASE)
        || word & 0xff0f_ffff == PUT_SPECIAL_M_OPCODE | u32::from(OUTGOING_BASE)
    {
        return Ok((
            PlanOperation::WriteBase {
                incoming: word & 0xff == u32::from(INCOMING_BASE),
                register: ((word >> 20) & 15) as u8,
            },
            EXCHANGE_BASE_WRITE_CYCLES,
            1,
        ));
    }
    if word & OPCODE_MASK == DELAY_PIC_OPCODE {
        let advance = ((word >> 19) & 0x7f) + 1;
        let value = (((word >> 18) & 1) << 18) | (word & PIC_RECEIVE_ADDRESS_MASK);
        return Ok((
            PlanOperation::IncomingControl(IncomingControl {
                stream: IncomingControlStream::Pic,
                value,
            }),
            advance,
            1,
        ));
    }
    if word & OPCODE_MASK == DELAY_XPIC_OPCODE {
        let advance = ((word >> 14) & 0xfff) + 1;
        let value = (((word >> 13) & 1) << 13) | (word & 0x1fff);
        return Ok((
            PlanOperation::IncomingControl(IncomingControl {
                stream: IncomingControlStream::Xpic,
                value,
            }),
            advance,
            1,
        ));
    }
    if is_send_control_pair(word) {
        let payload = *words
            .get(offset + 1)
            .ok_or(ExchangeError::Schedule("truncated SENDPICP payload"))?;
        let words = ((word >> 21) & 0x3f) + 1;
        return Ok((
            PlanOperation::Send {
                encoding: SendEncoding::PicPair,
                words,
                raw_operand: (word & SEND_ADDRESS_MASK) >> 3,
                send_control: Some((word & 7) as u8),
                controls: vec![
                    IncomingControl {
                        stream: IncomingControlStream::Xpic,
                        value: payload >> 18,
                    },
                    IncomingControl {
                        stream: IncomingControlStream::Pic,
                        value: (((word >> 27) & 1) << 18) | (payload & PIC_RECEIVE_ADDRESS_MASK),
                    },
                ],
            },
            words,
            2,
        ));
    }
    if is_send_control(word) {
        let words = ((word >> 21) & 0x3f) + 1;
        let selector = (word >> 18) & 3;
        let control = if selector < 2 {
            IncomingControl {
                stream: IncomingControlStream::Xpic,
                value: (selector << 13) | (word & 0x1fff),
            }
        } else {
            IncomingControl {
                stream: IncomingControlStream::Pic,
                value: ((selector - 2) << 18) | (word & PIC_RECEIVE_ADDRESS_MASK),
            }
        };
        return Ok((
            PlanOperation::Send {
                encoding: SendEncoding::Pic,
                words,
                raw_operand: word & PIC_RECEIVE_ADDRESS_MASK,
                send_control: None,
                controls: vec![control],
            },
            words,
            1,
        ));
    }
    if word & LONG_OPCODE_MASK == SEND_OPCODE {
        let words = ((word >> 21) & 0x3f) + 1;
        return Ok((
            PlanOperation::Send {
                encoding: SendEncoding::Explicit,
                words,
                raw_operand: (word & SEND_ADDRESS_MASK) >> 3,
                send_control: Some((word & 7) as u8),
                controls: Vec::new(),
            },
            words,
            1,
        ));
    }
    if is_send_off(word) {
        let words = (((word >> 21) & 0x3f) | (((word >> 14) & 0x3f) << 6)) + 1;
        return Ok((
            PlanOperation::Send {
                encoding: SendEncoding::Offset,
                words,
                raw_operand: (word & 0x3ff8) >> 3,
                send_control: Some((word & 7) as u8),
                controls: Vec::new(),
            },
            words,
            1,
        ));
    }
    if word & !0xff == SYNC_OPCODE {
        return Ok((PlanOperation::Sync((word & 0xff) as u8), 0, 1));
    }
    Ok((PlanOperation::Unknown(word), 0, 1))
}

pub(super) fn validate_tile_program(
    tile: usize,
    schedule: &TileProgramSchedule,
    words: &[u32],
) -> Result<PlanProgramDiagnostic, ExchangeError> {
    let diagnostic = diagnose_plan_program(words, None)?;
    if diagnostic.event_cycles != schedule.event_cycles {
        return Err(ExchangeError::Schedule("encoded tile horizon mismatch"));
    }
    if diagnostic
        .instructions
        .iter()
        .any(|instruction| matches!(instruction.operation, PlanOperation::Unknown(_)))
    {
        return Err(ExchangeError::Schedule(
            "unknown encoded exchange instruction",
        ));
    }

    let actual_bases = diagnostic
        .instructions
        .iter()
        .filter_map(|i| match i.operation {
            PlanOperation::WriteBase {
                incoming: false, ..
            } => Some((i.end_cycle, words[i.word_offset as usize])),
            _ => None,
        });
    let expected_bases = schedule
        .receive_events
        .iter()
        .filter(|e| matches!(e.kind, ReceiveEventKind::OutgoingBase(_)))
        .map(|e| (e.cycles, e.instruction));
    if !actual_bases.eq(expected_bases) {
        return Err(ExchangeError::Schedule("encoded outgoing base mismatch"));
    }
    let mut actual_controls = Vec::new();
    let mut actual_sends = Vec::new();
    for instruction in &diagnostic.instructions {
        match &instruction.operation {
            PlanOperation::IncomingControl(control) => {
                actual_controls.push((instruction.end_cycle, *control));
            }
            PlanOperation::Send {
                encoding,
                send_control,
                controls,
                ..
            } => {
                // Merged incoming writes happen when the send is issued, not
                // after all of its serial payload words have left the tile.
                actual_controls.extend(
                    controls
                        .iter()
                        .map(|control| (instruction.start_cycle + 1, *control)),
                );
                if *send_control != Some(0) {
                    actual_sends.push((instruction.start_cycle, instruction.end_cycle, *encoding));
                }
            }
            _ => {}
        }
    }
    actual_controls.sort_unstable_by_key(|entry| (entry.0, control_key(entry.1)));
    let mut expected_controls = schedule
        .receive_events
        .iter()
        .filter_map(|event| {
            let control = match event.kind {
                ReceiveEventKind::OutgoingBase(_) => return None,
                ReceiveEventKind::Pointer
                | ReceiveEventKind::PairedPointer
                | ReceiveEventKind::Format => IncomingControl {
                    stream: IncomingControlStream::Pic,
                    value: (((event.instruction >> 18) & 1) << 18)
                        | (event.instruction & PIC_RECEIVE_ADDRESS_MASK),
                },
                ReceiveEventKind::OrdinarySource
                | ReceiveEventKind::OrdinaryNeutral
                | ReceiveEventKind::PairedSource
                | ReceiveEventKind::PairedNeutral => IncomingControl {
                    stream: IncomingControlStream::Xpic,
                    value: (((event.instruction >> 13) & 1) << 13) | (event.instruction & 0x1fff),
                },
            };
            Some((event.cycles, control))
        })
        .collect::<Vec<_>>();
    expected_controls.sort_unstable_by_key(|entry| (entry.0, control_key(entry.1)));
    if actual_controls != expected_controls {
        let first_mismatch = actual_controls
            .iter()
            .zip(&expected_controls)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or_else(|| actual_controls.len().min(expected_controls.len()));
        let window_start = first_mismatch.saturating_sub(3);
        let window_end =
            (first_mismatch + 4).min(actual_controls.len().max(expected_controls.len()));
        tracing::debug!(
            tile,
            first_mismatch,
            actual_len = actual_controls.len(),
            expected_len = expected_controls.len(),
            actual = ?actual_controls.get(first_mismatch),
            expected = ?expected_controls.get(first_mismatch),
            actual_window = ?actual_controls.get(window_start..window_end.min(actual_controls.len())),
            expected_window = ?expected_controls.get(window_start..window_end.min(expected_controls.len())),
            "encoded incoming controls differ from the phase schedule"
        );
        return Err(ExchangeError::Schedule("encoded incoming-control mismatch"));
    }

    for expected in schedule.senders.iter() {
        let mut end = None;
        let contiguous = actual_sends
            .iter()
            .filter(|(start, end, _)| {
                *start < expected.timing.payload_end && expected.timing.payload_start < *end
            })
            .all(|&(start, next, _)| {
                let adjacent = start == end.unwrap_or(expected.timing.payload_start);
                end = Some(next);
                adjacent
            });
        if !contiguous || end != Some(expected.timing.payload_end) {
            return Err(ExchangeError::Schedule("encoded sender interval mismatch"));
        }
    }
    for (start, end, _) in actual_sends {
        let belongs_to_sender = schedule
            .senders
            .iter()
            .any(|sender| sender.timing.payload_start <= start && end <= sender.timing.payload_end);
        if !belongs_to_sender {
            return Err(ExchangeError::Schedule(
                "unexpected encoded outgoing interval",
            ));
        }
    }
    Ok(diagnostic)
}

fn control_key(control: IncomingControl) -> (u8, u32) {
    (
        match control.stream {
            IncomingControlStream::Pic => 0,
            IncomingControlStream::Xpic => 1,
        },
        control.value,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_all_sender_address_fields() {
        let row = [
            SYNC_SUPERVISOR_INSTRUCTION,
            encode_send(1, 3, 0x1a048).unwrap(),
            SEND_PICP_OPCODE | (7 << 21) | (0x1a04a << 3) | 3,
            0x1901_5000,
            RETURN_M10_INSTRUCTION,
        ];
        let normalized = normalized_exchange_address_words(&row);
        assert_eq!(normalized[1] ^ row[1], row[1] & SEND_ADDRESS_MASK);
        assert_eq!(normalized[2] ^ row[2], row[2] & SEND_ADDRESS_MASK);
        assert_eq!(normalized[0], row[0]);
        assert_eq!(normalized[4], row[4]);
    }

    #[test]
    fn restart_relocation_uses_encoded_addresses_for_both_send_widths() {
        for mode in [3, 7] {
            let shift = if mode & 4 != 0 { 3 } else { 2 };
            let source = 0x50000;
            let row = [
                encode_send(1, mode, source >> shift).unwrap(),
                SEND_PICP_OPCODE | (7 << 21) | (((source + 80) >> shift) << 3) | mode,
                0x1901_5000,
                RETURN_M10_INSTRUCTION,
            ];
            let groups = sender_address_instruction_groups(&row).unwrap();
            assert_eq!(groups, vec![vec![(0, 0), (1, 80)]]);
            for (word, offset) in &groups[0] {
                let mut instruction = row[*word];
                patch_sender_instruction(&mut instruction, source + offset).unwrap();
                assert_eq!(instruction, row[*word]);
                patch_sender_instruction(&mut instruction, source + 256 + offset).unwrap();
                assert_eq!(
                    ((instruction & SEND_ADDRESS_MASK) >> 3) << shift,
                    source + 256 + offset
                );
            }
        }
    }

    #[test]
    fn diagnostic_windows_handle_unbounded_radius_and_unplaced_rows() {
        let words = [delay(2), RETURN_M10_INSTRUCTION];
        let unplaced = diagnose_plan_program(&words, None).unwrap();
        assert_eq!(
            unplaced.render(),
            "word+0 cycles=0..3 Delay\nword+1 cycles=3..3 Return\n"
        );
        let placed = diagnose_plan_program(&words, Some(0x60000)).unwrap();
        assert_eq!(
            placed.render_around_address(0x60004, usize::MAX),
            "  0x60000 cycles=0..3 Delay\n> 0x60004 cycles=3..3 Return\n"
        );
        assert_eq!(
            placed.render_around_address(0x60004, 0),
            "> 0x60004 cycles=3..3 Return\n"
        );
        assert_eq!(
            diagnose_plan_program(&[], None)
                .unwrap()
                .render_around_address(0, usize::MAX),
            ""
        );
    }

    #[test]
    fn sdk_receiver_row_decodes_two_word_controls_as_single_instructions() {
        let words = [
            0x6400_0082,
            0x40a0_0032,
            0xf660_0000,
            0x0301_4048,
            0xf660_0000,
            0x0309_5000,
            0xf660_0000,
            0x0401_6000,
            0xf720_0000,
            0x1901_7000,
            0x43a0_0000,
        ];
        let decoded = diagnose_plan_program(&words, Some(0x60000)).unwrap();
        assert_eq!(decoded.row_words, words.len() as u32);
        assert_eq!(decoded.instructions.len(), 7);
        let paired = decoded
            .instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.operation,
                    PlanOperation::Send {
                        encoding: SendEncoding::PicPair,
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(paired.len(), 4);
        assert_eq!(paired[0].address, Some(0x60008));
        assert_eq!(paired[1].address, Some(0x60010));
        assert_eq!(paired[0].end_cycle - paired[0].start_cycle, 52);
        assert!(
            decoded
                .instructions
                .iter()
                .all(|instruction| !matches!(instruction.operation, PlanOperation::Unknown(_)))
        );
    }

    #[test]
    fn sdk_full_duplex_pair_decodes_absolute_send_restart() {
        let decoded = diagnose_plan_program(&[0xf54a_0109, 0x1901_5000], None).unwrap();
        assert_eq!(decoded.event_cycles, 43);
        assert_eq!(
            decoded.instructions[0].operation,
            PlanOperation::Send {
                encoding: SendEncoding::PicPair,
                words: 43,
                raw_operand: 0x14021,
                send_control: Some(1),
                controls: vec![
                    IncomingControl {
                        stream: IncomingControlStream::Xpic,
                        value: 0x640,
                    },
                    IncomingControl {
                        stream: IncomingControlStream::Pic,
                        value: 0x15000,
                    },
                ],
            }
        );
    }

    #[test]
    fn paired_control_decoder_preserves_the_pic_selector() {
        let events = [
            ReceiveEvent {
                cycles: 1,
                instruction: delay_xpic(0, 0, 0),
                kind: ReceiveEventKind::OrdinarySource,
            },
            ReceiveEvent {
                cycles: 1,
                instruction: delay_pic(0, 1, 1),
                kind: ReceiveEventKind::Format,
            },
        ];
        let (instruction, payload) = encode_send_control_pair(0, 0, 0, &events).unwrap();
        let decoded = diagnose_plan_program(&[instruction, payload], None).unwrap();
        let PlanOperation::Send { controls, .. } = &decoded.instructions[0].operation else {
            panic!("expected a paired control instruction");
        };
        assert!(controls.contains(&IncomingControl {
            stream: IncomingControlStream::Pic,
            value: 0x40001,
        }));
    }
}
