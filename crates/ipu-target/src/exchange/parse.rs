//! Decoding and validation for generated supervisor exchange programs.
//!
//! The decoder follows the exchange event timeline rather than the ordinary
//! supervisor instruction stream. In particular, `sendpicp` is one aligned
//! two-word supervisor instruction: the first word carries the send fields and
//! the second is inline PIC/XPIC payload, not an independently executed word.

use serde::{Deserialize, Serialize};

use super::{ExchangeError, ReceiveEventKind, TileProgramSchedule};
use crate::instruction::{
    DELAY_OPCODE, DELAY_OPCODE_MASK, DELAY_PIC_OPCODE, DELAY_XPIC_OPCODE, LONG_OPCODE_MASK,
    OPCODE_MASK, PIC_RECEIVE_ADDRESS_MASK, RETURN_M10_INSTRUCTION, SEND_ADDRESS_MASK, SEND_OPCODE,
    SYNC_OPCODE, is_send_control, is_send_control_pair, is_send_off,
};

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

impl PlanProgramDiagnostic {
    pub fn render(&self) -> String {
        let mut output = String::new();
        for instruction in &self.instructions {
            let location = instruction.address.map_or_else(
                || format!("word+{}", instruction.word_offset),
                |address| format!("0x{address:05x}"),
            );
            output.push_str(&format!(
                "{location} cycles={}..{} {:?}\n",
                instruction.start_cycle, instruction.end_cycle, instruction.operation
            ));
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
                    (start..start + width).contains(&address)
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
        let end = (focus + radius + 1).min(self.instructions.len());
        let mut output = String::new();
        for (index, instruction) in self.instructions[start..end].iter().enumerate() {
            let location = instruction.address.map_or_else(
                || format!("word+{}", instruction.word_offset),
                |instruction_address| format!("0x{instruction_address:05x}"),
            );
            let marker = if start + index == focus { ">" } else { " " };
            output.push_str(&format!(
                "{marker} {location} cycles={}..{} {:?}\n",
                instruction.start_cycle, instruction.end_cycle, instruction.operation
            ));
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
        .map(|event| {
            let control = match event.kind {
                ReceiveEventKind::Pointer | ReceiveEventKind::Format => IncomingControl {
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
            (event.cycles, control)
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

    for expected in &schedule.senders {
        let covering = actual_sends
            .iter()
            .filter(|(start, end, _)| *start < expected.end_cycles && expected.start_cycles < *end)
            .collect::<Vec<_>>();
        if covering.is_empty()
            || covering
                .first()
                .is_none_or(|entry| entry.0 != expected.start_cycles)
            || covering
                .last()
                .is_none_or(|entry| entry.1 != expected.end_cycles)
            || covering.windows(2).any(|pair| pair[0].1 != pair[1].0)
        {
            return Err(ExchangeError::Schedule("encoded sender interval mismatch"));
        }
    }
    for (start, end, _) in actual_sends {
        let belongs_to_sender = schedule
            .senders
            .iter()
            .any(|sender| sender.start_cycles <= start && end <= sender.end_cycles);
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
    use crate::exchange::{ReceiveEvent, encode_send_control_pair};
    use crate::instruction::{delay_pic, delay_xpic};

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
