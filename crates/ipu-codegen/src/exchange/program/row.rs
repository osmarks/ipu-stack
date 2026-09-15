//! Executable rows retain the relocation sites known during encoding. Consumers
//! may change addresses and base operands without interpreting the instruction
//! stream again; the independent decoder remains the oracle for imported rows.
use crate::exchange::program::{ExchangeError, PIC_RECEIVE_ADDRESS_MASK, SEND_ADDRESS_MASK};
use ipu_target::ipu21::instruction::{RETURN_M10_INSTRUCTION, encode_put_special_m};
use ipu_target::ipu21::registers::OUTGOING_BASE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRow {
    pub(crate) words: Vec<u32>,
    pub(crate) sends: Vec<SendAddress>,
    pub(crate) receive_pointers: Vec<u32>,
    pub(crate) outgoing_bases: Vec<OutgoingBaseWrite>,
}

/// One explicit source address, including a SENDPICP restart. `message` is the
/// caller's transfer identity, independent of insertion or execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SendAddress {
    pub word_offset: u32,
    pub message: u32,
    pub byte_offset: u32,
    // Byte-address scaling of this field: 2 for ordinary SEND, 3 for paired SEND.
    pub(crate) item_shift: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutgoingBaseWrite {
    pub word_offset: u32,
    pub register: u8,
}

impl EncodedRow {
    pub fn inactive() -> Self {
        Self {
            words: vec![RETURN_M10_INSTRUCTION],
            sends: Vec::new(),
            receive_pointers: Vec::new(),
            outgoing_bases: Vec::new(),
        }
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Discard relocation information only at the final emission boundary.
    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    pub fn send_addresses(&self) -> &[SendAddress] {
        &self.sends
    }

    pub fn outgoing_base_writes(&self) -> &[OutgoingBaseWrite] {
        &self.outgoing_bases
    }

    fn address_fields(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.sends
            .iter()
            .map(|site| (site.word_offset as usize, SEND_ADDRESS_MASK))
            .chain(
                self.receive_pointers
                    .iter()
                    .map(|&offset| (offset as usize, PIC_RECEIVE_ADDRESS_MASK)),
            )
    }

    /// Retain roles, routes, transfer sizes and timing; remove only addresses.
    pub fn normalized_words(&self) -> Vec<u32> {
        let mut words = self.words.clone();
        for (offset, mask) in self.address_fields() {
            words[offset] &= !mask;
        }
        words
    }

    /// Row sharing takes the union across all invocations, so an invocation
    /// with a zero/base-relative address still restores an earlier one's word.
    pub fn nonzero_address_word_offsets(&self) -> impl Iterator<Item = usize> + '_ {
        self.address_fields()
            .filter_map(|(offset, mask)| (self.words[offset] & mask != 0).then_some(offset))
    }

    pub fn relocated_send(&self, index: usize, byte_address: u32) -> Result<u32, ExchangeError> {
        let site = &self.sends[index];
        let item_address = byte_address >> site.item_shift;
        if byte_address & ((1 << site.item_shift) - 1) != 0 || item_address > SEND_ADDRESS_MASK >> 3
        {
            return Err(ExchangeError::Address(byte_address));
        }
        Ok((self.words[site.word_offset as usize] & !SEND_ADDRESS_MASK) | (item_address << 3))
    }

    pub fn set_send_address(
        &mut self,
        index: usize,
        byte_address: u32,
    ) -> Result<(), ExchangeError> {
        let word = self.relocated_send(index, byte_address)?;
        self.words[self.sends[index].word_offset as usize] = word;
        Ok(())
    }

    pub fn replace_outgoing_base_register(
        &mut self,
        from: u8,
        to: u8,
    ) -> Result<(), ExchangeError> {
        let instruction = encode_put_special_m(OUTGOING_BASE, to)?;
        for site in &mut self.outgoing_bases {
            if site.register == from {
                self.words[site.word_offset as usize] = instruction;
                site.register = to;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl EncodedRow {
    /// Check all sites, including zero-valued fields, against the independent
    /// decoder. Run this on incremental rows as well as complete encodings.
    pub(crate) fn assert_relocation_sites(&self, messages: impl Iterator<Item = u32>) {
        use crate::exchange::program::diagnostic::{
            IncomingControl, IncomingControlStream, PlanOperation, SendEncoding,
            diagnose_plan_program, normalized_exchange_address_words,
            sender_address_instruction_groups,
        };
        let groups = sender_address_instruction_groups(&self.words).unwrap();
        let messages = messages.collect::<Vec<_>>();
        assert_eq!(groups.len(), messages.len());
        let sends = groups
            .into_iter()
            .zip(messages)
            .flat_map(|(group, message)| {
                group
                    .into_iter()
                    .map(move |(word, byte_offset)| SendAddress {
                        word_offset: word as u32,
                        message,
                        byte_offset,
                        item_shift: if self.words[word] & 4 != 0 { 3 } else { 2 },
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(self.sends, sends);

        let pointer = |control: &IncomingControl| {
            control.stream == IncomingControlStream::Pic && control.value & (1 << 18) == 0
        };
        let decoded = diagnose_plan_program(&self.words, None).unwrap();
        let mut receives = Vec::new();
        let mut bases = Vec::new();
        for instruction in decoded.instructions {
            match instruction.operation {
                PlanOperation::IncomingControl(control) if pointer(&control) => {
                    receives.push(instruction.word_offset);
                }
                PlanOperation::Send {
                    encoding, controls, ..
                } if controls.iter().any(pointer) => {
                    receives.push(
                        instruction.word_offset + u32::from(encoding == SendEncoding::PicPair),
                    );
                }
                PlanOperation::WriteBase {
                    incoming: false,
                    register,
                } => bases.push(OutgoingBaseWrite {
                    word_offset: instruction.word_offset,
                    register,
                }),
                _ => {}
            }
        }
        assert_eq!(self.receive_pointers, receives);
        assert_eq!(self.outgoing_bases, bases);
        assert_eq!(
            self.normalized_words(),
            normalized_exchange_address_words(&self.words)
        );
        for (index, site) in self.sends.iter().enumerate() {
            for address in [0, 0x54320, 0xffffc, 0x1ffff8, 3, u32::MAX] {
                let mut word = self.words[site.word_offset as usize];
                let expected =
                    crate::exchange::program::patch_sender_instruction(&mut word, address)
                        .map(|()| word);
                assert_eq!(self.relocated_send(index, address), expected);
            }
        }
    }
}
