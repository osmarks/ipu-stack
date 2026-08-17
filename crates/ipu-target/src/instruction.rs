//! IPU21 supervisor instruction encoders used by generated tile programs.

use crate::exchange::ExchangeError;

pub(crate) const OPCODE_MASK: u32 = 0xfc00_0000;
pub(crate) const LONG_OPCODE_MASK: u32 = 0xf800_0000;
pub(crate) const DELAY_OPCODE_MASK: u32 = 0xfff8_0000;
pub(crate) const DELAY_OPCODE: u32 = 0x40a0_0000;
pub(crate) const DELAY_PIC_OPCODE: u32 = 0x6000_0000;
pub(crate) const DELAY_XPIC_OPCODE: u32 = 0x6400_0000;
pub(crate) const PIC_RECEIVE_ADDRESS_MASK: u32 = 0x3ffff;
pub(crate) const SEND_OPCODE: u32 = 0x7800_0000;
pub(crate) const SEND_ADDRESS_MASK: u32 = 0x001f_fff8;
pub(crate) const SEND_OFF_OPCODE: u32 = 0x7000_0000;
// See docs/EXCHANGE_INSTRUCTION_REFERENCE.md. SENDPICP is an aligned two-word
// supervisor instruction whose following word is inline PIC/XPIC payload.
pub(crate) const SEND_PIC_OPCODE: u32 = 0x7010_0000;
pub(crate) const SEND_PICP_OPCODE: u32 = 0xf000_0000;
pub(crate) const SEND_CONTROL_OPCODE_MASK: u32 = 0xf810_0000;
pub(crate) const SEND_PICP_OPCODE_MASK: u32 = 0xf000_0000;
pub(crate) const SEND_COUNT_MASK: u32 = 0x07e0_0000;

pub(crate) const SYNC_OPCODE: u32 = 0x4180_0000;
const SANS_OPCODE: u32 = 0x40c0_0000;
const BR_M_OPCODE: u32 = 0x4300_0000;
const CALL_M_IMMEDIATE_OPCODE: u32 = 0x1800_0000;
const SETZI_M_OPCODE: u32 = 0x1900_0000;
const PUT_SPECIAL_M_OPCODE: u32 = 0x4300_8000;
const LD32_M_IMMEDIATE_OPCODE: u32 = 0x0100_0000;
const ST32_M_IMMEDIATE_OPCODE: u32 = 0x4f00_0000;
const ADD_M_IMMEDIATE_OPCODE: u32 = 0x2200_0000;
const AND_M_IMMEDIATE_OPCODE: u32 = 0x4200_0000;
const SHL_M_IMMEDIATE_OPCODE: u32 = 0x4200_a000;
const BRZ_M_IMMEDIATE_OPCODE: u32 = 0x1300_0000;

/// Largest delay encodable by one processor or exchange delay instruction.
pub const MAX_PLAN_OFFSET_CYCLES: u32 = 0x8_0000;

pub const SANS_INACTIVE_INSTRUCTION: u32 = sans(0);
pub const SYNC_RECEIVE_INSTRUCTION: u32 = sync(0);
pub const SYNC_ANS_INSTRUCTION: u32 = sync(1);
pub const SYNC_SUPERVISOR_INSTRUCTION: u32 = sync(3);
pub const SYNC_ALL_INSTRUCTION: u32 = sync(7);
pub const SYNC_HOST_INSTRUCTION: u32 = sync(15);
pub const RETURN_M10_INSTRUCTION: u32 = br_m(10);

pub const fn sans(selector: u8) -> u32 {
    SANS_OPCODE | selector as u32
}

pub const fn sync(selector: u8) -> u32 {
    SYNC_OPCODE | selector as u32
}

pub const fn br_m(register: u8) -> u32 {
    BR_M_OPCODE | ((register as u32) << 20)
}

pub fn encode_br_m(register: u8) -> Result<u32, ExchangeError> {
    if register >= 16 {
        return Err(ExchangeError::Schedule("branch register"));
    }
    Ok(br_m(register))
}

pub fn encode_call_m_immediate(
    return_register: u8,
    target_address: u32,
) -> Result<u32, ExchangeError> {
    if return_register >= 16 || target_address & 0b11 != 0 || target_address >= 1 << 21 {
        return Err(ExchangeError::Schedule("call operand"));
    }
    Ok(CALL_M_IMMEDIATE_OPCODE | (u32::from(return_register) << 20) | (target_address >> 2))
}

pub fn encode_setzi_m(register: u8, immediate: u32) -> Result<u32, ExchangeError> {
    if register >= 16 || immediate >= 1 << 20 {
        return Err(ExchangeError::Schedule("setzi operand"));
    }
    Ok(setzi_m(register, immediate))
}

pub fn encode_put_special_m(special: u8, register: u8) -> Result<u32, ExchangeError> {
    if register >= 16 {
        return Err(ExchangeError::Schedule("put source register"));
    }
    Ok(PUT_SPECIAL_M_OPCODE | (u32::from(register) << 20) | u32::from(special))
}

pub fn encode_ld32_m_immediate(
    destination: u8,
    base: u8,
    delta: u8,
    word_offset: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(ExchangeError::Schedule("ld32 operand"));
    }
    Ok(LD32_M_IMMEDIATE_OPCODE
        | (u32::from(base) << 20)
        | (u32::from(destination) << 16)
        | (u32::from(delta) << 12)
        | u32::from(word_offset))
}

pub fn encode_st32_m_immediate(
    source: u8,
    base: u8,
    delta: u8,
    word_offset: u16,
) -> Result<u32, ExchangeError> {
    if source >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(ExchangeError::Schedule("st32 operand"));
    }
    Ok(ST32_M_IMMEDIATE_OPCODE
        | (u32::from(base) << 20)
        | (u32::from(source) << 16)
        | (u32::from(delta) << 12)
        | u32::from(word_offset))
}

pub fn encode_add_m_immediate(
    destination: u8,
    source: u8,
    immediate: i32,
) -> Result<u32, ExchangeError> {
    let immediate =
        i16::try_from(immediate).map_err(|_| ExchangeError::Schedule("add immediate operand"))?;
    if destination >= 16 || source >= 16 {
        return Err(ExchangeError::Schedule("add register operand"));
    }
    Ok(ADD_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate as u16))
}

pub fn encode_and_m_immediate(
    destination: u8,
    source: u8,
    immediate: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(ExchangeError::Schedule("and operand"));
    }
    Ok(AND_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate))
}

pub fn encode_shl_m_immediate(
    destination: u8,
    source: u8,
    immediate: u16,
) -> Result<u32, ExchangeError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(ExchangeError::Schedule("shift-left operand"));
    }
    Ok(SHL_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate))
}

pub fn encode_brz_m_immediate(register: u8, target_address: u32) -> Result<u32, ExchangeError> {
    if register >= 16 || target_address & 0b11 != 0 || target_address >= 1 << 21 {
        return Err(ExchangeError::Schedule("brz operand"));
    }
    Ok(BRZ_M_IMMEDIATE_OPCODE | (u32::from(register) << 20) | (target_address >> 2))
}

/// Encodes a processor delay of `cycles` cycles.
pub fn encode_delay_m(cycles: u32) -> Result<u32, ExchangeError> {
    if !(1..=MAX_PLAN_OFFSET_CYCLES).contains(&cycles) {
        return Err(ExchangeError::Schedule("processor delay range"));
    }
    Ok(delay(cycles - 1))
}

pub(crate) const fn setzi_m(register: u8, immediate: u32) -> u32 {
    SETZI_M_OPCODE | ((register as u32) << 20) | immediate
}

pub(crate) const fn put_special_from_m8(register: u8) -> u32 {
    PUT_SPECIAL_M_OPCODE | (8 << 20) | register as u32
}

pub const fn encode_exchange_delay(cycles: u32) -> u32 {
    DELAY_OPCODE | (cycles & 0x7ffff)
}

pub const fn encode_exchange_delay_pic(a: u32, b: u32, c: u32) -> u32 {
    DELAY_PIC_OPCODE | ((a << 19) & 0x03f8_0000) | ((b << 18) & 0x0004_0000) | (c & 0x3ffff)
}

pub const fn encode_exchange_delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    DELAY_XPIC_OPCODE | ((a << 14) & 0x03ff_c000) | ((b << 13) & 0x0000_2000) | (c & 0x1fff)
}

pub(crate) const fn delay(cycles: u32) -> u32 {
    encode_exchange_delay(cycles)
}

pub(crate) const fn delay_pic(a: u32, b: u32, c: u32) -> u32 {
    encode_exchange_delay_pic(a, b, c)
}

pub(crate) const fn delay_xpic(a: u32, b: u32, c: u32) -> u32 {
    encode_exchange_delay_xpic(a, b, c)
}

pub fn encode_send(
    count_minus_one: u32,
    direction: u32,
    base_word: u32,
) -> Result<u32, ExchangeError> {
    if count_minus_one > 63 || direction > 7 || base_word > 0x3_ffff {
        return Err(ExchangeError::Schedule("send instruction operand"));
    }
    Ok(SEND_OPCODE
        | ((count_minus_one << 21) & SEND_COUNT_MASK)
        | ((base_word << 3) & SEND_ADDRESS_MASK)
        | direction)
}

pub(crate) const fn send_off(count_minus_one: u32, direction: u32, base_word: u32) -> u32 {
    SEND_OFF_OPCODE
        | ((count_minus_one << 21) & SEND_COUNT_MASK)
        | (((count_minus_one >> 6) << 14) & 0x000f_c000)
        | ((base_word << 3) & 0x0000_3ff8)
        | (direction & 7)
}

pub(crate) fn is_send_control(instruction: u32) -> bool {
    instruction & SEND_CONTROL_OPCODE_MASK == SEND_PIC_OPCODE
}

pub(crate) fn is_send_control_pair(instruction: u32) -> bool {
    instruction & SEND_PICP_OPCODE_MASK == SEND_PICP_OPCODE
}

pub(crate) fn is_send_off(instruction: u32) -> bool {
    instruction & SEND_CONTROL_OPCODE_MASK == SEND_OFF_OPCODE
}

pub(crate) fn is_payload_send(instruction: u32) -> bool {
    instruction & LONG_OPCODE_MASK == SEND_OPCODE || is_send_off(instruction)
}
