//! Supervisor instruction encodings, including checked operand ranges.
//! Exchange sequencing and measured control-write timings live in ipu-exchange.
include!(concat!(env!("OUT_DIR"), "/instruction.rs"));
pub const SYNC_HOST_INSTRUCTION: u32 = sync(15);
pub const SYNC_ALL_INSTRUCTION: u32 = sync(7);
pub const SYNC_SUPERVISOR_INSTRUCTION: u32 = sync(3);
pub const SYNC_ANS_INSTRUCTION: u32 = sync(1);
pub const SYNC_RECEIVE_INSTRUCTION: u32 = sync(0);
pub const SANS_INACTIVE_INSTRUCTION: u32 = sans(0);
pub const RETURN_M10_INSTRUCTION: u32 = br_m(10);

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid supervisor instruction: {0}")]
pub struct InstructionError(pub &'static str);

pub const fn sans(selector: u8) -> u32 {
    SANS_OPCODE | selector as u32
}

pub const fn sync(selector: u8) -> u32 {
    SYNC_OPCODE | selector as u32
}

pub const fn br_m(register: u8) -> u32 {
    BR_M_OPCODE | ((register as u32) << 20)
}

pub fn encode_br_m(register: u8) -> Result<u32, InstructionError> {
    if register >= 16 {
        return Err(InstructionError("branch register"));
    }
    Ok(br_m(register))
}

pub fn encode_call_m_immediate(
    return_register: u8,
    target_address: u32,
) -> Result<u32, InstructionError> {
    if return_register >= 16 || target_address & 3 != 0 || target_address >= 1 << 21 {
        return Err(InstructionError("call operand"));
    }
    Ok(CALL_M_IMMEDIATE_OPCODE | (u32::from(return_register) << 20) | (target_address >> 2))
}

pub fn encode_setzi_m(register: u8, immediate: u32) -> Result<u32, InstructionError> {
    if register >= 16 || immediate >= 1 << 20 {
        return Err(InstructionError("setzi operand"));
    }
    Ok(setzi_m(register, immediate))
}

pub fn encode_put_special_m(special: u8, register: u8) -> Result<u32, InstructionError> {
    if register >= 16 {
        return Err(InstructionError("put source register"));
    }
    Ok(PUT_SPECIAL_M_OPCODE | (u32::from(register) << 20) | u32::from(special))
}

pub fn encode_ld32_m_immediate(
    destination: u8,
    base: u8,
    delta: u8,
    word_offset: u16,
) -> Result<u32, InstructionError> {
    if destination >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(InstructionError("ld32 operand"));
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
) -> Result<u32, InstructionError> {
    if source >= 16 || base >= 16 || delta >= 16 || word_offset >= 1 << 12 {
        return Err(InstructionError("st32 operand"));
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
) -> Result<u32, InstructionError> {
    let immediate =
        i16::try_from(immediate).map_err(|_| InstructionError("add immediate operand"))?;
    if destination >= 16 || source >= 16 {
        return Err(InstructionError("add register operand"));
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
) -> Result<u32, InstructionError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(InstructionError("and operand"));
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
) -> Result<u32, InstructionError> {
    if destination >= 16 || source >= 16 || immediate >= 1 << 12 {
        return Err(InstructionError("shift-left operand"));
    }
    Ok(SHL_M_IMMEDIATE_OPCODE
        | (u32::from(source) << 20)
        | (u32::from(destination) << 16)
        | u32::from(immediate))
}

pub fn encode_brz_m_immediate(register: u8, target_address: u32) -> Result<u32, InstructionError> {
    if register >= 16 || target_address & 3 != 0 || target_address >= 1 << 21 {
        return Err(InstructionError("brz operand"));
    }
    Ok(BRZ_M_IMMEDIATE_OPCODE | (u32::from(register) << 20) | (target_address >> 2))
}

/// Processor delay immediate. The encoded advance is immediate + 1 cycles.
/// Use encode_delay_m for checked cycle counts.
pub const fn encode_delay_immediate(immediate: u32) -> u32 {
    DELAY_OPCODE | (immediate & (MAX_PROCESSOR_DELAY_CYCLES - 1))
}

/// Encodes a processor delay of `cycles` cycles.
pub fn encode_delay_m(cycles: u32) -> Result<u32, InstructionError> {
    if !(1..=MAX_PROCESSOR_DELAY_CYCLES).contains(&cycles) {
        return Err(InstructionError("processor delay range"));
    }
    Ok(encode_delay_immediate(cycles - 1))
}

pub const fn setzi_m(register: u8, immediate: u32) -> u32 {
    SETZI_M_OPCODE | ((register as u32) << 20) | immediate
}

pub const fn put_special_from_m8(register: u8) -> u32 {
    PUT_SPECIAL_M_OPCODE | (8 << 20) | register as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_instruction_encoders_preserve_operands_and_reject_overflow() {
        let setzi = encode_setzi_m(15, (1 << 20) - 1).unwrap();
        assert_eq!((setzi >> 20) & 0xf, 15);
        assert_eq!(setzi & ((1 << 20) - 1), (1 << 20) - 1);

        let put = encode_put_special_m(0xa6, 8).unwrap();
        assert_eq!((put >> 20) & 0xf, 8);
        assert_eq!(put & 0xff, 0xa6);
        assert_eq!((encode_br_m(10).unwrap() >> 20) & 0xf, 10);

        let call = encode_call_m_immediate(10, 0x4c100).unwrap();
        assert_eq!((call >> 20) & 0xf, 10);
        assert_eq!((call & 0x7ffff) << 2, 0x4c100);

        assert!(encode_setzi_m(16, 0).is_err());
        assert!(encode_setzi_m(0, 1 << 20).is_err());
        assert!(encode_put_special_m(0, 16).is_err());
        assert!(encode_br_m(16).is_err());
        assert!(encode_call_m_immediate(16, 0).is_err());
        assert!(encode_call_m_immediate(0, 2).is_err());
        assert!(encode_call_m_immediate(0, 1 << 21).is_err());

        assert_eq!(encode_ld32_m_immediate(8, 11, 15, 1).unwrap(), 0x01b8_f001);
        assert_eq!(encode_st32_m_immediate(2, 11, 15, 0).unwrap(), 0x4fb2_f000);
        assert_eq!(encode_add_m_immediate(11, 11, -32).unwrap(), 0x22bb_ffe0);
        assert_eq!(encode_and_m_immediate(0, 8, 1).unwrap(), 0x4280_0001);
        assert_eq!(encode_shl_m_immediate(10, 7, 2).unwrap(), 0x427a_a002);
        assert!(encode_shl_m_immediate(0, 0, 1 << 12).is_err());
        assert_eq!(encode_brz_m_immediate(0, 0x4c100).unwrap(), 0x1301_3040);
    }
}
