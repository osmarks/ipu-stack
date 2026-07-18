use crate::Result;
use ipu_compiler::{LoweredTileProgram, LoweredTileStep};
use std::collections::BTreeMap;

const INCOMING_DBASE: u8 = 0xa4;
const INCOMING_DCOUNT: u8 = 0xa6;
const INCOMING_SBASE: u8 = 0xa7;

pub(crate) const WORKER_BARRIER: &str = "ipu_stack_static_worker_barrier";
pub(crate) const COMPLETE: &str = "ipu_stack_static_complete";
pub(crate) const COPY_U32: &str = "ipu_stack_static_copy_u32";

#[derive(Clone, Copy)]
pub(crate) struct HostCopy {
    pub source: u32,
    pub destination: u32,
    pub words: u32,
}

pub(crate) fn emit(
    program: &LoweredTileProgram,
    base: u32,
    symbols: &BTreeMap<String, u32>,
    plan_addresses: &[u32],
    host_inputs: &[(u32, Option<HostCopy>)],
    host_outputs: &[(u32, Option<HostCopy>)],
) -> Result<Vec<u8>> {
    let mut code = TileCode::new(base);
    let worker_barrier = symbol(symbols, WORKER_BARRIER)?;
    for &(address, copy) in host_inputs {
        code.call(address, 10)?;
        if let Some(copy) = copy {
            emit_copy(&mut code, symbols, copy)?;
        }
    }
    let mut plan_index = 0usize;
    for step in &program.steps {
        match step {
            LoweredTileStep::Exchange { row, .. } => {
                code.instruction(ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
                let active = row.first() != Some(&ipu_exchange::SANS_INACTIVE_INSTRUCTION);
                if active {
                    code.call(worker_barrier, 7)?;
                    code.put_special(INCOMING_SBASE, 15)?;
                    code.put_special(INCOMING_DBASE, 15)?;
                    code.setzi(8, 1)?;
                    code.put_special(INCOMING_DCOUNT, 8)?;
                }
                let target = plan_addresses
                    .get(plan_index)
                    .copied()
                    .ok_or("missing exchange plan address")?;
                plan_index += 1;
                code.call(target, 10)?;
            }
            LoweredTileStep::Compute(command) => {
                if command.input_addresses.len() != 2 {
                    return Err(format!(
                        "kernel {} on tile {} has {} inputs; the current ABI requires two",
                        command.specialization.operation,
                        program.tile,
                        command.input_addresses.len()
                    )
                    .into());
                }
                let kernel = symbol(
                    symbols,
                    &format!("ipu_stack_{}", command.specialization.operation),
                )?;
                code.setzi(2, command.output_address)?;
                code.setzi(3, command.input_addresses[0])?;
                code.setzi(4, command.input_addresses[1])?;
                code.call(kernel, 10)?;
            }
        }
    }
    if plan_index != plan_addresses.len() {
        return Err("unused exchange plan address".into());
    }
    for &(address, copy) in host_outputs {
        if let Some(copy) = copy {
            emit_copy(&mut code, symbols, copy)?;
        }
        code.call(address, 10)?;
    }
    code.jump(symbol(symbols, COMPLETE)?)?;
    Ok(code.words.into_iter().flat_map(u32::to_le_bytes).collect())
}

fn emit_copy(code: &mut TileCode, symbols: &BTreeMap<String, u32>, copy: HostCopy) -> Result<()> {
    code.setzi(2, copy.destination)?;
    code.setzi(3, copy.source)?;
    code.setzi(4, copy.words)?;
    code.call(symbol(symbols, COPY_U32)?, 10)
}

fn symbol(symbols: &BTreeMap<String, u32>, name: &str) -> Result<u32> {
    symbols
        .get(name)
        .copied()
        .ok_or_else(|| format!("static runtime has no {name} symbol").into())
}

struct TileCode {
    base: u32,
    words: Vec<u32>,
}

impl TileCode {
    fn new(base: u32) -> Self {
        Self {
            base,
            words: Vec::new(),
        }
    }

    fn address_after(&self, words: usize) -> Result<u32> {
        let words = self
            .words
            .len()
            .checked_add(words)
            .ok_or("tile program size overflow")?;
        self.base
            .checked_add(
                u32::try_from(words)?
                    .checked_mul(4)
                    .ok_or("tile program size overflow")?,
            )
            .ok_or_else(|| "tile program address overflow".into())
    }

    fn setzi(&mut self, register: u8, immediate: u32) -> Result<()> {
        self.words
            .push(ipu_exchange::encode_setzi_m(register, immediate)?);
        Ok(())
    }

    fn instruction(&mut self, instruction: u32) {
        self.words.push(instruction);
    }

    fn put_special(&mut self, special: u8, register: u8) -> Result<()> {
        self.words
            .push(ipu_exchange::encode_put_special_m(special, register)?);
        Ok(())
    }

    fn call(&mut self, target: u32, return_register: u8) -> Result<()> {
        let return_address = self.address_after(3)?;
        self.setzi(return_register, return_address)?;
        self.setzi(0, target)?;
        self.words.push(ipu_exchange::encode_br_m(0)?);
        Ok(())
    }

    fn jump(&mut self, target: u32) -> Result<()> {
        self.setzi(0, target)?;
        self.words.push(ipu_exchange::encode_br_m(0)?);
        Ok(())
    }
}
