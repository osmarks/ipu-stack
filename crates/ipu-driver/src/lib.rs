use ipu_package::{Application, HostCall, HostExchange, TILE_MEMORY_BASE};
use object::{Object, ObjectSegment};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::io;
use std::os::fd::RawFd;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{Ordering, fence};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace};

pub const CONFIG_BAR_SIZE: usize = 0x80000;
pub const TILE_MEMORY_SIZE: usize = 624 * 1024;
// The secondary loader installs framed application payload at the SDK image's
// launch slot. Applications reserve that word and enter at the following word.
pub const APPLICATION_LOAD_BASE: u32 = TILE_MEMORY_BASE + 0x10;
pub const HSP_MARK_MASK: u32 = 0xffff;
// The secondary loader consumes 23 tile batches before handing HSP ownership
// to the resident host-exchange program.
pub const HOST_EXCHANGE_HANDOFF_MARK: u32 = 23;
pub const TILES_PER_BATCH: usize = 64;
pub const FRAME_SIZE: usize = 1024;
pub const FRAME_HEADER_SIZE: usize = 16;
pub const FRAME_TRAILER_SIZE: usize = 16;
pub const FRAME_PAYLOAD_SIZE: usize = FRAME_SIZE - FRAME_HEADER_SIZE - FRAME_TRAILER_SIZE;
pub const DESCRIPTOR_AREA_SIZE: usize = 4096;
pub const TRANSPORT_SIZE: usize = 0x2842000;
// The Graphcore secondary loader does not acknowledge a one-frame application.
// Pad transport payloads to the smallest established working envelope.
pub const SECONDARY_LOADER_MIN_PAYLOAD_SIZE: usize = 0x4134;
const TILE_DEBUG_BASE: u32 = 0x30000;
const TILE_DEBUG_TILE_STRIDE: u32 = 0x40;
const TILE_DEBUG_EXCEPTION_STATE: u32 = 5;
const TILE_DEBUG_REGISTER_STRIDE: u32 = 4;
const TILE_DEBUG_BREAK_ON_SYNC: u32 = 1 << 5;
const TDI_CONTEXT_STATUS: u32 = 0;
const TDI_RUN_BREAK: u32 = 1;
const TDI_INSTRUCTION: u32 = 3;
const TDI_INSTRUCTION_OWNER: u32 = 4;
const TDI_EXCEPTION_CLEAR: u32 = 6;
const TDI_DATA: u32 = 7;
const TDI_STATUS: u32 = 8;
const TDI_STATUS_CLEAR: u32 = 9;

mod tdi_instruction {
    // IPU21 diagnostic instructions, named by their Tile Vertex ISA assembly.
    pub const GET_M0_PC: u32 = 0x4100_0000;
    pub const GET_M0_WSR: u32 = 0x4100_0001;
    pub const GET_M1_DEBUG_DATA: u32 = 0x4101_0070;
    pub const LOAD_M0_FROM_M1: u32 = 0x01f0_1000;
    pub const PUT_DEBUG_DATA_M0: u32 = 0x4300_8070;

    pub fn put_debug_data_m(register: u32) -> Option<u32> {
        (register < 16).then_some(PUT_DEBUG_DATA_M0 | (register << 20))
    }
}

/// IPU21 `$SSR.ETYPE` / `$WSR.ETYPE` values from the Tile Vertex ISA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TileException {
    None = 0,
    RetirementBreak = 1,
    PatchedBreak0 = 2,
    PatchedBreak1 = 3,
    BreakOnSync = 4,
    FloatingPoint = 5,
    MemoryConflict = 6,
    InvalidExchangeConfiguration = 7,
    InvalidMemoryAddress = 8,
    InvalidOperand = 9,
    InvalidProgramCounter = 10,
    DataBreak = 11,
    InvalidInstruction = 12,
    ExchangeError = 13,
    MemoryError = 14,
    InstructionBreak = 15,
}

impl TileException {
    pub const ALL: [Self; 16] = [
        Self::None,
        Self::RetirementBreak,
        Self::PatchedBreak0,
        Self::PatchedBreak1,
        Self::BreakOnSync,
        Self::FloatingPoint,
        Self::MemoryConflict,
        Self::InvalidExchangeConfiguration,
        Self::InvalidMemoryAddress,
        Self::InvalidOperand,
        Self::InvalidProgramCounter,
        Self::DataBreak,
        Self::InvalidInstruction,
        Self::ExchangeError,
        Self::MemoryError,
        Self::InstructionBreak,
    ];

    pub const fn architecture_name(self) -> &'static str {
        match self {
            Self::None => "TEXCPT_NONE",
            Self::RetirementBreak => "TEXCPT_RBRK",
            Self::PatchedBreak0 => "TEXCPT_PBRK0",
            Self::PatchedBreak1 => "TEXCPT_PBRK1",
            Self::BreakOnSync => "TEXCPT_BOS",
            Self::FloatingPoint => "TEXCPT_FP",
            Self::MemoryConflict => "TEXCPT_CONFLICT",
            Self::InvalidExchangeConfiguration => "TEXCPT_EXCONF",
            Self::InvalidMemoryAddress => "TEXCPT_INVALID_ADDR",
            Self::InvalidOperand => "TEXCPT_INVALID_OP",
            Self::InvalidProgramCounter => "TEXCPT_INVALID_PC",
            Self::DataBreak => "TEXCPT_DBRK",
            Self::InvalidInstruction => "TEXCPT_INVALID_INSTR",
            Self::ExchangeError => "TEXCPT_EXERR",
            Self::MemoryError => "TEXCPT_MEMERR",
            Self::InstructionBreak => "TEXCPT_IBRK",
        }
    }

    pub const fn is_debug(self) -> bool {
        matches!(
            self,
            Self::RetirementBreak
                | Self::PatchedBreak0
                | Self::PatchedBreak1
                | Self::BreakOnSync
                | Self::DataBreak
                | Self::InstructionBreak
        )
    }

    pub const fn from_status(status: u32) -> Self {
        // ETYPE occupies bits 4..=7 in both supervisor SSR and worker WSR.
        Self::ALL[((status >> 4) & 0xf) as usize]
    }
}

impl TryFrom<u8> for TileException {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::ALL.get(value as usize).copied().ok_or(value)
    }
}

impl std::fmt::Display for TileException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.architecture_name())
    }
}

pub mod pci {
    pub const CCSR: u32 = 0x2000;
    pub const AUTOLD_CSR: u32 = 0x2068;
    pub const AUTOLD_TARGET: u32 = 0x206c;
    pub const AUTOLD_DATA: u32 = 0x2070;
    pub const HSP_GS1_CONTROL: u32 = 0x302c;
    pub const HSP_GS2_CONTROL: u32 = 0x3030;
    pub const EXCHANGE_WINDOW_BASE: u32 = 0x3044;
    pub const AUTOLD_GO: u32 = 1 << 8;
    pub const AUTOLD_DATA_PRESENT: u32 = 1 << 20;
    pub const AUTOLD_CURRENT_TILE_SHIFT: u32 = 9;
    pub const AUTOLD_CURRENT_TILE_MASK: u32 = 0x7ff;
    pub const AUTOLD_ADDRESS_SHIFT: u32 = 10;
    pub const AUTOLD_ZONE_SHIFT: u32 = 20;
    pub const AUTOLD_LOAD_POINTER_MASK: u32 = 0x3f;
    pub const EXCHANGE_WINDOW_HEXOPT: u32 = 1 << 11;
    pub const NO_SNOOP: u32 = 1 << 1;
    pub const EXCHANGE_LINK_ECRC_ENABLE: u32 = 1 << 10;
}

mod xb {
    pub const INSTANCES_PER_INTERFACE: usize = 8;
    pub const INSTANCE_COUNT: usize = 16;
    const PRIMARY_DCXCR_BASE: u32 = 0x5404;
    const SECONDARY_DCXCR_BASE: u32 = 0x1b804;
    const INSTANCE_STRIDE: u32 = 0x400;
    const REQUEST_SHIFTS: [u32; 4] = [3, 9, 15, 21];

    pub fn address(instance: usize) -> u32 {
        let base = if instance < INSTANCES_PER_INTERFACE {
            PRIMARY_DCXCR_BASE
        } else {
            SECONDARY_DCXCR_BASE - INSTANCES_PER_INTERFACE as u32 * INSTANCE_STRIDE
        };
        base + instance as u32 * INSTANCE_STRIDE
    }

    pub fn control(request_id: u32) -> u32 {
        let mut value = 2;
        for shift in REQUEST_SHIFTS {
            value |= (request_id & 0x3f) << shift;
        }
        value
    }
}

const IOCTL_MAGIC: u32 = 0xed;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const fn ioctl_code(direction: u32, number: u32, size: u32) -> libc::c_ulong {
    ((direction << 30) | (size << 16) | (IOCTL_MAGIC << 8) | number) as libc::c_ulong
}
const USER_ATTACH: libc::c_ulong = ioctl_code(IOC_READ | IOC_WRITE, 10, 8);
const BUFFER_ATTACH: libc::c_ulong = ioctl_code(IOC_READ | IOC_WRITE, 11, 8);
const BUFFER_DETACH: libc::c_ulong = ioctl_code(IOC_READ | IOC_WRITE, 12, 8);
const BUFFER_DETACH_ALL: libc::c_ulong = ioctl_code(0, 18, 0);
const SET_IPU_ID: libc::c_ulong = ioctl_code(IOC_READ | IOC_WRITE, 14, 8);
const MAILBOX_WRITE_READ: libc::c_ulong = ioctl_code(IOC_READ | IOC_WRITE, 31, 8);
const STOP_MONITORING: libc::c_ulong = ioctl_code(0, 34, 0);
const START_MONITORING: libc::c_ulong = ioctl_code(0, 35, 0);
const ACCUMULATE_ERRORS: libc::c_ulong = ioctl_code(0, 36, 0);
const SET_HEXOPT_IDENTITY_TABLE: libc::c_ulong = ioctl_code(0, 44, 0);
const RESET_DEVICE: libc::c_ulong = ioctl_code(IOC_WRITE, 45, 4);
const SET_SECONDARY_IPU_ID: libc::c_ulong = ioctl_code(IOC_WRITE, 47, 8);

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("{operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("invalid device data: {0}")]
    Invalid(String),
    #[error("device timeout: {0}")]
    Timeout(String),
    #[error("package error: {0}")]
    Package(#[from] ipu_package::PackageError),
    #[error("ELF error: {0}")]
    Elf(#[from] object::Error),
}

fn io_error(operation: &'static str) -> DriverError {
    DriverError::Io {
        operation,
        source: io::Error::last_os_error(),
    }
}

#[repr(C)]
struct AttachBufferData {
    index: u32,
    size: u32,
    address: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MailboxMessage {
    words: [u32; 5],
}

#[repr(C)]
struct MailboxArgument {
    sent: *const MailboxMessage,
    received: *mut MailboxMessage,
}

pub struct Device {
    fd: RawFd,
    config: *mut u8,
}

impl Device {
    pub fn open(path: &str) -> Result<Self, DriverError> {
        info!(path, "opening IPU device");
        let path = CString::new(path).map_err(|_| DriverError::Invalid("NUL in path".into()))?;
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(io_error("open IPU device"));
        }
        let config = unsafe {
            libc::mmap(
                ptr::null_mut(),
                CONFIG_BAR_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if config == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(io_error("map configuration BAR"));
        }
        let device = Self {
            fd,
            config: config.cast(),
        };
        device.ioctl_value(USER_ATTACH, 0, "attach IPU")?;
        device.ioctl_value(SET_IPU_ID, 0, "set IPU ID")?;
        info!(path = %path.to_string_lossy(), "IPU device attached");
        Ok(device)
    }

    pub fn read_config(&self, offset: u32) -> Result<u32, DriverError> {
        self.check_offset(offset)?;
        Ok(unsafe { ptr::read_volatile(self.config.add(offset as usize).cast::<u32>()) })
    }

    pub fn write_config(&self, offset: u32, value: u32) -> Result<(), DriverError> {
        self.check_offset(offset)?;
        unsafe { ptr::write_volatile(self.config.add(offset as usize).cast::<u32>(), value) };
        Ok(())
    }

    pub fn initialize(&self) -> Result<(), DriverError> {
        info!("initializing IPU device");
        self.ioctl_value(STOP_MONITORING, 0, "stop monitoring")?;
        let request = MailboxMessage {
            words: [0x4078_020e, 0, 0, 0, 0],
        };
        let response = self.mailbox(request)?;
        if response.words[0] & 0xffff_0000 != request.words[0] & 0xffff_0000 {
            return Err(DriverError::Invalid("unexpected ICU reset response".into()));
        }
        self.ioctl_value(START_MONITORING, 0, "start monitoring")?;
        self.ioctl_value(ACCUMULATE_ERRORS, 0, "accumulate errors")?;
        self.ioctl_value(RESET_DEVICE, 0, "notify reset")?;
        let id = 0u32;
        self.ioctl_pointer(
            SET_SECONDARY_IPU_ID,
            (&id as *const u32).cast_mut().cast(),
            "set secondary IPU ID",
        )?;
        info!("IPU device initialized");
        Ok(())
    }

    pub fn replay_configuration(&self, bytes: &[u8]) -> Result<(), DriverError> {
        if bytes.len() < 12 || &bytes[..8] != b"IPUCFG1\0" {
            return Err(DriverError::Invalid("bad configuration image".into()));
        }
        let count = read_u32(bytes, 8)? as usize;
        info!(register_writes = count, "replaying device configuration");
        if bytes.len() != 12 + count * 8 {
            return Err(DriverError::Invalid("bad configuration image size".into()));
        }
        for index in 0..count {
            self.write_config(
                read_u32(bytes, 12 + index * 8)?,
                read_u32(bytes, 16 + index * 8)?,
            )?;
        }
        Ok(())
    }

    pub fn wait_mark(
        &self,
        register: u32,
        expected: u32,
        timeout: Duration,
    ) -> Result<(), DriverError> {
        trace!(
            register = format_args!("0x{register:x}"),
            expected,
            timeout_ms = timeout.as_millis(),
            "waiting for HSP mark"
        );
        let deadline = Instant::now() + timeout;
        loop {
            let observed = self.read_config(register)? & HSP_MARK_MASK;
            if observed == expected {
                trace!(
                    register = format_args!("0x{register:x}"),
                    observed, "observed HSP mark"
                );
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(DriverError::Timeout(format!(
                    "HSP register 0x{register:x}: expected {expected}, observed {observed}"
                )));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    pub fn set_mark(&self, mark: u32) -> Result<(), DriverError> {
        if mark > HSP_MARK_MASK {
            return Err(DriverError::Invalid("HSP mark out of range".into()));
        }
        self.write_config(pci::HSP_GS1_CONTROL, mark)
    }

    pub fn set_hexopt_identity_table(&self) -> Result<(), DriverError> {
        self.ioctl_value(SET_HEXOPT_IDENTITY_TABLE, 0, "set HEXOPT identity table")
    }

    pub fn set_break_on_sync(&self, physical_tile: u16, enabled: bool) -> Result<(), DriverError> {
        let offset = TILE_DEBUG_BASE
            + u32::from(physical_tile) * TILE_DEBUG_TILE_STRIDE
            + TILE_DEBUG_EXCEPTION_STATE * TILE_DEBUG_REGISTER_STRIDE;
        let mut state = self.read_config(offset)?;
        if enabled {
            state |= TILE_DEBUG_BREAK_ON_SYNC;
        } else {
            state &= !TILE_DEBUG_BREAK_ON_SYNC;
        }
        self.write_config(offset, state)
    }

    pub fn read_tile_word(&self, physical_tile: u16, address: u32) -> Result<u32, DriverError> {
        if address & 3 != 0
            || !(TILE_MEMORY_BASE..=TILE_MEMORY_BASE + TILE_MEMORY_SIZE as u32 - 4)
                .contains(&address)
        {
            return Err(DriverError::Invalid(format!(
                "invalid tile memory address 0x{address:x}"
            )));
        }
        self.with_stopped_tile_context(physical_tile, 0, || {
            self.write_tile_debug(physical_tile, TDI_DATA, address)?;
            for instruction in [
                tdi_instruction::GET_M1_DEBUG_DATA,
                tdi_instruction::LOAD_M0_FROM_M1,
                tdi_instruction::PUT_DEBUG_DATA_M0,
            ] {
                self.execute_tile_instruction(physical_tile, 0, instruction)?;
            }
            self.read_tile_debug(physical_tile, TDI_DATA)
        })
    }

    pub fn read_tile_program_counter(
        &self,
        physical_tile: u16,
        context: u32,
    ) -> Result<u32, DriverError> {
        self.with_stopped_tile_context(physical_tile, context, || {
            self.execute_tile_instruction(physical_tile, context, tdi_instruction::GET_M0_PC)?;
            self.execute_tile_instruction(
                physical_tile,
                context,
                tdi_instruction::PUT_DEBUG_DATA_M0,
            )?;
            self.read_tile_debug(physical_tile, TDI_DATA)
        })
    }

    pub fn read_tile_worker_status(
        &self,
        physical_tile: u16,
        context: u32,
    ) -> Result<u32, DriverError> {
        if context == 0 {
            return Err(DriverError::Invalid(
                "worker status requires a worker context".into(),
            ));
        }
        self.with_stopped_tile_context(physical_tile, context, || {
            self.execute_tile_instruction(physical_tile, context, tdi_instruction::GET_M0_WSR)?;
            self.execute_tile_instruction(
                physical_tile,
                context,
                tdi_instruction::PUT_DEBUG_DATA_M0,
            )?;
            self.read_tile_debug(physical_tile, TDI_DATA)
        })
    }

    pub fn read_tile_m_register(
        &self,
        physical_tile: u16,
        context: u32,
        register: u32,
    ) -> Result<u32, DriverError> {
        let instruction = tdi_instruction::put_debug_data_m(register)
            .ok_or_else(|| DriverError::Invalid("M register index out of range".into()))?;
        self.with_stopped_tile_context(physical_tile, context, || {
            self.execute_tile_instruction(physical_tile, context, instruction)?;
            self.read_tile_debug(physical_tile, TDI_DATA)
        })
    }

    fn with_stopped_tile_context<T>(
        &self,
        physical_tile: u16,
        context: u32,
        operation: impl FnOnce() -> Result<T, DriverError>,
    ) -> Result<T, DriverError> {
        let context_bit = 1 << context;
        let old_run_break = self.read_tile_debug(physical_tile, TDI_RUN_BREAK)?;
        let initial_state = self.tile_context_state(physical_tile, context)?;
        let already_stopped = matches!(initial_state, 2 | 3);
        if !already_stopped {
            self.write_tile_debug(physical_tile, TDI_RUN_BREAK, old_run_break | context_bit)?;
            let deadline = Instant::now() + Duration::from_millis(100);
            while !matches!(self.tile_context_state(physical_tile, context)?, 2 | 3) {
                if Instant::now() >= deadline {
                    return Err(DriverError::Timeout("stopping tile context".into()));
                }
            }
        }
        let result = operation();
        if !already_stopped {
            self.write_tile_debug(physical_tile, TDI_RUN_BREAK, old_run_break)?;
            if old_run_break & context_bit == 0 {
                self.write_tile_debug(physical_tile, TDI_EXCEPTION_CLEAR, context_bit)?;
            }
        }
        result
    }

    pub fn tile_context_state(&self, physical_tile: u16, context: u32) -> Result<u32, DriverError> {
        Ok((self.read_tile_debug(physical_tile, TDI_CONTEXT_STATUS)? >> (context * 2)) & 3)
    }

    fn read_tile_debug(&self, physical_tile: u16, register: u32) -> Result<u32, DriverError> {
        self.read_config(
            TILE_DEBUG_BASE
                + u32::from(physical_tile) * TILE_DEBUG_TILE_STRIDE
                + register * TILE_DEBUG_REGISTER_STRIDE,
        )
    }

    fn write_tile_debug(
        &self,
        physical_tile: u16,
        register: u32,
        value: u32,
    ) -> Result<(), DriverError> {
        self.write_config(
            TILE_DEBUG_BASE
                + u32::from(physical_tile) * TILE_DEBUG_TILE_STRIDE
                + register * TILE_DEBUG_REGISTER_STRIDE,
            value,
        )
    }

    fn execute_tile_instruction(
        &self,
        physical_tile: u16,
        context: u32,
        instruction: u32,
    ) -> Result<(), DriverError> {
        const INVALID_OR_DOUBLE: u32 = 0x6;
        const BUSY: u32 = 0x8;
        self.write_tile_debug(physical_tile, TDI_STATUS_CLEAR, INVALID_OR_DOUBLE)?;
        self.write_tile_debug(physical_tile, TDI_INSTRUCTION_OWNER, context)?;
        self.write_tile_debug(physical_tile, TDI_INSTRUCTION, instruction)?;
        let deadline = Instant::now() + Duration::from_millis(100);
        loop {
            let status = self.read_tile_debug(physical_tile, TDI_STATUS)?;
            if status & BUSY == 0 {
                if status & (INVALID_OR_DOUBLE | 1) != 0 {
                    return Err(DriverError::Invalid(format!(
                        "TDI instruction failed with status 0x{status:x}"
                    )));
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(DriverError::Timeout("TDI instruction".into()));
            }
        }
    }

    fn wait_autoloader(&self, timeout: Duration) -> Result<(), DriverError> {
        let deadline = Instant::now() + timeout;
        while self.read_config(pci::AUTOLD_CSR)? & pci::AUTOLD_GO != 0 {
            if Instant::now() >= deadline {
                return Err(DriverError::Timeout("hardware autoloader".into()));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        Ok(())
    }

    pub fn attach_buffer(
        &self,
        index: u32,
        address: *mut u8,
        size: usize,
    ) -> Result<(), DriverError> {
        let size = u32::try_from(size)
            .map_err(|_| DriverError::Invalid("host buffer too large".into()))?;
        if size == 0 {
            return Err(DriverError::Invalid("empty host buffer".into()));
        }
        let mut data = AttachBufferData {
            index,
            size,
            address: address as usize as u64,
        };
        self.ioctl_pointer(
            BUFFER_ATTACH,
            (&mut data as *mut AttachBufferData).cast(),
            "attach host buffer",
        )
    }

    pub fn detach_buffer(&self, index: u32) {
        unsafe { libc::ioctl(self.fd, BUFFER_DETACH, index as libc::c_ulong) };
    }

    pub fn detach_all_buffers(&self) -> Result<(), DriverError> {
        self.ioctl_value(BUFFER_DETACH_ALL, 0, "detach all host buffers")
    }

    fn mailbox(&self, request: MailboxMessage) -> Result<MailboxMessage, DriverError> {
        let mut response = MailboxMessage::default();
        let mut argument = MailboxArgument {
            sent: &request,
            received: &mut response,
        };
        self.ioctl_pointer(
            MAILBOX_WRITE_READ,
            (&mut argument as *mut MailboxArgument).cast(),
            "ICU mailbox transaction",
        )?;
        Ok(response)
    }

    fn ioctl_value(
        &self,
        request: libc::c_ulong,
        value: libc::c_ulong,
        operation: &'static str,
    ) -> Result<(), DriverError> {
        if unsafe { libc::ioctl(self.fd, request, value) } != 0 {
            Err(io_error(operation))
        } else {
            Ok(())
        }
    }

    fn ioctl_pointer(
        &self,
        request: libc::c_ulong,
        value: *mut libc::c_void,
        operation: &'static str,
    ) -> Result<(), DriverError> {
        if unsafe { libc::ioctl(self.fd, request, value) } != 0 {
            Err(io_error(operation))
        } else {
            Ok(())
        }
    }

    fn check_offset(&self, offset: u32) -> Result<(), DriverError> {
        if offset & 3 != 0 || offset as usize > CONFIG_BAR_SIZE - 4 {
            Err(DriverError::Invalid(format!(
                "configuration offset 0x{offset:x}"
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.config.cast(), CONFIG_BAR_SIZE);
            libc::close(self.fd);
        }
    }
}

pub struct HostBuffer {
    data: *mut u8,
    size: usize,
}

impl HostBuffer {
    pub fn new(size: usize) -> Result<Self, DriverError> {
        let data = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if data == libc::MAP_FAILED {
            Err(io_error("allocate pinned host buffer"))
        } else {
            unsafe { ptr::write_bytes(data, 0, size) };
            Ok(Self {
                data: data.cast(),
                size,
            })
        }
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.size) }
    }

    pub fn attach(&self, device: &Device, index: u32) -> Result<(), DriverError> {
        device.attach_buffer(index, self.data, self.size)
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.data.cast(), self.size) };
    }
}

struct ExchangeBufferGuard<'a> {
    device: &'a Device,
    operational: [u32; xb::INSTANCE_COUNT],
    saved_ccsr: u32,
}

impl<'a> ExchangeBufferGuard<'a> {
    fn new(device: &'a Device) -> Result<Self, DriverError> {
        let saved_ccsr = device.read_config(pci::CCSR)?;
        let mut operational = [0; xb::INSTANCE_COUNT];
        for (instance, value) in operational.iter_mut().enumerate() {
            *value = xb::control((instance / xb::INSTANCES_PER_INTERFACE) as u32);
            device.write_config(xb::address(instance), xb::control(0x3f))?;
        }
        device.write_config(
            pci::CCSR,
            saved_ccsr & !(pci::NO_SNOOP | pci::EXCHANGE_LINK_ECRC_ENABLE),
        )?;
        let _ = device.read_config(pci::CCSR)?;
        Ok(Self {
            device,
            operational,
            saved_ccsr,
        })
    }

    fn restore_primary(&self) -> Result<(), DriverError> {
        for instance in 0..xb::INSTANCES_PER_INTERFACE {
            self.device
                .write_config(xb::address(instance), self.operational[instance])?;
        }
        Ok(())
    }

    fn restore_all(&self) -> Result<(), DriverError> {
        for (instance, value) in self.operational.iter().enumerate() {
            self.device.write_config(xb::address(instance), *value)?;
        }
        self.device.write_config(pci::CCSR, self.saved_ccsr)?;
        let _ = self.device.read_config(pci::CCSR)?;
        Ok(())
    }
}

impl Drop for ExchangeBufferGuard<'_> {
    fn drop(&mut self) {
        let _ = self.restore_all();
    }
}

pub struct Loader<'a> {
    device: &'a Device,
    bootloader: Vec<u8>,
}

struct BufferAttachment<'a> {
    device: &'a Device,
    index: u32,
}

impl<'a> BufferAttachment<'a> {
    fn new(
        device: &'a Device,
        index: u32,
        address: *mut u8,
        size: usize,
    ) -> Result<Self, DriverError> {
        device.attach_buffer(index, address, size)?;
        Ok(Self { device, index })
    }
}

impl Drop for BufferAttachment<'_> {
    fn drop(&mut self) {
        self.device.detach_buffer(self.index);
    }
}

impl<'a> Loader<'a> {
    pub fn new(device: &'a Device, bootloader_elf: &[u8]) -> Result<Self, DriverError> {
        let file = object::File::parse(bootloader_elf)?;
        let mut image = None;
        for segment in file.segments() {
            if segment.address() != u64::from(TILE_MEMORY_BASE) || segment.size() == 0 {
                continue;
            }
            if image.is_some() {
                return Err(DriverError::Invalid(
                    "multiple bootloader load segments".into(),
                ));
            }
            image = Some(segment.data()?.to_vec());
        }
        let mut bootloader =
            image.ok_or_else(|| DriverError::Invalid("missing bootloader segment".into()))?;
        if bootloader.len() > DESCRIPTOR_AREA_SIZE {
            return Err(DriverError::Invalid(
                "bootloader exceeds descriptor area".into(),
            ));
        }
        bootloader.resize(bootloader.len().div_ceil(FRAME_SIZE) * FRAME_SIZE, 0);
        Ok(Self { device, bootloader })
    }

    pub fn load(&self, app: &Application, final_mark: u32) -> Result<(), DriverError> {
        app.validate()?;
        let tile_count = app.tiles.len();
        info!(tile_count, final_mark, "loading application");
        if tile_count == 0
            || !tile_count.is_multiple_of(TILES_PER_BATCH)
            || tile_count - 1 > pci::AUTOLD_CURRENT_TILE_MASK as usize
        {
            return Err(DriverError::Invalid(
                "tile count is not complete loader batches".into(),
            ));
        }
        self.install_bootloader(tile_count)?;
        self.device
            .write_config(pci::EXCHANGE_WINDOW_BASE, pci::EXCHANGE_WINDOW_HEXOPT)?;
        let mut transport = HostBuffer::new(TRANSPORT_SIZE)?;
        let _attachment = BufferAttachment::new(self.device, 0, transport.data, transport.size)?;
        let guard = ExchangeBufferGuard::new(self.device)?;
        guard.restore_primary()?;

        for (batch, first) in (0..tile_count).step_by(TILES_PER_BATCH).enumerate() {
            let bytes = transport.bytes_mut();
            bytes.fill(0);
            let mut cursor = DESCRIPTOR_AREA_SIZE;
            for slot in 0..TILES_PER_BATCH {
                let tile = &app.tiles[first + slot];
                let physical = tile.physical_tile;
                if tile.segments.iter().map(|segment| segment.address).min()
                    != Some(APPLICATION_LOAD_BASE)
                {
                    return Err(DriverError::Invalid(format!(
                        "tile {physical} does not begin at secondary-loader address 0x{APPLICATION_LOAD_BASE:x}"
                    )));
                }
                let mut image = app.tile_image(physical)?;
                image.resize(image.len().max(SECONDARY_LOADER_MIN_PAYLOAD_SIZE), 0);
                let framed = frame_tile(physical, &image)?;
                if cursor + framed.len() > bytes.len() {
                    return Err(DriverError::Invalid("tile batch exceeds transport".into()));
                }
                bytes[slot * 8..slot * 8 + 4].copy_from_slice(&(cursor as u32).to_le_bytes());
                bytes[slot * 8 + 4..slot * 8 + 8]
                    .copy_from_slice(&((framed.len() / FRAME_SIZE) as u32).to_le_bytes());
                bytes[cursor..cursor + framed.len()].copy_from_slice(&framed);
                cursor += framed.len();
            }
            fence(Ordering::SeqCst);
            debug!(
                batch,
                first_tile = first,
                transport_bytes = cursor,
                "submitting application tile batch"
            );
            self.device.set_mark(1)?;
            self.device
                .wait_mark(pci::HSP_GS1_CONTROL, 0, Duration::from_secs(10))
                .map_err(|error| {
                    DriverError::Timeout(format!("bootloader batch {batch}: {error}"))
                })?;
        }

        transport.bytes_mut().fill(0xff);
        fence(Ordering::SeqCst);
        self.device
            .wait_mark(pci::HSP_GS1_CONTROL, 0, Duration::from_secs(10))
            .map_err(|error| DriverError::Timeout(format!("before execute sentinel: {error}")))?;
        self.device.set_mark(1)?;
        self.device
            .wait_mark(pci::HSP_GS1_CONTROL, 0, Duration::from_secs(10))
            .map_err(|error| DriverError::Timeout(format!("execute sentinel: {error}")))?;
        self.device.set_mark(if final_mark == 0 {
            (tile_count / TILES_PER_BATCH) as u32
        } else {
            final_mark
        })?;
        self.device
            .wait_mark(pci::HSP_GS1_CONTROL, 0, Duration::from_secs(10))
            .map_err(|error| DriverError::Timeout(format!("application initial sync: {error}")))?;
        guard.restore_all()?;
        info!(
            tile_count,
            "application loaded and initial synchronization completed"
        );
        Ok(())
    }

    fn install_bootloader(&self, tile_count: usize) -> Result<(), DriverError> {
        debug!(
            tile_count,
            bootloader_bytes = self.bootloader.len(),
            "installing secondary bootloader"
        );
        self.device.write_config(pci::AUTOLD_CSR, 0)?;
        for (index, chunk) in self.bootloader.chunks_exact(4).enumerate() {
            let word = [chunk[0], chunk[1], chunk[2], chunk[3]];
            self.device
                .write_config(pci::AUTOLD_DATA, u32::from_le_bytes(word))?;
            if index & 31 == 31 {
                let _ = self.device.read_config(pci::AUTOLD_CSR)?;
            }
        }
        let zone = 32 << pci::AUTOLD_ZONE_SHIFT;
        let kib = (self.bootloader.len() / 1024) as u32;
        let load_pointer = (self.bootloader.len() as u32 / 4 - 1) & pci::AUTOLD_LOAD_POINTER_MASK;
        self.device.write_config(
            pci::AUTOLD_TARGET,
            zone | (kib << pci::AUTOLD_ADDRESS_SHIFT),
        )?;
        self.device.write_config(
            pci::AUTOLD_CSR,
            pci::AUTOLD_DATA_PRESENT | pci::AUTOLD_GO | load_pointer,
        )?;
        self.device.wait_autoloader(Duration::from_secs(2))?;
        self.device.write_config(
            pci::AUTOLD_TARGET,
            zone | ((TILE_MEMORY_SIZE as u32 / 1024) << pci::AUTOLD_ADDRESS_SHIFT) | kib,
        )?;
        self.device.write_config(
            pci::AUTOLD_CSR,
            ((tile_count as u32 - 1) << pci::AUTOLD_CURRENT_TILE_SHIFT)
                | pci::AUTOLD_GO
                | load_pointer,
        )?;
        self.device.wait_autoloader(Duration::from_secs(2))
    }
}

pub struct HostSession<'a> {
    device: &'a Device,
    protocol: HostExchange,
    pages: HashMap<u32, HostBuffer>,
    attached_pages: Vec<u32>,
}

impl<'a> HostSession<'a> {
    pub fn new(device: &'a Device, protocol: HostExchange) -> Result<Self, DriverError> {
        let mut pages = HashMap::new();
        for page in &protocol.pages {
            pages.insert(page.index, HostBuffer::new(page.size as usize)?);
        }
        Ok(Self {
            device,
            protocol,
            pages,
            attached_pages: Vec::new(),
        })
    }

    pub fn attach(&mut self) -> Result<(), DriverError> {
        info!(
            pages = self.protocol.attach_order.len(),
            "attaching host exchange pages"
        );
        self.device.detach_all_buffers()?;
        for index in &self.protocol.attach_order {
            let page = self
                .pages
                .get(index)
                .ok_or_else(|| DriverError::Invalid(format!("missing host page {index}")))?;
            self.device
                .write_config(pci::EXCHANGE_WINDOW_BASE, pci::EXCHANGE_WINDOW_HEXOPT)?;
            if let Err(error) = self.device.attach_buffer(*index, page.data, page.size) {
                for attached in self.attached_pages.drain(..).rev() {
                    self.device.detach_buffer(attached);
                }
                return Err(error);
            }
            self.attached_pages.push(*index);
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), DriverError> {
        info!(
            startup_mark = self.protocol.startup_mark,
            "starting host exchange session"
        );
        self.device.set_mark(1)?;
        self.device
            .wait_mark(pci::HSP_GS2_CONTROL, 0, Duration::from_secs(10))?;
        self.attach()?;
        self.device.write_config(pci::HSP_GS2_CONTROL, 1)?;
        self.device
            .wait_mark(pci::HSP_GS2_CONTROL, 0, Duration::from_secs(10))?;
        info!("host exchange session started");
        Ok(())
    }

    pub fn invoke(&mut self, name: &str, input: &[u8]) -> Result<Vec<u8>, DriverError> {
        if self.attached_pages.len() != self.protocol.attach_order.len() {
            return Err(DriverError::Invalid("host session not attached".into()));
        }
        let call = self.prepare(name, input)?;
        self.drive(call)
    }

    pub fn prepare(&mut self, name: &str, input: &[u8]) -> Result<HostCall, DriverError> {
        let call = self
            .protocol
            .calls
            .iter()
            .find(|call| call.name == name)
            .cloned()
            .ok_or_else(|| DriverError::Invalid(format!("unknown host call {name}")))?;
        copy_input(&mut self.pages, &call, input)?;
        let command = self
            .pages
            .get_mut(&self.protocol.command_page)
            .ok_or_else(|| DriverError::Invalid("missing command page".into()))?;
        let offset = self.protocol.command_offset as usize;
        command.bytes_mut()[offset..offset + 4].copy_from_slice(&call.command.to_le_bytes());
        fence(Ordering::SeqCst);
        Ok(call)
    }

    pub fn invoke_prepared(&mut self, name: &str) -> Result<Vec<u8>, DriverError> {
        if self.attached_pages.len() != self.protocol.attach_order.len() {
            return Err(DriverError::Invalid("host session not attached".into()));
        }
        let call = self
            .protocol
            .calls
            .iter()
            .find(|call| call.name == name)
            .cloned()
            .ok_or_else(|| DriverError::Invalid(format!("unknown host call {name}")))?;
        self.drive(call)
    }

    fn drive(&mut self, call: HostCall) -> Result<Vec<u8>, DriverError> {
        info!(
            call = call.name,
            command = call.command,
            phases = call.phases,
            "invoking host exchange call"
        );
        for phase in 0..call.phases {
            self.device
                .wait_mark(pci::HSP_GS2_CONTROL, 0, Duration::from_secs(10))?;
            self.device.write_config(pci::HSP_GS2_CONTROL, 1)?;
            self.device
                .wait_mark(pci::HSP_GS2_CONTROL, 0, Duration::from_secs(10))
                .map_err(|error| {
                    DriverError::Timeout(format!(
                        "host call {} phase {phase}/{}: {error}",
                        call.name, call.phases
                    ))
                })?;
        }
        let output = copy_output(&mut self.pages, &call)?;
        info!(
            call = call.name,
            output_bytes = output.len(),
            "host exchange call completed"
        );
        Ok(output)
    }
}

impl Drop for HostSession<'_> {
    fn drop(&mut self) {
        for index in self.attached_pages.drain(..).rev() {
            self.device.detach_buffer(index);
        }
    }
}

fn copy_input(
    pages: &mut HashMap<u32, HostBuffer>,
    call: &HostCall,
    input: &[u8],
) -> Result<(), DriverError> {
    let expected = call
        .inputs
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0) as usize;
    if input.len() != expected {
        return Err(DriverError::Invalid(format!(
            "{} expects {expected} bytes",
            call.name
        )));
    }
    for slice in &call.inputs {
        let page = pages
            .get_mut(&slice.page)
            .ok_or_else(|| DriverError::Invalid("missing input page".into()))?;
        let destination = slice.page_offset as usize;
        let source = slice.file_offset as usize;
        page.bytes_mut()[destination..destination + slice.size as usize]
            .copy_from_slice(&input[source..source + slice.size as usize]);
    }
    Ok(())
}

fn copy_output(
    pages: &mut HashMap<u32, HostBuffer>,
    call: &HostCall,
) -> Result<Vec<u8>, DriverError> {
    let size = call
        .outputs
        .iter()
        .map(|slice| slice.file_offset + slice.size)
        .max()
        .unwrap_or(0) as usize;
    let mut output = vec![0; size];
    fence(Ordering::SeqCst);
    for slice in &call.outputs {
        let page = pages
            .get_mut(&slice.page)
            .ok_or_else(|| DriverError::Invalid("missing output page".into()))?;
        let source = slice.page_offset as usize;
        let destination = slice.file_offset as usize;
        output[destination..destination + slice.size as usize]
            .copy_from_slice(&page.bytes_mut()[source..source + slice.size as usize]);
    }
    Ok(output)
}

pub fn frame_tile(physical_tile: u32, image: &[u8]) -> Result<Vec<u8>, DriverError> {
    if image.is_empty() {
        return Err(DriverError::Invalid("cannot frame empty tile image".into()));
    }
    let frames = image.len().div_ceil(FRAME_PAYLOAD_SIZE);
    let mut output = vec![0; frames * FRAME_SIZE];
    for frame in 0..frames {
        let destination = &mut output[frame * FRAME_SIZE..(frame + 1) * FRAME_SIZE];
        destination[..4].copy_from_slice(&(frame as u32).to_le_bytes());
        destination[4..8].copy_from_slice(&physical_tile.to_le_bytes());
        let flags = if frame + 1 == frames { 8 << 24 } else { 0u32 };
        destination[8..12].copy_from_slice(&flags.to_le_bytes());
        let source = frame * FRAME_PAYLOAD_SIZE;
        let size = FRAME_PAYLOAD_SIZE.min(image.len() - source);
        destination[FRAME_HEADER_SIZE..FRAME_HEADER_SIZE + size]
            .copy_from_slice(&image[source..source + size]);
        destination[FRAME_SIZE - FRAME_TRAILER_SIZE..].fill(0xff);
    }
    Ok(output)
}

pub fn read_configuration(path: impl AsRef<Path>) -> Result<Vec<u8>, DriverError> {
    fs::read(path).map_err(|source| DriverError::Io {
        operation: "read configuration image",
        source,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, DriverError> {
    let bytes = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| DriverError::Invalid("truncated data".into()))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub fn block_device_interrupt_signals() -> Result<(), DriverError> {
    let mut set = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    if unsafe { libc::sigfillset(&mut set) } != 0
        || unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, ptr::null_mut()) } != 0
    {
        Err(io_error("block device interrupt signals"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_marks_final_packet() {
        let data = vec![0x5a; FRAME_PAYLOAD_SIZE + 1];
        let framed = frame_tile(17, &data).unwrap();
        assert_eq!(framed.len(), 2 * FRAME_SIZE);
        assert_eq!(u32::from_le_bytes(framed[4..8].try_into().unwrap()), 17);
        assert_eq!(u32::from_le_bytes(framed[8..12].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_le_bytes(framed[FRAME_SIZE + 8..FRAME_SIZE + 12].try_into().unwrap()),
            8 << 24
        );
        assert!(
            framed[FRAME_SIZE - FRAME_TRAILER_SIZE..FRAME_SIZE]
                .iter()
                .all(|byte| *byte == 0xff)
        );
    }

    #[test]
    fn decodes_every_ipu21_exception_type() {
        for (value, exception) in TileException::ALL.into_iter().enumerate() {
            assert_eq!(TileException::try_from(value as u8), Ok(exception));
            assert_eq!(TileException::from_status((value as u32) << 4), exception);
            assert!(exception.architecture_name().starts_with("TEXCPT_"));
        }
        assert_eq!(TileException::try_from(16), Err(16));
        assert!(TileException::BreakOnSync.is_debug());
        assert!(!TileException::InvalidExchangeConfiguration.is_debug());
    }
}
