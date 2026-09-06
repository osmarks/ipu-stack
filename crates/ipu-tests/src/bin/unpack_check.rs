//! Bitwise hardware checks for transposed-AMP unpacking, including padding.
use anyhow::{Result, ensure};
use clap::Parser;
use ipu_codegen::{
    ComputeStep, StepProfile, TileAddress, TileProgram, TileProgramData, TileStep,
    build_tile_program_package,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::Runtime;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/unpack-check")]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/device.lock")]
    device_lock: PathBuf,
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    fs::create_dir_all(&args.output)?;
    let device = PathBuf::from("device").canonicalize()?;
    let kernel = fs::read_to_string(device.join("unpack_transposed_amp_f16.S"))?;
    let mut source = format!(
        "#include \"{}\"\n",
        device.join("static_runtime.S").display()
    );
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut expected = Vec::new();
    let mut cases = Vec::new();
    let mut rng = fastrand::Rng::with_seed(0x756e7061636b);
    for (rows, columns, matrices) in [
        (0u32, 0u32, 1u32),
        (1, 1, 1),
        (9, 114, 1),
        (16, 116, 1),
        (17, 7, 3),
        (31, 65, 2),
        (48, 128, 2),
    ] {
        for interleaved in [false, true] {
            let tile = programs.len() as u16;
            let physical_rows = rows.max(1).div_ceil(16) * 16;
            let physical_columns = columns.max(1).div_ceil(2) * 2;
            let count = matrices * physical_rows * physical_columns;
            let symbol = format!("unpack_check_{tile}");
            for (name, value) in [
                ("CALL_SYMBOL", symbol.clone()),
                ("LOGICAL_ROWS", rows.to_string()),
                ("PHYSICAL_ROWS", physical_rows.to_string()),
                ("LOGICAL_COLUMNS", columns.to_string()),
                ("PHYSICAL_COLUMNS", physical_columns.to_string()),
            ] {
                source += &format!("#undef UNPACK_{name}\n#define UNPACK_{name} {value}\n");
            }
            source += &kernel.replace(".Lunpack_", &format!(".Lunpack_{tile}_"));
            source.push('\n');
            let input: Vec<u16> = (0..count).map(|_| rng.u16(..)).collect();
            let mut result = vec![0u16; count as usize];
            for matrix in 0..matrices {
                let base = matrix * physical_rows * physical_columns;
                for row in 0..rows {
                    for column in 0..columns {
                        result[(base + row * physical_columns + column) as usize] = input[(base
                            + row / 16 * physical_columns * 16
                            + column * 16
                            + row % 16)
                            as usize];
                    }
                }
            }
            // Four-byte, deliberately not eight-byte, aligned pointers; guard
            // both ends of the output and initialize its padding to nonzero bits.
            let input_address = if interleaved { 0x88004 } else { 0x65004 };
            let output_address = 0x5c004;
            let guard = 0xd37au16;
            let mut wanted = vec![guard; 2];
            wanted.extend(result);
            wanted.extend([guard; 2]);
            let bytes: Vec<_> = wanted.iter().flat_map(|x| x.to_le_bytes()).collect();
            slices.push(RegionSlice {
                tile: u32::from(ipu_exchange::c600_logical_to_physical(tile)),
                tile_address: output_address - 4,
                file_offset: expected.len() as u64,
                size: bytes.len() as u64,
            });
            expected.extend(bytes);
            data.push(TileProgramData {
                tile,
                address: output_address - 4,
                data: vec![guard; wanted.len()]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            });
            data.push(TileProgramData {
                tile,
                address: input_address,
                data: input.into_iter().flat_map(u16::to_le_bytes).collect(),
            });
            programs.push(TileProgram {
                tile,
                steps: vec![TileStep::Compute(ComputeStep {
                    symbol,
                    output_address: TileAddress::Absolute(output_address),
                    input_addresses: vec![TileAddress::Absolute(input_address)],
                    arguments: vec![matrices, rows, physical_rows, columns, physical_columns],
                    profile: StepProfile {
                        before: Some(0x7f000),
                        after: Some(0x7f004),
                    },
                })],
            });
            cases.push((rows, columns, matrices, interleaved));
        }
    }
    let wrapper = args.output.join("unpack_check.S");
    for tile in programs.len() as u16..1472 {
        programs.push(TileProgram {
            tile,
            steps: vec![],
        });
    }
    fs::write(&wrapper, source)?;
    let outputs = [
        Binding {
            name: "result".into(),
            dtype: "u8".into(),
            shape: vec![expected.len() as u32],
            slices,
        },
        Binding {
            name: "cycles".into(),
            dtype: "u32".into(),
            shape: vec![cases.len() as u32, 2],
            slices: (0..cases.len())
                .map(|tile| RegionSlice {
                    tile: u32::from(ipu_exchange::c600_logical_to_physical(tile as u16)),
                    tile_address: 0x7f000,
                    file_offset: (tile * 8) as u64,
                    size: 8,
                })
                .collect(),
        },
    ];
    let application = build_tile_program_package(
        &programs,
        &data,
        &outputs,
        &Toolchain::from_sdk(&args.sdk),
        &wrapper,
    )?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&args.device_lock)?;
    lock.lock()?;
    let runtime = Runtime::open("/dev/ipu0", &fs::read(args.configuration)?)?;
    runtime.load(
        &application,
        &fs::read(args.sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"))?,
        application.host_exchange.startup_mark,
    )?;
    let mut session = runtime.host_session(&application)?;
    session.start()?;
    let call = session.invoke_streaming_deferred("run", &[0; 4])?;
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    let actual = session.collect(&call)?;
    fs::write(args.output.join("output.bin"), &actual)?;
    ensure!(
        actual.starts_with(&expected),
        "unpack mismatch at byte {:?}",
        actual.iter().zip(&expected).position(|(a, b)| a != b)
    );
    for (case, times) in cases.iter().zip(actual[expected.len()..].chunks_exact(8)) {
        let before = u32::from_le_bytes(times[..4].try_into()?);
        let after = u32::from_le_bytes(times[4..].try_into()?);
        println!("case={case:?} cycles={}", after.wrapping_sub(before));
    }
    println!(
        "cases={cases:?} checkedBytes={} bitwiseTest=PASS",
        expected.len()
    );
    Ok(())
}
