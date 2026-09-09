//! Hardware checks and timings for FP8 casts, packing, tails, and bank placement.
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
    #[arg(long, default_value = "device/cast_f8.cpp")]
    kernel: PathBuf,
    /// Compare a historical kernel which cannot pack row-major input.
    #[arg(long)]
    existing_layouts_only: bool,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long, default_value = "artifacts/fp8-cast/check")]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/device.lock")]
    device_lock: PathBuf,
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    fs::create_dir_all(&args.output)?;
    let device = PathBuf::from("device").canonicalize()?;
    let assembly = args.output.join("cast.S");
    let status = std::process::Command::new(args.sdk.join("bin/popc"))
        .args([
            "--target",
            "ipu21",
            "-O2",
            "--S",
            "-DINPUT_BYTES=2",
            "-DOUTPUT_BYTES=1",
            "-DCAST_VERTEX=Cast2To1",
        ])
        .arg(&args.kernel)
        .arg("-o")
        .arg(&assembly)
        .status()?;
    ensure!(status.success(), "cast compilation failed");
    let source = format!(
        "#include \"{}\"\n#include \"{}\"\n#include \"{}\"\n#define WORKER_CALL_SYMBOL cast_check\n#define WORKER_CODELET_SYMBOL __runCodelet_Cast2To1\n#define WORKER_ARGUMENTS $m3,$m2,$m4,$m5,$m6,$m7,$m8,$m9\n#define WORKER_FRAME_BYTES 32\n#include \"{}\"\n",
        device.join("static_runtime.S").display(),
        device.join("worker_support.S").display(),
        assembly.canonicalize()?.display(),
        device.join("worker_call.S").display(),
    );
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut expected = Vec::new();
    let mut cases = Vec::new();
    let mut rng = fastrand::Rng::with_seed(0x63617374);
    for (rows, columns) in [
        (1u32, 1u32),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 16),
        (5, 48),
        (33, 48),
        (38, 80),
        (9, 12),
        (6, 24),
        (1, 47),
        (1, 48),
        (1, 49),
        (1, 128),
        (2, 128),
        (5, 64),
        (6, 64),
        (7, 64),
        (32, 192),
        (65, 128),
        (95, 32), // Workers straddle the pipeline threshold.
        (96, 48), // Pipelined full panel followed by a half-panel tail.
        (97, 64),
        (184, 96), // Largest cast in the full B1 ViT profile.
        (7, 1024), // Row-major stride exceeds packed-stride encoding.
    ] {
        for mode in 0..4 {
            // dense, already packed, row-major -> packed
            if mode >= 2 && args.existing_layouts_only {
                continue;
            }
            if (mode == 1 && !columns.is_multiple_of(16))
                || (mode >= 2 && !columns.is_multiple_of(4))
            {
                continue;
            }
            for (placement, input_address, output_address) in [
                ("separate-standard", 0x60000u32, 0x70008u32),
                ("shared-standard", 0x60000, 0x65008),
                ("interleaved", 0xa0000, 0x90008),
            ] {
                for scale in [-4i8, 0, 3] {
                    let tile = programs.len() as u16;
                    let source_count = rows * columns;
                    let output_columns = if mode == 0 {
                        columns
                    } else {
                        if mode == 3 {
                            columns.next_multiple_of(32) + 64
                        } else {
                            columns.next_multiple_of(32)
                        }
                    };
                    let count = rows * output_columns;
                    let output_address = if placement == "shared-standard" && count <= 2048 {
                        0x62008
                    } else {
                        output_address
                    };
                    // Keep large inputs disjoint even in the shared-bank case.
                    let output_address = if placement == "shared-standard" {
                        output_address.max(input_address + source_count * 2 + 8)
                    } else {
                        output_address
                    };
                    let logical: Vec<u8> = (0..source_count)
                        .map(|_| {
                            let value = rng.u8(..);
                            if value == 0x80 { 0 } else { value }
                        })
                        .collect();
                    let mut input = vec![0u16; source_count as usize];
                    let mut result = vec![0u8; count as usize];
                    for row in 0..rows {
                        for column in 0..columns {
                            let logical_index = (row * columns + column) as usize;
                            let input_index = if mode == 1 {
                                (column / 16 * rows * 16 + row * 16 + column % 16) as usize
                            } else {
                                logical_index
                            };
                            let output_index = if mode != 0 {
                                (column / 32 * rows * 32 + row * 32 + column % 32) as usize
                            } else {
                                logical_index
                            };
                            input[input_index] = half::f16::from_f32(
                                ipu_codegen::f143::f143_to_f32(logical[logical_index], scale),
                            )
                            .to_bits();
                            result[output_index] = logical[logical_index];
                        }
                    }
                    // Guard the sub-word tail and both sides of the allocation.
                    let mut wanted = vec![0xa5; 8];
                    wanted.extend(result);
                    wanted.resize((count as usize).next_multiple_of(8) + 16, 0xa5);
                    slices.push(RegionSlice {
                        tile: u32::from(ipu_exchange::c600_logical_to_physical(tile)),
                        tile_address: output_address - 8,
                        file_offset: expected.len() as u64,
                        size: wanted.len() as u64,
                    });
                    data.push(TileProgramData {
                        tile,
                        address: output_address - 8,
                        data: vec![0xa5; wanted.len()],
                    });
                    expected.extend(wanted);
                    let mut bytes: Vec<_> = input.into_iter().flat_map(u16::to_le_bytes).collect();
                    bytes.resize(bytes.len().next_multiple_of(8), 0);
                    data.push(TileProgramData {
                        tile,
                        address: input_address,
                        data: bytes,
                    });
                    data.push(TileProgramData {
                        tile,
                        address: 0x7f000,
                        data: vec![0; 8],
                    });
                    programs.push(TileProgram {
                        tile,
                        steps: vec![TileStep::Compute(ComputeStep {
                            symbol: "cast_check".into(),
                            output_address: TileAddress::Absolute(output_address),
                            input_addresses: vec![TileAddress::Absolute(input_address)],
                            arguments: vec![
                                count,
                                0,
                                (i32::from(scale)) as u32,
                                if mode == 0 || (rows == 1 && source_count == count) {
                                    0
                                } else {
                                    rows
                                },
                                source_count,
                                if mode >= 2 { columns } else { 0 },
                            ],
                            profile: StepProfile {
                                before: Some(0x7f000),
                                after: Some(0x7f004),
                            },
                        })],
                    });
                    cases.push((rows, columns, mode, placement, scale));
                }
            }
        }
    }
    let wrapper = args.output.join("cast_check.S");
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
    // The final streaming phase is deferred. Wait for every supervisor before
    // reading its host page, otherwise the last batch can still contain the
    // preceding tensor chunk rather than the timestamps.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    for tile in &application.tiles {
        let physical = tile.physical_tile as u16;
        loop {
            let state = runtime.device().tile_context_state(physical, 0)?;
            if state == 0 {
                break;
            }
            if state == 3 {
                let completed = application
                    .debug_symbols
                    .iter()
                    .find(|symbol| symbol.name == ipu_codegen::COMPLETED_SYMBOL)
                    .map(|symbol| symbol.address);
                ensure!(
                    Some(runtime.device().read_tile_program_counter(physical, 0)?) == completed,
                    "tile {physical} failed before completion"
                );
                break;
            }
            ensure!(
                std::time::Instant::now() < deadline,
                "tile {physical} did not finish"
            );
        }
    }
    let actual = session.collect(&call)?;
    fs::write(args.output.join("output.bin"), &actual)?;
    ensure!(
        actual.starts_with(&expected),
        "cast mismatch at byte {:?}",
        actual.iter().zip(&expected).position(|(a, b)| a != b)
    );
    for (case, times) in cases.iter().zip(actual[expected.len()..].chunks_exact(8)) {
        let before = u32::from_le_bytes(times[..4].try_into()?);
        let after = u32::from_le_bytes(times[4..].try_into()?);
        let cycles = after.wrapping_sub(before);
        ensure!(cycles < 100_000, "invalid timing for {case:?}: {cycles}");
        println!("case={case:?} cycles={cycles}");
    }
    println!(
        "cases={} checkedBytes={} bitwiseTest=PASS",
        cases.len(),
        expected.len()
    );
    Ok(())
}
