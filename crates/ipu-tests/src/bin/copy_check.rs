//! Execute affine packing tasks from exchange-gather.py with production helpers.
use anyhow::{Result, ensure};
use clap::Parser;
use ipu_codegen::{
    ComputeStep, StepProfile, TileAddress, TileProgram, TileProgramData, TileStep,
    build_tile_program_package,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Arguments {
    tasks: PathBuf,
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/device.lock")]
    device_lock: PathBuf,
}

fn number(v: &serde_json::Value, key: &str) -> u32 {
    v[key].as_u64().unwrap().try_into().unwrap()
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    fs::create_dir_all(&args.output)?;
    let cases: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&args.tasks)?)?;
    ensure!(cases.len() <= 1472, "too many cases");
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut expected = Vec::new();
    for (tile, case) in cases.iter().enumerate() {
        let bytes = number(case, "bytes") as usize;
        // Isolate packing from model placement: preserve standard/interleaved
        // classes, using fixture ranges clear of host exchange support.
        let input_address = 0x60000;
        let output_address = 0x90000;
        ensure!(bytes <= 0x10000, "case exceeds fixture buffer size");
        let input: Vec<u8> = (0..bytes)
            .map(|i| ((i as u32).wrapping_mul(2654435761).rotate_left(13) >> 17) as u8)
            .collect();
        let mut output = vec![0xcdu8; bytes];
        let mut covered = vec![false; bytes];
        let tasks = case["tasks"].as_array().unwrap();
        ensure!(
            !tasks.is_empty(),
            "direct relay has no packing work to measure"
        );
        let mut steps = Vec::new();
        for (index, task) in tasks.iter().enumerate() {
            let source = number(task, "source");
            let destination = number(task, "destination");
            let size = number(task, "row_bytes");
            let rows = number(task, "rows");
            let ss = number(task, "source_stride");
            let ds = number(task, "destination_stride");
            ensure!(
                [source, destination, size, ss, ds]
                    .iter()
                    .all(|x| x % 8 == 0),
                "unaligned task"
            );
            for row in 0..rows {
                let a = (source + row * ss) as usize;
                let b = (destination + row * ds) as usize;
                ensure!(
                    a + size as usize <= bytes && b + size as usize <= bytes,
                    "out of bounds task"
                );
                ensure!(
                    covered[b..b + size as usize].iter().all(|x| !x),
                    "overlapping tasks"
                );
                covered[b..b + size as usize].fill(true);
                output[b..b + size as usize].copy_from_slice(&input[a..a + size as usize]);
            }
            steps.push(TileStep::Compute(ComputeStep {
                symbol: ipu_codegen::COPY_STRIDED_U64_SYMBOL.into(),
                output_address: TileAddress::Absolute(output_address + destination),
                input_addresses: vec![TileAddress::Absolute(input_address + source)],
                arguments: vec![size / 8, rows, ss, ds],
                profile: StepProfile {
                    before: (index == 0).then_some(0x7f000),
                    after: (index + 1 == tasks.len()).then_some(0x7f004),
                },
            }));
        }
        for (address, bytes) in [(input_address, input), (output_address, vec![0xcd; bytes])] {
            data.push(TileProgramData {
                tile: tile as u16,
                address,
                data: bytes,
            });
        }
        slices.push(RegionSlice {
            tile: u32::from(ipu_exchange::c600_logical_to_physical(tile as u16)),
            tile_address: output_address,
            file_offset: expected.len() as u64,
            size: bytes as u64,
        });
        expected.extend(output);
        programs.push(TileProgram {
            tile: tile as u16,
            steps,
        });
    }
    for tile in programs.len() as u16..1472 {
        programs.push(TileProgram {
            tile,
            steps: vec![],
        });
    }
    let outputs = [
        Binding {
            name: "result".into(),
            dtype: "u8".into(),
            shape: vec![expected.len() as u32],
            slices,
        },
        ipu_tests::cycle_binding(cases.len().try_into()?, 0x7f000),
    ];
    let application = build_tile_program_package(
        &programs,
        &data,
        &outputs,
        &Toolchain::from_sdk(&args.sdk),
        &PathBuf::from("device/static_runtime.S"),
    )?;
    let device = ipu_tests::KernelDevice::load(
        &args.sdk,
        &args.configuration,
        &args.device_lock,
        &application,
    )?;
    let runtime = device.runtime();
    let mut session = runtime.host_session(&application)?;
    session.start()?;
    let call = session.invoke_streaming_deferred("run", &[0; 4])?;
    let actual = session.finish(&call)?;
    ensure!(
        actual.starts_with(&expected),
        "packing mismatch at byte {:?}",
        actual.iter().zip(&expected).position(|(a, b)| a != b)
    );
    let results: Vec<_> = cases.iter().zip(actual[expected.len()..].chunks_exact(8)).map(|(case, times)| {
        let before = u32::from_le_bytes(times[..4].try_into().unwrap());
        let after = u32::from_le_bytes(times[4..].try_into().unwrap());
        serde_json::json!({"phase": case["phase"], "relay": case["tile"], "bytes": case["bytes"], "calls": case["tasks"].as_array().unwrap().len(), "cycles": after.wrapping_sub(before)})
    }).collect();
    fs::write(
        args.output.join("cycles.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    println!(
        "cases={} checkedBytes={} bitwiseTest=PASS",
        cases.len(),
        expected.len()
    );
    Ok(())
}
