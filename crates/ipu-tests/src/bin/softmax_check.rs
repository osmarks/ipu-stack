//! Device correctness and timing checks for row softmax, including masked tails.
use anyhow::{Result, ensure};
use clap::Parser;
use half::f16;
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
    #[arg(long, default_value = "random", value_parser = ["random", "constant", "extreme"])]
    pattern: String,
    #[arg(long, value_delimiter = ',', default_value = "1,2,5,6,7,8,12,17")]
    rows: Vec<u32>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,2,15,16,17,31,64,65,729,768"
    )]
    keys: Vec<u32>,
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, default_value = "device/attention_softmax_f16.S")]
    kernel: PathBuf,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long, default_value = "artifacts/softmax-upgrade/check")]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/device.lock")]
    device_lock: PathBuf,
}
fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    fs::create_dir_all(&args.output)?;
    let device = PathBuf::from("device").canonicalize()?;
    let mut source = format!(
        "#include \"{}\"\n",
        device.join("static_runtime.S").display()
    );
    let mut programs = (0..1472)
        .map(|tile| TileProgram {
            tile,
            steps: vec![],
        })
        .collect::<Vec<_>>();
    let mut data = vec![];
    let mut slices = vec![];
    let mut cases = vec![];
    let mut file_offset = 0;
    let mut rng = fastrand::Rng::with_seed(729);
    for &keys in &args.keys {
        let padded = if keys == 729 {
            768
        } else {
            keys.div_ceil(16) * 16
        };
        for (tag, path) in [("old", &args.reference), ("new", &args.kernel)] {
            for (name, value) in [
                ("FULL_BLOCK", u32::from(keys == padded).to_string()),
                ("KEY_BLOCK_COLUMNS", padded.to_string()),
                (
                    "SCALE_BITS",
                    format!("0x{:08x}", (1.0f32 / 72.0f32.sqrt()).to_bits()),
                ),
                ("SOFTMAX_SYMBOL", format!("softmax_{tag}_{keys}")),
            ] {
                source += &format!("#undef ATTENTION_{name}\n#define ATTENTION_{name} {value}\n");
            }
            source += "#undef SOFTMAX_FRAME_BYTES\n";
            source += &fs::read_to_string(path)?
                .replace(".Lsoftmax_", &format!(".L{tag}_{keys}_"))
                .replace("SOFTMAX_MAX_PANEL", &format!("MAXP_{tag}_{keys}"))
                .replace("SOFTMAX_EXP_PANEL", &format!("EXPP_{tag}_{keys}"))
                .replace("SOFTMAX_SPLIT_INIT", &format!("INIT_{tag}_{keys}"))
                .replace("SOFTMAX_SPLIT_ROW", &format!("ROW_{tag}_{keys}"))
                .replace("SOFTMAX_EXP_PAIR", &format!("EXP_{tag}_{keys}"))
                .replace("SOFTMAX_STORE_QUAD", &format!("QUAD_{tag}_{keys}"));
            source.push('\n');
        }
        for &rows in &args.rows {
            ensure!(
                rows > 0 && rows <= 18 && keys > 0 && keys <= 768,
                "test shape exceeds reserved memory"
            );
            let tile = cases.len() as u16;
            let size = rows * (padded + 16) * 2;
            let mut scores = vec![0x7e00u16; (rows * padded) as usize];
            for r in 0..rows {
                for k in 0..keys {
                    let value = if args.pattern == "extreme" {
                        if k % 3 == 0 { 65504.0 } else { -65504.0 }
                    } else if args.pattern == "constant" || r % 3 == 0 {
                        0.0
                    } else {
                        (rng.f32() - 0.5) * 64.0
                    };
                    scores[(k / 16 * rows * 16 + r * 16 + k % 16) as usize] =
                        f16::from_f32(value).to_bits();
                }
            }
            data.push(TileProgramData {
                tile,
                address: 0x88000,
                data: scores.iter().flat_map(|v| v.to_le_bytes()).collect(),
            });
            let old = 0x5c008;
            let new = old + size + 16;
            // Host SEND cannot share an SRAM element with its executing code.
            // Reserve whole output elements, including gaps between cases.
            data.push(TileProgramData {
                tile,
                address: 0x5c000,
                data: vec![0x55; 0x10000],
            });
            for (index, (tag, address)) in [("old", old), ("new", new)].into_iter().enumerate() {
                programs[tile as usize]
                    .steps
                    .push(TileStep::Compute(ComputeStep {
                        symbol: format!("softmax_{tag}_{keys}"),
                        output_address: TileAddress::Absolute(address),
                        input_addresses: vec![TileAddress::Absolute(0x88000)],
                        arguments: vec![rows, keys],
                        profile: StepProfile {
                            before: Some(0x7f000 + index as u32 * 8),
                            after: Some(0x7f004 + index as u32 * 8),
                        },
                    }));
            }
            data.push(TileProgramData {
                tile,
                address: 0x7c000,
                data: vec![0; 0x4000],
            });
            let physical = u32::from(ipu_exchange::c600_logical_to_physical(tile));
            for (address, size) in [(0x7f000, 16), (old - 8, size + 16), (new - 8, size + 16)] {
                for start in (0..size).step_by(256) {
                    let chunk = (size - start).min(256);
                    slices.push(RegionSlice {
                        tile: physical,
                        tile_address: address + start,
                        file_offset,
                        size: u64::from(chunk),
                    });
                    file_offset += u64::from(chunk);
                }
            }
            cases.push((rows, keys, padded, scores));
        }
    }
    let wrapper = args.output.join("softmax_check.S");
    fs::write(&wrapper, source)?;
    let outputs = vec![Binding {
        name: "results".into(),
        dtype: "u8".into(),
        shape: vec![file_offset as u32],
        slices,
    }];
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
    let executed = match session.invoke_streaming_deferred("run", &[0; 4]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "completion={:?}",
                application
                    .debug_symbols
                    .iter()
                    .filter(|s| s.name == ipu_codegen::COMPLETED_SYMBOL)
                    .collect::<Vec<_>>()
            );
            for logical in 0..cases.len() {
                let tile = ipu_exchange::c600_logical_to_physical(logical as u16);
                for context in 0..=6 {
                    let state = runtime.device().tile_context_state(tile, context)?;
                    if state == 3 {
                        let pc = runtime.device().read_tile_program_counter(tile, context)?;
                        eprintln!(
                            "case={:?} tile={tile} context={context} status={:x} pc={pc:x} {:?}",
                            &cases[logical].0..=&cases[logical].1,
                            runtime.device().read_tile_context_status(tile, context)?,
                            application.symbolize_pc(u32::from(tile), pc)
                        );
                        if context == 0 {
                            eprintln!(
                                "regs={:?}",
                                (0..12)
                                    .map(|r| runtime
                                        .device()
                                        .read_tile_m_register(tile, context, r))
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
            return Err(e.into());
        }
    };
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    let completed_pc = application
        .debug_symbols
        .iter()
        .find(|symbol| symbol.name == ipu_codegen::COMPLETED_SYMBOL)
        .unwrap()
        .address;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    for tile in &application.tiles {
        let physical = tile.physical_tile as u16;
        loop {
            let state = runtime.device().tile_context_state(physical, 0)?;
            if state == 0 {
                break;
            }
            if state == 3
                && runtime.device().read_tile_program_counter(physical, 0)? == completed_pc
            {
                ensure!(
                    runtime
                        .device()
                        .read_tile_word(physical, tile.diagnostic_address)?
                        == 1
                );
                break;
            }
            ensure!(
                std::time::Instant::now() < deadline,
                "tile {physical} did not finish"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        for worker in 1..=6 {
            ensure!(
                runtime.device().tile_context_state(physical, worker)? == 0,
                "tile {physical} worker {worker} did not halt"
            );
        }
    }
    let output = session.collect(&executed)?;
    fs::write(args.output.join("output.bin"), &output)?;

    let mut offset = 0;
    let mut measurements = vec![];
    for (rows, keys, padded, scores) in cases {
        let time: Vec<_> = output[offset..offset + 16]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        offset += 16;
        let size = (rows * (padded + 16) * 2) as usize;
        let mut errors = [0.0f64; 2];
        for (variant, error) in errors.iter_mut().enumerate() {
            ensure!(
                output[offset..offset + 8] == [0x55; 8]
                    && output[offset + 8 + size..offset + 16 + size] == [0x55; 8],
                "guard rows={rows} keys={keys} variant={variant}"
            );
            let bytes = &output[offset + 8..offset + 8 + size];
            offset += size + 16;
            let readhalf = |i: usize| {
                f16::from_bits(u16::from_le_bytes(
                    bytes[i * 2..i * 2 + 2].try_into().unwrap(),
                ))
                .to_f64()
            };
            let readfloat =
                |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as f64;
            for r in 0..rows {
                let index = |k: u32| (k / 16 * rows * 16 + r * 16 + k % 16) as usize;
                let maximum = (0..keys)
                    .map(|k| f16::from_bits(scores[index(k)]).to_f64())
                    .fold(f64::NEG_INFINITY, f64::max);
                let expected: Vec<_> = (0..keys)
                    .map(|k| {
                        ((f16::from_bits(scores[index(k)]).to_f64() - maximum) / 72f64.sqrt()).exp()
                    })
                    .collect();
                let sum: f64 = expected.iter().sum();
                let denominator = readfloat((rows * padded * 2 + rows * 4 + r * 4) as usize);
                let max = readfloat((rows * padded * 2 + r * 4) as usize);
                ensure!(
                    denominator.is_finite()
                        && denominator > 0.0
                        && (max - maximum / 72f64.sqrt()).abs() < 0.01,
                    "bad row state rows={rows} keys={keys} variant={variant} max={max} denominator={denominator}"
                );
                let actual_sum: f64 = (0..keys).map(|k| readhalf(index(k))).sum();
                ensure!(
                    (actual_sum - denominator).abs() < 0.001,
                    "bad sum {actual_sum} {denominator}"
                );
                for k in 0..padded {
                    let actual = readhalf(index(k));
                    if k >= keys {
                        ensure!(
                            actual == 0.0,
                            "padding rows={rows} keys={keys} k={k}: {actual}"
                        );
                    } else {
                        let delta = (actual / denominator - expected[k as usize] / sum).abs();
                        ensure!(
                            delta.is_finite() && delta < 0.001,
                            "probability rows={rows} keys={keys} variant={variant} r={r} k={k} delta={delta}"
                        );
                        *error = error.max(delta);
                    }
                }
            }
        }
        measurements.push(serde_json::json!({"rows":rows,"keys":keys,"padded":padded,"old":time[1].wrapping_sub(time[0]),"new":time[3].wrapping_sub(time[2]),"max_probability_error":errors}));
    }
    fs::write(
        args.output.join("measurements.json"),
        serde_json::to_vec_pretty(&measurements)?,
    )?;
    println!("{} softmax cases passed", measurements.len());
    Ok(())
}
