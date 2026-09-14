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
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    split_rows: bool,
    #[arg(long)]
    fp8_output: bool,
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
fn softmax_source(path: &std::path::Path) -> Result<String> {
    let mut source = String::new();
    for line in fs::read_to_string(path)?.lines() {
        if let Some(name) = line
            .strip_prefix("#include \"attention_softmax_")
            .and_then(|s| s.strip_suffix('"'))
        {
            source += &softmax_source(
                &path
                    .parent()
                    .unwrap()
                    .join(format!("attention_softmax_{name}")),
            )?;
        } else {
            source.push_str(line);
            source.push('\n');
        }
    }
    Ok(source)
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    ensure!(
        args.keys.len().saturating_mul(args.rows.len()) <= 1472,
        "one tile is needed per test case"
    );
    fs::create_dir_all(&args.output)?;
    let device = PathBuf::from("device").canonicalize()?;
    let mut source = format!(
        "#include \"{}\"\n",
        device.join("static_runtime.S").display()
    );
    let mut programs = Vec::new();
    let mut data = vec![];
    let mut slices = vec![];
    let mut cases = vec![];
    let mut file_offset = 0;
    let mut rng = fastrand::Rng::with_seed(729);
    for &keys in &args.keys {
        let padded = if keys == 729 {
            768
        } else {
            keys.div_ceil(if args.fp8_output { 32 } else { 16 })
                * if args.fp8_output { 32 } else { 16 }
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
            source += "#undef SOFTMAX_FRAME_BYTES\n#undef ATTENTION_OUTPUT_F8\n";
            if args.fp8_output && tag == "new" {
                source += "#define ATTENTION_OUTPUT_F8\n#define ATTENTION_OUTPUT_SCALE -4\n";
            }
            source += &softmax_source(path)?
                .replace(".Lsoftmax_", &format!(".L{tag}_{keys}_"))
                .replace("SOFTMAX_MAX_PANEL", &format!("MAXP_{tag}_{keys}"))
                .replace("SOFTMAX_EXP_PANEL", &format!("EXPP_{tag}_{keys}"))
                .replace("SOFTMAX_SPLIT_INIT", &format!("INIT_{tag}_{keys}"))
                .replace("SOFTMAX_SPLIT_ROW", &format!("ROW_{tag}_{keys}"))
                .replace("SOFTMAX_EXP_PAIR", &format!("EXP_{tag}_{keys}"))
                .replace("SOFTMAX_STORE_QUAD", &format!("QUAD_{tag}_{keys}"))
                .replace("SOFTMAX_CONFIG_OUTPUT", &format!("CONFIG_{tag}_{keys}"))
                .replace("SOFTMAX_NEXT_OUTPUT", &format!("NEXT_{tag}_{keys}"))
                .replace("SOFTMAX_FP8_STORE_EIGHT", &format!("EIGHT_{tag}_{keys}"))
                .replace("SOFTMAX_BEGIN_TAIL", &format!("BEGIN_{tag}_{keys}"))
                .replace("SOFTMAX_END_TAIL", &format!("END_{tag}_{keys}"));
            source.push('\n');
        }
        for &rows in &args.rows {
            ensure!(
                rows > 0 && rows <= 18 && keys > 0 && keys <= 768,
                "test shape exceeds reserved memory"
            );
            let tile = cases.len() as u16;
            let size = rows * (padded * 2 + 64);
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
            let mut steps = Vec::new();
            for (index, (tag, address)) in [("old", old), ("new", new)].into_iter().enumerate() {
                steps.push(TileStep::Compute(ComputeStep {
                    symbol: format!("softmax_{tag}_{keys}"),
                    output_address: TileAddress::Absolute(address),
                    input_addresses: vec![TileAddress::Absolute(0x88000)],
                    arguments: vec![rows, keys, u32::from(args.split_rows && keys >= 128)],
                    profile: StepProfile {
                        before: Some(0x7f000 + index as u32 * 8),
                        after: Some(0x7f004 + index as u32 * 8),
                    },
                }));
            }
            programs.push(TileProgram { tile, steps });
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
    let device = ipu_tests::KernelDevice::load(
        &args.sdk,
        &args.configuration,
        &args.device_lock,
        &application,
    )?;
    let runtime = device.runtime();
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
            for (logical, (rows, keys, _, _)) in cases.iter().enumerate() {
                let tile = ipu_exchange::c600_logical_to_physical(logical as u16);
                for context in 0..=6 {
                    let state = runtime.device().tile_context_state(tile, context)?;
                    if state == 3 {
                        let pc = runtime.device().read_tile_program_counter(tile, context)?;
                        eprintln!(
                            "rows={rows} keys={keys} tile={tile} context={context} status={:x} pc={pc:x} {:?}",
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
    let output = session.finish(&executed)?;
    ipu_tests::completion::diagnose_completion(
        runtime,
        &application,
        std::time::Duration::from_secs(10),
    )?;
    fs::write(args.output.join("output.bin"), &output)?;

    let mut offset = 0;
    let mut measurements = vec![];
    for (rows, keys, padded, scores) in cases {
        let time: Vec<_> = output[offset..offset + 16]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        offset += 16;
        let size = (rows * (padded * 2 + 64)) as usize;
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
            let fp8 = args.fp8_output && variant == 1;
            let read_probability = |r: u32, k: u32| {
                if fp8 {
                    ipu_codegen::f143::f143_to_f32(
                        bytes[(k / 32 * rows * 32 + r * 32 + k % 32) as usize],
                        -4,
                    ) as f64
                } else {
                    readhalf((k / 16 * rows * 16 + r * 16 + k % 16) as usize)
                }
            };
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
                let denominator = readfloat(
                    (rows * padded * if fp8 { 1 } else { 2 } + rows * 4 + r * 4) as usize,
                );
                let max = readfloat((rows * padded * if fp8 { 1 } else { 2 } + r * 4) as usize);
                ensure!(
                    denominator.is_finite()
                        && denominator > 0.0
                        && (max - maximum / 72f64.sqrt()).abs() < 0.01,
                    "bad row state rows={rows} keys={keys} variant={variant} max={max} denominator={denominator}"
                );
                let actual_sum: f64 = (0..keys).map(|k| read_probability(r, k)).sum();
                ensure!(
                    (actual_sum - denominator).abs()
                        < 0.001 + if fp8 { 0.07 } else { 0.002 } * actual_sum,
                    "bad sum {actual_sum} {denominator}"
                );
                for k in 0..padded {
                    let actual = read_probability(r, k);
                    if k >= keys {
                        ensure!(
                            actual == 0.0,
                            "padding rows={rows} keys={keys} k={k}: {actual}"
                        );
                    } else {
                        let delta = (actual / denominator - expected[k as usize] / sum).abs();
                        ensure!(
                            delta.is_finite() && delta < if fp8 { 0.02 } else { 0.001 },
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
