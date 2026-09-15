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
    #[command(flatten)]
    device: ipu_tests::KernelDeviceOptions,
    #[arg(long, default_value = "device/attention_softmax_f16.S")]
    kernel: PathBuf,
    #[arg(long, default_value = "artifacts/softmax-upgrade/check")]
    output: PathBuf,
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
        {
            let tag = "check";
            let path = &args.kernel;
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
            source += "#undef SOFTMAX_FRAME_BYTES\n#undef SOFTMAX_ROWS\n#undef SOFTMAX_KEYS\n#undef SOFTMAX_SPLIT\n#undef ATTENTION_OUTPUT_F8\n";
            if args.fp8_output {
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
            // Separate memory elements deliberately rule out hidden adjacency
            // between probabilities, persistent statistics, and worker scratch.
            let probability = 0x6c008;
            let statistics = 0x74008;
            let reduction = 0x76008;
            let tail = 0x78008;
            let mut buffers = vec![
                (
                    probability,
                    rows * padded * if args.fp8_output { 1 } else { 2 },
                ),
                (statistics, rows * 8),
                (reduction, rows * 24),
            ];
            let mut addresses = vec![0x88000, statistics, reduction];
            if args.fp8_output && keys != padded {
                addresses.push(tail);
                buffers.push((tail, rows * 32));
            }
            // Host SEND cannot share an SRAM element with its executing code.
            data.push(TileProgramData {
                tile,
                address: 0x6c000,
                data: vec![0x55; 0x10000],
            });
            programs.push(TileProgram {
                tile,
                steps: vec![TileStep::Compute(ComputeStep {
                    symbol: format!("softmax_check_{keys}"),
                    output_address: TileAddress::Absolute(probability),
                    input_addresses: addresses.into_iter().map(TileAddress::Absolute).collect(),
                    arguments: vec![rows, keys, u32::from(args.split_rows && keys >= 128)],
                    profile: StepProfile {
                        before: Some(0x7f000),
                        after: Some(0x7f004),
                    },
                })],
            });
            data.push(TileProgramData {
                tile,
                address: 0x7c000,
                data: vec![0; 0x4000],
            });
            let physical = u32::from(ipu_target::c600::logical_to_physical(tile));
            for (address, size) in std::iter::once((0x7f000, 8)).chain(
                buffers
                    .iter()
                    .map(|&(address, size)| (address - 8, size + 16)),
            ) {
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
            cases.push((rows, keys, padded, scores, buffers));
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
        &Toolchain::from_sdk(&args.device.sdk),
        &wrapper,
    )?;
    let device = ipu_tests::KernelDevice::load(&args.device, &application)?;
    let runtime = device.runtime();
    let mut session = runtime.host_session(&application)?;
    session.start()?;
    let executed = session
        .invoke_streaming_deferred("run", &[0; 4])
        .inspect_err(|_| {
            ipu_tests::completion::report_kernel_faults(
                runtime,
                &application,
                cases
                    .iter()
                    .map(|(rows, keys, ..)| format!("rows={rows} keys={keys}")),
            );
        })?;
    let output = session.finish(&executed)?;
    ipu_tests::completion::diagnose_completion(
        runtime,
        &application,
        std::time::Duration::from_secs(10),
    )?;
    fs::write(args.output.join("output.bin"), &output)?;

    let mut offset = 0;
    let mut measurements = vec![];
    for (rows, keys, padded, scores, buffers) in cases {
        let time: Vec<_> = output[offset..offset + 8]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        offset += 8;
        let mut contents = Vec::new();
        for (address, size) in buffers {
            let size = size as usize;
            ensure!(
                output[offset..offset + 8] == [0x55; 8]
                    && output[offset + 8 + size..offset + 16 + size] == [0x55; 8],
                "guard rows={rows} keys={keys} address=0x{address:x}"
            );
            contents.push(&output[offset + 8..offset + 8 + size]);
            offset += size + 16;
        }
        let bytes = contents[0];
        let statistics = contents[1];
        let mut error = 0.0f64;
        let readhalf = |i: usize| {
            f16::from_bits(u16::from_le_bytes(
                bytes[i * 2..i * 2 + 2].try_into().unwrap(),
            ))
            .to_f64()
        };
        let readfloat =
            |i: usize| f32::from_le_bytes(statistics[i..i + 4].try_into().unwrap()) as f64;
        let fp8 = args.fp8_output;
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
            let denominator = readfloat(((rows + r) * 4) as usize);
            let max = readfloat((r * 4) as usize);
            ensure!(
                denominator.is_finite()
                    && denominator > 0.0
                    && (max - maximum / 72f64.sqrt()).abs() < 0.01,
                "bad row state rows={rows} keys={keys} max={max} denominator={denominator}"
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
                        "probability rows={rows} keys={keys} r={r} k={k} delta={delta}"
                    );
                    error = error.max(delta);
                }
            }
        }
        measurements.push(serde_json::json!({"rows":rows,"keys":keys,"padded":padded,"cycles":time[1].wrapping_sub(time[0]),"max_probability_error":error}));
    }
    fs::write(
        args.output.join("measurements.json"),
        serde_json::to_vec_pretty(&measurements)?,
    )?;
    println!("{} softmax cases passed", measurements.len());
    Ok(())
}
