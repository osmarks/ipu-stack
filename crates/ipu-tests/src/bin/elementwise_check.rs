//! Direct device checks for elementwise kernels, including row and worker tails.
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
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long)]
    fp8: bool,
    #[arg(long)]
    residual: bool,
    #[arg(long, default_value = "device")]
    source: PathBuf,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long, default_value = "artifacts/elementwise-upgrade/check")]
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
        "#include \"{}\"\n#include \"{}\"\n",
        device.join("static_runtime.S").display(),
        device.join("worker_support.S").display()
    );
    let mut kernels = vec![
        (
            "add_f16",
            "AddF16",
            "elementwise_f16.cpp",
            vec!["-DVERTEX_AddF16"],
            None,
            "$m3,$m4,$m2,$m5,$m6,$m7",
        ),
        (
            "layer_norm_f16",
            "LayerNormF16",
            "elementwise_f16.cpp",
            vec!["-DVERTEX_LayerNormF16"],
            Some("layer_norm_f16.S"),
            "",
        ),
        (
            "add_layer_norm_f16",
            "AddLayerNormF16",
            "elementwise_f16.cpp",
            vec!["-DVERTEX_LayerNormF16", "-DNORM_WITH_ADD"],
            Some("layer_norm_f16.S"),
            "",
        ),
        (
            "layer_norm_moments",
            "LayerNormMoments",
            "layer_norm_distributed.cpp",
            vec!["-DVERTEX_LayerNormMoments"],
            Some("layer_norm_moments.S"),
            "",
        ),
        (
            "layer_norm_apply",
            "LayerNormApply",
            "layer_norm_distributed.cpp",
            vec!["-DVERTEX_LayerNormApply"],
            None,
            "$m3,$m4,$m5,$m6,$m2,$m7,$m8,$m9",
        ),
    ];
    if args.fp8 {
        kernels.extend([(
            "layer_norm_f8",
            "LayerNormF8",
            "elementwise_f16.cpp",
            vec!["-DVERTEX_LayerNormF16", "-DNORM_FP8"],
            Some("layer_norm_f16.S"),
            "",
        )]);
    }
    if args.residual {
        kernels.push((
            "add_layer_norm_moments",
            "AddLayerNormMoments",
            "layer_norm_distributed.cpp",
            vec!["-DVERTEX_LayerNormMoments", "-DNORM_STORE_SUM"],
            Some("layer_norm_moments.S"),
            "",
        ));
    }
    for (index, (symbol, vertex, cpp, flags, wrapper, registers)) in kernels.into_iter().enumerate()
    {
        let assembly = args.output.join(format!("{symbol}.S"));
        let status = std::process::Command::new(args.sdk.join("bin/popc"))
            .args(["--target", "ipu21", "-O2", "--S", "-I"])
            .arg(&args.source)
            .args(flags)
            .arg(args.source.join(cpp))
            .arg("-o")
            .arg(&assembly)
            .status()?;
        ensure!(status.success(), "{symbol} compilation failed");
        source += &fs::read_to_string(assembly)?.replace(".L", &format!(".Lcode{index}_"));
        source += "\n";
        if let Some(wrapper) = wrapper {
            if symbol == "add_layer_norm_f16" {
                source += "#define NORM_WITH_ADD\n";
            }
            if symbol == "add_layer_norm_moments" {
                source += "#define NORM_STORE_SUM\n";
            }
            if symbol == "layer_norm_f8" {
                source += "#define NORM_FP8\n";
            }
            source += &fs::read_to_string(args.source.join(wrapper))?
                .replace(".L", &format!(".Lwrap{index}_"));
            source += "\n#undef NORM_STORE_SUM\n#undef MOMENTS_SYMBOL\n#undef MOMENTS_VERTEX\n#undef MOMENTS_ROWS\n#undef MOMENTS_WIDTH\n#undef NORM_FP8\n#undef NORM_WITH_ADD\n#undef NORM_SYMBOL\n#undef NORM_VERTEX\n#undef NORM_ROWS\n#undef NORM_WIDTH\n#undef NORM_GAMMA\n#undef NORM_BETA\n#undef NORM_SHIFT\n#undef NORM_STACK\n#undef NORM_SCRATCH\n";
        } else {
            source += &format!(
                "#define WORKER_CALL_SYMBOL {symbol}\n#define WORKER_CODELET_SYMBOL __runCodelet_{vertex}\n#define WORKER_ARGUMENTS {registers}\n#define WORKER_FRAME_BYTES 48\n#include \"{}\"\n#undef WORKER_CALL_SYMBOL\n#undef WORKER_CODELET_SYMBOL\n#undef WORKER_ARGUMENTS\n#undef WORKER_FRAME_BYTES\n",
                device.join("worker_call.S").display()
            );
        }
    }
    if args.fp8 {
        source += &format!("#include \"{}\"\n", device.join("gelu_f8.S").display());
    }
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let mut slices = Vec::new();
    let mut expected = Vec::<f32>::new();
    let mut cases = Vec::new();
    let mut rng = fastrand::Rng::with_seed(0x6e6f726d);
    for width in [
        1u32, 2, 3, 4, 6, 7, 12, 14, 16, 24, 32, 72, 144, 288, 576, 1152, 1728, 2152,
    ] {
        for rows in [1, 3] {
            for mode in 0..if args.residual {
                12
            } else if args.fp8 {
                10
            } else {
                6
            } {
                if (6..10).contains(&mode) && !args.fp8 {
                    continue;
                }
                if mode >= 6 && !width.is_multiple_of(4) {
                    continue;
                }
                if mode >= 2 && width % 2 != 0 {
                    continue;
                }
                // Add, broadcast add, ordinary LN, add+LN, distributed moments,
                // and distributed apply (equal-sized feature shards).
                for offset in [0, 4, 8] {
                    if mode >= 6 && offset != 0 {
                        continue;
                    }
                    // Offset 8 tests safe in-place F16 output; moments use
                    // a separate FP32 output allocation by definition.
                    if offset == 8 && mode == 4 {
                        continue;
                    }
                    let tile = programs.len() as u16;
                    let elements = width * rows;
                    let input_stride = if mode == 8 || mode == 9 {
                        width.next_multiple_of(32) + 64
                    } else {
                        width
                    };
                    let output_columns = if mode == 9 {
                        width.next_multiple_of(32) + 64
                    } else if mode == 7 {
                        width.next_multiple_of(32)
                    } else {
                        width
                    };
                    let rounded = |x| f16::from_f32(x).to_f32();
                    let input: Vec<f32> = (0..elements)
                        .map(|_| {
                            if width == 1152 && offset == 8 {
                                100.0
                            } else {
                                rounded(
                                    (rng.f32() - 0.5) * 4.0
                                        + if mode < 8 && (width == 14 || width == 144) {
                                            100.0
                                        } else {
                                            0.0
                                        },
                                )
                            }
                        })
                        .collect();
                    let right: Vec<f32> = (0..if mode == 1 { width } else { elements })
                        .map(|_| rounded(rng.f32() - 0.5))
                        .collect();
                    let gamma: Vec<f32> = (0..width).map(|_| rounded(0.5 + rng.f32())).collect();
                    let beta: Vec<f32> = (0..width).map(|_| rounded(rng.f32() - 0.5)).collect();
                    let x: Vec<f32> = input
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            if mode == 3 || mode >= 10 {
                                rounded(x + right[i])
                            } else {
                                x
                            }
                        })
                        .collect();
                    let mut wanted = Vec::new();
                    let mut moments = Vec::<f32>::new();
                    for row in 0..rows as usize {
                        let slice = &x[row * width as usize..(row + 1) * width as usize];
                        let mean = slice.iter().map(|&v| v as f64).sum::<f64>() / width as f64;
                        let variance = slice
                            .iter()
                            .map(|&v| (v as f64 - mean).powi(2))
                            .sum::<f64>();
                        moments.extend([mean as f32, variance as f32]);
                        if mode == 4 || mode == 11 {
                            wanted.extend([mean as f32, variance as f32]);
                        } else {
                            for (col, &v) in slice.iter().enumerate() {
                                wanted.push(if mode < 2 {
                                    rounded(v + right[(row * width as usize + col) % right.len()])
                                } else if mode == 10 {
                                    v
                                } else if mode >= 8 {
                                    0.5 * v
                                        * (1.0 + (0.7978846 * (v + 0.044715 * v.powi(3))).tanh())
                                } else {
                                    rounded(
                                        (((v as f64 - mean)
                                            / (variance / width as f64 + 1e-6).sqrt())
                                            * gamma[col] as f64
                                            + beta[col] as f64)
                                            as f32,
                                    )
                                });
                            }
                        }
                    }
                    if (6..10).contains(&mode) && mode % 2 == 1 {
                        let mut packed = vec![0.0; (rows * output_columns) as usize];
                        for row in 0..rows {
                            for col in 0..width {
                                packed[((col / 32 * rows + row) * 32 + col % 32) as usize] =
                                    wanted[(row * width + col) as usize];
                            }
                        }
                        wanted = packed;
                    }
                    let mut physical_input = vec![f32::NAN; (rows * input_stride) as usize];
                    for row in 0..rows as usize {
                        physical_input[row * input_stride as usize
                            ..row * input_stride as usize + width as usize]
                            .copy_from_slice(
                                &input[row * width as usize..(row + 1) * width as usize],
                            );
                    }
                    let addresses = [
                        0x60000 + offset,
                        0x64000 + offset,
                        0x68000 + offset,
                        0x6a000 + offset,
                    ];
                    for (values, address) in [&physical_input, &right, &gamma, &beta]
                        .into_iter()
                        .zip(addresses)
                    {
                        if offset == 8 && address == addresses[0] {
                            continue;
                        }
                        data.push(TileProgramData {
                            tile,
                            address,
                            data: values
                                .iter()
                                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                                .collect(),
                        });
                    }
                    data.push(TileProgramData {
                        tile,
                        address: 0x6c000,
                        data: moments.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    });
                    let output = if offset == 8 {
                        addresses[0]
                    } else {
                        0x70008 + if mode == 4 || mode == 11 { 0 } else { offset }
                    };
                    let bytes_per_element = if mode == 4 || mode == 11 {
                        4
                    } else if (6..10).contains(&mode) {
                        1
                    } else {
                        2
                    };
                    let byte_count = wanted.len() * bytes_per_element;
                    let guarded = (byte_count + 16).next_multiple_of(8);
                    let mut initial = vec![0xa5; guarded];
                    if offset == 8 {
                        let source_bytes: Vec<_> = input
                            .iter()
                            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                            .collect();
                        initial[8..8 + source_bytes.len()].copy_from_slice(&source_bytes);
                    }
                    data.push(TileProgramData {
                        tile,
                        address: output - 8,
                        data: initial,
                    });
                    slices.push(RegionSlice {
                        tile: u32::from(ipu_exchange::c600_logical_to_physical(tile)),
                        tile_address: output - 8,
                        file_offset: cases
                            .iter()
                            .map(
                                |(_, _, _, _, _, bytes): &(u32, u32, u32, u32, usize, usize)| {
                                    *bytes as u64
                                },
                            )
                            .sum(),
                        size: guarded as u64,
                    });
                    cases.push((width, rows, mode, offset, wanted.len(), guarded));
                    expected.extend(wanted);
                    data.push(TileProgramData {
                        tile,
                        address: 0x7f000,
                        data: vec![0; 8],
                    });
                    let (symbol, inputs, arguments) = match mode {
                        0 | 1 => (
                            "add_f16",
                            vec![addresses[0], addresses[1]],
                            vec![elements, elements, right.len() as u32],
                        ),
                        2 => (
                            "layer_norm_f16",
                            vec![addresses[0], addresses[2], addresses[3]],
                            vec![rows, width],
                        ),
                        3 => ("add_layer_norm_f16", addresses.to_vec(), vec![rows, width]),
                        4 => ("layer_norm_moments", vec![addresses[0]], vec![rows, width]),
                        10 | 11 => (
                            "add_layer_norm_moments",
                            vec![
                                addresses[0],
                                addresses[1],
                                if mode == 10 { output } else { 0x74000 },
                            ],
                            vec![rows, width],
                        ),
                        6 | 7 => (
                            "layer_norm_f8",
                            vec![addresses[0], addresses[2], addresses[3]],
                            vec![rows, width, (-4i32) as u32, mode % 2],
                        ),
                        8 | 9 => (
                            "gelu_f8",
                            vec![addresses[0]],
                            vec![
                                rows,
                                width,
                                (-4i32) as u32,
                                mode % 2,
                                input_stride,
                                output_columns,
                            ],
                        ),
                        _ => (
                            "layer_norm_apply",
                            vec![addresses[0], addresses[2], addresses[3], 0x6c000],
                            vec![rows, width, 1],
                        ),
                    };
                    if mode == 11 {
                        data.push(TileProgramData {
                            tile,
                            address: 0x74000,
                            data: vec![0xa5; elements as usize * 2],
                        });
                    }
                    programs.push(TileProgram {
                        tile,
                        steps: vec![TileStep::Compute(ComputeStep {
                            symbol: symbol.into(),
                            output_address: TileAddress::Absolute(if mode == 10 {
                                0x6c000
                            } else {
                                output
                            }),
                            input_addresses: inputs
                                .into_iter()
                                .map(TileAddress::Absolute)
                                .collect(),
                            arguments,
                            profile: StepProfile {
                                before: Some(0x7f000),
                                after: Some(0x7f004),
                            },
                        })],
                    });
                }
            }
        }
    }
    let bytes: usize = cases.iter().map(|case| case.5).sum();
    let wrapper = args.output.join("elementwise_check.S");
    fs::write(&wrapper, source)?;
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
            shape: vec![bytes as u32],
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
    let call = session.invoke_streaming_deferred("run", &[0; 4]).inspect_err(|_| {
        for (tile, case) in cases.iter().enumerate() {
            let physical = ipu_exchange::c600_logical_to_physical(tile as u16);
            for context in 0..=6 {
                if runtime.device().tile_context_state(physical, context).ok() == Some(3) {
                    let pc = runtime.device().read_tile_program_counter(physical, context).unwrap_or(0);
                    eprintln!("case={case:?} tile={tile} context={context} pc={pc:x} symbol={:?} exception={:?}", application.symbolize_pc(u32::from(physical), pc), runtime.device().read_tile_context_status(physical, context).map(ipu_driver::TileException::from_status));
                }
            }
        }
    })?;
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
    let mut at = 0;
    let mut logical = 0;
    for (case, times) in cases.iter().zip(actual[bytes..].chunks_exact(8)) {
        let (width, rows, mode, offset, count, guarded) = *case;
        let size = if mode == 4 || mode == 11 {
            4
        } else if (6..10).contains(&mode) {
            1
        } else {
            2
        };
        ensure!(
            actual[at..at + 8]
                .iter()
                .chain(actual[at + 8 + count * size..at + guarded].iter())
                .all(|&b| b == 0xa5),
            "guard corruption: {case:?}"
        );
        for i in 0..count {
            let begin = at + 8 + i * size;
            let value = if mode == 4 || mode == 11 {
                f32::from_le_bytes(actual[begin..begin + 4].try_into()?)
            } else if (6..10).contains(&mode) {
                ipu_codegen::f143::f143_to_f32(actual[begin], -4)
            } else {
                f16::from_bits(u16::from_le_bytes(actual[begin..begin + 2].try_into()?)).to_f32()
            };
            let wanted = expected[logical + i];
            ensure!(
                value.is_finite()
                    && (value - wanted).abs()
                        <= if (6..10).contains(&mode) {
                            0.02 + 0.13 * wanted.abs()
                        } else {
                            0.002 + 0.002 * wanted.abs()
                        },
                "mismatch: {case:?} index={i} expected={wanted} actual={value}"
            );
        }
        at += guarded;
        logical += count;
        let before = u32::from_le_bytes(times[..4].try_into()?);
        let after = u32::from_le_bytes(times[4..].try_into()?);
        println!(
            "width={width} rows={rows} mode={mode} offset={offset} cycles={}",
            after.wrapping_sub(before)
        );
    }
    println!("cases={} numericalTest=PASS", cases.len());
    Ok(())
}
