//! Numerically check and time kernels against a supplied reference source tree.
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
    #[command(flatten)]
    device: ipu_tests::KernelDeviceOptions,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, default_value = "device")]
    source: PathBuf,
    /// Additionally require bitwise equality; useful for instruction scheduling changes.
    #[arg(long)]
    exact: bool,
    /// Compare the row-broadcast bias + GeLU kernel instead of GeLU/reduction.
    #[arg(long)]
    bias_gelu: bool,
    #[arg(long, default_value_t = 1, requires = "bias_gelu", value_parser = clap::value_parser!(u32).range(1..=3))]
    bias_rows: u32,
    /// Exercise word-aligned views that cannot use 64-bit memory instructions.
    #[arg(long)]
    offset_four: bool,
    #[arg(long, default_value = "artifacts/layout-sweep/kernel-upgrade")]
    output: PathBuf,
}

fn call(symbol: &str, output: u32, inputs: &[u32], arguments: &[u32]) -> TileStep {
    TileStep::Compute(ComputeStep {
        symbol: symbol.into(),
        output_address: TileAddress::Absolute(output),
        input_addresses: inputs.iter().copied().map(TileAddress::Absolute).collect(),
        arguments: arguments.to_vec(),
        profile: StepProfile::default(),
    })
}

fn main() -> Result<()> {
    ipu_runtime::init_tracing();
    let args = Arguments::parse();
    fs::create_dir_all(&args.output)?;
    let device = args.source.canonicalize()?;
    let mut source = format!(
        "#include \"{}\"\n",
        device.join("static_runtime.S").display()
    );
    if args.bias_gelu {
        source += "#define GELU_WITH_BIAS\n#define REFERENCE_GELU_WITH_BIAS\n";
    }
    for name in ["gelu_f16.S", "reduce_add_f16.S"] {
        // Namespace the reference's symbols, local labels and GELU macros.
        source += &fs::read_to_string(args.reference.join(name))?
            .replace("ipu_stack_", "")
            .replace("gelu_tanh_approx_f16", "reference_gelu_tanh_approx_f16")
            .replace("reduce_sum_f16", "reference_reduce_sum_f16")
            .replace("bias_gelu_f16", "reference_bias_gelu_f16")
            .replace(".L", ".Lreference_")
            .replace("GELU_", "REFERENCE_GELU_")
            .replace("LOAD_GELU_CONSTANTS", "LOAD_REFERENCE_GELU_CONSTANTS");
        source.push('\n');
        source += &fs::read_to_string(device.join(name))?;
    }
    source += "\n.section .text.kernel_test_benign_fp,\"ax\",@progbits\n.supervisor\n.p2align 2\n.globl kernel_test_benign_fp\nkernel_test_benign_fp:\nget $m0, $FP_ICTL\nldconst $m1, 0xfffffff8\nand $m0, $m0, $m1\nput $FP_ICTL, $m0\nbr $m10\n";
    source += "\n.section .text.kernel_test_strict_fp,\"ax\",@progbits\n.supervisor\n.p2align 2\n.globl kernel_test_strict_fp\nkernel_test_strict_fp:\nget $m0, $FP_ICTL\nsetzi $m1, 7\nor $m0, $m0, $m1\nput $FP_ICTL, $m0\nbr $m10\n";
    if args.bias_gelu {
        // MIX reads old accumulator values even when the destination is dead.
        // Poison those values to verify the kernel initializes its own state.
        source += r#"
.section .text.kernel_test_poison_acc,"ax",@progbits
.supervisor
.p2align 2
.globl kernel_test_poison_acc
kernel_test_poison_acc:
setzi $m0, .Lpoison_acc
runall $m0, $mzero, 0
sync TEXCH_SYNCZONE_LOCAL
br $m10
.worker
.Lpoison_acc:
setzi $a0, 0
setzi $a1, 0
ldconst $a2, 0x7fc07fc0
ldconst $a3, 0x7fc07fc0
f16v4istacc $a0:1, $a0:1, $a2:3, 0
exitz $mzero
"#;
    }
    let wrapper = args.output.join("kernel_equivalence.S");
    fs::write(&wrapper, source)?;
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let finite = (0..=u16::MAX)
        .filter(|bits| bits & 0x7c00 != 0x7c00)
        .collect::<Vec<_>>();
    let mut cases = finite
        .chunks(1024)
        .map(|chunk| (chunk.to_vec(), 1u32, false))
        .collect::<Vec<_>>();
    let mut random = fastrand::Rng::with_seed(0x67656c75);
    // Odd numbers of pairs, partial worker waves, and in-place output.
    for elements in [2, 6, 14, 16, 18, 30, 94, 96, 98, 1408, 2152, 2208] {
        for inplace in [false, true] {
            cases.push((
                (0..elements)
                    .map(|_| finite[random.usize(..finite.len())])
                    .collect(),
                1,
                inplace,
            ));
        }
    }
    for partials in [2, 3, 4, 15, 27, 28]
        .into_iter()
        .filter(|_| !args.bias_gelu)
    {
        for elements in [8, 16, 40, 48, 56, 96, 104, 576, 1472, 2208] {
            for inplace in [false, true] {
                let values = (0..elements * partials)
                    .map(|_| f16::from_f32((random.f32() - 0.5) * 0.5).to_bits())
                    .collect();
                cases.push((values, partials, inplace));
            }
        }
    }
    if args.bias_gelu {
        for (values, _, _) in &mut cases {
            *values = values.repeat(args.bias_rows as usize);
        }
    }
    let bias_value = |column: usize| f16::from_f32((column as i32 % 17 - 8) as f32 / 16.0);
    let mut output_addresses = Vec::new();
    for (index, (values, partials, inplace)) in cases.iter().enumerate() {
        let tile = index as u16;
        let elements = values.len() as u32 / partials;
        let mut bytes = Vec::new();
        let mut append = |halves: &[u16]| {
            while bytes.len() % 8 != if args.offset_four { 4 } else { 0 } {
                bytes.push(0);
            }
            let address = 0x80000 + bytes.len() as u32;
            bytes.extend(halves.iter().flat_map(|half| half.to_le_bytes()));
            address
        };
        let input = append(values);
        let bias = args.bias_gelu.then(|| {
            append(
                &(0..elements as usize / args.bias_rows as usize)
                    .map(|column| bias_value(column).to_bits())
                    .collect::<Vec<_>>(),
            )
        });
        let canary = append(&[0x3555; 8]);
        let mut initial = vec![0x3555; elements as usize + 8];
        if *inplace {
            initial[..elements as usize].copy_from_slice(&values[..elements as usize]);
        }
        let old = append(&initial);
        let new = append(&initial);
        output_addresses.push(new);
        let name = if args.bias_gelu {
            "bias_gelu_f16"
        } else if *partials == 1 {
            "gelu_tanh_approx_f16"
        } else {
            "reduce_sum_f16"
        };
        let mut steps = Vec::new();
        // Extreme finite GELU inputs overflow the existing polynomial. Compare
        // their outputs with benign exceptions, preserving other FP controls.
        if *partials == 1 {
            steps.push(call("kernel_test_benign_fp", 0, &[], &[0]));
        }

        for (symbol, output) in [(format!("reference_{name}"), old), (name.into(), new)] {
            if *partials == 1 && output == new {
                steps.push(call("kernel_test_strict_fp", 0, &[], &[0]));
                if args.bias_gelu {
                    steps.push(call("kernel_test_poison_acc", 0, &[], &[0]));
                }
            }
            let initial = if *inplace { output } else { input };
            let inputs = if let Some(bias) = bias {
                vec![initial, bias]
            } else if *partials == 1 {
                vec![initial]
            } else {
                vec![initial, input + elements * 2]
            };
            let scalars = if args.bias_gelu {
                vec![args.bias_rows, elements / args.bias_rows]
            } else if *partials == 1 {
                vec![elements]
            } else {
                vec![partials - 1, elements]
            };
            let mut step = call(&symbol, output, &inputs, &scalars);
            if let TileStep::Compute(call) = &mut step {
                let base = if output == old { 0xe7000 } else { 0xe7008 };
                call.profile = StepProfile {
                    before: Some(base),
                    after: Some(base + 4),
                };
            }
            steps.push(step);
        }
        if args.exact {
            steps.push(call(
                "static_assert_equal_u32",
                old,
                &[old, new],
                &[elements / 2],
            ));
        }
        for output in [old, new] {
            steps.push(call(
                "static_assert_equal_u32",
                canary,
                &[output + elements * 2, canary],
                &[4],
            ));
        }
        programs.push(TileProgram { tile, steps });
        data.push(TileProgramData {
            tile,
            address: 0xe7000,
            data: vec![0; 16],
        });
        data.push(TileProgramData {
            tile,
            address: 0x80000,
            data: bytes,
        });
    }
    let topology = ipu_target::ipu21::fabric::Topology::c600();
    let mut outputs = vec![Binding {
        name: "timestamps".into(),
        dtype: "u32".into(),
        shape: vec![cases.len() as u32, 4],
        slices: (0..cases.len())
            .map(|tile| {
                Ok(RegionSlice {
                    tile: u32::from(topology.physical(tile as u16)?),
                    tile_address: 0xe7000,
                    file_offset: tile as u64 * 16,
                    size: 16,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    }];
    let mut offset = 0;
    let mut slices = Vec::new();
    for (tile, (values, partials, _)) in cases.iter().enumerate() {
        let size = values.len() as u64 / u64::from(*partials) * 2;
        slices.push(RegionSlice {
            tile: u32::from(topology.physical(tile as u16)?),
            tile_address: output_addresses[tile],
            file_offset: offset,
            size,
        });
        offset += size;
    }
    outputs.push(Binding {
        name: "values".into(),
        dtype: "f16".into(),
        shape: vec![offset as u32 / 2],
        slices,
    });
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
                cases.iter().map(|(values, partials, inplace)| {
                    format!(
                        "elements={} partials={partials} inplace={inplace}",
                        values.len() / *partials as usize
                    )
                }),
            );
        })?;
    let output = session.finish(&executed)?;
    ipu_tests::completion::diagnose_completion(
        runtime,
        &application,
        std::time::Duration::from_secs(10),
    )?;
    fs::write(args.output.join("output.bin"), &output)?;
    let mut position = cases.len() * 16;
    let mut maximum_error = [0f64; 2];
    for (case, (values, partials, _)) in cases.iter().enumerate() {
        let elements = values.len() / *partials as usize;
        for element in 0..elements {
            let actual = f16::from_bits(u16::from_le_bytes(
                output[position..position + 2].try_into()?,
            ))
            .to_f64();
            position += 2;
            let reference = if *partials == 1 {
                let mut x = f16::from_bits(values[element]).to_f64();
                if args.bias_gelu {
                    let width = elements / args.bias_rows as usize;
                    x = f16::from_f64(x + bias_value(element % width).to_f64()).to_f64();
                }
                x * 0.5
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            } else {
                values
                    .chunks_exact(elements)
                    .map(|partial| f16::from_bits(partial[element]).to_f64())
                    .sum()
            };
            let error = (actual - reference).abs();
            ensure!(
                error <= 0.01 + reference.abs() * 0.002,
                "case={case} element={element} actual={actual} reference={reference}"
            );
            let family = usize::from(*partials != 1);
            maximum_error[family] = maximum_error[family].max(error);
        }
    }
    let mut measurements = Vec::new();
    for (tile, (values, partials, inplace)) in cases.iter().enumerate() {
        let cycles = output[tile * 16..tile * 16 + 16]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        measurements.push(serde_json::json!({
            "elements": values.len() as u32 / partials, "partials": partials,
            "bias_rows": args.bias_gelu.then_some(args.bias_rows),
            "offset_four": args.offset_four,
            "inplace": inplace, "old": cycles[1].wrapping_sub(cycles[0]),
            "new": cycles[3].wrapping_sub(cycles[2]),
        }));
    }
    fs::write(
        args.output.join("measurements.json"),
        serde_json::to_vec_pretty(&measurements)?,
    )?;
    println!(
        "kernelEquivalence=PASS cases={} finiteGeluInputs={} inPlaceAndCanaries=true maxAbsoluteErrors={maximum_error:?}",
        cases.len(),
        finite.len()
    );
    Ok(())
}
