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
use ipu_runtime::Runtime;
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    /// Additionally require bitwise equality; useful for instruction scheduling changes.
    #[arg(long)]
    exact: bool,
    #[arg(long, default_value = "c600-init.ipucfg")]
    configuration: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/kernel-upgrade")]
    output: PathBuf,
    #[arg(long, default_value = "artifacts/layout-sweep/device.lock")]
    device_lock: PathBuf,
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
    let device = PathBuf::from("device").canonicalize()?;
    let mut source = format!(
        "#include \"{}\"\n",
        device.join("static_runtime.S").display()
    );
    for name in ["gelu_f16.S", "reduce_add_f16.S"] {
        // Namespace the reference's symbols, local labels and GELU macros.
        source += &fs::read_to_string(args.reference.join(name))?
            .replace("ipu_stack_", "reference_ipu_stack_")
            .replace(".L", ".Lreference_")
            .replace("GELU_", "REFERENCE_GELU_")
            .replace("LOAD_GELU_CONSTANTS", "LOAD_REFERENCE_GELU_CONSTANTS");
        source.push('\n');
        source += &fs::read_to_string(device.join(name))?;
    }
    source += "\n.section .text.kernel_test_benign_fp,\"ax\",@progbits\n.supervisor\n.p2align 2\n.globl kernel_test_benign_fp\nkernel_test_benign_fp:\nget $m0, $FP_ICTL\nldconst $m1, 0xfffffff8\nand $m0, $m0, $m1\nput $FP_ICTL, $m0\nbr $m10\n";
    source += "\n.section .text.kernel_test_strict_fp,\"ax\",@progbits\n.supervisor\n.p2align 2\n.globl kernel_test_strict_fp\nkernel_test_strict_fp:\nget $m0, $FP_ICTL\nsetzi $m1, 7\nor $m0, $m0, $m1\nput $FP_ICTL, $m0\nbr $m10\n";
    let wrapper = args.output.join("kernel_equivalence.S");
    fs::write(&wrapper, source)?;
    let mut programs = (0..1472)
        .map(|tile| TileProgram {
            tile,
            steps: vec![],
        })
        .collect::<Vec<_>>();
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
    for elements in [2, 6, 14, 16, 18, 30, 94, 96, 98, 1408, 2208] {
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
    for partials in [2, 3, 4, 15, 27, 28] {
        for elements in [8, 16, 40, 48, 56, 96, 104, 576, 1472, 2208] {
            for inplace in [false, true] {
                let values = (0..elements * partials)
                    .map(|_| f16::from_f32((random.f32() - 0.5) * 0.5).to_bits())
                    .collect();
                cases.push((values, partials, inplace));
            }
        }
    }
    let mut output_addresses = Vec::new();
    for (index, (values, partials, inplace)) in cases.iter().enumerate() {
        let tile = index as u16;
        let elements = values.len() as u32 / partials;
        let mut bytes = Vec::new();
        let mut append = |halves: &[u16]| {
            while bytes.len() % 8 != 0 {
                bytes.push(0);
            }
            let address = 0x5c000 + bytes.len() as u32;
            bytes.extend(halves.iter().flat_map(|half| half.to_le_bytes()));
            address
        };
        let input = append(values);
        let canary = append(&[0x3555; 8]);
        let mut initial = vec![0x3555; elements as usize + 8];
        if *inplace {
            initial[..elements as usize].copy_from_slice(&values[..elements as usize]);
        }
        let old = append(&initial);
        let new = append(&initial);
        output_addresses.push(new);
        let name = if *partials == 1 {
            "ipu_stack_gelu_tanh_approx_f16"
        } else {
            "ipu_stack_reduce_sum_f16"
        };
        let steps = &mut programs[index].steps;
        // Extreme finite GELU inputs overflow the existing polynomial. Compare
        // their outputs with benign exceptions, preserving other FP controls.
        if *partials == 1 {
            steps.push(call("kernel_test_benign_fp", 0, &[], &[0]));
        }

        for (symbol, output) in [(format!("reference_{name}"), old), (name.into(), new)] {
            if *partials == 1 && output == new {
                steps.push(call("kernel_test_strict_fp", 0, &[], &[0]));
            }
            let initial = if *inplace { output } else { input };
            let inputs = if *partials == 1 {
                vec![initial]
            } else {
                vec![initial, input + elements * 2]
            };
            let scalars = if *partials == 1 {
                vec![elements]
            } else {
                vec![partials - 1, elements]
            };
            let mut step = call(&symbol, output, &inputs, &scalars);
            if let TileStep::Compute(call) = &mut step {
                let base = if output == old { 0x7f000 } else { 0x7f008 };
                call.profile = StepProfile {
                    before: Some(base),
                    after: Some(base + 4),
                };
            }
            steps.push(step);
        }
        if args.exact {
            steps.push(call(
                "ipu_stack_static_assert_equal_u32",
                old,
                &[old, new],
                &[elements / 2],
            ));
        }
        for output in [old, new] {
            steps.push(call(
                "ipu_stack_static_assert_equal_u32",
                canary,
                &[output + elements * 2, canary],
                &[4],
            ));
        }
        data.push(TileProgramData {
            tile,
            address: 0x7f000,
            data: vec![0; 16],
        });
        data.push(TileProgramData {
            tile,
            address: 0x5c000,
            data: bytes,
        });
    }
    let topology = ipu_exchange::Topology::c600();
    let mut outputs = vec![Binding {
        name: "timestamps".into(),
        dtype: "u32".into(),
        shape: vec![cases.len() as u32, 4],
        slices: (0..cases.len())
            .map(|tile| {
                Ok(RegionSlice {
                    tile: u32::from(topology.physical(tile as u16)?),
                    tile_address: 0x7f000,
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
    let executed = session.invoke_streaming_deferred("run", &[0; 4])?;
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
                let x = f16::from_bits(values[element]).to_f64();
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
