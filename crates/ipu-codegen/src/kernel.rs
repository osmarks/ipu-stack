//! Machine-readable ABI contracts for tile-local kernel calls.

use crate::{GemmKernelMode, KernelRequirements, KernelRun, Precision, TileKernel, TileKernelSpec};

pub const OUTPUT_REGISTER: u8 = 2;
pub const FIRST_INPUT_REGISTER: u8 = 3;
pub const RETURN_REGISTER: u8 = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSymbols {
    Exact(&'static str),
    RowSpecialized {
        small: &'static str,
        large: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelAvailability {
    Implemented,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarArgument {
    pub register: u8,
    pub name: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    pub symbols: KernelSymbols,
    pub availability: KernelAvailability,
    pub output_register: u8,
    pub input_registers: Vec<u8>,
    pub scalar_arguments: Vec<ScalarArgument>,
    pub return_register: u8,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelAbiError {
    #[error("kernel requirements do not match the tile-kernel family")]
    RequirementMismatch,
    #[error("kernel run has {actual} pointer operands, ABI requires {expected}")]
    PointerArity { expected: usize, actual: usize },
    #[error("kernel operand {0} is fragmented into multiple views")]
    FragmentedOperand(usize),
}

pub fn tile_kernel_abi(
    kernel: &TileKernelSpec,
    requirements: &KernelRequirements,
) -> Result<KernelAbi, KernelAbiError> {
    let precision = match requirements {
        KernelRequirements::Operator(requirements) => requirements.output.format.precision,
        KernelRequirements::Conversion { output, .. } => output.format.precision,
    };
    let (symbols, availability, inputs, scalars) = match kernel {
        TileKernelSpec::Gemm { multiply, mode, .. } => {
            if !matches!(requirements, KernelRequirements::Operator(_)) {
                return Err(KernelAbiError::RequirementMismatch);
            }
            let symbols = gemm_symbols(*multiply, *mode);
            let scalars = if matches!(multiply, Precision::F8F143 { .. }) {
                scalar_arguments(2, &["scale_exponent"])
            } else {
                Vec::new()
            };
            (symbols.0, symbols.1, 2, scalars)
        }
        TileKernelSpec::Gelu => (
            exact_symbol(
                precision,
                "ipu_stack_gelu_exact_f16",
                "ipu_stack_gelu_exact_f32",
            ),
            KernelAvailability::Required,
            1,
            scalar_arguments(1, &["element_count"]),
        ),
        TileKernelSpec::Add => (
            exact_symbol(precision, "ipu_stack_add_f16", "ipu_stack_add_f32"),
            KernelAvailability::Required,
            2,
            scalar_arguments(
                2,
                &[
                    "element_count",
                    "left_broadcast_stride",
                    "right_broadcast_stride",
                ],
            ),
        ),
        TileKernelSpec::FlashAttention { .. } => (
            exact_symbol(
                precision,
                "ipu_stack_flash_attention_f16",
                "ipu_stack_flash_attention_f32",
            ),
            KernelAvailability::Required,
            3,
            scalar_arguments(3, &["descriptor_address"]),
        ),
        TileKernelSpec::Cast { from, to } => (
            KernelSymbols::Exact(cast_symbol(*from, *to)),
            KernelAvailability::Required,
            1,
            scalar_arguments(1, &["element_count"]),
        ),
        TileKernelSpec::Rearrange { .. } => (
            KernelSymbols::Exact("ipu_stack_rearrange"),
            KernelAvailability::Required,
            1,
            scalar_arguments(1, &["descriptor_address"]),
        ),
    };
    Ok(KernelAbi {
        symbols,
        availability,
        output_register: OUTPUT_REGISTER,
        input_registers: (0..inputs)
            .map(|index| FIRST_INPUT_REGISTER + index as u8)
            .collect(),
        scalar_arguments: scalars,
        return_register: RETURN_REGISTER,
    })
}

pub fn validate_kernel_run(run: &KernelRun) -> Result<KernelAbi, KernelAbiError> {
    let TileKernel::Planned(kernel) = &run.kernel;
    let abi = tile_kernel_abi(kernel, &run.requirements)?;
    if run.inputs.len() != abi.input_registers.len() {
        return Err(KernelAbiError::PointerArity {
            expected: abi.input_registers.len(),
            actual: run.inputs.len(),
        });
    }
    if let Some(index) = run
        .inputs
        .iter()
        .position(|operand| operand.views.len() != 1)
    {
        return Err(KernelAbiError::FragmentedOperand(index));
    }
    Ok(abi)
}

fn gemm_symbols(precision: Precision, mode: GemmKernelMode) -> (KernelSymbols, KernelAvailability) {
    let symbols = match (precision, mode) {
        (Precision::F16, GemmKernelMode::Initialize) => KernelSymbols::RowSpecialized {
            small: "ipu_stack_gemm_f16_init_small_rows",
            large: "ipu_stack_gemm_f16_init_large_rows",
        },
        (Precision::F16, GemmKernelMode::Accumulate) => KernelSymbols::RowSpecialized {
            small: "ipu_stack_gemm_f16_accumulate_small_rows",
            large: "ipu_stack_gemm_f16_accumulate_large_rows",
        },
        (Precision::F32, GemmKernelMode::Initialize) => KernelSymbols::RowSpecialized {
            small: "ipu_stack_gemm_f32_init_small_rows",
            large: "ipu_stack_gemm_f32_init_large_rows",
        },
        (Precision::F32, GemmKernelMode::Accumulate) => KernelSymbols::RowSpecialized {
            small: "ipu_stack_gemm_f32_accumulate_small_rows",
            large: "ipu_stack_gemm_f32_accumulate_large_rows",
        },
        (Precision::F8F143 { .. }, GemmKernelMode::Initialize) => {
            KernelSymbols::Exact("ipu_stack_gemm_f8_init")
        }
        (Precision::F8F143 { .. }, GemmKernelMode::Accumulate) => {
            KernelSymbols::Exact("ipu_stack_gemm_f8_accumulate")
        }
    };
    let availability = if matches!(precision, Precision::F8F143 { .. }) {
        KernelAvailability::Required
    } else {
        KernelAvailability::Implemented
    };
    (symbols, availability)
}

fn exact_symbol(
    precision: Precision,
    f16_symbol: &'static str,
    f32_symbol: &'static str,
) -> KernelSymbols {
    KernelSymbols::Exact(match precision {
        Precision::F16 => f16_symbol,
        Precision::F32 => f32_symbol,
        Precision::F8F143 { .. } => "ipu_stack_unsupported_f8_kernel",
    })
}

fn cast_symbol(from: Precision, to: Precision) -> &'static str {
    match (from, to) {
        (Precision::F16, Precision::F32) => "ipu_stack_cast_f16_f32",
        (Precision::F32, Precision::F16) => "ipu_stack_cast_f32_f16",
        (Precision::F8F143 { .. }, Precision::F16) => "ipu_stack_cast_f8_f16",
        (Precision::F8F143 { .. }, Precision::F32) => "ipu_stack_cast_f8_f32",
        (Precision::F16, Precision::F8F143 { .. }) => "ipu_stack_cast_f16_f8",
        (Precision::F32, Precision::F8F143 { .. }) => "ipu_stack_cast_f32_f8",
        _ => "ipu_stack_cast_identity",
    }
}

fn scalar_arguments(input_count: u8, names: &[&'static str]) -> Vec<ScalarArgument> {
    names
        .iter()
        .enumerate()
        .map(|(index, name)| ScalarArgument {
            register: FIRST_INPUT_REGISTER + input_count + index as u8,
            name,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccumulationPrecision, Layout, MemoryClass, OperandRequirement, OperatorRequirements,
        OutputAliasing, TensorFormat, TensorTiling,
    };

    #[test]
    fn randomized_gemm_abis_resolve_to_retained_symbols() {
        let mut random = fastrand::Rng::with_seed(0x6162_6921);
        for _ in 0..64 {
            let precision = if random.bool() {
                Precision::F16
            } else {
                Precision::F32
            };
            let mode = if random.bool() {
                GemmKernelMode::Initialize
            } else {
                GemmKernelMode::Accumulate
            };
            let format = TensorFormat {
                precision,
                layout: Layout {
                    order: crate::ElementOrder::RowMajor,
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            };
            let operand = OperandRequirement::new(format, 8);
            let requirements = KernelRequirements::Operator(OperatorRequirements {
                inputs: vec![operand.clone(), operand.clone()],
                output: operand,
                output_aliasing: OutputAliasing::Fresh,
                memory_relations: Vec::new(),
            });
            let abi = tile_kernel_abi(
                &TileKernelSpec::Gemm {
                    multiply: precision,
                    accumulate: AccumulationPrecision::F32,
                    mode,
                },
                &requirements,
            )
            .unwrap();
            assert_eq!(abi.availability, KernelAvailability::Implemented);
            assert!(matches!(abi.symbols, KernelSymbols::RowSpecialized { .. }));
            assert_eq!(abi.input_registers, [3, 4]);
            assert_eq!(abi.return_register, 10);
        }
    }
}
