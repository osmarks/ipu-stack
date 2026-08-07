//! Machine-readable ABI contracts for tile-local kernel calls.

use crate::{
    ComputeStep, GemmKernelMode, KernelRequirements, KernelRun, LowProgram, LowShard, LowShardId,
    Precision, StepProfile, StorageError, TileAddress, TileKernel, TileKernelSpec, TileWork,
    TileWorkList, view_byte_spans,
};
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompilation {
    pub source: &'static str,
    pub name: String,
    pub flags: Vec<String>,
    pub retained_symbols: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelBuildPlan {
    pub compilations: Vec<KernelCompilation>,
    gemm_rows: BTreeMap<Precision, Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedKernelCall {
    pub symbol: String,
    pub arguments: Vec<u32>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelAbiError {
    #[error("kernel requirements do not match the tile-kernel family")]
    RequirementMismatch,
    #[error("kernel run has {actual} pointer operands, ABI requires {expected}")]
    PointerArity { expected: usize, actual: usize },
    #[error("kernel operand {0} is fragmented into multiple views")]
    FragmentedOperand(usize),
    #[error("kernel {0:?} has no device implementation")]
    Unavailable(TileKernelSpec),
    #[error("GEMM output view does not have a matrix row axis")]
    MissingGemmRows,
    #[error("GEMM precision {precision:?} needs more than two row specializations: {rows:?}")]
    TooManyGemmRows {
        precision: Precision,
        rows: Vec<u32>,
    },
    #[error("kernel element count overflowed")]
    ElementCountOverflow,
    #[error("GEMM row count {0} is not present in the compilation plan")]
    UnplannedGemmRows(u32),
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelMaterializationError {
    #[error(transparent)]
    Abi(#[from] KernelAbiError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("shard {0} has no assigned address")]
    UnplacedShard(u32),
    #[error("kernel operand view is not one contiguous physical byte span")]
    FragmentedView,
    #[error("placed kernel address overflowed")]
    AddressOverflow,
}

impl KernelBuildPlan {
    /// Derives device objects from the finalized schedule, so row variants are
    /// compiler specializations rather than a fixed collection of binaries.
    pub fn from_program(program: &LowProgram) -> Result<Self, KernelAbiError> {
        let mut rows = BTreeMap::<Precision, BTreeSet<u32>>::new();
        for tile in &program.tiles {
            collect_kernel_rows(tile, &mut rows)?;
        }
        let mut plan = Self::default();
        for (precision, values) in rows {
            let values = values.into_iter().collect::<Vec<_>>();
            if values.len() > 2 {
                return Err(KernelAbiError::TooManyGemmRows {
                    precision,
                    rows: values,
                });
            }
            let small = values[0];
            let large = *values.last().expect("nonempty GEMM row set");
            let (source, prefix) = match precision {
                Precision::F16 => ("gemm_f16_64_amp.S", "f16"),
                Precision::F32 => ("gemm_f32_64_amp.S", "f32"),
                Precision::F8F143 { .. } => continue,
            };
            let symbols = [
                format!("ipu_stack_gemm_{prefix}_init_small_rows"),
                format!("ipu_stack_gemm_{prefix}_init_large_rows"),
                format!("ipu_stack_gemm_{prefix}_accumulate_small_rows"),
                format!("ipu_stack_gemm_{prefix}_accumulate_large_rows"),
            ];
            plan.compilations.push(KernelCompilation {
                source,
                name: format!("gemm_{prefix}_r{small}_r{large}"),
                flags: vec![
                    format!("-DGEMM_SMALL_ROWS={small}"),
                    format!("-DGEMM_LARGE_ROWS={large}"),
                ],
                retained_symbols: symbols.into_iter().collect(),
            });
            plan.gemm_rows.insert(precision, values);
        }
        Ok(plan)
    }

    pub fn call(&self, run: &KernelRun) -> Result<PlannedKernelCall, KernelAbiError> {
        let abi = validate_kernel_run(run)?;
        let TileKernel::Planned(kernel) = &run.kernel;
        if abi.availability != KernelAvailability::Implemented {
            return Err(KernelAbiError::Unavailable(kernel.clone()));
        }
        let symbol = match (&abi.symbols, kernel) {
            (KernelSymbols::Exact(symbol), _) => (*symbol).to_owned(),
            (
                KernelSymbols::RowSpecialized { small, large },
                TileKernelSpec::Gemm { multiply, .. },
            ) => {
                let rows = gemm_rows(run)?;
                let planned = self
                    .gemm_rows
                    .get(multiply)
                    .ok_or(KernelAbiError::UnplannedGemmRows(rows))?;
                if planned.first() == Some(&rows) {
                    (*small).to_owned()
                } else if planned.last() == Some(&rows) {
                    (*large).to_owned()
                } else {
                    return Err(KernelAbiError::UnplannedGemmRows(rows));
                }
            }
            _ => return Err(KernelAbiError::RequirementMismatch),
        };
        Ok(PlannedKernelCall {
            symbol,
            arguments: scalar_values(run, &abi)?,
        })
    }

    pub fn retained_symbols(&self) -> impl Iterator<Item = &str> {
        self.compilations
            .iter()
            .flat_map(|compilation| compilation.retained_symbols.iter().map(String::as_str))
    }
}

/// Resolves one scheduled call after placement has assigned each shard base.
/// Layout conversion supplies the byte offset; the build plan supplies the
/// linked specialization and ABI scalar values.
pub fn materialize_kernel_run(
    run: &KernelRun,
    shards: &[LowShard],
    shard_addresses: &BTreeMap<LowShardId, u32>,
    plan: &KernelBuildPlan,
) -> Result<ComputeStep, KernelMaterializationError> {
    materialize_kernel_run_with_addresses(run, shards, shard_addresses, plan, &BTreeMap::new())
}

pub fn materialize_kernel_run_with_addresses(
    run: &KernelRun,
    shards: &[LowShard],
    shard_addresses: &BTreeMap<LowShardId, u32>,
    plan: &KernelBuildPlan,
    overrides: &BTreeMap<LowShardId, TileAddress>,
) -> Result<ComputeStep, KernelMaterializationError> {
    let call = plan.call(run)?;
    let resolve = |view: &crate::ShardView| {
        let shard = shards.get(view.shard.index() as usize).ok_or(
            KernelMaterializationError::UnplacedShard(view.shard.index()),
        )?;
        let spans = view_byte_spans(shard, view)?;
        let [span] = spans.as_slice() else {
            return Err(KernelMaterializationError::FragmentedView);
        };
        let base = overrides.get(&view.shard).copied().unwrap_or_else(|| {
            TileAddress::Absolute(
                shard_addresses
                    .get(&view.shard)
                    .copied()
                    .unwrap_or_default(),
            )
        });
        if !overrides.contains_key(&view.shard) && !shard_addresses.contains_key(&view.shard) {
            return Err(KernelMaterializationError::UnplacedShard(
                view.shard.index(),
            ));
        }
        add_address_offset(base, span.offset)
    };
    let output_address = resolve(&run.output)?;
    let input_addresses = run
        .inputs
        .iter()
        .map(|operand| resolve(&operand.views[0]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ComputeStep {
        symbol: call.symbol,
        output_address,
        input_addresses,
        arguments: call.arguments,
        profile: StepProfile::default(),
    })
}

fn add_address_offset(
    address: TileAddress,
    offset: u32,
) -> Result<TileAddress, KernelMaterializationError> {
    Ok(match address {
        TileAddress::Absolute(address) => TileAddress::Absolute(
            address
                .checked_add(offset)
                .ok_or(KernelMaterializationError::AddressOverflow)?,
        ),
        TileAddress::RepeatPointer {
            index,
            offset: existing,
        } => TileAddress::RepeatPointer {
            index,
            offset: existing
                .checked_add(offset)
                .ok_or(KernelMaterializationError::AddressOverflow)?,
        },
    })
}

fn collect_kernel_rows(
    tile: &TileWorkList,
    rows: &mut BTreeMap<Precision, BTreeSet<u32>>,
) -> Result<(), KernelAbiError> {
    for work in &tile.work {
        match work {
            TileWork::Kernel(run) => {
                let abi = validate_kernel_run(run)?;
                let TileKernel::Planned(kernel) = &run.kernel;
                if abi.availability != KernelAvailability::Implemented {
                    return Err(KernelAbiError::Unavailable(kernel.clone()));
                }
                if let TileKernelSpec::Gemm { multiply, .. } = kernel {
                    rows.entry(*multiply).or_default().insert(gemm_rows(run)?);
                }
            }
            TileWork::Repeat(repeat) => collect_kernel_rows(&repeat.body, rows)?,
            TileWork::Exchange(_) => {}
        }
    }
    Ok(())
}

fn gemm_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    run.output
        .extents
        .get(rank.checked_sub(2).ok_or(KernelAbiError::MissingGemmRows)?)
        .map(|extent| extent.physical_end - extent.start)
        .filter(|&rows| rows != 0)
        .ok_or(KernelAbiError::MissingGemmRows)
}

fn scalar_values(run: &KernelRun, abi: &KernelAbi) -> Result<Vec<u32>, KernelAbiError> {
    let count = run
        .output
        .extents
        .iter()
        .try_fold(1u32, |product, extent| {
            product
                .checked_mul(extent.physical_end - extent.start)
                .ok_or(KernelAbiError::ElementCountOverflow)
        })?;
    abi.scalar_arguments
        .iter()
        .map(|argument| match argument.name {
            "element_count" => Ok(count),
            "scale_exponent" => match &run.kernel {
                TileKernel::Planned(TileKernelSpec::Gemm {
                    multiply: Precision::F8F143 { scale_exponent },
                    ..
                }) => Ok(u32::from_ne_bytes(i32::from(*scale_exponent).to_ne_bytes())),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            _ => Err(KernelAbiError::RequirementMismatch),
        })
        .collect()
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
        AccumulationPrecision, ComputeGraph, Layout, MemoryClass, OperandRequirement,
        OperatorRequirements, OutputAliasing, PipelineConfig, TensorFormat, TensorTiling,
        ToyCostModel, lower, lower_to_tiles,
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

    #[test]
    fn randomized_gemm_plans_compile_and_select_scheduled_row_specializations() {
        let mut random = fastrand::Rng::with_seed(0x7370_6563);
        for _ in 0..32 {
            let tiles = 1_u16 << random.u32(0..=3);
            let rows_per_tile = random.u32(1..=12);
            let mut graph = ComputeGraph::new();
            let left = graph
                .host_input("left", [u32::from(tiles) * rows_per_tile, 64])
                .unwrap();
            let right = graph.parameter("right", [64, 64]).unwrap();
            let result = graph.gemm(left, right).unwrap();
            graph.set_outputs([result]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_input(
                    left,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::amp_left(64, tiles),
                    },
                )
                .with_input(
                    right,
                    TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::amp_right(64, tiles),
                    },
                );
            let mid = lower(&graph, &config, &ToyCostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let plan = KernelBuildPlan::from_program(&low).unwrap();
            let addresses = low
                .shards
                .iter()
                .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(plan.compilations.len(), 1);
            assert!(
                plan.compilations[0]
                    .flags
                    .iter()
                    .any(|flag| flag == &format!("-DGEMM_SMALL_ROWS={rows_per_tile}"))
            );
            for run in low
                .tiles
                .iter()
                .flat_map(|tile| &tile.work)
                .filter_map(|work| {
                    if let TileWork::Kernel(run) = work {
                        Some(run)
                    } else {
                        None
                    }
                })
            {
                let call = plan.call(run).unwrap();
                assert!(plan.retained_symbols().any(|symbol| symbol == call.symbol));
                assert!(call.arguments.is_empty());
                let compute = materialize_kernel_run(run, &low.shards, &addresses, &plan).unwrap();
                assert_eq!(compute.symbol, call.symbol);
                assert_eq!(compute.input_addresses.len(), 2);
            }
        }
    }
}
