//! Machine-readable ABI contracts for tile-local kernel calls.

use crate::mid::{AMP_COLUMN_MICRO, AMP_INNER_BLOCK};
use crate::{
    AmpOrder, BlockMajorOrder, ComputeStep, ElementOrder, GemmKernelMode, GemmWeightLoad,
    KernelRequirements, KernelRun, LowProgram, LowShard, LowShardId, Precision, StepProfile,
    StorageError, TileAddress, TileKernel, TileKernelSpec, TileWorkList, TileWorkRef,
    view_byte_spans,
};
use std::collections::{BTreeMap, BTreeSet};

pub const OUTPUT_REGISTER: u8 = 2;
pub const FIRST_INPUT_REGISTER: u8 = 3;
pub const RETURN_REGISTER: u8 = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSymbols {
    Exact(&'static str),
    RowSpecialized { small: String, large: String },
    AttentionSpecialized,
    AttentionStageSpecialized,
    RearrangeSpecialized,
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
    gemm_rows: BTreeMap<(Precision, GemmWeightLoad, u32, u32), Vec<u32>>,
    gemm_symbols: BTreeMap<(Precision, GemmWeightLoad, u32, u32, GemmKernelMode, u32), String>,
    attention_symbols: BTreeMap<AttentionKernelShape, String>,
    attention_stage_symbols: Vec<(TileKernelSpec, u32, String)>,
    rearrange_symbols: BTreeMap<(RearrangeTarget, u32, u32, u32, u32), String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RearrangeTarget {
    AmpLeft,
    AmpTransposedRight,
    BlockMajor { row_block: u16, column_block: u16 },
}

impl RearrangeTarget {
    fn from_order(order: ElementOrder) -> Option<Self> {
        match order {
            ElementOrder::Amp(AmpOrder::Left) => Some(Self::AmpLeft),
            ElementOrder::Amp(AmpOrder::TransposedRight) => Some(Self::AmpTransposedRight),
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }) => Some(Self::BlockMajor {
                row_block,
                column_block,
            }),
            _ => None,
        }
    }

    const fn codelet_index(self) -> u32 {
        match self {
            Self::AmpLeft => 0,
            Self::AmpTransposedRight => 1,
            Self::BlockMajor { .. } => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AttentionKernelShape {
    matrices: u32,
    query_rows: u32,
    key_rows: u32,
    query_dimension: u32,
    value_dimension: u32,
    scale_bits: u32,
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
    #[error("kernel element count overflowed")]
    ElementCountOverflow,
    #[error("GEMM row count {0} is not present in the compilation plan")]
    UnplannedGemmRows(u32),
    #[error("kernel {symbol} requires an element count divisible by {divisor}, got {count}")]
    UnsupportedElementCount {
        symbol: &'static str,
        count: u32,
        divisor: u32,
    },
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum KernelMaterializationError {
    #[error(transparent)]
    Abi(#[from] KernelAbiError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("shard {0} has no assigned address")]
    UnplacedShard(u32),
    #[error("kernel operand view of shard {shard} has {spans} physical byte spans")]
    FragmentedView { shard: u32, spans: usize },
    #[error("placed kernel address overflowed")]
    AddressOverflow,
}

fn specialized_gemm_symbol(
    prefix: &str,
    mode: GemmKernelMode,
    weight_suffix: &str,
    inner_block: u32,
    output_columns: u32,
    size: &str,
    small_rows: u32,
    large_rows: u32,
) -> String {
    let operation = match mode {
        GemmKernelMode::Initialize => "init",
        GemmKernelMode::Accumulate => "accumulate",
    };
    format!(
        "ipu_stack_gemm_{prefix}_{operation}_{size}_rows{weight_suffix}_k{inner_block}_c{output_columns}_r{small_rows}_r{large_rows}"
    )
}

impl KernelBuildPlan {
    /// Derives device objects from the finalized schedule, so row variants are
    /// compiler specializations rather than a fixed collection of binaries.
    pub fn from_program(program: &LowProgram) -> Result<Self, KernelAbiError> {
        let mut rows = BTreeMap::<(Precision, GemmWeightLoad, u32, u32), BTreeSet<u32>>::new();
        let mut gelu = false;
        let mut reduction_add = false;
        let mut rearrangements = BTreeSet::new();
        let mut attention = BTreeSet::new();
        let mut attention_stages = Vec::new();
        for tile in &program.tiles {
            collect_kernels(
                program,
                tile,
                &mut rows,
                &mut gelu,
                &mut reduction_add,
                &mut rearrangements,
                &mut attention,
                &mut attention_stages,
            )?;
        }
        let mut plan = Self::default();
        for ((precision, weights, inner_block, output_columns), values) in rows {
            let values = values.into_iter().collect::<Vec<_>>();
            let (source, prefix) = match precision {
                Precision::F16 => ("gemm_f16_amp.S", "f16"),
                Precision::F32 => ("gemm_f32_64_amp.S", "f32"),
                Precision::F8F143 { .. } => continue,
            };
            let weight_suffix = if weights == GemmWeightLoad::Interleaved {
                "_interleaved"
            } else {
                ""
            };
            for pair in values.chunks(2) {
                let small = pair[0];
                let large = *pair.last().expect("nonempty GEMM row pair");
                let symbols = [
                    (GemmKernelMode::Initialize, "small", small),
                    (GemmKernelMode::Initialize, "large", large),
                    (GemmKernelMode::Accumulate, "small", small),
                    (GemmKernelMode::Accumulate, "large", large),
                ]
                .map(|(mode, size, _)| {
                    specialized_gemm_symbol(
                        prefix,
                        mode,
                        weight_suffix,
                        inner_block,
                        output_columns,
                        size,
                        small,
                        large,
                    )
                });
                for (mode, row_index) in [
                    (GemmKernelMode::Initialize, 0usize),
                    (GemmKernelMode::Accumulate, 2usize),
                ] {
                    plan.gemm_symbols.insert(
                        (precision, weights, inner_block, output_columns, mode, small),
                        symbols[row_index].clone(),
                    );
                    if pair.len() == 2 {
                        plan.gemm_symbols.insert(
                            (precision, weights, inner_block, output_columns, mode, large),
                            symbols[row_index + 1].clone(),
                        );
                    }
                }
                let single_rows = pair.len() == 1;
                let mut flags = vec![
                    format!("-DGEMM_SMALL_ROWS={small}"),
                    format!("-DGEMM_LARGE_ROWS={large}"),
                    format!("-DGEMM_OUTPUT_COLUMNS={output_columns}"),
                    format!("-DGEMM_INNER_BLOCK_DIMENSION={inner_block}"),
                    format!("-DGEMM_INIT_SMALL_SYMBOL={}", symbols[0]),
                    format!("-DGEMM_INIT_LARGE_SYMBOL={}", symbols[1]),
                    format!("-DGEMM_ACCUMULATE_SMALL_SYMBOL={}", symbols[2]),
                    format!("-DGEMM_ACCUMULATE_LARGE_SYMBOL={}", symbols[3]),
                ];
                if single_rows {
                    flags.push("-DGEMM_SINGLE_ROWS=1".into());
                }
                if weights == GemmWeightLoad::Interleaved {
                    flags.push("-DGEMM_INTERLEAVED_WEIGHTS=1".into());
                }
                let retained_symbols = if single_rows {
                    vec![symbols[0].clone(), symbols[2].clone()]
                } else {
                    symbols.into_iter().collect()
                };
                plan.compilations.push(KernelCompilation {
                    source,
                    name: format!(
                        "gemm_{prefix}{weight_suffix}_k{inner_block}_c{output_columns}_r{small}_r{large}"
                    ),
                    flags,
                    retained_symbols,
                });
            }
            plan.gemm_rows
                .insert((precision, weights, inner_block, output_columns), values);
        }
        if gelu {
            plan.compilations.push(KernelCompilation {
                source: "gelu_f16.S",
                name: "gelu_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec!["ipu_stack_gelu_exact_f16".into()],
            });
        }
        if reduction_add {
            plan.compilations.push(KernelCompilation {
                source: "reduce_add_f16.S",
                name: "reduce_add_f16".into(),
                flags: Vec::new(),
                retained_symbols: vec![
                    "ipu_stack_reduce_sum_2_f16".into(),
                    "ipu_stack_reduce_sum_3_f16".into(),
                    "ipu_stack_reduce_sum_4_f16".into(),
                ],
            });
        }
        let has_rearrange_codelets = !rearrangements.is_empty();
        for (order, logical_rows, physical_rows, logical_columns, physical_columns) in
            rearrangements
        {
            let order_index = order.codelet_index();
            let (row_block, column_block) = match order {
                RearrangeTarget::BlockMajor {
                    row_block,
                    column_block,
                } => (row_block, column_block),
                _ => (AMP_INNER_BLOCK as u16, AMP_COLUMN_MICRO as u16),
            };
            let suffix = format!(
                "o{order_index}_r{logical_rows}_p{physical_rows}_c{logical_columns}_p{physical_columns}"
            );
            let vertex = format!("RearrangeRowMajorToAmpF16_{suffix}");
            let codelet = format!("__runCodelet_{vertex}");
            let call = format!("ipu_stack_rearrange_row_major_to_amp_f16_{suffix}");
            if order == RearrangeTarget::AmpTransposedRight
                && logical_rows == 64
                && physical_rows == 64
                && logical_columns == 16
                && physical_columns == 16
            {
                plan.compilations.push(KernelCompilation {
                    source: "rearrange_transposed_right_f16.S",
                    name: format!("rearrange_transposed_right_f16_{suffix}"),
                    flags: vec![format!("-DREARRANGE_CALL_SYMBOL={call}")],
                    retained_symbols: vec![call.clone()],
                });
                plan.rearrange_symbols.insert(
                    (
                        order,
                        logical_rows,
                        physical_rows,
                        logical_columns,
                        physical_columns,
                    ),
                    call,
                );
                continue;
            }
            plan.compilations.push(KernelCompilation {
                source: "rearrange_f16.cpp",
                name: format!("rearrange_f16_codelet_{suffix}"),
                flags: vec![
                    "-O2".into(),
                    format!("-DREARRANGE_TARGET_ORDER={order_index}"),
                    format!("-DREARRANGE_LOGICAL_ROWS={logical_rows}"),
                    format!("-DREARRANGE_PHYSICAL_ROWS={physical_rows}"),
                    format!("-DREARRANGE_LOGICAL_COLUMNS={logical_columns}"),
                    format!("-DREARRANGE_PHYSICAL_COLUMNS={physical_columns}"),
                    format!("-DREARRANGE_INNER_DIMENSION={AMP_COLUMN_MICRO}"),
                    format!("-DREARRANGE_ROW_BLOCK={row_block}"),
                    format!("-DREARRANGE_COLUMN_BLOCK={column_block}"),
                    format!("-DREARRANGE_VERTEX_NAME={vertex}"),
                ],
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "rearrange_f16.S",
                name: format!("rearrange_f16_wrapper_{suffix}"),
                flags: vec![
                    format!("-DREARRANGE_CALL_SYMBOL={call}"),
                    format!("-DREARRANGE_CODELET_SYMBOL={codelet}"),
                ],
                retained_symbols: vec![call.clone()],
            });
            plan.rearrange_symbols.insert(
                (
                    order,
                    logical_rows,
                    physical_rows,
                    logical_columns,
                    physical_columns,
                ),
                call,
            );
        }
        if has_rearrange_codelets || !attention.is_empty() || !attention_stages.is_empty() {
            plan.compilations.push(KernelCompilation {
                source: "worker_support.S",
                name: "worker_support".into(),
                flags: Vec::new(),
                retained_symbols: Vec::new(),
            });
        }
        for shape in attention {
            let suffix = format!(
                "m{}_q{}_k{}_d{}_v{}_{:08x}",
                shape.matrices,
                shape.query_rows,
                shape.key_rows,
                shape.query_dimension,
                shape.value_dimension,
                shape.scale_bits,
            );
            let call_symbol = format!("ipu_stack_flash_attention_online_f16_{suffix}");
            let vertex = format!("FlashAttentionOnlineF16_{suffix}");
            let codelet = format!("__runCodelet_{vertex}");
            let common_flags = vec![
                format!("-DATTENTION_MATRICES={}", shape.matrices),
                format!("-DATTENTION_QUERY_ROWS={}", shape.query_rows),
                format!("-DATTENTION_KEY_ROWS={}", shape.key_rows),
                format!("-DATTENTION_QUERY_DIMENSION={}", shape.query_dimension),
                format!("-DATTENTION_VALUE_DIMENSION={}", shape.value_dimension),
                format!("-DATTENTION_SCALE={}", f32::from_bits(shape.scale_bits)),
            ];
            let mut codelet_flags = common_flags;
            codelet_flags.push(format!("-DATTENTION_VERTEX_NAME={vertex}"));
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_online_f16.cpp",
                name: format!("flash_attention_codelet_{suffix}"),
                flags: codelet_flags,
                retained_symbols: Vec::new(),
            });
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_online_f16.S",
                name: format!("flash_attention_wrapper_{suffix}"),
                flags: vec![
                    format!("-DATTENTION_CALL_SYMBOL={call_symbol}"),
                    format!("-DATTENTION_CODELET_SYMBOL={codelet}"),
                ],
                retained_symbols: vec![call_symbol.clone()],
            });
            plan.attention_symbols.insert(shape, call_symbol);
        }
        if !attention_stages.is_empty() {
            let mut query_rows = attention_stages
                .iter()
                .map(|(_, rows)| *rows)
                .collect::<BTreeSet<_>>();
            let small_query = query_rows
                .pop_first()
                .ok_or(KernelAbiError::RequirementMismatch)?;
            let large_query = query_rows.pop_last().unwrap_or(small_query);
            let mut key_rows = attention_stages
                .iter()
                .filter_map(|(kernel, _)| match kernel {
                    TileKernelSpec::AttentionSoftmax {
                        key_block_columns, ..
                    } => Some(*key_block_columns),
                    _ => None,
                })
                .collect::<BTreeSet<_>>();
            let small_key = key_rows
                .pop_first()
                .ok_or(KernelAbiError::RequirementMismatch)?;
            let large_key = key_rows.pop_last().unwrap_or(small_key);
            let configuration: Option<(u32, u32, u32, u32)> =
                attention_stages
                    .iter()
                    .fold(None, |configuration, (kernel, _)| match kernel {
                        TileKernelSpec::AttentionSoftmax { head_dimension, .. } => {
                            Some(configuration.unwrap_or((*head_dimension, 0, 0, AMP_INNER_BLOCK)))
                        }
                        TileKernelSpec::AttentionMerge {
                            value_dimension,
                            padded_value_dimension,
                            key_block_columns,
                            ..
                        } => {
                            let mut value = configuration.unwrap_or((
                                0,
                                *value_dimension,
                                *padded_value_dimension,
                                *key_block_columns,
                            ));
                            value.1 = *value_dimension;
                            value.2 = *padded_value_dimension;
                            value.3 = *key_block_columns;
                            Some(value)
                        }
                        _ => configuration,
                    });
            let (head_dimension, value_dimension, padded_value_dimension, key_block_columns) =
                configuration.ok_or(KernelAbiError::RequirementMismatch)?;
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_f16.cpp",
                name: format!(
                    "flash_attention_blocks_q{small_query}_q{large_query}_d{head_dimension}_v{value_dimension}"
                ),
                flags: vec![
                    "-Os".into(),
                    format!("-DATTENTION_HEAD_DIMENSION={head_dimension}"),
                    format!(
                        "-DATTENTION_PADDED_HEAD_DIMENSION={}",
                        head_dimension.div_ceil(16) * 16
                    ),
                    format!("-DATTENTION_VALUE_DIMENSION={value_dimension}"),
                    format!("-DATTENTION_PADDED_VALUE_DIMENSION={padded_value_dimension}"),
                    format!("-DATTENTION_KEY_BLOCK_COLUMNS={key_block_columns}"),
                    format!("-DATTENTION_SMALL_QUERY_ROWS={small_query}"),
                    format!("-DATTENTION_LARGE_QUERY_ROWS={large_query}"),
                    format!("-DATTENTION_SMALL_KEY_ROWS={small_key}"),
                    format!("-DATTENTION_LARGE_KEY_ROWS={large_key}"),
                ],
                retained_symbols: Vec::new(),
            });
            let mut retained_symbols = Vec::new();
            for (kernel, rows) in attention_stages {
                let size = if rows == small_query {
                    "small"
                } else {
                    "large"
                };
                let symbol = match &kernel {
                    TileKernelSpec::AttentionSoftmax {
                        key_block_columns, ..
                    } => {
                        let key_size = if *key_block_columns == small_key {
                            "small"
                        } else {
                            "large"
                        };
                        format!("ipu_stack_attention_softmax_{size}_query_{key_size}_key_f16")
                    }
                    TileKernelSpec::AttentionMerge { .. } => {
                        format!("ipu_stack_attention_merge_{size}_query_f16")
                    }
                    _ => return Err(KernelAbiError::RequirementMismatch),
                };
                if !retained_symbols.contains(&symbol) {
                    retained_symbols.push(symbol.clone());
                }
                plan.attention_stage_symbols.push((kernel, rows, symbol));
            }
            plan.compilations.push(KernelCompilation {
                source: "flash_attention_f16.S",
                name: "flash_attention_blocks_wrapper".into(),
                flags: Vec::new(),
                retained_symbols,
            });
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
                KernelSymbols::RowSpecialized { .. },
                TileKernelSpec::Gemm {
                    multiply,
                    mode,
                    weights,
                    inner_block,
                    output_columns,
                    ..
                },
            ) => {
                let rows = gemm_rows(run)?;
                let planned = self
                    .gemm_rows
                    .get(&(*multiply, *weights, *inner_block, *output_columns))
                    .ok_or(KernelAbiError::UnplannedGemmRows(rows))?;
                if !planned.contains(&rows) {
                    return Err(KernelAbiError::UnplannedGemmRows(rows));
                }
                self.gemm_symbols
                    .get(&(
                        *multiply,
                        *weights,
                        *inner_block,
                        *output_columns,
                        *mode,
                        rows,
                    ))
                    .cloned()
                    .ok_or(KernelAbiError::UnplannedGemmRows(rows))?
            }
            (KernelSymbols::AttentionSpecialized, TileKernelSpec::FlashAttention { .. }) => self
                .attention_symbols
                .get(&attention_shape(run)?)
                .cloned()
                .ok_or(KernelAbiError::RequirementMismatch)?,
            (KernelSymbols::AttentionStageSpecialized, _) => {
                let rows = gemm_rows(run)?;
                self.attention_stage_symbols
                    .iter()
                    .find(|(planned, planned_rows, _)| planned == kernel && *planned_rows == rows)
                    .map(|(_, _, symbol)| symbol.clone())
                    .ok_or(KernelAbiError::RequirementMismatch)?
            }
            (KernelSymbols::RearrangeSpecialized, TileKernelSpec::Rearrange { .. }) => self
                .rearrange_symbols
                .get(&rearrangement_specialization(
                    match kernel {
                        TileKernelSpec::Rearrange {
                            to: crate::Layout { order, .. },
                            ..
                        } => RearrangeTarget::from_order(*order)
                            .ok_or(KernelAbiError::RequirementMismatch)?,
                        _ => return Err(KernelAbiError::RequirementMismatch),
                    },
                    matrix_extent(run, true, false)?,
                    matrix_extent(run, false, false)?,
                    matrix_extent(run, true, true)?,
                    matrix_extent(run, false, true)?,
                ))
                .cloned()
                .ok_or(KernelAbiError::RequirementMismatch)?,
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
    overrides: &BTreeMap<LowShardId, TileAddress>,
) -> Result<ComputeStep, KernelMaterializationError> {
    let call = plan.call(run)?;
    let resolve = |view: &crate::ShardView| {
        let shard = shards.get(view.shard.index() as usize).ok_or(
            KernelMaterializationError::UnplacedShard(view.shard.index()),
        )?;
        let spans = view_byte_spans(shard, view)?;
        let [span] = spans.as_slice() else {
            return Err(KernelMaterializationError::FragmentedView {
                shard: view.shard.index(),
                spans: spans.len(),
            });
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

fn collect_kernels(
    program: &LowProgram,
    tile: &TileWorkList,
    rows: &mut BTreeMap<(Precision, GemmWeightLoad, u32, u32), BTreeSet<u32>>,
    gelu: &mut bool,
    reduction_add: &mut bool,
    rearrangements: &mut BTreeSet<(RearrangeTarget, u32, u32, u32, u32)>,
    attention: &mut BTreeSet<AttentionKernelShape>,
    attention_stages: &mut Vec<(TileKernelSpec, u32)>,
) -> Result<(), KernelAbiError> {
    for work in program.work(tile) {
        match work {
            TileWorkRef::Kernel(run) => {
                let abi = validate_kernel_run(run)?;
                let TileKernel::Planned(kernel) = &run.kernel;
                if abi.availability != KernelAvailability::Implemented {
                    return Err(KernelAbiError::Unavailable(kernel.clone()));
                }
                if let TileKernelSpec::Gemm {
                    multiply,
                    weights,
                    inner_block,
                    output_columns,
                    ..
                } = kernel
                {
                    rows.entry((*multiply, *weights, *inner_block, *output_columns))
                        .or_default()
                        .insert(gemm_rows(run)?);
                } else if matches!(kernel, TileKernelSpec::Gelu) {
                    *gelu = true;
                } else if matches!(kernel, TileKernelSpec::ReductionSum { .. }) {
                    *reduction_add = true;
                } else if let TileKernelSpec::Rearrange {
                    from:
                        crate::Layout {
                            order: ElementOrder::RowMajor,
                            ..
                        },
                    to: crate::Layout { order, .. },
                } = kernel
                    && let Some(target) = RearrangeTarget::from_order(*order)
                {
                    rearrangements.insert(rearrangement_specialization(
                        target,
                        matrix_extent(run, true, false)?,
                        matrix_extent(run, false, false)?,
                        matrix_extent(run, true, true)?,
                        matrix_extent(run, false, true)?,
                    ));
                } else if matches!(kernel, TileKernelSpec::FlashAttention { .. }) {
                    attention.insert(attention_shape(run)?);
                } else if matches!(
                    kernel,
                    TileKernelSpec::AttentionSoftmax { .. } | TileKernelSpec::AttentionMerge { .. }
                ) {
                    let stage = (kernel.clone(), gemm_rows(run)?);
                    if !attention_stages.contains(&stage) {
                        attention_stages.push(stage);
                    }
                }
            }
            TileWorkRef::Repeat(repeat) => collect_kernels(
                program,
                &repeat.body,
                rows,
                gelu,
                reduction_add,
                rearrangements,
                attention,
                attention_stages,
            )?,
            TileWorkRef::Exchange(_) | TileWorkRef::LocalCopy(_) | TileWorkRef::Checkpoint(..) => {}
        }
    }
    Ok(())
}

fn attention_shape(run: &KernelRun) -> Result<AttentionKernelShape, KernelAbiError> {
    let TileKernel::Planned(TileKernelSpec::FlashAttention {
        options,
        accumulate,
    }) = &run.kernel
    else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    if options.causal || *accumulate != crate::AccumulationPrecision::F32 {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let [query, key, value] = run.inputs.as_slice() else {
        return Err(KernelAbiError::RequirementMismatch);
    };
    let extents = |operand: &crate::KernelOperand| {
        let [view] = operand.views.as_slice() else {
            return None;
        };
        Some(
            view.extents
                .iter()
                .map(|extent| extent.physical_end - extent.start)
                .collect::<Vec<_>>(),
        )
    };
    let query = extents(query).ok_or(KernelAbiError::RequirementMismatch)?;
    let key = extents(key).ok_or(KernelAbiError::RequirementMismatch)?;
    let value = extents(value).ok_or(KernelAbiError::RequirementMismatch)?;
    if query.len() < 2 || query.len() != key.len() || query.len() != value.len() {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let rank = query.len();
    if query[..rank - 2] != key[..rank - 2]
        || query[..rank - 2] != value[..rank - 2]
        || query[rank - 1] != key[rank - 1]
        || key[rank - 2] != value[rank - 2]
    {
        return Err(KernelAbiError::RequirementMismatch);
    }
    let matrices = query[..rank - 2]
        .iter()
        .try_fold(1u32, |product, &extent| product.checked_mul(extent))
        .ok_or(KernelAbiError::ElementCountOverflow)?;
    let scale = options
        .scale
        .as_value()
        .unwrap_or_else(|| 1.0 / (query[rank - 1] as f32).sqrt());
    Ok(AttentionKernelShape {
        matrices,
        query_rows: query[rank - 2],
        key_rows: key[rank - 2],
        query_dimension: query[rank - 1],
        value_dimension: value[rank - 1],
        scale_bits: scale.to_bits(),
    })
}

fn gemm_rows(run: &KernelRun) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    let output_order = match &run.requirements {
        KernelRequirements::Operator(requirements) => &requirements.output.format.layout.order,
        KernelRequirements::Conversion { .. } => return Err(KernelAbiError::RequirementMismatch),
    };
    let matrix_column_axis = rank
        .checked_sub(
            if matches!(output_order, ElementOrder::Amp(AmpOrder::TransposedOutput)) {
                2
            } else {
                1
            },
        )
        .ok_or(KernelAbiError::MissingGemmRows)?;
    run.output
        .extents
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != matrix_column_axis)
        .try_fold(1u32, |rows, extent| {
            rows.checked_mul(extent.1.physical_end - extent.1.start)
        })
        .filter(|&rows| rows != 0)
        .ok_or(KernelAbiError::MissingGemmRows)
}

fn matrix_extent(run: &KernelRun, logical: bool, columns: bool) -> Result<u32, KernelAbiError> {
    let rank = run.output.extents.len();
    let axis = rank
        .checked_sub(if columns { 1 } else { 2 })
        .ok_or(KernelAbiError::RequirementMismatch)?;
    let extent = &run.output.extents[axis];
    Ok(if logical {
        extent.logical_end - extent.start
    } else {
        extent.physical_end - extent.start
    })
}

fn rearrangement_specialization(
    order: RearrangeTarget,
    logical_rows: u32,
    physical_rows: u32,
    logical_columns: u32,
    physical_columns: u32,
) -> (RearrangeTarget, u32, u32, u32, u32) {
    if physical_rows == AMP_INNER_BLOCK
        && logical_rows < physical_rows
        && matches!(
            order,
            RearrangeTarget::AmpTransposedRight | RearrangeTarget::BlockMajor { .. }
        )
    {
        (order, 0, physical_rows, 0, physical_columns)
    } else {
        (
            order,
            logical_rows,
            physical_rows,
            logical_columns,
            physical_columns,
        )
    }
}

fn scalar_values(run: &KernelRun, abi: &KernelAbi) -> Result<Vec<u32>, KernelAbiError> {
    let count = element_count(run)?;
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
            "initial_block" => match &run.kernel {
                TileKernel::Planned(TileKernelSpec::AttentionMerge { initial, .. }) => {
                    Ok(u32::from(*initial))
                }
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            "final_block" => match &run.kernel {
                TileKernel::Planned(TileKernelSpec::AttentionMerge { final_block, .. }) => {
                    Ok(u32::from(*final_block))
                }
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            "logical_rows" => matrix_extent(run, true, false),
            "physical_rows" => matrix_extent(run, false, false),
            "logical_columns" => matrix_extent(run, true, true),
            "physical_columns" => matrix_extent(run, false, true),
            "target_order" => match &run.kernel {
                TileKernel::Planned(TileKernelSpec::Rearrange {
                    to: crate::Layout { order, .. },
                    ..
                }) => RearrangeTarget::from_order(*order)
                    .map(RearrangeTarget::codelet_index)
                    .ok_or(KernelAbiError::RequirementMismatch),
                _ => Err(KernelAbiError::RequirementMismatch),
            },
            _ => Err(KernelAbiError::RequirementMismatch),
        })
        .collect()
}

fn element_count(run: &KernelRun) -> Result<u32, KernelAbiError> {
    run.output.extents.iter().try_fold(1u32, |product, extent| {
        product
            .checked_mul(extent.physical_end - extent.start)
            .ok_or(KernelAbiError::ElementCountOverflow)
    })
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
        TileKernelSpec::Gemm {
            multiply,
            mode,
            weights,
            output_columns,
            ..
        } => {
            if !matches!(requirements, KernelRequirements::Operator(_)) {
                return Err(KernelAbiError::RequirementMismatch);
            }
            if *weights == GemmWeightLoad::Interleaved && *multiply != Precision::F16 {
                return Err(KernelAbiError::RequirementMismatch);
            }
            let symbols = gemm_symbols(*multiply, *mode, *weights, *output_columns);
            let scalars = if matches!(multiply, Precision::F8F143 { .. }) {
                scalar_arguments(2, &["scale_exponent"])
            } else {
                Vec::new()
            };
            (symbols.0, symbols.1, 2, scalars)
        }
        TileKernelSpec::Gelu => {
            let symbol = gelu_symbol(requirements).unwrap_or("ipu_stack_unsupported_gelu");
            (
                KernelSymbols::Exact(symbol),
                if symbol == "ipu_stack_unsupported_gelu" {
                    KernelAvailability::Required
                } else {
                    KernelAvailability::Implemented
                },
                1,
                scalar_arguments(1, &["element_count"]),
            )
        }
        TileKernelSpec::ReductionSum { inputs } => {
            if precision != Precision::F16 {
                return Err(KernelAbiError::RequirementMismatch);
            }
            let symbol = match inputs {
                2 => "ipu_stack_reduce_sum_2_f16",
                3 => "ipu_stack_reduce_sum_3_f16",
                4 => "ipu_stack_reduce_sum_4_f16",
                _ => return Err(KernelAbiError::RequirementMismatch),
            };
            (
                KernelSymbols::Exact(symbol),
                KernelAvailability::Implemented,
                *inputs,
                scalar_arguments(*inputs, &["element_count"]),
            )
        }
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
            KernelSymbols::AttentionSpecialized,
            if matches!(requirements, KernelRequirements::Operator(requirements)
                if requirements.output.format.precision == Precision::F32
                    && requirements.inputs.iter().all(|input| input.format.precision == Precision::F16))
            {
                KernelAvailability::Implemented
            } else {
                KernelAvailability::Required
            },
            3,
            Vec::new(),
        ),
        TileKernelSpec::AttentionSoftmax { .. } => (
            KernelSymbols::AttentionStageSpecialized,
            KernelAvailability::Implemented,
            1,
            Vec::new(),
        ),
        TileKernelSpec::AttentionMerge { .. } => (
            KernelSymbols::AttentionStageSpecialized,
            KernelAvailability::Implemented,
            2,
            scalar_arguments(2, &["initial_block", "final_block"]),
        ),
        TileKernelSpec::Cast { from, to } => (
            KernelSymbols::Exact(cast_symbol(*from, *to)),
            KernelAvailability::Required,
            1,
            scalar_arguments(1, &["element_count"]),
        ),
        TileKernelSpec::Rearrange { from, to }
            if precision == Precision::F16
                && from.order == ElementOrder::RowMajor
                && matches!(
                    to.order,
                    ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                        | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                ) =>
        {
            (
                KernelSymbols::RearrangeSpecialized,
                KernelAvailability::Implemented,
                1,
                scalar_arguments(
                    1,
                    &[
                        "logical_rows",
                        "physical_rows",
                        "target_order",
                        "logical_columns",
                        "physical_columns",
                    ],
                ),
            )
        }
        TileKernelSpec::Rearrange { .. } => (
            KernelSymbols::Exact("ipu_stack_rearrange"),
            KernelAvailability::Required,
            1,
            Vec::new(),
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
    if matches!(kernel, TileKernelSpec::Gelu) {
        let KernelSymbols::Exact(symbol) = abi.symbols else {
            return Err(KernelAbiError::RequirementMismatch);
        };
        let divisor = 2;
        let count = element_count(run)?;
        if !count.is_multiple_of(divisor) {
            return Err(KernelAbiError::UnsupportedElementCount {
                symbol,
                count,
                divisor,
            });
        }
    }
    if matches!(kernel, TileKernelSpec::ReductionSum { .. }) {
        let count = element_count(run)?;
        if !count.is_multiple_of(8) {
            return Err(KernelAbiError::UnsupportedElementCount {
                symbol: "ipu_stack_reduce_sum_f16",
                count,
                divisor: 8,
            });
        }
    }
    Ok(abi)
}

fn gelu_symbol(requirements: &KernelRequirements) -> Option<&'static str> {
    let KernelRequirements::Operator(requirements) = requirements else {
        return None;
    };
    let [input] = requirements.inputs.as_slice() else {
        return None;
    };
    if input.format.precision != Precision::F16
        || requirements.output.format.precision != Precision::F16
    {
        return None;
    }
    let input_layout = &input.format.layout;
    let output_layout = &requirements.output.format.layout;
    (input_layout == output_layout).then_some("ipu_stack_gelu_exact_f16")
}

fn gemm_symbols(
    precision: Precision,
    mode: GemmKernelMode,
    weights: GemmWeightLoad,
    output_columns: u32,
) -> (KernelSymbols, KernelAvailability) {
    let weight_suffix = if weights == GemmWeightLoad::Interleaved {
        "_interleaved"
    } else {
        ""
    };
    let prefix = match precision {
        Precision::F16 => "f16",
        Precision::F32 => "f32",
        Precision::F8F143 { .. } => "f8",
    };
    let row_symbols = |operation: &str| KernelSymbols::RowSpecialized {
        small: format!(
            "ipu_stack_gemm_{prefix}_{operation}_small_rows{weight_suffix}_c{output_columns}"
        ),
        large: format!(
            "ipu_stack_gemm_{prefix}_{operation}_large_rows{weight_suffix}_c{output_columns}"
        ),
    };
    let symbols = match (precision, mode, weights) {
        (Precision::F16, GemmKernelMode::Initialize, GemmWeightLoad::Interleaved) => {
            row_symbols("init")
        }
        (Precision::F16, GemmKernelMode::Accumulate, GemmWeightLoad::Interleaved) => {
            row_symbols("accumulate")
        }
        (Precision::F16, GemmKernelMode::Initialize, GemmWeightLoad::Standard) => {
            row_symbols("init")
        }
        (Precision::F16, GemmKernelMode::Accumulate, GemmWeightLoad::Standard) => {
            row_symbols("accumulate")
        }
        (Precision::F32, GemmKernelMode::Initialize, _) => row_symbols("init"),
        (Precision::F32, GemmKernelMode::Accumulate, _) => row_symbols("accumulate"),
        (Precision::F8F143 { .. }, GemmKernelMode::Initialize, _) => {
            KernelSymbols::Exact("ipu_stack_gemm_f8_init")
        }
        (Precision::F8F143 { .. }, GemmKernelMode::Accumulate, _) => {
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
        AccumulationPrecision, ComputeGraph, Ipu21CostModel, Layout, MemoryClass,
        OperandRequirement, OperatorRequirements, OutputAliasing, PipelineConfig, TensorFormat,
        TensorTiling, lower, lower_to_tiles,
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
            let weights = if precision == Precision::F16 && random.bool() {
                GemmWeightLoad::Interleaved
            } else {
                GemmWeightLoad::Standard
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
                    weights,
                    inner_block: 64,
                    output_columns: [32, 64, 128][random.usize(0..3)],
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
            let batch = random.u32(1..=4);
            let mut graph = ComputeGraph::new();
            let left = graph
                .host_input("left", [batch, u32::from(tiles) * rows_per_tile, 64])
                .unwrap();
            let right = graph.parameter("right", [64, 64]).unwrap();
            let result = graph.gemm(left, right).unwrap();
            graph.set_outputs([result]).unwrap();
            let config = PipelineConfig::new(tiles)
                .with_active_tile_counts([tiles])
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
                        layout: Layout::block_major_matrix(64, tiles),
                    },
                );
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&mid, &config).unwrap();
            let plan = KernelBuildPlan::from_program(&low).unwrap();
            let addresses = low
                .shards
                .iter()
                .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(plan.compilations.len(), 1);
            let planned_rows = plan.gemm_rows.values().next().unwrap();
            assert!(
                plan.compilations[0]
                    .flags
                    .iter()
                    .any(|flag| flag == &format!("-DGEMM_SMALL_ROWS={}", planned_rows[0]))
            );
            assert!(
                plan.compilations[0]
                    .flags
                    .iter()
                    .any(|flag| flag == "-DGEMM_SINGLE_ROWS=1")
            );
            assert_eq!(plan.compilations[0].retained_symbols.len(), 2);
            for run in low
                .tiles
                .iter()
                .flat_map(|tile| low.work(tile))
                .filter_map(|work| {
                    if let TileWorkRef::Kernel(run) = work {
                        Some(run)
                    } else {
                        None
                    }
                })
            {
                let call = plan.call(run).unwrap();
                assert!(plan.retained_symbols().any(|symbol| symbol == call.symbol));
                assert!(call.arguments.is_empty());
                let compute =
                    materialize_kernel_run(run, &low.shards, &addresses, &plan, &BTreeMap::new())
                        .unwrap();
                assert_eq!(compute.symbol, call.symbol);
                assert_eq!(compute.input_addresses.len(), 2);
            }
        }
    }

    #[test]
    fn randomized_gelu_abis_select_supported_layout_paths() {
        let mut random = fastrand::Rng::with_seed(0x6765_6c75);
        for _ in 0..64 {
            let tiles = 1_u16 << random.u32(0..=5);
            let input_layout = if random.bool() {
                Layout::amp_left_result(tiles)
            } else {
                Layout::row_sharded(tiles)
            };
            let output_layout = input_layout.clone();
            let requirement = |layout| {
                OperandRequirement::new(
                    TensorFormat {
                        precision: Precision::F16,
                        layout,
                    },
                    8,
                )
            };
            let requirements = KernelRequirements::Operator(OperatorRequirements {
                inputs: vec![requirement(input_layout)],
                output: requirement(output_layout),
                output_aliasing: OutputAliasing::Fresh,
                memory_relations: Vec::new(),
            });
            let abi = tile_kernel_abi(&TileKernelSpec::Gelu, &requirements).unwrap();
            assert_eq!(abi.availability, KernelAvailability::Implemented);
            assert_eq!(abi.input_registers, [3]);
            assert_eq!(abi.scalar_arguments[0].register, 4);
            assert_eq!(
                abi.symbols,
                KernelSymbols::Exact("ipu_stack_gelu_exact_f16")
            );
        }
    }
}
