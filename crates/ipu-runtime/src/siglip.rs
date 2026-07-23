use crate::{
    BlockLayout, Result, block_binding_typed, blocked_matrix_f8_f143_by_block, blocked_matrix_f16,
    f143_block_scales, f143_scale,
};
use half::f16;
use ipu_compiler::{
    Allocation, AllocationKind, AppendAffineLayerNormConfig, BlockPlacement, BlockedGemmConfig,
    CompileError, FlashAttentionConfig, FlashAttentionPlan, GemmDataType, MemoryArena,
    MemoryPolicy, Phase, RowShardPlacement, RowShardTransitionConfig, Schedule, TensorId,
    allocate_from_occupied_arenas, append_a16_to_a16_row_shards_reblocked_in_arenas,
    append_add_affine_layer_norm_f16_with_memory_policy, append_add_f16_row_shards_in_place,
    append_affine_layer_norm_f16_with_memory_policy, append_bias_f16_c16_in_arenas,
    append_blocked_gemm_f16_with_a16_blocks_with_memory_policy,
    append_blocked_gemm_f16_with_a16_input_with_memory_policy,
    append_c16_to_a16_blocks_gelu_f16_in_arenas, append_c16_to_a16_row_shards,
    append_c16_to_a16_row_shards_reblocked_in_arenas,
    append_flash_attention_from_a16_qkv_in_arenas,
    append_flash_attention_to_a16_row_shards_in_arenas, choose_gemm_row_block_for,
    choose_gemm_row_block_for_shape, choose_gemm_row_block_for_shape_max_rows,
    choose_row_shard_rows_for_copies_in_arenas, end_tensor_lifetimes,
    gemm_row_block_candidates_for, gemm_row_block_cost, make_tensors_resident,
    make_tensors_resident_since, set_f8_weight_block_scales_in_phases,
    set_native_f8_weight_block_scales_in_phases,
};
use ipu_models::SiglipWeights;
use ipu_package::{Binding, RegionSlice};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::sync::Arc;
use tracing::{info, info_span};

#[derive(Default)]
pub struct HostTensorSet {
    pub bindings: Vec<Binding>,
    pub bytes: Vec<u8>,
    pub resident_bindings: Vec<Binding>,
    pub resident_bytes: Vec<u8>,
}

impl HostTensorSet {
    pub fn push_input(&mut self, binding: Binding, bytes: Vec<u8>) -> Result<()> {
        push_host_tensor(&mut self.bindings, &mut self.bytes, binding, bytes)
    }

    pub fn push(&mut self, binding: Binding, bytes: Vec<u8>) -> Result<()> {
        push_host_tensor(
            &mut self.resident_bindings,
            &mut self.resident_bytes,
            binding,
            bytes,
        )
    }
}

fn push_host_tensor(
    bindings: &mut Vec<Binding>,
    destination: &mut Vec<u8>,
    binding: Binding,
    bytes: Vec<u8>,
) -> Result<()> {
    let binding_bytes = binding.slices.iter().map(|slice| slice.size).sum::<u64>();
    if binding_bytes != bytes.len() as u64 {
        return Err(format!(
            "host tensor {} has {} binding bytes but {} data bytes",
            binding.name,
            binding_bytes,
            bytes.len()
        )
        .into());
    }
    bindings.push(binding);
    destination.extend(bytes);
    Ok(())
}

pub struct SiglipEncoderLayer {
    pub output: Vec<RowShardPlacement>,
    pub norm2: Vec<RowShardPlacement>,
    pub mlp_gelu: Vec<BlockPlacement>,
    pub attention: FlashAttentionPlan,
    pub diagnostics: Option<SiglipEncoderDiagnostics>,
    pub profile_stages: Vec<SiglipProfileStage>,
}

#[derive(Clone, Debug)]
pub struct SiglipEncoderDiagnostics {
    pub input: Vec<RowShardPlacement>,
    pub norm1: Vec<RowShardPlacement>,
    pub qkv: [Vec<RowShardPlacement>; 3],
    pub attention_hidden: Vec<RowShardPlacement>,
    pub attention_residual: Vec<RowShardPlacement>,
}

#[derive(Clone, Debug)]
pub struct DeferredResidualAdd {
    sources: Vec<(TensorId, TensorId)>,
    phases: [Phase; 2],
}

pub fn defer_terminal_residual_add(schedule: &mut Schedule) -> Result<Option<DeferredResidualAdd>> {
    let Some(Phase::Compute { commands, .. }) = schedule.phases.last() else {
        return Ok(None);
    };
    if commands.is_empty()
        || commands
            .iter()
            .any(|command| command.specialization.operation != "add_f16")
    {
        return Ok(None);
    }
    let Some(Phase::Exchange { transfers }) =
        schedule.phases.get(schedule.phases.len().saturating_sub(2))
    else {
        return Ok(None);
    };
    if !transfers.is_empty() {
        return Ok(None);
    }
    let sources = commands
        .iter()
        .map(|command| {
            if command.inputs.len() != 2 || command.output != command.inputs[0] {
                return Err("terminal residual add has an incompatible command ABI".into());
            }
            Ok((command.output, command.inputs[1]))
        })
        .collect::<Result<Vec<_>>>()?;
    let deferred = schedule.phases.split_off(schedule.phases.len() - 2);
    let phases: [Phase; 2] = deferred
        .try_into()
        .map_err(|_| "terminal residual add did not contain two phases")?;
    Ok(Some(DeferredResidualAdd { sources, phases }))
}

pub fn materialize_deferred_residual_add(
    schedule: &mut Schedule,
    mut deferred: DeferredResidualAdd,
) -> Result<()> {
    let phase = schedule.phases.len();
    let Phase::Compute { op, .. } = &mut deferred.phases[1] else {
        return Err("deferred residual tail has no compute phase".into());
    };
    op.0 = phase + 1;
    schedule.phases.extend(deferred.phases);
    Ok(())
}

pub fn fuse_deferred_residual_into_layer_norm(
    schedule: &mut Schedule,
    phase_start: usize,
    deferred: DeferredResidualAdd,
) -> Result<()> {
    let DeferredResidualAdd { sources, .. } = deferred;
    let (compute_phase, commands) = schedule
        .phases
        .iter_mut()
        .enumerate()
        .skip(phase_start)
        .find_map(|(phase, candidate)| match candidate {
            Phase::Compute { commands, .. }
                if !commands.is_empty()
                    && commands.iter().all(|command| {
                        command.specialization.operation == "layer_norm_affine_f16"
                    }) =>
            {
                Some((phase, commands))
            }
            _ => None,
        })
        .ok_or("deferred residual add has no following affine LayerNorm")?;
    if commands.len() != sources.len() {
        return Err("deferred residual add and LayerNorm shard counts differ".into());
    }
    let sources = sources.into_iter().collect::<HashMap<_, _>>();
    if sources.len() != commands.len() {
        return Err("deferred residual add has duplicate destination shards".into());
    }
    let required = commands
        .iter()
        .filter_map(|command| {
            sources
                .get(&command.inputs[0])
                .map(|&source| (source, command.tile))
        })
        .collect::<HashSet<_>>();
    let live = required
        .iter()
        .copied()
        .filter(|&(tensor, tile)| schedule.allocations.is_live_at(tensor, tile, compute_phase))
        .collect::<HashSet<_>>();
    for command in commands {
        let command = Arc::make_mut(command);
        let source = sources
            .get(&command.inputs[0])
            .copied()
            .ok_or("deferred residual add has no source for a LayerNorm shard")?;
        if !live.contains(&(source, command.tile)) {
            return Err("deferred residual source is not colocated with LayerNorm".into());
        }
        command.inputs.insert(1, source);
        let specialization = Arc::make_mut(&mut command.specialization);
        specialization.operation = "add_layer_norm_affine_f16".into();
        specialization.role = "add-and-normalize".into();
        command
            .metadata
            .insert("label".into(), "fused residual add and LayerNorm".into());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SiglipProfileStage {
    pub name: String,
    pub phases: std::ops::Range<usize>,
}

pub struct SiglipMapHead {
    pub output: Vec<RowShardPlacement>,
    pub attention: FlashAttentionPlan,
}

fn log_attention_blocking(stage: &str, attention: &FlashAttentionPlan) {
    info!(
        stage,
        query_block_rows = attention.query_block_rows,
        key_block_rows = attention.key_block_rows,
        key_block_columns = attention.key_block_columns,
        tasks = attention.tasks.len(),
        key_value_blocks = attention.key_values.len(),
        "planned SigLIP attention blocking"
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AttentionKernelVariant {
    pub small_query_rows: u16,
    pub large_query_rows: u16,
    pub small_key_rows: u16,
    pub large_key_rows: u16,
}

impl AttentionKernelVariant {
    pub fn suffix(self) -> String {
        format!(
            "q{}_{}_k{}_{}",
            self.small_query_rows, self.large_query_rows, self.small_key_rows, self.large_key_rows
        )
    }
}

pub fn attention_kernel_variant(plan: &FlashAttentionPlan) -> AttentionKernelVariant {
    let query_rows = plan.tasks.iter().map(|task| task.query_rows);
    let key_rows = plan.key_values.iter().map(|block| block.key_rows);
    AttentionKernelVariant {
        small_query_rows: query_rows.clone().min().unwrap(),
        large_query_rows: query_rows.max().unwrap(),
        small_key_rows: key_rows.clone().min().unwrap(),
        large_key_rows: key_rows.max().unwrap(),
    }
}

pub fn consolidate_attention_kernel_variants(
    schedule: &mut Schedule,
    plans: &[FlashAttentionPlan],
) -> AttentionKernelVariant {
    let variants = plans
        .iter()
        .map(attention_kernel_variant)
        .collect::<Vec<_>>();
    let domain = AttentionKernelVariant {
        small_query_rows: variants
            .iter()
            .map(|variant| variant.small_query_rows)
            .min()
            .unwrap(),
        large_query_rows: variants
            .iter()
            .map(|variant| variant.large_query_rows)
            .max()
            .unwrap(),
        small_key_rows: variants
            .iter()
            .map(|variant| variant.small_key_rows)
            .min()
            .unwrap(),
        large_key_rows: variants
            .iter()
            .map(|variant| variant.large_key_rows)
            .max()
            .unwrap(),
    };
    assert!(variants.iter().all(|variant| {
        [variant.small_query_rows, variant.large_query_rows]
            .into_iter()
            .all(|rows| rows == domain.small_query_rows || rows == domain.large_query_rows)
            && [variant.small_key_rows, variant.large_key_rows]
                .into_iter()
                .all(|rows| rows == domain.small_key_rows || rows == domain.large_key_rows)
    }));
    let local_suffixes = variants
        .iter()
        .map(|variant| (format!("_{}", variant.suffix()), *variant))
        .collect::<Vec<_>>();

    for phase in &mut schedule.phases {
        let ipu_compiler::Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            let operation = command.specialization.operation.as_ref();
            if !operation.starts_with("attention_") {
                continue;
            }
            let Some((suffix, local)) = local_suffixes
                .iter()
                .find(|(suffix, _)| operation.ends_with(suffix))
            else {
                continue;
            };
            let base = operation.strip_suffix(suffix).unwrap();
            let query_small = row_role(local.small_query_rows, domain.small_query_rows);
            let query_large = row_role(local.large_query_rows, domain.small_query_rows);
            let key_small = row_role(local.small_key_rows, domain.small_key_rows);
            let key_large = row_role(local.large_key_rows, domain.small_key_rows);
            let base = base
                .replace("small_query", "@QUERY_SMALL@")
                .replace("large_query", "@QUERY_LARGE@")
                .replace("small_key", "@KEY_SMALL@")
                .replace("large_key", "@KEY_LARGE@")
                .replace("small_rows", "@QUERY_SMALL_ROWS@")
                .replace("large_rows", "@QUERY_LARGE_ROWS@")
                .replace("@QUERY_SMALL@", &format!("{query_small}_query"))
                .replace("@QUERY_LARGE@", &format!("{query_large}_query"))
                .replace("@KEY_SMALL@", &format!("{key_small}_key"))
                .replace("@KEY_LARGE@", &format!("{key_large}_key"))
                .replace("@QUERY_SMALL_ROWS@", &format!("{query_small}_rows"))
                .replace("@QUERY_LARGE_ROWS@", &format!("{query_large}_rows"));
            Arc::make_mut(&mut command.specialization).operation =
                format!("{base}_{}", domain.suffix()).into();
        }
    }
    domain
}

fn row_role(rows: u16, small_rows: u16) -> &'static str {
    if rows == small_rows { "small" } else { "large" }
}

fn specialize_attention_phases(
    schedule: &mut Schedule,
    phase_start: usize,
    plan: &FlashAttentionPlan,
) {
    let suffix = attention_kernel_variant(plan).suffix();
    for phase in &mut schedule.phases[phase_start..] {
        let ipu_compiler::Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            let operation = command.specialization.operation.as_ref();
            let specialized = if operation.starts_with("attention_qk_")
                || operation.starts_with("attention_pv_")
                || operation.starts_with("attention_softmax_")
                || operation.starts_with("attention_merge_")
                || operation == "attention_f32_to_f16"
            {
                Some(format!("{operation}_{suffix}"))
            } else {
                None
            };
            if let Some(specialized) = specialized {
                Arc::make_mut(&mut command.specialization).operation = specialized.into();
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiglipWeightStorage {
    F16,
    F143,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiglipLinearPrecision {
    F16,
    F143Expanded,
    F143Native { activation_scale: i8 },
}

impl SiglipLinearPrecision {
    fn gemm_data_type(self, values: impl IntoIterator<Item = f32>) -> GemmDataType {
        match self {
            Self::F16 => GemmDataType::F16,
            Self::F143Expanded => GemmDataType::F16F8Weights {
                scale: f143_scale(values),
            },
            Self::F143Native { activation_scale } => GemmDataType::F8F143 {
                input_scale: activation_scale,
                weight_scale: f143_scale(values),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiglipEncoderPrecision {
    pub qkv: SiglipLinearPrecision,
    pub attention_output: SiglipLinearPrecision,
    pub mlp_up: SiglipLinearPrecision,
    pub mlp_down: SiglipLinearPrecision,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SiglipEncoderTuning {
    /// Zero uses automatic or persistent row blocking according to the mode below.
    pub gemm_row_block_rows: u16,
    /// Choose row blocking independently for GEMMs whose inputs can be reblocked.
    pub automatic_gemm_row_blocks: bool,
    /// Zero uses 64-column K blocks for GEMMs with row-sharded inputs.
    pub row_gemm_inner_block_columns: u16,
    /// Per-projection K-block overrides; zero inherits `row_gemm_inner_block_columns`.
    pub qkv_inner_block_columns: u16,
    pub attention_output_inner_block_columns: u16,
    pub mlp_up_inner_block_columns: u16,
    /// Zero selects an output block width from the GEMM shape and device occupancy.
    pub gemm_output_block_columns: u16,
    /// Per-projection output-block overrides; zero inherits `gemm_output_block_columns`.
    pub qkv_output_block_columns: u16,
    pub attention_output_block_columns: u16,
    pub mlp_up_output_block_columns: u16,
    pub mlp_down_output_block_columns: u16,
    /// Zero asks the attention planner to saturate the available tiles.
    pub attention_query_block_rows: u16,
    /// Zero asks the attention planner to choose the K/V block size.
    pub attention_key_block_rows: u16,
}

fn choose_gemm_output_block_columns(
    rows: u16,
    row_block_rows: u16,
    inner_block_columns: u16,
    columns: u16,
    tile_count: u16,
    data_type: GemmDataType,
    requested: u16,
) -> Result<u16> {
    let fits_transfer = |output_columns: u16| {
        u32::from(inner_block_columns)
            .checked_mul(u32::from(output_columns))
            .and_then(|elements| elements.checked_mul(data_type.weight_element_bytes()))
            .is_some_and(|bytes| bytes <= ipu_exchange::MAX_TRANSFER_WORDS * 4)
    };
    if requested != 0 {
        if !matches!(requested, 32 | 64 | 128)
            || !columns.is_multiple_of(requested)
            || !fits_transfer(requested)
        {
            return Err(format!(
                "GEMM output block width {requested} must be supported, divide {columns}, and fit one weight transfer"
            )
            .into());
        }
        return Ok(requested);
    }
    const BASELINE: u16 = 64;
    if !columns.is_multiple_of(BASELINE) {
        return Err(format!("GEMM output columns {columns} are not divisible by 64").into());
    }
    let row_shards = rows.div_ceil(row_block_rows);
    let tile_count = u32::from(tile_count);
    let mut best = BASELINE;
    let mut best_blocks = u32::from(row_shards) * u32::from(columns / BASELINE);
    let mut best_waves = best_blocks.div_ceil(tile_count);
    let baseline_waves = best_waves;
    let mut best_is_saturated = best_blocks * 4 >= best_waves * tile_count * 3;
    for candidate in [32, 64, 128] {
        if !columns.is_multiple_of(candidate) || !fits_transfer(candidate) {
            continue;
        }
        let blocks = u32::from(row_shards) * u32::from(columns / candidate);
        let waves = blocks.div_ceil(tile_count);
        let is_saturated = blocks * 4 >= waves * tile_count * 3;
        if !is_saturated || waves > baseline_waves {
            continue;
        }
        if !best_is_saturated || waves < best_waves || (waves == best_waves && blocks > best_blocks)
        {
            best = candidate;
            best_blocks = blocks;
            best_waves = waves;
            best_is_saturated = true;
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn choose_shared_gemm_row_block(
    rows: u16,
    tile_count: u16,
    first_inner_dimension: u16,
    first_inner_block: u16,
    first_columns: u16,
    first_output_block: u16,
    first_type: GemmDataType,
    second_inner_dimension: u16,
    second_inner_block: u16,
    second_columns: u16,
    second_output_block: u16,
    second_type: GemmDataType,
) -> Option<u16> {
    let second_row_shards = gemm_row_block_candidates_for(
        rows,
        second_inner_block,
        second_columns,
        second_output_block,
        tile_count,
        second_type,
    )
    .into_iter()
    .map(|target| rows.div_ceil(target))
    .collect::<std::collections::BTreeSet<_>>();
    let maximum_second_source_rows =
        u16::try_from(ipu_exchange::MAX_TRANSFER_WORDS * 4 / (u32::from(second_inner_block) * 2))
            .ok()?;
    gemm_row_block_candidates_for(
        rows,
        first_inner_block,
        first_columns,
        first_output_block,
        tile_count,
        first_type,
    )
    .into_iter()
    .filter(|target| second_row_shards.contains(&rows.div_ceil(*target)))
    .filter(|target| rows.div_ceil(rows.div_ceil(*target)) <= maximum_second_source_rows)
    .min_by_key(|target| {
        let first = gemm_row_block_cost(
            rows,
            *target,
            first_inner_dimension,
            first_columns,
            first_output_block,
            tile_count,
        )
        .expect("shared GEMM row candidate has valid first dimensions");
        let second = gemm_row_block_cost(
            rows,
            *target,
            second_inner_dimension,
            second_columns,
            second_output_block,
            tile_count,
        )
        .expect("shared GEMM row candidate has valid second dimensions");
        (first.0 + second.0, first.1 + second.1, first.2 + second.2)
    })
}

fn a16_reblock_staging_upper_bound(
    rows: u16,
    source_row_target: u16,
    destination_row_target: u16,
    inner_block_columns: u16,
    inner_dimension: u16,
    tile_count: u16,
) -> u32 {
    let source_rows = rows.div_ceil(rows.div_ceil(source_row_target));
    let destination_rows = rows.div_ceil(rows.div_ceil(destination_row_target));
    let destination_blocks = usize::from(rows.div_ceil(destination_row_target))
        * usize::from(inner_dimension / inner_block_columns);
    let blocks_per_tile = destination_blocks.div_ceil(usize::from(tile_count));
    let fragments_per_block = if source_rows == destination_rows {
        1
    } else {
        usize::from(destination_rows.div_ceil(source_rows)) + 1
    };
    let panel_count = u32::from(inner_block_columns / 16);
    let fragment_bytes = (panel_count - 1) * u32::from(source_rows) * 32
        + u32::from(source_rows.min(destination_rows)) * 32;
    u32::try_from(blocks_per_tile.saturating_mul(fragments_per_block))
        .unwrap_or(u32::MAX)
        .saturating_mul(fragment_bytes)
}

impl SiglipEncoderPrecision {
    pub fn uniform(storage: SiglipWeightStorage) -> Self {
        let precision = match storage {
            SiglipWeightStorage::F16 => SiglipLinearPrecision::F16,
            SiglipWeightStorage::F143 => SiglipLinearPrecision::F143Expanded,
        };
        Self {
            qkv: precision,
            attention_output: precision,
            mlp_up: precision,
            mlp_down: precision,
        }
    }

    /// Whether two layer instances use the same kernels, layouts, and memory
    /// objects. Native-FP8 exponent scales are call arguments and therefore do
    /// not change the reusable program body.
    pub fn has_same_execution_shape(self, other: Self) -> bool {
        fn linear_shape(precision: SiglipLinearPrecision) -> u8 {
            match precision {
                SiglipLinearPrecision::F16 => 0,
                SiglipLinearPrecision::F143Expanded => 1,
                SiglipLinearPrecision::F143Native { .. } => 2,
            }
        }

        linear_shape(self.qkv) == linear_shape(other.qkv)
            && linear_shape(self.attention_output) == linear_shape(other.attention_output)
            && linear_shape(self.mlp_up) == linear_shape(other.mlp_up)
            && linear_shape(self.mlp_down) == linear_shape(other.mlp_down)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn append_host_a16_matrix(
    schedule: &mut Schedule,
    name: &str,
    values: &[f32],
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    append_host_a16_matrix_in_arenas(
        schedule,
        name,
        values,
        rows,
        columns,
        row_block_dimension,
        &[MemoryArena::high(data_base, data_limit)],
        host,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_host_a16_matrix_in_arenas(
    schedule: &mut Schedule,
    name: &str,
    values: &[f32],
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    arenas: &[MemoryArena],
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    if rows == 0
        || columns == 0
        || !columns.is_multiple_of(16)
        || row_block_dimension == 0
        || arenas.is_empty()
        || values.len() != usize::from(rows) * usize::from(columns)
    {
        return Err("host A16 matrix has incompatible dimensions or data".into());
    }
    let mut next_tensor = schedule.allocations.next_tensor_id();
    let row_grid = rows.div_ceil(row_block_dimension);
    let base_rows = rows / row_grid;
    let larger_shards = rows % row_grid;
    let mut shards = Vec::with_capacity(usize::from(row_grid));
    let mut resident_pressure = vec![0u64; usize::from(schedule.tile_count)];
    for allocation in &schedule.allocations {
        if allocation.kind == AllocationKind::Home
            && allocation.live_from == 0
            && allocation.live_until == usize::MAX
        {
            resident_pressure[usize::from(allocation.tile)] += u64::from(allocation.size);
        }
    }
    let mut occupied = schedule.allocations.occupied_intervals_by_tile(
        schedule.tile_count,
        0,
        usize::MAX,
        arenas.iter().map(|arena| arena.base).min().unwrap(),
        arenas.iter().map(|arena| arena.limit).max().unwrap(),
    );
    let mut row_start = 0;
    for shard_index in 0..row_grid {
        let shard_rows = base_rows + u16::from(shard_index < larger_shards);
        let bytes = u32::from(shard_rows) * u32::from(columns) * 2;
        let (tile, address) = (0..schedule.tile_count)
            .filter_map(|tile| {
                let mut candidate = occupied[usize::from(tile)].clone();
                let address =
                    allocate_from_occupied_arenas(&mut candidate, bytes, arenas, 8).ok()?;
                Some((resident_pressure[usize::from(tile)], tile, address))
            })
            .min()
            .map(|(_, tile, address)| (tile, address))
            .ok_or_else(|| format!("no tile can hold {bytes} bytes for host matrix {name}"))?;
        let allocated =
            allocate_from_occupied_arenas(&mut occupied[usize::from(tile)], bytes, arenas, 8)?;
        debug_assert_eq!(allocated, address);
        resident_pressure[usize::from(tile)] += u64::from(bytes);
        let tensor = TensorId(next_tensor);
        next_tensor += 1;
        shards.push(RowShardPlacement {
            tile,
            row_start,
            rows: shard_rows,
            columns,
            tensor,
            address,
        });
        schedule.allocations.push(Allocation {
            tensor,
            tile,
            address,
            size: bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        row_start += shard_rows;
    }
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for shard in &shards {
        for panel in 0..columns / 16 {
            for row in shard.row_start..shard.row_start + shard.rows {
                for column in panel * 16..panel * 16 + 16 {
                    let value =
                        values[usize::from(row) * usize::from(columns) + usize::from(column)];
                    bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
                }
            }
        }
    }
    host.push_input(row_shard_binding(name, rows, columns, &shards), bytes)?;
    Ok(shards)
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_post_layer_norm(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    let memory = MemoryPolicy::contiguous(data_base, data_limit);
    append_siglip_post_layer_norm_with_memory_policy(schedule, input, model, &memory, host)
}

pub fn append_siglip_post_layer_norm_with_memory_policy(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    memory: &MemoryPolicy,
    host: &mut HostTensorSet,
) -> Result<Vec<RowShardPlacement>> {
    let columns = u16::try_from(model.config.hidden_size)?;
    let allocation_start = schedule.allocations.len();
    let norm = append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        input,
        AppendAffineLayerNormConfig {
            data_base: memory
                .transient
                .iter()
                .map(|arena| arena.base)
                .min()
                .unwrap(),
            data_limit: memory
                .transient
                .iter()
                .map(|arena| arena.limit)
                .max()
                .unwrap(),
            epsilon_bits: model.config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    push_named_layer_norm_affine(
        schedule,
        host,
        "post_layernorm.affine",
        columns,
        &norm.affine,
        &model.tensor_f32("vision_model.post_layernorm.weight")?,
        &model.tensor_f32("vision_model.post_layernorm.bias")?,
        allocation_start,
    )?;
    end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;
    Ok(norm.output)
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_map_head(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    rows: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    host: &mut HostTensorSet,
) -> Result<SiglipMapHead> {
    let memory = MemoryPolicy::contiguous(data_base, data_limit);
    append_siglip_map_head_with_memory_policy(
        schedule,
        input,
        model,
        rows,
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        &memory,
        host,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_map_head_with_memory_policy(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    rows: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    memory: &MemoryPolicy,
    host: &mut HostTensorSet,
) -> Result<SiglipMapHead> {
    const PROBE_ROWS: u16 = 12;
    let config = &model.config;
    let columns = u16::try_from(config.hidden_size)?;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64)?;
    let probe = model.tensor_f32("vision_model.head.probe")?;
    let mut repeated_probe = Vec::with_capacity(usize::from(PROBE_ROWS) * probe.len());
    for _ in 0..PROBE_ROWS {
        repeated_probe.extend_from_slice(&probe);
    }
    let probe = append_host_a16_matrix_in_arenas(
        schedule,
        "map.probe",
        &repeated_probe,
        PROBE_ROWS,
        columns,
        PROBE_ROWS,
        &memory.resident,
        host,
    )?;
    let in_weight = model.tensor_f32("vision_model.head.attention.in_proj_weight")?;
    let in_bias = model.tensor_f32("vision_model.head.attention.in_proj_bias")?;
    let query = append_a16_linear_c16_with_memory_policy(
        schedule,
        &probe,
        PROBE_ROWS,
        columns,
        columns,
        columns,
        columns,
        0,
        &in_weight,
        &in_bias,
        "map.query",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        memory,
        host,
    )?;
    end_tensor_lifetimes(schedule, probe.iter().map(|shard| shard.tensor))?;
    let query = append_c16_to_a16_row_shards(
        schedule,
        &query,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;

    let key_value = append_a16_linear_c16_with_memory_policy(
        schedule,
        input,
        rows,
        columns,
        columns * 2,
        columns,
        columns * 2,
        columns,
        &in_weight,
        &in_bias,
        "map.key_value",
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        memory,
        host,
    )?;
    let key = append_c16_to_a16_row_shards(
        schedule,
        &projection_blocks(&key_value, 0, columns),
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    let value = append_c16_to_a16_row_shards(
        schedule,
        &projection_blocks(&key_value, 1, columns),
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, key_value.iter().map(|block| block.tensor))?;
    end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;

    let attention_phase_start = schedule.phases.len();
    let attention = append_flash_attention_from_a16_qkv_in_arenas(
        schedule,
        &query,
        &key,
        &value,
        FlashAttentionConfig {
            batch_size: 1,
            query_sequence_length: PROBE_ROWS,
            sequence_length: rows,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads)?,
            query_block_rows: PROBE_ROWS,
            key_block_rows: 0,
            tile_count,
            data_base,
            data_limit,
        },
        &memory.transient,
    )?;
    log_attention_blocking("map", &attention);
    specialize_attention_phases(schedule, attention_phase_start, &attention);
    end_tensor_lifetimes(
        schedule,
        query
            .iter()
            .chain(&key)
            .chain(&value)
            .map(|shard| shard.tensor),
    )?;
    let attention_shards = append_flash_attention_to_a16_row_shards_in_arenas(
        schedule,
        &attention,
        &memory.transient,
    )?;
    let projected = append_a16_linear_c16_with_memory_policy(
        schedule,
        &attention_shards,
        PROBE_ROWS,
        columns,
        columns,
        columns,
        columns,
        0,
        &model.tensor_f32("vision_model.head.attention.out_proj.weight")?,
        &model.tensor_f32("vision_model.head.attention.out_proj.bias")?,
        "map.attention_output",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        memory,
        host,
    )?;
    end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    let residual = append_c16_to_a16_row_shards(
        schedule,
        &projected,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, projected.iter().map(|block| block.tensor))?;

    let norm_allocation_start = schedule.allocations.len();
    let norm = append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        &residual,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    push_named_layer_norm_affine(
        schedule,
        host,
        "map.layernorm",
        columns,
        &norm.affine,
        &model.tensor_f32("vision_model.head.layernorm.weight")?,
        &model.tensor_f32("vision_model.head.layernorm.bias")?,
        norm_allocation_start,
    )?;
    let up = append_a16_linear_c16_with_memory_policy(
        schedule,
        &norm.output,
        PROBE_ROWS,
        columns,
        intermediate_columns,
        columns,
        u16::try_from(config.intermediate_size)?,
        0,
        &model.tensor_f32("vision_model.head.mlp.fc1.weight")?,
        &model.tensor_f32("vision_model.head.mlp.fc1.bias")?,
        "map.mlp_up",
        PROBE_ROWS,
        tile_count,
        data_base,
        data_limit,
        memory,
        host,
    )?;
    end_tensor_lifetimes(schedule, norm.output.iter().map(|shard| shard.tensor))?;
    let gelu = append_c16_to_a16_blocks_gelu_f16_in_arenas(schedule, &up, &memory.transient)?;
    end_tensor_lifetimes(schedule, up.iter().map(|block| block.tensor))?;
    let down = append_blocked_gemm_f16_with_a16_blocks_with_memory_policy(
        schedule,
        &gelu,
        gemm_config(
            PROBE_ROWS,
            intermediate_columns,
            columns,
            PROBE_ROWS,
            tile_count,
            data_base,
            data_limit,
            GemmDataType::F16,
            true,
        ),
        memory,
    )?;
    let down_weight = model.tensor_f32("vision_model.head.mlp.fc2.weight")?;
    host.push(
        block_binding_typed(
            "map.mlp_down.weight",
            intermediate_columns,
            columns,
            &down.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&down.right, BlockLayout::AmpB16x16, |row, column| {
            if usize::from(row) < config.intermediate_size {
                down_weight[usize::from(column) * config.intermediate_size + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, down.right.iter().map(|block| block.tensor))?;
    end_tensor_lifetimes(schedule, gelu.iter().map(|block| block.tensor))?;
    let down_bias = append_bias_f16_c16_in_arenas(schedule, &down.output, &memory.resident)?;
    let bias = model.tensor_f32("vision_model.head.mlp.fc2.bias")?;
    host.push(
        block_binding_typed(
            "map.mlp_down.bias",
            u16::try_from(down_bias.len())?,
            64,
            &down_bias,
            "f16",
            2,
        ),
        blocked_matrix_f16(&down_bias, BlockLayout::AmpC16F16, |_row, column| {
            bias[usize::from(column)]
        }),
    )?;
    make_tensors_resident(schedule, down_bias.iter().map(|block| block.tensor))?;
    let output = append_c16_to_a16_row_shards(
        schedule,
        &down.output,
        RowShardTransitionConfig {
            columns,
            data_base,
            data_limit,
        },
    )?;
    end_tensor_lifetimes(schedule, down.output.iter().map(|block| block.tensor))?;
    let output = append_add_f16_row_shards_in_place(schedule, &output, &residual)?;
    end_tensor_lifetimes(schedule, residual.iter().map(|shard| shard.tensor))?;
    Ok(SiglipMapHead { output, attention })
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_encoder_layer(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    layer: usize,
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    memory: &MemoryPolicy,
    weight_storage: SiglipWeightStorage,
    retain_profile_metadata: bool,
    retain_diagnostics: bool,
    host: &mut HostTensorSet,
) -> Result<SiglipEncoderLayer> {
    append_siglip_encoder_layer_with_precision(
        schedule,
        input,
        model,
        layer,
        rows,
        columns,
        row_block_dimension,
        tile_count,
        memory,
        SiglipEncoderPrecision::uniform(weight_storage),
        SiglipEncoderTuning::default(),
        retain_profile_metadata,
        retain_diagnostics,
        host,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_encoder_layer_with_precision(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    layer: usize,
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    memory: &MemoryPolicy,
    precision: SiglipEncoderPrecision,
    tuning: SiglipEncoderTuning,
    retain_profile_metadata: bool,
    retain_diagnostics: bool,
    host: &mut HostTensorSet,
) -> Result<SiglipEncoderLayer> {
    append_siglip_encoder_layer_batched_with_precision(
        schedule,
        input,
        model,
        layer,
        1,
        rows,
        columns,
        row_block_dimension,
        tile_count,
        memory,
        precision,
        tuning,
        retain_profile_metadata,
        retain_diagnostics,
        host,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn append_siglip_encoder_layer_batched_with_precision(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    model: &SiglipWeights,
    layer: usize,
    batch_size: u16,
    rows: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    memory: &MemoryPolicy,
    precision: SiglipEncoderPrecision,
    tuning: SiglipEncoderTuning,
    retain_profile_metadata: bool,
    retain_diagnostics: bool,
    host: &mut HostTensorSet,
) -> Result<SiglipEncoderLayer> {
    memory.validate()?;
    // The standalone kernel planners use this ordinary-memory window before
    // policy-aware composition relocates movable operands into their arenas.
    let data_base = ipu_package::IPU21_INTERLEAVED_MEMORY_LIMIT;
    let data_limit = ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE;
    let span = info_span!("siglip_encoder_layer", layer);
    let _guard = span.enter();
    let config = &model.config;
    let sequence_length = u16::try_from(model.sequence_length())?;
    if batch_size == 0 || u32::from(rows) != u32::from(batch_size) * u32::from(sequence_length) {
        return Err("SigLIP encoder rows must equal batch size times sequence length".into());
    }
    let tuned_gemm_rows = if tuning.gemm_row_block_rows == 0 {
        row_block_dimension
    } else {
        tuning.gemm_row_block_rows
    };
    let tuned_gemm_inner = if tuning.row_gemm_inner_block_columns == 0 {
        64
    } else {
        tuning.row_gemm_inner_block_columns
    };
    let projection_inner = |requested: u16| {
        if requested == 0 {
            tuned_gemm_inner
        } else {
            requested
        }
    };
    let projection_output = |requested: u16| {
        if requested == 0 {
            tuning.gemm_output_block_columns
        } else {
            requested
        }
    };
    let qkv_inner = projection_inner(tuning.qkv_inner_block_columns);
    let prefix = format!("encoder_layer_{layer:02}");
    let layer_phase_start = schedule.phases.len();
    info!(stage = "norm1_qkv", "planning SigLIP encoder stage");
    let norm_allocation_start = schedule.allocations.len();
    let norm = append_affine_layer_norm_f16_with_memory_policy(
        schedule,
        input,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm1",
        columns,
        &norm.affine,
        norm_allocation_start,
    )?;

    let projection_names = ["q_proj", "k_proj", "v_proj"];
    let projection_weights = projection_names.map(|projection| {
        model
            .tensor_f32(
                &model
                    .layer_name(layer, &format!("self_attn.{projection}.weight"))
                    .unwrap(),
            )
            .unwrap()
    });
    let qkv_data_type = precision.qkv.gemm_data_type(
        projection_weights
            .iter()
            .flat_map(|weights| weights.iter().copied()),
    );
    let qkv_maximum_materialized_rows = if matches!(qkv_data_type, GemmDataType::F8F143 { .. }) {
        None
    } else {
        let preferred_arena = memory
            .transient
            .first()
            .ok_or("SigLIP encoder requires a transient memory arena")?;
        Some(
            (preferred_arena.limit - preferred_arena.base)
                .checked_div(u32::from(columns) * 2)
                .and_then(|rows| u16::try_from(rows).ok())
                .ok_or("preferred transient arena cannot hold one QKV row")?,
        )
    };
    let mut qkv_row_block_dimension = if matches!(qkv_data_type, GemmDataType::F8F143 { .. }) {
        row_block_dimension
    } else if tuning.automatic_gemm_row_blocks && tuning.gemm_row_block_rows == 0 {
        choose_gemm_row_block_for_shape_max_rows(
            rows,
            columns,
            qkv_inner,
            columns * 3,
            64,
            tile_count,
            qkv_data_type,
            qkv_maximum_materialized_rows.unwrap(),
        )
        .ok_or("QKV GEMM has no row blocking whose output fits the preferred transient arena")?
    } else {
        tuned_gemm_rows
    };
    let qkv_phase_start = schedule.phases.len();
    let qkv_output_block_columns = choose_gemm_output_block_columns(
        rows,
        qkv_row_block_dimension,
        qkv_inner,
        columns * 3,
        tile_count,
        qkv_data_type,
        projection_output(tuning.qkv_output_block_columns),
    )?;
    if tuning.automatic_gemm_row_blocks
        && tuning.gemm_row_block_rows == 0
        && !matches!(qkv_data_type, GemmDataType::F8F143 { .. })
    {
        qkv_row_block_dimension = choose_gemm_row_block_for_shape_max_rows(
            rows,
            columns,
            qkv_inner,
            columns * 3,
            qkv_output_block_columns,
            tile_count,
            qkv_data_type,
            qkv_maximum_materialized_rows.unwrap(),
        )
        .ok_or("QKV GEMM has no row blocking for its selected output blocks")?;
    }
    info!(
        operation = "qkv",
        row_block_rows = qkv_row_block_dimension,
        output_block_columns = qkv_output_block_columns,
        "selected SigLIP GEMM blocking"
    );
    let qkv_allocation_start = schedule.allocations.len();
    let qkv = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &norm.output,
        gemm_config_with_inner(
            rows,
            columns,
            columns * 3,
            qkv_output_block_columns,
            qkv_inner,
            qkv_row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            qkv_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.qkv.weight"),
        columns,
        columns * 3,
        &qkv.right,
        qkv_data_type,
        qkv_phase_start,
        |row, column| {
            let projection = usize::from(column / columns);
            let output = usize::from(column % columns);
            projection_weights[projection][output * usize::from(columns) + usize::from(row)]
        },
    )?;
    make_tensors_resident_since(
        schedule,
        qkv_allocation_start,
        qkv.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, norm.output.iter().map(|shard| shard.tensor))?;
    }

    let projection_biases = projection_names.map(|projection| {
        model
            .tensor_f32(
                &model
                    .layer_name(layer, &format!("self_attn.{projection}.bias"))
                    .unwrap(),
            )
            .unwrap()
    });
    let qkv_bias_allocation_start = schedule.allocations.len();
    let qkv_bias = append_bias_f16_c16_in_arenas(schedule, &qkv.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.qkv.bias"),
            u16::try_from(qkv_bias.len())?,
            qkv_output_block_columns,
            &qkv_bias,
            "f16",
            2,
        ),
        blocked_matrix_f16(&qkv_bias, BlockLayout::AmpC16F16, |_row, column| {
            let projection = usize::from(column / columns);
            projection_biases[projection][usize::from(column % columns)]
        }),
    )?;
    make_tensors_resident_since(
        schedule,
        qkv_bias_allocation_start,
        qkv_bias.iter().map(|block| block.tensor),
    )?;
    let preferred_qkv_rows = if tuning.automatic_gemm_row_blocks {
        qkv_row_block_dimension
    } else {
        row_block_dimension
    };
    let qkv_destination_rows = choose_row_shard_rows_for_copies_in_arenas(
        schedule,
        rows,
        columns,
        preferred_qkv_rows,
        3,
        &memory.transient,
    )
    .ok_or_else(|| {
        CompileError::Memory("QKV has no feasible shared A16 row-shard placement".into())
    })?;
    info!(
        source_row_block_rows = qkv_row_block_dimension,
        destination_row_block_rows = qkv_destination_rows,
        "selected shared QKV row-shard placement"
    );
    let qkv_shards = (0..3)
        .map(|projection| {
            let blocks = projection_blocks(&qkv.output, projection, columns);
            if qkv_destination_rows == qkv_row_block_dimension {
                append_c16_to_a16_row_shards(
                    schedule,
                    &blocks,
                    RowShardTransitionConfig {
                        columns,
                        data_base,
                        data_limit,
                    },
                )
            } else {
                append_c16_to_a16_row_shards_reblocked_in_arenas(
                    schedule,
                    &blocks,
                    columns,
                    qkv_destination_rows,
                    &memory.transient,
                )
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    end_tensor_lifetimes(schedule, qkv.output.iter().map(|block| block.tensor))?;

    info!(stage = "attention", "planning SigLIP encoder stage");
    let attention_phase_start = schedule.phases.len();
    let attention = append_flash_attention_from_a16_qkv_in_arenas(
        schedule,
        &qkv_shards[0],
        &qkv_shards[1],
        &qkv_shards[2],
        FlashAttentionConfig {
            batch_size,
            query_sequence_length: 0,
            sequence_length,
            hidden_size: columns,
            attention_heads: u16::try_from(config.num_attention_heads)?,
            query_block_rows: tuning.attention_query_block_rows,
            key_block_rows: tuning.attention_key_block_rows,
            tile_count,
            data_base,
            data_limit,
        },
        &memory.transient,
    )?;
    log_attention_blocking("encoder", &attention);
    specialize_attention_phases(schedule, attention_phase_start, &attention);
    if !retain_diagnostics {
        end_tensor_lifetimes(
            schedule,
            qkv_shards
                .iter()
                .flat_map(|shards| shards.iter().map(|shard| shard.tensor)),
        )?;
    }
    let attention_shards = append_flash_attention_to_a16_row_shards_in_arenas(
        schedule,
        &attention,
        &memory.transient,
    )?;
    info!(stage = "attention_output", "planning SigLIP encoder stage");
    let output_weight = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.weight")?)?;
    let output_data_type = precision
        .attention_output
        .gemm_data_type(output_weight.iter().copied());
    let attention_output_inner = projection_inner(tuning.attention_output_inner_block_columns);
    let output_row_block_dimension = if matches!(output_data_type, GemmDataType::F8F143 { .. }) {
        row_block_dimension
    } else {
        tuned_gemm_rows
    };
    let output_projection_phase_start = schedule.phases.len();
    let output_projection_block_columns = choose_gemm_output_block_columns(
        rows,
        output_row_block_dimension,
        attention_output_inner,
        columns,
        tile_count,
        output_data_type,
        projection_output(tuning.attention_output_block_columns),
    )?;
    let output_projection_allocation_start = schedule.allocations.len();
    let output_projection = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &attention_shards,
        gemm_config_with_inner(
            rows,
            columns,
            columns,
            output_projection_block_columns,
            attention_output_inner,
            output_row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            output_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.attention_output.weight"),
        columns,
        columns,
        &output_projection.right,
        output_data_type,
        output_projection_phase_start,
        |row, column| output_weight[usize::from(column) * usize::from(columns) + usize::from(row)],
    )?;
    make_tensors_resident_since(
        schedule,
        output_projection_allocation_start,
        output_projection.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, attention_shards.iter().map(|shard| shard.tensor))?;
    }
    let output_bias = model.tensor_f32(&model.layer_name(layer, "self_attn.out_proj.bias")?)?;
    let output_adjustment_allocation_start = schedule.allocations.len();
    let output_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &output_projection.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.attention_output.bias"),
            u16::try_from(output_adjustment.len())?,
            64,
            &output_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &output_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| output_bias[usize::from(column)],
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        output_adjustment_allocation_start,
        output_adjustment.iter().map(|block| block.tensor),
    )?;
    // Residual edges keep their preferred layout until the current placement
    // can no longer hold the residual, projection, and normalized output
    // together. Only that edge is then transitioned to a smaller shard grid.
    let residual_source_rows = input.iter().map(|shard| shard.rows).max().unwrap_or(1);
    let residual_rows = choose_row_shard_rows_for_copies_in_arenas(
        schedule,
        rows,
        columns,
        residual_source_rows,
        2,
        &memory.transient,
    )
    .ok_or_else(|| {
        CompileError::Memory("attention residual has no feasible row-shard placement".into())
    })?;
    let residual_input = if residual_rows == residual_source_rows {
        input.to_vec()
    } else {
        let transitioned = append_a16_to_a16_row_shards_reblocked_in_arenas(
            schedule,
            input,
            residual_rows,
            &memory.transient,
        )?;
        if !retain_diagnostics {
            end_tensor_lifetimes(schedule, input.iter().map(|shard| shard.tensor))?;
        }
        transitioned
    };
    info!(
        source_row_block_rows = residual_source_rows,
        destination_row_block_rows = residual_rows,
        "selected attention residual row-shard placement"
    );
    let projected_shards = if output_row_block_dimension == residual_rows {
        append_c16_to_a16_row_shards(
            schedule,
            &output_projection.output,
            RowShardTransitionConfig {
                columns,
                data_base,
                data_limit,
            },
        )?
    } else {
        append_c16_to_a16_row_shards_reblocked_in_arenas(
            schedule,
            &output_projection.output,
            columns,
            residual_rows,
            &memory.transient,
        )?
    };
    end_tensor_lifetimes(
        schedule,
        output_projection.output.iter().map(|block| block.tensor),
    )?;
    info!(stage = "norm2_mlp", "planning SigLIP encoder stage");
    let norm2_phase_start = schedule.phases.len();
    let norm2_allocation_start = schedule.allocations.len();
    let norm2 = append_add_affine_layer_norm_f16_with_memory_policy(
        schedule,
        &projected_shards,
        &residual_input,
        AppendAffineLayerNormConfig {
            data_base,
            data_limit,
            epsilon_bits: config.layer_norm_eps.to_bits(),
        },
        memory,
    )?;
    let attention_residual = projected_shards;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, residual_input.iter().map(|shard| shard.tensor))?;
    }
    push_layer_norm_affine(
        schedule,
        host,
        model,
        layer,
        "layer_norm2",
        columns,
        &norm2.affine,
        norm2_allocation_start,
    )?;
    let intermediate_columns = u16::try_from(config.intermediate_size.div_ceil(64) * 64)?;
    let mlp_up_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.weight")?)?;
    let mlp_up_data_type = precision
        .mlp_up
        .gemm_data_type(mlp_up_weight.iter().copied());
    let mlp_down_weight = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.weight")?)?;
    let mlp_down_data_type = precision
        .mlp_down
        .gemm_data_type(mlp_down_weight.iter().copied());
    let mlp_up_inner = projection_inner(tuning.mlp_up_inner_block_columns);
    let mut mlp_up_row_block_dimension = if matches!(mlp_up_data_type, GemmDataType::F8F143 { .. })
    {
        row_block_dimension
    } else if tuning.automatic_gemm_row_blocks && tuning.gemm_row_block_rows == 0 {
        choose_gemm_row_block_for(
            rows,
            mlp_up_inner,
            intermediate_columns,
            64,
            tile_count,
            mlp_up_data_type,
        )
        .ok_or("MLP-up GEMM shape has no feasible row blocking")?
    } else {
        tuned_gemm_rows
    };
    let mlp_up_phase_start = schedule.phases.len();
    let mlp_up_output_block_columns = choose_gemm_output_block_columns(
        rows,
        mlp_up_row_block_dimension,
        mlp_up_inner,
        intermediate_columns,
        tile_count,
        mlp_up_data_type,
        projection_output(tuning.mlp_up_output_block_columns),
    )?;
    let mut planned_mlp_down_row_block_dimension = None;
    if tuning.automatic_gemm_row_blocks
        && tuning.gemm_row_block_rows == 0
        && !matches!(mlp_up_data_type, GemmDataType::F8F143 { .. })
    {
        let independent_up = choose_gemm_row_block_for_shape(
            rows,
            columns,
            mlp_up_inner,
            intermediate_columns,
            mlp_up_output_block_columns,
            tile_count,
            mlp_up_data_type,
        )
        .ok_or("MLP-up GEMM has no row blocking for its selected output blocks")?;
        let independent_down = choose_gemm_row_block_for_shape(
            rows,
            intermediate_columns,
            mlp_up_output_block_columns,
            columns,
            64,
            tile_count,
            mlp_down_data_type,
        )
        .ok_or("MLP-down GEMM shape has no feasible row blocking")?;
        let staging_bound = a16_reblock_staging_upper_bound(
            rows,
            independent_up,
            independent_down,
            mlp_up_output_block_columns,
            intermediate_columns,
            tile_count,
        );
        if staging_bound <= ipu_exchange::EXCHANGE_WINDOW_BYTES {
            mlp_up_row_block_dimension = independent_up;
            planned_mlp_down_row_block_dimension = Some(independent_down);
        } else {
            let shared = choose_shared_gemm_row_block(
                rows,
                tile_count,
                columns,
                mlp_up_inner,
                intermediate_columns,
                mlp_up_output_block_columns,
                mlp_up_data_type,
                intermediate_columns,
                mlp_up_output_block_columns,
                columns,
                64,
                mlp_down_data_type,
            )
            .ok_or("MLP GEMMs have no feasible shared row blocking")?;
            mlp_up_row_block_dimension = shared;
            planned_mlp_down_row_block_dimension = Some(shared);
        }
    }
    info!(
        operation = "mlp_up",
        row_block_rows = mlp_up_row_block_dimension,
        output_block_columns = mlp_up_output_block_columns,
        "selected SigLIP GEMM blocking"
    );
    let mlp_up_allocation_start = schedule.allocations.len();
    let mlp_up = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        &norm2.output,
        gemm_config_with_inner(
            rows,
            columns,
            intermediate_columns,
            mlp_up_output_block_columns,
            mlp_up_inner,
            mlp_up_row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            mlp_up_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.mlp_up.weight"),
        columns,
        intermediate_columns,
        &mlp_up.right,
        mlp_up_data_type,
        mlp_up_phase_start,
        |row, column| {
            if usize::from(column) < config.intermediate_size {
                mlp_up_weight[usize::from(column) * usize::from(columns) + usize::from(row)]
            } else {
                0.0
            }
        },
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_up_allocation_start,
        mlp_up.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, norm2.output.iter().map(|shard| shard.tensor))?;
    }
    let mlp_up_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc1.bias")?)?;
    let mlp_up_adjustment_allocation_start = schedule.allocations.len();
    let mlp_up_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &mlp_up.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_up.bias"),
            u16::try_from(mlp_up_adjustment.len())?,
            mlp_up_output_block_columns,
            &mlp_up_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &mlp_up_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| mlp_up_bias.get(usize::from(column)).copied().unwrap_or(0.0),
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_up_adjustment_allocation_start,
        mlp_up_adjustment.iter().map(|block| block.tensor),
    )?;
    let mlp_gelu =
        append_c16_to_a16_blocks_gelu_f16_in_arenas(schedule, &mlp_up.output, &memory.transient)?;
    let mlp_down_inner_block_dimension = mlp_gelu
        .first()
        .map(|block| block.columns)
        .ok_or("MLP GeLU produced no input blocks")?;
    if mlp_gelu
        .iter()
        .any(|block| block.columns != mlp_down_inner_block_dimension)
    {
        return Err("MLP GeLU blocks have inconsistent column widths".into());
    }
    let mlp_down_row_block_dimension = if let Some(planned) = planned_mlp_down_row_block_dimension {
        planned
    } else if tuning.automatic_gemm_row_blocks && tuning.gemm_row_block_rows == 0 {
        choose_gemm_row_block_for_shape(
            rows,
            intermediate_columns,
            mlp_down_inner_block_dimension,
            columns,
            64,
            tile_count,
            mlp_down_data_type,
        )
        .ok_or("MLP-down GEMM shape has no feasible row blocking")?
    } else {
        tuned_gemm_rows
    };
    info!(
        operation = "mlp_down",
        row_block_rows = mlp_down_row_block_dimension,
        inner_block_columns = mlp_down_inner_block_dimension,
        "selected SigLIP GEMM blocking"
    );
    end_tensor_lifetimes(schedule, mlp_up.output.iter().map(|block| block.tensor))?;
    let mlp_down_phase_start = schedule.phases.len();
    let mlp_down_output_block_columns = choose_gemm_output_block_columns(
        rows,
        mlp_down_row_block_dimension,
        mlp_down_inner_block_dimension,
        columns,
        tile_count,
        mlp_down_data_type,
        projection_output(tuning.mlp_down_output_block_columns),
    )?;
    let mlp_down_allocation_start = schedule.allocations.len();
    let mlp_down = append_blocked_gemm_f16_with_a16_blocks_with_memory_policy(
        schedule,
        &mlp_gelu,
        gemm_config_with_inner(
            rows,
            intermediate_columns,
            columns,
            mlp_down_output_block_columns,
            mlp_down_inner_block_dimension,
            mlp_down_row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            mlp_down_data_type,
            retain_profile_metadata,
        ),
        memory,
    )?;
    push_gemm_weight(
        schedule,
        host,
        &format!("{prefix}.mlp_down.weight"),
        intermediate_columns,
        columns,
        &mlp_down.right,
        mlp_down_data_type,
        mlp_down_phase_start,
        |row, column| {
            if usize::from(row) < config.intermediate_size {
                mlp_down_weight[usize::from(column) * config.intermediate_size + usize::from(row)]
            } else {
                0.0
            }
        },
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_down_allocation_start,
        mlp_down.right.iter().map(|block| block.tensor),
    )?;
    if !retain_diagnostics {
        end_tensor_lifetimes(schedule, mlp_gelu.iter().map(|block| block.tensor))?;
    }
    let mlp_down_bias = model.tensor_f32(&model.layer_name(layer, "mlp.fc2.bias")?)?;
    let mlp_down_adjustment_allocation_start = schedule.allocations.len();
    let mlp_down_adjustment =
        append_bias_f16_c16_in_arenas(schedule, &mlp_down.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{prefix}.mlp_down.bias"),
            u16::try_from(mlp_down_adjustment.len())?,
            64,
            &mlp_down_adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(
            &mlp_down_adjustment,
            BlockLayout::AmpC16F16,
            |_row, column| mlp_down_bias[usize::from(column)],
        ),
    )?;
    make_tensors_resident_since(
        schedule,
        mlp_down_adjustment_allocation_start,
        mlp_down_adjustment.iter().map(|block| block.tensor),
    )?;
    let output_row_block_dimension = attention_residual
        .iter()
        .map(|shard| shard.rows)
        .max()
        .ok_or("attention residual has no row shards")?;
    let output = if mlp_down_row_block_dimension == output_row_block_dimension {
        append_c16_to_a16_row_shards(
            schedule,
            &mlp_down.output,
            RowShardTransitionConfig {
                columns,
                data_base,
                data_limit,
            },
        )?
    } else {
        append_c16_to_a16_row_shards_reblocked_in_arenas(
            schedule,
            &mlp_down.output,
            columns,
            output_row_block_dimension,
            &memory.transient,
        )?
    };
    end_tensor_lifetimes(schedule, mlp_down.output.iter().map(|block| block.tensor))?;
    let output = append_add_f16_row_shards_in_place(schedule, &output, &attention_residual)?;
    if !retain_diagnostics {
        end_tensor_lifetimes(
            schedule,
            attention_residual.iter().map(|shard| shard.tensor),
        )?;
    }
    info!(stage = "complete", "planned SigLIP encoder layer");
    let layer_phase_end = schedule.phases.len();
    Ok(SiglipEncoderLayer {
        output,
        norm2: norm2.output,
        mlp_gelu,
        attention,
        diagnostics: retain_diagnostics.then(|| SiglipEncoderDiagnostics {
            input: input.to_vec(),
            norm1: norm.output,
            qkv: qkv_shards.try_into().expect("QKV has three projections"),
            attention_hidden: attention_shards,
            attention_residual,
        }),
        profile_stages: vec![
            SiglipProfileStage {
                name: format!("{prefix}.norm1_qkv"),
                phases: layer_phase_start..attention_phase_start,
            },
            SiglipProfileStage {
                name: format!("{prefix}.attention"),
                phases: attention_phase_start..output_projection_phase_start,
            },
            SiglipProfileStage {
                name: format!("{prefix}.attention_output"),
                phases: output_projection_phase_start..norm2_phase_start,
            },
            SiglipProfileStage {
                name: format!("{prefix}.norm2_mlp_up"),
                phases: norm2_phase_start..mlp_down_phase_start,
            },
            SiglipProfileStage {
                name: format!("{prefix}.mlp_down"),
                phases: mlp_down_phase_start..layer_phase_end,
            },
        ],
    })
}

fn gemm_config(
    rows: u16,
    inner_dimension: u16,
    columns: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    data_type: GemmDataType,
    retain_profile_metadata: bool,
) -> BlockedGemmConfig {
    gemm_config_with_inner(
        rows,
        inner_dimension,
        columns,
        64,
        64,
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        data_type,
        retain_profile_metadata,
    )
}

fn gemm_config_with_inner(
    rows: u16,
    inner_dimension: u16,
    columns: u16,
    block_dimension: u16,
    inner_block_dimension: u16,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    data_type: GemmDataType,
    retain_profile_metadata: bool,
) -> BlockedGemmConfig {
    BlockedGemmConfig {
        rows,
        inner_dimension,
        columns,
        block_dimension,
        inner_block_dimension,
        row_block_dimension,
        tile_count,
        data_base,
        data_limit,
        data_type,
        retain_profile_metadata,
    }
}

fn push_gemm_weight(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    name: &str,
    rows: u16,
    columns: u16,
    placements: &[BlockPlacement],
    data_type: GemmDataType,
    phase_start: usize,
    value: impl Fn(u16, u16) -> f32 + Sync,
) -> Result<()> {
    let (dtype, bytes) = match data_type {
        GemmDataType::F16 => (
            "f16",
            blocked_matrix_f16(placements, BlockLayout::AmpB16x16, value),
        ),
        GemmDataType::F16F8Weights { .. } => {
            let scales = f143_block_scales(placements, &value);
            let phase_end = schedule.phases.len();
            set_f8_weight_block_scales_in_phases(
                schedule,
                phase_start..phase_end,
                placements,
                &scales,
            )?;
            (
                "f8-f143-block-scaled",
                blocked_matrix_f8_f143_by_block(placements, BlockLayout::AmpB16x16, &scales, value),
            )
        }
        GemmDataType::F8F143 { input_scale, .. } => {
            let scales = f143_block_scales(placements, &value);
            let phase_end = schedule.phases.len();
            set_native_f8_weight_block_scales_in_phases(
                schedule,
                phase_start..phase_end,
                input_scale,
                placements,
                &scales,
            )?;
            (
                "f8-f143-block-scaled",
                blocked_matrix_f8_f143_by_block(placements, BlockLayout::AmpB32x16, &scales, value),
            )
        }
        GemmDataType::F32 => {
            return Err("SigLIP weight serialization does not support this GEMM data type".into());
        }
    };
    host.push(
        block_binding_typed(
            name,
            rows,
            columns,
            placements,
            dtype,
            u64::from(data_type.weight_element_bytes()),
        ),
        bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_a16_linear_c16_with_memory_policy(
    schedule: &mut Schedule,
    input: &[RowShardPlacement],
    rows: u16,
    padded_inner: u16,
    padded_output: u16,
    actual_inner: u16,
    actual_output: u16,
    output_offset: u16,
    weight: &[f32],
    bias: &[f32],
    name: &str,
    row_block_dimension: u16,
    tile_count: u16,
    data_base: u32,
    data_limit: u32,
    memory: &MemoryPolicy,
    host: &mut HostTensorSet,
) -> Result<Vec<BlockPlacement>> {
    if weight.len() < usize::from(output_offset + actual_output) * usize::from(actual_inner)
        || bias.len() < usize::from(output_offset + actual_output)
    {
        return Err(format!("{name} weight or bias is smaller than its declared slice").into());
    }
    let gemm = append_blocked_gemm_f16_with_a16_input_with_memory_policy(
        schedule,
        input,
        gemm_config(
            rows,
            padded_inner,
            padded_output,
            row_block_dimension,
            tile_count,
            data_base,
            data_limit,
            GemmDataType::F16,
            true,
        ),
        memory,
    )?;
    host.push(
        block_binding_typed(
            &format!("{name}.weight"),
            padded_inner,
            padded_output,
            &gemm.right,
            "f16",
            2,
        ),
        blocked_matrix_f16(&gemm.right, BlockLayout::AmpB16x16, |row, column| {
            if row < actual_inner && column < actual_output {
                weight[usize::from(output_offset + column) * usize::from(actual_inner)
                    + usize::from(row)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, gemm.right.iter().map(|block| block.tensor))?;
    let adjustment = append_bias_f16_c16_in_arenas(schedule, &gemm.output, &memory.resident)?;
    host.push(
        block_binding_typed(
            &format!("{name}.bias"),
            u16::try_from(adjustment.len())?,
            64,
            &adjustment,
            "f16",
            2,
        ),
        blocked_matrix_f16(&adjustment, BlockLayout::AmpC16F16, |_row, column| {
            if column < actual_output {
                bias[usize::from(output_offset + column)]
            } else {
                0.0
            }
        }),
    )?;
    make_tensors_resident(schedule, adjustment.iter().map(|block| block.tensor))?;
    Ok(gemm.output)
}

fn push_named_layer_norm_affine(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    name: &str,
    columns: u16,
    placements: &[RowShardPlacement],
    weight: &[f32],
    bias: &[f32],
    allocation_start: usize,
) -> Result<()> {
    if weight.len() != usize::from(columns) || bias.len() != usize::from(columns) {
        return Err(format!("{name} affine dimensions do not match {columns} columns").into());
    }
    let mut bytes = Vec::with_capacity(placements.len() * usize::from(columns) * 2);
    for placement in placements {
        for row in placement.row_start..placement.row_start + placement.rows {
            let values = match row {
                0 => weight,
                1 => bias,
                _ => {
                    return Err(format!("layer norm affine row {row} is outside scale/bias").into());
                }
            };
            bytes.extend(
                values
                    .iter()
                    .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes()),
            );
        }
    }
    host.push(row_shard_binding(name, 2, columns, placements), bytes)?;
    make_tensors_resident_since(
        schedule,
        allocation_start,
        placements.iter().map(|placement| placement.tensor),
    )?;
    Ok(())
}

fn push_layer_norm_affine(
    schedule: &mut Schedule,
    host: &mut HostTensorSet,
    model: &SiglipWeights,
    layer: usize,
    norm: &str,
    columns: u16,
    placements: &[RowShardPlacement],
    allocation_start: usize,
) -> Result<()> {
    let weight = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.weight"))?)?;
    let bias = model.tensor_f32(&model.layer_name(layer, &format!("{norm}.bias"))?)?;
    push_named_layer_norm_affine(
        schedule,
        host,
        &format!("encoder_layer_{layer:02}.{norm}.affine"),
        columns,
        placements,
        &weight,
        &bias,
        allocation_start,
    )
}

fn row_shard_binding(name: &str, rows: u16, columns: u16, shards: &[RowShardPlacement]) -> Binding {
    let topology = ipu_exchange::Topology::c600();
    let mut file_offset = 0u64;
    let slices = shards
        .iter()
        .map(|shard| {
            let size = u64::from(shard.rows) * u64::from(shard.columns) * 2;
            let slice = RegionSlice {
                tile: u32::from(topology.physical(shard.tile).unwrap()),
                tile_address: shard.address,
                file_offset,
                size,
            };
            file_offset += size;
            slice
        })
        .collect();
    Binding {
        name: name.into(),
        dtype: "f16".into(),
        shape: vec![u32::from(rows), u32::from(columns)],
        slices,
    }
}

fn projection_blocks(
    output: &[BlockPlacement],
    projection: u16,
    hidden: u16,
) -> Vec<BlockPlacement> {
    let first_column = projection * hidden;
    output
        .iter()
        .filter(|block| {
            block.column_start >= first_column && block.column_start < first_column + hidden
        })
        .map(|block| BlockPlacement {
            block_column: block.block_column - first_column / block.columns,
            column_start: block.column_start - first_column,
            ..*block
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipu_compiler::{KernelCommand, OpId, SpecializationKey};
    use std::collections::BTreeMap;

    fn command(operation: &'static str, output: usize, inputs: &[usize]) -> KernelCommand {
        KernelCommand {
            tile: 0,
            output: TensorId(output),
            inputs: inputs.iter().copied().map(TensorId).collect(),
            arguments: vec![1, 16, 1],
            specialization: Arc::new(SpecializationKey {
                operation: operation.into(),
                shape: vec![1, 16],
                worker_count: 6,
                role: String::new().into(),
                alignment: 8,
                abi: ipu_compiler::KernelAbi::Generic,
            }),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn automatic_output_blocks_respect_weight_transfer_capacity() {
        let output =
            choose_gemm_output_block_columns(4096, 18, 128, 1152, 1472, GemmDataType::F16, 0)
                .unwrap();
        let bytes = u32::from(output) * 128 * GemmDataType::F16.weight_element_bytes();
        assert!(bytes <= ipu_exchange::MAX_TRANSFER_WORDS * 4);
        assert!(
            choose_gemm_output_block_columns(4096, 18, 128, 1152, 1472, GemmDataType::F16, 128,)
                .is_err()
        );
    }

    #[test]
    fn native_fp8_scales_are_instance_data_not_execution_shape() {
        let precision = |scale| SiglipEncoderPrecision {
            qkv: SiglipLinearPrecision::F143Native {
                activation_scale: scale,
            },
            attention_output: SiglipLinearPrecision::F143Expanded,
            mlp_up: SiglipLinearPrecision::F143Expanded,
            mlp_down: SiglipLinearPrecision::F143Expanded,
        };

        assert!(precision(-6).has_same_execution_shape(precision(1)));
        assert!(
            !precision(-6).has_same_execution_shape(SiglipEncoderPrecision {
                qkv: SiglipLinearPrecision::F143Expanded,
                ..precision(-6)
            })
        );
    }

    #[test]
    fn terminal_residual_add_fuses_into_following_layer_norm() {
        let mut schedule = Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: Vec::new().into(),
            tile_count: 1,
            peak_sram: BTreeMap::new(),
        };
        for tensor in 1..=4 {
            schedule.allocations.push(Allocation {
                tensor: TensorId(tensor),
                tile: 0,
                address: 0x60000 + tensor as u32 * 0x100,
                size: 32,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            });
        }
        schedule.phases.push(Phase::Exchange {
            transfers: Vec::new(),
        });
        schedule.phases.push(Phase::Compute {
            op: OpId(1),
            commands: vec![command("add_f16", 1, &[1, 2]).into()],
        });
        let deferred = defer_terminal_residual_add(&mut schedule).unwrap().unwrap();
        assert!(schedule.phases.is_empty());

        schedule.phases.push(Phase::Exchange {
            transfers: Vec::new(),
        });
        schedule.phases.push(Phase::Compute {
            op: OpId(1),
            commands: vec![command("layer_norm_affine_f16", 4, &[1, 3]).into()],
        });
        fuse_deferred_residual_into_layer_norm(&mut schedule, 0, deferred).unwrap();

        let Phase::Compute { commands, .. } = &schedule.phases[1] else {
            panic!("expected fused compute phase");
        };
        assert_eq!(
            commands[0].specialization.operation,
            "add_layer_norm_affine_f16"
        );
        assert_eq!(commands[0].inputs, [TensorId(1), TensorId(2), TensorId(3)]);
    }
}
