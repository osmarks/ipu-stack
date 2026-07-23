use crate::{
    Allocation, AllocationKind, BlockPlacement, BlockedGemmConfig, CompileError, GemmDataType,
    KernelCommand, OpId, Phase, Schedule, SpecializationKey, TensorId, Transfer, plan_blocked_gemm,
};
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockedMlpConfig {
    pub batch: u16,
    pub width: u16,
    pub layers: u16,
    pub block_dimension: u16,
    pub inner_block_dimension: u16,
    pub row_block_dimension: u16,
    pub tile_count: u16,
    pub data_base: u32,
    pub data_limit: u32,
    pub data_type: GemmDataType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedMlpPlan {
    pub schedule: Schedule,
    pub input: Vec<BlockPlacement>,
    pub weights: Vec<Vec<BlockPlacement>>,
    pub output: Vec<BlockPlacement>,
}

pub fn plan_blocked_mlp(config: BlockedMlpConfig) -> Result<BlockedMlpPlan, CompileError> {
    if config.layers == 0 {
        return Err(CompileError::Graph(
            "blocked MLP requires at least one layer".into(),
        ));
    }
    let element_bytes = config.data_type.element_bytes();
    let mut layer_plans = Vec::with_capacity(usize::from(config.layers));
    let mut data_base = config.data_base;
    let mut tensor_base = 0usize;
    for _ in 0..config.layers {
        let mut plan = plan_blocked_gemm(BlockedGemmConfig {
            rows: config.batch,
            inner_dimension: config.width,
            columns: config.width,
            block_dimension: config.block_dimension,
            inner_block_dimension: config.inner_block_dimension,
            row_block_dimension: config.row_block_dimension,
            tile_count: config.tile_count,
            data_base,
            data_limit: config.data_limit,
            data_type: config.data_type,
            retain_profile_metadata: true,
        })?;
        data_base = resident_end(&plan, element_bytes)?
            .checked_add(31)
            .map(|address| address & !31)
            .ok_or_else(|| CompileError::Memory("MLP data arena alignment overflow".into()))?;
        remap_tensors(&mut plan, tensor_base);
        tensor_base = maximum_tensor(&plan)
            .checked_add(1)
            .ok_or_else(|| CompileError::Graph("MLP tensor ID overflow".into()))?;
        layer_plans.push(plan);
    }

    let final_tensor_base = tensor_base;
    let mut output = Vec::with_capacity(layer_plans.last().unwrap().output.len());
    let mut final_cursors = vec![data_base; usize::from(config.tile_count)];
    for (index, source) in layer_plans.last().unwrap().output.iter().enumerate() {
        let tile = u16::try_from(index % usize::from(config.tile_count))
            .map_err(|_| CompileError::Graph("MLP output tile overflow".into()))?;
        let size = u32::from(source.rows) * u32::from(source.columns) * element_bytes;
        let address = final_cursors[usize::from(tile)];
        final_cursors[usize::from(tile)] = address
            .checked_add(size)
            .ok_or_else(|| CompileError::Memory("MLP output address overflow".into()))?;
        output.push(BlockPlacement {
            tensor: TensorId(final_tensor_base + index),
            tile,
            address,
            block_row: source.block_row,
            block_column: source.block_column,
            row_start: source.row_start,
            rows: source.rows,
            column_start: source.column_start,
            columns: source.columns,
        });
    }
    if let Some((tile, end)) = final_cursors
        .iter()
        .copied()
        .enumerate()
        .find(|(_, end)| *end > config.data_limit)
    {
        return Err(CompileError::Memory(format!(
            "MLP resident data exhausts tile {tile}: 0x{end:x} exceeds 0x{:x}",
            config.data_limit
        )));
    }

    let input = layer_plans[0].left.clone();
    let weights = layer_plans.iter().map(|plan| plan.right.clone()).collect();
    let mut phases = Vec::new();
    let mut allocations = Vec::new();
    let mut previous_output: Option<Vec<BlockPlacement>> = None;
    for (layer, mut plan) in layer_plans.into_iter().enumerate() {
        annotate_layer(&mut plan.schedule, layer);
        let deferred_output = if layer + 1 < usize::from(config.layers) {
            Some(defer_single_wave_output(&mut plan)?)
        } else {
            None
        };
        if let Some(source) = previous_output.as_ref() {
            append_activation_transition(
                &mut phases,
                source,
                &plan.left,
                layer - 1,
                config.data_type,
            )?;
        }
        append_schedule(&mut phases, &mut allocations, &mut plan.schedule)?;
        previous_output = Some(deferred_output.unwrap_or(plan.output));
    }
    for placement in &output {
        allocations.push(Allocation {
            tensor: placement.tensor,
            tile: placement.tile,
            address: placement.address,
            size: u32::from(placement.rows) * u32::from(placement.columns) * element_bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
    }
    append_activation_transition(
        &mut phases,
        previous_output.as_ref().unwrap(),
        &output,
        usize::from(config.layers) - 1,
        config.data_type,
    )?;

    Ok(BlockedMlpPlan {
        schedule: Schedule {
            layouts: Vec::new(),
            phases,
            allocations: allocations.into(),
            tile_count: config.tile_count,
            peak_sram: BTreeMap::new(),
        },
        input,
        weights,
        output,
    })
}

fn defer_single_wave_output(
    plan: &mut crate::BlockedGemmPlan,
) -> Result<Vec<BlockPlacement>, CompileError> {
    if plan.output.len() > usize::from(plan.schedule.tile_count) || plan.schedule.phases.len() < 3 {
        return Err(CompileError::Graph(
            "MLP accumulator forwarding requires a single-wave GEMM".into(),
        ));
    }
    let evacuation_phase = plan.schedule.phases.len() - 2;
    let commands = match &plan.schedule.phases[evacuation_phase - 1] {
        Phase::Compute { commands, .. } if commands.len() == plan.output.len() => commands,
        _ => {
            return Err(CompileError::Graph(
                "MLP could not identify the final GEMM accumulator phase".into(),
            ));
        }
    };
    let mut accumulators = Vec::with_capacity(plan.output.len());
    for (output, command) in plan.output.iter().zip(commands) {
        let allocation = plan
            .schedule
            .allocations
            .iter()
            .find(|allocation| {
                allocation.tensor == command.output
                    && allocation.tile == command.tile
                    && allocation.kind == AllocationKind::Home
            })
            .ok_or_else(|| CompileError::Graph("MLP accumulator has no home allocation".into()))?;
        accumulators.push(BlockPlacement {
            tensor: command.output,
            tile: command.tile,
            address: allocation.address,
            block_row: output.block_row,
            block_column: output.block_column,
            row_start: output.row_start,
            rows: output.rows,
            column_start: output.column_start,
            columns: output.columns,
        });
    }
    let discarded_outputs = plan
        .output
        .iter()
        .map(|placement| placement.tensor)
        .collect::<HashSet<_>>();
    plan.schedule.allocations.retain(|allocation| {
        allocation.live_from < evacuation_phase && !discarded_outputs.contains(&allocation.tensor)
    });
    plan.schedule.phases.truncate(evacuation_phase);
    Ok(accumulators)
}

fn annotate_layer(schedule: &mut Schedule, layer: usize) {
    for phase in &mut schedule.phases {
        let Phase::Compute { commands, .. } = phase else {
            continue;
        };
        for command in commands {
            let command = Arc::make_mut(command);
            command.metadata.insert("layer".into(), layer.to_string());
            if let Some(label) = command.metadata.get_mut("label") {
                *label = format!("MLP layer {layer}: {label}");
            }
        }
    }
}

fn resident_end(plan: &crate::BlockedGemmPlan, element_bytes: u32) -> Result<u32, CompileError> {
    plan.left
        .iter()
        .chain(&plan.right)
        .chain(&plan.output)
        .map(|placement| {
            placement
                .address
                .checked_add(
                    u32::from(placement.rows) * u32::from(placement.columns) * element_bytes,
                )
                .ok_or_else(|| CompileError::Memory("MLP resident range overflow".into()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| CompileError::Graph("empty GEMM plan in MLP".into()))
}

fn maximum_tensor(plan: &crate::BlockedGemmPlan) -> usize {
    plan.schedule.allocations.maximum_tensor_id().unwrap_or(0)
}

fn remap_tensors(plan: &mut crate::BlockedGemmPlan, base: usize) {
    for placement in plan
        .left
        .iter_mut()
        .chain(&mut plan.right)
        .chain(&mut plan.output)
    {
        placement.tensor.0 += base;
    }
    for allocation in &mut plan.schedule.allocations {
        allocation.tensor.0 += base;
    }
    for phase in &mut plan.schedule.phases {
        match phase {
            Phase::Exchange { transfers } => {
                for transfer in transfers {
                    transfer.tensor.0 += base;
                }
            }
            Phase::Compute { commands, .. } => {
                for command in commands {
                    let command = Arc::make_mut(command);
                    command.output.0 += base;
                    for input in &mut command.inputs {
                        input.0 += base;
                    }
                }
            }
        }
    }
}

fn append_schedule(
    phases: &mut Vec<Phase>,
    allocations: &mut Vec<Allocation>,
    schedule: &mut Schedule,
) -> Result<(), CompileError> {
    let phase_base = phases.len();
    for allocation in &mut schedule.allocations {
        if allocation.live_from != 0 {
            allocation.live_from = allocation
                .live_from
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("MLP allocation phase overflow".into()))?;
        }
        if allocation.live_until != usize::MAX {
            allocation.live_until = allocation
                .live_until
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("MLP allocation phase overflow".into()))?;
        }
        if let AllocationKind::ExchangeStaging { phase } = &mut allocation.kind {
            *phase = phase
                .checked_add(phase_base)
                .ok_or_else(|| CompileError::Graph("MLP staging phase overflow".into()))?;
        }
    }
    for phase in &mut schedule.phases {
        if let Phase::Compute { op, .. } = phase {
            op.0 =
                op.0.checked_add(phase_base)
                    .ok_or_else(|| CompileError::Graph("MLP operation ID overflow".into()))?;
        }
    }
    allocations.append(&mut schedule.allocations);
    phases.append(&mut schedule.phases);
    Ok(())
}

fn append_activation_transition(
    phases: &mut Vec<Phase>,
    source: &[BlockPlacement],
    destination: &[BlockPlacement],
    layer: usize,
    data_type: GemmDataType,
) -> Result<(), CompileError> {
    if source.len() != destination.len() {
        return Err(CompileError::Graph(
            "MLP activation block counts differ between layers".into(),
        ));
    }
    let compute_phase = phases.len() + 1;
    let mut transfers = Vec::with_capacity(source.len());
    let mut commands = Vec::with_capacity(source.len());
    let mut occupied_destinations = HashSet::default();
    for (source, destination) in source.iter().zip(destination) {
        if source.row_start != destination.row_start
            || source.rows != destination.rows
            || source.column_start != destination.column_start
            || source.columns != destination.columns
        {
            return Err(CompileError::Graph(
                "MLP activation block layouts differ between layers".into(),
            ));
        }
        let bytes = u32::from(source.rows) * u32::from(source.columns) * data_type.element_bytes();
        if source.tile != destination.tile {
            if !occupied_destinations.insert(destination.tile) {
                return Err(CompileError::Graph(
                    "MLP transition assigns multiple blocks to one destination tile".into(),
                ));
            }
            transfers.push(Transfer {
                source_tile: source.tile,
                destination_tile: destination.tile,
                tensor: source.tensor,
                bytes,
                staging_address: Some(ipu_exchange::EXCHANGE_WINDOW_BASE),
            });
        }
        commands.push(KernelCommand {
            tile: destination.tile,
            output: destination.tensor,
            inputs: vec![source.tensor, source.tensor],
            arguments: vec![
                u32::from(source.rows),
                u32::from(source.rows / 6),
                u32::from(source.rows % 6),
            ],
            specialization: Arc::new(SpecializationKey {
                operation: match data_type {
                    GemmDataType::F16
                    | GemmDataType::F16F8Weights { .. }
                    | GemmDataType::F8F143 { .. } => "gelu_f16_c16_to_a16",
                    GemmDataType::F32 => "gelu_c16_to_a8",
                }
                .into(),
                shape: vec![usize::from(source.rows), usize::from(source.columns)],
                worker_count: 6,
                role: format!("layer-{layer}").into(),
                alignment: 8,
            }),
            metadata: BTreeMap::from([
                ("label".into(), format!("MLP layer {layer} GeLU")),
                ("layer".into(), layer.to_string()),
                ("rows".into(), source.rows.to_string()),
                ("columns".into(), source.columns.to_string()),
                ("output_block_row".into(), source.block_row.to_string()),
                (
                    "output_block_column".into(),
                    source.block_column.to_string(),
                ),
            ]),
        });
    }
    phases.push(Phase::Exchange { transfers });
    phases.push(Phase::Compute {
        op: OpId(compute_phase),
        commands: commands.into_iter().map(Arc::new).collect(),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_mlp_has_one_weight_layout_per_layer_and_fused_boundaries() {
        let layers = 8;
        let row_block_dimension = crate::choose_gemm_row_block(512, 64, 2048, 64, 1472).unwrap();
        let plan = plan_blocked_mlp(BlockedMlpConfig {
            batch: 512,
            width: 2048,
            layers,
            block_dimension: 64,
            inner_block_dimension: 64,
            row_block_dimension,
            tile_count: 1472,
            data_base: 0xa0000,
            data_limit: 0xe8000,
            data_type: GemmDataType::F32,
        })
        .unwrap();

        assert_eq!(plan.weights.len(), usize::from(layers));
        assert_eq!(plan.input.len(), plan.output.len());
        assert!(plan.input.iter().zip(&plan.output).all(|(input, output)| {
            input.row_start == output.row_start
                && input.rows == output.rows
                && input.column_start == output.column_start
                && input.columns == output.columns
        }));
        let gelu_phases = plan
            .schedule
            .phases
            .iter()
            .filter(|phase| {
                matches!(phase, Phase::Compute { commands, .. }
                    if commands.first().is_some_and(|command|
                        command.specialization.operation == "gelu_c16_to_a8"))
            })
            .count();
        assert_eq!(gelu_phases, usize::from(layers));
        let copy_phases = plan
            .schedule
            .phases
            .iter()
            .filter(|phase| {
                matches!(phase, Phase::Compute { commands, .. }
                    if commands.first().is_some_and(|command|
                        command.specialization.operation == "copy_u64"))
            })
            .count();
        assert_eq!(copy_phases, 1);
        let homes = plan
            .schedule
            .allocations
            .iter()
            .filter(|allocation| allocation.kind.has_home_address())
            .map(|allocation| (allocation.tensor, allocation.tile))
            .collect::<HashSet<_>>();
        for (phase_index, phase) in plan.schedule.phases.iter().enumerate() {
            if let Phase::Compute { commands, .. } = phase {
                for command in commands {
                    assert!(
                        homes.contains(&(command.output, command.tile)),
                        "phase {phase_index} kernel {} has no output allocation for tensor {} on tile {}",
                        command.specialization.operation,
                        command.output.0,
                        command.tile
                    );
                }
            }
        }
        assert!(plan.schedule.allocations.iter().all(|allocation| {
            allocation.address < 0xa0000
                || allocation.address.saturating_add(allocation.size) <= 0xe8000
        }));
    }
}
