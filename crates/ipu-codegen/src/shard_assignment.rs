//! Global assignment of packed tensor-grid coordinates to logical tiles.
//!
//! Layouts describe tensor coordinates, while this pass chooses which tile
//! owns each coordinate. Values with the same packed grid share an assignment,
//! keeping operator operands aligned. Rearrangements are the communication
//! edges used to choose among inexpensive structured permutations.

use crate::ShardView;
use crate::low::{
    LowLoweringError, LowLoweringResult, LowShard, LowShardId, ShardExtent, intersect_extents,
};
use crate::mid::{MidGraph, MidOperationKind, TensorTiling};
use crate::storage::view_byte_spans;
use ipu_exchange::{MAX_TRANSFER_WORDS, Topology, plan_event_cycles};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TileGrid(Vec<u16>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct AssignmentScore {
    estimated_cycles: u128,
    remote_bytes: u128,
    route_work: u128,
}

struct ConversionPattern {
    input_grid: TileGrid,
    output_grid: TileGrid,
    input_count: usize,
    outputs: Vec<Vec<IntersectionPattern>>,
    transfers: Vec<TransferPattern>,
}

struct IntersectionPattern {
    sources: Vec<(usize, usize)>,
    bytes: u64,
}

struct TransferPattern {
    source: usize,
    bytes: u64,
    span_words: Vec<u32>,
}

#[derive(Clone)]
struct RouteTiming {
    sender: u32,
    receivers: Vec<u32>,
}

pub(crate) fn assign_shards(
    graph: &MidGraph,
    canonical: &[Vec<LowShardId>],
    shards: &mut [LowShard],
) -> LowLoweringResult<()> {
    let grids = graph
        .values
        .iter()
        .filter_map(|value| packed_grid(&value.tensor_type.format.layout.tiling))
        .collect::<BTreeSet<_>>();
    let mut assignments = grids
        .iter()
        .map(|grid| Ok((grid.clone(), identity_assignment(grid)?)))
        .collect::<LowLoweringResult<BTreeMap<_, _>>>()?;
    let patterns = conversion_patterns(graph, canonical, shards)?;

    // Coordinate descent lets a grid account for every conversion in which it
    // participates rather than committing based on the first use alone.
    for _ in 0..3 {
        let mut changed = false;
        for grid in &grids {
            let current = assignments
                .get(grid)
                .cloned()
                .ok_or(LowLoweringError::IdOverflow)?;
            let mut best = current.clone();
            let mut best_score = graph_score(&patterns, &assignments)?;
            for candidate in structured_assignments(grid)? {
                assignments.insert(grid.clone(), candidate.clone());
                let score = graph_score(&patterns, &assignments)?;
                if score < best_score {
                    best = candidate;
                    best_score = score;
                }
            }
            changed |= best != current;
            assignments.insert(grid.clone(), best);
        }
        if !changed {
            break;
        }
    }

    let non_identity_grids = assignments
        .iter()
        .filter_map(|(grid, assignment)| {
            (identity_assignment(grid).ok().as_ref() != Some(assignment)).then_some(grid)
        })
        .collect::<Vec<_>>();
    if !non_identity_grids.is_empty() {
        tracing::debug!(
            ?non_identity_grids,
            score = ?graph_score(&patterns, &assignments)?,
            "selected structured shard assignments"
        );
    }

    for value in &graph.values {
        let Some(grid) = packed_grid(&value.tensor_type.format.layout.tiling) else {
            continue;
        };
        let assignment = assignments.get(&grid).ok_or(LowLoweringError::IdOverflow)?;
        for (ordinal, shard) in canonical[value.id.index() as usize]
            .iter()
            .copied()
            .enumerate()
        {
            shards[shard.index() as usize].tile = assignment[ordinal];
        }
    }
    Ok(())
}

fn conversion_patterns(
    graph: &MidGraph,
    canonical: &[Vec<LowShardId>],
    shards: &[LowShard],
) -> LowLoweringResult<Vec<ConversionPattern>> {
    let mut patterns = Vec::new();
    for operation in &graph.operations {
        if !matches!(operation.kind, MidOperationKind::Rearrange { .. }) {
            continue;
        }
        let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
        else {
            return Err(LowLoweringError::ResultArity);
        };
        let Some(input_grid) = packed_grid(
            &graph.values[input.index() as usize]
                .tensor_type
                .format
                .layout
                .tiling,
        ) else {
            continue;
        };
        let Some(output_grid) = packed_grid(
            &graph.values[output.index() as usize]
                .tensor_type
                .format
                .layout
                .tiling,
        ) else {
            continue;
        };
        let inputs = &canonical[input.index() as usize];
        let mut transfer_ids = BTreeMap::<(usize, Vec<ShardExtent>), usize>::new();
        let mut transfers = Vec::new();
        let mut output_patterns = Vec::new();
        for &output_shard in &canonical[output.index() as usize] {
            let output_shard = &shards[output_shard.index() as usize];
            let mut intersections = BTreeMap::<Vec<ShardExtent>, Vec<usize>>::new();
            for (input_ordinal, &input_shard) in inputs.iter().enumerate() {
                if let Some(extents) = intersect_extents(
                    &shards[input_shard.index() as usize].extents,
                    &output_shard.extents,
                ) {
                    intersections
                        .entry(extents)
                        .or_default()
                        .push(input_ordinal);
                }
            }
            let mut output = Vec::with_capacity(intersections.len());
            for (extents, sources) in intersections {
                let mut source_transfers = Vec::with_capacity(sources.len());
                for source in sources {
                    let transfer = if let Some(transfer) =
                        transfer_ids.get(&(source, extents.clone())).copied()
                    {
                        transfer
                    } else {
                        let input_shard = inputs[source];
                        let view = ShardView {
                            shard: input_shard,
                            extents: extents.clone(),
                        };
                        let spans = view_byte_spans(&shards[input_shard.index() as usize], &view)?;
                        let transfer = transfers.len();
                        transfers.push(TransferPattern {
                            source,
                            bytes: spans.iter().map(|span| u64::from(span.bytes)).sum(),
                            span_words: spans
                                .iter()
                                .flat_map(|span| {
                                    let words = span.bytes.div_ceil(4);
                                    (0..words.div_ceil(MAX_TRANSFER_WORDS)).map(move |chunk| {
                                        (words - chunk * MAX_TRANSFER_WORDS).min(MAX_TRANSFER_WORDS)
                                    })
                                })
                                .collect(),
                        });
                        transfer_ids.insert((source, extents.clone()), transfer);
                        transfer
                    };
                    source_transfers.push((source, transfer));
                }
                let transfer = &transfers[source_transfers[0].1];
                output.push(IntersectionPattern {
                    sources: source_transfers,
                    bytes: transfer.bytes,
                });
            }
            output_patterns.push(output);
        }
        patterns.push(ConversionPattern {
            input_grid,
            output_grid,
            input_count: inputs.len(),
            outputs: output_patterns,
            transfers,
        });
    }
    Ok(patterns)
}

fn graph_score(
    patterns: &[ConversionPattern],
    assignments: &BTreeMap<TileGrid, Vec<u16>>,
) -> LowLoweringResult<AssignmentScore> {
    patterns
        .iter()
        .try_fold(AssignmentScore::default(), |total, pattern| {
            let score = conversion_score(pattern, assignments)?;
            Ok(AssignmentScore {
                estimated_cycles: total.estimated_cycles + score.estimated_cycles,
                remote_bytes: total.remote_bytes + score.remote_bytes,
                route_work: total.route_work + score.route_work,
            })
        })
}

fn conversion_score(
    pattern: &ConversionPattern,
    assignments: &BTreeMap<TileGrid, Vec<u16>>,
) -> LowLoweringResult<AssignmentScore> {
    let input_tiles = value_assignment(&pattern.input_grid, assignments, pattern.input_count)?;
    let output_tiles = value_assignment(&pattern.output_grid, assignments, pattern.outputs.len())?;
    let mut destinations = vec![Vec::<u16>::new(); pattern.transfers.len()];
    let tile_count = assignments.values().map(Vec::len).max().unwrap_or(0);
    let mut local_pressure = vec![0u128; tile_count];
    let mut remote_bytes = 0u128;

    for (output_ordinal, intersections) in pattern.outputs.iter().enumerate() {
        let destination_tile = output_tiles[output_ordinal];
        for intersection in intersections {
            let input_ordinal = intersection
                .sources
                .iter()
                .copied()
                .find(|(source, _)| input_tiles[*source] == destination_tile)
                .unwrap_or(intersection.sources[0]);
            if input_tiles[input_ordinal.0] == destination_tile {
                local_pressure[usize::from(destination_tile)] +=
                    u128::from(intersection.bytes.div_ceil(16));
            } else {
                remote_bytes += u128::from(intersection.bytes);
                destinations[input_ordinal.1].push(destination_tile);
            }
        }
    }

    let mut pressure = local_pressure;
    let mut route_work = 0u128;
    let topology = Topology::c600();
    let mut route_cache = BTreeMap::<(u16, Vec<u16>, u32), RouteTiming>::new();
    for (transfer_index, mut destinations) in destinations.into_iter().enumerate() {
        if destinations.is_empty() {
            continue;
        }
        destinations.sort_unstable();
        destinations.dedup();
        let input_ordinal = pattern.transfers[transfer_index].source;
        let source_tile = input_tiles[input_ordinal];
        let transfer = &pattern.transfers[transfer_index];
        for &words in &transfer.span_words {
            let timing = if let Some(timing) =
                route_cache.get(&(source_tile, destinations.clone(), words))
            {
                timing.clone()
            } else {
                let timing = if destinations.len() == 1 {
                    let plan = topology.point_to_point(source_tile, destinations[0], words)?;
                    RouteTiming {
                        sender: plan_event_cycles(&plan.sender)?,
                        receivers: vec![plan_event_cycles(&plan.receiver)?],
                    }
                } else {
                    let plan = topology.multicast(source_tile, &destinations, words, 0)?;
                    RouteTiming {
                        sender: plan_event_cycles(&plan.sender)?,
                        receivers: plan
                            .receivers
                            .iter()
                            .map(|row| plan_event_cycles(row))
                            .collect::<Result<_, _>>()?,
                    }
                };
                route_cache.insert((source_tile, destinations.clone(), words), timing.clone());
                timing
            };
            pressure[usize::from(source_tile)] += u128::from(timing.sender);
            route_work += u128::from(timing.sender);
            for (&destination_tile, &cycles) in destinations.iter().zip(&timing.receivers) {
                pressure[usize::from(destination_tile)] += u128::from(cycles);
                route_work += u128::from(cycles);
            }
        }
    }
    let maximum_pressure = pressure.into_iter().max().unwrap_or(0);
    // A pure maximum-endpoint objective overfits small route-timing changes
    // and can increase work across most other tiles. A small mean-pressure
    // term keeps the search sensitive to the critical endpoint without
    // accepting globally busier exchange schedules.
    let mean_pressure_term = route_work.div_ceil(
        u128::try_from(tile_count)
            .unwrap_or(1)
            .saturating_mul(16)
            .max(1),
    );
    Ok(AssignmentScore {
        estimated_cycles: maximum_pressure + mean_pressure_term,
        remote_bytes,
        route_work,
    })
}

fn value_assignment<'a>(
    grid: &TileGrid,
    assignments: &'a BTreeMap<TileGrid, Vec<u16>>,
    count: usize,
) -> LowLoweringResult<&'a [u16]> {
    let assignment = assignments.get(grid).ok_or(LowLoweringError::IdOverflow)?;
    if assignment.len() != count {
        return Err(LowLoweringError::IdOverflow);
    }
    Ok(assignment)
}

fn packed_grid(tiling: &TensorTiling) -> Option<TileGrid> {
    let strides = tiling.axis_strides().ok()?;
    let mut expected = u32::from(tiling.replicas);
    let mut dimensions = Vec::new();
    if tiling.replicas > 1 {
        dimensions.push(tiling.replicas);
    }
    for (axis, stride) in tiling.axes.iter().zip(strides) {
        if stride != expected || axis.partitions == 0 {
            return None;
        }
        expected = expected.checked_mul(u32::from(axis.partitions))?;
        if axis.partitions > 1 {
            dimensions.push(axis.partitions);
        }
    }
    (expected == u32::from(tiling.tile_count)).then_some(TileGrid(dimensions))
}

fn identity_assignment(grid: &TileGrid) -> LowLoweringResult<Vec<u16>> {
    let count = grid
        .0
        .iter()
        .try_fold(1u32, |count, radix| count.checked_mul(u32::from(*radix)))
        .ok_or(LowLoweringError::IdOverflow)?;
    (0..count)
        .map(|tile| u16::try_from(tile).map_err(|_| LowLoweringError::IdOverflow))
        .collect()
}

fn structured_assignments(grid: &TileGrid) -> LowLoweringResult<Vec<Vec<u16>>> {
    let identity = identity_assignment(grid)?;
    let mut candidates = BTreeSet::from([identity.clone()]);
    let mut stride = 1u32;
    for boundary in 0..grid.0.len().saturating_sub(1) {
        let fast = u32::from(grid.0[boundary]);
        let slow = u32::from(grid.0[boundary + 1]);
        for slow_factor in divisors(slow) {
            for fast_low in divisors_including_one(fast) {
                let mut assignment = Vec::with_capacity(identity.len());
                for tile in
                    0..u32::try_from(identity.len()).map_err(|_| LowLoweringError::IdOverflow)?
                {
                    let lower = tile % stride;
                    let pair = (tile / stride) % (fast * slow);
                    let higher = tile / (stride * fast * slow);
                    let fast_coordinate = pair % fast;
                    let slow_coordinate = pair / fast;
                    let permuted_pair = fast_coordinate % fast_low
                        + fast_low * (slow_coordinate % slow_factor)
                        + fast_low * slow_factor * (fast_coordinate / fast_low)
                        + fast * slow_factor * (slow_coordinate / slow_factor);
                    let mapped = lower + stride * (permuted_pair + fast * slow * higher);
                    assignment
                        .push(u16::try_from(mapped).map_err(|_| LowLoweringError::IdOverflow)?);
                }
                candidates.insert(assignment);
            }
        }
        stride = stride
            .checked_mul(fast)
            .ok_or(LowLoweringError::IdOverflow)?;
    }
    Ok(candidates.into_iter().collect())
}

fn divisors(value: u32) -> impl Iterator<Item = u32> {
    (2..=value).filter(move |candidate| value.is_multiple_of(*candidate))
}

fn divisors_including_one(value: u32) -> impl Iterator<Item = u32> {
    (1..=value).filter(move |candidate| value.is_multiple_of(*candidate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputeGraph, TensorShape};
    use crate::low::{ShardDefinition, shard_extents};
    use crate::mid::{
        AmpOrder, ElementOrder, Layout, MemoryClass, MemoryEstimate, MidOperation, MidValue,
        MidValueId, Precision, TensorType,
    };

    #[test]
    fn randomized_structured_assignments_are_permutations() {
        let mut random = fastrand::Rng::with_seed(0x706c_6163_656d_656e);
        for _ in 0..128 {
            let dimensions = (0..random.u32(1..=3))
                .map(|_| [2, 4, 8, 16][random.usize(0..4)])
                .collect::<Vec<_>>();
            let grid = TileGrid(dimensions);
            for assignment in structured_assignments(&grid).unwrap() {
                let mut sorted = assignment.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, identity_assignment(&grid).unwrap());
            }
        }
    }

    #[test]
    fn adjacent_factor_interleaving_preserves_coarse_groups() {
        let grid = TileGrid(vec![16, 32]);
        let candidates = structured_assignments(&grid).unwrap();
        assert!(candidates.iter().any(|assignment| {
            (0..512usize).all(|ordinal| {
                assignment[ordinal] / 32 == u16::try_from(ordinal / 32).unwrap()
                    && assignment[ordinal] % 2 == u16::try_from(ordinal / 16 % 2).unwrap()
            })
        }));
    }

    #[test]
    fn siglip_grid_transpose_selects_no_worse_global_assignment() {
        let shape = TensorShape(vec![4, 256, 4096]);
        let mut source_layout = Layout::amp_output_grid(512, 16, 32);
        source_layout.order = ElementOrder::Amp(AmpOrder::Left);
        source_layout.memory_class = MemoryClass::Ipu21Standard;
        let destination_layout = Layout::amp_left_grid(64, 512, 32, 16);
        let source_type = TensorType::new(shape.0.clone(), Precision::F16, source_layout.clone());
        let destination_type =
            TensorType::new(shape.0.clone(), Precision::F16, destination_layout.clone());
        let origin = ComputeGraph::new()
            .host_input("value", shape.0.clone())
            .unwrap();
        let graph = MidGraph {
            values: vec![
                MidValue {
                    id: MidValueId::from_index(0),
                    tensor_type: source_type.clone(),
                    origin,
                },
                MidValue {
                    id: MidValueId::from_index(1),
                    tensor_type: destination_type.clone(),
                    origin,
                },
            ],
            operations: vec![MidOperation {
                source: None,
                inputs: vec![MidValueId::from_index(0)],
                results: vec![MidValueId::from_index(1)],
                kind: MidOperationKind::Rearrange {
                    from: source_layout,
                    to: destination_layout,
                },
                operator_plan: None,
                conversion_plan: None,
                estimated_cycles: 0,
                memory: MemoryEstimate::default(),
            }],
            ..MidGraph::default()
        };
        let mut shards = Vec::new();
        let mut canonical = vec![Vec::new(), Vec::new()];
        for (value, tensor_type) in [source_type, destination_type].iter().enumerate() {
            for extents in shard_extents(tensor_type).unwrap() {
                let id = LowShardId::from_index(shards.len() as u32);
                shards.push(LowShard {
                    id,
                    tile: canonical[value].len() as u16,
                    tensor_type: tensor_type.clone(),
                    extents,
                    definition: ShardDefinition::Value(MidValueId::from_index(value as u32)),
                });
                canonical[value].push(id);
            }
        }
        let identity = [TileGrid(vec![32, 16]), TileGrid(vec![16, 32])]
            .into_iter()
            .map(|grid| Ok((grid.clone(), identity_assignment(&grid)?)))
            .collect::<LowLoweringResult<BTreeMap<_, _>>>()
            .unwrap();
        let patterns = conversion_patterns(&graph, &canonical, &shards).unwrap();
        let identity_score = graph_score(&patterns, &identity).unwrap();
        assign_shards(&graph, &canonical, &mut shards).unwrap();
        let selected = [
            (
                TileGrid(vec![32, 16]),
                canonical[0]
                    .iter()
                    .map(|id| shards[id.index() as usize].tile)
                    .collect(),
            ),
            (
                TileGrid(vec![16, 32]),
                canonical[1]
                    .iter()
                    .map(|id| shards[id.index() as usize].tile)
                    .collect(),
            ),
        ]
        .into_iter()
        .collect();
        let selected_score = graph_score(&patterns, &selected).unwrap();
        assert_eq!(selected, identity);
        assert_eq!(selected_score, identity_score);
    }
}
