//! Opt-in exact placement diagnostics, using the allocator's own requests.
use super::*;
use crate::memory::TileMemoryMap;
use crate::package::{PackageBuildError, PackageBuildResult};
use ipu_package::{Application, TILE_MEMORY_BASE, TILE_MEMORY_SIZE};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct Allocation {
    start: u32,
    end: u32,
    payload_end: u32,
    label: usize,
    kind: &'static str,
    first: u32,
    last: u32,
    shards: Vec<u32>,
}

#[derive(Serialize)]
struct Tile {
    logical: u16,
    physical: u16,
    allocations: Vec<Allocation>,
}

#[derive(Serialize)]
struct Profile {
    schema_version: u32,
    start: u32,
    end: u32,
    labels: Vec<String>,
    tiles: Vec<Tile>,
}

#[derive(Default)]
struct Labels {
    ids: BTreeMap<String, usize>,
    values: Vec<String>,
}

impl Labels {
    fn intern(&mut self, label: String) -> usize {
        if let Some(&id) = self.ids.get(&label) {
            return id;
        }
        let id = self.values.len();
        self.values.push(label.clone());
        self.ids.insert(label, id);
        id
    }
}

fn collect(
    program: &LowProgram,
    placement: &Placement,
    support: &TileMemoryMap,
    application: &Application,
) -> PackageBuildResult<Profile> {
    let analysis = analyze_allocations(program)?;
    let names = program
        .inputs
        .iter()
        .flat_map(|input| input.shards.iter().map(move |shard| (*shard, input)))
        .collect::<BTreeMap<_, _>>();
    let mut labels = Labels::default();
    let mut tiles = (0..application.tiles.len())
        .map(|logical| Tile {
            logical: logical as u16,
            physical: ipu_exchange::c600_logical_to_physical(logical as u16),
            allocations: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut members = vec![BTreeMap::new(); tiles.len()];
    for (&root, group) in &analysis.members {
        members[usize::from(program.shards[group[0]].tile)].insert(root, group.clone());
    }
    let mut iterated = vec![Vec::new(); tiles.len()];
    for group in &analysis.iterated {
        iterated[usize::from(group.tile)].push(group.clone());
    }
    for tile in &mut tiles {
        let index = usize::from(tile.logical);
        let grouped = iterated[index]
            .iter()
            .flat_map(|group| {
                group
                    .shards
                    .iter()
                    .map(|shard| analysis.root_of_member[shard.index() as usize])
            })
            .collect();
        let requests = allocation_requests(
            program,
            &iterated[index],
            &grouped,
            &members[index],
            &analysis.root_of_member,
            &analysis.root_requirements,
            &analysis.root_lifetimes,
            &analysis.conflicts,
        )?;
        for request in requests {
            let first_root = request.assignments[0].0;
            let base =
                placement.shard_addresses[&program.shards[members[index][&first_root][0]].id];
            let end = base
                + if base >= IPU21_INTERLEAVED_MEMORY_BASE {
                    request.region1_stride.map_or(request.bytes, |stride| {
                        stride * request.assignments.len() as u32
                    })
                } else {
                    request.bytes
                };
            for (i, &(root, _)) in request.assignments.iter().enumerate() {
                let group = &members[index][&root];
                let shard = &program.shards[group[0]];
                let start = placement.shard_addresses[&shard.id];
                let next = request.assignments.get(i + 1).map_or(end, |&(root, _)| {
                    placement.shard_addresses[&program.shards[members[index][&root][0]].id]
                });
                let bytes = allocation_bytes(program, group, analysis.root_requirements[&root])?;
                let payload = group
                    .iter()
                    .map(|&member| shard_storage_bytes(&program.shards[member]))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .max()
                    .unwrap_or(0);
                let input = group
                    .iter()
                    .find_map(|&member| names.get(&program.shards[member].id));
                let kind = match input.map(|input| input.kind) {
                    Some(crate::GraphInputKind::Parameter) => "parameter",
                    Some(crate::GraphInputKind::Host) => "activation",
                    None if matches!(shard.definition, ShardDefinition::Value(_)) => "activation",
                    None => "scratch",
                };
                let name = input.map_or_else(
                    || match shard.definition {
                        ShardDefinition::Value(value) => {
                            let value = &program.logical_values[value.index() as usize];
                            format!(
                                "value {} · origin {}",
                                value.id.index(),
                                value.origin.index()
                            )
                        }
                        _ => format!("{:?}", shard.definition),
                    },
                    |input| input.name.clone(),
                );
                let label = labels.intern(format!(
                    "{name}\n{:?} {:?} · {:?}",
                    shard.tensor_type.format.precision,
                    shard.tensor_type.shape.0,
                    shard.tensor_type.format.layout.memory_class
                ));
                tile.allocations.push(Allocation {
                    start,
                    end: start + bytes,
                    payload_end: start + payload,
                    label,
                    kind,
                    first: request.lifetime.first,
                    last: request.lifetime.last,
                    shards: group
                        .iter()
                        .map(|&i| program.shards[i].id.index())
                        .collect(),
                });
                if start + bytes < next {
                    tile.allocations.push(Allocation {
                        start: start + bytes,
                        end: next,
                        payload_end: start + bytes,
                        label: labels.intern("Repeat stride padding".into()),
                        kind: "padding",
                        first: request.lifetime.first,
                        last: request.lifetime.last,
                        shards: vec![],
                    });
                }
            }
        }
        for allocation in support.allocations() {
            let host_aperture = !program.requires_finite_scratch
                && (allocation.reserved.start, allocation.reserved.end) == HOST_SCRATCH_RANGE;
            for (first, last) in if host_aperture {
                vec![(0, 0), (u32::MAX, u32::MAX)]
            } else {
                vec![(0, u32::MAX)]
            } {
                tile.allocations.push(Allocation {
                    start: allocation.reserved.start,
                    end: allocation.reserved.end,
                    payload_end: allocation.range.end,
                    label: labels.intern(allocation.name.into()),
                    kind: "support",
                    first,
                    last,
                    shards: vec![],
                });
            }
        }
        // Host descriptors can occupy otherwise unused per-tile tensor space.
        // Record image segments outside global reservations as well.
        let mut occupied = crate::memory::merge_ranges(
            tile.allocations.iter().map(|a| (a.start, a.end)).collect(),
        );
        if let Some(image) = application
            .tiles
            .iter()
            .find(|image| image.physical_tile == u32::from(tile.physical))
        {
            for segment in &image.segments {
                for (start, end) in uncovered(
                    segment.address,
                    segment.address + segment.memory_size,
                    &occupied,
                ) {
                    tile.allocations.push(Allocation {
                        start,
                        end,
                        payload_end: end,
                        label: labels.intern("Per-tile image data / host descriptors".into()),
                        kind: "support",
                        first: 0,
                        last: u32::MAX,
                        shards: vec![],
                    });
                    occupied.push((start, end));
                }
                occupied = crate::memory::merge_ranges(occupied);
            }
        }
        // Display the loader-inaccessible SRAM tail explicitly.
        tile.allocations.push(Allocation {
            start: IPU21_APPLICATION_MEMORY_LIMIT,
            end: TILE_MEMORY_BASE + TILE_MEMORY_SIZE,
            payload_end: IPU21_APPLICATION_MEMORY_LIMIT,
            label: labels.intern("Outside application load range".into()),
            kind: "padding",
            first: 0,
            last: u32::MAX,
            shards: vec![],
        });
        tile.allocations.sort_by_key(|a| (a.first, a.start, a.last));
    }
    Ok(Profile {
        schema_version: 1,
        start: TILE_MEMORY_BASE,
        end: TILE_MEMORY_BASE + TILE_MEMORY_SIZE,
        labels: labels.values,
        tiles,
    })
}

fn uncovered(start: u32, end: u32, occupied: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut cursor = start;
    let mut result = Vec::new();
    for &(a, b) in occupied {
        if a >= end {
            break;
        }
        if b <= cursor {
            continue;
        }
        if cursor < a {
            result.push((cursor, a));
        }
        cursor = cursor.max(b);
    }
    if cursor < end {
        result.push((cursor, end));
    }
    result
}

pub(crate) fn write(
    directory: &Path,
    program: &LowProgram,
    placement: &Placement,
    support: &TileMemoryMap,
    application: &Application,
) -> PackageBuildResult<()> {
    let profile = collect(program, placement, support, application)?;
    let json =
        serde_json::to_string(&profile).map_err(|e| PackageBuildError::Invalid(e.to_string()))?;
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("placement-{}", std::process::id()));
    std::fs::write(path.with_extension("json"), &json)?;
    std::fs::write(
        path.with_extension("html"),
        include_str!("profile.html").replace("__PROFILE_JSON__", &json.replace('<', "\\u003c")),
    )?;
    tracing::info!(path = %path.with_extension("html").display(), "wrote exact tile placement profile");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeGraph, Ipu21CostModel, PipelineConfig};

    #[test]
    fn image_gaps_do_not_duplicate_tensor_or_support_allocations() {
        assert_eq!(
            uncovered(10, 50, &[(0, 12), (20, 30), (35, 40), (48, 60)]),
            vec![(12, 20), (30, 35), (40, 48)]
        );
        assert!(uncovered(20, 30, &[(10, 40)]).is_empty());
    }
    #[test]
    fn report_covers_placed_aliases_and_reused_requests_exactly() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("input</script>", [8, 64]).unwrap();
        let weights = (0..3)
            .map(|i| graph.parameter(format!("weights.{i}"), [64, 64]).unwrap())
            .collect::<Vec<_>>();
        let sequence = graph.value_sequence("weights", weights).unwrap();
        let y = graph
            .repeat(3, [x], [], [sequence], |body, args| {
                let product = body.gemm(args.carried[0], args.iterated[0])?;
                Ok(vec![body.add(product, args.carried[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([y]).unwrap();
        let mut config = PipelineConfig::new(8);
        for input in graph.inputs() {
            config.inputs.insert(
                input.value,
                crate::TensorFormat {
                    precision: crate::Precision::F16,
                    layout: if input.kind == crate::GraphInputKind::Host {
                        crate::Layout::amp_left(64, 8)
                    } else {
                        crate::Layout::block_major_matrix(64, 8)
                    },
                },
            );
        }
        let mid = crate::lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = crate::lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
        let placement = place(&low).unwrap();
        let application = Application {
            tiles: (0..8)
                .map(|tile| ipu_package::TileImage {
                    physical_tile: u32::from(ipu_exchange::c600_logical_to_physical(tile)),
                    entry_point: 0,
                    command_address: 0,
                    diagnostic_address: 0,
                    segments: vec![],
                })
                .collect(),
            ..Default::default()
        };
        let mut support = TileMemoryMap::new();
        support
            .reserve(
                "host exchange aperture",
                HOST_SCRATCH_RANGE.0..HOST_SCRATCH_RANGE.1,
            )
            .unwrap();
        let report = collect(&low, &placement, &support, &application).unwrap();
        for input in low
            .inputs
            .iter()
            .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        {
            for shard in &input.shards {
                let tile = &report.tiles[usize::from(low.shards[shard.index() as usize].tile)];
                let allocation = tile
                    .allocations
                    .iter()
                    .find(|a| a.shards.contains(&shard.index()))
                    .unwrap();
                assert_eq!((allocation.first, allocation.last), (0, u32::MAX));
                assert!(allocation.start >= HOST_SCRATCH_RANGE.1);
                // This must also exclude deliberate writable aliases of parameters.
                for run in &low.kernel_runs {
                    assert!(
                        run.outputs()
                            .all(|out| !allocation.shards.contains(&out.shard.index()))
                    );
                }
            }
        }
        for (&shard, &address) in &placement.shard_addresses {
            let tile = usize::from(low.shards[shard.index() as usize].tile);
            let matches = report.tiles[tile]
                .allocations
                .iter()
                .filter(|a| a.shards.contains(&shard.index()))
                .collect::<Vec<_>>();
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].start, address);
            assert!(
                matches[0].end
                    >= address + shard_storage_bytes(&low.shards[shard.index() as usize]).unwrap()
            );
        }
        for tile in &report.tiles {
            for (i, a) in tile.allocations.iter().enumerate() {
                assert!(a.start <= a.payload_end && a.payload_end <= a.end);
                for b in &tile.allocations[i + 1..] {
                    if a.start < b.end && b.start < a.end {
                        assert!(
                            a.last < b.first || b.last < a.first,
                            "overlapping live allocations"
                        );
                    }
                }
            }
        }
    }
}
