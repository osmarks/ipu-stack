//! Opt-in exact placement diagnostics, using the allocator's own requests.
use super::*;
use crate::memory::TileMemoryMap;
use crate::package::{PackageBuildError, PackageBuildResult};
use ipu_package::{Application, TILE_MEMORY_BASE, TILE_MEMORY_SIZE};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct Allocation {
    start: u32,
    end: u32,
    payload_end: u32,
    label: usize,
    kind: String,
    first: u32,
    last: u32,
    shards: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct Tile {
    logical: u16,
    physical: u16,
    allocations: Vec<Allocation>,
}

#[derive(Serialize, Deserialize)]
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
    let mut analysis = analyze_allocations(program)?;
    analysis
        .tiles
        .resize_with(application.tiles.len(), TileAllocations::default);
    let names = program
        .inputs
        .iter()
        .flat_map(|input| {
            program
                .value_shards(input.value)
                .iter()
                .map(move |shard| (*shard, input))
        })
        .collect::<BTreeMap<_, _>>();
    let mut labels = Labels::default();
    let mut tiles = (0..application.tiles.len())
        .map(|logical| Tile {
            logical: logical as u16,
            physical: ipu_exchange::c600_logical_to_physical(logical as u16),
            allocations: Vec::new(),
        })
        .collect::<Vec<_>>();
    for tile in &mut tiles {
        let members = &analysis.tiles[usize::from(tile.logical)].members;
        let requests = allocation_requests(program, &analysis, tile.logical)?;
        let address = |root| {
            let first = members[&root][0];
            placement.shard_addresses[&program.shards[first].id] - analysis.member_offsets[first]
        };
        for request in requests {
            let first_root = request.assignments[0].0;
            let base = address(first_root);
            let end = base
                + if base >= IPU21_INTERLEAVED_MEMORY_BASE {
                    request.region1_stride.map_or(request.bytes, |stride| {
                        stride * request.assignments.len() as u32
                    })
                } else {
                    request.bytes
                };
            for (i, &(root, _)) in request.assignments.iter().enumerate() {
                let group = &members[&root];
                let shard = &program.shards[group[0]];
                let start = address(root);
                let next = request
                    .assignments
                    .get(i + 1)
                    .map_or(end, |&(root, _)| address(root));
                let requirement = analysis.root_requirements[&root];
                let bytes =
                    allocation_bytes(program, group, &analysis.member_offsets, requirement)?;
                let payload = allocation_bytes(
                    program,
                    group,
                    &analysis.member_offsets,
                    Requirement::default(),
                )?;
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
                    kind: kind.into(),
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
                        kind: "padding".into(),
                        first: request.lifetime.first,
                        last: request.lifetime.last,
                        shards: vec![],
                    });
                }
            }
        }
        for allocation in placement
            .auxiliary_allocations
            .get(usize::from(tile.logical))
            .into_iter()
            .flatten()
        {
            tile.allocations.push(Allocation {
                start: allocation.address,
                end: allocation.address + allocation.bytes,
                payload_end: allocation.address + allocation.bytes,
                label: labels.intern(allocation.name.clone()),
                kind: "support".into(),
                first: allocation.first,
                last: allocation.last,
                shards: vec![],
            });
        }
        for allocation in support.allocations() {
            let host_aperture = !program.requires_finite_scratch
                && (allocation.reserved.start, allocation.reserved.end) == HOST_SCRATCH_RANGE;
            for &(first, last) in if host_aperture {
                &[(0, 0), (u32::MAX, u32::MAX)][..]
            } else {
                &[(0, u32::MAX)][..]
            } {
                tile.allocations.push(Allocation {
                    start: allocation.reserved.start,
                    end: allocation.reserved.end,
                    payload_end: allocation.range.end,
                    label: labels.intern(allocation.name.into()),
                    kind: "support".into(),
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
                for (start, end) in crate::memory::uncovered_ranges(
                    segment.address,
                    segment.address + segment.memory_size,
                    &occupied,
                ) {
                    tile.allocations.push(Allocation {
                        start,
                        end,
                        payload_end: end,
                        label: labels.intern("Per-tile image data / host descriptors".into()),
                        kind: "support".into(),
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
            kind: "padding".into(),
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

pub(crate) fn write(
    directory: &Path,
    program: &LowProgram,
    placement: &Placement,
    support: &TileMemoryMap,
    application: &Application,
) -> PackageBuildResult<()> {
    let profile = collect(program, placement, support, application)?;
    std::fs::create_dir_all(directory)?;
    let path = directory.join(format!("placement-{}", std::process::id()));
    write_json(&path.with_extension("json"), &profile)?;
    write_html(&profile, &path.with_extension("html"))?;
    tracing::info!(path = %path.with_extension("html").display(), "wrote exact tile placement profile");
    Ok(())
}

const TILES_PER_CHUNK: usize = 16;

fn write_json(path: &Path, value: &impl Serialize) -> PackageBuildResult<()> {
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer(&mut writer, value)
        .map_err(|e| PackageBuildError::Invalid(e.to_string()))?;
    writer.flush()?;
    Ok(())
}

/// Render an existing exact placement JSON report without rebuilding a package.
/// The HTML loads visible tiles from an adjacent `.data` directory over HTTP.
pub fn render_memory_profile(input: impl std::io::Read, output: &Path) -> PackageBuildResult<()> {
    let profile: Profile = serde_json::from_reader(std::io::BufReader::new(input))
        .map_err(|e| PackageBuildError::Invalid(e.to_string()))?;
    if profile.schema_version != 1 || profile.start >= profile.end {
        return Err(PackageBuildError::Invalid(
            "unsupported memory profile".into(),
        ));
    }
    write_html(&profile, output)
}

// First-use order puts later occupants below earlier allocations. Preserve this
// geometry in the index so unloaded tiles occupy their exact scroll positions.
fn allocation_lanes(tile: &Tile) -> Vec<Vec<&Allocation>> {
    let mut lanes: Vec<Vec<&Allocation>> = vec![Vec::new()];
    for allocation in &tile.allocations {
        let slot = lanes.iter_mut().find_map(|lane| {
            let i = lane.partition_point(|a| a.start < allocation.start);
            ((i == 0 || lane[i - 1].end <= allocation.start)
                && (i == lane.len() || allocation.end <= lane[i].start))
                .then_some((lane, i))
        });
        if let Some((lane, i)) = slot {
            lane.insert(i, allocation);
        } else {
            lanes.push(vec![allocation]);
        }
    }
    lanes
}

fn write_html(profile: &Profile, output: &Path) -> PackageBuildResult<()> {
    let directory = output.with_extension("data");
    std::fs::create_dir_all(&directory)?;
    let mut tiles = Vec::new();
    for (chunk_id, chunk) in profile.tiles.chunks(TILES_PER_CHUNK).enumerate() {
        let lanes = chunk.iter().map(allocation_lanes).collect::<Vec<_>>();
        for (tile, lanes) in chunk.iter().zip(&lanes) {
            tiles.push(serde_json::json!({
                "logical": tile.logical, "physical": tile.physical,
                "lane_count": lanes.len(), "allocation_count": tile.allocations.len(),
                "reused_count": tile.allocations.len() - lanes[0].len(),
            }));
        }
        write_json(&directory.join(format!("tiles-{chunk_id}.json")), &lanes)?;
    }
    let index = serde_json::json!({
        "start": profile.start, "end": profile.end, "labels": profile.labels,
        "tiles": tiles, "chunk_size": TILES_PER_CHUNK,
        "directory": directory.file_name().unwrap().to_string_lossy(),
    });
    std::fs::write(
        output,
        include_str!("profile.html").replace(
            "__PROFILE_JSON__",
            &index.to_string().replace('<', "\\u003c"),
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeGraph, Ipu21CostModel, PipelineConfig};

    #[test]
    fn rendering_splits_tiles_and_retains_reuse_geometry() {
        let directory = std::env::temp_dir().join(format!(
            "ipu-memory-render-{}-{}",
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let allocation = serde_json::json!({
            "start": 100, "end": 120, "payload_end": 116,
            "label": 0, "kind": "activation", "first": 0, "last": 1, "shards": [5],
        });
        let mut reused = allocation.clone();
        reused["first"] = 2.into();
        reused["last"] = 3.into();
        let input = serde_json::json!({
            "schema_version": 1, "start": 100, "end": 200,
            "labels": ["input</script>"],
            "tiles": (0..17).map(|tile| serde_json::json!({
                "logical": tile, "physical": 16-tile,
                "allocations": [allocation, reused],
            })).collect::<Vec<_>>(),
        })
        .to_string();
        let output = directory.join("memory report.html");
        render_memory_profile(input.as_bytes(), &output).unwrap();
        let html = std::fs::read_to_string(&output).unwrap();
        assert!(html.contains("input\\u003c/script>"));
        assert!(!html.contains("\"shards\""));
        let chunk: Vec<Vec<Vec<Allocation>>> = serde_json::from_reader(
            std::fs::File::open(directory.join("memory report.data/tiles-1.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(chunk.len(), 1);
        assert_eq!(chunk[0].len(), 2);
        assert_eq!(chunk[0][0][0].first, 0);
        assert_eq!(chunk[0][1][0].first, 2);
        assert_eq!(chunk[0][1][0].shards, [5]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn image_gaps_do_not_duplicate_tensor_or_support_allocations() {
        assert_eq!(
            crate::memory::uncovered_ranges(10, 50, &[(0, 12), (20, 30), (35, 40), (48, 60)]),
            vec![(12, 20), (30, 35), (40, 48)]
        );
        assert!(crate::memory::uncovered_ranges(20, 30, &[(10, 40)]).is_empty());
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
            for shard in low.value_shards(input.value) {
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
            let lanes = allocation_lanes(tile);
            assert_eq!(
                lanes.iter().map(Vec::len).sum::<usize>(),
                tile.allocations.len()
            );
            for lane in lanes {
                assert!(lane.windows(2).all(|pair| pair[0].end <= pair[1].start));
            }
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
