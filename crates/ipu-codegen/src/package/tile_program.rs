//! Package already resolved tile programs for low-level diagnostics.
use super::*;

/// Builds an application from address-resolved tile programs.
///
/// This is the low-level counterpart to [`build_package`]. It deliberately has
/// no tensor bindings: callers supply initialized tile data and inspect it
/// through driver diagnostics. A zero-payload `run` rendezvous starts execution
/// after loading, so breakpoints in the program cannot race the loader.
/// Initial values in the host aperture are staged and copied in after this
/// rendezvous, before the supplied device steps execute.
pub fn build_tile_program_package(
    programs: &[TileProgram],
    data: &[TileProgramData],
    outputs: &[Binding],
    toolchain: &Toolchain,
    runtime_source: &std::path::Path,
) -> PackageBuildResult<Application> {
    let topology = Topology::c600();
    let execution_tiles = u16::try_from(topology.tile_count())?;
    if programs.len() != usize::from(execution_tiles)
        || programs
            .iter()
            .enumerate()
            .any(|(tile, program)| usize::from(program.tile) != tile)
    {
        return Err(invalid(
            "finalized tile programs must cover every C600 logical tile in order",
        ));
    }
    if data
        .iter()
        .any(|segment| segment.tile >= execution_tiles || segment.data.is_empty())
    {
        return Err(invalid(
            "tile-program data has an invalid tile or empty payload",
        ));
    }

    let mut programs = programs.to_vec();
    let (mut data, aperture) = split_aperture_data(data)?;
    let runtime_artifact = toolchain.compile(runtime_source, "static_runtime", &[])?;
    let objects = vec![fs::read(runtime_artifact.object)?];
    let kernels = KernelBuildPlan::default();
    let mut retained_runtime = vec![
        COMPLETE_SYMBOL.into(),
        HOST_RUN_SYMBOL.into(),
        REPEAT_CALL_SYMBOL.into(),
        WORKER_BARRIER_SYMBOL.into(),
    ];
    for program in &programs {
        collect_compute_symbols(&mut retained_runtime, &program.steps);
    }
    if !aperture.is_empty() {
        retained_runtime.push(crate::COPY_U32_SYMBOL.into());
    }
    retained_runtime.sort_unstable();
    retained_runtime.dedup();
    let layout = link_runtime(
        &objects,
        runtime_symbols(0, 0, 0)?,
        &kernels,
        &retained_runtime,
    )?;
    let symbols = layout
        .symbols
        .clone()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let mut memory = TileMemoryMap::new();
    reserve_linked_image(&mut memory, &layout, "linked runtime")?;
    // Explicit tile programs bring fixed data addresses; protect linked code
    // before admitting those externally supplied ranges.
    protect_executable_elements(
        &mut memory,
        layout
            .segments
            .iter()
            .map(|segment| segment.address..segment.address + segment.size as u32),
    )?;
    memory.reserve(
        "host exchange aperture",
        ipu_exchange::EXCHANGE_WINDOW_BASE
            ..ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
    )?;
    memory.reserve("runtime state", RUNTIME_STATE_BASE..crate::IPU21_DATA_BASE)?;
    let mut tile_data = vec![Vec::<(u32, u32)>::new(); usize::from(execution_tiles)];
    for segment in &data {
        let bytes = u32::try_from(segment.data.len())?;
        let end = segment
            .address
            .checked_add(bytes)
            .ok_or_else(|| invalid("tile data range overflow"))?;
        tile_data[usize::from(segment.tile)].push((segment.address, end));
    }
    let mut tile_rows = vec![Vec::<(u32, u32)>::new(); usize::from(execution_tiles)];
    for program in &programs {
        let mut rows = BTreeMap::new();
        collect_exchange_rows(&mut rows, &program.steps)?;
        tile_rows[usize::from(program.tile)].extend(rows);
    }
    for tile in 0..execution_tiles {
        for &(data_start, data_end) in &tile_data[usize::from(tile)] {
            if let Some(&(row_start, row_end)) = tile_rows[usize::from(tile)]
                .iter()
                .find(|&&(row_start, row_end)| data_start < row_end && row_start < data_end)
            {
                return Err(invalid(format!(
                    "tile {tile} data at 0x{data_start:x}..0x{data_end:x} overlaps exchange row 0x{row_start:x}..0x{row_end:x}"
                )));
            }
        }
    }
    // The host handshake overwrites its aperture. Stage requested initial
    // contents elsewhere and copy them in at entry to the device program.
    // Stage independently per tile; a common hole is not required. Region 1
    // cannot hold executable code and its reservations precede host metadata.
    for mut segment in aperture {
        let tile = usize::from(segment.tile);
        let mut scratch = TileMemoryMap::new();
        for (start, end) in crate::memory::merge_ranges(
            tile_data[tile]
                .iter()
                .chain(&tile_rows[tile])
                .copied()
                .collect(),
        ) {
            scratch.reserve("replay data and rows", start..end)?;
        }
        let bytes = u32::try_from(segment.data.len())?;
        let staging = scratch
            .allocate(MemoryRequest {
                name: "host aperture initial contents",
                bytes,
                alignment: 4,
                bounds: ipu_package::IPU21_INTERLEAVED_MEMORY_BASE
                    ..ipu_package::IPU21_APPLICATION_MEMORY_LIMIT,
                end_alignment: 4,
                guard_after: 0,
            })?
            .range;
        programs[tile].steps.insert(
            0,
            crate::TileStep::Compute(crate::ComputeStep {
                symbol: crate::COPY_U32_SYMBOL.into(),
                output_address: crate::TileAddress::Absolute(segment.address),
                input_addresses: vec![crate::TileAddress::Absolute(staging.start)],
                arguments: vec![bytes / 4],
                profile: Default::default(),
            }),
        );
        segment.address = staging.start;
        tile_data[tile].push((staging.start, staging.end));
        data.push(segment);
    }
    // Generated and linked code use common addresses on every tile, so choose
    // them against the union of tile-local data and row ranges. Data on one
    // tile may otherwise legally share an address with a row on another tile.
    let tile_local_ranges = tile_data
        .into_iter()
        .chain(tile_rows)
        .flatten()
        .collect::<Vec<_>>();
    let tile_local_ranges = crate::memory::merge_ranges(tile_local_ranges);
    for &(start, end) in &tile_local_ranges {
        memory.reserve("tile-local data or exchange rows", start..end)?;
    }
    for (start, end) in tile_local_ranges {
        // These externally fixed ranges precede code placement. Exclude their
        // entire executable elements: instruction fetch conflicts with data
        // access even when the byte ranges do not overlap.
        let element = ipu_package::TILE_MEMORY_ELEMENT_SIZE;
        let executable_end = end.min(ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT);
        if start < executable_end {
            for (free_start, free_end) in memory
                .free_ranges(start / element * element..executable_end.div_ceil(element) * element)
            {
                memory.reserve("tile data memory elements", free_start..free_end)?;
            }
        }
    }

    let launch = Binding {
        name: "run-gate".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: u32::from(topology.physical(0)?),
            tile_address: COMPLETION_ADDRESS + 4,
            file_offset: 0,
            size: 4,
        }],
    };
    let finish = Binding {
        name: "run-finish".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: u32::from(topology.physical(0)?),
            tile_address: COMPLETION_ADDRESS + 8,
            file_offset: 0,
            size: 4,
        }],
    };
    let mut run_outputs = outputs.to_vec();
    run_outputs.push(finish);
    let host_bounds = crate::IPU21_DATA_BASE..ipu_package::IPU21_APPLICATION_MEMORY_LIMIT;
    let sizing_host_base = memory.next_free(
        RUNTIME_EXECUTABLE_START,
        RUNTIME_EXECUTABLE_START..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        8,
        "host programs",
    )?;
    let provisional_ranges = memory.free_ranges(host_bounds.clone());
    let provisional_host = host::plan(
        &[],
        std::slice::from_ref(&launch),
        &run_outputs,
        execution_tiles,
        sizing_host_base,
        &vec![provisional_ranges; usize::from(execution_tiles)],
    )?;
    let host_code_bytes = provisional_host
        .end
        .checked_sub(sizing_host_base)
        .ok_or_else(|| invalid("host program size underflow"))?;
    let host_code = allocate_package_code(&mut memory, "host programs", host_code_bytes, 8, 0)?;
    protect_executable_elements(&mut memory, [host_code.range.clone()])?;
    let host_ranges = memory.free_ranges(host_bounds.clone());
    let host = host::plan(
        &[],
        std::slice::from_ref(&launch),
        &run_outputs,
        execution_tiles,
        host_code.range.start,
        &vec![host_ranges; usize::from(execution_tiles)],
    )?;
    if host.end - host_code.range.start > host_code_bytes {
        return Err(invalid("host program grew after placement"));
    }
    let host_data_ranges = host
        .segments
        .iter()
        .flatten()
        .filter(|segment| segment.flags & SEGMENT_EXECUTE == 0)
        .map(|segment| (segment.address, segment.address + segment.memory_size))
        .collect::<Vec<_>>();
    for (start, end) in crate::memory::merge_ranges(host_data_ranges) {
        memory.reserve("host program data", start..end)?;
    }

    let sizing_address = memory.next_free(
        RUNTIME_EXECUTABLE_START,
        RUNTIME_EXECUTABLE_START..ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT,
        8,
        "generated tile programs",
    )?;
    let maximum_bytes = programs.iter().try_fold(0u32, |maximum, program| {
        let physical = topology.physical(program.tile)?;
        let generated = emit(
            program,
            &symbols,
            &host.programs[usize::from(physical)],
            &CodegenOptions {
                code_address: sizing_address,
                ..CodegenOptions::default()
            },
        )?;
        Ok::<_, PackageBuildError>(maximum.max(u32::try_from(generated.bytes.len())?))
    })?;
    let code_address = allocate_package_code(
        &mut memory,
        "generated tile programs",
        maximum_bytes,
        ipu_package::TILE_MEMORY_ELEMENT_SIZE,
        0,
    )?
    .range
    .start;
    let generated = programs
        .iter()
        .map(|program| {
            let physical = topology.physical(program.tile)?;
            Ok(emit(
                program,
                &symbols,
                &host.programs[usize::from(physical)],
                &CodegenOptions {
                    code_address,
                    ..CodegenOptions::default()
                },
            )?)
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;

    let mut segments = vec![Vec::new(); usize::from(execution_tiles)];
    for segment in &data {
        let physical = topology.physical(segment.tile)?;
        segments[usize::from(physical)].push(Segment {
            address: segment.address,
            memory_size: u32::try_from(segment.data.len())?,
            data: segment.data.clone(),
            flags: SEGMENT_READ | SEGMENT_WRITE,
        });
    }
    for (physical, host_segments) in host.segments.iter().enumerate() {
        segments[physical].extend(host_segments.iter().cloned());
    }
    let context = TileBuildContext {
        objects: &objects,
        kernel_plan: &kernels,
        retained_runtime: &retained_runtime,
        code_address,
        host_staging_address: host.staging_address,
    };
    let mut tiles = Vec::with_capacity(usize::from(execution_tiles));
    for logical in 0..execution_tiles {
        let physical = topology.physical(logical)?;
        tiles.push(build_tile(
            u32::from(physical),
            u32::from(logical),
            &generated[usize::from(logical)],
            &segments[usize::from(physical)],
            &context,
        )?);
    }
    tiles.sort_unstable_by_key(|tile| tile.physical_tile);
    let mut application = Application {
        tiles,
        ..Application::default()
    };
    add_linked_debug_map(&mut application, &layout)?;
    for (logical, program) in generated.iter().enumerate() {
        let physical = u32::from(topology.physical(u16::try_from(logical)?)?);
        add_generated_debug_map(&mut application, physical, code_address, program)?;
    }
    application.outputs.push(Binding {
        name: "completion".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: 0,
            tile_address: COMPLETION_ADDRESS,
            file_offset: 0,
            size: 4,
        }],
    });
    application.outputs.extend(run_outputs);
    application.inputs.push(launch);
    application.entry_points.push(EntryPoint {
        name: "run".into(),
        command: 0,
        external_syncs: 0,
    });
    application.host_exchange = host.protocol;
    application.validate()?;
    Ok(application)
}

fn collect_exchange_rows(
    rows: &mut BTreeMap<u32, u32>,
    steps: &[crate::TileStep],
) -> PackageBuildResult<()> {
    for step in steps {
        match step {
            crate::TileStep::Exchange(exchange) => {
                let bytes = u32::try_from(exchange.program.words.len())?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("exchange row size overflow"))?;
                let end = exchange
                    .program
                    .address
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("exchange row range overflow"))?;
                rows.entry(exchange.program.address)
                    .and_modify(|existing| *existing = (*existing).max(end))
                    .or_insert(end);
            }
            crate::TileStep::Repeat(repeat) => collect_exchange_rows(rows, &repeat.body)?,
            crate::TileStep::Compute(_) | crate::TileStep::Checkpoint(_) => {}
        }
    }
    Ok(())
}

fn collect_compute_symbols(symbols: &mut Vec<String>, steps: &[crate::TileStep]) {
    for step in steps {
        let profile = match step {
            crate::TileStep::Compute(compute) => {
                symbols.push(compute.symbol.clone());
                &compute.profile
            }
            crate::TileStep::Repeat(repeat) => {
                collect_compute_symbols(symbols, &repeat.body);
                &repeat.profile
            }
            crate::TileStep::Exchange(exchange) => &exchange.profile,
            crate::TileStep::Checkpoint(checkpoint) => &checkpoint.profile,
        };
        if profile.before.is_some() || profile.after.is_some() {
            symbols.push(SAMPLE_CYCLE_SYMBOL.into());
        }
    }
}

/// Split loader data from initial values that must be installed after the host
/// handshake. Coalesce aperture fragments per tile, preserving byte alignment
/// and insertion order; the runtime helper copies complete words.
fn split_aperture_data(
    data: &[crate::TileProgramData],
) -> PackageBuildResult<(Vec<crate::TileProgramData>, Vec<crate::TileProgramData>)> {
    let start = ipu_exchange::EXCHANGE_WINDOW_BASE;
    let end = start + ipu_exchange::EXCHANGE_WINDOW_BYTES;
    let mut loader = Vec::new();
    let mut aperture = BTreeMap::<u16, Vec<crate::TileProgramData>>::new();
    for segment in data {
        let limit = segment
            .address
            .checked_add(u32::try_from(segment.data.len())?)
            .ok_or_else(|| invalid("tile data range overflow"))?;
        for (from, to, deferred) in [
            (segment.address, limit.min(start), false),
            (segment.address.max(start), limit.min(end), true),
            (segment.address.max(end), limit, false),
        ] {
            if from >= to {
                continue;
            }
            let part = crate::TileProgramData {
                tile: segment.tile,
                address: from,
                data: segment.data
                    [(from - segment.address) as usize..(to - segment.address) as usize]
                    .to_vec(),
            };
            if deferred {
                aperture.entry(segment.tile).or_default().push(part);
            } else {
                loader.push(part);
            }
        }
    }
    let aperture = aperture
        .into_iter()
        .map(|(tile, parts)| {
            let start = parts.iter().map(|p| p.address).min().unwrap() & !3;
            let end = (parts
                .iter()
                .map(|p| p.address + p.data.len() as u32)
                .max()
                .unwrap()
                + 3)
                & !3;
            let mut data = vec![0; (end - start) as usize];
            for part in parts {
                let offset = (part.address - start) as usize;
                data[offset..offset + part.data.len()].copy_from_slice(&part.data);
            }
            crate::TileProgramData {
                tile,
                address: start,
                data,
            }
        })
        .collect();
    Ok((loader, aperture))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aperture_initial_values_are_word_aligned_and_kept_out_of_loader_data() {
        let start = ipu_exchange::EXCHANGE_WINDOW_BASE;
        let end = start + ipu_exchange::EXCHANGE_WINDOW_BYTES;
        let input = vec![
            crate::TileProgramData {
                tile: 0,
                address: start - 2,
                data: vec![1, 2, 3, 4, 5],
            },
            crate::TileProgramData {
                tile: 0,
                address: start + 6,
                data: vec![6],
            },
            crate::TileProgramData {
                tile: 1,
                address: end - 1,
                data: vec![7, 8],
            },
        ];
        let (loader, staged) = split_aperture_data(&input).unwrap();
        assert_eq!(loader.len(), 2);
        assert_eq!(loader[0].address, start - 2);
        assert_eq!(loader[0].data, [1, 2]);
        assert_eq!(loader[1].address, end);
        assert_eq!(loader[1].data, [8]);
        assert_eq!(staged.len(), 2);
        assert_eq!(staged[0].address, start);
        assert_eq!(staged[0].data, [3, 4, 5, 0, 0, 0, 6, 0]);
        assert_eq!(staged[1].address, end - 4);
        assert_eq!(staged[1].data, [0, 0, 0, 7]);
    }
}
