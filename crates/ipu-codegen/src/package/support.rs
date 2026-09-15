//! Linked code and concrete package reservations. Sizing consumes provisional
//! addresses only to run the real emitters; it neither places tensors nor schedules.
use super::bindings::PackageBindings;
use super::profile::{self, instrument_profile, profile_step_count};
use super::{
    PackageBuildError, PackageBuildResult, RUNTIME_EXECUTABLE_START, active_topology,
    allocate_package_code, check_exchange_budget, invalid, link_runtime, linked_end,
    protect_executable_elements, reserve_exchange_setup, reserve_fixed_runtime_memory,
    reserve_linked_image, runtime_retained_symbols,
};
use crate::host;
use crate::kernel::KernelObjects;
use crate::low::LowProgram;
use crate::memory::{
    MemoryAllocation, MemoryRequest, PROFILE_END_CYCLE, PROFILE_START_CYCLE, TileMemoryMap,
};
use crate::{CodegenOptions, PipelineConfig, TileProgramLowering, emit};
use ipu_elf::LinkedImage;
use ipu_target::ipu21::fabric::Topology;
use ipu_target::ipu21::memory::IPU21_DATA_BASE;
use rayon::prelude::*;

/// Reservations measured with the same host/tile emitters used for final output.
/// Writable auxiliaries participate in tensor placement through `profile_requests`;
/// code, rows and host descriptors define its remaining `available_ranges`.
pub(crate) struct PackageSupport {
    pub(crate) memory: TileMemoryMap,
    pub(crate) available_ranges: Vec<(u32, u32)>,
    pub(crate) profile_requests: Vec<Vec<crate::place::AuxiliaryRequest>>,
    pub(crate) exchange_code_base: u32,
    pub(super) objects: Vec<Vec<u8>>,
    pub(super) kernel_plan: KernelObjects,
    pub(super) retained_runtime: Vec<&'static str>,
    pub(super) layout: LinkedImage,
    pub(super) physical_to_logical: Vec<u16>,
    pub(super) code_address: u32,
    pub(super) generated_code_bytes: u32,
    pub(super) host_code_base: u32,
    pub(super) host_code_bytes: u32,
    pub(super) host_data: Option<MemoryAllocation>,
    pub(super) exchange_rows: Option<MemoryAllocation>,
}

pub(crate) fn size_support(
    program: &LowProgram,
    provisional_placement: &crate::Placement,
    provisional_exchanges: &[crate::PhysicalExchangePhase],
    config: &PipelineConfig,
    objects: Vec<Vec<u8>>,
    kernel_plan: KernelObjects,
    invocations: u32,
) -> PackageBuildResult<PackageSupport> {
    let topology = active_topology(program.tile_count)?;
    let retained_runtime = runtime_retained_symbols(program, config);
    let layout = tracing::info_span!("link_runtime").in_scope(|| -> PackageBuildResult<_> {
        link_runtime(&objects, 0, 0, 0, &kernel_plan, &retained_runtime)
    })?;
    let linked_end = linked_end(&layout)?;
    let mut memory = TileMemoryMap::new();
    reserve_linked_image(&mut memory, &layout, "linked runtime and kernels")?;
    reserve_fixed_runtime_memory(&mut memory)?;

    let execution_tile_count = u16::try_from(Topology::c600().tile_count())?;
    let exchange_table_bytes = crate::tile::compact_exchange_table_bytes(
        provisional_exchanges,
        execution_tile_count,
        program.tile_count,
    )?;
    let mut repeat_bytes = vec![0usize; usize::from(program.tile_count)];
    let mut arithmetic_savings = repeat_bytes.clone();
    for phase in provisional_exchanges {
        for (tile, patches) in phase.repeat_patches.iter().enumerate() {
            for patch in patches {
                let words = &patch.values;
                repeat_bytes[tile] += 4 * words.len();
                if crate::arithmetic_progression(words).is_some() {
                    arithmetic_savings[tile] += 4 * words.len();
                }
            }
        }
    }
    tracing::info!(
        exchange_table_bytes,
        maximum_uncompressed_repeat_patch_bytes = repeat_bytes.iter().max().copied().unwrap_or(0),
        maximum_elided_arithmetic_patch_bytes =
            arithmetic_savings.iter().max().copied().unwrap_or(0),
        "exchange table and repeat patch storage"
    );
    check_exchange_budget(u64::from(exchange_table_bytes), config)?;
    let profile_requests = if config.profiling {
        (0..execution_tile_count)
            .map(|logical| {
                let steps = if logical < program.tile_count {
                    profile_step_count(program, &program.tiles[usize::from(logical)])
                } else {
                    profile::inactive_profile_work(program).len()
                };
                let bytes = u32::try_from(steps + 1)?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("profile storage size overflow"))?;
                Ok(vec![crate::place::AuxiliaryRequest {
                    name: "cycle profile samples".into(),
                    bytes,
                    alignment: 4,
                    first: 0,
                    last: u32::MAX,
                }])
            })
            .collect::<PackageBuildResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    // Addresses do not affect instruction sizing. Final emission and host
    // bindings use the auxiliary allocations chosen alongside tensors.
    let provisional_profile_addresses = config.profiling.then(|| {
        vec![
            ipu_target::ipu21::memory::IPU21_INTERLEAVED_MEMORY_BASE;
            usize::from(execution_tile_count)
        ]
    });
    let execution_topology = Topology::c600();
    let mut physical_to_logical = vec![None; usize::from(execution_tile_count)];
    for logical in 0..execution_tile_count {
        let physical = execution_topology.physical(logical)?;
        physical_to_logical[usize::from(physical)] = Some(logical);
    }
    let physical_to_logical = physical_to_logical
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("execution topology does not cover every physical tile"))?;
    let provisional_bindings = PackageBindings::new(
        program,
        provisional_placement,
        &topology,
        &physical_to_logical,
        provisional_profile_addresses.as_deref(),
    )?;
    let sizing_host_base = memory.next_free(
        RUNTIME_EXECUTABLE_START,
        RUNTIME_EXECUTABLE_START..ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "host programs",
    )?;
    // Sizing does not emit these descriptor addresses. Do not depend on holes
    // left by provisional tensor placement to discover the required reservation.
    let provisional_auxiliary_ranges = vec![
        vec![(
            IPU21_DATA_BASE,
            ipu_target::ipu21::loader_abi::APPLICATION_LOAD_LIMIT,
        )];
        usize::from(execution_tile_count)
    ];
    let provisional_host = host::plan(
        &provisional_bindings.weights,
        &provisional_bindings.inputs,
        &provisional_bindings.outputs,
        execution_tile_count,
        sizing_host_base,
        &provisional_auxiliary_ranges,
    )?;
    let host_code_bytes = provisional_host
        .end
        .checked_sub(sizing_host_base)
        .ok_or_else(|| invalid("host program size underflow"))?;
    let host_code = (host_code_bytes != 0)
        .then(|| {
            allocate_package_code(
                &mut memory,
                "host programs",
                host_code_bytes,
                8,
                ipu_target::ipu21::memory::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
            )
        })
        .transpose()?;
    let host_code_base = host_code
        .as_ref()
        .map_or(sizing_host_base, |code| code.range.start);
    let provisional_host = host::plan(
        &provisional_bindings.weights,
        &provisional_bindings.inputs,
        &provisional_bindings.outputs,
        execution_tile_count,
        host_code_base,
        &provisional_auxiliary_ranges,
    )?;
    let provisional_finalizer = TileProgramLowering::new(
        program,
        provisional_placement,
        provisional_exchanges,
        // Sizing only: emission uses fixed-width address materialization.
        RUNTIME_EXECUTABLE_START,
        execution_tile_count,
        false,
    )?;
    let sizing_code_address = memory.next_free(
        RUNTIME_EXECUTABLE_START,
        RUNTIME_EXECUTABLE_START..ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
        4,
        "generated tile programs",
    )?;
    let generated_code_bytes =
        tracing::info_span!("size_tile_code").in_scope(|| -> PackageBuildResult<_> {
            physical_to_logical
                .par_iter()
                .enumerate()
                .map(|(physical, &logical)| {
                    let host = &provisional_host.programs[physical];
                    let mut tile_program = provisional_finalizer.lower_tile(logical)?;
                    if let Some(addresses) = &provisional_profile_addresses {
                        instrument_profile(
                            program,
                            provisional_exchanges,
                            logical,
                            u32::try_from(physical)?,
                            &mut tile_program,
                            addresses[usize::from(logical)],
                        )?;
                    }
                    // Row sharing can change after final placement. Reserve its
                    // optional setup call through the same emitter used below.
                    reserve_exchange_setup(&mut tile_program.steps);
                    let generated = emit(
                        &tile_program,
                        &layout.symbols,
                        host,
                        &CodegenOptions {
                            invocations,
                            code_address: sizing_code_address,
                            initial_profile_address: config
                                .profiling
                                .then_some(PROFILE_START_CYCLE),
                            final_profile_address: config.profiling.then_some(PROFILE_END_CYCLE),
                        },
                    )?;
                    Ok::<_, PackageBuildError>(u32::try_from(generated.bytes.len())?)
                })
                .collect::<PackageBuildResult<Vec<_>>>()?
                .into_iter()
                .max()
                .ok_or_else(|| invalid("execution topology has no tiles"))
        })?;
    tracing::info!(linked_end, host_code_base, host_code_bytes, generated_code_bytes,
        exchange_table_bytes,
        executable_ranges = ?memory.free_ranges(RUNTIME_EXECUTABLE_START..ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT),
        "placing generated tile programs");
    let tile_code = (generated_code_bytes != 0)
        .then(|| {
            allocate_package_code(
                &mut memory,
                "generated tile programs",
                generated_code_bytes,
                8,
                0,
            )
        })
        .transpose()?;
    let code_address = tile_code
        .as_ref()
        .map_or(sizing_code_address, |code| code.range.start);
    // Code can share executable elements. Close their remaining holes only
    // after all code is placed, before allocating writable rows/descriptors.
    protect_executable_elements(
        &mut memory,
        layout
            .segments
            .iter()
            .map(|segment| segment.range.clone())
            .chain(
                host_code
                    .iter()
                    .chain(tile_code.iter())
                    .map(|code| code.reserved.clone()),
            ),
    )?;
    let exchange_rows = (exchange_table_bytes != 0)
        .then(|| {
            memory.allocate(MemoryRequest {
                name: "exchange row tables",
                bytes: exchange_table_bytes,
                // Executed exchange rows may not share an SRAM element with
                // any transfer source or destination. Reserve whole elements
                // at both ends so storage placement cannot use a prefix of the
                // row table's first element.
                alignment: ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE,
                bounds: IPU21_DATA_BASE..ipu_target::ipu21::memory::IPU21_EXECUTABLE_MEMORY_LIMIT,
                end_alignment: ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE,
                guard_after: ipu_target::ipu21::memory::IPU21_SUPERVISOR_FETCH_LOOKAHEAD,
            })
        })
        .transpose()?;
    let exchange_code_base = exchange_rows
        .as_ref()
        .map_or(IPU21_DATA_BASE, |allocation| allocation.range.start);
    // Reserve descriptors before tensors. Their contents depend on final addresses,
    // but their undeduplicated size does not. Final host emission may still reuse
    // packets, leaving part of this reservation unused.
    let host_data_bytes = provisional_host.descriptor_bytes;
    let host_data = (host_data_bytes != 0)
        .then(|| {
            memory.allocate(MemoryRequest {
                name: "host descriptors",
                bytes: host_data_bytes,
                alignment: 4,
                bounds: IPU21_DATA_BASE..ipu_target::ipu21::loader_abi::APPLICATION_LOAD_LIMIT,
                end_alignment: 4,
                guard_after: 0,
            })
        })
        .transpose()?;
    let mut available_ranges =
        memory.free_ranges(IPU21_DATA_BASE..ipu_target::ipu21::loader_abi::APPLICATION_LOAD_LIMIT);
    available_ranges.insert(0, crate::place::HOST_SCRATCH_RANGE);
    tracing::info!(
        linked_end,
        profile_bytes = profile_requests
            .iter()
            .flatten()
            .map(|r| r.bytes)
            .max()
            .unwrap_or(0),
        exchange_table_bytes,
        host_code_bytes,
        host_data_bytes,
        generated_code_bytes,
        code_address,
        ?available_ranges,
        "allocated package support memory"
    );
    Ok(PackageSupport {
        memory,
        available_ranges,
        profile_requests,
        exchange_code_base,
        objects,
        kernel_plan,
        retained_runtime,
        layout,
        physical_to_logical,
        code_address,
        generated_code_bytes,
        host_code_base,
        host_code_bytes,
        host_data,
        exchange_rows,
    })
}
