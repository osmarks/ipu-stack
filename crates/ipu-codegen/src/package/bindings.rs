//! Host-visible bindings and physical-tile memory ranges for any placement.

use super::*;

pub(super) struct PackageBindings {
    pub inputs: Vec<Binding>,
    pub weights: Vec<Binding>,
    pub outputs: Vec<Binding>,
}

impl PackageBindings {
    pub(super) fn new(
        program: &LowProgram,
        placement: &crate::Placement,
        topology: &Topology,
        physical_to_logical: &[u16],
        profile_addresses: Option<&[u32]>,
    ) -> PackageBuildResult<Self> {
        let mut inputs = Vec::new();
        let mut weights = Vec::new();
        for input in &program.inputs {
            let bindings = match input.kind {
                crate::GraphInputKind::Host => &mut inputs,
                crate::GraphInputKind::Parameter => &mut weights,
            };
            bindings.push(binding(
                program,
                placement,
                topology,
                input.name.clone(),
                program.value_views(input.value),
            )?);
        }
        let mut outputs = program
            .outputs
            .iter()
            .enumerate()
            .map(|(index, output)| {
                binding(
                    program,
                    placement,
                    topology,
                    format!("output.{index}"),
                    program.value_views(*output),
                )
            })
            .collect::<PackageBuildResult<Vec<_>>>()?;
        if let Some(addresses) = profile_addresses {
            outputs.push(cycle_binding(
                "profile.start-cycle",
                PROFILE_START_CYCLE,
                program.tile_count,
                topology,
            ));
            outputs.push(profile_binding(program, physical_to_logical, addresses)?);
            outputs.push(cycle_binding(
                "profile.end-cycle",
                PROFILE_END_CYCLE,
                program.tile_count,
                topology,
            ));
        }
        Ok(Self {
            inputs,
            weights,
            outputs,
        })
    }
}

pub(super) fn auxiliary_ranges(
    placement: &crate::Placement,
    topology: &Topology,
    execution_tile_count: u16,
    inactive_ranges: &[(u32, u32)],
) -> PackageBuildResult<Vec<Vec<(u32, u32)>>> {
    let mut ranges = vec![inactive_ranges.to_vec(); usize::from(execution_tile_count)];
    for (logical, unused) in placement.tile_auxiliary_ranges.iter().enumerate() {
        ranges[usize::from(topology.physical(u16::try_from(logical)?)?)] = unused.clone();
    }
    Ok(ranges)
}

fn cycle_binding(name: &str, address: u32, tile_count: u16, topology: &Topology) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u32".into(),
        shape: vec![u32::from(tile_count)],
        slices: (0..tile_count)
            .map(|tile| RegionSlice {
                tile: u32::from(
                    topology
                        .physical(tile)
                        .expect("active topology contains tile"),
                ),
                tile_address: address,
                file_offset: u64::from(tile) * 4,
                size: 4,
            })
            .collect(),
    }
}

fn binding(
    program: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    name: String,
    views: &[crate::ShardView],
) -> PackageBuildResult<Binding> {
    let first = views
        .first()
        .and_then(|view| program.shards.get(view.shard.index() as usize))
        .ok_or_else(|| invalid("binding has no shards"))?;
    let dtype = match first.tensor_type.format.precision {
        crate::Precision::F8F143 { .. } => "f8f143",
        crate::Precision::F16 => "f16",
        crate::Precision::F32 => "f32",
    };
    let mut file_offset = 0u64;
    let slices = views
        .iter()
        .map(|view| {
            let shard = &program.shards[view.shard.index() as usize];
            if view.extents != shard.extents {
                return Err(invalid(
                    "host binding requires canonical whole-buffer storage",
                ));
            }
            let size = u64::from(shard_storage_bytes(shard)?);
            let slice = RegionSlice {
                tile: u32::from(topology.physical(shard.tile)?),
                tile_address: *placement
                    .shard_addresses
                    .get(&view.shard)
                    .ok_or_else(|| invalid("binding shard is not placed"))?,
                file_offset,
                size,
            };
            file_offset = file_offset
                .checked_add(size)
                .ok_or_else(|| invalid("binding file offset overflow"))?;
            Ok(slice)
        })
        .collect::<PackageBuildResult<Vec<_>>>()?;
    Ok(Binding {
        name,
        dtype: dtype.into(),
        shape: first.tensor_type.shape.0.clone(),
        slices,
    })
}
