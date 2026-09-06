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
        profile_address: Option<u32>,
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
                &input.shards,
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
                    &output.shards,
                )
            })
            .collect::<PackageBuildResult<Vec<_>>>()?;
        if let Some(address) = profile_address {
            outputs.push(cycle_binding(
                "profile.start-cycle",
                PROFILE_START_CYCLE,
                program.tile_count,
                topology,
            ));
            outputs.push(profile_binding(program, physical_to_logical, address)?);
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
    program: &LowProgram,
    placement: &crate::Placement,
    topology: &Topology,
    execution_tile_count: u16,
    inactive_ranges: &[(u32, u32)],
) -> PackageBuildResult<Vec<Vec<(u32, u32)>>> {
    let mut ranges = vec![inactive_ranges.to_vec(); usize::from(execution_tile_count)];
    for logical in 0..program.tile_count {
        ranges[usize::from(topology.physical(logical)?)] =
            placement.tile_auxiliary_ranges[usize::from(logical)].clone();
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
    shards: &[crate::BlockValueId],
) -> PackageBuildResult<Binding> {
    let first = shards
        .first()
        .and_then(|id| program.shards.get(id.index() as usize))
        .ok_or_else(|| invalid("binding has no shards"))?;
    let dtype = match first.tensor_type.format.precision {
        crate::Precision::F8F143 { .. } => "f8f143",
        crate::Precision::F16 => "f16",
        crate::Precision::F32 => "f32",
    };
    let mut file_offset = 0u64;
    let slices = shards
        .iter()
        .map(|id| {
            let shard = &program.shards[id.index() as usize];
            let size = u64::from(shard_storage_bytes(shard)?);
            let slice = RegionSlice {
                tile: u32::from(topology.physical(shard.tile)?),
                tile_address: *placement
                    .shard_addresses
                    .get(id)
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
