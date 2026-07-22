use crate::CompileError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowRequirement {
    pub tile: u16,
    /// Independently aligned regions required by one indivisible operation.
    pub regions: Vec<u32>,
}

/// Partitions indivisible operations into the fewest greedy passes that fit a
/// per-tile address window. The result contains indices into `requirements`.
///
/// This is deliberately independent of exchange instructions and tensor
/// layouts. Callers describe only the storage consumed on each destination;
/// they remain responsible for emitting transfers and allocations.
pub(crate) fn partition_address_window(
    requirements: &[WindowRequirement],
    tile_count: u16,
    base: u32,
    limit: u32,
    alignment: u32,
) -> Result<Vec<Vec<usize>>, CompileError> {
    if tile_count == 0
        || base >= limit
        || !alignment.is_power_of_two()
        || requirements
            .iter()
            .any(|requirement| requirement.tile >= tile_count)
    {
        return Err(CompileError::Memory(
            "invalid per-tile address-window requirements".into(),
        ));
    }

    let mut pending = (0..requirements.len()).collect::<Vec<_>>();
    let mut passes = Vec::new();
    while !pending.is_empty() {
        let mut cursors = vec![base; usize::from(tile_count)];
        let mut selected = Vec::new();
        let mut deferred = Vec::new();
        for index in pending {
            let requirement = &requirements[index];
            let cursor = &mut cursors[usize::from(requirement.tile)];
            let mut candidate = *cursor;
            for &size in &requirement.regions {
                candidate = align_up(
                    candidate.checked_add(size).ok_or_else(|| {
                        CompileError::Memory("address-window allocation overflow".into())
                    })?,
                    alignment,
                );
            }
            if candidate <= limit {
                *cursor = candidate;
                selected.push(index);
            } else {
                deferred.push(index);
            }
        }
        if selected.is_empty() {
            let requirement = &requirements[deferred[0]];
            return Err(CompileError::Memory(format!(
                "one operation requires more than {} bytes in tile {} address window",
                limit - base,
                requirement.tile
            )));
        }
        passes.push(selected);
        pending = deferred;
    }
    Ok(passes)
}

fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_partitioning_is_per_tile_and_preserves_every_operation() {
        let requirements = vec![
            WindowRequirement {
                tile: 0,
                regions: vec![40, 24],
            },
            WindowRequirement {
                tile: 0,
                regions: vec![64],
            },
            WindowRequirement {
                tile: 1,
                regions: vec![96],
            },
            WindowRequirement {
                tile: 1,
                regions: Vec::new(),
            },
        ];

        let passes = partition_address_window(&requirements, 2, 0x1000, 0x1060, 16).unwrap();
        assert_eq!(passes.len(), 2);
        let mut indices = passes.into_iter().flatten().collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, (0..requirements.len()).collect::<Vec<_>>());
    }

    #[test]
    fn oversized_indivisible_operation_is_rejected() {
        let error = partition_address_window(
            &[WindowRequirement {
                tile: 1,
                regions: vec![65],
            }],
            2,
            0x1000,
            0x1040,
            16,
        )
        .unwrap_err();
        assert!(error.to_string().contains("tile 1"));
    }
}
