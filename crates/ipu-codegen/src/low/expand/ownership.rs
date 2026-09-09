//! Resolve copy regions before expanding their replica recipients.
//!
//! Equal source regions are one piece of data with several possible owners.
//! Equal destination requests share the intersection geometry. Local replicas
//! retain priority; all other recipients select the same default sender, so
//! multicast membership is determined without repeating source intersections.

use super::*;

#[derive(Clone)]
struct Owners {
    first: (usize, BlockValueId),
    local: BTreeMap<u16, (usize, BlockValueId)>,
}

impl Owners {
    fn insert(&mut self, tile: u16, owner: (usize, BlockValueId)) {
        self.first = self.first.min(owner);
        self.local
            .entry(tile)
            .and_modify(|previous| *previous = (*previous).min(owner))
            .or_insert(owner);
    }

    fn merge(&mut self, other: &Self) {
        for (&tile, &owner) in &other.local {
            self.insert(tile, owner);
        }
    }

    fn select(&self, tile: u16) -> BlockValueId {
        self.local.get(&tile).unwrap_or(&self.first).1
    }
}

pub(super) struct CopyRegions {
    sources: Vec<(Vec<ShardExtent>, Arc<Owners>)>,
    axis: Option<usize>,
    prefix_ends: Vec<u32>,
    targets: BTreeMap<Vec<ShardExtent>, Vec<(Vec<ShardExtent>, Arc<Owners>)>>,
}

impl CopyRegions {
    pub(super) fn new(shards: &[BlockValue], sources: &[BlockValueId]) -> Self {
        let mut regions = BTreeMap::<Vec<ShardExtent>, Owners>::new();
        for (index, &id) in sources.iter().enumerate() {
            let shard = &shards[id.index() as usize];
            // Intersection geometry excludes padding. The selected shard still
            // determines shared padding later, where the copy order is known.
            let mut extents = shard.extents.clone();
            for extent in &mut extents {
                extent.physical_end = extent.logical_end;
            }
            regions
                .entry(extents)
                .and_modify(|owners| owners.insert(shard.tile, (index, id)))
                .or_insert_with(|| Owners {
                    first: (index, id),
                    local: BTreeMap::from([(shard.tile, (index, id))]),
                });
        }
        let mut sources = regions
            .into_iter()
            .map(|(extents, owners)| (extents, Arc::new(owners)))
            .collect::<Vec<_>>();
        // Search one discriminating axis before doing full intersections. Prefix
        // maxima retain overlapping regions; this is not a disjoint-grid assumption.
        let rank = sources.first().map_or(0, |(extents, _)| extents.len());
        let axis = (0..rank)
            .filter(|_| sources.iter().all(|(e, _)| e.len() == rank))
            .max_by_key(|&axis| {
                sources
                    .iter()
                    .map(|(e, _)| e[axis].start)
                    .collect::<BTreeSet<_>>()
                    .len()
            });
        let mut prefix_ends = Vec::new();
        if let Some(axis) = axis {
            sources.sort_by_key(|(e, _)| e[axis].start);
            let mut end = 0;
            prefix_ends = sources
                .iter()
                .map(|(e, _)| {
                    end = end.max(e[axis].logical_end);
                    end
                })
                .collect();
        }
        Self {
            sources,
            axis,
            prefix_ends,
            targets: BTreeMap::new(),
        }
    }

    pub(super) fn intersections(
        &mut self,
        target: &[ShardExtent],
        tile: u16,
    ) -> Vec<(Vec<ShardExtent>, BlockValueId)> {
        let mut key = target.to_vec();
        for extent in &mut key {
            extent.physical_end = extent.logical_end;
        }
        let intersections = self.targets.entry(key).or_insert_with_key(|target| {
            let mut intersections = BTreeMap::<Vec<ShardExtent>, Arc<Owners>>::new();
            let sources = if let Some(axis) = self.axis {
                let Some(extent) = target.get(axis) else {
                    return Vec::new();
                };
                let first = self.prefix_ends.partition_point(|&end| end <= extent.start);
                let last = self
                    .sources
                    .partition_point(|(e, _)| e[axis].start < extent.logical_end);
                &self.sources[first.min(last)..last]
            } else {
                &self.sources
            };
            for (source, owners) in sources {
                if let Some(extents) = intersect_extents(source, target) {
                    // Distinct source regions can clip to the same request.
                    // Preserve the original input order when selecting an owner.
                    intersections
                        .entry(extents)
                        .and_modify(|previous| Arc::make_mut(previous).merge(owners))
                        .or_insert_with(|| owners.clone());
                }
            }
            intersections.into_iter().collect()
        });
        intersections
            .iter()
            .map(|(extents, owners)| (extents.clone(), owners.select(tile)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard(id: u32, tile: u16, start: u32, end: u32, padding: u32) -> BlockValue {
        BlockValue {
            id: BlockValueId(id),
            tile,
            tensor_type: TensorType::new([4096, 128], Precision::F16, Layout::row_sharded(1)),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start,
                    logical_end: end,
                    physical_end: end + padding,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: 128,
                    physical_end: 128,
                },
            ],
            definition: ShardDefinition::Staging,
        }
    }

    // Independent reference: the destination-first expansion used before
    // replica ownership was grouped. Keep ordering and clipped-region merging.
    fn reference(
        shards: &[BlockValue],
        sources: &[BlockValueId],
        target: &[ShardExtent],
        tile: u16,
    ) -> Vec<(Vec<ShardExtent>, BlockValueId)> {
        let mut groups = BTreeMap::<Vec<ShardExtent>, Vec<BlockValueId>>::new();
        for &source in sources {
            if let Some(extents) =
                intersect_extents(&shards[source.index() as usize].extents, target)
            {
                groups.entry(extents).or_default().push(source);
            }
        }
        groups
            .into_iter()
            .map(|(extents, owners)| {
                let selected = owners
                    .iter()
                    .copied()
                    .find(|id| shards[id.index() as usize].tile == tile)
                    .unwrap_or(owners[0]);
                (extents, selected)
            })
            .collect()
    }

    #[test]
    fn region_ownership_preserves_replica_selection_and_clipped_overlaps() {
        let mut random = fastrand::Rng::with_seed(0x6d75_6c74_6963_6173);
        for case in 0..128 {
            let orient = |mut extents: Vec<ShardExtent>| {
                if case % 2 == 0 {
                    extents.swap(0, 1);
                }
                for (axis, extent) in extents.iter_mut().enumerate() {
                    extent.axis = axis as u16;
                }
                extents
            };
            let shards = (0..32)
                .map(|id| {
                    let start = random.u32(0..8) * 16;
                    let end = start + random.u32(1..9) * 16;
                    let mut value = shard(id, random.u16(0..16), start, end, random.u32(0..4));
                    value.extents = orient(value.extents);
                    value
                })
                .collect::<Vec<_>>();
            let mut sources = shards.iter().map(|s| s.id).collect::<Vec<_>>();
            random.shuffle(&mut sources);
            let mut regions = CopyRegions::new(&shards, &sources);
            for _ in 0..16 {
                let start = random.u32(0..16) * 8;
                let end = start + random.u32(1..16) * 8;
                let target = orient(shard(0, 0, start, end, 8).extents);
                for tile in 0..20 {
                    assert_eq!(
                        regions.intersections(&target, tile),
                        reference(&shards, &sources, &target, tile)
                    );
                }
            }
        }
    }

    #[test]
    fn replicas_share_geometry_but_keep_local_owners() {
        let shards = (0..1472)
            .map(|id| shard(id, id as u16, 0, 4096, id % 8))
            .collect::<Vec<_>>();
        let sources = shards.iter().map(|s| s.id).collect::<Vec<_>>();
        let mut regions = CopyRegions::new(&shards, &sources);
        for tile in 0..1472 {
            assert_eq!(
                regions.intersections(&shards[tile as usize].extents, tile)[0].1,
                sources[tile as usize]
            );
        }
        assert_eq!(regions.sources.len(), 1);
        assert_eq!(regions.targets.len(), 1);
        assert_eq!(
            regions.intersections(&shards[0].extents, 1472)[0].1,
            sources[0]
        );
    }

    #[test]
    #[ignore = "manual comparison with destination-first copy expansion"]
    fn benchmark_copy_regions() {
        for (source_parts, source_replicas, target_parts, target_replicas) in [
            (1, 1472, 1, 1472),
            (64, 16, 32, 32),
            (1472, 1, 1472, 1),
            (1, 1, 64, 23),
            (64, 1, 1, 1472),
        ] {
            let shards = (0..source_parts * source_replicas)
                .map(|id| {
                    let part = id / source_replicas;
                    shard(
                        id,
                        id as u16,
                        part * 4096 / source_parts,
                        (part + 1) * 4096 / source_parts,
                        0,
                    )
                })
                .collect::<Vec<_>>();
            let sources = shards.iter().map(|s| s.id).collect::<Vec<_>>();
            let targets = (0..target_parts * target_replicas)
                .map(|id| {
                    let part = id / target_replicas;
                    shard(
                        id,
                        id as u16,
                        part * 4096 / target_parts,
                        (part + 1) * 4096 / target_parts,
                        0,
                    )
                })
                .collect::<Vec<_>>();
            let started = Instant::now();
            let expected = targets
                .iter()
                .map(|target| reference(&shards, &sources, &target.extents, target.tile))
                .collect::<Vec<_>>();
            let old = started.elapsed();
            let started = Instant::now();
            let mut regions = CopyRegions::new(&shards, &sources);
            let actual = targets
                .iter()
                .map(|target| regions.intersections(&target.extents, target.tile))
                .collect::<Vec<_>>();
            let new = started.elapsed();
            assert_eq!(actual, expected);
            eprintln!(
                "sources={source_parts}x{source_replicas} targets={target_parts}x{target_replicas} old_us={} new_us={} source_regions={} target_regions={}",
                old.as_micros(),
                new.as_micros(),
                regions.sources.len(),
                regions.targets.len()
            );
        }
    }
}
