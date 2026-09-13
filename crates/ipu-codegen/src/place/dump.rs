//! Standalone placement problems for allocator experiments.

use super::*;
use std::hash::{Hash, Hasher};
use std::io::Write;

pub(super) fn capture(tile: u16, requests: &[AllocationRequest], arena: &Arena, placed: bool) {
    let Some(directory) = std::env::var_os("IPU_STACK_PLACEMENT_DUMP") else {
        return;
    };
    if placed && std::env::var_os("IPU_STACK_PLACEMENT_DUMP_ALL").is_none() {
        return;
    }
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec_pretty(&problem(tile, requests, arena, placed))?;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hash);
        let directory = std::path::Path::new(&directory);
        std::fs::create_dir_all(directory)?;
        let path = directory.join(format!("tile-{tile}-{:016x}.json", hash.finish()));
        // Parallel candidate evaluation can encounter the same problem again.
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => file.write_all(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        Ok(())
    })();
    if let Err(error) = result {
        tracing::warn!(tile, %error, "could not dump placement constraints");
    }
}

fn problem(
    tile: u16,
    requests: &[AllocationRequest],
    arena: &Arena,
    placed: bool,
) -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "tile": tile,
        "placed": placed,
        "ranges": arena.ranges,
        "interleaved_offset": arena.interleaved_offset,
        "region_boundary": IPU21_INTERLEAVED_MEMORY_BASE,
        "region0_element_bytes": TILE_MEMORY_ELEMENT_SIZE,
        "region1_element_bytes": IPU21_INTERLEAVED_ELEMENT_SIZE,
        "host_scratch_range": HOST_SCRATCH_RANGE,
        "requests": requests,
        "assigned_root_spans": arena.root_spans,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_preserves_region_stride_and_conflict_constraints() {
        let requests = [AllocationRequest {
            auxiliary: None,
            class: MemoryClass::Ipu21Standard,
            region1_stride: Some(IPU21_INTERLEAVED_ELEMENT_SIZE),
            lifetime: Lifetime {
                first: 0,
                last: u32::MAX,
                seen: true,
            },
            bytes: 32768,
            alignment: 8,
            assignments: vec![(2, 0), (7, 16384)],
            conflicts: vec![9],
        }];
        let arena = Arena::new(&[HOST_SCRATCH_RANGE, (524288, 600000)], 1024);
        let value = problem(40, &requests, &arena, false);
        assert_eq!(value["requests"][0]["region1_stride"], 32768);
        assert_eq!(
            value["requests"][0]["assignments"][1],
            serde_json::json!([7, 16384])
        );
        assert_eq!(value["requests"][0]["conflicts"], serde_json::json!([9]));
        assert_eq!(value["requests"][0]["lifetime"]["last"], u32::MAX);
        assert_eq!(value["interleaved_offset"], 1024);
        assert_eq!(
            value["host_scratch_range"],
            serde_json::json!(HOST_SCRATCH_RANGE)
        );
        assert_eq!(value["ranges"], serde_json::json!(arena.ranges));
    }
}
