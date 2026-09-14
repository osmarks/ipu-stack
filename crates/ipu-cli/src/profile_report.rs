use anyhow::{Context, Result};
use ipu_package::{ProfileExchangeActivityKind, ProfileReport, ProfileStepKind};
use ipu_profile::cycle_origin;
use std::{collections::HashMap, fs, path::Path};

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Payload {
    clock_hz: u64,
    tile_count: usize,
    sample_count: usize,
    strings: Vec<String>,
    metadata: Vec<u32>,
    metadata_sets: Vec<[u32; 2]>,
    activity_sets: Vec<Vec<[u32; 5]>>,
    // phase, epoch, operation, kernel, metadata, kind, exchange event cycles
    steps: Vec<[u32; 7]>,
    tiles: Vec<Tile>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Tile {
    physical_tile: u32,
    // step, relative start, duration, activity stream
    samples: Vec<[u32; 4]>,
}

fn payload(report: &ProfileReport) -> Payload {
    fn intern<T: Clone + Eq + std::hash::Hash>(
        values: &mut Vec<T>,
        indices: &mut HashMap<T, u32>,
        value: T,
    ) -> u32 {
        *indices.entry(value).or_insert_with_key(|value| {
            let index = values.len() as u32;
            values.push(value.clone());
            index
        })
    }

    fn intern_string(
        values: &mut Vec<String>,
        indices: &mut HashMap<String, u32>,
        value: &str,
    ) -> u32 {
        indices
            .get(value)
            .copied()
            .unwrap_or_else(|| intern(values, indices, value.to_owned()))
    }

    let mut strings = Vec::new();
    let mut string_indices = HashMap::new();
    let mut metadata_sets = Vec::<Vec<[u32; 2]>>::new();
    let mut metadata_indices = HashMap::<Vec<[u32; 2]>, u32>::new();
    let mut activity_sets = Vec::<Vec<[u32; 5]>>::new();
    let mut activity_indices = HashMap::<Vec<[u32; 5]>, u32>::new();
    let mut steps = Vec::<[u32; 7]>::new();
    let mut step_indices = HashMap::<[u32; 7], u32>::new();
    let base_cycle = cycle_origin(report);
    let tiles = report
        .tiles
        .iter()
        .map(|tile| {
            let samples = tile
                .samples
                .iter()
                .map(|sample| {
                    let metadata = sample
                        .step
                        .metadata
                        .iter()
                        .map(|entry| {
                            [
                                intern_string(&mut strings, &mut string_indices, &entry.name),
                                intern_string(&mut strings, &mut string_indices, &entry.value),
                            ]
                        })
                        .collect::<Vec<_>>();
                    let metadata = intern(&mut metadata_sets, &mut metadata_indices, metadata);
                    let step = [
                        sample.step.phase,
                        sample.step.epoch,
                        intern_string(&mut strings, &mut string_indices, &sample.step.operation),
                        intern_string(&mut strings, &mut string_indices, &sample.step.kernel),
                        metadata,
                        match sample.step.kind {
                            ProfileStepKind::Exchange => 0,
                            ProfileStepKind::Compute => 1,
                            ProfileStepKind::Synchronization => 2,
                            ProfileStepKind::Idle => 3,
                        },
                        sample.step.exchange_event_cycles,
                    ];
                    let step = intern(&mut steps, &mut step_indices, step);
                    let activities = sample
                        .step
                        .exchange_activities
                        .iter()
                        .map(|activity| {
                            [
                                match activity.kind {
                                    ProfileExchangeActivityKind::Send => 0,
                                    ProfileExchangeActivityKind::Receive => 1,
                                    ProfileExchangeActivityKind::PartnerBusy => 2,
                                },
                                activity.start_cycle,
                                activity.end_cycle,
                                u32::from(activity.fanout),
                                u32::from(activity.paired),
                            ]
                        })
                        .collect::<Vec<_>>();
                    let activities = intern(&mut activity_sets, &mut activity_indices, activities);
                    [
                        step,
                        sample.start_cycle.wrapping_sub(base_cycle),
                        sample.end_cycle.wrapping_sub(sample.start_cycle),
                        activities,
                    ]
                })
                .collect::<Vec<_>>();
            Tile {
                physical_tile: tile.physical_tile,
                samples,
            }
        })
        .collect::<Vec<_>>();
    let total_samples: usize = report.tiles.iter().map(|tile| tile.samples.len()).sum();
    let mut metadata = Vec::new();
    let metadata_sets = metadata_sets
        .into_iter()
        .map(|entries| {
            let start = metadata.len() as u32;
            let count = entries.len() as u32;
            for [name, value] in entries {
                metadata.extend([name, value]);
            }
            [start, count]
        })
        .collect::<Vec<_>>();
    Payload {
        clock_hz: report.clock_hz,
        tile_count: report.tiles.len(),
        sample_count: total_samples,
        strings,
        metadata,
        metadata_sets,
        activity_sets,
        steps,
        tiles,
    }
}

const PROFILE_REPORT_HTML: &str = include_str!("profile_report.html");

/// Keep sample metadata small and resident; fetch transfer streams only on demand.
/// Each chunk is bounded by event count, except an indivisible single stream.
pub(crate) fn write(report: &ProfileReport, output: &Path, single_file: bool) -> Result<()> {
    let mut data = payload(report);
    let data = if !single_file {
        let directory = output.with_extension("data");
        fs::create_dir_all(&directory)?;
        let directory_name = directory
            .file_name()
            .context("profile output needs a filename")?
            .to_string_lossy();
        let mut activities = std::mem::take(&mut data.activity_sets);
        // Group streams by their earliest phase, so zooming one exchange does
        // not fetch chunks containing unrelated phases from every source tile.
        let mut phase_keys = vec![(u32::MAX, u32::MAX); activities.len()];
        for tile in &data.tiles {
            for sample in &tile.samples {
                let step = &data.steps[sample[0] as usize];
                let key = (step[1], step[0]);
                let index = sample[3] as usize;
                phase_keys[index] = phase_keys[index].min(key);
            }
        }
        let mut order = (0..activities.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| phase_keys[index]);
        let mut remap = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new as u32;
        }
        for tile in &mut data.tiles {
            for sample in &mut tile.samples {
                sample[3] = remap[sample[3] as usize];
            }
        }
        let activities = order
            .into_iter()
            .map(|index| std::mem::take(&mut activities[index]))
            .collect::<Vec<_>>();
        let mut chunks = Vec::new();
        let mut summaries = Vec::new();
        let mut counts = Vec::new();
        let mut begin = 0;
        let mut events = 0;
        for (index, activity) in activities.iter().enumerate() {
            summaries.push(summarize(activity));
            let count = activity.len();
            counts.push(count);
            events += count;
            if events >= 20_000 || index + 1 == activities.len() {
                let name = format!("transfers-{}.json", chunks.len());
                fs::write(
                    directory.join(&name),
                    serde_json::to_vec(&activities[begin..=index])?,
                )?;
                chunks.push(serde_json::json!([begin, index + 1, events, name]));
                begin = index + 1;
                events = 0;
            }
        }
        let mut data = serde_json::to_value(data)?;
        data["activitySummaries"] = serde_json::json!(summaries);
        data["activityCounts"] = serde_json::json!(counts);
        data["activityChunks"] = serde_json::json!(chunks);
        // URL-encode path components in the browser, including spaces and '#'.
        data["dataDirectory"] = serde_json::json!(directory_name);
        fs::write(directory.join("index.json"), serde_json::to_vec(&data)?)?;
        serde_json::json!({"index": format!("{directory_name}/index.json")})
    } else {
        serde_json::to_value(data)?
    };
    let json = serde_json::to_string(&data)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    fs::write(
        output,
        PROFILE_REPORT_HTML.replace("__PROFILE_JSON__", &json),
    )?;
    Ok(())
}

// A bounded overview of each stream, independent of its number of transfers.
// Each bin stores the dominant activity by duration, retaining paired/multicast
// texture. Detailed streams remain lossless in the sidecar chunks.
fn summarize(activities: &[[u32; 5]]) -> (u32, String) {
    let mut events = Vec::with_capacity(activities.len() * 2);
    let mut end = 0;
    for &[kind, start, stop, fanout, paired] in activities {
        let kind = kind as usize;
        let paired = paired != 0;
        let multicast = fanout > if paired { 2 } else { 1 };
        events.push((start, kind, multicast, paired, 1i32));
        events.push((stop, kind, multicast, paired, -1));
        end = end.max(stop);
    }
    if end == 0 {
        return (0, String::new());
    }
    events.sort_unstable_by_key(|e| e.0);
    let mut bins = [[0u64; 32]; 32];
    let mut counts = [0i32; 5];
    let mut cursor = 0;
    for (time, kind, multicast, paired, delta) in events {
        let base = match (counts[0] > 0, counts[1] > 0, counts[2] > 0) {
            (true, true, _) => 3,
            (true, false, _) => 1,
            (false, true, _) => 2,
            (false, false, true) => 4,
            _ => 0,
        };
        let code = base | if counts[3] > 0 { 8 } else { 0 } | if counts[4] > 0 { 16 } else { 0 };
        // Scale coordinates by 32 to avoid rounding away short bins.
        let start = u64::from(cursor) * 32;
        let stop = u64::from(time) * 32;
        for (bin, weights) in bins.iter_mut().enumerate() {
            let lo = (bin as u64 * u64::from(end)).max(start);
            let hi = ((bin + 1) as u64 * u64::from(end)).min(stop);
            weights[code] += hi.saturating_sub(lo);
        }
        counts[kind] += delta;
        counts[3] += i32::from(multicast) * delta;
        counts[4] += i32::from(paired) * delta;
        cursor = time;
    }
    let codes = bins
        .iter()
        .map(|weights| {
            let (code, _) = weights
                .iter()
                .enumerate()
                .max_by_key(|(_, weight)| **weight)
                .unwrap();
            char::from(b'@' + code as u8)
        })
        .collect::<String>();
    (end, codes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_summary_keeps_gaps_bidirectionality_and_textures() {
        let summary = summarize(&[[0, 0, 16, 4, 1], [1, 8, 24, 1, 0], [2, 28, 32, 0, 0]]);
        let codes = summary.1.bytes().map(|b| b - b'@').collect::<Vec<_>>();
        assert_eq!(&codes[0..8], &[25; 8]);
        assert_eq!(&codes[8..16], &[27; 8]);
        assert_eq!(&codes[16..24], &[2; 8]);
        assert_eq!(&codes[24..28], &[0; 4]);
        assert_eq!(&codes[28..32], &[4; 4]);
    }
    #[test]
    fn external_chunks_reconstruct_the_lossless_streams() {
        use ipu_package::{CycleSample, ProfileExchangeActivity, ProfileStep, TileProfile};
        let mut report = ProfileReport {
            clock_hz: 1_500_000_000,
            tiles: vec![TileProfile {
                physical_tile: 3,
                samples: (0..3)
                    .rev()
                    .map(|phase| CycleSample {
                        start_cycle: phase * 100000,
                        end_cycle: (phase + 1) * 100000,
                        step: ProfileStep {
                            local_index: phase,
                            phase,
                            epoch: 0,
                            operation: "test </script> & unicode λ".into(),
                            kernel: String::new(),
                            kind: ProfileStepKind::Exchange,
                            metadata: vec![ipu_package::ProfileMetadata {
                                name: "source".into(),
                                value: "shared projection".into(),
                            }],
                            exchange_event_cycles: 100000,
                            exchange_activities: (0..10001)
                                .map(|i| ProfileExchangeActivity {
                                    fanout: 2,
                                    paired: phase == 1,
                                    kind: ProfileExchangeActivityKind::Send,
                                    start_cycle: i * 4 + phase,
                                    end_cycle: i * 4 + phase + 1,
                                })
                                .collect(),
                        },
                    })
                    .collect(),
            }],
        };
        let before = serde_json::to_value(payload(&report)).unwrap();
        let mut duplicate = report.tiles[0].clone();
        duplicate.physical_tile = 7;
        report.tiles.push(duplicate);
        let after = serde_json::to_value(payload(&report)).unwrap();
        for table in [
            "strings",
            "metadata",
            "metadataSets",
            "steps",
            "activitySets",
        ] {
            assert_eq!(
                before[table], after[table],
                "duplicate tile enlarged {table}"
            );
        }
        assert_eq!(after["tiles"][0]["samples"], after["tiles"][1]["samples"]);
        let directory =
            std::env::temp_dir().join(format!("ipu-profile-report-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let output = directory.join("profile # λ.html");
        write(&report, &output, false).unwrap();
        let data_dir = output.with_extension("data");
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(data_dir.join("index.json")).unwrap()).unwrap();
        let mut recovered = Vec::new();
        for chunk in index["activityChunks"].as_array().unwrap() {
            assert_eq!(chunk[0].as_u64().unwrap() as usize, recovered.len());
            let values: Vec<serde_json::Value> = serde_json::from_slice(
                &fs::read(data_dir.join(chunk[3].as_str().unwrap())).unwrap(),
            )
            .unwrap();
            recovered.extend(values);
            assert_eq!(chunk[1].as_u64().unwrap() as usize, recovered.len());
        }
        let original = serde_json::to_value(payload(&report)).unwrap();
        for (before, after) in original["tiles"][0]["samples"]
            .as_array()
            .unwrap()
            .iter()
            .zip(index["tiles"][0]["samples"].as_array().unwrap())
        {
            assert_eq!(
                original["activitySets"][before[3].as_u64().unwrap() as usize],
                recovered[after[3].as_u64().unwrap() as usize]
            );
            assert_eq!(
                &before.as_array().unwrap()[..3],
                &after.as_array().unwrap()[..3]
            );
        }
        assert!(fs::metadata(&output).unwrap().len() < 100000);
        write(&report, &output, true).unwrap();
        let html = fs::read_to_string(output).unwrap();
        assert!(!html.contains("test </script>"));
        assert!(html.contains("\\u003c/script\\u003e"));
        fs::remove_dir_all(directory).unwrap();
    }
}
