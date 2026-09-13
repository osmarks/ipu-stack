use anyhow::{Context, Result};
use ipu_package::{ProfileExchangeActivityKind, ProfileReport, ProfileStepKind};
use ipu_profile::cycle_origin;
use std::{collections::HashMap, fs, path::Path};

fn payload(report: &ProfileReport) -> serde_json::Value {
    #[derive(Clone, Copy, Hash, PartialEq, Eq)]
    struct StepKey {
        phase: u32,
        epoch: u32,
        operation: u32,
        kernel: u32,
        metadata: u32,
        kind: u8,
        exchange_event_cycles: u32,
    }

    fn intern_string(
        values: &mut Vec<String>,
        indices: &mut HashMap<String, u32>,
        value: &str,
    ) -> u32 {
        if let Some(index) = indices.get(value) {
            return *index;
        }
        let index = values.len() as u32;
        values.push(value.into());
        indices.insert(value.into(), index);
        index
    }

    let mut strings = Vec::new();
    let mut string_indices = HashMap::new();
    let mut metadata_sets = Vec::<Vec<[u32; 2]>>::new();
    let mut metadata_indices = HashMap::<Vec<[u32; 2]>, u32>::new();
    let mut activity_sets = Vec::<Vec<[u32; 5]>>::new();
    let mut activity_indices = HashMap::<Vec<[u32; 5]>, u32>::new();
    let mut steps = Vec::<StepKey>::new();
    let mut step_indices = HashMap::<StepKey, u32>::new();
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
                    let metadata = *metadata_indices.entry(metadata.clone()).or_insert_with(|| {
                        let index = metadata_sets.len() as u32;
                        metadata_sets.push(metadata);
                        index
                    });
                    let step = StepKey {
                        phase: sample.step.phase,
                        epoch: sample.step.epoch,
                        operation: intern_string(
                            &mut strings,
                            &mut string_indices,
                            &sample.step.operation,
                        ),
                        kernel: intern_string(
                            &mut strings,
                            &mut string_indices,
                            &sample.step.kernel,
                        ),
                        metadata,
                        kind: match sample.step.kind {
                            ProfileStepKind::Exchange => 0,
                            ProfileStepKind::Compute => 1,
                            ProfileStepKind::Synchronization => 2,
                            ProfileStepKind::Idle => 3,
                        },
                        exchange_event_cycles: sample.step.exchange_event_cycles,
                    };
                    let step = *step_indices.entry(step).or_insert_with(|| {
                        let index = steps.len() as u32;
                        steps.push(step);
                        index
                    });
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
                    let activities =
                        *activity_indices
                            .entry(activities.clone())
                            .or_insert_with(|| {
                                let index = activity_sets.len() as u32;
                                activity_sets.push(activities);
                                index
                            });
                    serde_json::json!([
                        step,
                        sample.start_cycle.wrapping_sub(base_cycle),
                        sample.end_cycle.wrapping_sub(sample.start_cycle),
                        activities,
                    ])
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "physicalTile": tile.physical_tile,
                "samples": samples,
            })
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
    let steps = steps
        .into_iter()
        .map(|step| {
            serde_json::json!([
                step.phase,
                step.epoch,
                step.operation,
                step.kernel,
                step.metadata,
                step.kind,
                step.exchange_event_cycles,
            ])
        })
        .collect::<Vec<_>>();
    let payload = serde_json::json!({
        "clockHz": report.clock_hz,
        "tileCount": report.tiles.len(),
        "sampleCount": total_samples,
        "strings": strings,
        "metadata": metadata,
        "metadataSets": metadata_sets,
        "activitySets": activity_sets,
        "steps": steps,
        "tiles": tiles,
    });
    payload
}

const PROFILE_REPORT_HTML: &str = include_str!("profile_report.html");

/// Keep sample metadata small and resident; fetch transfer streams only on demand.
/// Each chunk is bounded by event count, except an indivisible single stream.
pub(crate) fn write(report: &ProfileReport, output: &Path, single_file: bool) -> Result<()> {
    let mut data = payload(report);
    if !single_file {
        let directory = output.with_extension("data");
        fs::create_dir_all(&directory)?;
        let directory_name = directory
            .file_name()
            .context("profile output needs a filename")?
            .to_string_lossy();
        let serde_json::Value::Array(activities) = data["activitySets"].take() else {
            unreachable!()
        };
        // Group streams by their earliest phase, so zooming one exchange does
        // not fetch chunks containing unrelated phases from every source tile.
        let mut phase_keys = vec![(u64::MAX, u64::MAX); activities.len()];
        for tile in data["tiles"].as_array().unwrap() {
            for sample in tile["samples"].as_array().unwrap() {
                let step = &data["steps"][sample[0].as_u64().unwrap() as usize];
                let key = (step[1].as_u64().unwrap(), step[0].as_u64().unwrap());
                let index = sample[3].as_u64().unwrap() as usize;
                phase_keys[index] = phase_keys[index].min(key);
            }
        }
        let mut order = (0..activities.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| phase_keys[index]);
        let mut remap = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new;
        }
        for tile in data["tiles"].as_array_mut().unwrap() {
            for sample in tile["samples"].as_array_mut().unwrap() {
                sample[3] = serde_json::json!(remap[sample[3].as_u64().unwrap() as usize]);
            }
        }
        let mut activities = activities;
        let activities = order
            .into_iter()
            .map(|index| activities[index].take())
            .collect::<Vec<_>>();
        let mut chunks = Vec::new();
        let mut summaries = Vec::new();
        let mut counts = Vec::new();
        let mut begin = 0;
        let mut events = 0;
        for (index, activity) in activities.iter().enumerate() {
            summaries.push(summarize(activity));
            let count = activity.as_array().unwrap().len();
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
        data["activitySets"] = serde_json::json!([]);
        data["activitySummaries"] = serde_json::json!(summaries);
        data["activityCounts"] = serde_json::json!(counts);
        data["activityChunks"] = serde_json::json!(chunks);
        // URL-encode path components in the browser, including spaces and '#'.
        data["dataDirectory"] = serde_json::json!(directory_name);
        fs::write(directory.join("index.json"), serde_json::to_vec(&data)?)?;
        data = serde_json::json!({"index": format!("{directory_name}/index.json")});
    }
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
fn summarize(value: &serde_json::Value) -> serde_json::Value {
    let activities = value.as_array().unwrap();
    let mut events = Vec::with_capacity(activities.len() * 2);
    let mut end = 0;
    for activity in activities {
        let a = activity.as_array().unwrap();
        let kind = a[0].as_u64().unwrap() as usize;
        let start = a[1].as_u64().unwrap() as u32;
        let stop = a[2].as_u64().unwrap() as u32;
        let paired = a[4].as_u64().unwrap() != 0;
        let multicast = a[3].as_u64().unwrap() > if paired { 2 } else { 1 };
        events.push((start, kind, multicast, paired, 1i32));
        events.push((stop, kind, multicast, paired, -1));
        end = end.max(stop);
    }
    if end == 0 {
        return serde_json::json!([0, ""]);
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
    serde_json::json!([end, codes])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_summary_keeps_gaps_bidirectionality_and_textures() {
        let summary = summarize(&serde_json::json!([
            [0, 0, 16, 4, 1],
            [1, 8, 24, 1, 0],
            [2, 28, 32, 0, 0]
        ]));
        let codes = summary[1]
            .as_str()
            .unwrap()
            .bytes()
            .map(|b| b - b'@')
            .collect::<Vec<_>>();
        assert_eq!(&codes[0..8], &[25; 8]);
        assert_eq!(&codes[8..16], &[27; 8]);
        assert_eq!(&codes[16..24], &[2; 8]);
        assert_eq!(&codes[24..28], &[0; 4]);
        assert_eq!(&codes[28..32], &[4; 4]);
    }
    #[test]
    fn external_chunks_reconstruct_the_lossless_streams() {
        use ipu_package::{CycleSample, ProfileExchangeActivity, ProfileStep, TileProfile};
        let report = ProfileReport {
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
                            metadata: vec![],
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
        let original = payload(&report);
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
