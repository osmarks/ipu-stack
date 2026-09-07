//! Run with `cargo test --release -p ipu-codegen benchmark_span_sort -- --ignored --nocapture`.
use super::*;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "manual CPU benchmark"]
fn benchmark_span_sort() {
    let mut cases = Vec::new();
    for (name, order) in [
        ("amp-output", ElementOrder::Amp(AmpOrder::Output)),
        (
            "block-major",
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 64,
            }),
        ),
    ] {
        let block = super::tests::shard(
            crate::Layout {
                order,
                tiling: crate::TensorTiling::replicated(1),
                memory_class: crate::MemoryClass::Ipu21Standard,
            },
            &[256, 256],
        );
        cases.push((
            name,
            logical_byte_spans(block.storage(), &block.extents).unwrap(),
        ));
    }
    let sorted = (0..65536)
        .map(|i| ByteSpan {
            offset: i * 4,
            bytes: 4,
        })
        .collect::<Vec<_>>();
    let mut random = sorted.clone();
    fastrand::Rng::with_seed(42).shuffle(&mut random);
    cases.extend([("sorted", sorted), ("random", random)]);
    println!("case,count,comparison_ns,radix_ns,speedup");
    for (name, spans) in cases {
        for count in [64, 128, 256, 300, 512, 1024, 4096, 16384, 65536] {
            let Some(input) = spans.get(..count) else {
                continue;
            };
            let mut comparison = input.to_vec();
            comparison.sort_unstable_by_key(|span| span.offset);
            let mut radix = input.to_vec();
            radsort::sort_by_key(&mut radix, |span| span.offset);
            assert_eq!(comparison, radix);
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..5 {
                for choice in [round % 2, 1 - round % 2] {
                    let iterations = (1_000_000 / count).clamp(8, 2048);
                    let mut elapsed = 0;
                    for _ in 0..iterations {
                        let mut work = input.to_vec();
                        let start = Instant::now();
                        if choice == 0 {
                            work.sort_unstable_by_key(|span| span.offset);
                        } else {
                            radsort::sort_by_key(&mut work, |span| span.offset);
                        }
                        elapsed += start.elapsed().as_nanos();
                        black_box(work);
                    }
                    times[choice].push(elapsed as f64 / iterations as f64);
                }
            }
            for time in &mut times {
                time.sort_by(f64::total_cmp);
            }
            println!(
                "{name},{count},{:.0},{:.0},{:.2}",
                times[0][2],
                times[1][2],
                times[0][2] / times[1][2]
            );
        }
    }
}
