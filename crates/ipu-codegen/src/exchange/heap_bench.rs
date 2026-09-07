//! Manual queue benchmark; run with --release --ignored --nocapture.
use super::*;
use min_max_heap::MinMaxHeap;
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "manual heap benchmark"]
fn compare_ready_heaps() {
    for count in [1_000, 10_000, 100_000] {
        let mut rng = fastrand::Rng::with_seed(0x68656170);
        let keys: Vec<_> = (0..count)
            .map(|index| ReadyTransfer {
                earliest_start: Reverse(0),
                endpoint_pressure: rng.u64(1..100_000),
                fanout: rng.usize(1..8),
                words: rng.u32(1..512),
                source: Reverse(rng.u16(0..1472)),
                index: Reverse(index),
            })
            .collect();
        // Lazy availability refreshes followed by removal, using the actual key
        // type, deterministic total ordering, and shrinking ready queues.
        macro_rules! run {
            ($heap:expr, $push:expr, $pop:expr) => {{
                let start = Instant::now();
                let mut heap = $heap;
                for &key in &keys {
                    ($push)(&mut heap, key);
                }
                let mut checksum = 0u64;
                while let Some(mut key) = ($pop)(&mut heap) {
                    if key.earliest_start.0 < 4 {
                        key.earliest_start.0 += 1;
                        key.endpoint_pressure /= 2;
                        ($push)(&mut heap, key);
                    } else {
                        checksum = checksum.wrapping_mul(31).wrapping_add(key.index.0 as u64);
                    }
                }
                (start.elapsed().as_secs_f64(), black_box(checksum))
            }};
        }
        for _ in 0..5 {
            let binary = run!(
                BinaryHeap::new(),
                |h: &mut BinaryHeap<_>, k| h.push(k),
                |h: &mut BinaryHeap<_>| h.pop()
            );
            let max = run!(
                MinMaxHeap::new(),
                |h: &mut MinMaxHeap<_>, k| h.push(k),
                |h: &mut MinMaxHeap<_>| h.pop_max()
            );
            let min = run!(
                MinMaxHeap::new(),
                |h: &mut MinMaxHeap<_>, k| h.push(Reverse(k)),
                |h: &mut MinMaxHeap<_>| h.pop_min().map(|Reverse(k)| k)
            );
            assert_eq!(binary.1, max.1);
            assert_eq!(binary.1, min.1);
            println!(
                "count={count} binary={:.6} minmax_max={:.6} minmax_min={:.6}",
                binary.0, max.0, min.0
            );
        }
    }
}
