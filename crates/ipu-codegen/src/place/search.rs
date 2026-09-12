//! Bounded address search for tiles that fail the cheap allocation orders.
//!
//! Domains are intervals of aligned starts, not enumerated byte addresses.
//! Every decision propagates lifetime and physical-element exclusions into all
//! remaining domains. Branching samples hole edges and element boundaries;
//! exhaustion therefore means unknown, never a proof of infeasibility.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Domain {
    pub first: u32,
    pub last: u32,
    pub alignment: u32,
    pub bytes: u32,
}

impl Domain {
    fn without(self, start: u32, end: u32) -> impl Iterator<Item = Self> {
        // An allocation [x, x + bytes) overlaps [start, end) exactly when
        // start - bytes < x < end. Checked subtraction handles the zero edge.
        let before = start.checked_sub(self.bytes).map(|last| Self {
            last: self.last.min(last / self.alignment * self.alignment),
            ..self
        });
        let after = align_up(end, self.alignment).ok().map(|first| Self {
            first: self.first.max(first),
            ..self
        });
        [before, after]
            .into_iter()
            .flatten()
            .filter(|domain| domain.first <= domain.last)
    }

    fn starts(self) -> impl Iterator<Item = u32> {
        let element = if self.first >= IPU21_INTERLEAVED_MEMORY_BASE {
            IPU21_INTERLEAVED_ELEMENT_SIZE
        } else {
            TILE_MEMORY_ELEMENT_SIZE
        };
        let mut starts = vec![self.first, self.last];
        let mut boundary = self.first.div_ceil(element) * element;
        while boundary <= self.last.saturating_add(self.bytes) {
            if let Ok(start) = align_up(boundary, self.alignment)
                && self.first <= start
                && start <= self.last
            {
                starts.push(start);
            }
            if let Some(start) = boundary.checked_sub(self.bytes) {
                let start = start / self.alignment * self.alignment;
                if self.first <= start && start <= self.last {
                    starts.push(start);
                }
            }
            let Some(next) = boundary.checked_add(element) else {
                break;
            };
            boundary = next;
        }
        starts.sort_unstable();
        starts.dedup();
        starts.into_iter()
    }
}

impl AllocationRequest {
    /// Shared eligibility/stride rules for greedy allocation and address search.
    pub(super) fn domains<'a>(
        &'a self,
        ranges: &'a [(u32, u32)],
        interleaved_offset: u32,
    ) -> impl Iterator<Item = (usize, Domain)> + 'a {
        ranges
            .iter()
            .enumerate()
            .flat_map(move |(index, &(base, limit))| {
                [false, true].into_iter().filter_map(move |region1| {
                    if HOST_SCRATCH_RANGE.0 <= base
                        && limit <= HOST_SCRATCH_RANGE.1
                        && (self.lifetime.first == 0 || self.lifetime.last == u32::MAX)
                    {
                        return None;
                    }
                    let (base, limit, element) = if region1 {
                        (
                            base.max(IPU21_INTERLEAVED_MEMORY_BASE.checked_add(
                                if self.class == MemoryClass::Ipu21Interleaved {
                                    interleaved_offset
                                } else {
                                    0
                                },
                            )?),
                            limit,
                            IPU21_INTERLEAVED_ELEMENT_SIZE,
                        )
                    } else {
                        if self.class == MemoryClass::Ipu21Interleaved {
                            return None;
                        }
                        (
                            base,
                            limit.min(IPU21_INTERLEAVED_MEMORY_BASE),
                            TILE_MEMORY_ELEMENT_SIZE,
                        )
                    };
                    let alignment = self.alignment.max(if self.region1_stride.is_some() {
                        element
                    } else {
                        1
                    });
                    let bytes = if region1 && let Some(stride) = self.region1_stride {
                        stride.checked_mul(u32::try_from(self.assignments.len()).ok()?)?
                    } else {
                        self.bytes
                    };
                    let first = align_up(base, alignment).ok()?;
                    let last = limit.checked_sub(bytes)? / alignment * alignment;
                    (first <= last).then_some((
                        index,
                        Domain {
                            first,
                            last,
                            alignment,
                            bytes,
                        },
                    ))
                })
            })
    }
}

#[derive(Debug)]
pub(super) struct Outcome {
    pub placement: Option<Vec<(u32, u32)>>,
    pub nodes: usize,
    pub excess_live_bytes: u64,
}

/// Keep pathological tiles bounded without changing successful greedy placements.
const NODE_BUDGET: usize = 4096;
const PROPAGATION_BUDGET: usize = 2_000_000;

pub(super) fn place(requests: &[AllocationRequest], arena: &Arena) -> Outcome {
    solve(requests, arena, NODE_BUDGET, PROPAGATION_BUDGET)
}

fn solve(
    requests: &[AllocationRequest],
    arena: &Arena,
    node_budget: usize,
    propagation_budget: usize,
) -> Outcome {
    let domains: Vec<Vec<Domain>> = requests
        .iter()
        .map(|request| {
            request
                .domains(&arena.ranges, arena.interleaved_offset)
                .map(|(_, d)| d)
                .collect()
        })
        .collect();
    let capacity: u64 = arena.ranges.iter().map(|&(a, b)| u64::from(b - a)).sum();
    let minimum: Vec<u64> = domains
        .iter()
        .map(|d| d.iter().map(|d| u64::from(d.bytes)).min().unwrap_or(0))
        .collect();
    let peak = requests
        .iter()
        .map(|r| {
            requests
                .iter()
                .zip(&minimum)
                .filter(|(q, _)| {
                    q.lifetime.first <= r.lifetime.first && r.lifetime.first <= q.lifetime.last
                })
                .map(|(_, &bytes)| bytes)
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    if peak > capacity {
        return Outcome {
            placement: None,
            nodes: 0,
            excess_live_bytes: peak - capacity,
        };
    }
    let owner: BTreeMap<usize, usize> = requests
        .iter()
        .enumerate()
        .flat_map(|(i, r)| r.assignments.iter().map(move |&(root, _)| (root, i)))
        .collect();
    let mut edges = vec![vec![false; requests.len()]; requests.len()];
    for (i, r) in requests.iter().enumerate() {
        for root in &r.conflicts {
            if let Some(&j) = owner.get(root)
                && i != j
            {
                edges[i][j] = true;
                edges[j][i] = true;
            }
        }
    }
    let mut search = Search {
        requests,
        edges,
        nodes: 0,
        work: 0,
        node_budget,
        propagation_budget,
    };
    let mut placement = vec![None; requests.len()];
    let found = search.visit(domains, &mut placement);
    Outcome {
        placement: found.then(|| placement.into_iter().map(Option::unwrap).collect()),
        nodes: search.nodes,
        excess_live_bytes: 0,
    }
}

struct Search<'a> {
    requests: &'a [AllocationRequest],
    edges: Vec<Vec<bool>>,
    nodes: usize,
    work: usize,
    node_budget: usize,
    propagation_budget: usize,
}

impl Search<'_> {
    fn visit(&mut self, domains: Vec<Vec<Domain>>, placement: &mut [Option<(u32, u32)>]) -> bool {
        if self.nodes >= self.node_budget || self.work >= self.propagation_budget {
            return false;
        }
        self.nodes += 1;
        let Some(index) = (0..domains.len())
            .filter(|&i| placement[i].is_none())
            .min_by_key(|&i| {
                let r = &self.requests[i];
                let starts: u64 = domains[i]
                    .iter()
                    .map(|d| u64::from((d.last - d.first) / d.alignment) + 1)
                    .sum();
                (
                    !domains[i].is_empty(),
                    r.lifetime.last != u32::MAX,
                    starts.saturating_mul(u64::from(r.alignment)) / u64::from(r.bytes.max(1)),
                    std::cmp::Reverse(self.edges[i].iter().filter(|&&edge| edge).count()),
                    std::cmp::Reverse(r.bytes),
                    i,
                )
            })
        else {
            return true;
        };
        let r = &self.requests[index];
        for domain in &domains[index] {
            for start in domain.starts() {
                let end = start + domain.bytes;
                let mut next = domains.clone();
                let mut viable = true;
                for j in 0..next.len() {
                    if j == index || placement[j].is_some() {
                        continue;
                    }
                    let q = &self.requests[j];
                    let overlap =
                        q.lifetime.first <= r.lifetime.last && r.lifetime.first <= q.lifetime.last;
                    if self.edges[index][j] {
                        for (a, b) in element_spans(start, end) {
                            self.work += next[j].len();
                            next[j] = next[j].iter().flat_map(|&d| d.without(a, b)).collect();
                        }
                    } else if overlap {
                        self.work += next[j].len();
                        next[j] = next[j]
                            .iter()
                            .flat_map(|&d| d.without(start, end))
                            .collect();
                    }
                    if next[j].is_empty() {
                        viable = false;
                        break;
                    }
                }
                if viable {
                    placement[index] = Some((start, end));
                    if self.visit(next, placement) {
                        return true;
                    }
                    placement[index] = None;
                }
                if self.nodes >= self.node_budget || self.work >= self.propagation_budget {
                    return false;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Problem {
        ranges: Vec<(u32, u32)>,
        interleaved_offset: u32,
        requests: Vec<AllocationRequest>,
    }

    fn check(problem: &Problem, placement: &[(u32, u32)]) {
        assert_eq!(problem.requests.len(), placement.len());
        let owners: BTreeMap<_, _> = problem
            .requests
            .iter()
            .enumerate()
            .flat_map(|(i, r)| r.assignments.iter().map(move |&(root, _)| (root, i)))
            .collect();
        for (i, (&(a, b), r)) in placement.iter().zip(&problem.requests).enumerate() {
            assert!(problem.ranges.iter().any(|&(lo, hi)| lo <= a && b <= hi));
            assert_eq!(a % r.alignment.max(1), 0);
            assert!(
                a < IPU21_INTERLEAVED_MEMORY_BASE && b <= IPU21_INTERLEAVED_MEMORY_BASE
                    || a >= IPU21_INTERLEAVED_MEMORY_BASE
            );
            if r.class == MemoryClass::Ipu21Interleaved {
                assert!(a >= IPU21_INTERLEAVED_MEMORY_BASE + problem.interleaved_offset);
            }
            let region1 = a >= IPU21_INTERLEAVED_MEMORY_BASE;
            let size = if region1 {
                r.region1_stride
                    .map(|stride| stride * r.assignments.len() as u32)
                    .unwrap_or(r.bytes)
            } else {
                r.bytes
            };
            assert_eq!(b - a, size);
            if r.region1_stride.is_some() {
                assert_eq!(
                    a % if region1 {
                        IPU21_INTERLEAVED_ELEMENT_SIZE
                    } else {
                        TILE_MEMORY_ELEMENT_SIZE
                    },
                    0
                );
            }
            if r.lifetime.first == 0 || r.lifetime.last == u32::MAX {
                assert!(b <= HOST_SCRATCH_RANGE.0 || a >= HOST_SCRATCH_RANGE.1);
            }
            for (j, &(c, d)) in placement[..i].iter().enumerate() {
                let q = &problem.requests[j];
                if r.lifetime.first <= q.lifetime.last && q.lifetime.first <= r.lifetime.last {
                    assert!(b <= c || d <= a, "{i} overlaps {j}");
                }
            }
            for root in &r.conflicts {
                if let Some(&j) = owners.get(root)
                    && i != j
                {
                    let (c, d) = placement[j];
                    // Compare element IDs directly, independently of the domain subtraction.
                    for address in [a, b - 1] {
                        let element = if address < IPU21_INTERLEAVED_MEMORY_BASE {
                            TILE_MEMORY_ELEMENT_SIZE
                        } else {
                            IPU21_INTERLEAVED_ELEMENT_SIZE
                        };
                        assert!(
                            address / element < c / element
                                || address / element > (d - 1) / element
                        );
                    }
                    assert!(b <= c || d <= a);
                }
            }
        }
    }

    #[test]
    fn captured_fragmentation_and_capacity_are_distinct() {
        for (text, fits) in [
            (include_str!("testdata/fragmented.json"), true),
            (include_str!("testdata/overcapacity.json"), false),
        ] {
            let problem: Problem = serde_json::from_str(text).unwrap();
            let result = place(
                &problem.requests,
                &Arena::new(&problem.ranges, problem.interleaved_offset),
            );
            eprintln!(
                "capture fit={fits}: nodes={} deficit={} recovered={}",
                result.nodes,
                result.excess_live_bytes,
                result.placement.is_some()
            );
            assert_eq!(result.placement.is_some(), fits);
            if let Some(placement) = result.placement {
                check(&problem, &placement);
            } else {
                assert_eq!(result.excess_live_bytes, 16328);
            }
        }
    }

    #[test]
    fn address_branching_recovers_a_dense_seeded_bank_case() {
        // Generated with tools/placement_experiment.py, seed 20260917, eight
        // elements. All three greedy orders and 64 order-repair trials fail.
        let problem: Problem =
            serde_json::from_str(include_str!("testdata/bank-branching.json")).unwrap();
        let result = place(
            &problem.requests,
            &Arena::new(&problem.ranges, problem.interleaved_offset),
        );
        assert!(
            result.nodes > problem.requests.len(),
            "exercise backtracking"
        );
        check(&problem, &result.placement.unwrap());
        let exhausted = solve(
            &problem.requests,
            &Arena::new(&problem.ranges, problem.interleaved_offset),
            1,
            1000,
        );
        assert!(exhausted.placement.is_none());
        assert_eq!(
            exhausted.excess_live_bytes, 0,
            "budget exhaustion is not a capacity proof"
        );
    }

    #[test]
    fn propagation_preserves_every_aligned_legal_start() {
        for bytes in 1..24 {
            for first in (0..64).step_by(4) {
                let d = Domain {
                    first,
                    last: 64,
                    alignment: 4,
                    bytes,
                };
                for a in 0..90 {
                    let b = a + 7;
                    let remaining: Vec<_> = d.without(a, b).collect();
                    for start in (first..=64).step_by(4) {
                        assert_eq!(
                            remaining
                                .iter()
                                .any(|r| r.first <= start && start <= r.last),
                            start + bytes <= a || b <= start
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "external fixture corpus benchmark; set IPU_STACK_PLACEMENT_CORPUS"]
    fn fixture_corpus() {
        let directory = std::env::var_os("IPU_STACK_PLACEMENT_CORPUS").expect("fixture directory");
        let mut results = Vec::new();
        let mut paths = std::fs::read_dir(directory)
            .unwrap()
            .map(|p| p.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            let problem: Problem = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let now = std::time::Instant::now();
            let result = place(
                &problem.requests,
                &Arena::new(&problem.ranges, problem.interleaved_offset),
            );
            let seconds = now.elapsed().as_secs_f64();
            if let Some(placement) = &result.placement {
                check(&problem, placement);
            }
            results.push(
                serde_json::json!({"path":path,"nodes":result.nodes,"seconds":seconds,
                "placement":result.placement,"excess_live_bytes":result.excess_live_bytes}),
            );
        }
        println!("{}", serde_json::to_string(&results).unwrap());
    }
}
