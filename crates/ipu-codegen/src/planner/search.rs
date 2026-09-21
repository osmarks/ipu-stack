//! DP over a fixed high-operation order. A state contains only live boundary
//! representations; a path label contains historical cost and a predecessor.
//! This first version fixes parameter representations and disallows boundary
//! aliasing. Consequently every fragment is costed with the same resident set,
//! and maxima of its complete-context peaks compose without double-counting.

use super::candidates::{BoundaryValue, Candidate, LiveValues};
use super::{BoundaryLayouts, PlanningError, PlanningResult, SearchLimits};
use crate::config::PipelineConfig;
use crate::estimate::MemoryPeaks;
use crate::graph::{GraphInputKind, HighGraph, ValueId};
use crate::mid::{MidGraph, MidValueId};
use crate::tensor::{OwnerMap, TensorType};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

#[cfg(test)]
#[path = "property_tests.rs"]
mod property_tests;

pub(super) struct State {
    pub live: LiveValues,
    paths: Vec<Rc<Path>>,
}

struct Path {
    previous: Option<Rc<Path>>,
    candidate: Option<Rc<Candidate>>,
    cycles: u64,
    peak: MemoryPeaks,
}

pub(super) struct Search<'a> {
    high: &'a HighGraph,
    settings: &'a PipelineConfig,
    limits: SearchLimits,
    /// Birth boundary and last-use boundary. Parameters extend to graph exit.
    lifetimes: BTreeMap<ValueId, (usize, usize)>,
    states: Vec<HashMap<LiveValues, Vec<Rc<Path>>, foldhash::fast::FixedState>>,
    initial: Candidate,
}

fn memory(peak: MemoryPeaks) -> [u64; 4] {
    [
        peak.standard,
        peak.interleaved,
        peak.total,
        peak.maximum_standard_allocation,
    ]
}

impl<'a> Search<'a> {
    pub fn new(
        high: &'a HighGraph,
        layouts: &BoundaryLayouts,
        settings: &'a PipelineConfig,
        limits: SearchLimits,
    ) -> PlanningResult<Self> {
        if limits.states_per_boundary == Some(0) || limits.paths_per_state == Some(0) {
            return Err(PlanningError::InvalidFragment(
                "search limits must be positive",
            ));
        }
        let exit = high.operations().len();
        let mut lifetimes = BTreeMap::new();
        for input in high.inputs() {
            lifetimes.insert(
                input.value,
                (
                    0,
                    if input.kind == GraphInputKind::Parameter {
                        exit + 1
                    } else {
                        0
                    },
                ),
            );
        }
        for (index, operation) in high.operations().iter().enumerate() {
            for input in high.operation_inputs(operation) {
                let lifetime = lifetimes
                    .get_mut(&input)
                    .ok_or(PlanningError::InvalidFragment("high use before definition"))?;
                lifetime.1 = lifetime.1.max(index + 1);
            }
            for &output in &operation.results {
                lifetimes.insert(output, (index + 1, index + 1));
            }
        }
        for output in high.outputs() {
            lifetimes
                .get_mut(output)
                .ok_or(PlanningError::InvalidFragment("undefined graph output"))?
                .1 = exit + 1;
        }
        let mut live = LiveValues::new();
        for input in high
            .inputs()
            .iter()
            .filter(|input| lifetimes[&input.value].1 > 0)
        {
            let mut format = settings
                .inputs
                .get(&input.value)
                .cloned()
                .ok_or(PlanningError::UnassignedLayout(input.value))?;
            if let Some(layout) = layouts.get(&input.value).and_then(Option::as_ref) {
                format.layout = layout.clone();
            }
            live.insert(
                input.value,
                BoundaryValue {
                    tensor: TensorType {
                        shape: input.shape.clone(),
                        format,
                    },
                    owners: OwnerMap::default(),
                },
            );
        }
        let mut initial = Candidate::inputs(high, &live, settings.tile_count, 0);
        initial.graph.outputs = initial.bindings.values().copied().collect();
        initial
            .graph
            .refresh_estimates()
            .ok_or(PlanningError::InvalidFragment("initial storage"))?;
        let peak = initial.graph.peak_memory;
        let mut states = (0..=exit).map(|_| HashMap::default()).collect::<Vec<_>>();
        states[0].insert(
            live,
            vec![Rc::new(Path {
                previous: None,
                candidate: None,
                cycles: 0,
                peak,
            })],
        );
        Ok(Self {
            high,
            settings,
            limits,
            lifetimes,
            states,
            initial,
        })
    }

    pub fn take_states(&mut self, position: usize) -> Vec<State> {
        let mut states = std::mem::take(&mut self.states[position])
            .into_iter()
            .filter(|(_, paths)| !paths.is_empty())
            .map(|(live, paths)| State { live, paths })
            .collect::<Vec<_>>();
        states.sort_by_key(|state| state.paths.iter().map(|p| p.cycles).min());
        if let Some(limit) = self.limits.states_per_boundary {
            states.truncate(limit);
        }
        for state in &mut states {
            state.paths.sort_by_key(|p| (p.cycles, p.peak.total));
            if let Some(limit) = self.limits.paths_per_state {
                state.paths.truncate(limit);
            }
        }
        states
    }

    pub fn extend(
        &mut self,
        position: usize,
        state: &State,
        mut candidate: Candidate,
    ) -> PlanningResult<()> {
        if candidate.end <= position || candidate.end > self.high.operations().len() {
            return Err(PlanningError::InvalidFragment(
                "edge does not advance within graph",
            ));
        }
        // Until alias identities are part of the boundary key, no candidate may
        // export an alias or change a parameter representation behind the DP.
        if candidate
            .graph
            .operations
            .iter()
            .any(|op| !op.output_aliases.is_empty())
        {
            return Err(PlanningError::Unimplemented("aliased DP fragments"));
        }
        let imports = candidate
            .graph
            .inputs
            .iter()
            .map(|input| {
                let value = &candidate.graph.values[input.value.index() as usize];
                (
                    value.origin,
                    BoundaryValue {
                        tensor: value.tensor_type.clone(),
                        owners: value.owners.clone(),
                    },
                )
            })
            .collect::<LiveValues>();
        if imports != state.live {
            return Err(PlanningError::InvalidFragment(
                "fragment imports differ from boundary state",
            ));
        }
        let mut live = LiveValues::new();
        for (&origin, &(birth, death)) in &self.lifetimes {
            if birth <= candidate.end && death > candidate.end {
                let id = candidate
                    .bindings
                    .get(&origin)
                    .ok_or(PlanningError::InvalidFragment("missing live export"))?;
                let value = &candidate.graph.values[id.index() as usize];
                live.insert(
                    origin,
                    BoundaryValue {
                        tensor: value.tensor_type.clone(),
                        owners: value.owners.clone(),
                    },
                );
            }
        }
        candidate.graph.outputs = live.keys().map(|id| candidate.bindings[id]).collect();
        candidate
            .graph
            .refresh_estimates()
            .ok_or(PlanningError::InvalidFragment("uncostable mid fragment"))?;
        let end = candidate.end;
        let candidate = Rc::new(candidate);
        let paths = self.states[end].entry(live).or_default();
        for old in &state.paths {
            let local = candidate.graph.peak_memory;
            let peak = MemoryPeaks {
                standard: old.peak.standard.max(local.standard),
                interleaved: old.peak.interleaved.max(local.interleaved),
                total: old.peak.total.max(local.total),
                maximum_standard_allocation: old
                    .peak
                    .maximum_standard_allocation
                    .max(local.maximum_standard_allocation),
                // Row sharing/code residency need whole-program accounting.
                // They are not treated as a local, composable memory resource.
                exchange_rows: 0,
            };
            if !peak.fits_ipu21_with_budget(
                self.settings.standard_memory_reservation_bytes,
                self.settings.tile_memory_budget_bytes,
            ) {
                continue;
            }
            let cycles = old.cycles.saturating_add(candidate.graph.estimated_cycles);
            let dominates = |a_cycles, a_peak, b_cycles, b_peak| {
                a_cycles <= b_cycles
                    && memory(a_peak)
                        .into_iter()
                        .zip(memory(b_peak))
                        .all(|(a, b)| a <= b)
            };
            if paths
                .iter()
                .any(|p| dominates(p.cycles, p.peak, cycles, peak))
            {
                continue;
            }
            paths.retain(|p| !dominates(cycles, peak, p.cycles, p.peak));
            paths.push(Rc::new(Path {
                previous: Some(Rc::clone(old)),
                candidate: Some(Rc::clone(&candidate)),
                cycles,
                peak,
            }));
        }
        Ok(())
    }

    pub fn finish(mut self) -> PlanningResult<MidGraph> {
        let exit = self.high.operations().len();
        let best = self
            .take_states(exit)
            .into_iter()
            .flat_map(|state| state.paths)
            .min_by_key(|p| (p.cycles, p.peak.total))
            .ok_or(PlanningError::NoPlan(exit))?;
        let mut path = Vec::new();
        let mut cursor = best;
        while let Some(previous) = &cursor.previous {
            path.push(Rc::clone(cursor.candidate.as_ref().unwrap()));
            cursor = Rc::clone(previous);
        }
        let mut graph = self.initial.graph;
        let mut bindings = self.initial.bindings;
        for candidate in path.into_iter().rev() {
            let imports = candidate
                .graph
                .inputs
                .iter()
                .map(|input| bindings[&candidate.graph.values[input.value.index() as usize].origin])
                .collect::<Vec<MidValueId>>();
            let outputs = crate::mid::append_fragment(
                &candidate.graph,
                &imports,
                &OwnerMap::default(),
                None,
                None,
                graph.tile_count,
                &mut graph.values,
                &mut graph.operations,
            )
            .ok_or(PlanningError::InvalidFragment(
                "concatenating selected fragments",
            ))?;
            for (&local, output) in candidate.graph.outputs.iter().zip(outputs) {
                bindings.insert(
                    candidate.graph.values[local.index() as usize].origin,
                    output,
                );
            }
        }
        graph.outputs = self.high.outputs().iter().map(|id| bindings[id]).collect();
        graph
            .refresh_estimates()
            .ok_or(PlanningError::InvalidFragment("selected graph"))?;
        if !graph.peak_memory.fits_ipu21_with_budget(
            self.settings.standard_memory_reservation_bytes,
            self.settings.tile_memory_budget_bytes,
        ) {
            return Err(PlanningError::NoPlan(exit));
        }
        Ok(graph)
    }
}
