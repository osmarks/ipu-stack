//! DP over a fixed high-operation order. A state contains only live boundary
//! representations; a path label contains historical cost and a predecessor.
//! Parameters begin with compact reservations. Their first consumer can replace
//! them; selected resident storage is charged against all earlier scratch peaks.

use super::budget::Memory;
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
    memory: Memory,
}

pub(super) struct Search<'a> {
    high: &'a HighGraph,
    settings: &'a PipelineConfig,
    limits: SearchLimits,
    /// Birth and last-use boundaries for representations, not storage lifetime.
    lifetimes: BTreeMap<ValueId, (usize, usize)>,
    states: Vec<HashMap<LiveValues, Vec<Rc<Path>>, foldhash::fast::FixedState>>,
    initial: Candidate,
    /// Unconstrained parameter -> first consumer boundary. Defaults are ranking
    /// reservations until this boundary, not hard per-tile capacity charges.
    first_use: BTreeMap<ValueId, usize>,
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
        let first_use = high
            .inputs()
            .iter()
            .filter(|input| {
                input.kind == GraphInputKind::Parameter
                    && layouts.get(&input.value).and_then(Option::as_ref).is_none()
            })
            .filter_map(|input| {
                high.operations()
                    .iter()
                    .position(|op| high.operation_inputs(op).any(|id| id == input.value))
                    .map(|position| (input.value, position))
            })
            .collect();
        let mut lifetimes = BTreeMap::new();
        for input in high.inputs() {
            lifetimes.insert(input.value, (0, 0));
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
        // Preserve declared resident inputs even if this graph never reads them.
        for input in high
            .inputs()
            .iter()
            .filter(|i| i.kind == GraphInputKind::Parameter)
        {
            if lifetimes[&input.value].1 == 0 {
                lifetimes.get_mut(&input.value).unwrap().1 = exit + 1;
            }
        }
        let mut live = LiveValues::new();
        for input in high
            .inputs()
            .iter()
            .filter(|input| lifetimes[&input.value].1 > 0)
        {
            let mut format = match settings.inputs.get(&input.value) {
                Some(format) => format.clone(),
                None if input.kind == GraphInputKind::Parameter => {
                    super::parameters::default_format(high, input.value, settings)?
                }
                None => return Err(PlanningError::UnassignedLayout(input.value)),
            };
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
            .validate()
            .map_err(|_| PlanningError::InvalidFragment("initial storage"))?;
        let mut memory = Memory::default();
        let (_, peak) = crate::estimate::analyze_observed(
            settings.target,
            &initial.graph,
            &BTreeMap::new(),
            &mut memory,
        )
        .ok_or(PlanningError::InvalidFragment("initial storage"))?;
        initial.graph.peak_memory = peak;
        let mut states = (0..=exit).map(|_| HashMap::default()).collect::<Vec<_>>();
        states[0].insert(
            live,
            vec![Rc::new(Path {
                previous: None,
                candidate: None,
                cycles: 0,
                peak,
                memory,
            })],
        );
        Ok(Self {
            high,
            settings,
            limits,
            lifetimes,
            states,
            initial,
            first_use,
        })
    }

    pub fn selectable_parameters(&self, position: usize) -> Vec<ValueId> {
        self.first_use
            .iter()
            .filter_map(|(&id, &first)| (first == position).then_some(id))
            .collect()
    }

    pub fn take_states(&mut self, position: usize) -> Vec<State> {
        let mut states = std::mem::take(&mut self.states[position])
            .into_iter()
            .filter(|(_, paths)| !paths.is_empty())
            .map(|(live, paths)| State { live, paths })
            .collect::<Vec<_>>();
        states.sort_by_key(|state| state.paths.iter().map(|p| (p.cycles, p.peak.total)).min());
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
        // Aliases need allocation identities in the boundary state and memory
        // composition. Do not silently cost shared storage as independent values.
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
        if imports.keys().ne(state.live.keys())
            || state.live.iter().any(|(id, value)| {
                imports.get(id) != Some(value) && self.first_use.get(id) != Some(&position)
            })
        {
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
            .validate()
            .map_err(|_| PlanningError::InvalidFragment("invalid mid fragment"))?;
        let mut local = Memory::default();
        let (cycles, peak) = crate::estimate::analyze_observed(
            self.settings.target,
            &candidate.graph,
            &BTreeMap::new(),
            &mut local,
        )
        .ok_or(PlanningError::InvalidFragment("uncostable mid fragment"))?;
        candidate.graph.estimated_cycles = cycles.total;
        candidate.graph.estimated_exchange_cycles = cycles.exchange;
        candidate.graph.peak_memory = peak;
        let end = candidate.end;
        let candidate = Rc::new(candidate);
        let paths = self.states[end].entry(live).or_default();
        for old in &state.paths {
            let mut memory = local.clone();
            for (peak, old) in memory.nonresident.iter_mut().zip(&old.memory.nonresident) {
                peak.include(*old);
            }
            memory.nonresident_total = memory.nonresident_total.max(old.memory.nonresident_total);
            // Later consumers can increase padding/alignment requirements even
            // when the resident layout is unchanged. Preserve those requirements.
            for (&id, old_bytes) in &old.memory.parameters {
                if self.first_use.get(&id) == Some(&position) {
                    continue;
                }
                let bytes = memory
                    .parameters
                    .entry(id)
                    .or_insert_with(|| old_bytes.clone());
                for (bytes, old) in bytes.iter_mut().zip(old_bytes) {
                    bytes.standard = bytes.standard.max(old.standard);
                    bytes.interleaved = bytes.interleaved.max(old.interleaved);
                }
            }
            let peak = memory.peak(|_| true);
            // Undecided layouts impose no per-tile lower bound: they may move
            // away from that tile entirely. Their default footprint ranks only.
            let mut lower =
                memory.peak(|id| self.first_use.get(&id).is_none_or(|&first| first < end));
            let resident_total = memory
                .parameters
                .iter()
                .map(|(id, bytes)| {
                    if self.first_use.get(id).is_some_and(|&first| first >= end) {
                        let value =
                            &candidate.graph.values[candidate.bindings[id].index() as usize];
                        value.tensor_type.shape.elements()
                            * value.tensor_type.format.precision.bytes()
                    } else {
                        bytes.iter().map(|bytes| bytes.total()).sum()
                    }
                })
                .sum::<u64>();
            lower.total = lower.total.max(
                memory
                    .nonresident_total
                    .saturating_add(resident_total)
                    .div_ceil(u64::from(self.settings.tile_count)),
            );
            if !lower.fits_with_budget(
                self.settings.target,
                self.settings.standard_memory_reservation_bytes,
                self.settings.tile_memory_budget_bytes,
            ) {
                continue;
            }
            let cycles = old.cycles.saturating_add(candidate.graph.estimated_cycles);
            if paths
                .iter()
                .any(|p| p.cycles <= cycles && p.memory.dominates(&memory))
            {
                continue;
            }
            paths.retain(|p| !(cycles <= p.cycles && memory.dominates(&p.memory)));
            paths.push(Rc::new(Path {
                previous: Some(Rc::clone(old)),
                candidate: Some(Rc::clone(&candidate)),
                cycles,
                peak,
                memory,
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
            // Planning order does not imply loading order. Replace the original
            // graph input's format; it remains resident from execution entry.
            for input in &candidate.graph.inputs {
                if input.kind == GraphInputKind::Parameter {
                    let value = &candidate.graph.values[input.value.index() as usize];
                    let destination = &mut graph.values[bindings[&value.origin].index() as usize];
                    destination.tensor_type = value.tensor_type.clone();
                    destination.owners = value.owners.clone();
                }
            }
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
            .refresh_estimates(self.settings.target)
            .ok_or(PlanningError::InvalidFragment("selected graph"))?;
        if !graph.peak_memory.fits_with_budget(
            self.settings.target,
            self.settings.standard_memory_reservation_bytes,
            self.settings.tile_memory_budget_bytes,
        ) {
            return Err(PlanningError::NoPlan(exit));
        }
        Ok(graph)
    }
}
