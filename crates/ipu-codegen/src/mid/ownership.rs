//! Change value ownership and make the resulting operand movement explicit.

use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
    OperandIndexing, ProgramError, ResultSite, independent_copy_prefix,
};
use crate::tensor::{AmpOrder, ElementOrder, OwnerMap};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Input/result homes are base assignments for existing ownership groups, with
/// the constructor's relative rotations retained. Operator defaults choose a
/// working domain for newly produced storage, including any local embeddings
/// declared by the family. Input homes and explicit results take precedence;
/// aliases and Repeat sequences retain one consistent assignment per group.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OwnerChoices {
    #[serde(default)]
    pub inputs: BTreeMap<crate::ValueId, OwnerMap>,
    #[serde(default)]
    pub operators: BTreeMap<crate::OperationId, OwnerMap>,
    #[serde(default, with = "super::site::map")]
    pub results: BTreeMap<ResultSite, OwnerMap>,
}

impl OwnerChoices {
    /// A joint proposal relabels the affected input homes and operator domains.
    /// Result-specific choices remain more specific, with the same relabeling.
    pub(crate) fn remap_tiles(
        &mut self,
        inputs: impl Iterator<Item = crate::ValueId>,
        operators: impl Iterator<Item = crate::OperationId>,
        mapping: &[u16],
        tile_count: u16,
    ) -> Result<(), ProgramError> {
        if !mapping.iter().copied().eq(0..tile_count) {
            for input in inputs {
                self.inputs.entry(input).or_default();
            }
            for operator in operators {
                self.operators.entry(operator).or_default();
            }
        }
        crate::tensor::remap_owners(
            self.inputs
                .values_mut()
                .chain(self.operators.values_mut())
                .chain(self.results.values_mut()),
            mapping,
            tile_count,
        )
        .map_err(Into::into)
    }
}

impl MidProgram {
    /// Relabel an already constructed program before expansion, for explicit
    /// transfer captures and equivalence checks. Search changes Recipe choices.
    pub(crate) fn remap_tiles(&mut self, mapping: &[u16]) -> Result<(), ProgramError> {
        crate::tensor::remap_owners(
            self.values.iter_mut().map(|value| &mut value.owners),
            mapping,
            self.tile_count,
        )
        .map_err(Into::into)
    }

    pub(crate) fn apply_ownership(&mut self, choices: &OwnerChoices) -> Result<(), ProgramError> {
        let mut homes = BTreeMap::new();
        let mut input_groups = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        for input in &self.inputs {
            let value = &self.values[input.value.index() as usize];
            inputs.insert(value.origin);
            input_groups.insert(value.storage_group);
            if let Some(map) = choices.inputs.get(&value.origin) {
                agree(
                    &mut homes,
                    value.storage_group,
                    &map.canonicalized(self.tile_count),
                )?;
            }
        }
        if let Some(input) = choices.inputs.keys().find(|id| !inputs.contains(id)) {
            return Err(ProgramError::Invalid(format!(
                "owner choice names unknown input {input:?}"
            )));
        }
        let mut results = BTreeMap::new();
        for operation in self.walk_operations() {
            for (index, &value) in operation.results.iter().enumerate() {
                if let Some(site) = operation.result_site(index) {
                    results.insert(site, value);
                }
            }
        }
        let mut explicit = BTreeMap::new();
        for (site, map) in &choices.results {
            let map = map.canonicalized(self.tile_count);
            let value = results.get(site).ok_or_else(|| {
                ProgramError::Invalid(format!("owner choice is unavailable at {site:?}"))
            })?;
            let value = &self.values[value.index() as usize];
            if input_groups.contains(&value.storage_group) {
                let current = value.owners.with_rotation(0).canonicalized(self.tile_count);
                let home = homes.get(&value.storage_group).unwrap_or(&current);
                if home != &map {
                    return Err(ProgramError::Invalid(format!(
                        "result {site:?} shares input storage; choose the input's home consistently"
                    )));
                }
            }
            agree(&mut explicit, value.storage_group, &map)?;
        }
        homes.extend(explicit.clone());
        for operation in self.walk_operations() {
            // Repeat results alias carried storage rather than allocating a new
            // group. Their homes are supplied by the seed and body bindings.
            if matches!(operation.kind, MidOperationKind::Repeat(_)) {
                continue;
            }
            let Some(map) = operation.source.and_then(|id| choices.operators.get(&id)) else {
                continue;
            };
            for &result in &operation.results {
                let value = &self.values[result.index() as usize];
                let group = value.storage_group;
                if !input_groups.contains(&group) && !explicit.contains_key(&group) {
                    map.validate(1, self.tile_count)?;
                    let base = value
                        .owners
                        .with_rotation(0)
                        .canonicalized(self.tile_count)
                        .in_domain(map, self.tile_count)
                        .ok_or_else(|| {
                            ProgramError::Invalid(
                                "family owners exceed the chosen working domain".into(),
                            )
                        })?;
                    agree(&mut homes, group, &base.canonicalized(self.tile_count))?;
                }
            }
        }
        // Validate all assignments before replacing any value's ownership.
        let owners = self
            .values
            .iter()
            .map(|value| {
                let owners = if let Some(base) = homes.get(&value.storage_group) {
                    base.validate(1, self.tile_count)?;
                    base.shifted(i32::from(value.owners.rotation()), self.tile_count)
                        .ok_or_else(|| ProgramError::Invalid("invalid owner domain".into()))?
                } else {
                    value.owners.clone()
                };
                owners.validate(
                    value.tensor_type.format.layout.tiling.tile_count,
                    self.tile_count,
                )?;
                Ok(owners.canonicalized(self.tile_count))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        for (value, owners) in self.values.iter_mut().zip(owners) {
            value.owners = owners;
        }
        bind_owners(&mut self.operations, &mut self.values)?;
        Ok(())
    }
}

fn agree(
    homes: &mut BTreeMap<MidValueId, OwnerMap>,
    group: MidValueId,
    map: &OwnerMap,
) -> Result<(), ProgramError> {
    if let Some(previous) = homes.insert(group, map.clone())
        && previous != *map
    {
        return Err(ProgramError::Invalid(format!(
            "inconsistent owner choices for shared storage group {group:?}"
        )));
    }
    Ok(())
}

impl MidProgram {
    /// Rotate independent preparation sources, including inside Repeat bodies.
    pub(crate) fn with_disjoint_copy_sources(&self, checkpoints: bool) -> Option<Self> {
        self.with_owner_changes(|ops, values, required| {
            disjoint_sources(ops, values, required, self.tile_count, checkpoints)
        })
    }

    /// Group independent reductions without crossing a region boundary.
    pub(crate) fn with_overlapped_reductions(&self, limit: usize) -> Option<Self> {
        self.with_owner_changes(|ops, values, required| {
            overlap_reductions(ops, values, required, self.tile_count, limit)
        })
    }

    fn with_owner_changes(
        &self,
        mut change: impl FnMut(&mut Vec<MidOperation>, &mut [MidValue], &[MidValueId]) -> bool,
    ) -> Option<Self> {
        fn region(
            operations: &mut Vec<MidOperation>,
            values: &mut [MidValue],
            required: &[MidValueId],
            change: &mut impl FnMut(&mut Vec<MidOperation>, &mut [MidValue], &[MidValueId]) -> bool,
        ) -> bool {
            let mut changed = false;
            for op in operations.iter_mut() {
                if let MidOperationKind::Repeat(repeat) = &mut op.kind {
                    changed |= region(
                        &mut repeat.body.operations,
                        values,
                        &repeat.body.yields,
                        change,
                    );
                }
            }
            change(operations, values, required) || changed
        }
        let mut result = self.clone();
        region(
            &mut result.operations,
            &mut result.values,
            &result.outputs,
            &mut change,
        )
        .then_some(result)
    }
}

fn disjoint_sources(
    operations: &[MidOperation],
    values: &mut [MidValue],
    required: &[MidValueId],
    tile_count: u16,
    checkpoints: bool,
) -> bool {
    let storage_groups = values
        .iter()
        .map(|value| value.storage_group)
        .collect::<Vec<_>>();
    let mut uses = vec![0; values.len()];
    let mut sums = BTreeSet::new();
    for operation in operations.iter() {
        for input in operation.read_values() {
            uses[storage_groups[input.index() as usize].index() as usize] += 1;
        }
        if matches!(
            operation.kind,
            MidOperationKind::Compute(Compute::Sum { .. })
        ) {
            sums.extend(operation.results.iter().copied());
        }
    }
    let mut changed = false;
    let mut index = 0;
    while index < operations.len() {
        let count = independent_copy_prefix(&operations[index..], checkpoints, &storage_groups);
        let sources = operations[index..index + count]
            .iter()
            .filter_map(|operation| {
                let MidOperationKind::Copy { mapping, .. } = &operation.kind else {
                    return None;
                };
                let input = *operation.inputs.first()?;
                let value = &values[input.index() as usize];
                (mapping.view.is_some()
                    && sums.contains(&input)
                    && uses[value.storage_group.index() as usize] == 1
                    && !required.iter().any(|output| {
                        values[output.index() as usize].storage_group == value.storage_group
                    })
                    && value.tensor_type.format.layout.order
                        == ElementOrder::Amp(AmpOrder::TransposedLeft))
                .then_some(input)
            })
            .collect::<Vec<_>>();
        let total = sources
            .iter()
            .map(|&id| owner_count(values, id))
            .sum::<u32>();
        if sources.len() > 1 && total <= u32::from(tile_count) {
            let offset = values[sources[0].index() as usize].owners.rotation();
            changed |= rotate_owners(values, tile_count, sources, offset);
        }
        index += count.max(1);
    }
    changed
}

fn overlap_reductions(
    operations: &mut Vec<MidOperation>,
    values: &mut [MidValue],
    required: &[MidValueId],
    tile_count: u16,
    limit: usize,
) -> bool {
    let groups = values.iter().map(|v| v.storage_group).collect::<Vec<_>>();
    let overlaps = |a: &[MidValueId], b: &[MidValueId]| {
        a.iter().any(|a| {
            b.iter()
                .any(|b| groups[a.index() as usize] == groups[b.index() as usize])
        })
    };
    let conflicts = |a: &MidOperation, b: &MidOperation| {
        overlaps(&a.results, &b.inputs)
            || overlaps(&a.results, &b.results)
            || overlaps(&a.inputs, &b.results)
    };
    let mut eligible = BTreeSet::new();
    let mut used = BTreeSet::new();
    let mut forbidden = required
        .iter()
        .map(|id| groups[id.index() as usize])
        .collect::<BTreeSet<_>>();
    for op in operations.iter() {
        if matches!(op.kind, MidOperationKind::Compute(Compute::Sum { .. }))
            && op.results.len() == 1
        {
            eligible.insert(groups[op.results[0].index() as usize]);
        }
        for input in op.read_values() {
            let group = groups[input.index() as usize];
            used.insert(group);
            if !matches!(op.kind, MidOperationKind::Copy { .. }) {
                forbidden.insert(group);
            }
        }
    }
    eligible.retain(|group| used.contains(group) && !forbidden.contains(group));
    let eligible = |op: &MidOperation| {
        op.results.len() == 1
            && matches!(op.kind, MidOperationKind::Compute(Compute::Sum { .. }))
            && eligible.contains(&groups[op.results[0].index() as usize])
    };
    let mut changed = false;
    let mut start = 0;
    while start < operations.len() {
        if !eligible(&operations[start]) {
            start += 1;
            continue;
        }
        let mut selected = vec![start];
        let owners = |op: &MidOperation| owner_count(values, op.results[0]);
        let mut total = owners(&operations[start]);
        for next in start + 1..operations.len() {
            let operation = &operations[next];
            if selected.len() >= limit
                || !(matches!(
                    operation.kind,
                    MidOperationKind::Copy { .. } | MidOperationKind::Compute(Compute::Sum { .. })
                ) || matches!(&operation.kind, MidOperationKind::Compute(Compute::Product(product)) if product.output_aliases.is_empty()))
                || selected
                    .iter()
                    .any(|&index| conflicts(&operations[index], operation))
            {
                break;
            }
            if eligible(operation) {
                total += owners(operation);
                if total > u32::from(tile_count) {
                    break;
                }
                selected.push(next);
            }
        }
        if selected.len() < 2 {
            start += 1;
            continue;
        }
        let insertion = selected.last().copied().unwrap() + 1 - selected.len();
        let mut sums = Vec::new();
        for index in selected.into_iter().rev() {
            sums.push(operations.remove(index));
        }
        sums.reverse();
        let offset = values[sums[0].results[0].index() as usize]
            .owners
            .rotation();
        rotate_owners(
            values,
            tile_count,
            sums.iter().map(|sum| sum.results[0]),
            offset,
        );
        start = insertion + sums.len();
        operations.splice(insertion..insertion, sums);
        changed = true;
    }
    changed
}

fn owner_count(values: &[MidValue], value: MidValueId) -> u32 {
    u32::from(
        values[value.index() as usize]
            .tensor_type
            .format
            .layout
            .tiling
            .tile_count,
    )
}

/// Rotate complete alias groups together; callers check the combined owner count.
fn rotate_owners(
    values: &mut [MidValue],
    tile_count: u16,
    sources: impl IntoIterator<Item = MidValueId>,
    mut offset: u16,
) -> bool {
    let mut changed = false;
    for source in sources {
        let group = values[source.index() as usize].storage_group;
        for alias in values
            .iter_mut()
            .filter(|alias| alias.storage_group == group)
        {
            let owners = alias.owners.with_rotation(offset);
            changed |= alias.owners != owners;
            alias.owners = owners;
        }
        offset = ((u32::from(offset) + owner_count(values, source)) % u32::from(tile_count)) as u16;
    }
    changed
}

/// Preserve each compute operand's declared result ownership with explicit
/// copies after homes change. Aliased inputs include accumulation/donation
/// bindings beyond the callable operands. Distributed sums select their own
/// contributor traffic; ordinary copies already declare both endpoints.
pub(crate) fn bind_owners(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
) -> Result<(), ProgramError> {
    let mut rewritten = Vec::with_capacity(operations.len());
    for mut operation in std::mem::take(operations) {
        match &mut operation.kind {
            MidOperationKind::Repeat(repeat) => {
                let bindings = operation.inputs.iter().copied().chain(
                    repeat
                        .iterated_inputs
                        .iter()
                        .filter_map(|sequence| sequence.first().copied()),
                );
                for (&argument, binding) in repeat.body.arguments.iter().zip(bindings) {
                    let source = &values[binding.index() as usize];
                    let owners = source.owners.clone();
                    let group = source.storage_group;
                    values[argument.index() as usize].owners = owners;
                    values[argument.index() as usize].storage_group = group;
                }
                for sequence in &repeat.iterated_inputs {
                    if let Some(&first) = sequence.first()
                        && sequence.iter().any(|value| {
                            values[value.index() as usize].owners
                                != values[first.index() as usize].owners
                        })
                    {
                        return Err(ProgramError::Invalid(
                            "Repeat sequence members require one owner assignment".into(),
                        ));
                    }
                }
                for (&result, &input) in operation.results.iter().zip(&operation.inputs) {
                    if values[result.index() as usize].owners
                        != values[input.index() as usize].owners
                    {
                        return Err(ProgramError::Invalid(
                            "Repeat results and carried inputs require the same owners".into(),
                        ));
                    }
                }
                bind_owners(&mut repeat.body.operations, values)?;
                for (index, (yielded, &input)) in repeat
                    .body
                    .yields
                    .iter_mut()
                    .zip(&operation.inputs)
                    .enumerate()
                {
                    let owners = values[input.index() as usize].owners.clone();
                    if values[yielded.index() as usize].owners != owners {
                        let site = operation
                            .site
                            .clone()
                            .unwrap_or_else(|| super::LocalSite::from("repeat"));
                        *yielded = append_owner_copy(
                            operation.source,
                            Some(site.child("owner.yield").at(index as u32)),
                            *yielded,
                            owners,
                            values,
                            &mut repeat.body.operations,
                        );
                    }
                }
            }
            MidOperationKind::Compute(compute) => {
                for (index, input) in operation.inputs.iter_mut().enumerate() {
                    let result = match compute {
                        Compute::Kernel { operands, .. } => {
                            operands.get(index).map(|operand| match operand {
                                OperandIndexing::Elementwise { result } => *result,
                                OperandIndexing::Local(_) => 0,
                            })
                        }
                        Compute::Product(product) => (index < product.operands.len()).then_some(0),
                        Compute::Sum { .. } => None,
                    };
                    let mut result = result;
                    for &(output, alias_input) in compute.output_aliases() {
                        if alias_input != index {
                            continue;
                        }
                        if result.is_some_and(|result| {
                            values[operation.results[result].index() as usize].owners
                                != values[operation.results[output].index() as usize].owners
                        }) {
                            return Err(ProgramError::Invalid(format!(
                                "operation {:?} requires incompatible homes for input {index}",
                                operation.source
                            )));
                        }
                        result = Some(output);
                    }
                    let Some(result) = result else {
                        continue;
                    };
                    let owners = values[operation.results[result].index() as usize]
                        .owners
                        .clone();
                    if values[input.index() as usize].owners != owners {
                        *input = append_owner_copy(
                            operation.source,
                            operation
                                .site
                                .as_ref()
                                .map(|site| site.child("owner.copy").at(index as u32)),
                            *input,
                            owners,
                            values,
                            &mut rewritten,
                        );
                    }
                }
            }
            _ => {}
        }
        rewritten.push(operation);
    }
    *operations = rewritten;
    Ok(())
}

fn append_owner_copy(
    source: Option<crate::OperationId>,
    site: Option<super::LocalSite>,
    input: MidValueId,
    owners: OwnerMap,
    values: &mut Vec<MidValue>,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    let mut value = values[input.index() as usize].clone();
    value.id = MidValueId::from_index(values.len() as u32);
    value.owners = owners;
    value.storage_group = value.id;
    let id = value.id;
    values.push(value);
    operations.push(MidOperation {
        source,
        site,
        inputs: vec![input],
        results: vec![id],
        kind: MidOperationKind::Copy {
            policy: crate::CopyPolicy::Automatic,
            packing: crate::PackingPolicy::Automatic,
            mapping: CoordinateMapping::default(),
            reuse_local: true,
        },
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
    });
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::MemoryPeaks;
    use crate::graph::{ComputeGraph, ValueId};
    use crate::kernel::TileKernelSpec;
    use crate::mid::{MidRegion, MidRepeat, ReductionStaging};
    use crate::tensor::{AxisFactorView, Layout, Precision, TensorType};

    #[test]
    fn operator_domains_preserve_local_embeddings_and_relocate_donation_buffers() {
        use crate::GraphInputKind;
        use crate::mid::MidInput;
        use crate::tensor::OwnerMap;

        let id = MidValueId::from_index;
        let mut program = MidProgram {
            tile_count: 8,
            values: [5, 1, 5]
                .into_iter()
                .enumerate()
                .map(|(index, tile)| MidValue {
                    id: id(index as u32),
                    origin: ValueId::from_index(index as u32),
                    storage_group: id(index as u32),
                    owners: OwnerMap::embedded(vec![tile]),
                    tensor_type: TensorType::new([4, 16], Precision::F16, Layout::row_sharded(1)),
                })
                .collect(),
            inputs: (0..2)
                .map(|index| MidInput {
                    name: format!("input{index}"),
                    kind: GraphInputKind::Host,
                    value: id(index),
                })
                .collect(),
            outputs: vec![id(2)],
            operations: vec![MidOperation {
                site: None,
                source: None,
                inputs: vec![id(0), id(1)],
                results: vec![id(2)],
                kind: MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::Gelu,
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: vec![(0, 1)],
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            }],
            ..MidProgram::default()
        };
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("x", [4, 16]).unwrap();
        graph.gelu(input).unwrap();
        let source = graph.operations()[0].id;
        program.operations[0].source = Some(source);
        program.operations[0].site = Some("gelu".into());
        let mut choices = OwnerChoices::default();
        choices
            .operators
            .insert(source, OwnerMap::embedded((0..8).rev().collect::<Vec<_>>()));
        program.apply_ownership(&choices).unwrap();
        program.validate().unwrap();
        let graph = crate::low::expand::expand_tiles(&program, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let placement = crate::place(&low).unwrap();
        let donation = program.operations.last().unwrap().inputs[1];
        let donation = low.value_shards(donation)[0];
        let output = low.value_shards(id(2))[0];
        assert_eq!(
            low.shards[low.value_shards(id(1))[0].index() as usize].tile,
            1
        );
        // The family's owner 5 maps to tile 2; it does not become owner 0 of
        // the new domain. Both the callable input and donated storage follow it.
        assert_eq!(low.shards[donation.index() as usize].tile, 2);
        assert_eq!(low.shards[output.index() as usize].tile, 2);
        assert_eq!(
            placement.shard_addresses[&donation],
            placement.shard_addresses[&output]
        );
    }

    #[test]
    fn independent_copy_roots_rotate_without_moving_partials_or_shared_results() {
        let mut program = MidProgram {
            tile_count: 16,
            ..MidProgram::default()
        };
        let mut layout = Layout::row_sharded(4);
        layout.order = ElementOrder::Amp(AmpOrder::TransposedLeft);
        for id in 0..6 {
            program.values.push(MidValue {
                id: MidValueId::from_index(id),
                owners: crate::tensor::OwnerMap::default(),
                tensor_type: TensorType::new([1, 16, 16], Precision::F16, layout.clone()),
                origin: ValueId::from_index(id),
                storage_group: MidValueId::from_index(id),
            });
        }
        let operation = |input, output, primitive| MidOperation {
            site: None,
            source: None,
            inputs: vec![MidValueId::from_index(input)],
            results: vec![MidValueId::from_index(output)],
            kind: primitive,
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let copy = |input, output| {
            operation(
                input,
                output,
                MidOperationKind::Copy {
                    policy: crate::CopyPolicy::Automatic,
                    packing: crate::PackingPolicy::Automatic,
                    mapping: CoordinateMapping {
                        offsets: vec![],
                        view: Some(AxisFactorView {
                            split_axis: 2,
                            merge_axis: 0,
                            factor: 1,
                            reversed: false,
                        }),
                    },
                    reuse_local: false,
                },
            )
        };
        program.operations = vec![
            operation(
                4,
                0,
                MidOperationKind::Compute(Compute::Sum {
                    axis: 0,
                    staging: ReductionStaging::Complete,
                }),
            ),
            operation(
                5,
                1,
                MidOperationKind::Compute(Compute::Sum {
                    axis: 0,
                    staging: ReductionStaging::Complete,
                }),
            ),
            copy(0, 2),
            copy(1, 3),
        ];
        let groups = (0..6).map(MidValueId::from_index).collect::<Vec<_>>();
        assert_eq!(
            independent_copy_prefix(&program.operations[2..], true, &groups),
            2
        );
        let rotated = program.with_disjoint_copy_sources(true).unwrap();
        assert_eq!(
            rotated
                .values
                .iter()
                .map(|v| v.owners.rotation())
                .collect::<Vec<_>>(),
            [0, 4, 0, 0, 0, 0]
        );
        assert!(
            program
                .values
                .iter()
                .all(|v| v.owners == crate::tensor::OwnerMap::default())
        );
        let mut delayed = program.clone();
        delayed.operations.insert(1, copy(4, 5));
        let overlapped = delayed.with_overlapped_reductions(2).unwrap();
        assert_eq!(
            overlapped.operations[0].results,
            [MidValueId::from_index(5)]
        );
        assert_eq!(
            overlapped.operations[1].results,
            [MidValueId::from_index(0)]
        );
        assert_eq!(
            overlapped.operations[2].results,
            [MidValueId::from_index(1)]
        );
        assert_eq!(overlapped.values[1].owners.rotation(), 4);
        // The same transformations must work in the repeated encoder, while
        // treating its yields as externally live values.
        let wrap = |mut body: MidProgram, yields: Vec<MidValueId>| {
            let operations = std::mem::take(&mut body.operations);
            body.outputs.clear();
            body.operations.push(MidOperation {
                site: None,
                source: None,
                inputs: vec![],
                results: vec![],
                kind: MidOperationKind::Repeat(MidRepeat {
                    count: 27,
                    carried_inputs: 0,
                    invariant_inputs: 0,
                    iterated_inputs: vec![],
                    body: MidRegion {
                        arguments: vec![],
                        operations,
                        yields,
                        estimated_cycles: 0,
                        peak_memory: MemoryPeaks::default(),
                    },
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            });
            body
        };
        let repeated = wrap(program.clone(), vec![]);
        let rotated_repeat = repeated.with_disjoint_copy_sources(true).unwrap();
        assert_eq!(rotated_repeat.values, rotated.values);
        let repeated = wrap(delayed.clone(), vec![]);
        let grouped_repeat = repeated.with_overlapped_reductions(2).unwrap();
        let MidOperationKind::Repeat(repeat) = &grouped_repeat.operations[0].kind else {
            unreachable!()
        };
        assert_eq!(repeat.body.operations, overlapped.operations);
        assert_eq!(grouped_repeat.values, overlapped.values);
        assert!(
            wrap(program.clone(), vec![MidValueId::from_index(1)])
                .with_disjoint_copy_sources(true)
                .is_none()
        );
        assert!(
            wrap(delayed.clone(), vec![MidValueId::from_index(1)])
                .with_overlapped_reductions(2)
                .is_none()
        );

        assert!(delayed.with_overlapped_reductions(1).is_none());
        delayed.operations[1].kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Gelu,
            operands: Vec::new(),
            output_aliases: vec![(0, 0)],
        });
        assert!(delayed.with_overlapped_reductions(2).is_none());
        delayed.operations[1] = copy(5, 4); // Would overwrite a delayed partial.
        assert!(delayed.with_overlapped_reductions(2).is_none());

        program.outputs.push(MidValueId::from_index(1));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId::from_index(1);
        program.outputs.push(MidValueId::from_index(3));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId::from_index(3);
        program.operations.push(copy(1, 2));
        assert!(program.with_disjoint_copy_sources(true).is_none());
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), copy(2, 3)], false, &groups),
            1
        );
        let mut boundary = copy(1, 3);
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("x", [16]).unwrap();
        graph.gelu(input).unwrap();
        boundary.source = Some(graph.operations()[0].id);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], true, &groups),
            1
        );
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &groups),
            2
        );
        let mut aliases = groups.clone();
        aliases[1] = MidValueId::from_index(2);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &aliases),
            1
        );
        aliases = groups.clone();
        aliases[3] = MidValueId::from_index(0);
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), boundary.clone()], false, &aliases),
            1
        );
        program.operations.truncate(4);
        program.operations[3] = boundary;
        assert!(program.with_disjoint_copy_sources(true).is_none());
        assert!(program.with_disjoint_copy_sources(false).is_some());
    }
}
