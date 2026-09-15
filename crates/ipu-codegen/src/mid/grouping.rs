//! Propose independent work and result homes, then apply only the requested order.
//!
//! Ownership uses the ordinary result-home policy. Grouping never moves a value
//! to another tile, and it never crosses a Repeat boundary.
use super::{
    Compute, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId, ProgramError,
    ResultSite, WorkSite, independent_copy_prefix,
};
use crate::tensor::{AmpOrder, ElementOrder, OwnerMap};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReductionGroup {
    pub members: Vec<WorkSite>,
}

/// A joint search proposal. The homes enter Recipe's existing ownership policy;
/// only the named reduction order is retained as a grouping choice.
#[derive(Clone, Debug, Default)]
pub(crate) struct GroupProposal {
    pub reductions: Vec<ReductionGroup>,
    pub homes: BTreeMap<ResultSite, OwnerMap>,
}

impl MidProgram {
    pub(crate) fn propose_reduction_groups(&self, limit: usize) -> GroupProposal {
        let mut proposal = GroupProposal::default();
        if limit < 2 {
            return proposal;
        }
        visit_regions(
            &self.operations,
            &self.outputs,
            &mut |operations, required| {
                let eligible = reduction_outputs(operations, &self.values, required);
                let mut start = 0;
                while start < operations.len() {
                    if !eligible.contains(&start) {
                        start += 1;
                        continue;
                    }
                    let mut selected = vec![start];
                    for next in start + 1..operations.len() {
                        let op = &operations[next];
                        if selected.len() >= limit
                            || !(matches!(
                                op.kind,
                                MidOperationKind::Copy { .. }
                                    | MidOperationKind::Compute(Compute::Sum { .. })
                            ) || matches!(&op.kind, MidOperationKind::Compute(Compute::Product(product)) if product.output_aliases.is_empty()))
                            || selected
                                .iter()
                                .any(|&index| conflicts(&operations[index], op, &self.values))
                        {
                            break;
                        }
                        if eligible.contains(&next) {
                            let mut candidate = selected.clone();
                            candidate.push(next);
                            if separate_homes(
                                candidate.iter().map(|&i| &operations[i]),
                                &self.values,
                                self.tile_count,
                            )
                            .is_none()
                            {
                                break;
                            }
                            selected = candidate;
                        }
                    }
                    if selected.len() < 2 {
                        start += 1;
                        continue;
                    }
                    let homes = separate_homes(
                        selected.iter().map(|&i| &operations[i]),
                        &self.values,
                        self.tile_count,
                    )
                    .unwrap();
                    proposal.homes.extend(homes);
                    proposal.reductions.push(ReductionGroup {
                        members: selected
                            .iter()
                            .map(|&i| operations[i].work_site().unwrap())
                            .collect(),
                    });
                    start = selected.last().copied().unwrap() + 1;
                }
            },
        );
        proposal
    }

    pub(crate) fn propose_preparation_homes(
        &self,
        checkpoints: bool,
    ) -> BTreeMap<ResultSite, OwnerMap> {
        let mut homes = BTreeMap::new();
        let groups = self
            .values
            .iter()
            .map(|v| v.storage_group)
            .collect::<Vec<_>>();
        visit_regions(
            &self.operations,
            &self.outputs,
            &mut |operations, required| {
                let mut uses = vec![0usize; self.values.len()];
                let mut producers = BTreeMap::new();
                for operation in operations {
                    for input in operation.read_values() {
                        uses[groups[input.index() as usize].index() as usize] += 1;
                    }
                    if matches!(
                        operation.kind,
                        MidOperationKind::Compute(Compute::Sum { .. })
                    ) && operation.results.len() == 1
                    {
                        producers.insert(operation.results[0], operation);
                    }
                }
                let mut index = 0;
                while index < operations.len() {
                    let count = independent_copy_prefix(&operations[index..], checkpoints, &groups);
                    let sources = operations[index..index + count]
                        .iter()
                        .filter_map(|operation| {
                            let MidOperationKind::Copy { mapping, .. } = &operation.kind else {
                                return None;
                            };
                            let input = *operation.inputs.first()?;
                            let value = &self.values[input.index() as usize];
                            (mapping.view.is_some()
                                && uses[value.storage_group.index() as usize] == 1
                                && !required
                                    .iter()
                                    .any(|id| groups[id.index() as usize] == value.storage_group)
                                && value.tensor_type.format.layout.order
                                    == ElementOrder::Amp(AmpOrder::TransposedLeft))
                            .then(|| producers.get(&input).copied())
                            .flatten()
                        })
                        .collect::<Vec<_>>();
                    if let Some(proposed) =
                        separate_homes(sources.into_iter(), &self.values, self.tile_count)
                    {
                        homes.extend(proposed);
                    }
                    index += count.max(1);
                }
            },
        );
        homes
    }

    /// Delay each named group to its last member. Check the complete request
    /// before mutation: missing members, region crossings, reused members and
    /// read/write hazards cannot silently turn an explicit choice into a no-op.
    pub(crate) fn group_reductions(
        &mut self,
        groups: &[ReductionGroup],
    ) -> Result<(), ProgramError> {
        if groups.is_empty() {
            return Ok(());
        }
        let mut remaining = BTreeSet::new();
        for group in groups {
            if group.members.len() < 2
                || group
                    .members
                    .iter()
                    .any(|site| !remaining.insert(site.clone()))
            {
                return Err(ProgramError::Invalid(
                    "reduction groups need distinct members and at least two reductions".into(),
                ));
            }
        }
        let mut checked = BTreeMap::new();
        let mut error = None;
        visit_regions(&self.operations, &self.outputs, &mut |operations, _| {
            let indexes = operations
                .iter()
                .enumerate()
                .filter_map(|(i, op)| op.work_site().map(|site| (site, i)))
                .collect::<BTreeMap<_, _>>();
            for group in groups {
                let Some(indices) = group
                    .members
                    .iter()
                    .map(|site| indexes.get(site).copied())
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let valid = indices.windows(2).all(|pair| pair[0] < pair[1])
                    && indices.iter().all(|&i| {
                        matches!(
                            operations[i].kind,
                            MidOperationKind::Compute(Compute::Sum { .. })
                        )
                    })
                    && indices.iter().all(|&i| {
                        (i + 1..=*indices.last().unwrap()).all(|j| {
                            !matches!(operations[j].kind, MidOperationKind::Repeat(_))
                                && !conflicts(&operations[i], &operations[j], &self.values)
                        })
                    });
                if !valid {
                    error = Some(ProgramError::Invalid(format!(
                        "reduction group crosses a dependency: {:?}",
                        group.members
                    )));
                    continue;
                }
                for site in &group.members {
                    remaining.remove(site);
                }
                checked.insert(
                    group.members.last().unwrap().clone(),
                    group.members.iter().cloned().collect::<BTreeSet<_>>(),
                );
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        if !remaining.is_empty() {
            return Err(ProgramError::Invalid(format!(
                "reduction group has unavailable members or crosses a region: {remaining:?}"
            )));
        }
        reorder_region(&mut self.operations, &checked);
        Ok(())
    }
}

fn visit_regions(
    operations: &[MidOperation],
    required: &[MidValueId],
    visit: &mut impl FnMut(&[MidOperation], &[MidValueId]),
) {
    for op in operations {
        if let MidOperationKind::Repeat(repeat) = &op.kind {
            visit_regions(&repeat.body.operations, &repeat.body.yields, visit);
        }
    }
    visit(operations, required);
}

fn conflicts(a: &MidOperation, b: &MidOperation, values: &[MidValue]) -> bool {
    let same = |a: &MidValueId, b: &MidValueId| {
        values[a.index() as usize].storage_group == values[b.index() as usize].storage_group
    };
    a.results.iter().any(|output| {
        b.read_values()
            .chain(&b.results)
            .any(|input| same(output, input))
    }) || b
        .results
        .iter()
        .any(|output| a.read_values().any(|input| same(output, input)))
}

fn reduction_outputs(
    operations: &[MidOperation],
    values: &[MidValue],
    required: &[MidValueId],
) -> BTreeSet<usize> {
    let group = |id: MidValueId| values[id.index() as usize].storage_group;
    let mut forbidden = required.iter().copied().map(group).collect::<BTreeSet<_>>();
    let mut used = BTreeSet::new();
    for op in operations {
        for input in op.read_values() {
            used.insert(group(*input));
            if !matches!(op.kind, MidOperationKind::Copy { .. }) {
                forbidden.insert(group(*input));
            }
        }
    }
    operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            (matches!(op.kind, MidOperationKind::Compute(Compute::Sum { .. }))
                && op.results.len() == 1
                && op.work_site().is_some()
                && used.contains(&group(op.results[0]))
                && !forbidden.contains(&group(op.results[0])))
            .then_some(i)
        })
        .collect()
}

/// Preserve every selected domain and find disjoint cyclic ranges within them.
/// Different embeddings need physical overlap checks, not just summed counts.
fn separate_homes<'a>(
    operations: impl Iterator<Item = &'a MidOperation>,
    values: &[MidValue],
    tiles: u16,
) -> Option<BTreeMap<ResultSite, OwnerMap>> {
    let sources = operations
        .map(|op| Some((op.result_site(0)?, &values[op.results[0].index() as usize])))
        .collect::<Option<Vec<_>>>()?;
    if sources.len() < 2
        || sources
            .iter()
            .map(|(_, v)| u32::from(v.tensor_type.format.layout.tiling.tile_count))
            .sum::<u32>()
            > u32::from(tiles)
    {
        return None;
    }
    let mut occupied = BTreeSet::new();
    let mut offset = sources[0].1.owners.rotation();
    let mut homes = BTreeMap::new();
    for (site, value) in sources {
        let count = value.tensor_type.format.layout.tiling.tile_count;
        let (home, used) = (0..tiles).find_map(|shift| {
            let home = value
                .owners
                .with_rotation(((u32::from(offset) + u32::from(shift)) % u32::from(tiles)) as u16);
            home.validate(count, tiles).ok()?;
            let used = (0..count)
                .map(|owner| home.tile(owner, tiles))
                .collect::<Option<BTreeSet<_>>>()?;
            occupied.is_disjoint(&used).then_some((home, used))
        })?;
        occupied.extend(used);
        offset = ((u32::from(home.rotation()) + u32::from(count)) % u32::from(tiles)) as u16;
        homes.insert(site, home);
    }
    Some(homes)
}

fn reorder_region(
    operations: &mut Vec<MidOperation>,
    groups: &BTreeMap<WorkSite, BTreeSet<WorkSite>>,
) {
    let mut delayed = BTreeMap::new();
    for (last, members) in groups {
        for site in members {
            delayed.insert(site.clone(), last);
        }
    }
    let mut pending = BTreeMap::<WorkSite, Vec<MidOperation>>::new();
    let mut result = Vec::with_capacity(operations.len());
    for mut operation in std::mem::take(operations) {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            reorder_region(&mut repeat.body.operations, groups);
        }
        if let Some(site) = operation.work_site()
            && let Some(last) = delayed.get(&site)
        {
            pending.entry((*last).clone()).or_default().push(operation);
            if site == **last {
                result.extend(pending.remove(*last).unwrap());
            }
        } else {
            result.push(operation);
        }
    }
    debug_assert!(pending.is_empty());
    *operations = result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ComputeGraph, ValueId};
    use crate::kernel::TileKernelSpec;
    use crate::mid::{CoordinateMapping, MidRegion, MidRepeat, ReductionStaging};
    use crate::tensor::{AxisFactorView, Layout, Precision, TensorType};

    impl MidProgram {
        fn separate_preparation(&self, checkpoints: bool) -> Option<Self> {
            let homes = self.propose_preparation_homes(checkpoints);
            if homes.is_empty() {
                return None;
            }
            let mut program = self.clone();
            program
                .apply_ownership(&super::super::OwnerChoices {
                    results: homes,
                    ..Default::default()
                })
                .unwrap();
            (program != *self).then_some(program)
        }
        fn overlap_reductions(&self, limit: usize) -> Option<Self> {
            let proposal = self.propose_reduction_groups(limit);
            if proposal.reductions.is_empty() {
                return None;
            }
            let mut program = self.clone();
            program
                .apply_ownership(&super::super::OwnerChoices {
                    results: proposal.homes,
                    ..Default::default()
                })
                .unwrap();
            program.group_reductions(&proposal.reductions).unwrap();
            (program != *self).then_some(program)
        }
    }
    #[test]
    fn independent_copy_roots_rotate_without_moving_partials_or_shared_results() {
        let mut provenance = ComputeGraph::new();
        let input = provenance.host_input("x", [16]).unwrap();
        provenance.gelu(input).unwrap();
        let source = provenance.operations()[0].id;
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
            site: Some(format!("output.{output}").as_str().into()),
            source: Some(source),
            inputs: vec![MidValueId::from_index(input)],
            results: vec![MidValueId::from_index(output)],
            kind: primitive,
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
        let rotated = program.separate_preparation(true).unwrap();
        let mut domains = program.clone();
        domains.values[0].owners = OwnerMap::embedded(vec![4, 5, 6, 7, 0, 1, 2, 3]);
        domains.values[1].owners = OwnerMap::embedded((0..8).collect::<Vec<_>>());
        let proposed = domains.propose_preparation_homes(true);
        let tiles = proposed
            .values()
            .map(|home| {
                (0..4)
                    .map(|i| home.tile(i, 16).unwrap())
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(tiles.len(), 2);
        assert!(
            tiles[0].is_disjoint(&tiles[1]),
            "different domains need physical overlap checks"
        );
        domains.values[0].owners = OwnerMap::embedded(vec![0, 1, 2, 3]);
        domains.values[1].owners = domains.values[0].owners.clone();
        assert!(domains.propose_preparation_homes(true).is_empty());
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
        let choice = delayed.propose_reduction_groups(2);
        let mut ordered = delayed.clone();
        ordered.group_reductions(&choice.reductions).unwrap();
        assert_eq!(
            ordered.values, delayed.values,
            "grouping must preserve selected result homes"
        );
        let mut duplicate = choice.reductions.clone();
        duplicate.extend(choice.reductions.clone());
        let before = ordered.clone();
        assert!(ordered.group_reductions(&duplicate).is_err());
        assert_eq!(ordered, before);
        let overlapped = delayed.overlap_reductions(2).unwrap();
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
                    },
                }),
            });
            body
        };
        let repeated = wrap(program.clone(), vec![]);
        let rotated_repeat = repeated.separate_preparation(true).unwrap();
        assert_eq!(rotated_repeat.values, rotated.values);
        let repeated = wrap(delayed.clone(), vec![]);
        let grouped_repeat = repeated.overlap_reductions(2).unwrap();
        let MidOperationKind::Repeat(repeat) = &grouped_repeat.operations[0].kind else {
            unreachable!()
        };
        assert_eq!(repeat.body.operations, overlapped.operations);
        assert_eq!(grouped_repeat.values, overlapped.values);
        assert!(
            wrap(program.clone(), vec![MidValueId::from_index(1)])
                .separate_preparation(true)
                .is_none()
        );
        assert!(
            wrap(delayed.clone(), vec![MidValueId::from_index(1)])
                .overlap_reductions(2)
                .is_none()
        );

        assert!(delayed.overlap_reductions(1).is_none());
        delayed.operations[1].kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Gelu,
            operands: Vec::new(),
            output_aliases: vec![(0, 0)],
        });
        assert!(delayed.overlap_reductions(2).is_none());
        delayed.operations[1] = copy(5, 4); // Would overwrite a delayed partial.
        assert!(delayed.overlap_reductions(2).is_none());

        program.outputs.push(MidValueId::from_index(1));
        assert!(program.separate_preparation(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId::from_index(1);
        program.outputs.push(MidValueId::from_index(3));
        assert!(program.separate_preparation(true).is_none());
        program.outputs.clear();
        program.values[3].storage_group = MidValueId::from_index(3);
        program.operations.push(copy(1, 2));
        assert!(program.separate_preparation(true).is_none());
        assert_eq!(
            independent_copy_prefix(&[copy(0, 2), copy(2, 3)], false, &groups),
            1
        );
        let mut boundary = copy(1, 3);
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("x", [16]).unwrap();
        let first = graph.gelu(input).unwrap();
        graph.gelu(first).unwrap();
        boundary.source = Some(graph.operations()[1].id);
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
        assert!(program.separate_preparation(true).is_none());
        assert!(program.separate_preparation(false).is_some());
    }
}
