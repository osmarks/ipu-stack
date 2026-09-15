//! Change value ownership and make the resulting operand movement explicit.

use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
    OperandIndexing, ProgramError, ResultSite,
};
use crate::tensor::OwnerMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Input homes are base assignments retaining constructor rotations. Explicit
/// result homes name the actual assignment of that result. Operator defaults choose a
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
        let results = self.named_results().collect::<BTreeMap<_, _>>();
        let mut explicit = BTreeMap::new();
        for (site, map) in &choices.results {
            let value = results.get(site).ok_or_else(|| {
                ProgramError::Invalid(format!("owner choice is unavailable at {site:?}"))
            })?;
            let value = &self.values[value.index() as usize];
            // Convert the requested result assignment into the common base of
            // its alias group. Other aliases retain their relative rotations.
            map.validate(
                value.tensor_type.format.layout.tiling.tile_count,
                self.tile_count,
            )?;
            let map = map
                .shifted(-i32::from(value.owners.rotation()), self.tile_count)
                .ok_or_else(|| ProgramError::Invalid("invalid result owner domain".into()))?
                .canonicalized(self.tile_count);
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
    use crate::graph::{ComputeGraph, ValueId};
    use crate::kernel::TileKernelSpec;
    use crate::tensor::{Layout, Precision, TensorType};

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
}
