//! Relabel device tiles and bind compute operands to their result ownership.
use crate::mid::MidOperationKind;
use crate::mid::{
    CoordinateMapping, MidOperation, MidProgram, MidValue, MidValueId, OperandIndexing,
    ProgramError,
};
use crate::tensor::OwnerMap;
impl MidProgram {
    /// Relabel an already constructed program before expansion, for explicit
    /// transfer captures and the global recipe mapping.
    pub(crate) fn remap_tiles(&mut self, mapping: &[u16]) -> Result<(), ProgramError> {
        crate::tensor::remap_owners(
            self.values.iter_mut().map(|value| &mut value.owners),
            mapping,
            self.tile_count,
        )
        .map_err(Into::into)
    }
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
                for (yielded, &input) in repeat.body.yields.iter_mut().zip(&operation.inputs) {
                    let owners = values[input.index() as usize].owners.clone();
                    if values[yielded.index() as usize].owners != owners {
                        *yielded = append_owner_copy(
                            operation.source,
                            *yielded,
                            owners,
                            values,
                            &mut repeat.body.operations,
                        );
                    }
                }
            }
            compute => {
                for (index, input) in operation.inputs.iter_mut().enumerate() {
                    let result = match compute {
                        _ => operation.operands.get(index).map(|operand| match operand {
                            OperandIndexing::Elementwise { result } => *result,
                            OperandIndexing::Local(_) | OperandIndexing::Fragment(_) => 0,
                        }),
                    };
                    let mut result = result;
                    for &(output, alias_input) in &operation.output_aliases {
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
                            *input,
                            owners,
                            values,
                            &mut rewritten,
                        );
                    }
                }
            }
        }
        rewritten.push(operation);
    }
    *operations = rewritten;
    Ok(())
}

fn append_owner_copy(
    source: Option<crate::OperationId>,

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
        output_windows: Vec::new(),
        source,
        inputs: vec![input],
        results: vec![id],
        kind: MidOperationKind::Copy {
            policy: crate::CopyPolicy::Automatic,
            packing: crate::PackingPolicy::Automatic,
            mapping: CoordinateMapping::default(),
        },
        operands: Vec::new(),
        output_aliases: Vec::new(),
    });
    id
}
