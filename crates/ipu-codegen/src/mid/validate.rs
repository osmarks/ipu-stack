//! Check executable bindings at construction and rewrite boundaries. Physical
//! address/access legality is checked later, against concrete kernel calls.

use super::{MidGraph, MidOperation, MidValueId};
use crate::OperandIndexing;
use crate::mid::MidOperationKind;
use std::collections::BTreeSet;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    #[error("invalid mid program: {0}")]
    Invalid(String),
    #[error(transparent)]
    Layout(#[from] crate::tensor::LayoutError),
}

type ProgramResult<T> = Result<T, ProgramError>;

impl MidGraph {
    pub(crate) fn validate(&self) -> ProgramResult<()> {
        let invalid = |message| ProgramError::Invalid(message);
        for (index, value) in self.values.iter().enumerate() {
            if value.id.index() as usize != index
                || value.storage_group.index() as usize >= self.values.len()
            {
                return Err(invalid(format!("invalid value binding {:?}", value.id)));
            }
            value
                .tensor_type
                .format
                .layout
                .validate_tile_count(self.tile_count)?;
            value.owners.validate(
                value.tensor_type.format.layout.tiling.tile_count,
                self.tile_count,
            )?;
        }
        let mut defined = BTreeSet::new();
        self.validate_region(
            &self
                .inputs
                .iter()
                .map(|input| input.value)
                .collect::<Vec<_>>(),
            &self.operations,
            &self.outputs,
            &mut defined,
        )
    }

    fn validate_region(
        &self,
        arguments: &[MidValueId],
        operations: &[MidOperation],
        yields: &[MidValueId],
        defined: &mut BTreeSet<MidValueId>,
    ) -> ProgramResult<()> {
        let invalid = |message| ProgramError::Invalid(message);
        let mut available = BTreeSet::new();
        let define = |id: MidValueId, defined: &mut BTreeSet<_>| {
            if id.index() as usize >= self.values.len() || !defined.insert(id) {
                return Err(invalid(format!("invalid or multiply defined value {id:?}")));
            }
            Ok(())
        };
        for &argument in arguments {
            define(argument, defined)?;
            available.insert(argument);
        }
        for operation in operations {
            for &input in operation.read_values() {
                if !available.contains(&input) {
                    return Err(invalid(format!(
                        "operation {:?} reads unavailable value {input:?}",
                        operation.source
                    )));
                }
            }
            for &output in &operation.results {
                define(output, defined)?;
            }
            let arity = (operation.inputs.len(), operation.results.len());
            let valid = match &operation.kind {
                MidOperationKind::Copy { .. } => {
                    arity == (1, 1)
                        && self.values[operation.inputs[0].index() as usize]
                            .tensor_type
                            .format
                            .precision
                            == self.values[operation.results[0].index() as usize]
                                .tensor_type
                                .format
                                .precision
                }

                MidOperationKind::Repeat(repeat) => {
                    self.validate_region(
                        &repeat.body.arguments,
                        &repeat.body.operations,
                        &repeat.body.yields,
                        defined,
                    )?;
                    repeat.count != 0
                        && repeat.carried_inputs + repeat.invariant_inputs == arity.0
                        && repeat.carried_inputs == arity.1
                        && repeat.body.yields.len() == arity.1
                        && repeat.body.arguments.len() == arity.0 + repeat.iterated_inputs.len()
                        && repeat
                            .iterated_inputs
                            .iter()
                            .all(|sequence| sequence.len() == repeat.count as usize)
                }
                kernel => {
                    let operands = &operation.operands;
                    arity.1 != 0
                        && operands.len() <= arity.0
                        && operation.output_windows.len() <= arity.1
                        && operation.output_windows.iter().zip(&operation.results).all(
                            |(window, result)| {
                                let rank = self.values[result.index() as usize]
                                    .tensor_type
                                    .shape
                                    .0
                                    .len();
                                window.0.iter().all(|&(axis, start, end)| {
                                    usize::from(axis) < rank && start < end
                                })
                            },
                        )
                        && operation
                            .inputs
                            .iter()
                            .zip(operands)
                            .all(|(&input, indexing)| {
                                let input = &self.values[input.index() as usize].tensor_type;
                                match indexing {
                                    OperandIndexing::Elementwise { result } => {
                                        operation.results.get(*result).is_some_and(|result| {
                                            crate::tensor::Broadcast::new(
                                                &input.shape.0,
                                                &self.values[result.index() as usize]
                                                    .tensor_type
                                                    .shape
                                                    .0,
                                            )
                                            .is_some()
                                        })
                                    }
                                    OperandIndexing::Local(window)
                                    | OperandIndexing::Fragment(window) => {
                                        window.0.iter().all(|&(axis, start, end)| {
                                            usize::from(axis) < input.shape.0.len() && start < end
                                        })
                                    }
                                }
                            })
                        && match kernel {
                            MidOperationKind::Gemm {
                                inner_block,
                                output_columns,
                                ..
                            } => {
                                arity.0 >= 2
                                    && arity.1 == 1
                                    && *inner_block > 0
                                    && *output_columns > 0
                            }
                            MidOperationKind::Cast { from, to } => {
                                arity.0 >= 1
                                    && arity.1 == 1
                                    && operands.len() == 1
                                    && self.values[operation.inputs[0].index() as usize]
                                        .tensor_type
                                        .format
                                        .precision
                                        == *from
                                    && self.values[operation.results[0].index() as usize]
                                        .tensor_type
                                        .format
                                        .precision
                                        == *to
                                    && self.values[operation.inputs[0].index() as usize]
                                        .tensor_type
                                        .shape
                                        == self.values[operation.results[0].index() as usize]
                                            .tensor_type
                                            .shape
                            }
                            _ => true,
                        }
                }
            };
            let aliases_valid = operation
                .output_aliases
                .as_slice()
                .iter()
                .all(|&(output, input)| output < arity.1 && input < arity.0);
            if !valid || !aliases_valid {
                return Err(invalid(format!(
                    "invalid executable operation {:?}",
                    operation.source
                )));
            }
            available.extend(&operation.results);
        }
        for output in yields {
            if !available.contains(output) {
                return Err(invalid(format!(
                    "region yields unavailable value {output:?}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GraphInputKind, Layout, MidInput, MidRegion, MidRepeat, MidValue, OperandIndexing,
        Precision, TensorType, ValueId,
    };

    fn program(repeated: bool) -> MidGraph {
        let id = MidValueId::from_index;
        let gelu = MidOperation {
            source: None,
            inputs: vec![id(if repeated { 2 } else { 0 })],
            results: vec![id(if repeated { 3 } else { 1 })],
            kind: MidOperationKind::Gelu,
            operands: vec![OperandIndexing::Elementwise { result: 0 }],
            output_aliases: vec![],
            output_windows: vec![],
        };
        MidGraph {
            tile_count: 4,
            values: (0..if repeated { 4 } else { 2 })
                .map(|i| MidValue {
                    id: id(i),
                    origin: ValueId::from_index(i),
                    storage_group: id(i),
                    owners: Default::default(),
                    tensor_type: TensorType::new([4, 16], Precision::F16, Layout::row_sharded(4)),
                })
                .collect(),
            inputs: vec![MidInput {
                name: "x".into(),
                kind: GraphInputKind::Host,
                value: id(0),
            }],
            outputs: vec![id(1)],
            operations: vec![if repeated {
                MidOperation {
                    source: None,
                    inputs: vec![id(0)],
                    results: vec![id(1)],
                    kind: MidOperationKind::Repeat(MidRepeat {
                        count: 3,
                        carried_inputs: 1,
                        invariant_inputs: 0,
                        iterated_inputs: vec![],
                        body: MidRegion {
                            arguments: vec![id(2)],
                            operations: vec![gelu],
                            yields: vec![id(3)],
                        },
                    }),
                    operands: vec![],
                    output_aliases: vec![],
                    output_windows: vec![],
                }
            } else {
                gelu
            }],
            ..MidGraph::default()
        }
    }

    #[test]
    fn executable_bindings_reject_forward_reads_and_duplicate_definitions() {
        let original = program(false);
        original.validate().unwrap();
        let mut forward = original.clone();
        forward.operations[0].inputs[0] = forward.outputs[0];
        assert!(matches!(forward.validate(), Err(ProgramError::Invalid(_))));
        let mut duplicate = original.clone();
        duplicate.operations.push(duplicate.operations[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(ProgramError::Invalid(_))
        ));
        let mut alias = original;
        let operation = alias
            .operations
            .iter_mut()
            .find(|op| {
                !matches!(
                    op.kind,
                    MidOperationKind::Copy { .. } | MidOperationKind::Repeat(_)
                )
            })
            .unwrap();
        operation.output_aliases.push((usize::MAX, 0));
        assert!(matches!(alias.validate(), Err(ProgramError::Invalid(_))));
    }

    #[test]
    fn invalid_owner_embeddings_fail_before_tile_expansion() {
        let original = program(false);
        let input = original.inputs[0].value;
        for tiles in [vec![], vec![0, 0, 1, 2], vec![0, 1, 2, 4]] {
            let mut bad = original.clone();
            bad.values[input.index() as usize].owners = crate::tensor::OwnerMap::embedded(tiles);
            assert!(matches!(
                bad.validate(),
                Err(ProgramError::Layout(
                    crate::tensor::LayoutError::InvalidOwnerMap { .. }
                ))
            ));
            assert!(crate::low::expand::expand_tiles(&bad, false).is_err());
        }
    }

    #[test]
    fn repeat_body_reads_only_its_declared_bindings() {
        let mut program = program(true);
        program.validate().unwrap();
        let outer = program.inputs[0].value;
        let body = program
            .operations
            .iter_mut()
            .find_map(|op| match &mut op.kind {
                MidOperationKind::Repeat(repeat) => Some(&mut repeat.body),
                _ => None,
            })
            .unwrap();
        body.operations[0].inputs[0] = outer;
        assert!(matches!(program.validate(), Err(ProgramError::Invalid(_))));
    }
}
