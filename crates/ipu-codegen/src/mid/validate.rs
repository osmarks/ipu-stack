//! Check executable bindings at construction and rewrite boundaries. Physical
//! address/access legality is checked later, against concrete kernel calls.

use super::{MidOperation, MidOperationKind, MidProgram, MidValueId};
use crate::{Compute, OperandIndexing, TileKernelSpec};
use std::collections::BTreeSet;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    #[error("invalid mid program: {0}")]
    Invalid(String),
    #[error(transparent)]
    Layout(#[from] crate::tensor::LayoutError),
}

type ProgramResult<T> = Result<T, ProgramError>;

impl MidProgram {
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
                MidOperationKind::Compute(Compute::Product(product)) => {
                    arity.0 >= 2
                        && arity.1 == 1
                        && product.inner_block > 0
                        && product.output_columns > 0
                        && product
                            .axes
                            .left_inner
                            .resolve(
                                self.values[operation.inputs[0].index() as usize]
                                    .tensor_type
                                    .shape
                                    .0
                                    .len(),
                            )
                            .is_ok()
                        && product
                            .axes
                            .right_inner
                            .resolve(
                                self.values[operation.inputs[1].index() as usize]
                                    .tensor_type
                                    .shape
                                    .0
                                    .len(),
                            )
                            .is_ok()
                        && product
                            .axes
                            .output_column
                            .resolve(
                                self.values[operation.results[0].index() as usize]
                                    .tensor_type
                                    .shape
                                    .0
                                    .len(),
                            )
                            .is_ok()
                }
                MidOperationKind::Compute(Compute::Kernel {
                    kernel, operands, ..
                }) => {
                    arity.1 != 0
                        && operands.len() <= arity.0
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
                                    OperandIndexing::Local(window) => {
                                        window.0.iter().all(|&(axis, start, end)| {
                                            usize::from(axis) < input.shape.0.len() && start < end
                                        })
                                    }
                                }
                            })
                        && match kernel {
                            TileKernelSpec::Gemm { .. } => false, // Distributed products have explicit axes and blocking.
                            TileKernelSpec::Cast { from, to } => {
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
                MidOperationKind::Compute(Compute::Sum { axis, .. }) => {
                    if arity != (1, 1) {
                        false
                    } else {
                        let input = &self.values[operation.inputs[0].index() as usize].tensor_type;
                        let output =
                            &self.values[operation.results[0].index() as usize].tensor_type;
                        let axis = usize::from(*axis);
                        axis < input.shape.0.len()
                            && input.format.precision == output.format.precision
                            && input
                                .shape
                                .0
                                .iter()
                                .enumerate()
                                .filter_map(|(i, d)| (i != axis).then_some(d))
                                .eq(&output.shape.0)
                    }
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
            };
            let aliases_valid = match &operation.kind {
                MidOperationKind::Compute(compute) => compute
                    .output_aliases()
                    .iter()
                    .all(|&(output, input)| output < arity.1 && input < arity.0),
                _ => true,
            };
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
    use crate::estimate::Ipu21CostModel;
    use crate::{ComputeGraph, PipelineConfig, Precision};

    fn program(repeated: bool) -> MidProgram {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [4, 16]).unwrap();
        let y = if repeated {
            graph
                .repeat(3, [x], [], [], |body, args| {
                    Ok(vec![body.gelu(args.carried[0])?])
                })
                .unwrap()[0]
        } else {
            graph.gelu(x).unwrap()
        };
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(4).with_automatic_input(x, Precision::F16);
        crate::planner::test_support::lower(&graph, &config, &Ipu21CostModel).unwrap()
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
            .find_map(|op| match &mut op.kind {
                MidOperationKind::Compute(Compute::Kernel { output_aliases, .. }) => {
                    Some(output_aliases)
                }
                _ => None,
            })
            .unwrap();
        operation.push((usize::MAX, 0));
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
