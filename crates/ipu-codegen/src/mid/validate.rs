//! Check executable bindings at construction and rewrite boundaries. Physical
//! address/access legality is checked later, against concrete kernel calls.

use super::{
    LoweringError, LoweringResult, MidOperation, MidOperationKind, MidProgram, MidValueId,
    OperandMaterialization, Primitive,
};
use std::collections::BTreeSet;

impl MidProgram {
    pub(crate) fn validate(&self) -> LoweringResult<()> {
        let invalid = |message| LoweringError::InvalidProgram(message);
        for (index, value) in self.values.iter().enumerate() {
            if value.id.index() as usize != index
                || value.storage_group.index() as usize >= self.values.len()
                || value.tile_offset >= self.tile_count
            {
                return Err(invalid(format!("invalid value binding {:?}", value.id)));
            }
            value
                .tensor_type
                .format
                .layout
                .validate_tile_count(self.tile_count)?;
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
    ) -> LoweringResult<()> {
        let invalid = |message| LoweringError::InvalidProgram(message);
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
                MidOperationKind::Primitive(Primitive::Copy { .. }) => {
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
                MidOperationKind::Primitive(Primitive::Compute {
                    operands,
                    output_aliases,
                    ..
                }) => {
                    arity.1 != 0
                        && operands.len() <= arity.0
                        && output_aliases
                            .iter()
                            .all(|&(output, input)| output < arity.1 && input < arity.0)
                }
                MidOperationKind::Primitive(Primitive::Sum { axis, .. }) => {
                    arity == (1, 1)
                        && usize::from(*axis)
                            < self.values[operation.inputs[0].index() as usize]
                                .tensor_type
                                .shape
                                .0
                                .len()
                }
                MidOperationKind::Convert(plan) => {
                    arity == (1, 1)
                        && plan.output.materialization == OperandMaterialization::Complete
                        && plan.input.format
                            == self.values[operation.inputs[0].index() as usize]
                                .tensor_type
                                .format
                        && plan.output.format
                            == self.values[operation.results[0].index() as usize]
                                .tensor_type
                                .format
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
            if !valid {
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
    use crate::{ComputeGraph, Ipu21CostModel, PipelineConfig, Precision};

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
        crate::mid::lower(&graph, &config, &Ipu21CostModel).unwrap()
    }

    #[test]
    fn executable_bindings_reject_forward_reads_and_duplicate_definitions() {
        let original = program(false);
        original.validate().unwrap();
        let mut forward = original.clone();
        forward.operations[0].inputs[0] = forward.outputs[0];
        assert!(matches!(
            forward.validate(),
            Err(LoweringError::InvalidProgram(_))
        ));
        let mut duplicate = original.clone();
        duplicate.operations.push(duplicate.operations[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(LoweringError::InvalidProgram(_))
        ));
        let mut alias = original;
        let operation = alias
            .operations
            .iter_mut()
            .find_map(|op| match &mut op.kind {
                MidOperationKind::Primitive(Primitive::Compute { output_aliases, .. }) => {
                    Some(output_aliases)
                }
                _ => None,
            })
            .unwrap();
        operation.push((usize::MAX, 0));
        assert!(matches!(
            alias.validate(),
            Err(LoweringError::InvalidProgram(_))
        ));
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
        assert!(matches!(
            program.validate(),
            Err(LoweringError::InvalidProgram(_))
        ));
    }
}
