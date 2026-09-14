//! Local Add and GeLU contracts. These kernels consume dense physical streams;
//! broadcast repetition and supported output epilogues are explicit here.

use super::*;

pub(super) fn call(run: &KernelRun) -> Result<KernelCall, KernelAbiError> {
    let inputs = match run.kernel {
        TileKernelSpec::Gelu => 1,
        TileKernelSpec::Add | TileKernelSpec::BiasGelu => 2,
        _ => return Err(KernelAbiError::RequirementMismatch),
    };
    run.check_arity(inputs, 1)?;
    if let Some(mut arguments) = output::fp8_arguments(run)? {
        let symbol = if run.kernel == TileKernelSpec::BiasGelu {
            if element_count(&run.inputs[1].extents)? != input_matrix_extent(run, true, true)? {
                return Err(KernelAbiError::RequirementMismatch);
            }
            "bias_gelu_f8"
        } else {
            arguments.push(input_matrix_extent(run, false, true)?);
            "gelu_f8"
        };
        arguments.push(matrix_extent(&run.outputs[0], false, true)?);
        return Ok(KernelCall::exact(symbol, arguments));
    }
    let count = element_count(&run.outputs[0].extents)?;
    match run.kernel {
        TileKernelSpec::Gelu => {
            let symbol = gelu_symbol(&run.requirements)
                .ok_or_else(|| KernelAbiError::Unavailable(run.kernel.clone()))?;
            if !count.is_multiple_of(2) {
                return Err(KernelAbiError::UnsupportedElementCount {
                    symbol,
                    count,
                    divisor: 2,
                });
            }
            Ok(KernelCall::exact(symbol, vec![count]))
        }
        TileKernelSpec::BiasGelu => {
            let width = output::f16_row_width(run)?;
            if run.inputs[0].extents != run.outputs[0].extents
                || element_count(&run.inputs[1].extents)? != width
            {
                return Err(KernelAbiError::RequirementMismatch);
            }
            Ok(KernelCall::exact(
                "bias_gelu_f16",
                vec![count / width, width],
            ))
        }
        TileKernelSpec::Add => {
            if run.requirements.outputs[0].format.precision != Precision::F16 {
                return Err(KernelAbiError::Unavailable(run.kernel.clone()));
            }
            for operand in &run.inputs {
                let input = &operand.extents;
                let output = &run.outputs[0].extents;
                if input.len() > output.len() {
                    return Err(KernelAbiError::RequirementMismatch);
                }
                let mut suffix = false;
                for (a, b) in input.iter().zip(&output[output.len() - input.len()..]) {
                    let n = a.physical_end - a.start;
                    let m = b.physical_end - b.start;
                    suffix |= n != 1;
                    // The codelet repeats a contiguous suffix, not arbitrary strides.
                    if (suffix && n != m) || n == 0 {
                        return Err(KernelAbiError::RequirementMismatch);
                    }
                }
            }
            Ok(KernelCall::exact(
                "add_f16",
                vec![
                    count,
                    element_count(&run.inputs[0].extents)?,
                    element_count(&run.inputs[1].extents)?,
                ],
            ))
        }
        _ => unreachable!(),
    }
}

fn gelu_symbol(requirements: &KernelRequirements) -> Option<&'static str> {
    let [input] = requirements.inputs.as_slice() else {
        return None;
    };
    (input.format.precision == Precision::F16
        && requirements.outputs[0].format.precision == Precision::F16
        && input.format.layout == requirements.outputs[0].format.layout)
        .then_some("gelu_tanh_approx_f16")
}
