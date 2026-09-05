//! GEMM tile phases and operand staging.

use super::*;

pub(super) fn split_gemm_matrices(
    run: &KernelRun,
    axis: usize,
    coordinates: &mut [u32],
    runs: &mut Vec<KernelRun>,
) -> ExpansionResult<()> {
    if axis < coordinates.len() {
        let extent = run
            .output
            .extents
            .get(axis)
            .ok_or(ExpansionError::InvalidOperatorPlan)?;
        if extent.logical_end != extent.physical_end {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        for coordinate in extent.start..extent.physical_end {
            coordinates[axis] = coordinate;
            split_gemm_matrices(run, axis + 1, coordinates, runs)?;
        }
        return Ok(());
    }

    let mut matrix = run.clone();
    narrow_gemm_matrix_view(&mut matrix.output, coordinates)?;
    for operand in &mut matrix.inputs {
        for view in &mut operand.views {
            narrow_gemm_matrix_view(view, coordinates)?;
        }
    }
    runs.push(matrix);
    Ok(())
}

pub(super) fn narrow_gemm_matrix_view(
    view: &mut ShardView,
    output_coordinates: &[u32],
) -> ExpansionResult<()> {
    let input_axes = view.extents.len().saturating_sub(2);
    if input_axes > output_coordinates.len() {
        return Err(ExpansionError::InvalidOperatorPlan);
    }
    let output_axis_offset = output_coordinates.len() - input_axes;
    for (axis, extent) in view.extents[..input_axes].iter_mut().enumerate() {
        if extent.physical_end - extent.start == 1 {
            continue;
        }
        let coordinate = output_coordinates[output_axis_offset + axis];
        if coordinate < extent.start || coordinate >= extent.physical_end {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        extent.start = coordinate;
        extent.logical_end = coordinate + 1;
        extent.physical_end = coordinate + 1;
    }
    Ok(())
}
