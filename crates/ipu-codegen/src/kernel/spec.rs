use crate::{AccumulationPrecision, GemmKernelMode, GemmWeightLoad, Layout, Precision};

/// A concrete tile-local callable produced during low lowering.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TileKernelSpec {
    FillZero,
    Gemm {
        multiply: Precision,
        accumulate: AccumulationPrecision,
        mode: GemmKernelMode,
        weights: GemmWeightLoad,
        inner_block: u32,
        output_columns: u32,
        rows: u32,
    },
    Gelu,
    ReductionSum {
        partials: u16,
    },
    Add,
    AttentionSoftmax {
        query_rows: u32,
        head_dimension: u32,
        key_columns: u32,
        padded_key_columns: u32,
    },
    AttentionMerge {
        query_rows: u32,
        value_dimension: u32,
        padded_value_dimension: u32,
        key_block_columns: u32,
        initial: bool,
        final_block: bool,
    },
    Cast {
        from: Precision,
        to: Precision,
    },
    Rearrange {
        from: Layout,
        to: Layout,
        matrices: u32,
        logical_rows: u32,
        physical_rows: u32,
        logical_columns: u32,
        physical_columns: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelSymbols {
    Exact(&'static str),
    Planned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelAvailability {
    Implemented,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarArgument {
    pub register: u8,
    pub name: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    pub symbols: KernelSymbols,
    pub availability: KernelAvailability,
    pub output_register: u8,
    pub input_registers: Vec<u8>,
    pub scalar_arguments: Vec<ScalarArgument>,
    pub return_register: u8,
}
