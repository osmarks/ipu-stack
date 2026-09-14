//! Resident runtime placement and calling convention. These defaults are
//! shared with device/static_runtime.S; they are not target memory capacities.
include!(concat!(env!("OUT_DIR"), "/runtime_layout.rs"));
