#[path = "../../build/abi.rs"]
mod abi;
fn main() {
    abi::generate("include/ipu21_instruction.def", "instruction.rs");
    abi::generate("include/ipu21_registers.def", "registers.rs");
    abi::generate("include/runtime_layout.def", "runtime_layout.rs");
}
