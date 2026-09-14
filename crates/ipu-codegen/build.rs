#[path = "../../build/abi.rs"]
mod abi;
fn main() {
    abi::generate("../../device/runtime_layout.def", "runtime_layout.rs");
}
