//! Shared build-time reader for constants consumed by Rust and device assembly.
//! The .def input uses one IPU_CONSTANT(name, Rust_type, expression) per line;
//! assembly defines that macro as .set before including the same file.
use std::path::Path;

pub fn generate(input: &str, output: &str) {
    println!("cargo:rerun-if-changed={input}");
    let source = std::fs::read_to_string(input).expect("read ABI constants");
    let mut rust = String::new();
    for line in source.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            rust.push_str(line);
        } else {
            let body = line
                .strip_prefix("IPU_CONSTANT(")
                .and_then(|line| line.strip_suffix(')'))
                .expect("expected IPU_CONSTANT(name, type, expression)");
            let mut fields = body.splitn(3, ',').map(str::trim);
            let name = fields.next().unwrap();
            let ty = fields.next().expect("constant type");
            let value = fields.next().expect("constant value");
            rust.push_str(&format!("pub const {name}: {ty} = {value};"));
        }
        rust.push('\n');
    }
    let directory = std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR");
    std::fs::write(Path::new(&directory).join(output), rust).expect("write ABI constants");
}
