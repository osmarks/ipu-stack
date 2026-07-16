fn main() {
    println!("cargo:rerun-if-changed=../../schemas/application.capnp");
    capnpc::CompilerCommand::new()
        .src_prefix("../../schemas")
        .file("../../schemas/application.capnp")
        .run()
        .expect("compile application.capnp");
}
