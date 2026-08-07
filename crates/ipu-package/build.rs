fn main() {
    println!("cargo:rerun-if-changed=../../schemas/application.capnp");
    println!("cargo:rerun-if-changed=../../schemas/profile.capnp");
    capnpc::CompilerCommand::new()
        .src_prefix("../../schemas")
        .file("../../schemas/application.capnp")
        .file("../../schemas/profile.capnp")
        .run()
        .expect("compile application.capnp");
}
