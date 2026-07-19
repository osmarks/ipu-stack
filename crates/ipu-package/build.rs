fn main() {
    println!("cargo:rerun-if-changed=../../schemas/application.capnp");
    println!("cargo:rerun-if-changed=../../schemas/profile.capnp");
    println!("cargo:rerun-if-changed=../../schemas/memory_profile.capnp");
    capnpc::CompilerCommand::new()
        .src_prefix("../../schemas")
        .file("../../schemas/application.capnp")
        .file("../../schemas/profile.capnp")
        .file("../../schemas/memory_profile.capnp")
        .run()
        .expect("compile application.capnp");
}
