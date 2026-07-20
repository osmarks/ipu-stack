use ipu_models::SiglipWeights;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> ipu_models::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let directory = std::env::args_os()
        .nth(1)
        .expect("usage: ipu-siglip-inspect MODEL_DIRECTORY");
    let model = SiglipWeights::open(directory)?;
    let parameters = model.vision_parameter_count()?;
    info!(
        parameters,
        fp16_bytes = parameters * 2,
        patch_grid = model.patch_grid(),
        sequence_length = model.sequence_length(),
        "validated complete SigLIP vision checkpoint"
    );
    Ok(())
}
