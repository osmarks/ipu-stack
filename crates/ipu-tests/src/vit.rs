//! Single-layer Big Vision ViT, including its learned-query MAP pool.
use anyhow::{Result, ensure};
use ipu_codegen::{AxisFactorView, ComputeGraph, GraphInput, ValueId};

#[derive(clap::Args)]
pub(crate) struct Options {
    #[arg(long, default_value_t = 1)]
    pub vit_batch: u32,
    /// Use a small image/width to validate the complete graph quickly.
    #[arg(long)]
    pub vit_small: bool,
    /// Override image side length while retaining the selected model width.
    #[arg(long)]
    pub vit_image_size: Option<u32>,
}

pub(crate) fn build(options: &Options) -> Result<ComputeGraph> {
    let (image, patch, width, hidden, heads) = if options.vit_small {
        // Retain So400m's 72-wide heads, including their packed-layout tails.
        (28, 14, 144, 288, 2)
    } else {
        (378, 14, 1152, 4304, 16)
    };
    ensure!(options.vit_batch > 0, "--vit-batch must be nonzero");
    let image = options.vit_image_size.unwrap_or(image);
    ensure!(
        image > 0 && image.is_multiple_of(patch),
        "ViT image side must be a positive multiple of {patch}"
    );
    let tokens = (image / patch)
        .checked_mul(image / patch)
        .ok_or_else(|| anyhow::anyhow!("ViT token count overflow"))?;
    let mut g = ComputeGraph::new();
    // Nonoverlapping convolution is GEMM on host-packed NHWC image patches.
    // The host supplies every pixel, in [patch_y, patch_x, y, x, channel] order.
    let image = g.host_input(
        "vit.image.patches",
        [options.vit_batch, tokens, patch * patch * 3],
    )?;
    let mut x = dense(&mut g, image, "embedding", patch * patch * 3, width)?;
    let position = g.parameter("vit.position", [1, tokens, width])?;
    x = g.add(x, position)?;
    let normalized = norm(&mut g, x, "encoder.attention_norm", width)?;
    let attended = attention(
        &mut g,
        normalized,
        normalized,
        "encoder.attention",
        width,
        heads,
    )?;
    x = g.add(x, attended)?;
    let normalized = norm(&mut g, x, "encoder.mlp_norm", width)?;
    let update = mlp(&mut g, normalized, "encoder.mlp", width, hidden)?;
    x = g.add(x, update)?;
    x = norm(&mut g, x, "encoder.final_norm", width)?;

    // A replicated probe is expressed as a batch-shaped parameter. Input
    // generation repeats the same learned vector across batches.
    let probe = g.parameter("vit.map.probe", [options.vit_batch, 1, width])?;
    x = attention(&mut g, probe, x, "map.attention", width, heads)?;
    let normalized = norm(&mut g, x, "map.norm", width)?;
    let update = mlp(&mut g, normalized, "map.mlp", width, hidden)?;
    x = g.add(x, update)?;
    g.set_outputs([x])?; // [batch, 1, width], singleton probe axis retained.
    Ok(g)
}

fn dense(g: &mut ComputeGraph, x: ValueId, name: &str, input: u32, output: u32) -> Result<ValueId> {
    let weight = g.parameter(format!("vit.{name}.weight"), [input, output])?;
    let bias = g.parameter(format!("vit.{name}.bias"), [1, 1, output])?;
    let x = g.gemm(x, weight)?;
    Ok(g.add(x, bias)?)
}

fn norm(g: &mut ComputeGraph, x: ValueId, name: &str, width: u32) -> Result<ValueId> {
    let scale = g.parameter(format!("vit.{name}.scale"), [1, 1, width])?;
    let bias = g.parameter(format!("vit.{name}.bias"), [1, 1, width])?;
    Ok(g.layer_norm(x, scale, bias)?)
}

fn mlp(g: &mut ComputeGraph, x: ValueId, name: &str, width: u32, hidden: u32) -> Result<ValueId> {
    let x = dense(g, x, &format!("{name}.up"), width, hidden)?;
    let x = g.gelu(x)?;
    dense(g, x, &format!("{name}.down"), hidden, width)
}

fn attention(
    g: &mut ComputeGraph,
    q: ValueId,
    kv: ValueId,
    name: &str,
    width: u32,
    heads: u32,
) -> Result<ValueId> {
    let query = dense(g, q, &format!("{name}.query"), width, width)?;
    let key = dense(g, kv, &format!("{name}.key"), width, width)?;
    let value = dense(g, kv, &format!("{name}.value"), width, width)?;
    let query = g.split_heads(query, heads)?;
    let key = g.split_heads(key, heads)?;
    let value = g.split_heads(value, heads)?;
    let x = g.flash_attention(query, key, value)?;
    let x = g.view(x, AxisFactorView::new(2, 0, heads).inverse())?;
    dense(g, x, &format!("{name}.output"), width, width)
}

pub(crate) fn random_input(input: &GraphInput, seed: u64, index: u64) -> Option<f32> {
    if !input.name.starts_with("vit.") {
        return None;
    }
    let width = *input.shape.0.last().unwrap();
    if input.name.ends_with(".scale") {
        return Some(1.0);
    }
    let scale = if input.name.ends_with(".bias") {
        1e-6
    } else if input.name == "vit.position" {
        (width as f32).sqrt().recip()
    } else if input.name.ends_with(".weight") {
        (2.0 / (input.shape.0[0] + width) as f32).sqrt()
    } else if input.name == "vit.map.probe" {
        (1.0 / width as f32).sqrt()
    } else {
        0.5
    };
    let index = if input.name == "vit.map.probe" {
        index % u64::from(width)
    } else {
        index
    };
    Some(super::gaussian(seed, index) * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipu_codegen::OperationKind;

    #[test]
    fn complete_single_layer_topology_and_shapes() -> Result<()> {
        let g = build(&Options {
            vit_batch: 1,
            vit_small: false,
            vit_image_size: None,
        })?;
        let image = g
            .inputs()
            .iter()
            .find(|i| i.name == "vit.image.patches")
            .unwrap();
        assert_eq!(image.shape.0, [1, 729, 588]);
        assert_eq!(image.shape.elements(), 378 * 378 * 3);
        assert_eq!(
            g.operations()
                .iter()
                .filter(|o| matches!(o.kind, OperationKind::LayerNorm))
                .count(),
            4
        );
        assert_eq!(
            g.operations()
                .iter()
                .filter(|o| matches!(o.kind, OperationKind::FlashAttention(_)))
                .count(),
            2
        );
        assert_eq!(
            g.operations()
                .iter()
                .filter(|o| matches!(o.kind, OperationKind::Gemm(_)))
                .count(),
            13
        );
        let probe = g
            .inputs()
            .iter()
            .find(|i| i.name == "vit.map.probe")
            .unwrap();
        assert_eq!(probe.shape.0, [1, 1, 1152]);
        Ok(())
    }
}
