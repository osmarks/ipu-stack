//! ViT benchmarks and explicitly approximate model-capacity probes.
use anyhow::{Result, ensure};
use ipu_codegen::{AxisFactorView, ComputeGraph, GraphInput, ValueId};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Model {
    #[default]
    SiglipSo400m,
    /// PE-Core-L14-336 dimensions and bookends; omits RoPE and retains epsilon 1e-6.
    PeCoreL14Capacity,
}

#[derive(clap::Args)]
pub(crate) struct Options {
    #[arg(long, value_enum, default_value_t = Model::SiglipSo400m)]
    pub vit_model: Model,
    #[arg(long, default_value_t = 1)]
    pub vit_batch: u32,
    /// Distinct encoder layers, represented by one structured Repeat when >1.
    #[arg(long, default_value_t = 1)]
    pub vit_layers: u32,
    /// Use a small image/width to validate the complete graph quickly.
    #[arg(long)]
    pub vit_small: bool,
    /// Override image side length while retaining the selected model width.
    #[arg(long)]
    pub vit_image_size: Option<u32>,
}

pub(crate) fn build(options: &Options, fuse_qkv: bool) -> Result<ComputeGraph> {
    let pe = options.vit_model == Model::PeCoreL14Capacity;
    let (image, patch, width, hidden, heads) = match (pe, options.vit_small) {
        (false, false) => (378, 14, 1152, 4304, 16),
        // Retain So400m's 72-wide heads, including their packed-layout tails.
        (false, true) => (28, 14, 144, 288, 2),
        (true, false) => (336, 14, 1024, 4096, 16),
        (true, true) => (28, 14, 128, 512, 2),
    };
    if pe {
        tracing::warn!(
            "PE capacity probe: RoPE omitted; layernorm epsilon remains 1e-6 instead of PE's 1e-5; numerical results are not PE model accuracy"
        );
    }
    ensure!(options.vit_batch > 0, "--vit-batch must be nonzero");
    ensure!(options.vit_layers > 0, "--vit-layers must be nonzero");
    let image = options.vit_image_size.unwrap_or(image);
    ensure!(
        image > 0 && image.is_multiple_of(patch),
        "ViT image side must be a positive multiple of {patch}"
    );
    let tokens = (image / patch)
        .checked_mul(image / patch)
        .ok_or_else(|| anyhow::anyhow!("ViT token count overflow"))?;
    let tokens = tokens + u32::from(pe);
    let mut g = ComputeGraph::new();
    // Nonoverlapping convolution is GEMM on host-packed NHWC image patches.
    // The host supplies every pixel, in [patch_y, patch_x, y, x, channel] order.
    let image = g.host_input(
        if pe {
            "vit.pe.image.patches"
        } else {
            "vit.image.patches"
        },
        [options.vit_batch, tokens, patch * patch * 3],
    )?;
    let mut x = if pe {
        // A zero patch represents the class slot. With a bias-free projection,
        // folding the learned class embedding into position[0] is exact.
        let weight = g.parameter("vit.embedding.weight", [patch * patch * 3, width])?;
        g.gemm(image, weight)?
    } else {
        dense(&mut g, image, "embedding", patch * patch * 3, width)?
    };
    let position = g.parameter("vit.position", [1, tokens, width])?;
    x = g.add(x, position)?;
    if pe {
        x = norm(&mut g, x, "embedding.norm", width)?;
    }
    x = if options.vit_layers == 1 {
        encoder(&mut g, x, width, hidden, heads, fuse_qkv)?
    } else {
        // Build the body once, then bind each parameter to an iterated sequence.
        // Import its ordinary operations through the same checked graph API.
        let mut template = ComputeGraph::new();
        let state = template.host_input("state", [options.vit_batch, tokens, width])?;
        let result = encoder(&mut template, state, width, hidden, heads, fuse_qkv)?;
        let parameters = &template.inputs()[1..];
        let mut sequences = Vec::new();
        for input in parameters {
            let mut layers = Vec::new();
            for layer in 0..options.vit_layers {
                let name =
                    input
                        .name
                        .replacen("vit.encoder.", &format!("vit.encoder.layer{layer}."), 1);
                layers.push(g.parameter(name, input.shape.0.clone())?);
            }
            sequences.push(g.value_sequence(input.name.clone(), layers)?);
        }
        g.repeat(options.vit_layers, [x], [], sequences, |body, arguments| {
            let mut values = std::collections::BTreeMap::from([(state, arguments.carried[0])]);
            values.extend(
                parameters
                    .iter()
                    .zip(&arguments.iterated)
                    .map(|(input, &value)| (input.value, value)),
            );
            for operation in template.operations() {
                let outputs = body.operation(
                    operation.kind.clone(),
                    operation.inputs.iter().map(|value| values[value]),
                    operation
                        .results
                        .iter()
                        .map(|value| template.value_shape(*value).unwrap().clone()),
                )?;
                values.extend(operation.results.iter().copied().zip(outputs));
            }
            Ok(vec![values[&result]])
        })?[0]
    };
    x = norm(&mut g, x, "encoder.final_norm", width)?;

    // A replicated probe is expressed as a batch-shaped parameter. Input
    // generation repeats the same learned vector across batches.
    let probe = g.parameter("vit.map.probe", [options.vit_batch, 1, width])?;
    x = attention(
        &mut g,
        probe,
        x,
        "map.attention",
        width,
        if pe { 8 } else { heads },
        fuse_qkv,
    )?;
    let normalized = norm(&mut g, x, "map.norm", width)?;
    let update = mlp(&mut g, normalized, "map.mlp", width, hidden)?;
    x = g.add(x, update)?;
    if pe {
        let projection = g.parameter("vit.projection.weight", [width, width])?;
        x = g.gemm(x, projection)?;
    }
    g.set_outputs([x])?; // [batch, 1, width], singleton probe axis retained.
    Ok(g)
}

fn encoder(
    g: &mut ComputeGraph,
    mut x: ValueId,
    width: u32,
    hidden: u32,
    heads: u32,
    fuse_qkv: bool,
) -> Result<ValueId> {
    let normalized = norm(g, x, "encoder.attention_norm", width)?;
    let attended = attention(
        g,
        normalized,
        normalized,
        "encoder.attention",
        width,
        heads,
        fuse_qkv,
    )?;
    x = g.add(x, attended)?;
    let normalized = norm(g, x, "encoder.mlp_norm", width)?;
    let update = mlp(g, normalized, "encoder.mlp", width, hidden)?;
    x = g.add(x, update)?;
    Ok(x)
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
    fuse_qkv: bool,
) -> Result<ValueId> {
    let (query, key, value) = if fuse_qkv {
        // Only projections of the same activation can share a GEMM. MAP's
        // learned query remains separate from its image-key/value projection.
        let shared_query = q == kv;
        let suffix = if shared_query { "qkv" } else { "kv" };
        let count = if shared_query { 3 } else { 2 };
        let projected = dense(g, kv, &format!("{name}.{suffix}"), width, count * width)?;
        let query = if shared_query {
            g.slice(projected, 2, 0, width)?
        } else {
            dense(g, q, &format!("{name}.query"), width, width)?
        };
        let key = g.slice(projected, 2, (count - 2) * width, width)?;
        let value = g.slice(projected, 2, (count - 1) * width, width)?;
        (query, key, value)
    } else {
        (
            dense(g, q, &format!("{name}.query"), width, width)?,
            dense(g, kv, &format!("{name}.key"), width, width)?,
            dense(g, kv, &format!("{name}.value"), width, width)?,
        )
    };
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
    if input.name == "vit.pe.image.patches"
        && index % (u64::from(input.shape.0[1]) * u64::from(input.shape.0[2]))
            < u64::from(input.shape.0[2])
    {
        return Some(0.0);
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
        // Concatenation does not change an individual projection's fan-out.
        let output = if input.name.ends_with(".qkv.weight") {
            width / 3
        } else if input.name.ends_with(".kv.weight") {
            width / 2
        } else {
            width
        };
        (2.0 / (input.shape.0[0] + output) as f32).sqrt()
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
    fn pe_capacity_includes_class_token_and_different_bookends() -> Result<()> {
        let graph = build(
            &Options {
                vit_model: Model::PeCoreL14Capacity,
                vit_batch: 2,
                vit_layers: 24,
                vit_small: false,
                vit_image_size: None,
            },
            true,
        )?;
        let image = &graph.inputs()[0];
        assert_eq!(image.shape.0, [2, 577, 588]);
        for batch in 0..2 {
            let start = batch * 577 * 588;
            assert_eq!(random_input(image, 7, start), Some(0.0));
            assert_eq!(random_input(image, 7, start + 587), Some(0.0));
            assert_ne!(random_input(image, 7, start + 588), Some(0.0));
        }
        assert_eq!(
            graph.value_shape(graph.outputs()[0]).unwrap().0,
            [2, 1, 1024]
        );
        assert!(
            !graph
                .inputs()
                .iter()
                .any(|i| i.name == "vit.embedding.bias")
        );
        assert!(
            graph
                .inputs()
                .iter()
                .any(|i| i.name == "vit.embedding.norm.scale")
        );
        assert!(
            graph
                .inputs()
                .iter()
                .any(|i| i.name == "vit.projection.weight")
        );
        let pool = graph
            .operations()
            .iter()
            .find(|op| matches!(op.kind, OperationKind::FlashAttention(_)))
            .unwrap();
        // Eight pool heads (128 channels), versus sixteen encoder heads (64).
        assert_eq!(graph.value_shape(pool.inputs[0]).unwrap().0, [16, 1, 128]);
        // Includes the deliberately batch-expanded probe; class+position are folded.
        let parameters: u64 = graph
            .inputs()
            .iter()
            .filter(|i| i.kind == ipu_codegen::GraphInputKind::Parameter)
            .map(|i| i.shape.elements())
            .sum();
        assert_eq!(parameters, 317_151_232);
        Ok(())
    }

    #[test]
    fn repeated_encoder_has_distinct_parameters_and_one_time_bookends() -> Result<()> {
        for fused in [false, true] {
            let graph = build(
                &Options {
                    vit_model: Model::SiglipSo400m,
                    vit_batch: 1,
                    vit_layers: 2,
                    vit_small: true,
                    vit_image_size: None,
                },
                fused,
            )?;
            let repeats = graph
                .operations()
                .iter()
                .filter_map(|op| match &op.kind {
                    OperationKind::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(repeats.len(), 1);
            assert_eq!(repeats[0].count, 2);
            assert_eq!(graph.sequences().len(), if fused { 12 } else { 16 });
            for sequence in graph.sequences() {
                assert_eq!(sequence.values.len(), 2);
                assert_ne!(sequence.values[0], sequence.values[1]);
            }
            assert_eq!(
                repeats[0]
                    .body
                    .operations
                    .iter()
                    .filter(|op| matches!(op.kind, OperationKind::LayerNorm))
                    .count(),
                2
            );
            assert_eq!(
                graph
                    .operations()
                    .iter()
                    .filter(|op| matches!(op.kind, OperationKind::LayerNorm))
                    .count(),
                2
            );
            assert_eq!(
                graph.value_shape(graph.outputs()[0]).unwrap().0,
                [1, 1, 144]
            );
        }
        Ok(())
    }

    #[test]
    fn complete_single_layer_topology_and_shapes() -> Result<()> {
        for fused in [false, true] {
            let g = build(
                &Options {
                    vit_model: Model::SiglipSo400m,
                    vit_batch: 1,
                    vit_layers: 1,
                    vit_small: false,
                    vit_image_size: None,
                },
                fused,
            )?;
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
                if fused { 10 } else { 13 }
            );
            let probe = g
                .inputs()
                .iter()
                .find(|i| i.name == "vit.map.probe")
                .unwrap();
            assert_eq!(probe.shape.0, [1, 1, 1152]);
            if fused {
                for (name, shape) in [
                    ("vit.encoder.attention.qkv.weight", [1152, 3456]),
                    ("vit.map.attention.kv.weight", [1152, 2304]),
                ] {
                    let weight = g.inputs().iter().find(|i| i.name == name).unwrap();
                    assert_eq!(weight.shape.0, shape);
                }
            }
        }
        Ok(())
    }
}
