use crate::CompileError;
use ipu_package::{
    IPU21_INTERLEAVED_MEMORY_BASE, IPU21_INTERLEAVED_MEMORY_LIMIT, TILE_MEMORY_BASE,
    TILE_MEMORY_SIZE,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use tracing::info;

const FP16_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiglipVisionConfig {
    pub image_size: u16,
    pub patch_size: u16,
    pub channels: u16,
    pub hidden_size: u16,
    pub intermediate_size: u16,
    pub layers: u16,
    pub heads: u16,
    pub matrix_padding: u16,
}

impl Default for SiglipVisionConfig {
    fn default() -> Self {
        Self {
            image_size: 384,
            patch_size: 14,
            channels: 3,
            hidden_size: 1152,
            intermediate_size: 4304,
            layers: 27,
            heads: 16,
            matrix_padding: 64,
        }
    }
}

impl SiglipVisionConfig {
    pub fn patch_grid(self) -> u16 {
        self.image_size / self.patch_size
    }

    pub fn sequence_length(self) -> u32 {
        u32::from(self.patch_grid()).pow(2)
    }

    fn validate(self) -> Result<(), CompileError> {
        if self.image_size == 0
            || self.patch_size == 0
            || self.channels == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.layers == 0
            || self.heads == 0
            || self.matrix_padding == 0
            || !self.matrix_padding.is_power_of_two()
            || !self.hidden_size.is_multiple_of(self.heads)
        {
            return Err(CompileError::Graph(
                "invalid SigLIP vision configuration".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiglipWeightStage {
    Embedding,
    Encoder {
        layer: u16,
        operation: EncoderWeightOperation,
    },
    PostLayerNorm,
    MapHead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderWeightOperation {
    LayerNorm1,
    QueryKeyValue,
    AttentionOutput,
    LayerNorm2,
    MlpInput,
    MlpOutput,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiglipWeightTensor {
    pub name: String,
    pub stage: SiglipWeightStage,
    pub logical_shape: Vec<u32>,
    pub resident_shape: Vec<u32>,
    pub bytes: u64,
    pub layout: SiglipWeightLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiglipWeightLayout {
    Linear,
    AmpB16x16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiglipResidencyOptions {
    pub tile_count: u16,
    /// Executable bytes actually occupied from the start of each tile's SRAM.
    pub code_bytes_per_tile: u32,
    /// Exchange SRAM left available for transient send/receive operands.
    pub exchange_scratch_bytes_per_tile: u32,
    /// Normal SRAM retained for activations and non-interleaved kernel state.
    pub working_bytes_per_tile: u32,
    pub shard_alignment: u32,
}

impl Default for SiglipResidencyOptions {
    fn default() -> Self {
        Self {
            tile_count: 1472,
            code_bytes_per_tile: 4096,
            exchange_scratch_bytes_per_tile: 12 * 1024,
            working_bytes_per_tile: 12 * 1024,
            shard_alignment: 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SiglipMemoryClass {
    CodeSlack,
    Normal,
    BootstrapExchange,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiglipWeightShard {
    pub tensor: usize,
    pub tensor_offset: u64,
    pub tile: u16,
    pub address: u32,
    pub bytes: u32,
    pub memory_class: SiglipMemoryClass,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiglipMemoryPlan {
    pub config: SiglipVisionConfig,
    pub weights: Vec<SiglipWeightTensor>,
    pub shards: Vec<SiglipWeightShard>,
    pub encoder_bytes: u64,
    pub boundary_map_bytes: u64,
    pub normal_capacity: u64,
    pub bootstrap_capacity: u64,
    pub bootstrap_bytes: u64,
    pub per_tile_bytes: BTreeMap<u16, u64>,
}

#[derive(Clone, Copy)]
struct Arena {
    base: u32,
    limit: u32,
    class: SiglipMemoryClass,
}

#[derive(Clone)]
struct TileCursor {
    arenas: Vec<Arena>,
    addresses: Vec<u32>,
}

impl TileCursor {
    fn new(arenas: Vec<Arena>) -> Self {
        let addresses = arenas.iter().map(|arena| arena.base).collect();
        Self { arenas, addresses }
    }

    fn allocate(
        &mut self,
        maximum: u32,
        alignment: u32,
        contiguous: bool,
    ) -> Option<(u32, u32, SiglipMemoryClass)> {
        for (index, arena) in self.arenas.iter().copied().enumerate() {
            let address = align(self.addresses[index].max(arena.base), alignment);
            if address < arena.limit {
                let available = arena.limit - address;
                if !contiguous || available >= maximum {
                    let bytes = maximum.min(available);
                    let result = (address, bytes, arena.class);
                    self.addresses[index] = address + bytes;
                    return Some(result);
                }
            }
        }
        None
    }
}

pub fn plan_siglip_memory(
    config: SiglipVisionConfig,
    options: SiglipResidencyOptions,
) -> Result<SiglipMemoryPlan, CompileError> {
    config.validate()?;
    validate_options(options)?;
    let weights = weight_manifest(config);
    let encoder_bytes = weights
        .iter()
        .filter(|weight| weight.stage != SiglipWeightStage::MapHead)
        .map(|weight| weight.bytes)
        .sum();
    let boundary_map_bytes = weights
        .iter()
        .filter(|weight| weight.stage == SiglipWeightStage::MapHead)
        .map(|weight| weight.bytes)
        .sum();

    let exchange_limit = ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
    let code_end = align(
        TILE_MEMORY_BASE
            .checked_add(options.code_bytes_per_tile)
            .ok_or_else(|| CompileError::Memory("SigLIP code range overflow".into()))?,
        options.shard_alignment,
    );
    let bootstrap_exchange_base = align(
        ipu_exchange::EXCHANGE_WINDOW_BASE + options.exchange_scratch_bytes_per_tile,
        options.shard_alignment,
    );
    let tile_limit = TILE_MEMORY_BASE + TILE_MEMORY_SIZE;
    let resident_high_limit = tile_limit
        .checked_sub(options.working_bytes_per_tile)
        .ok_or_else(|| CompileError::Memory("SigLIP working-set reservation overflow".into()))?;
    let arenas = vec![
        Arena {
            base: code_end,
            limit: ipu_exchange::EXCHANGE_WINDOW_BASE,
            class: SiglipMemoryClass::CodeSlack,
        },
        Arena {
            base: exchange_limit,
            limit: IPU21_INTERLEAVED_MEMORY_BASE,
            class: SiglipMemoryClass::Normal,
        },
        Arena {
            base: IPU21_INTERLEAVED_MEMORY_LIMIT,
            limit: resident_high_limit,
            class: SiglipMemoryClass::Normal,
        },
        Arena {
            base: bootstrap_exchange_base,
            limit: exchange_limit,
            class: SiglipMemoryClass::BootstrapExchange,
        },
    ];
    let normal_per_tile: u64 = arenas
        .iter()
        .filter(|arena| arena.class != SiglipMemoryClass::BootstrapExchange)
        .map(|arena| u64::from(arena.limit - arena.base))
        .sum();
    let bootstrap_per_tile = u64::from(exchange_limit - bootstrap_exchange_base);
    let normal_capacity = normal_per_tile * u64::from(options.tile_count);
    let bootstrap_capacity = (normal_per_tile + bootstrap_per_tile) * u64::from(options.tile_count);
    if encoder_bytes > bootstrap_capacity {
        return Err(CompileError::Memory(format!(
            "SigLIP encoder requires {encoder_bytes} bytes but resident and bootstrap arenas hold {bootstrap_capacity}"
        )));
    }

    let mut cursors = vec![TileCursor::new(arenas); usize::from(options.tile_count)];
    let mut shards = Vec::new();
    let mut tile_loads = (0..options.tile_count)
        .map(|tile| Reverse((0u64, tile)))
        .collect::<BinaryHeap<_>>();
    for (tensor_index, tensor) in weights.iter().enumerate() {
        if tensor.stage == SiglipWeightStage::MapHead {
            continue;
        }
        let mut offset = 0u64;
        while offset < tensor.bytes {
            let desired = u32::try_from((tensor.bytes - offset).min(8 * 1024))
                .map_err(|_| CompileError::Memory("SigLIP shard size overflow".into()))?;
            let contiguous = tensor.layout == SiglipWeightLayout::AmpB16x16;
            let mut skipped = Vec::new();
            let (tile_bytes, tile, first_allocation) = loop {
                let Some(Reverse((tile_bytes, tile))) = tile_loads.pop() else {
                    return Err(CompileError::Memory(format!(
                        "SigLIP encoder has no contiguous tile region for {}",
                        tensor.name
                    )));
                };
                if let Some(allocation) = cursors[usize::from(tile)].allocate(
                    desired,
                    options.shard_alignment,
                    contiguous,
                ) {
                    break (tile_bytes, tile, allocation);
                }
                skipped.push(Reverse((tile_bytes, tile)));
            };
            tile_loads.extend(skipped);
            let mut remaining = desired;
            let mut next_allocation = Some(first_allocation);
            while remaining != 0 {
                let (address, bytes, memory_class) =
                    if let Some(allocation) = next_allocation.take() {
                        allocation
                    } else {
                        cursors[usize::from(tile)]
                        .allocate(remaining, options.shard_alignment, contiguous)
                        .ok_or_else(|| {
                            CompileError::Memory(format!(
                                "SigLIP encoder allocation exhausted tile {tile} while placing {}",
                                tensor.name
                            ))
                        })?
                    };
                shards.push(SiglipWeightShard {
                    tensor: tensor_index,
                    tensor_offset: offset,
                    tile,
                    address,
                    bytes,
                    memory_class,
                });
                offset += u64::from(bytes);
                remaining -= bytes;
            }
            tile_loads.push(Reverse((tile_bytes + u64::from(desired), tile)));
        }
    }
    let per_tile_bytes = tile_loads
        .into_iter()
        .map(|Reverse((bytes, tile))| (tile, bytes))
        .collect();
    let bootstrap_bytes = shards
        .iter()
        .filter(|shard| shard.memory_class == SiglipMemoryClass::BootstrapExchange)
        .map(|shard| u64::from(shard.bytes))
        .sum();

    info!(
        tile_count = options.tile_count,
        encoder_bytes,
        boundary_map_bytes,
        normal_capacity,
        bootstrap_capacity,
        bootstrap_bytes,
        code_bytes_per_tile = options.code_bytes_per_tile,
        working_bytes_per_tile = options.working_bytes_per_tile,
        exchange_scratch_bytes_per_tile = options.exchange_scratch_bytes_per_tile,
        "planned resident SigLIP weights"
    );

    Ok(SiglipMemoryPlan {
        config,
        weights,
        shards,
        encoder_bytes,
        boundary_map_bytes,
        normal_capacity,
        bootstrap_capacity,
        bootstrap_bytes,
        per_tile_bytes,
    })
}

fn validate_options(options: SiglipResidencyOptions) -> Result<(), CompileError> {
    let code_capacity = ipu_exchange::EXCHANGE_WINDOW_BASE - TILE_MEMORY_BASE;
    if options.tile_count == 0
        || options.code_bytes_per_tile > code_capacity
        || options.exchange_scratch_bytes_per_tile > ipu_exchange::EXCHANGE_WINDOW_BYTES
        || options.working_bytes_per_tile
            > TILE_MEMORY_BASE + TILE_MEMORY_SIZE - IPU21_INTERLEAVED_MEMORY_LIMIT
        || options.shard_alignment == 0
        || !options.shard_alignment.is_power_of_two()
    {
        return Err(CompileError::Graph(
            "invalid SigLIP residency options".into(),
        ));
    }
    Ok(())
}

fn weight_manifest(config: SiglipVisionConfig) -> Vec<SiglipWeightTensor> {
    let hidden = u32::from(config.hidden_size);
    let intermediate = u32::from(config.intermediate_size);
    let padded_intermediate = pad(intermediate, u32::from(config.matrix_padding));
    let patch_features = u32::from(config.patch_size).pow(2) * u32::from(config.channels);
    let sequence = config.sequence_length();
    let mut weights = vec![
        matrix(
            "embeddings.patch_embedding.weight",
            SiglipWeightStage::Embedding,
            &[hidden, patch_features],
            &[
                hidden,
                pad(patch_features, u32::from(config.matrix_padding)),
            ],
        ),
        tensor(
            "embeddings.patch_embedding.bias",
            SiglipWeightStage::Embedding,
            &[hidden],
            &[hidden],
        ),
        tensor(
            "embeddings.position_embedding.weight",
            SiglipWeightStage::Embedding,
            &[sequence, hidden],
            &[sequence, hidden],
        ),
    ];
    for layer in 0..config.layers {
        let stage = |operation| SiglipWeightStage::Encoder { layer, operation };
        weights.extend([
            tensor_pair(
                &format!("encoder.layers.{layer}.layer_norm1"),
                stage(EncoderWeightOperation::LayerNorm1),
                hidden,
            ),
            matrix(
                &format!("encoder.layers.{layer}.self_attn.qkv.weight"),
                stage(EncoderWeightOperation::QueryKeyValue),
                &[hidden * 3, hidden],
                &[hidden * 3, hidden],
            ),
            tensor(
                &format!("encoder.layers.{layer}.self_attn.qkv.bias"),
                stage(EncoderWeightOperation::QueryKeyValue),
                &[hidden * 3],
                &[hidden * 3],
            ),
            matrix(
                &format!("encoder.layers.{layer}.self_attn.out_proj.weight"),
                stage(EncoderWeightOperation::AttentionOutput),
                &[hidden, hidden],
                &[hidden, hidden],
            ),
            tensor(
                &format!("encoder.layers.{layer}.self_attn.out_proj.bias"),
                stage(EncoderWeightOperation::AttentionOutput),
                &[hidden],
                &[hidden],
            ),
            tensor_pair(
                &format!("encoder.layers.{layer}.layer_norm2"),
                stage(EncoderWeightOperation::LayerNorm2),
                hidden,
            ),
            matrix(
                &format!("encoder.layers.{layer}.mlp.fc1.weight"),
                stage(EncoderWeightOperation::MlpInput),
                &[intermediate, hidden],
                &[padded_intermediate, hidden],
            ),
            tensor(
                &format!("encoder.layers.{layer}.mlp.fc1.bias"),
                stage(EncoderWeightOperation::MlpInput),
                &[intermediate],
                &[padded_intermediate],
            ),
            matrix(
                &format!("encoder.layers.{layer}.mlp.fc2.weight"),
                stage(EncoderWeightOperation::MlpOutput),
                &[hidden, intermediate],
                &[hidden, padded_intermediate],
            ),
            tensor(
                &format!("encoder.layers.{layer}.mlp.fc2.bias"),
                stage(EncoderWeightOperation::MlpOutput),
                &[hidden],
                &[hidden],
            ),
        ]);
    }
    weights.push(tensor_pair(
        "post_layernorm",
        SiglipWeightStage::PostLayerNorm,
        hidden,
    ));
    weights.extend(map_head_weights(hidden, intermediate, padded_intermediate));
    weights
}

fn tensor_pair(prefix: &str, stage: SiglipWeightStage, elements: u32) -> SiglipWeightTensor {
    tensor(
        &format!("{prefix}.weight_and_bias"),
        stage,
        &[2, elements],
        &[2, elements],
    )
}

fn map_head_weights(
    hidden: u32,
    intermediate: u32,
    padded_intermediate: u32,
) -> Vec<SiglipWeightTensor> {
    let stage = SiglipWeightStage::MapHead;
    vec![
        tensor("head.probe", stage, &[1, hidden], &[1, hidden]),
        matrix(
            "head.attention.in_proj_weight",
            stage,
            &[hidden * 3, hidden],
            &[hidden * 3, hidden],
        ),
        tensor(
            "head.attention.in_proj_bias",
            stage,
            &[hidden * 3],
            &[hidden * 3],
        ),
        matrix(
            "head.attention.out_proj.weight",
            stage,
            &[hidden, hidden],
            &[hidden, hidden],
        ),
        tensor("head.attention.out_proj.bias", stage, &[hidden], &[hidden]),
        tensor_pair("head.layernorm", stage, hidden),
        matrix(
            "head.mlp.fc1.weight",
            stage,
            &[intermediate, hidden],
            &[padded_intermediate, hidden],
        ),
        tensor(
            "head.mlp.fc1.bias",
            stage,
            &[intermediate],
            &[padded_intermediate],
        ),
        matrix(
            "head.mlp.fc2.weight",
            stage,
            &[hidden, intermediate],
            &[hidden, padded_intermediate],
        ),
        tensor("head.mlp.fc2.bias", stage, &[hidden], &[hidden]),
    ]
}

fn tensor(
    name: &str,
    stage: SiglipWeightStage,
    logical_shape: &[u32],
    resident_shape: &[u32],
) -> SiglipWeightTensor {
    SiglipWeightTensor {
        name: name.into(),
        stage,
        logical_shape: logical_shape.into(),
        resident_shape: resident_shape.into(),
        bytes: resident_shape
            .iter()
            .map(|&value| u64::from(value))
            .product::<u64>()
            * FP16_BYTES,
        layout: SiglipWeightLayout::Linear,
    }
}

fn matrix(
    name: &str,
    stage: SiglipWeightStage,
    logical_shape: &[u32],
    resident_shape: &[u32],
) -> SiglipWeightTensor {
    SiglipWeightTensor {
        layout: SiglipWeightLayout::AmpB16x16,
        ..tensor(name, stage, logical_shape, resident_shape)
    }
}

fn pad(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn align(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_has_expected_structure_and_balanced_shards() {
        let config = SiglipVisionConfig::default();
        let plan = plan_siglip_memory(config, SiglipResidencyOptions::default()).unwrap();
        assert_eq!(config.patch_grid(), 27);
        assert_eq!(config.sequence_length(), 729);
        assert!(plan.encoder_bytes > plan.normal_capacity);
        assert!(plan.encoder_bytes <= plan.bootstrap_capacity);
        let minimum_bootstrap = plan.encoder_bytes - plan.normal_capacity;
        assert!(plan.bootstrap_bytes >= minimum_bootstrap);
        assert!(plan.bootstrap_bytes < plan.bootstrap_capacity - plan.normal_capacity);
        assert_eq!(
            plan.shards
                .iter()
                .map(|shard| u64::from(shard.bytes))
                .sum::<u64>(),
            plan.encoder_bytes
        );
        let minimum = plan.per_tile_bytes.values().min().unwrap();
        let maximum = plan.per_tile_bytes.values().max().unwrap();
        assert!(
            maximum - minimum <= 8 * 1024,
            "per-tile weight imbalance is {} bytes",
            maximum - minimum
        );
        assert!(plan.weights.iter().any(|weight| {
            weight.stage == SiglipWeightStage::MapHead && weight.name == "head.probe"
        }));
    }

    #[test]
    fn model_dimensions_are_parametric() {
        let config = SiglipVisionConfig {
            image_size: 224,
            patch_size: 16,
            hidden_size: 768,
            intermediate_size: 3072,
            layers: 12,
            heads: 12,
            ..SiglipVisionConfig::default()
        };
        let plan = plan_siglip_memory(config, SiglipResidencyOptions::default()).unwrap();
        assert_eq!(config.sequence_length(), 196);
        assert_eq!(
            plan.weights
                .iter()
                .filter(|weight| matches!(weight.stage, SiglipWeightStage::Encoder { .. }))
                .count(),
            usize::from(config.layers) * 10
        );
        assert_eq!(plan.bootstrap_bytes, 0);
    }

    #[test]
    fn insufficient_residency_reports_an_error() {
        let error = plan_siglip_memory(
            SiglipVisionConfig::default(),
            SiglipResidencyOptions {
                tile_count: 100,
                ..SiglipResidencyOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, CompileError::Memory(_)));
    }
}
