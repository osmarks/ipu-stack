use half::f16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use std::{fs::File, path::Path};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model metadata error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("invalid SigLIP model: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SiglipVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_eps: f32,
    pub num_attention_heads: usize,
    #[serde(default = "default_image_channels")]
    pub num_channels: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
}

const fn default_layer_norm_epsilon() -> f32 {
    1e-6
}

const fn default_image_channels() -> usize {
    3
}

#[derive(Deserialize)]
struct SiglipConfigFile {
    vision_config: SiglipVisionConfig,
}

pub struct SiglipWeights {
    tensors: TensorArchive,
    pub config: SiglipVisionConfig,
}

pub struct TensorArchive {
    mapping: Mmap,
}

impl TensorArchive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // The mapping is read-only and remains owned by the archive for every tensor view.
        let mapping = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&mapping)?;
        Ok(Self { mapping })
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.with_tensor(name, |tensor| Ok(tensor.shape().to_vec()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.with_tensor(name, |tensor| {
            if tensor.dtype() != Dtype::F32 {
                return Err(ModelError::Invalid(format!(
                    "{name} has {:?} data, expected F32",
                    tensor.dtype()
                )));
            }
            Ok(tensor
                .data()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect())
        })
    }

    pub fn tensor_f16(&self, name: &str) -> Result<Vec<f16>> {
        Ok(self
            .tensor_f32(name)?
            .into_iter()
            .map(f16::from_f32)
            .collect())
    }

    fn with_tensor<T>(
        &self,
        name: &str,
        operation: impl FnOnce(TensorView<'_>) -> Result<T>,
    ) -> Result<T> {
        let tensors = SafeTensors::deserialize(&self.mapping)?;
        operation(tensors.tensor(name)?)
    }
}

impl SiglipWeights {
    pub fn open(model_directory: impl AsRef<Path>) -> Result<Self> {
        let directory = model_directory.as_ref();
        let config: SiglipConfigFile =
            serde_json::from_reader(File::open(directory.join("config.json"))?)?;
        let model = Self {
            tensors: TensorArchive::open(directory.join("model.safetensors"))?,
            config: config.vision_config,
        };
        model.validate()?;
        info!(
            image_size = model.config.image_size,
            patch_size = model.config.patch_size,
            sequence_length = model.sequence_length(),
            hidden_size = model.config.hidden_size,
            layers = model.config.num_hidden_layers,
            heads = model.config.num_attention_heads,
            "loaded SigLIP vision weight index"
        );
        Ok(model)
    }

    pub fn patch_grid(&self) -> usize {
        self.config.image_size / self.config.patch_size
    }

    pub fn sequence_length(&self) -> usize {
        self.patch_grid().pow(2)
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.tensors.tensor_shape(name)
    }

    pub fn tensor_f16(&self, name: &str) -> Result<Vec<f16>> {
        self.tensors.tensor_f16(name)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.tensors.tensor_f32(name)
    }

    pub fn vision_parameter_count(&self) -> Result<usize> {
        let tensors = SafeTensors::deserialize(&self.tensors.mapping)?;
        tensors
            .names()
            .into_iter()
            .filter(|name| name.starts_with("vision_model."))
            .try_fold(0usize, |total, name| {
                let elements = tensors.tensor(name)?.shape().iter().product::<usize>();
                total
                    .checked_add(elements)
                    .ok_or_else(|| ModelError::Invalid("vision parameter count overflow".into()))
            })
    }

    pub fn layer_name(&self, layer: usize, suffix: &str) -> Result<String> {
        if layer >= self.config.num_hidden_layers {
            return Err(ModelError::Invalid(format!(
                "encoder layer {layer} is outside 0..{}",
                self.config.num_hidden_layers
            )));
        }
        Ok(format!("vision_model.encoder.layers.{layer}.{suffix}"))
    }

    fn validate(&self) -> Result<()> {
        let config = &self.config;
        if config.image_size == 0
            || config.patch_size == 0
            || config.hidden_size == 0
            || config.num_hidden_layers == 0
            || config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
            return Err(ModelError::Invalid(
                "vision dimensions and layer counts must be positive and heads must divide hidden size"
                    .into(),
            ));
        }
        let patch_elements = config.num_channels * config.patch_size.pow(2);
        self.require_shape(
            "vision_model.embeddings.patch_embedding.weight",
            &[
                config.hidden_size,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ],
        )?;
        self.require_shape(
            "vision_model.embeddings.patch_embedding.bias",
            &[config.hidden_size],
        )?;
        self.require_shape(
            "vision_model.embeddings.position_embedding.weight",
            &[self.sequence_length(), config.hidden_size],
        )?;
        if patch_elements == 0 {
            return Err(ModelError::Invalid("patch projection is empty".into()));
        }
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("vision_model.encoder.layers.{layer}");
            for norm in ["layer_norm1", "layer_norm2"] {
                self.require_shape(&format!("{prefix}.{norm}.weight"), &[config.hidden_size])?;
                self.require_shape(&format!("{prefix}.{norm}.bias"), &[config.hidden_size])?;
            }
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                self.require_shape(
                    &format!("{prefix}.self_attn.{projection}.weight"),
                    &[config.hidden_size, config.hidden_size],
                )?;
                self.require_shape(
                    &format!("{prefix}.self_attn.{projection}.bias"),
                    &[config.hidden_size],
                )?;
            }
            self.require_shape(
                &format!("{prefix}.mlp.fc1.weight"),
                &[config.intermediate_size, config.hidden_size],
            )?;
            self.require_shape(
                &format!("{prefix}.mlp.fc1.bias"),
                &[config.intermediate_size],
            )?;
            self.require_shape(
                &format!("{prefix}.mlp.fc2.weight"),
                &[config.hidden_size, config.intermediate_size],
            )?;
            self.require_shape(&format!("{prefix}.mlp.fc2.bias"), &[config.hidden_size])?;
        }
        self.require_shape("vision_model.post_layernorm.weight", &[config.hidden_size])?;
        self.require_shape("vision_model.post_layernorm.bias", &[config.hidden_size])?;
        self.require_shape("vision_model.head.probe", &[1, 1, config.hidden_size])?;
        self.require_shape(
            "vision_model.head.attention.in_proj_weight",
            &[config.hidden_size * 3, config.hidden_size],
        )?;
        self.require_shape(
            "vision_model.head.attention.in_proj_bias",
            &[config.hidden_size * 3],
        )?;
        self.require_shape(
            "vision_model.head.attention.out_proj.weight",
            &[config.hidden_size, config.hidden_size],
        )?;
        self.require_shape(
            "vision_model.head.attention.out_proj.bias",
            &[config.hidden_size],
        )?;
        self.require_shape("vision_model.head.layernorm.weight", &[config.hidden_size])?;
        self.require_shape("vision_model.head.layernorm.bias", &[config.hidden_size])?;
        self.require_shape(
            "vision_model.head.mlp.fc1.weight",
            &[config.intermediate_size, config.hidden_size],
        )?;
        self.require_shape(
            "vision_model.head.mlp.fc1.bias",
            &[config.intermediate_size],
        )?;
        self.require_shape(
            "vision_model.head.mlp.fc2.weight",
            &[config.hidden_size, config.intermediate_size],
        )?;
        self.require_shape("vision_model.head.mlp.fc2.bias", &[config.hidden_size])?;
        Ok(())
    }

    fn require_shape(&self, name: &str, expected: &[usize]) -> Result<()> {
        let observed = self.tensor_shape(name)?;
        if observed != expected {
            return Err(ModelError::Invalid(format!(
                "{name} has shape {observed:?}, expected {expected:?}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_grid_uses_valid_convolution_extent() {
        let config = SiglipVisionConfig {
            hidden_size: 1152,
            image_size: 384,
            intermediate_size: 4304,
            layer_norm_eps: 1e-6,
            num_attention_heads: 16,
            num_channels: 3,
            num_hidden_layers: 27,
            patch_size: 14,
        };
        assert_eq!(config.image_size / config.patch_size, 27);
        assert_eq!((config.image_size / config.patch_size).pow(2), 729);
    }
}
