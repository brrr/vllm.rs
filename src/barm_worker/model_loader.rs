use anyhow::{Context, Result};
use candle_core::DType;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::core::engine::LLMEngine;
use crate::utils::config::{EngineConfig, SamplingParams};

/// Memory configuration for model loading
#[derive(Clone, Debug, Default)]
pub struct MemoryConfig {
    /// Maximum model context length
    pub max_model_len: Option<usize>,
    /// Maximum concurrent sequences
    pub max_num_seqs: Option<usize>,
    /// KV cache memory fraction (0.0-1.0)
    pub kv_fraction: Option<f32>,
}

/// ModelEngine - Wraps vLLM.rs LLMEngine for barm-worker inference
///
/// This struct manages the lifecycle of the vLLM.rs inference engine,
/// including model loading and inference execution.
#[derive(Clone)]
pub struct ModelEngine {
    /// The vLLM.rs engine instance (None until model is loaded)
    engine: Option<Arc<RwLock<LLMEngine>>>,
    /// Model name
    model_name: String,
    /// Memory configuration
    memory_config: MemoryConfig,
}

impl ModelEngine {
    /// Create a new ModelEngine
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    ///
    /// # Returns
    /// A new ModelEngine instance (engine not loaded yet)
    pub fn new(model_name: String) -> Self {
        Self {
            engine: None,
            model_name,
            memory_config: MemoryConfig::default(),
        }
    }

    /// Create a new ModelEngine with memory configuration
    pub fn with_memory_config(model_name: String, memory_config: MemoryConfig) -> Self {
        Self {
            engine: None,
            model_name,
            memory_config,
        }
    }

    /// Set memory configuration
    pub fn set_memory_config(&mut self, config: MemoryConfig) {
        self.memory_config = config;
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.engine.is_some()
    }

    /// Load model from cached assets
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing model assets (config.json, tokenizer.json, weights/)
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn load_model(&mut self, model_dir: PathBuf) -> Result<()> {
        info!("Loading model from: {:?}", model_dir);

        // Verify required files exist
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !config_path.exists() {
            anyhow::bail!("Config file not found: {:?}", config_path);
        }
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer file not found: {:?}", tokenizer_path);
        }

        // Get the weights directory
        let weights_dir = model_dir.join("weights");
        // Pass the model_dir as weight_path, not the weights subdirectory
        // The downloader will check for weights/shard-000.safetensors within model_dir
        let weight_path = model_dir.to_string_lossy().to_string();

        // Use memory config or sensible defaults
        let max_num_seqs = self.memory_config.max_num_seqs.unwrap_or(4);
        let max_model_len = self.memory_config.max_model_len.unwrap_or(512);
        let kv_fraction = self.memory_config.kv_fraction;

        info!("Using memory config: max_model_len={}, max_num_seqs={}, kv_fraction={:?}",
              max_model_len, max_num_seqs, kv_fraction);

        // Create engine config - point to local files
        // For local weights, use weight_path only (model_id should be None to match
        // the (None, Some(path), None) case in prepare_model_weights)
        let econfig = EngineConfig::new(
            None,                                             // model_id: None for local weights
            Some(weight_path),                                // weight_path: model directory
            None,                                             // weight_file
            None,                                             // hf_token
            None,                                             // hf_token_path
            Some(max_num_seqs),                               // max_num_seqs
            None,                                             // config_model_len
            Some(max_model_len),                              // max_model_len
            Some(1024),                                       // max_tokens
            None,                                             // isq
            Some(1),                                          // num_shards
            None,                                             // device_ids (will use default)
            None,                                             // generation_cfg
            None,                                             // seed
            None,                                             // prefix_cache
            None,                                             // prefix_cache_max_tokens
            None,                                             // fp8_kvcache
            Some(false),                                      // server_mode (false for embedded use)
            None,                                             // cpu_mem_fold
            kv_fraction,                                      // kv_fraction
            None,                                             // pd_config
            None,                                             // mcp_command
            None,                                             // mcp_config
            None,                                             // mcp_args
            None,                                             // disable_flash_attn
            None,                                             // tool_prompt_template
        );

        // Create engine on blocking runtime
        let new_engine = tokio::task::spawn_blocking(move || {
            LLMEngine::new(&econfig, DType::F32)
        })
        .await
        .context("Failed to spawn blocking task for model loading")?
        .context("Failed to load model")?;

        // Store the engine
        self.engine = Some(new_engine);
        info!("Model loaded successfully for: {}", self.model_name);

        Ok(())
    }

    /// Run inference with completion API (non-streaming)
    ///
    /// # Arguments
    /// * `prompt` - The input prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Temperature for sampling
    ///
    /// # Returns
    /// The generated text and token count
    pub async fn complete(&self, prompt: &str, max_tokens: usize, temperature: Option<f32>) -> Result<(String, usize)> {
        let engine = self.engine.clone();
        let prompt = prompt.to_string();

        let output = tokio::task::spawn_blocking(move || {
            use crate::utils::chat_template::Message;
            use std::vec;

            let messages = vec![Message {
                role: "user".to_string(),
                content: prompt,
                num_images: 0,
            }];

            let sampling_params = SamplingParams {
                temperature,
                max_tokens: Some(max_tokens),
                ..Default::default()
            };

            // Get the engine reference
            let engine_ref = engine.as_ref()
                .expect("Model not loaded, cannot run inference");

            // Use block_in_place to run async code in blocking context
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut e = engine_ref.write();
                    let receivers = e.generate_sync(&vec![sampling_params], &vec![messages], None, &vec![], &None)
                        .map_err(|e| anyhow::anyhow!("Failed to generate: {:?}", e))?;

                    // Clone tokenizer before dropping mutable access
                    let tokenizer = Arc::new(e.tokenizer.clone());
                    drop(e);

                    // Collect results
                    let mut output_text = String::new();
                    let mut token_count = 0usize;
                    for (_seq_id, _prompt_len, mut rx) in receivers {
                        // rx.recv() returns Option<StreamItem> (None when done)
                        while let Some(item) = rx.recv().await {
                            match item {
                                crate::core::engine::StreamItem::Completion((
                                    _prompt_start,
                                    _decode_start,
                                    _decode_finish,
                                    decoded_ids,
                                )) => {
                                    token_count += decoded_ids.len();
                                    // Decode token IDs to text
                                    if let Ok(decoded) = tokenizer.decode(&decoded_ids, true) {
                                        output_text = decoded;
                                    }
                                }
                                crate::core::engine::StreamItem::Done(_) => {
                                    break;
                                }
                                crate::core::engine::StreamItem::Error(e) => {
                                    anyhow::bail!("Inference error: {}", e);
                                }
                                _ => {}
                            }
                        }
                    }

                    Ok::<(String, usize), anyhow::Error>((output_text, token_count))
                })
            })
        })
        .await
        .context("Inference task failed")??;

        Ok(output)
    }

    /// Shutdown the engine and release resources
    pub fn shutdown(&mut self) {
        self.engine = None;
        info!("Model engine shut down for: {}", self.model_name);
    }
}

/// Extract model architecture from config.json
pub fn get_model_architecture(config_path: &PathBuf) -> Result<String> {
    let config_content = std::fs::read_to_string(config_path)
        .context("Failed to read config file")?;

    // Parse JSON to find architecture
    #[derive(Debug, serde::Deserialize)]
    struct ModelConfig {
        architecture: Option<String>,
        architectures: Option<Vec<String>>,
    }

    let config: ModelConfig = serde_json::from_str(&config_content)
        .context("Failed to parse config JSON")?;

    // Try architecture field first, then architectures
    if let Some(arch) = config.architecture {
        Ok(arch)
    } else if let Some(ref archs) = config.architectures {
        Ok(archs.first()
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string()))
    } else {
        anyhow::bail!("Could not determine model architecture from config");
    }
}

/// Detect device type from system
#[cfg(feature = "metal")]
pub fn detect_device() -> &'static str {
    "metal"
}

#[cfg(not(feature = "metal"))]
#[cfg(feature = "cuda")]
pub fn detect_device() -> &'static str {
    "cuda"
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub fn detect_device() -> &'static str {
    "cpu"
}
