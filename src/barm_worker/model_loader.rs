use anyhow::{Context, Result};
use candle_core::DType;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::core::engine::LLMEngine;
use crate::utils::config::{EngineConfig, SamplingParams};

/// ModelEngine - Wraps vLLM.rs LLMEngine for barm-worker inference
///
/// This struct manages the lifecycle of the vLLM.rs inference engine,
/// including model loading and inference execution.
#[derive(Clone)]
pub struct ModelEngine {
    /// The vLLM.rs engine instance (Arc<RwLock<LLMEngine>>)
    engine: Arc<RwLock<LLMEngine>>,
    /// Model name
    model_name: String,
    /// Whether the model is loaded
    is_loaded: bool,
}

impl ModelEngine {
    /// Create a new ModelEngine
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    ///
    /// # Returns
    /// A new ModelEngine instance
    pub fn new(model_name: String) -> Self {
        // Create a placeholder engine that will be replaced after loading
        let econfig = EngineConfig::new(
            None, None, None, None, None,
            Some(16), None, Some(32768), Some(1024),
            None, Some(1), None, None, None, None, None, Some(false),
            None, None, None, None, None, None, None, None, None
        );

        // This will fail, but we replace it immediately after loading
        let engine = LLMEngine::new(&econfig, DType::F32)
            .unwrap_or_else(|_| panic!("Failed to create placeholder engine"));

        Self {
            engine,
            model_name,
            is_loaded: false,
        }
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
        let weights_path = if weights_dir.exists() {
            weights_dir.to_string_lossy().to_string()
        } else {
            // If no weights dir, look for weights in model dir
            model_dir.to_string_lossy().to_string()
        };

        // Create engine config - point to local files
        let econfig = EngineConfig::new(
            Some(model_dir.to_string_lossy().to_string()), // model_id points to local dir
            Some(weights_path),                            // weight_path
            None,                                          // weight_file
            None,                                          // hf_token
            None,                                          // hf_token_path
            Some(16),                                      // max_num_seqs
            None,                                          // config_model_len
            Some(32768),                                   // max_model_len
            Some(1024),                                    // max_tokens
            None,                                          // isq
            Some(1),                                       // num_shards
            None,                                          // device_ids (will use default)
            None,                                          // generation_cfg
            None,                                          // seed
            None,                                          // prefix_cache
            None,                                          // prefix_cache_max_tokens
            None,                                          // fp8_kvcache
            Some(false),                                   // server_mode (false for embedded use)
            None,                                          // cpu_mem_fold
            None,                                          // kv_fraction
            None,                                          // pd_config
            None,                                          // mcp_command
            None,                                          // mcp_config
            None,                                          // mcp_args
            None,                                          // disable_flash_attn
            None,                                          // tool_prompt_template
        );

        // Create engine on blocking runtime
        let new_engine = tokio::task::spawn_blocking(move || {
            LLMEngine::new(&econfig, DType::F32)
        })
        .await
        .context("Failed to spawn blocking task for model loading")?
        .context("Failed to load model")?;

        // Store the engine by replacing the Arc
        self.engine = new_engine;
        self.is_loaded = true;
        info!("Model loaded successfully for: {}", self.model_name);

        Ok(())
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Run inference with completion API (non-streaming)
    ///
    /// # Arguments
    /// * `prompt` - The input prompt
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Temperature for sampling
    ///
    /// # Returns
    /// The generated text
    pub async fn complete(&self, prompt: &str, max_tokens: usize, temperature: Option<f32>) -> Result<String> {
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

            // Use block_in_place to run async code in blocking context
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut e = engine.write();
                    let receivers = e.generate_sync(&vec![sampling_params], &vec![messages], None, &vec![], &None)
                        .map_err(|e| anyhow::anyhow!("Failed to generate: {:?}", e))?;

                    // Clone tokenizer before dropping mutable access
                    let tokenizer = Arc::new(e.tokenizer.clone());
                    drop(e);

                    // Collect results
                    let mut output_text = String::new();
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

                    Ok::<String, anyhow::Error>(output_text)
                })
            })
        })
        .await
        .context("Inference task failed")??;

        Ok(output)
    }

    /// Shutdown the engine and release resources
    pub async fn shutdown(&mut self) {
        // Create a new empty engine to replace the current one
        let econfig = EngineConfig::new(
            None, None, None, None, None,
            Some(16), None, Some(32768), Some(1024),
            None, Some(1), None, None, None, None, None, Some(false),
            None, None, None, None, None, None, None, None, None
        );
        self.engine = LLMEngine::new(&econfig, DType::F32)
            .unwrap_or_else(|_| panic!("Failed to create shutdown engine"));
        self.is_loaded = false;
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
