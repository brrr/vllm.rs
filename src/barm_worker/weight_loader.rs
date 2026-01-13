use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use tokio::fs as tokio_fs;

/// Loader for model weights from various sources
///
/// This struct handles:
/// - Downloading weights from remote storage
/// - Loading weights from local files
/// - Caching downloaded weights
#[derive(Clone)]
pub struct WeightLoader {
    /// Local cache directory for weights
    cache_dir: PathBuf,
    /// Remote storage base URL (optional)
    remote_base_url: Option<String>,
}

impl WeightLoader {
    /// Create a new WeightLoader
    ///
    /// # Arguments
    /// * `cache_dir` - Directory to cache downloaded weights
    /// * `remote_base_url` - Optional base URL for remote weight storage
    pub fn new(cache_dir: PathBuf, remote_base_url: Option<String>) -> Self {
        // Ensure cache directory exists
        let cache_dir = if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)
                .unwrap_or_else(|e| tracing::warn!("Failed to create cache dir: {:?}", e));
            cache_dir
        } else {
            cache_dir
        };

        Self {
            cache_dir,
            remote_base_url,
        }
    }

    /// Get the model directory for a specific model
    fn model_dir(&self, model_name: &str) -> PathBuf {
        self.cache_dir.join(model_name)
    }

    /// Get the weights directory for a specific model
    fn weights_dir(&self, model_name: &str) -> PathBuf {
        self.model_dir(model_name).join("weights")
    }

    /// Load model configuration from bytes received via Zenoh
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `config_data` - Raw config bytes (JSON)
    ///
    /// # Returns
    /// Path to the saved config file
    pub async fn load_config(&self, model_name: &str, config_data: &[u8]) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);

        // Create model directory if it doesn't exist
        fs::create_dir_all(&model_dir)
            .with_context(|| format!("Failed to create model directory: {:?}", model_dir))?;

        let config_path = model_dir.join("config.json");

        // Write config to file
        fs::write(&config_path, config_data)
            .with_context(|| format!("Failed to write config to: {:?}", config_path))?;

        tracing::info!("Saved config for model '{}' to: {:?}", model_name, config_path);

        Ok(config_path)
    }

    /// Load tokenizer from bytes received via Zenoh
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `tokenizer_data` - Raw tokenizer bytes (JSON)
    ///
    /// # Returns
    /// Path to the saved tokenizer file
    pub async fn load_tokenizer(&self, model_name: &str, tokenizer_data: &[u8]) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);

        // Create model directory if it doesn't exist
        fs::create_dir_all(&model_dir)
            .with_context(|| format!("Failed to create model directory: {:?}", model_dir))?;

        let tokenizer_path = model_dir.join("tokenizer.json");

        // Write tokenizer to file
        fs::write(&tokenizer_path, tokenizer_data)
            .with_context(|| format!("Failed to write tokenizer to: {:?}", tokenizer_path))?;

        tracing::info!("Saved tokenizer for model '{}' to: {:?}", model_name, tokenizer_path);

        Ok(tokenizer_path)
    }

    /// Load tokenizer_config.json from bytes received via Zenoh
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `tokenizer_config_data` - Raw tokenizer_config bytes (JSON)
    ///
    /// # Returns
    /// Path to the saved tokenizer_config file
    pub async fn load_tokenizer_config(&self, model_name: &str, tokenizer_config_data: &[u8]) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);

        let tokenizer_config_path = model_dir.join("tokenizer_config.json");

        // Write tokenizer_config to file
        fs::write(&tokenizer_config_path, tokenizer_config_data)
            .with_context(|| format!("Failed to write tokenizer_config to: {:?}", tokenizer_config_path))?;

        tracing::info!("Saved tokenizer_config for model '{}' to: {:?}", model_name, tokenizer_config_path);

        Ok(tokenizer_config_path)
    }

    /// Load generation_config.json from bytes received via Zenoh
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `generation_config_data` - Raw generation_config bytes (JSON)
    ///
    /// # Returns
    /// Path to the saved generation_config file
    pub async fn load_generation_config(&self, model_name: &str, generation_config_data: &[u8]) -> Result<PathBuf> {
        let model_dir = self.model_dir(model_name);

        let generation_config_path = model_dir.join("generation_config.json");

        // Write generation_config to file
        fs::write(&generation_config_path, generation_config_data)
            .with_context(|| format!("Failed to write generation_config to: {:?}", generation_config_path))?;

        tracing::info!("Saved generation_config for model '{}' to: {:?}", model_name, generation_config_path);

        Ok(generation_config_path)
    }

    /// Load weight shard from bytes received via Zenoh
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `shard_index` - Index of the weight shard (e.g., "001")
    /// * `weight_data` - Raw weight bytes (safetensors format)
    ///
    /// # Returns
    /// Path to the saved weight shard file
    pub async fn load_weights(
        &self,
        model_name: &str,
        shard_index: &str,
        weight_data: &[u8],
    ) -> Result<PathBuf> {
        let weights_dir = self.weights_dir(model_name);

        // Create weights directory if it doesn't exist
        fs::create_dir_all(&weights_dir)
            .with_context(|| format!("Failed to create weights directory: {:?}", weights_dir))?;

        // Format shard filename with zero-padded index
        let shard_filename = format!("shard-{:03}.safetensors", shard_index.parse::<u32>().unwrap_or(0));
        let weight_path = weights_dir.join(&shard_filename);

        // Write weight shard to file
        fs::write(&weight_path, weight_data)
            .with_context(|| format!("Failed to write weight shard to: {:?}", weight_path))?;

        tracing::info!(
            "Saved weight shard '{}' for model '{}' to: {:?}",
            shard_filename,
            model_name,
            weight_path
        );

        Ok(weight_path)
    }

    /// Start receiving a chunked weight shard
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `shard_index` - Index of the weight shard (e.g., "000")
    ///
    /// # Returns
    /// Path to the pending weight shard file
    pub async fn start_chunked_weight_reception(
        &self,
        model_name: &str,
        shard_index: &str,
        file_size: u64,
        total_chunks: u32,
    ) -> Result<PathBuf> {
        let weights_dir = self.weights_dir(model_name);

        // Create weights directory if it doesn't exist
        fs::create_dir_all(&weights_dir)
            .with_context(|| format!("Failed to create weights directory: {:?}", weights_dir))?;

        // Format shard filename
        let shard_filename = format!("shard-{:03}.safetensors", shard_index.parse::<u32>().unwrap_or(0));
        let weight_path = weights_dir.join(&shard_filename);

        // Create the file
        let file = fs::File::create(&weight_path)
            .with_context(|| format!("Failed to create weight file: {:?}", weight_path))?;

        // Pre-allocate file space (Unix-specific)
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let _ = file.write_all_at(&[0u8; 1], file_size - 1); // Ignore errors, this is just a hint
        }

        tracing::info!(
            "Starting chunked weight reception: {} ({} bytes, {} chunks)",
            weight_path.display(), file_size, total_chunks
        );

        Ok(weight_path)
    }

    /// Write a chunk to an ongoing weight reception
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `shard_index` - Index of the weight shard
    /// * `chunk_index` - Index of the chunk
    /// * `chunk_data` - Raw chunk bytes
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn write_weight_chunk(
        &self,
        model_name: &str,
        shard_index: &str,
        chunk_index: u32,
        chunk_data: &[u8],
    ) -> Result<()> {
        let weights_dir = self.weights_dir(model_name);
        let shard_filename = format!("shard-{:03}.safetensors", shard_index.parse::<u32>().unwrap_or(0));
        let weight_path = weights_dir.join(&shard_filename);

        let offset = chunk_index as u64 * 1024 * 1024; // 1MB chunk size

        // Use blocking I/O in a tokio blocking task for file writes at specific offsets
        let chunk_data = chunk_data.to_vec();
        let weight_path_clone = weight_path.clone();

        tokio::task::spawn_blocking(move || {
            use std::io::{Write, Seek};
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .open(&weight_path_clone)
                .with_context(|| format!("Failed to open weight file: {:?}", weight_path_clone))?;

            file.seek(std::io::SeekFrom::Start(offset))
                .with_context(|| format!("Failed to seek to offset {}", offset))?;

            file.write_all(&chunk_data)
                .with_context(|| format!("Failed to write chunk at offset {}", offset))?;

            Ok::<(), anyhow::Error>(())
        }).await
        .with_context(|| "Failed to execute blocking I/O task")??;

        tracing::debug!("Wrote chunk {} for shard {}", chunk_index, shard_filename);

        Ok(())
    }

    /// Finalize chunked weight reception
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `shard_index` - Index of the weight shard
    ///
    /// # Returns
    /// Path to the completed weight shard file
    pub async fn finalize_chunked_weight_reception(
        &self,
        model_name: &str,
        shard_index: &str,
    ) -> Result<PathBuf> {
        let weights_dir = self.weights_dir(model_name);
        let shard_filename = format!("shard-{:03}.safetensors", shard_index.parse::<u32>().unwrap_or(0));
        let weight_path = weights_dir.join(&shard_filename);

        tracing::info!(
            "Finalized chunked weight reception: {:?}",
            weight_path
        );

        Ok(weight_path)
    }

    /// Load all weights for a model and return the weights directory
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    ///
    /// # Returns
    /// Path to the weights directory
    pub fn get_weights_dir(&self, model_name: &str) -> Result<PathBuf> {
        let weights_dir = self.weights_dir(model_name);

        if !weights_dir.exists() {
            anyhow::bail!("Weights directory does not exist: {:?}", weights_dir);
        }

        Ok(weights_dir)
    }

    /// Check if all model assets are cached
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    /// * `expected_shards` - Expected number of weight shards
    ///
    /// # Returns
    /// True if all assets are cached
    pub fn is_model_cached(&self, model_name: &str, expected_shards: u32) -> bool {
        let model_dir = self.model_dir(model_name);

        // Check config exists
        if !model_dir.join("config.json").exists() {
            return false;
        }

        // Check tokenizer exists
        if !model_dir.join("tokenizer.json").exists() {
            return false;
        }

        // Check all weight shards exist
        for i in 0..expected_shards {
            let shard_path = self.weights_dir(model_name).join(format!("shard-{:03}.safetensors", i));
            if !shard_path.exists() {
                return false;
            }
        }

        true
    }

    /// Check if weights are already cached
    ///
    /// # Arguments
    /// * `_model_id` - The model identifier
    ///
    /// # Returns
    /// True if weights are cached
    pub fn is_cached(&self, model_name: &str) -> bool {
        let model_dir = self.model_dir(model_name);
        model_dir.exists() && model_dir.join("config.json").exists()
    }

    /// Clear cached weights for a specific model
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    pub fn clear_cache(&self, model_name: &str) -> Result<()> {
        let model_dir = self.model_dir(model_name);

        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)
                .with_context(|| format!("Failed to clear cache for model: {}", model_name))?;
            tracing::info!("Cleared cache for model: {}", model_name);
        }

        Ok(())
    }

    /// Get the total size of cached weights for a model
    ///
    /// # Arguments
    /// * `model_name` - The model identifier
    ///
    /// # Returns
    /// Size in bytes
    pub fn cache_size(&self, model_name: &str) -> Result<u64> {
        let model_dir = self.model_dir(model_name);

        if !model_dir.exists() {
            return Ok(0);
        }

        let mut total_size = 0u64;

        for entry in fs::read_dir(&model_dir)
            .with_context(|| format!("Failed to read cache directory for: {}", model_name))?
        {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                total_size += path.metadata()?.len();
            } else if path.is_dir() {
                // Recursively calculate size of subdirectories (weights dir)
                for sub_entry in fs::read_dir(&path)? {
                    let sub_entry = sub_entry?;
                    if sub_entry.path().is_file() {
                        total_size += sub_entry.path().metadata()?.len();
                    }
                }
            }
        }

        Ok(total_size)
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_load_config() {
        let temp_dir = TempDir::new().unwrap();
        let loader = WeightLoader::new(temp_dir.path().to_path_buf(), None);

        let config_data = r#"{"model_type": "llama", "hidden_size": 4096}"#;

        let result = loader.load_config("test-model", config_data.as_bytes()).await;

        assert!(result.is_ok());
        let config_path = result.unwrap();
        assert!(config_path.exists());

        // Verify content
        let saved_content = fs::read_to_string(&config_path).unwrap();
        assert_eq!(saved_content, config_data);
    }

    #[tokio::test]
    async fn test_load_tokenizer() {
        let temp_dir = TempDir::new().unwrap();
        let loader = WeightLoader::new(temp_dir.path().to_path_buf(), None);

        let tokenizer_data = r#"{"version": "1.0", "tokens": []}"#;

        let result = loader.load_tokenizer("test-model", tokenizer_data.as_bytes()).await;

        assert!(result.is_ok());
        let tokenizer_path = result.unwrap();
        assert!(tokenizer_path.exists());
    }

    #[tokio::test]
    async fn test_load_weights() {
        let temp_dir = TempDir::new().unwrap();
        let loader = WeightLoader::new(temp_dir.path().to_path_buf(), None);

        let weight_data = b"safetensors header and weights data";

        let result = loader.load_weights("test-model", "001", weight_data).await;

        assert!(result.is_ok());
        let weight_path = result.unwrap();
        assert!(weight_path.exists());
        assert_eq!(weight_path.file_name().unwrap().to_str().unwrap(), "shard-001.safetensors");
    }

    #[test]
    fn test_is_model_cached() {
        let temp_dir = TempDir::new().unwrap();
        let loader = WeightLoader::new(temp_dir.path().to_path_buf(), None);

        // Initially not cached
        assert!(!loader.is_cached("test-model"));

        // Create config file
        let model_dir = temp_dir.path().join("test-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();

        // Now should be cached
        assert!(loader.is_cached("test-model"));
    }

    #[test]
    fn test_cache_size() {
        let temp_dir = TempDir::new().unwrap();
        let loader = WeightLoader::new(temp_dir.path().to_path_buf(), None);

        let model_dir = temp_dir.path().join("test-model");
        fs::create_dir_all(&model_dir).unwrap();

        // Write some test files
        fs::write(model_dir.join("config.json"), "{}").unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();

        let size = loader.cache_size("test-model").unwrap();
        assert!(size > 0);
    }
}
