use anyhow::{Context, Result};
use std::path::PathBuf;

pub mod zenoh_client;
pub mod weight_loader;
pub mod model_loader;

pub use zenoh_client::ZenohClient;
pub use weight_loader::WeightLoader;
pub use model_loader::{ModelEngine, MemoryConfig};

/// Run the barm worker with the given Zenoh peer endpoint
///
/// # Arguments
/// * `zenoh_peer` - The Zenoh peer endpoint to connect to
/// * `model_name` - The model name to receive assets for
/// * `cache_dir` - Directory to cache model assets
/// * `max_model_len` - Maximum model context length
/// * `max_num_seqs` - Maximum concurrent sequences
/// * `kv_fraction` - KV cache memory fraction
///
/// # Returns
/// Result indicating success or failure
pub async fn run(
    zenoh_peer: String,
    model_name: String,
    cache_dir: PathBuf,
    max_model_len: Option<usize>,
    max_num_seqs: Option<usize>,
    kv_fraction: Option<f32>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Log memory configuration
    tracing::info!("Barm worker memory config: max_model_len={:?}, max_num_seqs={:?}, kv_fraction={:?}",
        max_model_len, max_num_seqs, kv_fraction);

    // Create memory config from command-line args
    let memory_config = MemoryConfig {
        max_model_len,
        max_num_seqs,
        kv_fraction,
    };

    let mut client = ZenohClient::new(&zenoh_peer, cache_dir, memory_config).await
        .context("Failed to create Zenoh client")?;
    tracing::info!("Barm worker connected to {} for model: {}", zenoh_peer, model_name);
    client.run(&model_name).await
        .context("Failed to run Zenoh client")?;
    Ok(())
}
