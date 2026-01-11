use anyhow::{Context, Result};
use std::path::PathBuf;

pub mod zenoh_client;
pub mod weight_loader;

pub use zenoh_client::ZenohClient;
pub use weight_loader::WeightLoader;

/// Run the barm worker with the given Zenoh peer endpoint
///
/// # Arguments
/// * `zenoh_peer` - The Zenoh peer endpoint to connect to
/// * `model_name` - The model name to receive assets for
/// * `cache_dir` - Directory to cache model assets
///
/// # Returns
/// Result indicating success or failure
pub async fn run(zenoh_peer: String, model_name: String, cache_dir: PathBuf) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = ZenohClient::new(&zenoh_peer, cache_dir).await
        .context("Failed to create Zenoh client")?;
    tracing::info!("Barm worker connected to {} for model: {}", zenoh_peer, model_name);
    client.run(&model_name).await
        .context("Failed to run Zenoh client")?;
    Ok(())
}
