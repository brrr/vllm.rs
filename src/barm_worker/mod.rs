//! BARM Worker module for distributed model serving via Zenoh
//!
//! This module provides the worker-side functionality for the BARM distributed
//! inference system, including Zenoh communication and weight loading.

pub mod zenoh_client;
pub mod weight_loader;

pub use zenoh_client::ZenohClient;
pub use weight_loader::WeightLoader;

/// Run the barm worker with the given Zenoh peer endpoint
///
/// # Arguments
/// * `zenoh_peer` - The Zenoh peer endpoint to connect to
///
/// # Returns
/// Result indicating success or failure
pub async fn run(zenoh_peer: String) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let client = ZenohClient::new(&zenoh_peer).await?;
    tracing::info!("Barm worker connected to {}", zenoh_peer);
    client.run().await?;
    Ok(())
}
