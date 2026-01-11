use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::Subscriber;
use zenoh::sample::Sample;
use zenoh::Config;
use zenoh::Session;

use super::weight_loader::WeightLoader;

#[derive(Clone)]
/// Client for Zenoh-based communication with the BARM coordinator
///
/// This client handles:
/// - Registration with the coordinator
/// - Receiving model loading requests
/// - Receiving model assets (config, tokenizer, weights)
/// - Submitting inference results
/// - Heartbeat signaling
pub struct ZenohClient {
    /// The Zenoh session
    session: Arc<Session>,
    /// Weight loader for handling received model assets
    weight_loader: WeightLoader,
    /// Channel for sending shutdown signals
    shutdown_tx: mpsc::Sender<()>,
}

impl ZenohClient {
    /// Create a new ZenohClient connected to the given peer
    ///
    /// # Arguments
    /// * `peer` - The Zenoh peer endpoint to connect to
    /// * `cache_dir` - Directory for caching model assets
    ///
    /// # Returns
    /// A new ZenohClient instance
    pub async fn new(peer: &str, cache_dir: PathBuf) -> Result<Self> {
        let config_str = format!(r#"{{
            mode: "client",
            connect: {{
                endpoints: ["{}"]
            }}
        }}"#, peer);

        let config = Config::from_json5(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to create config: {:?}", e))?;

        let session = zenoh::open(config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to Zenoh: {}", e))?;

        let (shutdown_tx, _) = mpsc::channel(1);

        Ok(Self {
            session: Arc::new(session),
            weight_loader: WeightLoader::new(cache_dir, None),
            shutdown_tx,
        })
    }

    /// Register this worker with the coordinator
    ///
    /// # Returns
    /// The response from the coordinator containing worker ID and config
    pub async fn register(&self) -> Result<WorkerRegistrationResponse> {
        // TODO: Implement registration logic
        Ok(WorkerRegistrationResponse {
            worker_id: "TODO".to_string(),
            model_config: None,
        })
    }

    /// Submit an inference result to the coordinator
    ///
    /// # Arguments
    /// * `request_id` - The original request ID
    /// * `result` - The inference result
    pub async fn submit_result(&self, request_id: &str, result: &str) -> Result<()> {
        let key = format!("barm/result/{}", request_id);
        let publisher = self.session.declare_publisher(&key).await
            .map_err(|e| anyhow::anyhow!("Failed to declare publisher: {:?}", e))?;
        publisher.put(result).await
            .map_err(|e| anyhow::anyhow!("Failed to publish result: {:?}", e))?;
        Ok(())
    }

    /// Send a heartbeat signal periodically
    ///
    /// # Arguments
    /// * `worker_id` - This worker's ID
    pub async fn send_heartbeat(&self, worker_id: &str) -> Result<()> {
        // In Zenoh 1.x, we use liveliness tokens for heartbeats
        let key = format!("barm/worker/alive/{}", worker_id);
        // Refresh the liveliness token periodically
        let session = self.session.clone();
        let token_key = key.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = session.liveliness().declare_token(&token_key).await {
                    tracing::error!("Failed to refresh liveliness token: {:?}", e);
                }
            }
        });
        // Initial declaration
        self.session.liveliness().declare_token(&key).await
            .map_err(|e| anyhow::anyhow!("Failed to declare liveliness token: {:?}", e))?;
        Ok(())
    }

    /// Get a subscriber for receiving inference requests
    ///
    /// # Arguments
    /// * `worker_id` - This worker's ID
    ///
    /// # Returns
    /// A subscriber for request messages
    pub async fn subscribe_requests(&self, worker_id: &str) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        let key_expr = format!("barm/request/{}", worker_id);
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare subscriber: {:?}", e))?;
        Ok(subscriber)
    }

    /// Subscribe to model config updates
    ///
    /// # Arguments
    /// * `model_name` - The model name to subscribe to
    ///
    /// # Returns
    /// A subscriber for config messages
    pub async fn subscribe_config(&self, model_name: &str) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        let key_expr = format!("barm/model/{}/config", model_name);
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare config subscriber: {:?}", e))?;
        tracing::info!("Subscribed to config: {}", key_expr);
        Ok(subscriber)
    }

    /// Subscribe to tokenizer updates
    ///
    /// # Arguments
    /// * `model_name` - The model name to subscribe to
    ///
    /// # Returns
    /// A subscriber for tokenizer messages
    pub async fn subscribe_tokenizer(&self, model_name: &str) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        let key_expr = format!("barm/model/{}/tokenizer", model_name);
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare tokenizer subscriber: {:?}", e))?;
        tracing::info!("Subscribed to tokenizer: {}", key_expr);
        Ok(subscriber)
    }

    /// Subscribe to weight updates (sharded safetensors)
    ///
    /// # Arguments
    /// * `model_name` - The model name to subscribe to
    ///
    /// # Returns
    /// A subscriber for weight messages
    pub async fn subscribe_weights(&self, model_name: &str) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        let key_expr = format!("barm/model/{}/weights/shard-*", model_name);
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare weights subscriber: {:?}", e))?;
        tracing::info!("Subscribed to weights: {}", key_expr);
        Ok(subscriber)
    }

    /// Subscribe to model load requests
    ///
    /// # Returns
    /// A subscriber for model load requests
    pub async fn subscribe_model_load(&self) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        let key_expr = "barm/model/load".to_string();
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare model load subscriber: {:?}", e))?;
        tracing::info!("Subscribed to model load requests");
        Ok(subscriber)
    }

    /// Run the client, listening for all messages
    ///
    /// # Arguments
    /// * `model_name` - The model name to receive assets for
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn run(&self, model_name: &str) -> Result<()> {
        tracing::info!("Starting Zenoh client for model: {}", model_name);

        // Create subscribers for all asset types
        let config_sub = self.subscribe_config(model_name).await?;
        let tokenizer_sub = self.subscribe_tokenizer(model_name).await?;
        let weights_sub = self.subscribe_weights(model_name).await?;
        let model_load_sub = self.subscribe_model_load().await?;

        tracing::info!("All subscribers created. Starting event loop...");

        // Main event loop - run all handlers concurrently with shutdown signal
        let shutdown_rx = self.shutdown_tx.clone();
        // Clone data for spawned tasks since tokio::spawn requires 'static
        let model_name = model_name.to_string();
        let model_name_for_config = model_name.clone();
        let model_name_for_tokenizer = model_name.clone();
        let model_name_for_weights = model_name.clone();
        let self_for_config = self.clone();
        let self_for_tokenizer = self.clone();
        let self_for_weights = self.clone();
        let self_for_model_load = self.clone();

        // Spawn all handlers as independent tasks
        let config_h = tokio::spawn(async move {
            self_for_config.handle_config_subscriber(config_sub, &model_name_for_config).await
        });
        let tokenizer_h = tokio::spawn(async move {
            self_for_tokenizer.handle_tokenizer_subscriber(tokenizer_sub, &model_name_for_tokenizer).await
        });
        let weights_h = tokio::spawn(async move {
            self_for_weights.handle_weights_subscriber(weights_sub, &model_name_for_weights).await
        });
        let model_load_h = tokio::spawn(async move {
            self_for_model_load.handle_model_load_subscriber(model_load_sub).await
        });

        // Wait for shutdown signal or any handler error
        tokio::select! {
            _ = shutdown_rx.closed() => {
                tracing::info!("Shutdown signal received");
            }
            _ = config_h => {
                tracing::info!("Config handler ended");
            }
            _ = tokenizer_h => {
                tracing::info!("Tokenizer handler ended");
            }
            _ = weights_h => {
                tracing::info!("Weights handler ended");
            }
            _ = model_load_h => {
                tracing::info!("Model load handler ended");
            }
        }

        Ok(())
    }

    /// Handle config messages
    async fn handle_config_subscriber(
        &self,
        mut subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling config messages for model: {}", model_name);

        while let Ok(sample) = subscriber.recv_async().await {
            let path = sample.key_expr().to_string();
            tracing::info!("Received config for model {}: {}", model_name, path);

            let bytes = sample.payload().to_bytes();
            match self.weight_loader.load_config(model_name, bytes.as_ref()).await {
                Ok(config_path) => {
                    tracing::info!("Config saved to: {:?}", config_path);
                }
                Err(e) => {
                    tracing::error!("Failed to save config: {:?}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle tokenizer messages
    async fn handle_tokenizer_subscriber(
        &self,
        mut subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling tokenizer messages for model: {}", model_name);

        while let Ok(sample) = subscriber.recv_async().await {
            let path = sample.key_expr().to_string();
            tracing::info!("Received tokenizer for model {}: {}", model_name, path);

            let bytes = sample.payload().to_bytes();
            match self.weight_loader.load_tokenizer(model_name, bytes.as_ref()).await {
                Ok(tokenizer_path) => {
                    tracing::info!("Tokenizer saved to: {:?}", tokenizer_path);
                }
                Err(e) => {
                    tracing::error!("Failed to save tokenizer: {:?}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle weight messages (sharded safetensors)
    async fn handle_weights_subscriber(
        &self,
        mut subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling weight messages for model: {}", model_name);

        while let Ok(sample) = subscriber.recv_async().await {
            let path = sample.key_expr().to_string();
            tracing::info!("Received weight shard for model {}: {}", model_name, path);

            let bytes = sample.payload().to_bytes();
            // Extract shard index from path (e.g., "barm/model/model_name/weights/shard-001")
            let shard_index = extract_shard_index(&path)
                .unwrap_or_else(|| "0".to_string());

            match self.weight_loader.load_weights(model_name, &shard_index, bytes.as_ref()).await {
                Ok(weight_path) => {
                    tracing::info!("Weight shard saved to: {:?}", weight_path);
                }
                Err(e) => {
                    tracing::error!("Failed to save weight shard: {:?}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle model load requests
    async fn handle_model_load_subscriber(
        &self,
        mut subscriber: Subscriber<FifoChannelHandler<Sample>>,
    ) -> Result<()> {
        tracing::info!("Handling model load requests");

        while let Ok(sample) = subscriber.recv_async().await {
            let path = sample.key_expr().to_string();
            tracing::info!("Received model load request: {}", path);

            let bytes = sample.payload().to_bytes();
            match serde_json::from_slice::<ModelLoadRequest>(bytes.as_ref()) {
                Ok(request) => {
                    tracing::info!("Model load request: {:?}", request);
                    // TODO: Trigger model loading with the received config
                }
                Err(e) => {
                    tracing::error!("Failed to parse model load request: {:?}", e);
                }
            }
        }

        Ok(())
    }

    /// Signal shutdown to all handlers
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_tx.send(()).await
            .map_err(|e| anyhow::anyhow!("Failed to send shutdown signal: {:?}", e))
    }
}

/// Extract shard index from path
fn extract_shard_index(path: &str) -> Option<String> {
    // Expected format: barm/model/{model_name}/weights/shard-{number}
    if let Some(start) = path.rfind("shard-") {
        let shard_part = &path[start + 6..];
        // Take up to 3 characters (e.g., "001", "001\n")
        let end = shard_part.find(|c: char| !c.is_ascii_digit()).unwrap_or(shard_part.len());
        // Bounds check before slicing to avoid panic
        if end > 0 && end <= shard_part.len() {
            Some(shard_part[..end].to_string())
        } else {
            None
        }
    } else {
        None
    }
}

/// Response from coordinator during worker registration
#[derive(Debug)]
pub struct WorkerRegistrationResponse {
    /// The assigned worker ID
    pub worker_id: String,
    /// The model configuration to load (if any)
    pub model_config: Option<ModelConfig>,
}

/// Model configuration from the coordinator
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name or path
    pub model_id: String,
    /// Hardware acceleration type
    pub device: DeviceType,
    /// Additional model parameters
    pub params: serde_json::Value,
}

/// Request to load a model
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelLoadRequest {
    /// The model name
    pub model_name: String,
    /// Model architecture type
    pub architecture: String,
    /// Whether to overwrite existing files
    pub overwrite: bool,
}

/// Device type for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU execution
    Cpu,
    /// Metal GPU acceleration (Apple Silicon)
    Metal,
    /// CUDA GPU acceleration
    Cuda,
}
