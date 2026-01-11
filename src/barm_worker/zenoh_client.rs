use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, watch};
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::Subscriber;
use zenoh::sample::Sample;
use zenoh::Config;
use zenoh::Session;

use super::model_loader::ModelEngine;
use super::weight_loader::WeightLoader;

/// Shared shutdown signal using watch channel (supports cloning)
#[derive(Clone)]
struct ShutdownSignal {
    /// Watch sender for shutdown signal
    tx: watch::Sender<bool>,
    /// Watch receiver for shutdown signal
    rx: watch::Receiver<bool>,
}

impl ShutdownSignal {
    fn new() -> Self {
        let (tx, rx) = watch::channel(false);
        Self { tx, rx }
    }

    fn signal(&self) {
        let _ = self.tx.send(true);
    }
}

/// Track which model assets have been received
#[derive(Clone, Default)]
struct AssetTracker {
    pub config_received: Arc<AtomicBool>,
    pub tokenizer_received: Arc<AtomicBool>,
    pub weights_received: Arc<AtomicBool>,
}

impl AssetTracker {
    fn new() -> Self {
        Self {
            config_received: Arc::new(AtomicBool::new(false)),
            tokenizer_received: Arc::new(AtomicBool::new(false)),
            weights_received: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if all assets are received
    fn all_received(&self) -> bool {
        self.config_received.load(Ordering::SeqCst)
            && self.tokenizer_received.load(Ordering::SeqCst)
            && self.weights_received.load(Ordering::SeqCst)
    }

    /// Mark a specific asset as received
    fn mark_config_received(&self) {
        self.config_received.store(true, Ordering::SeqCst);
    }

    fn mark_tokenizer_received(&self) {
        self.tokenizer_received.store(true, Ordering::SeqCst);
    }

    fn mark_weights_received(&self) {
        self.weights_received.store(true, Ordering::SeqCst);
    }
}

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
    /// Model engine for inference (wrapped in tokio::sync::Mutex for Send compatibility)
    model_engine: Arc<Mutex<ModelEngine>>,
    /// Shutdown signal
    shutdown: ShutdownSignal,
    /// Asset tracker
    asset_tracker: Arc<AssetTracker>,
    /// Model name
    model_name: String,
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

        let shutdown = ShutdownSignal::new();

        Ok(Self {
            session: Arc::new(session),
            weight_loader: WeightLoader::new(cache_dir, None),
            model_engine: Arc::new(Mutex::new(ModelEngine::new(String::new()))),
            shutdown,
            asset_tracker: Arc::new(AssetTracker::new()),
            model_name: String::new(),
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
        // Use ** for recursive matching (Zenoh requires * to be preceded by / or $)
        let key_expr = format!("barm/model/{}/weights/**", model_name);
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

    /// Subscribe to inference requests
    ///
    /// # Returns
    /// A subscriber for inference request messages
    pub async fn subscribe_inference_requests(&self) -> Result<Subscriber<FifoChannelHandler<Sample>>> {
        // Subscribe to all inference requests for this model
        let key_expr = format!("barm/inference/{}", self.model_name);
        let subscriber = self.session.declare_subscriber(&key_expr).await
            .map_err(|e| anyhow::anyhow!("Failed to declare inference subscriber: {:?}", e))?;
        tracing::info!("Subscribed to inference requests: {}", key_expr);
        Ok(subscriber)
    }

    /// Run the client, listening for all messages
    ///
    /// # Arguments
    /// * `model_name` - The model name to receive assets for
    ///
    /// # Returns
    /// Result indicating success or failure
    pub async fn run(&mut self, model_name: &str) -> Result<()> {
        tracing::info!("Starting Zenoh client for model: {}", model_name);

        // Store model name
        self.model_name = model_name.to_string();

        // Initialize model engine
        {
            let mut guard = self.model_engine.lock().await;
            *guard = ModelEngine::new(model_name.to_string());
        }

        // Create subscribers for all asset types
        let config_sub = self.subscribe_config(model_name).await?;
        let tokenizer_sub = self.subscribe_tokenizer(model_name).await?;
        let weights_sub = self.subscribe_weights(model_name).await?;
        let model_load_sub = self.subscribe_model_load().await?;
        let inference_sub = self.subscribe_inference_requests().await?;

        tracing::info!("All subscribers created. Starting event loop...");

        // Main event loop - run all handlers concurrently with shutdown signal
        // Clone data for spawned tasks since tokio::spawn requires 'static
        let model_name = model_name.to_string();
        let model_name_for_config = model_name.clone();
        let model_name_for_tokenizer = model_name.clone();
        let model_name_for_weights = model_name.clone();
        let self_for_config = self.clone();
        let self_for_tokenizer = self.clone();
        let self_for_weights = self.clone();
        let self_for_model_load = self.clone();
        let self_for_inference = self.clone();
        let mut shutdown_rx = self.shutdown.rx.clone();

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
        let inference_h = tokio::spawn(async move {
            self_for_inference.handle_inference_requests(inference_sub).await
        });

        // Wait for shutdown signal or any handler error
        tokio::select! {
            _ = shutdown_rx.changed() => {
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
            _ = inference_h => {
                tracing::info!("Inference handler ended");
            }
        }

        Ok(())
    }

    /// Check if all assets are received and load model
    async fn check_and_load_model(&self) {
        if self.asset_tracker.all_received() {
            tracing::info!("All model assets received, loading model...");

            let model_dir = self.weight_loader.cache_dir().join(&self.model_name);

            // Lock the mutex and load the model (tokio::sync::Mutex guard is Send)
            let mut engine = self.model_engine.lock().await;
            if let Err(e) = engine.load_model(model_dir).await {
                tracing::error!("Failed to load model: {:?}", e);
                return;
            }

            tracing::info!("Model loaded successfully, running test inference...");

            // Run test inference
            drop(engine); // Release the lock before inference
            self.run_test_inference().await;
        }
    }

    /// Run a test inference
    async fn run_test_inference(&self) {
        let engine = self.model_engine.lock().await;
        if !engine.is_loaded() {
            tracing::warn!("Model not loaded, skipping test inference");
            return;
        }

        tracing::info!("Running test inference: 'what is the capital of China'");

        // Clone for use in async block
        let engine_clone = engine.clone();

        drop(engine);

        match engine_clone.complete("what is the capital of China", 100, Some(0.7)).await {
            Ok(result) => {
                tracing::info!("Test inference result: {}", result);

                // Submit result to coordinator (for demo purposes, use a test request ID)
                let request_id = "test-001".to_string();
                if let Err(e) = self.submit_result(&request_id, &result).await {
                    tracing::error!("Failed to submit test result: {:?}", e);
                }
            }
            Err(e) => {
                tracing::error!("Test inference failed: {:?}", e);
            }
        }
    }

    /// Handle config messages
    async fn handle_config_subscriber(
        &self,
        subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling config messages for model: {}", model_name);

        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let path = sample.key_expr().to_string();
                    tracing::info!("Received config for model {}: {}", model_name, path);

                    let bytes = sample.payload().to_bytes();
                    if let Err(e) = self.weight_loader.load_config(model_name, bytes.as_ref()).await {
                        tracing::error!("Failed to save config: {:?}", e);
                    } else {
                        // Mark config as received
                        self.asset_tracker.mark_config_received();

                        // Check if all assets are received
                        self.check_and_load_model().await;
                    }
                }
                Err(e) => {
                    tracing::error!("Config subscriber error: {:?}", e);
                    anyhow::bail!("Config subscriber error: {:?}", e);
                }
            }
        }
    }

    /// Handle tokenizer messages
    async fn handle_tokenizer_subscriber(
        &self,
        subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling tokenizer messages for model: {}", model_name);

        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let path = sample.key_expr().to_string();
                    tracing::info!("Received tokenizer for model {}: {}", model_name, path);

                    let bytes = sample.payload().to_bytes();
                    if let Err(e) = self.weight_loader.load_tokenizer(model_name, bytes.as_ref()).await {
                        tracing::error!("Failed to save tokenizer: {:?}", e);
                    } else {
                        // Mark tokenizer as received
                        self.asset_tracker.mark_tokenizer_received();

                        // Check if all assets are received
                        self.check_and_load_model().await;
                    }
                }
                Err(e) => {
                    tracing::error!("Tokenizer subscriber error: {:?}", e);
                    anyhow::bail!("Tokenizer subscriber error: {:?}", e);
                }
            }
        }
    }

    /// Handle weight messages (sharded safetensors)
    async fn handle_weights_subscriber(
        &self,
        subscriber: Subscriber<FifoChannelHandler<Sample>>,
        model_name: &str,
    ) -> Result<()> {
        tracing::info!("Handling weight messages for model: {}", model_name);

        let mut weight_count = 0;

        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let path = sample.key_expr().to_string();
                    tracing::info!("Received weight shard for model {}: {}", model_name, path);

                    let bytes = sample.payload().to_bytes();
                    // Extract shard index from path (e.g., "barm/model/model_name/weights/shard-001")
                    let shard_index = extract_shard_index(&path)
                        .unwrap_or_else(|| "0".to_string());

                    if let Err(e) = self.weight_loader.load_weights(model_name, &shard_index, bytes.as_ref()).await {
                        tracing::error!("Failed to save weight shard: {:?}", e);
                    } else {
                        weight_count += 1;
                        tracing::info!("Weight shard {} saved", weight_count);

                        // Mark weights as received (after first weight shard)
                        if weight_count == 1 {
                            self.asset_tracker.mark_weights_received();
                        }

                        // Check if all assets are received
                        self.check_and_load_model().await;
                    }
                }
                Err(e) => {
                    tracing::error!("Weights subscriber error: {:?}", e);
                    anyhow::bail!("Weights subscriber error: {:?}", e);
                }
            }
        }
    }

    /// Handle model load requests
    async fn handle_model_load_subscriber(
        &self,
        subscriber: Subscriber<FifoChannelHandler<Sample>>,
    ) -> Result<()> {
        tracing::info!("Handling model load requests");

        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let path = sample.key_expr().to_string();
                    tracing::info!("Received model load request: {}", path);

                    let bytes = sample.payload().to_bytes();
                    if let Ok(request) = serde_json::from_slice::<ModelLoadRequest>(bytes.as_ref()) {
                        tracing::info!("Model load request: {:?}", request);
                        // Trigger model loading with the received config
                        self.check_and_load_model().await;
                    } else {
                        tracing::error!("Failed to parse model load request");
                    }
                }
                Err(e) => {
                    tracing::error!("Model load subscriber error: {:?}", e);
                    anyhow::bail!("Model load subscriber error: {:?}", e);
                }
            }
        }
    }

    /// Handle inference requests
    async fn handle_inference_requests(
        &self,
        subscriber: Subscriber<FifoChannelHandler<Sample>>,
    ) -> Result<()> {
        tracing::info!("Handling inference requests");

        loop {
            match subscriber.recv_async().await {
                Ok(sample) => {
                    let path = sample.key_expr().to_string();
                    tracing::info!("Received inference request: {}", path);

                    let bytes = sample.payload().to_bytes();

                    // Parse inference request
                    if let Ok(request) = serde_json::from_slice::<InferenceRequest>(bytes.as_ref()) {
                        tracing::info!("Processing inference request: {:?}", request);

                        // Check if model is loaded BEFORE entering async context
                        let is_loaded = {
                            let engine = self.model_engine.lock().await;
                            engine.is_loaded()
                        };

                        if !is_loaded {
                            tracing::warn!("Model not loaded, cannot process inference request");

                            let error_result = serde_json::json!({
                                "error": "Model not loaded yet",
                                "request_id": request.request_id
                            });
                            let _ = self.submit_result(&request.request_id, &error_result.to_string()).await;
                            continue;
                        }

                        // Clone the engine for async operation
                        let engine_clone = {
                            let engine = self.model_engine.lock().await;
                            engine.clone()
                        };

                        // Run inference
                        match engine_clone.complete(
                            &request.prompt,
                            request.max_tokens.unwrap_or(1024),
                            request.temperature,
                        ).await {
                            Ok(result) => {
                                tracing::info!("Inference result: {}", result);

                                // Submit result back to coordinator
                                if let Err(e) = self.submit_result(&request.request_id, &result).await {
                                    tracing::error!("Failed to submit inference result: {:?}", e);
                                }
                            }
                            Err(e) => {
                                tracing::error!("Inference failed: {:?}", e);

                                // Submit error result
                                let error_result = serde_json::json!({
                                    "error": format!("{:?}", e),
                                    "request_id": request.request_id
                                });
                                if let Err(submit_err) = self.submit_result(
                                    &request.request_id,
                                    &error_result.to_string()
                                ).await {
                                    tracing::error!("Failed to submit error result: {:?}", submit_err);
                                }
                            }
                        }
                    } else {
                        tracing::error!("Failed to parse inference request");
                    }
                }
                Err(e) => {
                    tracing::error!("Inference subscriber error: {:?}", e);
                    anyhow::bail!("Inference subscriber error: {:?}", e);
                }
            }
        }
    }

    /// Signal shutdown to all handlers
    pub fn shutdown(&self) {
        self.shutdown.signal();
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

/// Inference request from coordinator
#[derive(Debug, Clone, serde::Deserialize)]
pub struct InferenceRequest {
    /// Unique request ID
    pub request_id: String,
    /// The prompt to process
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<isize>,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
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
