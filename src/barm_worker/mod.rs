use anyhow::{Context, Result};
use std::path::PathBuf;

pub mod zenoh_client;
pub mod weight_loader;
pub mod model_loader;

pub use zenoh_client::ZenohClient;
pub use weight_loader::WeightLoader;
pub use model_loader::{ModelEngine, MemoryConfig};

/// Trace context for distributed tracing
#[derive(Debug, Clone, Default)]
pub struct TraceContext {
    /// W3C traceparent header value
    pub traceparent: Option<String>,
    /// Trace ID extracted from traceparent
    pub trace_id: Option<String>,
    /// Span ID extracted from traceparent
    pub span_id: Option<String>,
}

impl TraceContext {
    /// Create a new TraceContext from a W3C traceparent string
    /// Format: version-traceId-spanId-traceFlags (e.g., 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01)
    pub fn from_traceparent(traceparent: &str) -> Self {
        let parts: Vec<&str> = traceparent.split('-').collect();
        let (trace_id, span_id) = if parts.len() >= 3 {
            (Some(parts[1].to_string()), Some(parts[2].to_string()))
        } else {
            (None, None)
        };

        Self {
            traceparent: Some(traceparent.to_string()),
            trace_id,
            span_id,
        }
    }

    /// Create a new trace context with generated IDs
    pub fn new() -> Self {
        // Generate a simple trace ID and span ID for logging
        let trace_id = format!("{:032x}", rand::random::<u128>());
        let span_id = format!("{:016x}", rand::random::<u64>());
        Self {
            traceparent: Some(format!("00-{}-{}{:016x}-01", &trace_id[..32], &span_id, 0)),
            trace_id: Some(trace_id),
            span_id: Some(span_id),
        }
    }
}

/// Configuration for barm worker runtime
#[derive(Debug, Clone)]
pub struct BarmWorkerConfig {
    /// Zenoh peer endpoint
    pub zenoh_peer: String,
    /// Model name
    pub model_name: String,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Trace context for distributed tracing
    pub trace_context: TraceContext,
    /// Worker ID for identification
    pub worker_id: String,
}

impl BarmWorkerConfig {
    /// Create a new config with required parameters
    pub fn new(
        zenoh_peer: String,
        model_name: String,
        cache_dir: PathBuf,
    ) -> Self {
        Self {
            zenoh_peer,
            model_name,
            cache_dir,
            memory_config: MemoryConfig::default(),
            trace_context: TraceContext::default(),
            worker_id: format!("vllm-rs-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("unknown")),
        }
    }

    /// Set memory configuration
    pub fn with_memory_config(mut self, max_model_len: Option<usize>, max_num_seqs: Option<usize>, kv_fraction: Option<f32>) -> Self {
        self.memory_config = MemoryConfig {
            max_model_len,
            max_num_seqs,
            kv_fraction,
        };
        self
    }

    /// Set trace context
    pub fn with_trace_context(mut self, traceparent: Option<String>) -> Self {
        if let Some(tp) = traceparent {
            self.trace_context = TraceContext::from_traceparent(&tp);
        }
        self
    }

    /// Set worker ID
    pub fn with_worker_id(mut self, worker_id: String) -> Self {
        self.worker_id = worker_id;
        self
    }
}

/// Run the barm worker with the given configuration
///
/// # Arguments
/// * `config` - Configuration for the barm worker
///
/// # Returns
/// Result indicating success or failure
pub async fn run(config: BarmWorkerConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Log trace context if available
    if let Some(ref tp) = config.trace_context.traceparent {
        tracing::info!(
            traceparent = %tp,
            trace_id = config.trace_context.trace_id.as_deref().unwrap_or("N/A"),
            span_id = config.trace_context.span_id.as_deref().unwrap_or("N/A"),
            "Barm worker started with trace context"
        );
    } else {
        tracing::info!("Barm worker started (no trace context)");
    }

    // Log memory configuration
    tracing::info!(
        worker_id = %config.worker_id,
        model = %config.model_name,
        max_model_len = ?config.memory_config.max_model_len,
        max_num_seqs = ?config.memory_config.max_num_seqs,
        kv_fraction = ?config.memory_config.kv_fraction,
        "Barm worker memory configuration"
    );

    let mut client = ZenohClient::new(
        &config.zenoh_peer,
        config.cache_dir,
        config.memory_config,
    )
    .await
    .context("Failed to create Zenoh client")?;

    tracing::info!(
        "Barm worker connected to {} for model: {}",
        config.zenoh_peer,
        config.model_name
    );

    client.run(&config.model_name)
        .await
        .context("Failed to run Zenoh client")?;

    Ok(())
}

/// Run the barm worker with legacy parameters (for backward compatibility)
///
/// # Arguments
/// * `zenoh_peer` - The Zenoh peer endpoint to connect to
/// * `model_name` - The model name to receive assets for
/// * `cache_dir` - Directory to cache model assets
/// * `max_model_len` - Maximum model context length
/// * `max_num_seqs` - Maximum concurrent sequences
/// * `kv_fraction` - KV cache memory fraction
/// * `traceparent` - W3C traceparent for distributed tracing
///
/// # Returns
/// Result indicating success or failure
#[deprecated(since = "0.1.0", note = "Use run(BarmWorkerConfig) instead")]
pub async fn run_legacy(
    zenoh_peer: String,
    model_name: String,
    cache_dir: PathBuf,
    max_model_len: Option<usize>,
    max_num_seqs: Option<usize>,
    kv_fraction: Option<f32>,
    traceparent: Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = BarmWorkerConfig::new(zenoh_peer, model_name, cache_dir)
        .with_memory_config(max_model_len, max_num_seqs, kv_fraction)
        .with_trace_context(traceparent);

    run(config).await
}
