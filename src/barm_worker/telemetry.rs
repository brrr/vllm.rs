//! Telemetry module for vLLM.rs barm-worker
//!
//! Provides OpenTelemetry integration for distributed tracing.

use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{trace, Resource};
use std::time::Duration;
use tracing_subscriber::layer::SubscriberExt;

/// vLLM.rs telemetry configuration
#[derive(Debug, Clone)]
pub struct VllmTelemetryConfig {
    /// Service name for tracing
    pub service_name: String,
    /// OTLP endpoint (optional)
    pub otlp_endpoint: Option<String>,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
}

impl Default for VllmTelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "vllm-rs".to_string(),
            otlp_endpoint: None,
            sampling_rate: 1.0,
        }
    }
}

/// Initialize OpenTelemetry tracing for vLLM.rs
pub fn init_vllm_telemetry(config: &VllmTelemetryConfig) -> anyhow::Result<()> {
    if let Some(endpoint) = &config.otlp_endpoint {
        setup_otlp_tracing(endpoint, &config.service_name, config.sampling_rate)?;
        tracing::info!(
            endpoint = %endpoint,
            service_name = %config.service_name,
            "vLLM.rs OpenTelemetry tracing initialized"
        );
    }
    Ok(())
}

/// Set up OTLP tracing
fn setup_otlp_tracing(
    endpoint: &str,
    service_name: &str,
    sampling_rate: f64,
) -> anyhow::Result<()> {
    // Create OTLP exporter with gRPC (tonic)
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(30))
        .build()?;

    // Create resource with service attributes
    let service_name_owned = service_name.to_string();
    let resource = Resource::builder_empty()
        .with_attributes([
            KeyValue::new("service.name", service_name_owned.clone()),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("deployment.environment", "development"),
        ])
        .build();

    // Create the pipeline with batch exporter
    let tracer_provider = trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .with_sampler(trace::Sampler::TraceIdRatioBased(sampling_rate))
        .build();

    // Set global tracer provider
    global::set_tracer_provider(tracer_provider);

    // Create a static string for the tracer name
    let static_name: &'static str = Box::leak(service_name_owned.into_boxed_str());

    // Get tracer using global::tracer
    let tracer = global::tracer(static_name);

    // Create OpenTelemetry tracing layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Set up the subscriber with the OpenTelemetry layer
    let subscriber = tracing_subscriber::fmt().finish().with(otel_layer);

    // Try to set global subscriber
    if tracing::subscriber::set_global_default(subscriber).is_err() {
        tracing::warn!("Global tracing subscriber already set");
    }

    Ok(())
}

/// Create a child span from trace context received in inference request
pub fn create_child_span_from_request(
    trace_id: &str,
    parent_span_id: &str,
    request_id: &str,
    model: &str,
) -> tracing::Span {
    // Create span with trace context as attributes
    // Note: Full OpenTelemetry context propagation requires more complex setup
    // For now, we record the parent trace info as span attributes
    tracing::info_span!(
        "vllm.inference",
        request_id = %request_id,
        model = %model,
        otel.trace_id = %trace_id,
        otel.parent_span_id = %parent_span_id,
    )
}
