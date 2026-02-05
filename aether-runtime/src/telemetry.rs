//! OpenTelemetry Telemetry Configuration
//!
//! Provides distributed tracing via OpenTelemetry with Jaeger export support.
//! Traces capture per-node execution with timing, token usage, and cache metrics.

use opentelemetry::trace::TracerProvider;
use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{self, Sampler};
use opentelemetry_sdk::Resource;
use tracing::info;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

// =============================================================================
// Configuration
// =============================================================================

/// Telemetry configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service name for traces
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Jaeger agent endpoint (host:port for UDP)
    pub jaeger_agent_endpoint: Option<String>,
    /// Jaeger collector endpoint (HTTP)
    pub jaeger_collector_endpoint: Option<String>,
    /// Sampling ratio (0.0 to 1.0)
    pub sampling_ratio: f64,
    /// Whether to enable console output
    pub console_output: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "aether-runtime".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            jaeger_agent_endpoint: std::env::var("JAEGER_AGENT_ENDPOINT").ok(),
            jaeger_collector_endpoint: std::env::var("JAEGER_COLLECTOR_ENDPOINT").ok(),
            sampling_ratio: std::env::var("OTEL_SAMPLING_RATIO")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            console_output: true,
        }
    }
}

impl TelemetryConfig {
    /// Check if Jaeger is configured
    pub fn has_jaeger(&self) -> bool {
        self.jaeger_agent_endpoint.is_some() || self.jaeger_collector_endpoint.is_some()
    }
}

// =============================================================================
// Initialization
// =============================================================================

/// Initialize telemetry with OpenTelemetry and tracing subscriber
///
/// Returns a guard that should be held for the lifetime of the application.
/// When dropped, it will flush pending traces.
pub fn init_telemetry(config: &TelemetryConfig) -> TelemetryGuard {
    // Set up the trace context propagator for distributed tracing
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Create resource with service info
    let resource = Resource::new(vec![
        KeyValue::new("service.name", config.service_name.clone()),
        KeyValue::new("service.version", config.service_version.clone()),
    ]);

    // Create the tracer config
    let trace_config = trace::Config::default()
        .with_resource(resource)
        .with_sampler(Sampler::TraceIdRatioBased(config.sampling_ratio));

    // Create the tracer provider builder
    let mut tracer_builder = trace::TracerProvider::builder()
        .with_config(trace_config);

    // Add Jaeger exporter if configured
    if config.has_jaeger() {
        // Note: opentelemetry-jaeger uses the agent endpoint from env vars by default
        // OTEL_EXPORTER_JAEGER_AGENT_HOST and OTEL_EXPORTER_JAEGER_AGENT_PORT
        // or OTEL_EXPORTER_JAEGER_ENDPOINT for HTTP collector

        if let Some(endpoint) = &config.jaeger_collector_endpoint {
            std::env::set_var("OTEL_EXPORTER_JAEGER_ENDPOINT", endpoint);
        }

        #[allow(deprecated)]
        match opentelemetry_jaeger::new_agent_pipeline()
            .with_service_name(&config.service_name)
            .build_batch(opentelemetry_sdk::runtime::Tokio)
        {
            Ok(exporter) => {
                tracer_builder = tracer_builder.with_batch_exporter(exporter);
                info!(
                    service = %config.service_name,
                    "Jaeger tracing enabled"
                );
            }
            Err(e) => {
                tracing::warn!("Failed to initialize Jaeger exporter: {}", e);
            }
        }
    }

    let tracer_provider = tracer_builder.build();
    let tracer = tracer_provider.tracer("aether-runtime");

    // Set global provider
    let _ = global::set_tracer_provider(tracer_provider);

    // Create OpenTelemetry layer for tracing
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // Build the subscriber
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("aether_runtime=info,tower_http=debug,axum=trace")
    });

    if config.console_output {
        // With console output
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().compact())
            .with(otel_layer)
            .init();
    } else {
        // Without console output (for production with only Jaeger)
        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .init();
    }

    info!(
        service = %config.service_name,
        version = %config.service_version,
        sampling_ratio = config.sampling_ratio,
        jaeger = config.has_jaeger(),
        "Telemetry initialized"
    );

    TelemetryGuard
}

/// Guard that flushes traces on drop
pub struct TelemetryGuard;

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        // Flush any remaining traces
        global::shutdown_tracer_provider();
    }
}

// =============================================================================
// Span Helpers
// =============================================================================

/// Create span attributes for node execution
pub fn node_execution_attributes(
    node_id: &str,
    node_type: &str,
    model: Option<&str>,
) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new("aether.node.id", node_id.to_string()),
        KeyValue::new("aether.node.type", node_type.to_string()),
    ];

    if let Some(m) = model {
        attrs.push(KeyValue::new("aether.llm.model", m.to_string()));
    }

    attrs
}

/// Create span attributes for DAG execution
pub fn dag_execution_attributes(execution_id: &str, node_count: usize) -> Vec<KeyValue> {
    vec![
        KeyValue::new("aether.dag.execution_id", execution_id.to_string()),
        KeyValue::new("aether.dag.node_count", node_count as i64),
    ]
}

/// Create span attributes for LLM calls
pub fn llm_call_attributes(
    model: &str,
    provider: &str,
    input_tokens: u32,
    output_tokens: u32,
    cache_hit: bool,
) -> Vec<KeyValue> {
    vec![
        KeyValue::new("aether.llm.model", model.to_string()),
        KeyValue::new("aether.llm.provider", provider.to_string()),
        KeyValue::new("aether.llm.input_tokens", input_tokens as i64),
        KeyValue::new("aether.llm.output_tokens", output_tokens as i64),
        KeyValue::new("aether.llm.total_tokens", (input_tokens + output_tokens) as i64),
        KeyValue::new("aether.llm.cache_hit", cache_hit),
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "aether-runtime");
        assert!(config.sampling_ratio >= 0.0 && config.sampling_ratio <= 1.0);
    }

    #[test]
    fn test_node_execution_attributes() {
        let attrs = node_execution_attributes("node1", "llm_fn", Some("gpt-4o"));
        assert_eq!(attrs.len(), 3);
    }

    #[test]
    fn test_dag_execution_attributes() {
        let attrs = dag_execution_attributes("exec-123", 5);
        assert_eq!(attrs.len(), 2);
    }

    #[test]
    fn test_llm_call_attributes() {
        let attrs = llm_call_attributes("gpt-4o", "openai", 100, 50, false);
        assert_eq!(attrs.len(), 6);
    }
}
