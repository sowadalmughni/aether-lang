//! OpenTelemetry Telemetry Configuration
//!
//! Provides distributed tracing via OpenTelemetry with OTLP export support.
//! OTLP works with Jaeger, Zipkin, and other backends.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{self, Sampler};
use opentelemetry_sdk::Resource;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub service_name: String,
    pub service_version: String,
    /// Backward-compatible field names from your old config.
    /// If set, treated as OTLP endpoint candidates.
    pub jaeger_agent_endpoint: Option<String>,
    pub jaeger_collector_endpoint: Option<String>,
    pub sampling_ratio: f64,
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
    pub fn has_jaeger(&self) -> bool {
        self.jaeger_agent_endpoint.is_some() || self.jaeger_collector_endpoint.is_some()
    }
}

// =============================================================================
// Initialization
// =============================================================================

pub fn init_telemetry(config: &TelemetryConfig) -> TelemetryGuard {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let resource = Resource::new(vec![
        KeyValue::new("service.name", config.service_name.clone()),
        KeyValue::new("service.version", config.service_version.clone()),
    ]);

    let trace_config = trace::Config::default()
        .with_resource(resource)
        .with_sampler(Sampler::TraceIdRatioBased(config.sampling_ratio));

    let mut tracer_builder = trace::TracerProvider::builder().with_config(trace_config);

    // Optional OTLP exporter setup (compatible with Jaeger backends)
    if config.has_jaeger() {
        let endpoint = config
            .jaeger_collector_endpoint
            .clone()
            .or(config.jaeger_agent_endpoint.clone())
            .unwrap_or_else(|| "http://localhost:4317".to_string());

        match opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(endpoint)
            .build_span_exporter()
        {
            Ok(exporter) => {
                tracer_builder =
                    tracer_builder.with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio);
                info!("OTLP tracing enabled");
            }
            Err(e) => {
                tracing::warn!("Failed to initialize OTLP exporter: {}", e);
            }
        }
    }

    let tracer_provider = tracer_builder.build();
    let tracer = tracer_provider.tracer(config.service_name.clone());

    let _ = global::set_tracer_provider(tracer_provider);

    // let otel_layer = tracing_opentelemetry::layer();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("aether_runtime=info,tower_http=debug,axum=trace"));

    if config.console_output {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().compact())
            // .with(otel_layer)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            // .with(otel_layer)
            .init();
    }

    info!(
        service = %config.service_name,
        version = %config.service_version,
        sampling_ratio = config.sampling_ratio,
        otlp_enabled = config.has_jaeger(),
        "Telemetry initialized"
    );

    TelemetryGuard
}

pub struct TelemetryGuard;

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        global::shutdown_tracer_provider();
    }
}

// =============================================================================
// Span Helpers
// =============================================================================

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

pub fn dag_execution_attributes(execution_id: &str, node_count: usize) -> Vec<KeyValue> {
    vec![
        KeyValue::new("aether.dag.execution_id", execution_id.to_string()),
        KeyValue::new("aether.dag.node_count", node_count as i64),
    ]
}

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
