//! Aether Runtime
//!
//! HTTP server for executing Aether DAG workflows with:
//! - Topological sorting with cycle detection for correct execution order
//! - Parallel execution of independent nodes (dependency-aware)
//! - LRU caching for LLM responses (exact-match, Level 1)
//! - Security middleware for prompt injection detection
//! - Prometheus metrics and OpenTelemetry tracing
//! - ExecutionContext for template substitution
//! - Error policies: Fail, Skip, Retry

use aether_core::Dag;
use axum::{
    extract::{State, Query},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
// use petgraph... removed
use prometheus::{TextEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::{info, instrument};
use uuid::Uuid;

use aether_runtime::{
    AppState, Metrics, execute_flow, DagExecutionResponse,
    context::ExecutionContext,
    llm::LlmConfig,
    security::{SecurityMiddleware, SecurityConfig, DefaultInputSanitizer},
    cache::{LlmCache, CacheConfig},
    telemetry,
};

// Logic moved to aether_runtime library


/// Request body for /execute endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    /// The DAG to execute
    pub dag: Dag,
    /// Optional execution context with variables
    #[serde(default)]
    pub context: Option<HashMap<String, serde_json::Value>>,
}

/// Query parameters for /execute endpoint
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ExecuteQueryParams {
    /// If true, force sequential execution (max_concurrency=1)
    #[serde(default)]
    pub sequential: bool,
}

/// Execute a DAG with parallel execution of independent nodes
#[instrument(skip(state, request, query))]
async fn execute_dag(
    State(state): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<ExecuteQueryParams>,
    Json(request): Json<ExecuteRequest>,
) -> Result<Json<DagExecutionResponse>, StatusCode> {
    let dag = request.dag;
    let execution_id = Uuid::new_v4().to_string();
    let sequential_mode = query.sequential;

    // Create execution context
    let context = if let Some(vars) = request.context {
        ExecutionContext::with_variables(&execution_id, vars)
    } else {
        ExecutionContext::new(&execution_id)
    };

    let response = execute_flow(&dag, &context, sequential_mode, &state, &execution_id).await;

    Ok(Json(response))
}

// =============================================================================
// HTTP Handlers
// =============================================================================

async fn metrics_handler(State(state): State<AppState>) -> Result<String, StatusCode> {
    let encoder = TextEncoder::new();
    let metric_families = state.metrics.registry.gather();

    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => Ok(metrics),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn health_check() -> &'static str {
    "OK"
}

#[derive(Serialize)]
struct CacheStatsResponse {
    entries: usize,
    hits: u64,
    misses: u64,
    hit_rate: f64,
}

async fn cache_stats(State(state): State<AppState>) -> Json<CacheStatsResponse> {
    let stats = state.cache.stats();
    Json(CacheStatsResponse {
        entries: state.cache.len(),
        hits: stats.hits,
        misses: stats.misses,
        hit_rate: stats.hit_rate(),
    })
}

async fn clear_cache(State(state): State<AppState>) -> &'static str {
    state.cache.clear();
    "Cache cleared"
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize telemetry (OpenTelemetry + tracing)
    let telemetry_config = telemetry::TelemetryConfig::default();
    let _telemetry_guard = telemetry::init_telemetry(&telemetry_config);

    info!("Starting Aether Runtime v{}", env!("CARGO_PKG_VERSION"));

    // Initialize LLM configuration
    let llm_config = Arc::new(LlmConfig::default());
    if llm_config.has_api_keys() {
        info!("LLM API keys detected - real API calls enabled");
    } else {
        info!("No LLM API keys - using mock responses");
    }

    // Initialize components
    let metrics = Arc::new(Metrics::new());
    let security_config = SecurityConfig::default();
    let sanitizer = DefaultInputSanitizer::new(security_config);
    let security = Arc::new(SecurityMiddleware::new(Box::new(sanitizer)));
    let cache = Arc::new(LlmCache::new(CacheConfig::default()));

    let app_state = AppState {
        metrics,
        security,
        cache,
        llm_config,
    };

    // Build the application
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/execute", post(execute_dag))
        .route("/metrics", get(metrics_handler))
        .route("/cache/stats", get(cache_stats))
        .route("/cache/clear", post(clear_cache))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Aether Runtime listening on http://0.0.0.0:3000");
    info!("Endpoints:");
    info!("  GET  /health       - Health check");
    info!("  POST /execute      - Execute a DAG");
    info!("  GET  /metrics      - Prometheus metrics");
    info!("  GET  /cache/stats  - Cache statistics");
    info!("  POST /cache/clear  - Clear cache");

    axum::serve(listener, app).await?;

    Ok(())
}


