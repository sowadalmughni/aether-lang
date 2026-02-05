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
    cache::LlmCache,
    telemetry::init_telemetry,
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::DagNode;

    fn make_test_node(id: &str, deps: Vec<&str>) -> DagNode {
        DagNode {
            id: id.to_string(),
            node_type: DagNodeType::Compute,
            name: Some(id.to_string()),
            prompt_template: None,
            prompt: None,
            template_refs: Vec::new(),
            model: None,
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            dependencies: deps.into_iter().map(|s| s.to_string()).collect(),
            cache_key_inputs: Vec::new(),
            render_policy: Default::default(),
            execution_hints: Default::default(),
            return_type: None,
            source_location: None,
        }
    }

    #[test]
    fn test_build_dependency_graph() {
        let dag = Dag::with_nodes(vec![
            make_test_node("a", vec![]),
            make_test_node("b", vec!["a"]),
            make_test_node("c", vec!["a"]),
            make_test_node("d", vec!["b", "c"]),
        ]);

        let level_info = get_execution_levels(&dag).unwrap();

        // Level 0: a (no deps)
        // Level 1: b, c (depend on a)
        // Level 2: d (depends on b and c)
        assert_eq!(level_info.levels.len(), 3);
        assert_eq!(level_info.levels[0], vec!["a"]);
        assert!(level_info.levels[1].contains(&"b".to_string()));
        assert!(level_info.levels[1].contains(&"c".to_string()));
        assert_eq!(level_info.levels[2], vec!["d"]);

        // Check node_levels mapping
        assert_eq!(level_info.node_levels.get("a"), Some(&0));
        assert_eq!(level_info.node_levels.get("b"), Some(&1));
        assert_eq!(level_info.node_levels.get("c"), Some(&1));
        assert_eq!(level_info.node_levels.get("d"), Some(&2));
    }

    #[test]
    fn test_cycle_detection() {
        let dag = Dag::with_nodes(vec![
            make_test_node("a", vec!["b"]),
            make_test_node("b", vec!["a"]),
        ]);

        let result = get_execution_levels(&dag);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Cycle"), "Expected cycle error, got: {}", err);
    }

    #[test]
    fn test_cycle_detection_three_nodes() {
        let dag = Dag::with_nodes(vec![
            make_test_node("a", vec!["c"]),
            make_test_node("b", vec!["a"]),
            make_test_node("c", vec!["b"]),
        ]);

        let result = get_execution_levels(&dag);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Cycle"), "Expected cycle error, got: {}", err);
    }

    #[test]
    fn test_parallel_independent_nodes() {
        let dag = Dag::with_nodes(vec![
            make_test_node("a", vec![]),
            make_test_node("b", vec![]),
            make_test_node("c", vec![]),
        ]);

        let level_info = get_execution_levels(&dag).unwrap();

        // All nodes are independent, should be in level 0
        assert_eq!(level_info.levels.len(), 1);
        assert_eq!(level_info.levels[0].len(), 3);
    }

    #[test]
    fn test_complex_dag_levels() {
        // Diamond pattern: a -> b, c -> d (plus e independent)
        let dag = Dag::with_nodes(vec![
            make_test_node("a", vec![]),
            make_test_node("b", vec!["a"]),
            make_test_node("c", vec!["a"]),
            make_test_node("d", vec!["b", "c"]),
            make_test_node("e", vec![]),  // Independent of the diamond
        ]);

        let level_info = get_execution_levels(&dag).unwrap();

        assert_eq!(level_info.levels.len(), 3);
        // Level 0: a and e (both independent)
        assert!(level_info.levels[0].contains(&"a".to_string()));
        assert!(level_info.levels[0].contains(&"e".to_string()));
        // Level 1: b and c (depend on a)
        assert!(level_info.levels[1].contains(&"b".to_string()));
        assert!(level_info.levels[1].contains(&"c".to_string()));
        // Level 2: d (depends on b and c)
        assert!(level_info.levels[2].contains(&"d".to_string()));
    }

    #[test]
    fn test_empty_dag() {
        let dag = Dag::with_nodes(vec![]);
        let level_info = get_execution_levels(&dag).unwrap();
        assert!(level_info.levels.is_empty());
        assert!(level_info.node_levels.is_empty());
    }

    #[test]
    fn test_single_node_dag() {
        let dag = Dag::with_nodes(vec![make_test_node("only", vec![])]);
        let level_info = get_execution_levels(&dag).unwrap();
        assert_eq!(level_info.levels.len(), 1);
        assert_eq!(level_info.levels[0], vec!["only".to_string()]);
    }

    // =========================================================================
    // Percentile Computation Tests
    // =========================================================================

    #[test]
    fn test_percentiles_empty_samples() {
        let percentiles = compute_percentiles(&[]);
        assert!(percentiles.p50.is_none());
        assert!(percentiles.p95.is_none());
        assert!(percentiles.p99.is_none());
    }

    #[test]
    fn test_percentiles_single_sample() {
        let percentiles = compute_percentiles(&[100]);
        assert_eq!(percentiles.p50, Some(100));
        assert_eq!(percentiles.p95, Some(100));
        assert_eq!(percentiles.p99, Some(100));
    }

    #[test]
    fn test_percentiles_two_samples() {
        let percentiles = compute_percentiles(&[10, 20]);
        // n=2, p50_idx = floor(0.50 * 1) = 0, p95_idx = floor(0.95 * 1) = 0, p99_idx = floor(0.99 * 1) = 0
        assert_eq!(percentiles.p50, Some(10));
        assert_eq!(percentiles.p95, Some(10));
        assert_eq!(percentiles.p99, Some(10));
    }

    #[test]
    fn test_percentiles_ten_samples() {
        // 10 samples: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let samples: Vec<u64> = (1..=10).collect();
        let percentiles = compute_percentiles(&samples);
        // n=10, indices:
        // p50_idx = floor(0.50 * 9) = floor(4.5) = 4 -> samples[4] = 5
        // p95_idx = floor(0.95 * 9) = floor(8.55) = 8 -> samples[8] = 9
        // p99_idx = floor(0.99 * 9) = floor(8.91) = 8 -> samples[8] = 9
        assert_eq!(percentiles.p50, Some(5));
        assert_eq!(percentiles.p95, Some(9));
        assert_eq!(percentiles.p99, Some(9));
    }

    #[test]
    fn test_percentiles_hundred_samples() {
        // 100 samples: [1, 2, 3, ..., 100]
        let samples: Vec<u64> = (1..=100).collect();
        let percentiles = compute_percentiles(&samples);
        // n=100, indices:
        // p50_idx = floor(0.50 * 99) = floor(49.5) = 49 -> samples[49] = 50
        // p95_idx = floor(0.95 * 99) = floor(94.05) = 94 -> samples[94] = 95
        // p99_idx = floor(0.99 * 99) = floor(98.01) = 98 -> samples[98] = 99
        assert_eq!(percentiles.p50, Some(50));
        assert_eq!(percentiles.p95, Some(95));
        assert_eq!(percentiles.p99, Some(99));
    }

    #[test]
    fn test_percentiles_unsorted_input() {
        // Unsorted input should still work
        let samples = vec![50, 10, 90, 30, 70, 20, 80, 40, 60, 100];
        let percentiles = compute_percentiles(&samples);
        // After sorting: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        // n=10, p50_idx = floor(0.50 * 9) = 4 -> 50
        assert_eq!(percentiles.p50, Some(50));
        assert_eq!(percentiles.p95, Some(90));
        assert_eq!(percentiles.p99, Some(90));
    }

    #[test]
    fn test_percentiles_with_duplicates() {
        let samples = vec![100, 100, 100, 100, 100, 200, 200, 200, 200, 200];
        let percentiles = compute_percentiles(&samples);
        // After sorting: [100, 100, 100, 100, 100, 200, 200, 200, 200, 200]
        // n=10, p50_idx = 4 -> 100
        assert_eq!(percentiles.p50, Some(100));
        // p95_idx = 8 -> 200
        assert_eq!(percentiles.p95, Some(200));
        assert_eq!(percentiles.p99, Some(200));
    }
}

