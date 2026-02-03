//! Aether Runtime
//!
//! HTTP server for executing Aether DAG workflows with:
//! - Topological sorting for correct execution order
//! - Parallel execution of independent nodes
//! - LRU caching for LLM responses
//! - Security middleware for prompt injection detection
//! - Prometheus metrics and OpenTelemetry tracing

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use prometheus::{Counter, Gauge, Histogram, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;
use tower_http::cors::CorsLayer;
use tracing::{info, instrument, span, warn, Level};
use uuid::Uuid;

mod cache;
mod llm;
mod security;
mod telemetry;

use cache::{CacheConfig, CacheKey, CachedResponse, LlmCache, current_timestamp};
use llm::{LlmClient, LlmConfig, LlmRequest, create_client};
use security::{DefaultInputSanitizer, SecurityConfig, SecurityError, SecurityMiddleware};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub node_type: String,
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dag {
    pub nodes: Vec<DagNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub node_id: String,
    pub output: String,
    pub execution_time_ms: u64,
    pub token_cost: u32,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagExecutionResponse {
    pub execution_id: String,
    pub results: Vec<ExecutionResult>,
    pub total_execution_time_ms: u64,
    pub total_token_cost: u32,
    pub parallelization_factor: f64,
    pub cache_hit_rate: f64,
    pub errors: Vec<String>,
}

// =============================================================================
// Application State
// =============================================================================

#[derive(Clone)]
pub struct AppState {
    pub metrics: Arc<Metrics>,
    pub security: Arc<SecurityMiddleware>,
    pub cache: Arc<LlmCache>,
    pub llm_config: Arc<LlmConfig>,
}

pub struct Metrics {
    pub registry: Registry,
    pub executed_nodes: Counter,
    pub token_cost: Counter,
    pub errors: Counter,
    pub cache_hits: Counter,
    pub cache_misses: Counter,
    pub execution_time: Histogram,
    pub parallel_nodes: Gauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let executed_nodes =
            Counter::new("aether_executed_nodes_total", "Total number of executed nodes")
                .expect("Failed to create executed_nodes counter");

        let token_cost = Counter::new("aether_token_cost_total", "Total simulated token cost")
            .expect("Failed to create token_cost counter");

        let errors = Counter::new("aether_errors_total", "Total number of errors")
            .expect("Failed to create errors counter");

        let cache_hits = Counter::new("aether_cache_hits_total", "Total cache hits")
            .expect("Failed to create cache_hits counter");

        let cache_misses = Counter::new("aether_cache_misses_total", "Total cache misses")
            .expect("Failed to create cache_misses counter");

        let execution_time = Histogram::with_opts(prometheus::HistogramOpts::new(
            "aether_execution_time_seconds",
            "Node execution time in seconds",
        ))
        .expect("Failed to create execution_time histogram");

        let parallel_nodes = Gauge::new(
            "aether_parallel_nodes",
            "Number of nodes executed in parallel in the last batch",
        )
        .expect("Failed to create parallel_nodes gauge");

        registry
            .register(Box::new(executed_nodes.clone()))
            .unwrap();
        registry.register(Box::new(token_cost.clone())).unwrap();
        registry.register(Box::new(errors.clone())).unwrap();
        registry.register(Box::new(cache_hits.clone())).unwrap();
        registry.register(Box::new(cache_misses.clone())).unwrap();
        registry
            .register(Box::new(execution_time.clone()))
            .unwrap();
        registry
            .register(Box::new(parallel_nodes.clone()))
            .unwrap();

        Self {
            registry,
            executed_nodes,
            token_cost,
            errors,
            cache_hits,
            cache_misses,
            execution_time,
            parallel_nodes,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// DAG Execution Engine
// =============================================================================

/// Build a directed graph from DAG nodes for topological sorting
fn build_dependency_graph(dag: &Dag) -> (DiGraph<String, ()>, HashMap<String, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

    // Add all nodes
    for node in &dag.nodes {
        let idx = graph.add_node(node.id.clone());
        node_indices.insert(node.id.clone(), idx);
    }

    // Add edges for dependencies
    for node in &dag.nodes {
        if let Some(&to_idx) = node_indices.get(&node.id) {
            for dep in &node.dependencies {
                if let Some(&from_idx) = node_indices.get(dep) {
                    graph.add_edge(from_idx, to_idx, ());
                }
            }
        }
    }

    (graph, node_indices)
}

/// Get execution levels (nodes that can run in parallel at each level)
fn get_execution_levels(dag: &Dag) -> Result<Vec<Vec<String>>, String> {
    let (graph, node_indices) = build_dependency_graph(dag);

    // Topological sort to detect cycles
    let sorted = toposort(&graph, None).map_err(|_| "Cycle detected in DAG".to_string())?;

    // Create reverse mapping
    let index_to_id: HashMap<NodeIndex, &String> =
        node_indices.iter().map(|(k, v)| (*v, k)).collect();

    // Group nodes by their depth level
    let mut node_levels: HashMap<String, usize> = HashMap::new();

    for idx in sorted {
        let node_id = index_to_id[&idx];
        let node = dag.nodes.iter().find(|n| &n.id == node_id).unwrap();

        // Level is max level of dependencies + 1
        let level = if node.dependencies.is_empty() {
            0
        } else {
            node.dependencies
                .iter()
                .filter_map(|dep| node_levels.get(dep))
                .max()
                .map(|l| l + 1)
                .unwrap_or(0)
        };

        node_levels.insert(node_id.clone(), level);
    }

    // Group by level
    let max_level = node_levels.values().max().copied().unwrap_or(0);
    let mut levels: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];

    for (node_id, level) in node_levels {
        levels[level].push(node_id);
    }

    Ok(levels)
}

/// Internal result for node execution
#[derive(Debug)]
struct NodeExecutionResult {
    output: String,
    token_cost: u32,
    cache_hit: bool,
}

/// Execute a single node
#[instrument(skip(security, cache, llm_config, outputs))]
async fn execute_node(
    node: &DagNode,
    security: &SecurityMiddleware,
    cache: &LlmCache,
    llm_config: &LlmConfig,
    outputs: &HashMap<String, String>,
) -> Result<NodeExecutionResult, String> {
    // Apply security checks if this is an LLM node with a prompt
    if node.node_type == "llm_fn" {
        if let Some(prompt) = &node.prompt {
            match security.process_prompt(prompt).await {
                Ok(_) => {
                    info!(node_id = %node.id, "Security check passed for prompt");
                }
                Err(SecurityError::ProfanityDetected(msg)) => {
                    warn!(node_id = %node.id, error = %msg, "Profanity detected in prompt");
                    return Err(format!("Security violation: {}", msg));
                }
                Err(SecurityError::PromptInjectionDetected(msg)) => {
                    warn!(node_id = %node.id, error = %msg, "Prompt injection detected");
                    return Err(format!("Security violation: {}", msg));
                }
                Err(SecurityError::SanitizationFailed(msg)) => {
                    warn!(node_id = %node.id, error = %msg, "Sanitization failed");
                    return Err(format!("Security error: {}", msg));
                }
            }

            // Check cache
            let cache_key = CacheKey::new(
                prompt.clone(),
                node.model.clone().unwrap_or_else(|| "default".to_string()),
            );
            let cache_key = if let Some(temp) = node.temperature {
                cache_key.with_temperature(temp)
            } else {
                cache_key
            };
            let cache_key = if let Some(max) = node.max_tokens {
                cache_key.with_max_tokens(max)
            } else {
                cache_key
            };

            if let Some(cached) = cache.get(&cache_key) {
                info!(node_id = %node.id, "Cache hit - returning cached response");
                return Ok(NodeExecutionResult {
                    output: cached.output,
                    token_cost: 0, // No cost for cached responses
                    cache_hit: true,
                });
            }

            // Substitute dependency outputs into prompt template
            let mut resolved_prompt = prompt.clone();
            for (dep_id, dep_output) in outputs {
                let placeholder = format!("{{{{{}}}}}", dep_id);
                resolved_prompt = resolved_prompt.replace(&placeholder, dep_output);
            }

            // Call LLM (real or mock based on feature flag and config)
            let model = node.model.clone().unwrap_or_else(|| llm_config.default_model.clone());
            let client = create_client(llm_config, &model);

            let request = LlmRequest {
                prompt: resolved_prompt,
                model: model.clone(),
                temperature: node.temperature,
                max_tokens: node.max_tokens,
                system_prompt: None,
            };

            let llm_response = client.complete(request).await.map_err(|e| e.to_string())?;

            let output = llm_response.content;
            let token_cost = llm_response.total_tokens;

            // Cache the result
            let response = CachedResponse {
                output: output.clone(),
                token_cost,
                cache_key: cache_key.hash(),
                cached_at: current_timestamp(),
            };
            cache.put(&cache_key, response);

            return Ok(NodeExecutionResult {
                output,
                token_cost,
                cache_hit: false,
            });
        }
    }

    // Simulate regular function call
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    Ok(NodeExecutionResult {
        output: format!("Function {} executed", node.id),
        token_cost: 0,
        cache_hit: false,
    })
}

/// Execute a DAG with parallel execution of independent nodes
#[instrument(skip(state))]
async fn execute_dag(
    State(state): State<AppState>,
    Json(dag): Json<Dag>,
) -> Result<Json<DagExecutionResponse>, StatusCode> {
    let execution_id = Uuid::new_v4().to_string();
    let start_time = Instant::now();

    info!(execution_id = %execution_id, node_count = dag.nodes.len(), "Starting DAG execution");

    // Get execution levels for parallel processing
    let levels = match get_execution_levels(&dag) {
        Ok(l) => l,
        Err(e) => {
            state.metrics.errors.inc();
            return Ok(Json(DagExecutionResponse {
                execution_id,
                results: Vec::new(),
                total_execution_time_ms: 0,
                total_token_cost: 0,
                parallelization_factor: 0.0,
                cache_hit_rate: 0.0,
                errors: vec![e],
            }));
        }
    };

    let mut results: Vec<ExecutionResult> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut total_token_cost = 0u32;
    let mut outputs: HashMap<String, String> = HashMap::new();
    let mut total_cache_hits = 0u32;
    let mut total_cache_misses = 0u32;
    let mut max_parallel = 0usize;

    // Create a map from node_id to node for quick lookup
    let node_map: HashMap<&str, &DagNode> = dag.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Execute level by level
    for (level_idx, level_nodes) in levels.iter().enumerate() {
        let level_start = Instant::now();
        let parallel_count = level_nodes.len();
        max_parallel = max_parallel.max(parallel_count);

        state.metrics.parallel_nodes.set(parallel_count as f64);

        let span = span!(Level::INFO, "execute_level", level = level_idx, nodes = parallel_count);
        let _enter = span.enter();

        info!(level = level_idx, parallel_nodes = parallel_count, "Executing level");

        // Execute all nodes in this level in parallel
        let mut join_set = JoinSet::new();

        for node_id in level_nodes {
            if let Some(node) = node_map.get(node_id.as_str()) {
                let node = (*node).clone();
                let security = state.security.clone();
                let cache = state.cache.clone();
                let llm_config = state.llm_config.clone();
                let current_outputs = outputs.clone();

                join_set.spawn(async move {
                    let node_start = Instant::now();
                    let result = execute_node(&node, &security, &cache, &llm_config, &current_outputs).await;
                    let execution_time_ms = node_start.elapsed().as_millis() as u64;
                    (node.id.clone(), result, execution_time_ms)
                });
            }
        }

        // Collect results from this level
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((node_id, Ok(node_result), execution_time_ms)) => {
                    state.metrics.executed_nodes.inc();
                    state
                        .metrics
                        .token_cost
                        .inc_by(node_result.token_cost as f64);
                    state
                        .metrics
                        .execution_time
                        .observe(execution_time_ms as f64 / 1000.0);

                    if node_result.cache_hit {
                        state.metrics.cache_hits.inc();
                        total_cache_hits += 1;
                    } else {
                        state.metrics.cache_misses.inc();
                        total_cache_misses += 1;
                    }

                    total_token_cost += node_result.token_cost;
                    outputs.insert(node_id.clone(), node_result.output.clone());

                    results.push(ExecutionResult {
                        node_id,
                        output: node_result.output,
                        execution_time_ms,
                        token_cost: node_result.token_cost,
                        cache_hit: node_result.cache_hit,
                    });
                }
                Ok((node_id, Err(e), _)) => {
                    state.metrics.errors.inc();
                    errors.push(format!("Node {}: {}", node_id, e));
                }
                Err(e) => {
                    state.metrics.errors.inc();
                    errors.push(format!("Task join error: {}", e));
                }
            }
        }

        info!(
            level = level_idx,
            level_time_ms = level_start.elapsed().as_millis(),
            "Level completed"
        );
    }

    let total_execution_time_ms = start_time.elapsed().as_millis() as u64;

    // Calculate parallelization factor (max parallel / total nodes)
    let parallelization_factor = if dag.nodes.is_empty() {
        0.0
    } else {
        max_parallel as f64 / dag.nodes.len() as f64
    };

    // Calculate cache hit rate
    let total_cache_ops = total_cache_hits + total_cache_misses;
    let cache_hit_rate = if total_cache_ops == 0 {
        0.0
    } else {
        total_cache_hits as f64 / total_cache_ops as f64
    };

    info!(
        execution_id = %execution_id,
        total_execution_time_ms,
        total_token_cost,
        nodes_executed = results.len(),
        errors_count = errors.len(),
        parallelization_factor,
        cache_hit_rate,
        "DAG execution completed"
    );

    Ok(Json(DagExecutionResponse {
        execution_id,
        results,
        total_execution_time_ms,
        total_token_cost,
        parallelization_factor,
        cache_hit_rate,
        errors,
    }))
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

    #[test]
    fn test_build_dependency_graph() {
        let dag = Dag {
            nodes: vec![
                DagNode {
                    id: "a".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec![],
                },
                DagNode {
                    id: "b".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec!["a".to_string()],
                },
                DagNode {
                    id: "c".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec!["a".to_string()],
                },
                DagNode {
                    id: "d".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec!["b".to_string(), "c".to_string()],
                },
            ],
        };

        let levels = get_execution_levels(&dag).unwrap();

        // Level 0: a (no deps)
        // Level 1: b, c (depend on a)
        // Level 2: d (depends on b and c)
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0], vec!["a"]);
        assert!(levels[1].contains(&"b".to_string()));
        assert!(levels[1].contains(&"c".to_string()));
        assert_eq!(levels[2], vec!["d"]);
    }

    #[test]
    fn test_cycle_detection() {
        let dag = Dag {
            nodes: vec![
                DagNode {
                    id: "a".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec!["b".to_string()],
                },
                DagNode {
                    id: "b".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec!["a".to_string()],
                },
            ],
        };

        let result = get_execution_levels(&dag);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cycle"));
    }

    #[test]
    fn test_parallel_independent_nodes() {
        let dag = Dag {
            nodes: vec![
                DagNode {
                    id: "a".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec![],
                },
                DagNode {
                    id: "b".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec![],
                },
                DagNode {
                    id: "c".to_string(),
                    node_type: "fn".to_string(),
                    prompt: None,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    dependencies: vec![],
                },
            ],
        };

        let levels = get_execution_levels(&dag).unwrap();

        // All nodes are independent, should be in level 0
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 3);
    }
}

