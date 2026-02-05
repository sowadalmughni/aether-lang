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

use aether_core::{NodeState, NodeStatus, ErrorPolicy};
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
mod context;
mod llm;
mod security;
mod telemetry;
mod template;

use cache::{CacheConfig, CacheKey, CachedResponse, LlmCache};
use context::ExecutionContext;
use llm::{LlmClient, LlmConfig, LlmRequest, create_client};
use security::{DefaultInputSanitizer, SecurityConfig, SecurityError, SecurityMiddleware};
use template::render_node_prompt;

// =============================================================================
// Percentile Computation
// =============================================================================

/// Percentile results for latency measurements
#[derive(Debug, Clone, Default)]
pub struct LatencyPercentiles {
    pub p50: Option<u64>,
    pub p95: Option<u64>,
    pub p99: Option<u64>,
}

/// Compute p50, p95, p99 percentiles from a vector of samples.
///
/// Uses the floor-based index method:
/// - p50 index = floor(0.50 * (n - 1))
/// - p95 index = floor(0.95 * (n - 1))
/// - p99 index = floor(0.99 * (n - 1))
///
/// Returns None for each percentile if the samples vector is empty.
pub fn compute_percentiles(samples: &[u64]) -> LatencyPercentiles {
    if samples.is_empty() {
        return LatencyPercentiles::default();
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let n = sorted.len();

    // Compute indices using floor method
    let p50_idx = ((0.50 * (n - 1) as f64).floor()) as usize;
    let p95_idx = ((0.95 * (n - 1) as f64).floor()) as usize;
    let p99_idx = ((0.99 * (n - 1) as f64).floor()) as usize;

    LatencyPercentiles {
        p50: Some(sorted[p50_idx]),
        p95: Some(sorted[p95_idx]),
        p99: Some(sorted[p99_idx]),
    }
}

// =============================================================================
// Data Structures (re-export from aether-core and add runtime-specific types)
// =============================================================================

// Re-export core types for public API
pub use aether_core::{Dag, DagNode, DagNodeType, NodeExecutionResult as ExecutionResult, DagExecutionResponse};

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

/// Result of topological sort with levels
#[derive(Debug)]
pub struct DagLevelInfo {
    /// Nodes grouped by execution level
    pub levels: Vec<Vec<String>>,
    /// Mapping of node_id to its level
    pub node_levels: HashMap<String, usize>,
}

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
/// Returns detailed level info including node-to-level mapping
fn get_execution_levels(dag: &Dag) -> Result<DagLevelInfo, String> {
    let (graph, node_indices) = build_dependency_graph(dag);

    // Topological sort to detect cycles
    let sorted = toposort(&graph, None).map_err(|cycle| {
        // Try to identify the node involved in the cycle
        let node_id = graph.node_weight(cycle.node_id())
            .map(|s| s.clone())
            .unwrap_or_else(|| "unknown".to_string());
        format!("Cycle detected in DAG involving node '{}'", node_id)
    })?;

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
    if node_levels.is_empty() {
        return Ok(DagLevelInfo {
            levels: Vec::new(),
            node_levels,
        });
    }
    let max_level = node_levels.values().max().copied().unwrap();
    let mut levels: Vec<Vec<String>> = vec![Vec::new(); max_level + 1];

    for (node_id, level) in &node_levels {
        levels[*level].push(node_id.clone());
    }

    Ok(DagLevelInfo {
        levels,
        node_levels,
    })
}


/// Internal result for node execution (not the API response type)
#[derive(Debug)]
struct InternalNodeResult {
    output: String,
    token_cost: u32,
    cache_hit: bool,
    input_tokens: u32,
    output_tokens: u32,
    rendered_prompt: Option<String>,
}

/// Execute a single node with template rendering, caching, and LLM call
#[instrument(skip(security, cache, llm_config, context, outputs))]
async fn execute_node(
    node: &DagNode,
    security: &SecurityMiddleware,
    cache: &LlmCache,
    llm_config: &LlmConfig,
    context: &ExecutionContext,
    outputs: &HashMap<String, String>,
) -> Result<InternalNodeResult, String> {
    // Apply security checks if this is an LLM node with a prompt
    if node.node_type == DagNodeType::LlmFn {
        // Try prompt_template first, then legacy prompt field
        let template = node.prompt_template.as_ref().or(node.prompt.as_ref());
        if let Some(template) = template {
            // Render the template with context and node outputs
            let render_result = render_node_prompt(node, context, outputs)
                .map_err(|e| format!("Template rendering failed: {}", e))?;

            let resolved_prompt = render_result.rendered;

            // Security check on rendered prompt
            match security.process_prompt(&resolved_prompt).await {
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

            // Check cache using the rendered prompt
            let cache_key = CacheKey::from_dag_node(node, &resolved_prompt);

            if let Some(cached) = cache.get(&cache_key) {
                info!(node_id = %node.id, "Cache hit - returning cached response");
                return Ok(InternalNodeResult {
                    output: cached.output,
                    token_cost: 0, // No cost for cached responses
                    cache_hit: true,
                    input_tokens: 0,
                    output_tokens: 0,
                    rendered_prompt: Some(resolved_prompt),
                });
            }

            // Call LLM (real or mock based on feature flag and config)
            let model = node.model.clone().unwrap_or_else(|| llm_config.default_model.clone());
            let client = create_client(llm_config, &model);

            let request = LlmRequest {
                prompt: resolved_prompt.clone(),
                model: model.clone(),
                temperature: node.temperature,
                max_tokens: node.max_tokens,
                system_prompt: node.system_prompt.clone(),
            };

            let llm_response = client.complete(request).await.map_err(|e| e.to_string())?;

            let output = llm_response.content.clone();
            let token_cost = llm_response.total_tokens;

            // Cache the result
            let response = CachedResponse::new(
                output.clone(),
                token_cost,
                cache_key.hash(),
            )
            .with_tokens(llm_response.input_tokens, llm_response.output_tokens)
            .with_model(&model, &llm_response.provider);

            cache.put(&cache_key, response);

            return Ok(InternalNodeResult {
                output,
                token_cost,
                cache_hit: false,
                input_tokens: llm_response.input_tokens,
                output_tokens: llm_response.output_tokens,
                rendered_prompt: Some(resolved_prompt),
            });
        }
    }

    // Simulate regular function call (non-LLM nodes)
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    Ok(InternalNodeResult {
        output: format!("Function {} executed", node.id),
        token_cost: 0,
        cache_hit: false,
        input_tokens: 0,
        output_tokens: 0,
        rendered_prompt: None,
    })
}

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
    let start_time = Instant::now();
    let sequential_mode = query.sequential;

    // Create execution context
    let context = if let Some(vars) = request.context {
        ExecutionContext::with_variables(&execution_id, vars)
    } else {
        ExecutionContext::new(&execution_id)
    };

    info!(
        execution_id = %execution_id,
        node_count = dag.nodes.len(),
        context_vars = context.variables.len(),
        sequential_mode = sequential_mode,
        "Starting DAG execution"
    );

    // Get execution levels for parallel processing
    let level_info = match get_execution_levels(&dag) {
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
                level_execution_times_ms: Vec::new(),
                max_concurrency_used: 0,
                node_execution_times_ms: HashMap::new(),
                node_levels: HashMap::new(),
                node_status: HashMap::new(),
                aborted: true,
                skipped_nodes: Vec::new(),
                tokens_saved: 0,
                node_latency_p50_ms: None,
                node_latency_p95_ms: None,
                node_latency_p99_ms: None,
                level_latency_p50_ms: None,
                level_latency_p95_ms: None,
                level_latency_p99_ms: None,
                sequential_mode,
            }));
        }
    };

    // Determine error policy from DAG metadata
    let error_policy = dag.nodes.first()
        .and_then(|n| n.execution_hints.error_policy.as_ref())
        .map(|s| ErrorPolicy::from_str(s))
        .unwrap_or(ErrorPolicy::Fail);

    let mut results: Vec<ExecutionResult> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut total_token_cost = 0u32;
    let mut outputs: HashMap<String, String> = HashMap::new();
    let mut total_cache_hits = 0u32;
    let mut total_cache_misses = 0u32;
    let mut max_parallel = 0usize;
    let mut level_times: Vec<u64> = Vec::new();
    let mut node_execution_times: HashMap<String, u64> = HashMap::new();
    let mut node_status_map: HashMap<String, NodeStatus> = HashMap::new();
    let mut skipped_nodes: Vec<String> = Vec::new();
    let mut aborted = false;
    let mut tokens_saved = 0u32;

    // Create a map from node_id to node for quick lookup
    let node_map: HashMap<&str, &DagNode> = dag.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Execute level by level
    for (level_idx, level_nodes) in level_info.levels.iter().enumerate() {
        // Check if we should abort
        if aborted && error_policy == ErrorPolicy::Fail {
            // Mark remaining nodes as skipped
            for node_id in level_nodes {
                node_status_map.insert(node_id.clone(), NodeStatus::skipped("Aborted due to previous failure"));
                skipped_nodes.push(node_id.clone());
            }
            continue;
        }

        let level_start = Instant::now();
        let parallel_count = level_nodes.len();
        // In sequential mode, max_parallel is always 1 (except for the first level with 0 nodes)
        let effective_parallel = if sequential_mode { 1.min(parallel_count) } else { parallel_count };
        max_parallel = max_parallel.max(effective_parallel);

        state.metrics.parallel_nodes.set(effective_parallel as f64);

        let span = span!(Level::INFO, "execute_level", level = level_idx, nodes = parallel_count, sequential = sequential_mode);
        let _enter = span.enter();

        info!(level = level_idx, parallel_nodes = parallel_count, sequential_mode = sequential_mode, "Executing level");

        if sequential_mode {
            // Sequential execution: run nodes one at a time
            for node_id in level_nodes {
                // Skip nodes whose dependencies failed (if Skip policy)
                if error_policy == ErrorPolicy::Skip {
                    let deps_failed = node_map.get(node_id.as_str())
                        .map(|n| n.dependencies.iter().any(|dep| {
                            node_status_map.get(dep)
                                .map(|s| matches!(s.state, NodeState::Failed | NodeState::Skipped))
                                .unwrap_or(false)
                        }))
                        .unwrap_or(false);

                    if deps_failed {
                        node_status_map.insert(node_id.clone(), NodeStatus::skipped("Dependency failed"));
                        skipped_nodes.push(node_id.clone());
                        continue;
                    }
                }

                if let Some(node) = node_map.get(node_id.as_str()) {
                    let node_start = Instant::now();
                    let result = execute_node(node, &state.security, &state.cache, &state.llm_config, &context, &outputs).await;
                    let execution_time_ms = node_start.elapsed().as_millis() as u64;

                    match result {
                        Ok(node_result) => {
                            state.metrics.executed_nodes.inc();
                            state.metrics.token_cost.inc_by(node_result.token_cost as f64);
                            state.metrics.execution_time.observe(execution_time_ms as f64 / 1000.0);

                            if node_result.cache_hit {
                                state.metrics.cache_hits.inc();
                                total_cache_hits += 1;
                                tokens_saved += node_result.input_tokens + node_result.output_tokens;
                            } else {
                                state.metrics.cache_misses.inc();
                                total_cache_misses += 1;
                            }

                            total_token_cost += node_result.token_cost;
                            outputs.insert(node_id.clone(), node_result.output.clone());
                            node_execution_times.insert(node_id.clone(), execution_time_ms);
                            node_status_map.insert(node_id.clone(), NodeStatus::succeeded());

                            results.push(ExecutionResult {
                                node_id: node_id.clone(),
                                output: node_result.output,
                                execution_time_ms,
                                token_cost: node_result.token_cost,
                                cache_hit: node_result.cache_hit,
                                rendered_prompt: node_result.rendered_prompt,
                                input_tokens: node_result.input_tokens,
                                output_tokens: node_result.output_tokens,
                                level: level_idx,
                            });
                        }
                        Err(e) => {
                            state.metrics.errors.inc();
                            errors.push(format!("Node {}: {}", node_id, e));
                            node_status_map.insert(node_id.clone(), NodeStatus::failed(&e));
                            node_execution_times.insert(node_id.clone(), 0);

                            if error_policy == ErrorPolicy::Fail {
                                aborted = true;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            // Parallel execution: spawn all nodes in this level
            let mut join_set = JoinSet::new();

            for node_id in level_nodes {
                // Skip nodes whose dependencies failed (if Skip policy)
                if error_policy == ErrorPolicy::Skip {
                    let deps_failed = node_map.get(node_id.as_str())
                        .map(|n| n.dependencies.iter().any(|dep| {
                            node_status_map.get(dep)
                                .map(|s| matches!(s.state, NodeState::Failed | NodeState::Skipped))
                                .unwrap_or(false)
                        }))
                        .unwrap_or(false);

                    if deps_failed {
                        node_status_map.insert(node_id.clone(), NodeStatus::skipped("Dependency failed"));
                        skipped_nodes.push(node_id.clone());
                        continue;
                    }
                }

                if let Some(node) = node_map.get(node_id.as_str()) {
                    let node = (*node).clone();
                    let security = state.security.clone();
                    let cache = state.cache.clone();
                    let llm_config = state.llm_config.clone();
                    let current_outputs = outputs.clone();
                    let ctx = context.clone();
                    let level = level_idx;

                    join_set.spawn(async move {
                        let node_start = Instant::now();
                        let result = execute_node(&node, &security, &cache, &llm_config, &ctx, &current_outputs).await;
                        let execution_time_ms = node_start.elapsed().as_millis() as u64;
                        (node.id.clone(), result, execution_time_ms, level)
                    });
                }
            }

            // Collect results from this level
            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok((node_id, Ok(node_result), execution_time_ms, level)) => {
                        state.metrics.executed_nodes.inc();
                        state.metrics.token_cost.inc_by(node_result.token_cost as f64);
                        state.metrics.execution_time.observe(execution_time_ms as f64 / 1000.0);

                        if node_result.cache_hit {
                            state.metrics.cache_hits.inc();
                            total_cache_hits += 1;
                            // Track tokens saved (from original cached response)
                            tokens_saved += node_result.input_tokens + node_result.output_tokens;
                        } else {
                            state.metrics.cache_misses.inc();
                            total_cache_misses += 1;
                        }

                        total_token_cost += node_result.token_cost;
                        outputs.insert(node_id.clone(), node_result.output.clone());
                        node_execution_times.insert(node_id.clone(), execution_time_ms);
                        node_status_map.insert(node_id.clone(), NodeStatus::succeeded());

                    results.push(ExecutionResult {
                        node_id,
                        output: node_result.output,
                        execution_time_ms,
                        token_cost: node_result.token_cost,
                        cache_hit: node_result.cache_hit,
                        rendered_prompt: node_result.rendered_prompt,
                        input_tokens: node_result.input_tokens,
                        output_tokens: node_result.output_tokens,
                        level,
                    });
                }
                Ok((node_id, Err(e), _, _)) => {
                    state.metrics.errors.inc();
                    errors.push(format!("Node {}: {}", node_id, e));
                    node_status_map.insert(node_id.clone(), NodeStatus::failed(&e));
                    node_execution_times.insert(node_id.clone(), 0);

                    if error_policy == ErrorPolicy::Fail {
                        aborted = true;
                    }
                }
                Err(e) => {
                    state.metrics.errors.inc();
                    errors.push(format!("Task join error: {}", e));
                    if error_policy == ErrorPolicy::Fail {
                        aborted = true;
                    }
                }
            }
        }
        } // end else (parallel execution)

        let level_time = level_start.elapsed().as_millis() as u64;
        level_times.push(level_time);

        info!(
            level = level_idx,
            level_time_ms = level_time,
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

    // Compute latency percentiles
    // Collect node latencies (only for succeeded/failed nodes, exclude skipped)
    let node_latency_samples: Vec<u64> = node_execution_times
        .iter()
        .filter(|(node_id, _)| {
            node_status_map
                .get(*node_id)
                .map(|s| matches!(s.state, NodeState::Succeeded | NodeState::Failed))
                .unwrap_or(false)
        })
        .map(|(_, &time)| time)
        .collect();

    let node_percentiles = compute_percentiles(&node_latency_samples);

    // Collect level latencies (exclude empty levels)
    let level_latency_samples: Vec<u64> = level_times.iter().copied().collect();
    let level_percentiles = compute_percentiles(&level_latency_samples);

    info!(
        execution_id = %execution_id,
        total_execution_time_ms,
        total_token_cost,
        nodes_executed = results.len(),
        errors_count = errors.len(),
        parallelization_factor,
        cache_hit_rate,
        tokens_saved,
        aborted,
        sequential_mode,
        node_latency_p50 = ?node_percentiles.p50,
        node_latency_p95 = ?node_percentiles.p95,
        node_latency_p99 = ?node_percentiles.p99,
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
        level_execution_times_ms: level_times,
        max_concurrency_used: max_parallel as u32,
        node_execution_times_ms: node_execution_times,
        node_levels: level_info.node_levels,
        node_status: node_status_map,
        aborted,
        skipped_nodes,
        tokens_saved,
        node_latency_p50_ms: node_percentiles.p50,
        node_latency_p95_ms: node_percentiles.p95,
        node_latency_p99_ms: node_percentiles.p99,
        level_latency_p50_ms: level_percentiles.p50,
        level_latency_p95_ms: level_percentiles.p95,
        level_latency_p99_ms: level_percentiles.p99,
        sequential_mode,
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

