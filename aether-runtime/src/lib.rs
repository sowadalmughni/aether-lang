//! Aether Runtime Library
use aether_core::{Dag, DagNode, DagNodeType, NodeState, NodeStatus, ErrorPolicy, NodeExecutionResult as ExecutionResult, DagExecutionResponse};
use prometheus::{Counter, Gauge, Histogram, Registry};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinSet;
use tracing::{info, instrument, span, warn, Level};
use uuid::Uuid;

pub mod cache;
pub mod context;
pub mod llm;
pub mod security;
pub mod telemetry;
pub mod template;

use cache::{LlmCache, CacheKey, CachedResponse};
use context::ExecutionContext;
use llm::{LlmConfig, LlmRequest, create_client};
use security::{SecurityMiddleware, SecurityError};
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

        registry.register(Box::new(executed_nodes.clone())).unwrap();
        registry.register(Box::new(token_cost.clone())).unwrap();
        registry.register(Box::new(errors.clone())).unwrap();
        registry.register(Box::new(cache_hits.clone())).unwrap();
        registry.register(Box::new(cache_misses.clone())).unwrap();
        registry.register(Box::new(execution_time.clone())).unwrap();
        registry.register(Box::new(parallel_nodes.clone())).unwrap();

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
// DAG Execution Logic
// =============================================================================

/// Result of topological sort with levels
#[derive(Debug)]
pub struct DagLevelInfo {
    pub levels: Vec<Vec<String>>,
    pub node_levels: HashMap<String, usize>,
}

fn build_dependency_graph(dag: &Dag) -> (DiGraph<String, ()>, HashMap<String, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut node_indices: HashMap<String, NodeIndex> = HashMap::new();

    // Add all nodes
    for node in &dag.nodes {
        let idx = graph.add_node(node.id.clone());
        node_indices.insert(node.id.clone(), idx);
    }

    // Add edges
    for node in &dag.nodes {
        if let Some(target_idx) = node_indices.get(&node.id) {
            for dep_id in &node.dependencies {
                if let Some(source_idx) = node_indices.get(dep_id) {
                    graph.add_edge(*source_idx, *target_idx, ());
                }
            }
        }
    }

    (graph, node_indices)
}

pub fn get_execution_levels(dag: &Dag) -> Result<DagLevelInfo, String> {
    let (graph, node_indices) = build_dependency_graph(dag);

    // Topological sort to detect cycles
    let sorted = toposort(&graph, None).map_err(|cycle| {
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

#[derive(Debug)]
pub struct InternalNodeResult {
    pub output: String,
    pub token_cost: u32,
    pub cache_hit: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub rendered_prompt: Option<String>,
}

#[instrument(skip(security, cache, llm_config, context, outputs))]
pub async fn execute_node(
    node: &DagNode,
    security: &SecurityMiddleware,
    cache: &LlmCache,
    llm_config: &LlmConfig,
    context: &ExecutionContext,
    outputs: &HashMap<String, String>,
) -> Result<InternalNodeResult, String> {
    if node.node_type == DagNodeType::LlmFn {
        let template = node.prompt_template.as_ref().or(node.prompt.as_ref());
        if let Some(template) = template {
            let render_result = render_node_prompt(node, context, outputs)
                .map_err(|e| format!("Template rendering failed: {}", e))?;

            let resolved_prompt = render_result.rendered;

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

            let cache_key = CacheKey::from_dag_node(node, &resolved_prompt);

            if let Some(cached) = cache.get(&cache_key) {
                info!(node_id = %node.id, "Cache hit - returning cached response");
                return Ok(InternalNodeResult {
                    output: cached.output,
                    token_cost: 0,
                    cache_hit: true,
                    input_tokens: 0,
                    output_tokens: 0,
                    rendered_prompt: Some(resolved_prompt),
                });
            }

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

#[instrument(skip(state, context))]
pub async fn execute_flow(
    dag: &Dag,
    context: &ExecutionContext,
    sequential_mode: bool,
    state: &AppState,
    execution_id: &str,
) -> DagExecutionResponse {
    let start_time = Instant::now();

    info!(
        execution_id = %execution_id,
        node_count = dag.nodes.len(),
        context_vars = context.variables.len(),
        sequential_mode = sequential_mode,
        "Starting DAG execution"
    );

    let level_info = match get_execution_levels(dag) {
        Ok(l) => l,
        Err(e) => {
            state.metrics.errors.inc();
            return DagExecutionResponse {
                execution_id: execution_id.to_string(),
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
            };
        }
    };

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

    let node_map: HashMap<&str, &DagNode> = dag.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    for (level_idx, level_nodes) in level_info.levels.iter().enumerate() {
        if aborted && error_policy == ErrorPolicy::Fail {
            for node_id in level_nodes {
                node_status_map.insert(node_id.clone(), NodeStatus::skipped("Aborted due to previous failure"));
                skipped_nodes.push(node_id.clone());
            }
            continue;
        }

        let level_start = Instant::now();
        let parallel_count = level_nodes.len();
        let effective_parallel = if sequential_mode { 1.min(parallel_count) } else { parallel_count };
        max_parallel = max_parallel.max(effective_parallel);

        state.metrics.parallel_nodes.set(effective_parallel as f64);

        let span = span!(Level::INFO, "execute_level", level = level_idx, nodes = parallel_count, sequential = sequential_mode);
        let _enter = span.enter();

        info!(level = level_idx, parallel_nodes = parallel_count, sequential_mode = sequential_mode, "Executing level");

        if sequential_mode {
            for node_id in level_nodes {
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
                    let result = execute_node(node, &state.security, &state.cache, &state.llm_config, context, &outputs).await;
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
            let mut join_set = JoinSet::new();

            for node_id in level_nodes {
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
                    let security = state.security.clone();
                    let cache = state.cache.clone();
                    let llm_config = state.llm_config.clone();
                    let context = (*context).clone();
                    let outputs = (*outputs).clone();
                    let node_cloned = (*node).clone(); 
                    
                    join_set.spawn(async move {
                        let node_start = Instant::now();
                        let result = execute_node(&node_cloned, &security, &cache, &llm_config, &context, &outputs).await;
                        let execution_time_ms = node_start.elapsed().as_millis() as u64;
                        (node_cloned.id, result, execution_time_ms)
                    });
                }
            }

            while let Some(res) = join_set.join_next().await {
                match res {
                    Ok((node_id, result, execution_time_ms)) => {
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
                                }
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Task execution panicked: {}", e));
                        if error_policy == ErrorPolicy::Fail {
                            aborted = true;
                        }
                    }
                }
            }
        }
        level_times.push(level_start.elapsed().as_millis() as u64);
    }

    let total_time_ms = start_time.elapsed().as_millis() as u64;
    let node_latencies: Vec<u64> = results.iter().map(|r| r.execution_time_ms).collect();
    let node_percentiles = compute_percentiles(&node_latencies);
    let level_percentiles = compute_percentiles(&level_times);

    let total_requests = total_cache_hits + total_cache_misses;
    let cache_hit_rate = if total_requests > 0 {
        total_cache_hits as f64 / total_requests as f64
    } else {
        0.0
    };

    let parallelization_factor = if sequential_mode {
        1.0
    } else {
        let sequential_sum: u64 = node_execution_times.values().sum();
        if total_time_ms > 0 {
            sequential_sum as f64 / total_time_ms as f64
        } else {
            1.0
        }
    };

    DagExecutionResponse {
        execution_id: execution_id.to_string(),
        results,
        total_execution_time_ms: total_time_ms,
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
    }
}
