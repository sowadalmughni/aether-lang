use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use prometheus::{Counter, Histogram, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;
use tracing::{info, instrument, span, Level, warn};
use uuid::Uuid;

mod security;
use security::{DefaultInputSanitizer, SecurityConfig, SecurityMiddleware, SecurityError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub node_type: String,
    pub prompt: Option<String>,
    pub model: Option<String>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagExecutionResponse {
    pub execution_id: String,
    pub results: Vec<ExecutionResult>,
    pub total_execution_time_ms: u64,
    pub total_token_cost: u32,
    pub errors: Vec<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub metrics: Arc<Metrics>,
    pub security: Arc<SecurityMiddleware>,
}

pub struct Metrics {
    pub registry: Registry,
    pub executed_nodes: Counter,
    pub token_cost: Counter,
    pub errors: Counter,
    pub execution_time: Histogram,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        
        let executed_nodes = Counter::new("aether_executed_nodes_total", "Total number of executed nodes")
            .expect("Failed to create executed_nodes counter");
        
        let token_cost = Counter::new("aether_token_cost_total", "Total simulated token cost")
            .expect("Failed to create token_cost counter");
        
        let errors = Counter::new("aether_errors_total", "Total number of errors")
            .expect("Failed to create errors counter");
        
        let execution_time = Histogram::with_opts(
            prometheus::HistogramOpts::new("aether_execution_time_seconds", "Node execution time in seconds")
        ).expect("Failed to create execution_time histogram");
        
        registry.register(Box::new(executed_nodes.clone())).unwrap();
        registry.register(Box::new(token_cost.clone())).unwrap();
        registry.register(Box::new(errors.clone())).unwrap();
        registry.register(Box::new(execution_time.clone())).unwrap();
        
        Self {
            registry,
            executed_nodes,
            token_cost,
            errors,
            execution_time,
        }
    }
}

#[instrument(skip(state))]
async fn execute_dag(
    State(state): State<AppState>,
    Json(dag): Json<Dag>,
) -> Result<Json<DagExecutionResponse>, StatusCode> {
    let execution_id = Uuid::new_v4().to_string();
    let start_time = Instant::now();
    
    info!(execution_id = %execution_id, "Starting DAG execution");
    
    let mut results = Vec::new();
    let mut errors = Vec::new();
    let mut total_token_cost = 0u32;
    
    // Simple topological execution (assumes nodes are already sorted)
    for node in &dag.nodes {
        let node_start = Instant::now();
        
        let span = span!(Level::INFO, "execute_node", node_id = %node.id, node_type = %node.node_type);
        let _enter = span.enter();
        
        match execute_node(node, &state.security).await {
            Ok(result) => {
                let execution_time_ms = node_start.elapsed().as_millis() as u64;
                
                // Update metrics
                state.metrics.executed_nodes.inc();
                state.metrics.token_cost.inc_by(result.token_cost as f64);
                state.metrics.execution_time.observe(execution_time_ms as f64 / 1000.0);
                
                total_token_cost += result.token_cost;
                results.push(ExecutionResult {
                    node_id: node.id.clone(),
                    output: result.output,
                    execution_time_ms,
                    token_cost: result.token_cost,
                });
                
                info!(node_id = %node.id, execution_time_ms, token_cost = result.token_cost, "Node executed successfully");
            }
            Err(e) => {
                state.metrics.errors.inc();
                errors.push(format!("Node {}: {}", node.id, e));
                info!(node_id = %node.id, error = %e, "Node execution failed");
            }
        }
    }
    
    let total_execution_time_ms = start_time.elapsed().as_millis() as u64;
    
    info!(
        execution_id = %execution_id,
        total_execution_time_ms,
        total_token_cost,
        nodes_executed = results.len(),
        errors_count = errors.len(),
        "DAG execution completed"
    );
    
    Ok(Json(DagExecutionResponse {
        execution_id,
        results,
        total_execution_time_ms,
        total_token_cost,
        errors,
    }))
}

#[derive(Debug, Serialize)]
struct NodeExecutionResult {
    output: String,
    token_cost: u32,
}

#[instrument(skip(security))]
async fn execute_node(node: &DagNode, security: &SecurityMiddleware) -> Result<NodeExecutionResult, String> {
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
        }
    }
    
    // Simulate some processing time
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    match node.node_type.as_str() {
        "llm_fn" => {
            // Simulate LLM call
            let simulated_token_cost = 50 + (node.prompt.as_ref().map(|p| p.len()).unwrap_or(0) / 4) as u32;
            Ok(NodeExecutionResult {
                output: "TODO".to_string(),
                token_cost: simulated_token_cost,
            })
        }
        "fn" => {
            // Simulate regular function call
            Ok(NodeExecutionResult {
                output: format!("Function {} executed", node.id),
                token_cost: 0,
            })
        }
        _ => Err(format!("Unknown node type: {}", node.node_type)),
    }
}

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    info!("Starting Aether Runtime");
    
    // Initialize metrics and security
    let metrics = Arc::new(Metrics::new());
    let security_config = SecurityConfig::default();
    let sanitizer = DefaultInputSanitizer::new(security_config);
    let security = Arc::new(SecurityMiddleware::new(Box::new(sanitizer)));
    let app_state = AppState { metrics, security };
    
    // Build the application
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/execute", post(execute_dag))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);
    
    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Aether Runtime listening on http://0.0.0.0:3000");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

