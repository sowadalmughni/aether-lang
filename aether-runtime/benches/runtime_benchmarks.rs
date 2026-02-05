use criterion::{criterion_group, criterion_main, Criterion};
use aether_core::{Dag, DagNode, DagNodeType};
use aether_runtime::{
    AppState, Metrics, execute_flow,
    context::ExecutionContext,
    llm::LlmConfig,
    security::{SecurityMiddleware, SecurityConfig},
    cache::{LlmCache, CacheConfig},
};
use std::sync::Arc;
use tokio::runtime::Runtime;
use std::collections::HashMap;

fn create_bench_state() -> AppState {
    let metrics = Arc::new(Metrics::new());
    let security = Arc::new(SecurityMiddleware::new(SecurityConfig::default()));
    // Use large capacity for bench to avoid eviction noise, or small to test eviction?
    // Let's use default.
    let cache = Arc::new(LlmCache::new(CacheConfig::default()));
    let llm_config = Arc::new(LlmConfig::default());

    AppState {
        metrics,
        security,
        cache,
        llm_config,
    }
}

fn create_simple_dag() -> Dag {
    // A -> B -> C
    let node_a = DagNode {
        id: "a".to_string(),
        node_type: DagNodeType::LlmFn,
        prompt: Some("Node A prompt".to_string()),
        ..DagNode::default()
    };
    let node_b = DagNode {
        id: "b".to_string(),
        node_type: DagNodeType::LlmFn,
        prompt: Some("Node B prompt".to_string()),
        dependencies: vec!["a".to_string()],
        ..DagNode::default()
    };
    let node_c = DagNode {
        id: "c".to_string(),
        node_type: DagNodeType::LlmFn,
        prompt: Some("Node C prompt".to_string()),
        dependencies: vec!["b".to_string()],
        ..DagNode::default()
    };

    Dag::with_nodes(vec![node_a, node_b, node_c])
}

fn benchmark_dag_execution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = create_bench_state();
    let dag = create_simple_dag();
    
    // We use a clean context for each run, or reuse?
    // ExecutionContext::new() is cheap.

    c.bench_function("execute_simple_dag_sequential", |b| {
        b.to_async(&rt).iter(|| async {
            let execution_id = "bench-id";
            let context = ExecutionContext::new(execution_id);
            execute_flow(
                &dag,
                &context,
                true, // sequential
                &state,
                execution_id
            ).await
        })
    });

    c.bench_function("execute_simple_dag_parallel", |b| {
        b.to_async(&rt).iter(|| async {
            let execution_id = "bench-id-par";
            let context = ExecutionContext::new(execution_id);
            execute_flow(
                &dag,
                &context,
                false, // parallel (though structure is sequential A->B->C)
                &state,
                execution_id
            ).await
        })
    });
}

fn create_parallel_dag() -> Dag {
    // 10 independent nodes
    let mut nodes = Vec::new();
    for i in 0..10 {
        nodes.push(DagNode {
            id: format!("node_{}", i),
            node_type: DagNodeType::Function, // simpler, faster
            ..DagNode::default()
        }); 
    }
    Dag::with_nodes(nodes)
}

fn benchmark_parallel_dag(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = create_bench_state();
    let dag = create_parallel_dag();

    c.bench_function("execute_parallel_dag_10_nodes", |b| {
        b.to_async(&rt).iter(|| async {
             let execution_id = "bench-id-par-10";
            let context = ExecutionContext::new(execution_id);
            execute_flow(
                &dag,
                &context,
                false, 
                &state,
                execution_id
            ).await
        })
    });
}

criterion_group!(benches, benchmark_dag_execution, benchmark_parallel_dag);
criterion_main!(benches);
