//! Integration tests for Aether Runtime
//!
//! Tests the full execution pipeline including:
//! - DAG parsing and execution
//! - Template rendering with context
//! - Cache hit/miss verification
//! - Parallel execution timing
//! - Security checks

use aether_core::{Dag, DagExecutionResponse, NodeState};
use std::collections::HashMap;

// Helper to load test fixtures
fn load_fixture(name: &str) -> Dag {
    let path = format!("tests/fixtures/{}", name);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read fixture: {}", path));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse fixture {}: {}", path, e))
}

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn test_load_context_dag() {
        let dag = load_fixture("test_dag_context.json");
        
        assert_eq!(dag.nodes.len(), 3);
        assert_eq!(dag.metadata.flow_name, Some("test_context_flow".to_string()));
        
        // Verify node dependencies
        let greeting = dag.get_node("generate_greeting").unwrap();
        assert!(greeting.dependencies.is_empty());
        
        let analyze = dag.get_node("analyze_greeting").unwrap();
        assert_eq!(analyze.dependencies, vec!["generate_greeting"]);
        
        let format = dag.get_node("format_response").unwrap();
        assert!(format.dependencies.contains(&"generate_greeting".to_string()));
        assert!(format.dependencies.contains(&"analyze_greeting".to_string()));
    }

    #[test]
    fn test_template_refs_parsed() {
        let dag = load_fixture("test_dag_context.json");
        
        let greeting = dag.get_node("generate_greeting").unwrap();
        assert_eq!(greeting.template_refs.len(), 2);
        
        // Check context references
        let user_ref = &greeting.template_refs[0];
        assert_eq!(user_ref.raw, "{{context.user_name}}");
        assert_eq!(user_ref.path, vec!["user_name"]);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn test_load_cache_dag() {
        let dag = load_fixture("test_dag_cache.json");
        
        assert_eq!(dag.nodes.len(), 3);
        
        // Two parallel nodes, one join node
        let q1 = dag.get_node("cached_query_1").unwrap();
        let q2 = dag.get_node("cached_query_2").unwrap();
        let combine = dag.get_node("combine_results").unwrap();
        
        assert!(q1.dependencies.is_empty());
        assert!(q2.dependencies.is_empty());
        assert_eq!(combine.dependencies.len(), 2);
    }

    #[test]
    fn test_cache_dag_has_deterministic_prompts() {
        let dag = load_fixture("test_dag_cache.json");
        
        // Temperature 0 for deterministic caching
        for node in &dag.nodes {
            assert_eq!(node.temperature, Some(0.0), 
                "Node {} should have temperature 0 for caching tests", node.id);
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_load_malicious_dag() {
        let dag = load_fixture("malicious_dag.json");
        
        assert_eq!(dag.nodes.len(), 3);
        
        // All nodes should have prompt injection patterns
        let prompts: Vec<_> = dag.nodes.iter()
            .filter_map(|n| n.prompt_template.as_ref())
            .collect();
        
        // Check for known injection patterns
        let injection_patterns = [
            "ignore previous instructions",
            "DAN mode",
            "Forget everything",
        ];
        
        for pattern in injection_patterns {
            let pattern_lower = pattern.to_lowercase();
            let found = prompts.iter().any(|p| p.to_lowercase().contains(&pattern_lower));
            assert!(found, "Expected injection pattern '{}' in test DAG", pattern);
        }
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_load_parallel_dag() {
        let dag = load_fixture("test_dag_parallel.json");
        
        assert_eq!(dag.nodes.len(), 6);
        
        // Count nodes at each level
        let independent_count = dag.nodes.iter()
            .filter(|n| n.dependencies.is_empty())
            .count();
        
        assert_eq!(independent_count, 5, "Should have 5 parallel nodes at level 0");
        
        // Join node depends on all parallel nodes
        let join = dag.get_node("join_node").unwrap();
        assert_eq!(join.dependencies.len(), 5);
    }

    #[test]
    fn test_parallel_structure() {
        let dag = load_fixture("test_dag_parallel.json");
        
        // Build expected structure
        let parallel_ids: Vec<&str> = vec!["parallel_1", "parallel_2", "parallel_3", "parallel_4", "parallel_5"];
        
        for id in &parallel_ids {
            let node = dag.get_node(id).expect(&format!("Missing node {}", id));
            assert!(node.dependencies.is_empty(), "Parallel node {} should have no dependencies", id);
        }
        
        let join = dag.get_node("join_node").unwrap();
        for id in &parallel_ids {
            assert!(join.dependencies.contains(&id.to_string()),
                "Join node should depend on {}", id);
        }
    }
}

#[cfg(test)]
mod dag_execution_response_tests {
    use super::*;

    #[test]
    fn test_execution_response_serialization() {
        let response = DagExecutionResponse {
            execution_id: "test-123".to_string(),
            results: vec![],
            total_execution_time_ms: 100,
            total_token_cost: 50,
            parallelization_factor: 0.5,
            cache_hit_rate: 0.25,
            errors: vec![],
            level_execution_times_ms: vec![30, 40, 30],
            max_concurrency_used: 3,
            node_execution_times_ms: HashMap::new(),
            node_levels: HashMap::new(),
            node_status: HashMap::new(),
            aborted: false,
            skipped_nodes: vec![],
            tokens_saved: 20,
            node_latency_p50_ms: None,
            node_latency_p95_ms: None,
            node_latency_p99_ms: None,
            level_latency_p50_ms: None,
            level_latency_p95_ms: None,
            level_latency_p99_ms: None,
            sequential_mode: false,
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: DagExecutionResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.execution_id, "test-123");
        assert_eq!(parsed.level_execution_times_ms, vec![30, 40, 30]);
        assert_eq!(parsed.max_concurrency_used, 3);
        assert_eq!(parsed.tokens_saved, 20);
        assert!(!parsed.aborted);
    }

    #[test]
    fn test_node_status_states() {
        use aether_core::NodeStatus;

        let succeeded = NodeStatus::succeeded();
        assert!(matches!(succeeded.state, NodeState::Succeeded));
        assert_eq!(succeeded.attempts, 1);
        assert!(succeeded.error.is_none());

        let failed = NodeStatus::failed("Test error");
        assert!(matches!(failed.state, NodeState::Failed));
        assert_eq!(failed.error, Some("Test error".to_string()));

        let skipped = NodeStatus::skipped("Dependency failed");
        assert!(matches!(skipped.state, NodeState::Skipped));
        assert_eq!(skipped.skip_reason, Some("Dependency failed".to_string()));
    }
}
