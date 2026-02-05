//! Snapshot tests for DAG JSON output
//!
//! These tests compile example files and snapshot the resulting DAG JSON
//! to ensure consistent output across changes.

use aether_compiler::{Codegen, CodegenConfig, Parser, SemanticAnalyzer};
use insta::assert_json_snapshot;

fn compile_flow(source: &str, flow_name: &str) -> aether_core::Dag {
    let program = Parser::new(source).unwrap().parse_program().unwrap();
    let ctx = SemanticAnalyzer::new().analyze(&program).unwrap();
    let config = CodegenConfig {
        source_file: Some("test.aether".to_string()),
        include_source_locations: false, // Disable for stable snapshots
        fold_constants: false,
        constants: std::collections::HashMap::new(),
    };
    Codegen::new(config).compile_flow(flow_name, &ctx).unwrap()
}

#[test]
fn test_simple_sentiment_dag() {
    let source = r#"
        enum Sentiment { Positive, Neutral, Negative }

        llm fn classify_sentiment(text: string) -> Sentiment {
            model: "gpt-4o",
            temperature: 0.1,
            prompt: "Classify sentiment: {{text}}"
        }

        flow analyze(input: string) -> Sentiment {
            let result = classify_sentiment(input);
            return result;
        }
    "#;

    let dag = compile_flow(source, "analyze");
    assert_json_snapshot!("simple_sentiment_dag", dag);
}

#[test]
fn test_parallel_flow_dag() {
    let source = r#"
        llm fn summarize(text: string) -> string {
            model: "gpt-4o",
            prompt: "Summarize: {{text}}"
        }

        llm fn extract_entities(text: string) -> string {
            model: "gpt-4o",
            prompt: "Extract entities: {{text}}"
        }

        flow analyze(doc: string) -> string {
            let summary = summarize(doc);
            let entities = extract_entities(doc);
            return summary;
        }
    "#;

    let dag = compile_flow(source, "analyze");
    
    // Verify parallel structure: both nodes have no dependencies
    assert_eq!(dag.nodes.len(), 2);
    for node in &dag.nodes {
        assert!(node.dependencies.is_empty(), "Parallel nodes should have no dependencies");
    }
    
    assert_json_snapshot!("parallel_flow_dag", dag);
}

#[test]
fn test_chained_flow_dag() {
    let source = r#"
        llm fn summarize(text: string) -> string {
            model: "gpt-4o",
            prompt: "Summarize: {{text}}"
        }

        llm fn analyze(summary: string) -> string {
            model: "gpt-4o",
            prompt: "Analyze: {{summary}}"
        }

        llm fn extract_actions(analysis: string) -> string {
            model: "gpt-4o",
            prompt: "Actions: {{analysis}}"
        }

        flow pipeline(doc: string) -> string {
            let summary = summarize(doc);
            let analysis = analyze(summary);
            let actions = extract_actions(analysis);
            return actions;
        }
    "#;

    let dag = compile_flow(source, "pipeline");
    
    // Verify chain structure
    assert_eq!(dag.nodes.len(), 3);
    
    let summary_node = dag.nodes.iter().find(|n| n.id == "summary").unwrap();
    let analysis_node = dag.nodes.iter().find(|n| n.id == "analysis").unwrap();
    let actions_node = dag.nodes.iter().find(|n| n.id == "actions").unwrap();
    
    assert!(summary_node.dependencies.is_empty());
    assert_eq!(analysis_node.dependencies, vec!["summary"]);
    assert_eq!(actions_node.dependencies, vec!["analysis"]);
    
    assert_json_snapshot!("chained_flow_dag", dag);
}

#[test]
fn test_template_refs_metadata() {
    let source = r#"
        llm fn greet(name: string, mood: string) -> string {
            model: "gpt-4o",
            prompt: "Say hello to {{name}} in a {{mood}} way"
        }

        flow hello(username: string, style: string) -> string {
            let greeting = greet(username, style);
            return greeting;
        }
    "#;

    let dag = compile_flow(source, "hello");
    
    assert_eq!(dag.nodes.len(), 1);
    let node = &dag.nodes[0];
    
    // Should have template refs for both parameters
    assert!(!node.template_refs.is_empty());
    
    assert_json_snapshot!("template_refs_metadata", dag);
}

#[test]
fn test_dag_metadata() {
    let source = r#"
        llm fn echo(msg: string) -> string {
            model: "gpt-4o",
            prompt: "Echo: {{msg}}"
        }

        flow test_flow(input: string, count: int) -> string {
            let out = echo(input);
            return out;
        }
    "#;

    let dag = compile_flow(source, "test_flow");
    
    // Verify metadata
    assert_eq!(dag.metadata.flow_name, Some("test_flow".to_string()));
    assert_eq!(dag.metadata.inputs.len(), 2);
    assert_eq!(dag.metadata.inputs[0].name, "input");
    assert_eq!(dag.metadata.inputs[1].name, "count");
    
    assert_json_snapshot!("dag_metadata", dag);
}
