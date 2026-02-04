//! Aether Compiler CLI (aetherc)
//!
//! Command-line interface for the Aether compiler.
//!
//! # Commands
//!
//! - `compile <file> -o <output>`: Parse, analyze, and emit DAG JSON
//! - `check <file>`: Parse and type check only, report errors
//! - `parse <file>`: Parse only, output AST as JSON
//! - `run <file>`: Compile and execute on runtime, display results
//!
//! # Examples
//!
//! ```bash
//! aetherc compile examples/sentiment.aether -o output.json
//! aetherc check examples/sentiment.aether
//! aetherc parse examples/sentiment.aether
//! aetherc run examples/sentiment.aether --runtime-url http://localhost:3000
//! ```

use aether_compiler::{Codegen, CodegenConfig, Parser, SemanticAnalyzer};
use clap::{Parser as ClapParser, Subcommand};
use miette::{miette, IntoDiagnostic, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use url::Url;

/// Default runtime URL
const DEFAULT_RUNTIME_URL: &str = "http://127.0.0.1:3000";

/// Environment variable for runtime URL override
const RUNTIME_URL_ENV: &str = "AETHER_RUNTIME_URL";

#[derive(ClapParser)]
#[command(name = "aetherc")]
#[command(author, version, about = "Aether language compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile Aether source to DAG JSON
    Compile {
        /// Input Aether source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file for DAG JSON (default: stdout)
        #[arg(short, long, value_name = "OUTPUT")]
        output: Option<PathBuf>,

        /// Flow name to compile (if multiple flows exist)
        #[arg(short, long)]
        flow: Option<String>,

        /// Compile all flows to separate files
        #[arg(long)]
        all: bool,
    },

    /// Parse and type-check Aether source (no output)
    Check {
        /// Input Aether source file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Parse Aether source and output AST as JSON
    Parse {
        /// Input Aether source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Pretty-print the AST JSON
        #[arg(short, long)]
        pretty: bool,
    },

    /// Compile and execute Aether source on the runtime
    Run {
        /// Input Aether source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Runtime URL (default: http://127.0.0.1:3000, or AETHER_RUNTIME_URL env var)
        #[arg(long, value_name = "URL")]
        runtime_url: Option<String>,

        /// Flow name to run (if multiple flows exist)
        #[arg(short, long)]
        flow: Option<String>,

        /// Context variables in JSON format (e.g., '{"key": "value"}')
        #[arg(short, long)]
        context: Option<String>,

        /// Save compiled DAG JSON alongside source file
        #[arg(long)]
        save_dag: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile {
            input,
            output,
            flow,
            all,
        } => cmd_compile(input, output, flow, all),
        Commands::Check { input } => cmd_check(input),
        Commands::Parse { input, pretty } => cmd_parse(input, pretty),
        Commands::Run {
            input,
            runtime_url,
            flow,
            context,
            save_dag,
        } => cmd_run(input, runtime_url, flow, context, save_dag),
    }
}

fn cmd_compile(
    input: PathBuf,
    output: Option<PathBuf>,
    flow: Option<String>,
    all: bool,
) -> Result<()> {
    let source = fs::read_to_string(&input)
        .into_diagnostic()
        .map_err(|e| miette!("Failed to read {}: {}", input.display(), e))?;

    // Parse
    let program = Parser::new(&source)
        .map_err(|e| miette!("Lexer error: {:?}", e))?
        .parse_program()
        .map_err(|e| miette!("Parse error: {:?}", e))?;

    // Semantic analysis
    let ctx = SemanticAnalyzer::new()
        .analyze(&program)
        .map_err(|errs| {
            let msgs: Vec<_> = errs.iter().map(|e| format!("  - {}", e)).collect();
            miette!("Semantic errors:\n{}", msgs.join("\n"))
        })?;

    // Configure codegen
    let config = CodegenConfig {
        source_file: Some(input.display().to_string()),
        include_source_locations: true,
        fold_constants: false,
        constants: std::collections::HashMap::new(),
    };
    let codegen = Codegen::new(config);

    if all {
        // Compile all flows
        let dags = codegen
            .compile_all_flows(&ctx)
            .map_err(|e| miette!("Codegen error: {}", e))?;

        if dags.is_empty() {
            return Err(miette!("No flows found in {}", input.display()));
        }

        for (name, dag) in dags {
            let json = dag
                .to_json()
                .into_diagnostic()
                .map_err(|e| miette!("JSON serialization error: {}", e))?;

            if let Some(ref out_dir) = output {
                let out_file = out_dir.join(format!("{}.json", name));
                fs::write(&out_file, &json)
                    .into_diagnostic()
                    .map_err(|e| miette!("Failed to write {}: {}", out_file.display(), e))?;
                eprintln!("Wrote {}", out_file.display());
            } else {
                println!("// Flow: {}", name);
                println!("{}", json);
            }
        }
    } else {
        // Compile a specific flow
        let flow_name = if let Some(name) = flow {
            name
        } else {
            // Find the first flow
            ctx.flow_calls
                .keys()
                .next()
                .cloned()
                .ok_or_else(|| miette!("No flows found in {}", input.display()))?
        };

        let dag = codegen
            .compile_flow(&flow_name, &ctx)
            .map_err(|e| miette!("Codegen error: {}", e))?;

        let json = dag
            .to_json()
            .into_diagnostic()
            .map_err(|e| miette!("JSON serialization error: {}", e))?;

        if let Some(out_path) = output {
            fs::write(&out_path, &json)
                .into_diagnostic()
                .map_err(|e| miette!("Failed to write {}: {}", out_path.display(), e))?;
            eprintln!("Wrote {}", out_path.display());
        } else {
            println!("{}", json);
        }
    }

    Ok(())
}

fn cmd_check(input: PathBuf) -> Result<()> {
    let source = fs::read_to_string(&input)
        .into_diagnostic()
        .map_err(|e| miette!("Failed to read {}: {}", input.display(), e))?;

    // Parse
    let program = Parser::new(&source)
        .map_err(|e| miette!("Lexer error: {:?}", e))?
        .parse_program()
        .map_err(|e| miette!("Parse error: {:?}", e))?;

    // Semantic analysis
    let ctx = SemanticAnalyzer::new().analyze(&program);

    match ctx {
        Ok(ctx) => {
            let llm_count = ctx.llm_functions.len();
            let flow_count = ctx.flow_calls.len();
            let type_count = ctx
                .symbols
                .global_symbols()
                .values()
                .filter(|s| {
                    matches!(
                        s.kind,
                        aether_compiler::semantic::SymbolKind::Struct { .. }
                            | aether_compiler::semantic::SymbolKind::Enum { .. }
                    )
                })
                .count();

            eprintln!("Check passed: {}", input.display());
            eprintln!(
                "  {} LLM function(s), {} flow(s), {} type(s)",
                llm_count, flow_count, type_count
            );
            Ok(())
        }
        Err(errs) => {
            eprintln!("Check failed: {}", input.display());
            for err in &errs {
                eprintln!("  - {}", err);
            }
            Err(miette!("{} error(s) found", errs.len()))
        }
    }
}

fn cmd_parse(input: PathBuf, pretty: bool) -> Result<()> {
    let source = fs::read_to_string(&input)
        .into_diagnostic()
        .map_err(|e| miette!("Failed to read {}: {}", input.display(), e))?;

    // Parse
    let program = Parser::new(&source)
        .map_err(|e| miette!("Lexer error: {:?}", e))?
        .parse_program()
        .map_err(|e| miette!("Parse error: {:?}", e))?;

    // Output AST as JSON
    let json = if pretty {
        serde_json::to_string_pretty(&program)
    } else {
        serde_json::to_string(&program)
    };

    let json = json
        .into_diagnostic()
        .map_err(|e| miette!("JSON serialization error: {}", e))?;

    println!("{}", json);
    Ok(())
}

/// Resolve runtime URL with precedence: flag > env > default
fn resolve_runtime_url(flag_url: Option<String>) -> Result<Url> {
    let url_str = if let Some(url) = flag_url {
        url
    } else if let Ok(env_url) = env::var(RUNTIME_URL_ENV) {
        env_url
    } else {
        DEFAULT_RUNTIME_URL.to_string()
    };

    validate_runtime_url(&url_str)
}

/// Validate that a URL is a valid http or https URL
fn validate_runtime_url(url_str: &str) -> Result<Url> {
    let url = Url::parse(url_str).map_err(|e| miette!("Invalid runtime URL '{}': {}", url_str, e))?;

    match url.scheme() {
        "http" | "https" => Ok(url),
        scheme => Err(miette!(
            "Invalid runtime URL scheme '{}'. Must be http or https.",
            scheme
        )),
    }
}

/// Truncate output string for display
fn truncate_output(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn cmd_run(
    input: PathBuf,
    runtime_url: Option<String>,
    flow: Option<String>,
    context_json: Option<String>,
    save_dag: bool,
) -> Result<()> {
    // Resolve and validate runtime URL
    let runtime_url = resolve_runtime_url(runtime_url)?;
    let execute_url = runtime_url
        .join("/execute")
        .map_err(|e| miette!("Failed to construct execute URL: {}", e))?;

    eprintln!("Runtime URL: {}", runtime_url);

    // Read source
    let source = fs::read_to_string(&input)
        .into_diagnostic()
        .map_err(|e| miette!("Failed to read {}: {}", input.display(), e))?;

    // Parse
    let program = Parser::new(&source)
        .map_err(|e| miette!("Lexer error: {:?}", e))?
        .parse_program()
        .map_err(|e| miette!("Parse error: {:?}", e))?;

    // Semantic analysis
    let ctx = SemanticAnalyzer::new().analyze(&program).map_err(|errs| {
        let msgs: Vec<_> = errs.iter().map(|e| format!("  - {}", e)).collect();
        miette!("Semantic errors:\n{}", msgs.join("\n"))
    })?;

    // Configure codegen
    let config = CodegenConfig {
        source_file: Some(input.display().to_string()),
        include_source_locations: true,
        fold_constants: false,
        constants: std::collections::HashMap::new(),
    };
    let codegen = Codegen::new(config);

    // Compile flow
    let flow_name = if let Some(name) = flow {
        name
    } else {
        ctx.flow_calls
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| miette!("No flows found in {}", input.display()))?
    };

    eprintln!("Compiling flow: {}", flow_name);

    let dag = codegen
        .compile_flow(&flow_name, &ctx)
        .map_err(|e| miette!("Codegen error: {}", e))?;

    let dag_json = dag
        .to_json()
        .into_diagnostic()
        .map_err(|e| miette!("JSON serialization error: {}", e))?;

    // Optionally save DAG JSON
    if save_dag {
        let dag_path = input.with_extension("dag.json");
        fs::write(&dag_path, &dag_json)
            .into_diagnostic()
            .map_err(|e| miette!("Failed to write {}: {}", dag_path.display(), e))?;
        eprintln!("Saved DAG to: {}", dag_path.display());
    }

    // Parse context if provided
    let context: HashMap<String, serde_json::Value> = if let Some(ctx_str) = context_json {
        serde_json::from_str(&ctx_str)
            .into_diagnostic()
            .map_err(|e| miette!("Invalid context JSON: {}", e))?
    } else {
        HashMap::new()
    };

    // Build request payload
    let request_body = serde_json::json!({
        "dag": serde_json::from_str::<serde_json::Value>(&dag_json).unwrap(),
        "context": context
    });

    eprintln!("Executing on runtime...\n");

    // Execute on runtime
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(execute_url.as_str())
        .header("Content-Type", "application/json")
        .body(request_body.to_string())
        .send()
        .into_diagnostic()
        .map_err(|e| miette!("Failed to connect to runtime at {}: {}", runtime_url, e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(miette!(
            "Runtime returned error {}: {}",
            status,
            truncate_output(&body, 200)
        ));
    }

    let result: serde_json::Value = response
        .json()
        .into_diagnostic()
        .map_err(|e| miette!("Failed to parse runtime response: {}", e))?;

    // Print formatted results
    println!("=== Execution Results ===\n");

    // Node outputs
    if let Some(results) = result.get("results").and_then(|r| r.as_array()) {
        println!("Node Outputs:");
        println!("{}", "-".repeat(60));
        for node_result in results {
            let node_id = node_result
                .get("node_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let output = node_result
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let cache_hit = node_result
                .get("cache_hit")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let cache_indicator = if cache_hit { " [CACHED]" } else { "" };

            println!("  {}{}", node_id, cache_indicator);
            println!("    Output: {}", truncate_output(output, 80));
        }
        println!();
    }

    // Execution summary
    println!("Summary:");
    println!("{}", "-".repeat(60));

    if let Some(time) = result.get("total_execution_time_ms") {
        println!("  Total Execution Time: {}ms", time);
    }

    if let Some(cost) = result.get("total_token_cost") {
        println!("  Total Token Cost: {}", cost);
    }

    // Token counts per node
    if let Some(results) = result.get("results").and_then(|r| r.as_array()) {
        println!("\n  Token Counts:");
        for node_result in results {
            let node_id = node_result
                .get("node_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let input_tokens = node_result
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = node_result
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "    {}: {} input, {} output",
                node_id, input_tokens, output_tokens
            );
        }
    }

    // Cache statistics
    if let Some(hit_rate) = result.get("cache_hit_rate") {
        let rate = hit_rate.as_f64().unwrap_or(0.0);
        println!("\n  Cache Hit Rate: {:.1}%", rate * 100.0);
    }

    if let Some(tokens_saved) = result.get("tokens_saved") {
        println!("  Tokens Saved: {}", tokens_saved);
    }

    // Errors
    if let Some(errors) = result.get("errors").and_then(|e| e.as_array()) {
        if !errors.is_empty() {
            println!("\nErrors:");
            println!("{}", "-".repeat(60));
            for error in errors {
                if let Some(err_str) = error.as_str() {
                    println!("  - {}", err_str);
                }
            }
        }
    }

    println!();
    Ok(())
}
