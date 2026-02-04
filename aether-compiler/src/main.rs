//! Aether Compiler CLI (aetherc)
//!
//! Command-line interface for the Aether compiler.
//!
//! # Commands
//!
//! - `compile <file> -o <output>`: Parse, analyze, and emit DAG JSON
//! - `check <file>`: Parse and type check only, report errors
//! - `parse <file>`: Parse only, output AST as JSON
//!
//! # Examples
//!
//! ```bash
//! aetherc compile examples/sentiment.aether -o output.json
//! aetherc check examples/sentiment.aether
//! aetherc parse examples/sentiment.aether
//! ```

use aether_compiler::{Codegen, CodegenConfig, Parser, SemanticAnalyzer};
use clap::{Parser as ClapParser, Subcommand};
use miette::{miette, IntoDiagnostic, Result};
use std::fs;
use std::path::PathBuf;

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
