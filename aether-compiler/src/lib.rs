//! Aether Compiler
//!
//! A compiler for the Aether programming language, designed for type-safe LLM orchestration.
//!
//! # Architecture
//!
//! The compiler follows a traditional pipeline:
//! 1. **Lexer** (`lexer.rs`): Tokenizes Aether source code
//! 2. **Parser** (`parser.rs`): Builds an Abstract Syntax Tree
//! 3. **AST** (`ast.rs`): Node definitions for the syntax tree
//! 4. **Semantic Analysis** (`semantic.rs`): Type checking and symbol resolution
//! 5. **Code Generation** (`codegen.rs`): DAG JSON emission
//!
//! # Example
//!
//! ```rust,ignore
//! use aether_compiler::{Parser, SemanticAnalyzer, Codegen};
//!
//! let source = r#"
//!     llm fn classify(text: string) -> Sentiment {
//!         model: "gpt-4o",
//!         prompt: "Classify: {{text}}"
//!     }
//!
//!     flow analyze(input: string) -> Sentiment {
//!         let result = classify(input);
//!         return result;
//!     }
//! "#;
//!
//! let program = Parser::new(source)?.parse_program()?;
//! let ctx = SemanticAnalyzer::new().analyze(&program)?;
//! let dag = Codegen::default().compile_flow("analyze", &ctx)?;
//! println!("{}", dag.to_json()?);
//! ```

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod semantic;

// Re-export main types
pub use ast::*;
pub use codegen::{Codegen, CodegenConfig};
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::{ParseError, Parser};
pub use semantic::{SemanticAnalyzer, SemanticContext, SemanticError};

// Re-export core types
pub use aether_core::{Dag, DagNode, DagNodeType, TemplateRef, TemplateRefKind};

