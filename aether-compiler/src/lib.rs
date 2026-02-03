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
//! 4. **Semantic Analysis** (`semantic.rs`): Type checking and symbol resolution (planned)
//! 5. **IR Generation** (`ir.rs`): Graph-based intermediate representation (planned)
//! 6. **Code Generation** (`codegen/`): Target-specific code emission (planned)
//!
//! # Example
//!
//! ```rust,ignore
//! use aether_compiler::{Lexer, Parser};
//!
//! let source = r#"
//!     llm fn classify(text: string) -> Sentiment {
//!         model: "gpt-4o",
//!         prompt: "Classify: {{text}}"
//!     }
//! "#;
//!
//! let tokens = Lexer::new(source).collect::<Vec<_>>();
//! let ast = Parser::new(&tokens).parse()?;
//! ```

pub mod ast;
pub mod lexer;
pub mod parser;

// Re-export main types
pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::{ParseError, Parser};
