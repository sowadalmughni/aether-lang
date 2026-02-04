//! Aether Core
//!
//! Shared types for the Aether programming language compiler and runtime.
//! This crate defines the DAG (Directed Acyclic Graph) structures that represent
//! compiled Aether programs, including template reference metadata for runtime
//! substitution.

pub mod dag;
pub mod error;

pub use dag::*;
pub use error::*;
