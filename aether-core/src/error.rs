//! Error types for Aether Core.

use thiserror::Error;

/// Errors that can occur when working with DAGs.
#[derive(Debug, Error)]
pub enum DagError {
    #[error("Node '{0}' not found in DAG")]
    NodeNotFound(String),

    #[error("Dependency cycle detected involving node '{0}'")]
    CycleDetected(String),

    #[error("Node '{node_id}' depends on non-existent node '{dependency}'")]
    MissingDependency { node_id: String, dependency: String },

    #[error("Template reference error: {0}")]
    TemplateError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Errors that can occur during template rendering.
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("Missing required context key: {0}")]
    MissingContextKey(String),

    #[error("Missing required node output: node '{node_id}', field '{field}'")]
    MissingNodeOutput { node_id: String, field: String },

    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    #[error("Value too long for key '{key}': {length} > {max_length}")]
    ValueTooLong {
        key: String,
        length: usize,
        max_length: usize,
    },

    #[error("Context key '{0}' not in allowlist")]
    ContextKeyNotAllowed(String),

    #[error("Security policy violation: {0}")]
    SecurityViolation(String),
}
