//! ExecutionContext for Aether runtime
//!
//! Provides context management for DAG execution. MVP uses in-memory storage.
//! The context subsystem is abstracted behind a store interface to support
//! future pluggable persistence backends (Redis, PostgreSQL, file system).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, instrument};

// =============================================================================
// Context Store Trait (for future persistence backends)
// =============================================================================

/// Trait for context storage backends.
///
/// MVP implements InMemoryContextStore. Future backends:
/// - RedisContextStore (behind `redis` feature flag)
/// - FileContextStore (behind `file-store` feature flag)
/// - PostgresContextStore (behind `postgres` feature flag)
pub trait ContextStore: Send + Sync {
    /// Get a value from the context store
    fn get(&self, exec_id: &str, key: &str) -> Option<Value>;

    /// Set a value in the context store
    fn set(&self, exec_id: &str, key: &str, value: Value);

    /// Remove a value from the context store
    fn remove(&self, exec_id: &str, key: &str) -> Option<Value>;

    /// Get a snapshot of all values for an execution
    fn snapshot(&self, exec_id: &str) -> HashMap<String, Value>;

    /// Clear all values for an execution
    fn clear(&self, exec_id: &str);

    /// Check if a key exists
    fn contains(&self, exec_id: &str, key: &str) -> bool;
}

// =============================================================================
// In-Memory Context Store (MVP)
// =============================================================================

/// In-memory implementation of ContextStore for MVP.
///
/// Thread-safe using RwLock. Data is lost on process restart.
#[derive(Debug, Default)]
pub struct InMemoryContextStore {
    /// Nested map: exec_id -> (key -> value)
    data: RwLock<HashMap<String, HashMap<String, Value>>>,
}

impl InMemoryContextStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    /// Get total number of execution contexts stored
    pub fn execution_count(&self) -> usize {
        self.data.read().unwrap().len()
    }

    /// Get total number of keys across all executions
    pub fn total_keys(&self) -> usize {
        self.data.read().unwrap().values().map(|m| m.len()).sum()
    }
}

impl ContextStore for InMemoryContextStore {
    fn get(&self, exec_id: &str, key: &str) -> Option<Value> {
        self.data
            .read()
            .unwrap()
            .get(exec_id)
            .and_then(|ctx| ctx.get(key).cloned())
    }

    fn set(&self, exec_id: &str, key: &str, value: Value) {
        let mut data = self.data.write().unwrap();
        data.entry(exec_id.to_string())
            .or_default()
            .insert(key.to_string(), value);
    }

    fn remove(&self, exec_id: &str, key: &str) -> Option<Value> {
        let mut data = self.data.write().unwrap();
        data.get_mut(exec_id).and_then(|ctx| ctx.remove(key))
    }

    fn snapshot(&self, exec_id: &str) -> HashMap<String, Value> {
        self.data
            .read()
            .unwrap()
            .get(exec_id)
            .cloned()
            .unwrap_or_default()
    }

    fn clear(&self, exec_id: &str) {
        let mut data = self.data.write().unwrap();
        data.remove(exec_id);
    }

    fn contains(&self, exec_id: &str, key: &str) -> bool {
        self.data
            .read()
            .unwrap()
            .get(exec_id)
            .map(|ctx| ctx.contains_key(key))
            .unwrap_or(false)
    }
}

// =============================================================================
// ExecutionContext
// =============================================================================

/// Context for a single DAG execution.
///
/// Holds variables that can be referenced in prompt templates via {{context.KEY}}.
/// Created per-request and lives for the duration of a single DAG execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Unique execution identifier
    pub execution_id: String,

    /// Context variables accessible via {{context.KEY}}
    #[serde(default)]
    pub variables: HashMap<String, Value>,

    /// Metadata about the execution (not accessible in templates)
    #[serde(default)]
    pub metadata: ContextMetadata,
}

/// Metadata about the execution context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// When the context was created (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// Source of the request (for auditing)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// User identifier (if authenticated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,

    /// Session identifier (for multi-turn conversations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Custom tags for filtering/grouping
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
}

impl ExecutionContext {
    /// Create a new execution context with the given ID
    pub fn new(execution_id: impl Into<String>) -> Self {
        Self {
            execution_id: execution_id.into(),
            variables: HashMap::new(),
            metadata: ContextMetadata::default(),
        }
    }

    /// Create a context with pre-populated variables
    pub fn with_variables(execution_id: impl Into<String>, variables: HashMap<String, Value>) -> Self {
        Self {
            execution_id: execution_id.into(),
            variables,
            metadata: ContextMetadata::default(),
        }
    }

    /// Get a variable value
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.variables.get(key)
    }

    /// Get a variable as a string (with automatic conversion)
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.variables.get(key).map(|v| match v {
            Value::String(s) => s.clone(),
            Value::Null => "null".to_string(),
            other => other.to_string(),
        })
    }

    /// Get a nested value using dot notation (e.g., "user.name")
    #[instrument(skip(self))]
    pub fn get_path(&self, path: &[String]) -> Option<&Value> {
        if path.is_empty() {
            return None;
        }

        let mut current = self.variables.get(&path[0])?;

        for key in &path[1..] {
            match current {
                Value::Object(map) => {
                    current = map.get(key)?;
                }
                _ => return None,
            }
        }

        Some(current)
    }

    /// Set a variable value
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.variables.insert(key.into(), value.into());
    }

    /// Set a string variable
    pub fn set_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(key.into(), Value::String(value.into()));
    }

    /// Remove a variable
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.variables.remove(key)
    }

    /// Check if a variable exists
    pub fn contains(&self, key: &str) -> bool {
        self.variables.contains_key(key)
    }

    /// Get all variable keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.variables.keys()
    }

    /// Merge another context's variables into this one
    pub fn merge(&mut self, other: &ExecutionContext) {
        for (key, value) in &other.variables {
            self.variables.insert(key.clone(), value.clone());
        }
    }

    /// Create a snapshot of all variables
    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.variables.clone()
    }

    /// Persist context to a store
    pub fn persist_to(&self, store: &dyn ContextStore) {
        for (key, value) in &self.variables {
            store.set(&self.execution_id, key, value.clone());
        }
        info!(
            execution_id = %self.execution_id,
            variable_count = self.variables.len(),
            "Context persisted to store"
        );
    }

    /// Load context from a store
    pub fn load_from(execution_id: impl Into<String>, store: &dyn ContextStore) -> Self {
        let exec_id = execution_id.into();
        let variables = store.snapshot(&exec_id);
        info!(
            execution_id = %exec_id,
            variable_count = variables.len(),
            "Context loaded from store"
        );
        Self {
            execution_id: exec_id,
            variables,
            metadata: ContextMetadata::default(),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new(uuid::Uuid::new_v4().to_string())
    }
}

// =============================================================================
// Context Manager
// =============================================================================

/// Manages multiple execution contexts with a pluggable store backend.
pub struct ContextManager {
    store: Arc<dyn ContextStore>,
}

impl ContextManager {
    /// Create a new context manager with the given store
    pub fn new(store: Arc<dyn ContextStore>) -> Self {
        Self { store }
    }

    /// Create a context manager with in-memory storage (MVP default)
    pub fn in_memory() -> Self {
        Self {
            store: Arc::new(InMemoryContextStore::new()),
        }
    }

    /// Get the underlying store
    pub fn store(&self) -> &dyn ContextStore {
        self.store.as_ref()
    }

    /// Create and persist a new context
    pub fn create_context(&self, execution_id: impl Into<String>) -> ExecutionContext {
        let ctx = ExecutionContext::new(execution_id);
        ctx.persist_to(self.store.as_ref());
        ctx
    }

    /// Load an existing context
    pub fn load_context(&self, execution_id: impl Into<String>) -> ExecutionContext {
        ExecutionContext::load_from(execution_id, self.store.as_ref())
    }

    /// Save a context to the store
    pub fn save_context(&self, context: &ExecutionContext) {
        context.persist_to(self.store.as_ref());
    }

    /// Clear a context from the store
    pub fn clear_context(&self, execution_id: &str) {
        self.store.clear(execution_id);
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::in_memory()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context_basic() {
        let mut ctx = ExecutionContext::new("test-123");

        ctx.set_string("user_name", "Alice");
        ctx.set("count", 42);
        ctx.set("active", true);

        assert_eq!(ctx.get_string("user_name"), Some("Alice".to_string()));
        assert_eq!(ctx.get("count"), Some(&Value::from(42)));
        assert_eq!(ctx.get("active"), Some(&Value::Bool(true)));
        assert!(ctx.contains("user_name"));
        assert!(!ctx.contains("nonexistent"));
    }

    #[test]
    fn test_execution_context_nested_path() {
        let mut ctx = ExecutionContext::new("test-123");

        let user = serde_json::json!({
            "name": "Alice",
            "profile": {
                "age": 30,
                "city": "Seattle"
            }
        });

        ctx.set("user", user);

        let path_name = vec!["user".to_string(), "name".to_string()];
        assert_eq!(
            ctx.get_path(&path_name),
            Some(&Value::String("Alice".to_string()))
        );

        let path_city = vec!["user".to_string(), "profile".to_string(), "city".to_string()];
        assert_eq!(
            ctx.get_path(&path_city),
            Some(&Value::String("Seattle".to_string()))
        );

        let path_invalid = vec!["user".to_string(), "nonexistent".to_string()];
        assert_eq!(ctx.get_path(&path_invalid), None);
    }

    #[test]
    fn test_in_memory_context_store() {
        let store = InMemoryContextStore::new();

        store.set("exec-1", "key1", Value::String("value1".to_string()));
        store.set("exec-1", "key2", Value::from(42));
        store.set("exec-2", "key1", Value::String("other".to_string()));

        assert_eq!(
            store.get("exec-1", "key1"),
            Some(Value::String("value1".to_string()))
        );
        assert_eq!(store.get("exec-1", "key2"), Some(Value::from(42)));
        assert_eq!(
            store.get("exec-2", "key1"),
            Some(Value::String("other".to_string()))
        );
        assert_eq!(store.get("exec-1", "nonexistent"), None);

        assert!(store.contains("exec-1", "key1"));
        assert!(!store.contains("exec-1", "nonexistent"));

        let snapshot = store.snapshot("exec-1");
        assert_eq!(snapshot.len(), 2);

        store.clear("exec-1");
        assert_eq!(store.snapshot("exec-1").len(), 0);
        assert_eq!(store.snapshot("exec-2").len(), 1);
    }

    #[test]
    fn test_context_manager() {
        let manager = ContextManager::in_memory();

        let mut ctx = manager.create_context("exec-123");
        ctx.set_string("greeting", "Hello");
        manager.save_context(&ctx);

        let loaded = manager.load_context("exec-123");
        assert_eq!(loaded.get_string("greeting"), Some("Hello".to_string()));

        manager.clear_context("exec-123");
        let cleared = manager.load_context("exec-123");
        assert!(cleared.variables.is_empty());
    }

    #[test]
    fn test_context_with_variables() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), Value::String("Bob".to_string()));
        vars.insert("age".to_string(), Value::from(25));

        let ctx = ExecutionContext::with_variables("test", vars);

        assert_eq!(ctx.get_string("name"), Some("Bob".to_string()));
        assert_eq!(ctx.get("age"), Some(&Value::from(25)));
    }

    #[test]
    fn test_context_merge() {
        let mut ctx1 = ExecutionContext::new("test-1");
        ctx1.set_string("a", "value_a");
        ctx1.set_string("b", "value_b");

        let mut ctx2 = ExecutionContext::new("test-2");
        ctx2.set_string("b", "overwritten");
        ctx2.set_string("c", "value_c");

        ctx1.merge(&ctx2);

        assert_eq!(ctx1.get_string("a"), Some("value_a".to_string()));
        assert_eq!(ctx1.get_string("b"), Some("overwritten".to_string()));
        assert_eq!(ctx1.get_string("c"), Some("value_c".to_string()));
    }

    #[test]
    fn test_context_serialization() {
        let mut ctx = ExecutionContext::new("test-123");
        ctx.set_string("name", "Alice");
        ctx.set("data", serde_json::json!({"key": "value"}));
        ctx.metadata.source = Some("test".to_string());

        let json = serde_json::to_string(&ctx).unwrap();
        let parsed: ExecutionContext = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.execution_id, "test-123");
        assert_eq!(parsed.get_string("name"), Some("Alice".to_string()));
        assert_eq!(parsed.metadata.source, Some("test".to_string()));
    }
}
