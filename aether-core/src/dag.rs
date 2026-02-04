//! DAG (Directed Acyclic Graph) types for Aether programs.
//!
//! These structures represent the compiled form of Aether programs. The compiler
//! emits DAG JSON that the runtime can execute. Template placeholders ({{...}})
//! are preserved with metadata for runtime substitution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Template Reference Types
// =============================================================================

/// The kind of template reference in a prompt template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemplateRefKind {
    /// Reference to a context variable: {{context.key}}
    Context,
    /// Reference to another node's output: {{node.id.output}}
    NodeOutput,
    /// Reference to a function parameter: {{param_name}}
    Parameter,
    /// Reference to a constant: {{const.NAME}}
    Constant,
    /// Reference to a local variable: {{variable}}
    Variable,
}

/// Sensitivity level for template references (for security policies).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    /// Low sensitivity - safe to log and cache
    Low,
    /// Medium sensitivity - cache but redact in logs
    Medium,
    /// High sensitivity - do not cache, redact in logs
    High,
}

impl Default for Sensitivity {
    fn default() -> Self {
        Sensitivity::Low
    }
}

/// A structured reference to a template placeholder.
///
/// Template references are extracted from prompt templates at compile time
/// and stored as metadata for runtime substitution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateRef {
    /// The raw placeholder string, e.g., "{{context.recipient_name}}"
    pub raw: String,

    /// The kind of reference
    pub kind: TemplateRefKind,

    /// Path components for nested access, e.g., ["recipient_name"] for context.recipient_name
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<String>,

    /// For node_output kind: the source node ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,

    /// For node_output kind: the field being accessed (usually "output")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,

    /// Whether this reference is required (error if missing)
    #[serde(default = "default_true")]
    pub required: bool,

    /// Sensitivity level for security policies
    #[serde(default)]
    pub sensitivity: Sensitivity,

    /// If the reference was folded at compile time, the resolved value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folded_value: Option<String>,

    /// Provenance info for debugging (source location, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

fn default_true() -> bool {
    true
}

/// Provenance information for debugging and cache transparency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Line number in source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Column number in source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
    /// Original expression before any transformations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_expr: Option<String>,
}

// =============================================================================
// Render Policy Types
// =============================================================================

/// Policy hints for runtime rendering of a node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderPolicy {
    /// Allowlist of context keys this node may access
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_context_keys: Vec<String>,

    /// Keys that should be redacted in logs
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redact_keys: Vec<String>,

    /// Whether to escape HTML/special characters in substituted values
    #[serde(default)]
    pub escape_html: bool,

    /// Maximum length for substituted values (0 = unlimited)
    #[serde(default)]
    pub max_value_length: usize,
}

// =============================================================================
// Execution Hints
// =============================================================================

/// Optional execution hints for runtime scheduling.
/// These are advisory - the runtime may ignore them based on conditions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionHints {
    /// Parallel group identifier (for future parallel {} blocks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_group: Option<String>,

    /// Maximum concurrent executions for this group
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<u32>,

    /// Error policy: "fail_fast", "collect", "partial"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_policy: Option<String>,

    /// Whether this node acts as a barrier (wait point)
    #[serde(default)]
    pub barrier: bool,

    /// Priority hint (higher = execute sooner when dependencies allow)
    #[serde(default)]
    pub priority: i32,

    /// Retry configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryConfig>,
}

/// Retry configuration for a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay in milliseconds
    pub base_delay_ms: u64,
    /// Backoff strategy: "constant", "linear", "exponential"
    pub strategy: String,
    /// Maximum delay in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_delay_ms: Option<u64>,
}

// =============================================================================
// DAG Node Types
// =============================================================================

/// The type of a DAG node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DagNodeType {
    /// An LLM function call
    LlmFn,
    /// A pure computation node
    Compute,
    /// A conditional branch
    Conditional,
    /// An input/parameter node
    Input,
    /// An output/return node
    Output,
}

/// A node in the DAG representing a unit of computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagNode {
    /// Unique identifier for this node
    pub id: String,

    /// The type of node
    pub node_type: DagNodeType,

    /// Human-readable name (usually the function name)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The prompt template with placeholders preserved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    /// Legacy field for backward compatibility (rendered prompt if no templates)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// Structured template references for runtime substitution
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub template_refs: Vec<TemplateRef>,

    /// LLM model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Temperature parameter for LLM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Maximum tokens for LLM response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// System prompt for LLM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// IDs of nodes this node depends on
    #[serde(default)]
    pub dependencies: Vec<String>,

    /// Explicit cache key inputs (if not derived from template_refs)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_key_inputs: Vec<String>,

    /// Render policy hints
    #[serde(default, skip_serializing_if = "is_default")]
    pub render_policy: RenderPolicy,

    /// Execution hints for scheduling
    #[serde(default, skip_serializing_if = "is_default")]
    pub execution_hints: ExecutionHints,

    /// Expected return type (for validation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,

    /// Source location for debugging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_location: Option<SourceLocation>,
}

fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    t == &T::default()
}

/// Source location information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

// =============================================================================
// DAG Structure
// =============================================================================

/// Metadata about the compiled DAG.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DagMetadata {
    /// Name of the flow this DAG was compiled from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_name: Option<String>,

    /// Source file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,

    /// Compiler version that produced this DAG
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiler_version: Option<String>,

    /// Compilation timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compiled_at: Option<String>,

    /// Input parameters expected by the flow
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<DagInput>,

    /// Output type of the flow
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_type: Option<String>,

    /// Constants that were available at compile time
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub constants: HashMap<String, String>,
}

/// An input parameter for the DAG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagInput {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(default)]
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<String>,
}

/// A compiled DAG representing an Aether flow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dag {
    /// Schema version for forward compatibility
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    /// Metadata about this DAG
    #[serde(default)]
    pub metadata: DagMetadata,

    /// The nodes in execution order (topologically sorted)
    pub nodes: Vec<DagNode>,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

impl Dag {
    /// Create a new empty DAG.
    pub fn new() -> Self {
        Self {
            schema_version: default_schema_version(),
            metadata: DagMetadata::default(),
            nodes: Vec::new(),
        }
    }

    /// Create a DAG with the given nodes.
    pub fn with_nodes(nodes: Vec<DagNode>) -> Self {
        Self {
            schema_version: default_schema_version(),
            metadata: DagMetadata::default(),
            nodes,
        }
    }

    /// Add a node to the DAG.
    pub fn add_node(&mut self, node: DagNode) {
        self.nodes.push(node);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&DagNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Validate that all dependencies exist.
    pub fn validate_dependencies(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let node_ids: std::collections::HashSet<_> = self.nodes.iter().map(|n| &n.id).collect();

        for node in &self.nodes {
            for dep in &node.dependencies {
                if !node_ids.contains(dep) {
                    errors.push(format!(
                        "Node '{}' depends on non-existent node '{}'",
                        node.id, dep
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Execution Result Types (for runtime)
// =============================================================================

/// State of a node during/after execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    /// Node has not started execution
    Pending,
    /// Node is currently executing
    Running,
    /// Node completed successfully
    Succeeded,
    /// Node failed with an error
    Failed,
    /// Node was skipped (due to dependency failure or abort)
    Skipped,
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState::Pending
    }
}

/// Status information for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub state: NodeState,
    /// Number of execution attempts (for retry tracking)
    #[serde(default)]
    pub attempts: u32,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Reason for skip if skipped
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self {
            state: NodeState::Pending,
            attempts: 0,
            error: None,
            skip_reason: None,
        }
    }
}

impl NodeStatus {
    pub fn succeeded() -> Self {
        Self {
            state: NodeState::Succeeded,
            attempts: 1,
            error: None,
            skip_reason: None,
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            state: NodeState::Failed,
            attempts: 1,
            error: Some(error.into()),
            skip_reason: None,
        }
    }

    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            state: NodeState::Skipped,
            attempts: 0,
            error: None,
            skip_reason: Some(reason.into()),
        }
    }

    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }
}

/// Result of executing a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeExecutionResult {
    pub node_id: String,
    pub output: String,
    pub execution_time_ms: u64,
    pub token_cost: u32,
    pub cache_hit: bool,
    /// The rendered prompt after substitution (for debugging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_prompt: Option<String>,
    /// Input tokens used (before cache)
    #[serde(default)]
    pub input_tokens: u32,
    /// Output tokens used (before cache)
    #[serde(default)]
    pub output_tokens: u32,
    /// Which level this node was in
    #[serde(default)]
    pub level: usize,
}

/// Error policy for handling failures during parallel execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorPolicy {
    /// Stop scheduling new nodes immediately on failure
    Fail,
    /// Continue executing independent nodes, skip dependents
    Skip,
    /// Retry failed nodes according to retry config
    Retry,
}

impl Default for ErrorPolicy {
    fn default() -> Self {
        ErrorPolicy::Fail
    }
}

impl ErrorPolicy {
    /// Parse from string (for DAG execution hints)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "skip" => ErrorPolicy::Skip,
            "retry" => ErrorPolicy::Retry,
            "fail" | "fail_fast" => ErrorPolicy::Fail,
            _ => ErrorPolicy::Fail,
        }
    }
}

/// Result of executing an entire DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagExecutionResponse {
    pub execution_id: String,
    pub results: Vec<NodeExecutionResult>,
    pub total_execution_time_ms: u64,
    pub total_token_cost: u32,
    pub parallelization_factor: f64,
    pub cache_hit_rate: f64,
    pub errors: Vec<String>,

    // === New fields for observability ===

    /// Execution time for each level (to prove parallel speedup)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub level_execution_times_ms: Vec<u64>,

    /// Maximum concurrency actually used during execution
    #[serde(default)]
    pub max_concurrency_used: u32,

    /// Mapping of node_id to execution time (for detailed analysis)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub node_execution_times_ms: HashMap<String, u64>,

    /// Mapping of node_id to level (for understanding DAG structure)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub node_levels: HashMap<String, usize>,

    /// Status of each node (state, attempts, errors)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub node_status: HashMap<String, NodeStatus>,

    /// Whether execution was aborted due to error
    #[serde(default)]
    pub aborted: bool,

    /// List of nodes that were skipped
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_nodes: Vec<String>,

    /// Total tokens saved by cache hits
    #[serde(default)]
    pub tokens_saved: u32,
}

// =============================================================================
// Builder Pattern for DagNode
// =============================================================================

impl DagNode {
    /// Create a new LLM function node.
    pub fn llm_fn(id: impl Into<String>) -> DagNodeBuilder {
        DagNodeBuilder {
            node: DagNode {
                id: id.into(),
                node_type: DagNodeType::LlmFn,
                name: None,
                prompt_template: None,
                prompt: None,
                template_refs: Vec::new(),
                model: None,
                temperature: None,
                max_tokens: None,
                system_prompt: None,
                dependencies: Vec::new(),
                cache_key_inputs: Vec::new(),
                render_policy: RenderPolicy::default(),
                execution_hints: ExecutionHints::default(),
                return_type: None,
                source_location: None,
            },
        }
    }

    /// Create a new compute node.
    pub fn compute(id: impl Into<String>) -> DagNodeBuilder {
        DagNodeBuilder {
            node: DagNode {
                id: id.into(),
                node_type: DagNodeType::Compute,
                name: None,
                prompt_template: None,
                prompt: None,
                template_refs: Vec::new(),
                model: None,
                temperature: None,
                max_tokens: None,
                system_prompt: None,
                dependencies: Vec::new(),
                cache_key_inputs: Vec::new(),
                render_policy: RenderPolicy::default(),
                execution_hints: ExecutionHints::default(),
                return_type: None,
                source_location: None,
            },
        }
    }
}

/// Builder for DagNode.
pub struct DagNodeBuilder {
    node: DagNode,
}

impl DagNodeBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.node.name = Some(name.into());
        self
    }

    pub fn prompt_template(mut self, template: impl Into<String>) -> Self {
        self.node.prompt_template = Some(template.into());
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.node.prompt = Some(prompt.into());
        self
    }

    pub fn template_ref(mut self, template_ref: TemplateRef) -> Self {
        self.node.template_refs.push(template_ref);
        self
    }

    pub fn template_refs(mut self, refs: Vec<TemplateRef>) -> Self {
        self.node.template_refs = refs;
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.node.model = Some(model.into());
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.node.temperature = Some(temp);
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.node.max_tokens = Some(tokens);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.node.system_prompt = Some(prompt.into());
        self
    }

    pub fn dependency(mut self, dep: impl Into<String>) -> Self {
        self.node.dependencies.push(dep.into());
        self
    }

    pub fn dependencies(mut self, deps: Vec<String>) -> Self {
        self.node.dependencies = deps;
        self
    }

    pub fn return_type(mut self, type_name: impl Into<String>) -> Self {
        self.node.return_type = Some(type_name.into());
        self
    }

    pub fn source_location(mut self, file: impl Into<String>, line: u32, column: u32) -> Self {
        self.node.source_location = Some(SourceLocation {
            file: file.into(),
            line,
            column,
        });
        self
    }

    pub fn build(self) -> DagNode {
        self.node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_serialization() {
        let dag = Dag::with_nodes(vec![
            DagNode::llm_fn("classify")
                .name("classify_sentiment")
                .model("gpt-4o")
                .temperature(0.1)
                .prompt_template("Classify the sentiment of: {{text}}")
                .template_ref(TemplateRef {
                    raw: "{{text}}".to_string(),
                    kind: TemplateRefKind::Parameter,
                    path: vec!["text".to_string()],
                    node_id: None,
                    field: None,
                    required: true,
                    sensitivity: Sensitivity::Low,
                    folded_value: None,
                    provenance: None,
                })
                .return_type("Sentiment")
                .build(),
        ]);

        let json = dag.to_json().unwrap();
        let parsed: Dag = Dag::from_json(&json).unwrap();
        assert_eq!(dag, parsed);
    }

    #[test]
    fn test_dag_validation() {
        let dag = Dag::with_nodes(vec![
            DagNode::llm_fn("a").build(),
            DagNode::llm_fn("b").dependency("a").build(),
            DagNode::llm_fn("c").dependency("nonexistent").build(),
        ]);

        let result = dag.validate_dependencies();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("nonexistent"));
    }

    #[test]
    fn test_template_ref_kinds() {
        let refs = vec![
            TemplateRef {
                raw: "{{context.user_name}}".to_string(),
                kind: TemplateRefKind::Context,
                path: vec!["user_name".to_string()],
                node_id: None,
                field: None,
                required: true,
                sensitivity: Sensitivity::Medium,
                folded_value: None,
                provenance: None,
            },
            TemplateRef {
                raw: "{{node.summarize.output}}".to_string(),
                kind: TemplateRefKind::NodeOutput,
                path: vec![],
                node_id: Some("summarize".to_string()),
                field: Some("output".to_string()),
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: None,
                provenance: None,
            },
        ];

        let node = DagNode::llm_fn("email")
            .prompt_template("Write to {{context.user_name}} about {{node.summarize.output}}")
            .template_refs(refs)
            .dependency("summarize")
            .build();

        assert_eq!(node.template_refs.len(), 2);
        assert_eq!(node.dependencies, vec!["summarize"]);
    }
}
