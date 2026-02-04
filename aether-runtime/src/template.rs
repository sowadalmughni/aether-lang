//! Prompt Template Engine for Aether runtime
//!
//! Handles substitution of template placeholders in prompts:
//! - {{context.KEY}} - Values from ExecutionContext
//! - {{node.NODE_ID.output}} - Outputs from previously executed nodes
//! - {{param_name}} - Direct parameter references (legacy)
//!
//! Designed for deterministic rendering to support reproducibility and caching.

use crate::context::ExecutionContext;
use aether_core::{DagNode, TemplateRef, TemplateRefKind, RenderPolicy, Sensitivity};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, instrument};

// =============================================================================
// Template Error Types
// =============================================================================

/// Errors that can occur during template rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateError {
    /// A required template reference was not found
    MissingRequired {
        placeholder: String,
        kind: String,
        path: Vec<String>,
    },
    /// A node output reference points to a non-existent node
    MissingNodeOutput {
        placeholder: String,
        node_id: String,
    },
    /// A context key is not in the allowed list
    ContextKeyNotAllowed {
        key: String,
        allowed: Vec<String>,
    },
    /// Value exceeded maximum length
    ValueTooLong {
        key: String,
        length: usize,
        max_length: usize,
    },
    /// Invalid template syntax
    InvalidSyntax {
        placeholder: String,
        reason: String,
    },
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::MissingRequired { placeholder, kind, path } => {
                write!(f, "Missing required {} reference '{}' (path: {:?})", kind, placeholder, path)
            }
            TemplateError::MissingNodeOutput { placeholder, node_id } => {
                write!(f, "Missing output from node '{}' for placeholder '{}'", node_id, placeholder)
            }
            TemplateError::ContextKeyNotAllowed { key, allowed } => {
                write!(f, "Context key '{}' not in allowed list: {:?}", key, allowed)
            }
            TemplateError::ValueTooLong { key, length, max_length } => {
                write!(f, "Value for '{}' exceeds max length ({} > {})", key, length, max_length)
            }
            TemplateError::InvalidSyntax { placeholder, reason } => {
                write!(f, "Invalid template syntax in '{}': {}", placeholder, reason)
            }
        }
    }
}

impl std::error::Error for TemplateError {}

// =============================================================================
// Render Result
// =============================================================================

/// Result of rendering a prompt template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResult {
    /// The fully rendered prompt
    pub rendered: String,
    /// Placeholders that were substituted
    pub substitutions: Vec<Substitution>,
    /// Warnings (non-fatal issues)
    pub warnings: Vec<String>,
    /// Whether any high-sensitivity values were substituted
    pub contains_sensitive: bool,
}

/// A single substitution that was made
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Substitution {
    pub placeholder: String,
    pub value_preview: String, // Truncated for logging
    pub sensitivity: Sensitivity,
    pub source: SubstitutionSource,
}

/// Source of a substitution value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubstitutionSource {
    Context { key: String },
    NodeOutput { node_id: String },
    Parameter { name: String },
    Constant { name: String },
    FoldedValue,
}

// =============================================================================
// Template Renderer
// =============================================================================

/// Options for template rendering
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// Whether to fail on missing optional references
    pub strict: bool,
    /// Maximum value length (0 = unlimited)
    pub max_value_length: usize,
    /// Whether to escape HTML in substituted values
    pub escape_html: bool,
    /// Whether to redact sensitive values in the result preview
    pub redact_sensitive: bool,
}

/// Render a prompt template with substitutions.
///
/// # Arguments
/// * `template` - The prompt template with {{...}} placeholders
/// * `template_refs` - Structured metadata about placeholders (from compiler)
/// * `context` - Execution context for {{context.KEY}} lookups
/// * `node_outputs` - Map of node_id -> output string for {{node.ID.output}}
/// * `policy` - Render policy with allowed keys and constraints
/// * `options` - Rendering options
///
/// # Returns
/// * `Ok(RenderResult)` - Successfully rendered prompt
/// * `Err(TemplateError)` - A required substitution failed
#[instrument(skip(template, context, node_outputs))]
pub fn render_prompt(
    template: &str,
    template_refs: &[TemplateRef],
    context: &ExecutionContext,
    node_outputs: &HashMap<String, String>,
    policy: &RenderPolicy,
    options: &RenderOptions,
) -> Result<RenderResult, TemplateError> {
    let mut rendered = template.to_string();
    let mut substitutions = Vec::new();
    let mut warnings = Vec::new();
    let mut contains_sensitive = false;

    // First pass: use structured template_refs from compiler
    for tref in template_refs {
        let value = resolve_template_ref(tref, context, node_outputs, policy)?;

        if let Some(mut val) = value {
            // Track sensitivity
            if matches!(tref.sensitivity, Sensitivity::High) {
                contains_sensitive = true;
            }

            // Apply length limit
            let max_len = if options.max_value_length > 0 {
                options.max_value_length
            } else if policy.max_value_length > 0 {
                policy.max_value_length
            } else {
                0
            };

            if max_len > 0 && val.len() > max_len {
                if tref.required {
                    return Err(TemplateError::ValueTooLong {
                        key: tref.raw.clone(),
                        length: val.len(),
                        max_length: max_len,
                    });
                }
                val.truncate(max_len);
                warnings.push(format!("Truncated '{}' to {} chars", tref.raw, max_len));
            }

            // Apply HTML escaping if needed
            if options.escape_html || policy.escape_html {
                val = escape_html(&val);
            }

            // Perform substitution
            rendered = rendered.replace(&tref.raw, &val);

            // Record substitution
            let value_preview = if options.redact_sensitive && matches!(tref.sensitivity, Sensitivity::High | Sensitivity::Medium) {
                "[REDACTED]".to_string()
            } else {
                truncate_preview(&val, 50)
            };

            substitutions.push(Substitution {
                placeholder: tref.raw.clone(),
                value_preview,
                sensitivity: tref.sensitivity.clone(),
                source: substitution_source_from_ref(tref),
            });
        } else if tref.required {
            return Err(TemplateError::MissingRequired {
                placeholder: tref.raw.clone(),
                kind: format!("{:?}", tref.kind),
                path: tref.path.clone(),
            });
        } else {
            // Optional reference not found - leave placeholder or replace with empty
            warnings.push(format!("Optional placeholder '{}' not resolved", tref.raw));
        }
    }

    // Second pass: handle any remaining {{...}} placeholders not in template_refs
    // This supports legacy prompts and simple parameter references
    rendered = render_legacy_placeholders(&rendered, node_outputs, &mut substitutions, &mut warnings);

    info!(
        substitution_count = substitutions.len(),
        warning_count = warnings.len(),
        contains_sensitive,
        rendered_length = rendered.len(),
        "Template rendered successfully"
    );

    Ok(RenderResult {
        rendered,
        substitutions,
        warnings,
        contains_sensitive,
    })
}

/// Resolve a single template reference to its value
fn resolve_template_ref(
    tref: &TemplateRef,
    context: &ExecutionContext,
    node_outputs: &HashMap<String, String>,
    policy: &RenderPolicy,
) -> Result<Option<String>, TemplateError> {
    // Check for compile-time folded value first
    if let Some(folded) = &tref.folded_value {
        return Ok(Some(folded.clone()));
    }

    match &tref.kind {
        TemplateRefKind::Context => {
            // Check allowed keys if policy is set
            if !policy.allowed_context_keys.is_empty() {
                let key = tref.path.first().map(|s| s.as_str()).unwrap_or("");
                if !policy.allowed_context_keys.iter().any(|k| k == key) {
                    return Err(TemplateError::ContextKeyNotAllowed {
                        key: key.to_string(),
                        allowed: policy.allowed_context_keys.clone(),
                    });
                }
            }

            // Resolve nested path in context
            let value = context.get_path(&tref.path);
            Ok(value.map(value_to_string))
        }

        TemplateRefKind::NodeOutput => {
            if let Some(node_id) = &tref.node_id {
                if let Some(output) = node_outputs.get(node_id) {
                    Ok(Some(output.clone()))
                } else if tref.required {
                    Err(TemplateError::MissingNodeOutput {
                        placeholder: tref.raw.clone(),
                        node_id: node_id.clone(),
                    })
                } else {
                    Ok(None)
                }
            } else {
                Err(TemplateError::InvalidSyntax {
                    placeholder: tref.raw.clone(),
                    reason: "NodeOutput reference missing node_id".to_string(),
                })
            }
        }

        TemplateRefKind::Parameter => {
            // Parameters are typically passed as node_outputs with the parameter name
            let param_name = tref.path.first().cloned().unwrap_or_else(|| tref.raw.trim_matches(|c| c == '{' || c == '}').to_string());
            Ok(node_outputs.get(&param_name).cloned())
        }

        TemplateRefKind::Constant => {
            // Constants should be folded at compile time
            // If we reach here, the constant wasn't resolved
            warn!(placeholder = %tref.raw, "Constant not folded at compile time");
            Ok(None)
        }

        TemplateRefKind::Variable => {
            // Variables are local to execution
            let var_name = tref.path.first().cloned().unwrap_or_else(|| tref.raw.trim_matches(|c| c == '{' || c == '}').to_string());
            Ok(node_outputs.get(&var_name).cloned())
        }
    }
}

/// Handle legacy {{placeholder}} syntax not covered by template_refs
fn render_legacy_placeholders(
    template: &str,
    node_outputs: &HashMap<String, String>,
    substitutions: &mut Vec<Substitution>,
    warnings: &mut Vec<String>,
) -> String {
    // Regex to match {{...}} placeholders
    let re = Regex::new(r"\{\{([^}]+)\}\}").unwrap();
    
    let mut result = template.to_string();
    
    for cap in re.captures_iter(template) {
        let full_match = cap.get(0).unwrap().as_str();
        let inner = cap.get(1).unwrap().as_str().trim();
        
        // Skip if already substituted (placeholder no longer exists)
        if !result.contains(full_match) {
            continue;
        }
        
        // Try to resolve from node_outputs using the inner content
        if let Some(value) = node_outputs.get(inner) {
            result = result.replace(full_match, value);
            substitutions.push(Substitution {
                placeholder: full_match.to_string(),
                value_preview: truncate_preview(value, 50),
                sensitivity: Sensitivity::Low,
                source: SubstitutionSource::Parameter { name: inner.to_string() },
            });
        } else {
            warnings.push(format!("Unresolved placeholder: {}", full_match));
        }
    }
    
    result
}

/// Convert a serde_json::Value to a string
fn value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "".to_string(),
        other => other.to_string(),
    }
}

/// Create a SubstitutionSource from a TemplateRef
fn substitution_source_from_ref(tref: &TemplateRef) -> SubstitutionSource {
    match &tref.kind {
        TemplateRefKind::Context => SubstitutionSource::Context {
            key: tref.path.join("."),
        },
        TemplateRefKind::NodeOutput => SubstitutionSource::NodeOutput {
            node_id: tref.node_id.clone().unwrap_or_default(),
        },
        TemplateRefKind::Parameter => SubstitutionSource::Parameter {
            name: tref.path.first().cloned().unwrap_or_default(),
        },
        TemplateRefKind::Constant => SubstitutionSource::Constant {
            name: tref.path.first().cloned().unwrap_or_default(),
        },
        TemplateRefKind::Variable => SubstitutionSource::Parameter {
            name: tref.path.first().cloned().unwrap_or_default(),
        },
    }
}

/// Escape HTML special characters
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Truncate a string for preview/logging
fn truncate_preview(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Simple render without structured template_refs (for legacy prompts)
pub fn render_simple(
    template: &str,
    context: &ExecutionContext,
    node_outputs: &HashMap<String, String>,
) -> Result<String, TemplateError> {
    let result = render_prompt(
        template,
        &[],
        context,
        node_outputs,
        &RenderPolicy::default(),
        &RenderOptions::default(),
    )?;
    Ok(result.rendered)
}

/// Render a DagNode's prompt template
pub fn render_node_prompt(
    node: &DagNode,
    context: &ExecutionContext,
    node_outputs: &HashMap<String, String>,
) -> Result<RenderResult, TemplateError> {
    let template = node.prompt_template.as_ref()
        .or(node.prompt.as_ref())
        .map(|s| s.as_str())
        .unwrap_or("");

    render_prompt(
        template,
        &node.template_refs,
        context,
        node_outputs,
        &node.render_policy,
        &RenderOptions::default(),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_context_placeholder() {
        let template = "Hello, {{context.user_name}}! How are you?";
        let mut ctx = ExecutionContext::new("test");
        ctx.set_string("user_name", "Alice");

        let template_refs = vec![TemplateRef {
            raw: "{{context.user_name}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["user_name".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(result.rendered, "Hello, Alice! How are you?");
        assert_eq!(result.substitutions.len(), 1);
    }

    #[test]
    fn test_render_node_output_placeholder() {
        let template = "Based on the summary: {{node.summarize.output}}, please analyze.";
        let ctx = ExecutionContext::new("test");

        let mut outputs = HashMap::new();
        outputs.insert("summarize".to_string(), "This is a summary of the document.".to_string());

        let template_refs = vec![TemplateRef {
            raw: "{{node.summarize.output}}".to_string(),
            kind: TemplateRefKind::NodeOutput,
            path: vec![],
            node_id: Some("summarize".to_string()),
            field: Some("output".to_string()),
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &outputs,
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(
            result.rendered,
            "Based on the summary: This is a summary of the document., please analyze."
        );
    }

    #[test]
    fn test_render_missing_required() {
        let template = "Hello, {{context.missing}}!";
        let ctx = ExecutionContext::new("test");

        let template_refs = vec![TemplateRef {
            raw: "{{context.missing}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["missing".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions::default(),
        );

        assert!(matches!(result, Err(TemplateError::MissingRequired { .. })));
    }

    #[test]
    fn test_render_optional_missing() {
        let template = "Hello{{context.suffix}}!";
        let ctx = ExecutionContext::new("test");

        let template_refs = vec![TemplateRef {
            raw: "{{context.suffix}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["suffix".to_string()],
            node_id: None,
            field: None,
            required: false,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        // Placeholder stays if optional and missing
        assert!(result.warnings.len() > 0);
    }

    #[test]
    fn test_render_nested_context() {
        let template = "City: {{context.user.profile.city}}";
        let mut ctx = ExecutionContext::new("test");
        ctx.set("user", serde_json::json!({
            "profile": {
                "city": "Seattle"
            }
        }));

        let template_refs = vec![TemplateRef {
            raw: "{{context.user.profile.city}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["user".to_string(), "profile".to_string(), "city".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(result.rendered, "City: Seattle");
    }

    #[test]
    fn test_render_context_key_not_allowed() {
        let template = "Secret: {{context.password}}";
        let mut ctx = ExecutionContext::new("test");
        ctx.set_string("password", "secret123");

        let template_refs = vec![TemplateRef {
            raw: "{{context.password}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["password".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::High,
            folded_value: None,
            provenance: None,
        }];

        let policy = RenderPolicy {
            allowed_context_keys: vec!["username".to_string(), "email".to_string()],
            ..Default::default()
        };

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &policy,
            &RenderOptions::default(),
        );

        assert!(matches!(result, Err(TemplateError::ContextKeyNotAllowed { .. })));
    }

    #[test]
    fn test_render_html_escaping() {
        let template = "User input: {{context.input}}";
        let mut ctx = ExecutionContext::new("test");
        ctx.set_string("input", "<script>alert('xss')</script>");

        let template_refs = vec![TemplateRef {
            raw: "{{context.input}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["input".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        }];

        let options = RenderOptions {
            escape_html: true,
            ..Default::default()
        };

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &options,
        ).unwrap();

        assert_eq!(
            result.rendered,
            "User input: &lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        );
    }

    #[test]
    fn test_render_legacy_placeholder() {
        let template = "Analyze: {{document}}";
        let ctx = ExecutionContext::new("test");

        let mut outputs = HashMap::new();
        outputs.insert("document".to_string(), "The document content".to_string());

        // No template_refs - uses legacy resolution
        let result = render_prompt(
            template,
            &[],
            &ctx,
            &outputs,
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(result.rendered, "Analyze: The document content");
    }

    #[test]
    fn test_render_folded_value() {
        let template = "Version: {{const.VERSION}}";
        let ctx = ExecutionContext::new("test");

        let template_refs = vec![TemplateRef {
            raw: "{{const.VERSION}}".to_string(),
            kind: TemplateRefKind::Constant,
            path: vec!["VERSION".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: Some("2.0.0".to_string()), // Folded at compile time
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(result.rendered, "Version: 2.0.0");
    }

    #[test]
    fn test_render_multiple_placeholders() {
        let template = "Dear {{context.recipient}}, regarding {{node.topic.output}}: {{context.message}}";
        let mut ctx = ExecutionContext::new("test");
        ctx.set_string("recipient", "Bob");
        ctx.set_string("message", "Please review.");

        let mut outputs = HashMap::new();
        outputs.insert("topic".to_string(), "Q3 Report".to_string());

        let template_refs = vec![
            TemplateRef {
                raw: "{{context.recipient}}".to_string(),
                kind: TemplateRefKind::Context,
                path: vec!["recipient".to_string()],
                node_id: None,
                field: None,
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: None,
                provenance: None,
            },
            TemplateRef {
                raw: "{{node.topic.output}}".to_string(),
                kind: TemplateRefKind::NodeOutput,
                path: vec![],
                node_id: Some("topic".to_string()),
                field: Some("output".to_string()),
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: None,
                provenance: None,
            },
            TemplateRef {
                raw: "{{context.message}}".to_string(),
                kind: TemplateRefKind::Context,
                path: vec!["message".to_string()],
                node_id: None,
                field: None,
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: None,
                provenance: None,
            },
        ];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &outputs,
            &RenderPolicy::default(),
            &RenderOptions::default(),
        ).unwrap();

        assert_eq!(result.rendered, "Dear Bob, regarding Q3 Report: Please review.");
        assert_eq!(result.substitutions.len(), 3);
    }

    #[test]
    fn test_render_simple() {
        let template = "Hello, {{name}}!";
        let ctx = ExecutionContext::new("test");
        let mut outputs = HashMap::new();
        outputs.insert("name".to_string(), "World".to_string());

        let result = render_simple(template, &ctx, &outputs).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_sensitivity_tracking() {
        let template = "Password: {{context.password}}";
        let mut ctx = ExecutionContext::new("test");
        ctx.set_string("password", "secret123");

        let template_refs = vec![TemplateRef {
            raw: "{{context.password}}".to_string(),
            kind: TemplateRefKind::Context,
            path: vec!["password".to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::High,
            folded_value: None,
            provenance: None,
        }];

        let result = render_prompt(
            template,
            &template_refs,
            &ctx,
            &HashMap::new(),
            &RenderPolicy::default(),
            &RenderOptions { redact_sensitive: true, ..Default::default() },
        ).unwrap();

        assert!(result.contains_sensitive);
        assert_eq!(result.substitutions[0].value_preview, "[REDACTED]");
    }
}
