//! Code Generation for Aether
//!
//! This module converts the analyzed AST into DAG JSON that the runtime can execute.
//! Key responsibilities:
//! - Convert flow definitions to DAG structures
//! - Extract LLM function calls as DagNodes
//! - Parse template placeholders and emit TemplateRef metadata
//! - Compute dependencies from data flow order
//! - Preserve type information for runtime validation

use crate::ast::{StringTemplate, TemplatePart};
use crate::semantic::{CallArg, FlowCall, LlmFnInfo, SemanticContext, SymbolKind};
use aether_core::{
    Dag, DagInput, DagMetadata, DagNode, DagNodeType, Provenance, Sensitivity, SourceLocation,
    TemplateRef, TemplateRefKind,
};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Compiler Configuration
// =============================================================================

/// Configuration for the code generator.
#[derive(Debug, Clone)]
pub struct CodegenConfig {
    /// Source file path (for metadata)
    pub source_file: Option<String>,
    /// Whether to include source locations in output
    pub include_source_locations: bool,
    /// Whether to attempt compile-time folding of constants
    pub fold_constants: bool,
    /// Known constants for compile-time folding
    pub constants: HashMap<String, String>,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            source_file: None,
            include_source_locations: true,
            fold_constants: false,
            constants: HashMap::new(),
        }
    }
}

// =============================================================================
// Code Generator
// =============================================================================

/// The code generator.
pub struct Codegen {
    config: CodegenConfig,
}

impl Codegen {
    pub fn new(config: CodegenConfig) -> Self {
        Self { config }
    }

    /// Compile a flow from the semantic context to a DAG.
    pub fn compile_flow(&self, flow_name: &str, ctx: &SemanticContext) -> Result<Dag, String> {
        // Get flow info from symbol table
        let flow_symbol = ctx
            .symbols
            .lookup(flow_name)
            .ok_or_else(|| format!("Flow '{}' not found", flow_name))?;

        let (params, return_type) = match &flow_symbol.kind {
            SymbolKind::Flow {
                params,
                return_type,
            } => (params.clone(), return_type.clone()),
            _ => return Err(format!("'{}' is not a flow", flow_name)),
        };

        // Get calls made in this flow
        let calls = ctx
            .flow_calls
            .get(flow_name)
            .ok_or_else(|| format!("No call info for flow '{}'", flow_name))?;

        // Build variable-to-node mapping for dependency tracking
        let mut var_to_node: HashMap<String, String> = HashMap::new();

        // Add flow parameters as available variables
        for (param_name, _) in &params {
            var_to_node.insert(param_name.clone(), format!("input.{}", param_name));
        }

        // Generate nodes for each LLM call
        let mut nodes = Vec::new();
        let mut node_counter = 0;

        for call in calls {
            if !call.is_llm_fn {
                continue; // Skip non-LLM calls for now
            }

            let llm_info = ctx
                .llm_functions
                .get(&call.callee)
                .ok_or_else(|| format!("LLM function '{}' not found", call.callee))?;

            // Generate unique node ID
            let node_id = if let Some(binding) = &call.binding {
                binding.clone()
            } else {
                node_counter += 1;
                format!("{}_{}", call.callee, node_counter)
            };

            // Compute dependencies
            let dependencies = self.compute_dependencies(&call.args, &var_to_node);

            // Build the prompt template with refs
            let (prompt_template, template_refs) =
                self.build_prompt_template(llm_info, &call.args, &var_to_node)?;

            // Build the node
            let mut node = DagNode::llm_fn(&node_id)
                .name(&call.callee)
                .dependencies(dependencies)
                .template_refs(template_refs)
                .return_type(llm_info.return_type.to_string())
                .build();

            // Set prompt template
            node.prompt_template = Some(prompt_template);

            // Set model parameters
            node.model = llm_info.model.clone();
            node.temperature = llm_info.temperature;
            node.max_tokens = llm_info.max_tokens;

            // Set system prompt if present
            if let Some(sys) = &llm_info.system_prompt {
                node.system_prompt = Some(self.render_template_string(sys));
            }

            // Set source location if configured
            if self.config.include_source_locations {
                if let Some(file) = &self.config.source_file {
                    node.source_location = Some(SourceLocation {
                        file: file.clone(),
                        line: call.span.start as u32,
                        column: 0,
                    });
                }
            }

            // Update variable mapping
            if let Some(binding) = &call.binding {
                var_to_node.insert(binding.clone(), node_id.clone());
            }

            nodes.push(node);
        }

        // Build DAG with metadata
        let mut dag = Dag::with_nodes(nodes);
        dag.metadata = DagMetadata {
            flow_name: Some(flow_name.to_string()),
            source_file: self.config.source_file.clone(),
            compiler_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            compiled_at: Some(chrono::Utc::now().to_rfc3339()),
            inputs: params
                .iter()
                .map(|(name, ty)| DagInput {
                    name: name.clone(),
                    type_name: ty.to_string(),
                    required: true,
                    default_value: None,
                })
                .collect(),
            output_type: Some(return_type.to_string()),
            constants: self.config.constants.clone(),
        };

        Ok(dag)
    }

    /// Compile all flows in the semantic context.
    pub fn compile_all_flows(&self, ctx: &SemanticContext) -> Result<HashMap<String, Dag>, String> {
        let mut dags = HashMap::new();

        for (name, symbol) in ctx.symbols.global_symbols() {
            if matches!(symbol.kind, SymbolKind::Flow { .. }) {
                let dag = self.compile_flow(name, ctx)?;
                dags.insert(name.clone(), dag);
            }
        }

        Ok(dags)
    }

    /// Compute dependencies from call arguments.
    fn compute_dependencies(
        &self,
        args: &[CallArg],
        var_to_node: &HashMap<String, String>,
    ) -> Vec<String> {
        let mut deps = HashSet::new();

        for arg in args {
            match arg {
                CallArg::Variable(name) => {
                    if let Some(node_id) = var_to_node.get(name) {
                        // Only add as dependency if it's a node (not an input param)
                        if !node_id.starts_with("input.") {
                            deps.insert(node_id.clone());
                        }
                    }
                }
                CallArg::FieldAccess { object, .. } => {
                    if let Some(node_id) = var_to_node.get(object) {
                        if !node_id.starts_with("input.") {
                            deps.insert(node_id.clone());
                        }
                    }
                }
                CallArg::Literal(_) => {}
            }
        }

        deps.into_iter().collect()
    }

    /// Build prompt template with template refs.
    fn build_prompt_template(
        &self,
        llm_info: &LlmFnInfo,
        args: &[CallArg],
        var_to_node: &HashMap<String, String>,
    ) -> Result<(String, Vec<TemplateRef>), String> {
        // Get the prompt template
        let template = llm_info
            .prompt
            .as_ref()
            .or(llm_info.user_prompt.as_ref())
            .ok_or_else(|| format!("No prompt for LLM function '{}'", llm_info.name))?;

        // Build argument mapping: param name -> call arg
        let arg_map: HashMap<String, &CallArg> = llm_info
            .params
            .iter()
            .zip(args.iter())
            .map(|((param_name, _), arg)| (param_name.clone(), arg))
            .collect();

        // Render template and extract refs
        let mut rendered = String::new();
        let mut refs = Vec::new();

        for part in &template.parts {
            match part {
                TemplatePart::Literal(s) => {
                    rendered.push_str(s);
                }
                TemplatePart::Variable(var_name) => {
                    // Parse the variable reference
                    let template_ref = self.parse_template_var(var_name, &arg_map, var_to_node)?;
                    
                    // Check for compile-time folding
                    if self.config.fold_constants {
                        if let Some(value) = self.try_fold_constant(&template_ref) {
                            // Fold the constant but keep provenance
                            rendered.push_str(&value);
                            let mut folded_ref = template_ref;
                            folded_ref.folded_value = Some(value);
                            refs.push(folded_ref);
                            continue;
                        }
                    }

                    // Preserve placeholder as-is
                    rendered.push_str(&format!("{{{{{}}}}}", var_name));
                    refs.push(template_ref);
                }
            }
        }

        Ok((rendered, refs))
    }

    /// Parse a template variable reference.
    fn parse_template_var(
        &self,
        var_name: &str,
        arg_map: &HashMap<String, &CallArg>,
        var_to_node: &HashMap<String, String>,
    ) -> Result<TemplateRef, String> {
        let raw = format!("{{{{{}}}}}", var_name);

        // Check for special prefixes
        if var_name.starts_with("context.") {
            let path = var_name["context.".len()..]
                .split('.')
                .map(|s| s.to_string())
                .collect();
            return Ok(TemplateRef {
                raw,
                kind: TemplateRefKind::Context,
                path,
                node_id: None,
                field: None,
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: None,
                provenance: None,
            });
        }

        if var_name.starts_with("node.") {
            let parts: Vec<_> = var_name["node.".len()..].split('.').collect();
            if parts.len() >= 2 {
                return Ok(TemplateRef {
                    raw,
                    kind: TemplateRefKind::NodeOutput,
                    path: vec![],
                    node_id: Some(parts[0].to_string()),
                    field: Some(parts[1].to_string()),
                    required: true,
                    sensitivity: Sensitivity::Low,
                    folded_value: None,
                    provenance: None,
                });
            }
        }

        if var_name.starts_with("const.") {
            let const_name = var_name["const.".len()..].to_string();
            return Ok(TemplateRef {
                raw,
                kind: TemplateRefKind::Constant,
                path: vec![const_name],
                node_id: None,
                field: None,
                required: true,
                sensitivity: Sensitivity::Low,
                folded_value: self.config.constants.get(&var_name["const.".len()..]).cloned(),
                provenance: None,
            });
        }

        // Check if it's a function parameter
        if let Some(arg) = arg_map.get(var_name) {
            return Ok(self.arg_to_template_ref(var_name, arg, var_to_node));
        }

        // Check if it's a known variable
        if let Some(node_id) = var_to_node.get(var_name) {
            if node_id.starts_with("input.") {
                return Ok(TemplateRef {
                    raw,
                    kind: TemplateRefKind::Parameter,
                    path: vec![var_name.to_string()],
                    node_id: None,
                    field: None,
                    required: true,
                    sensitivity: Sensitivity::Low,
                    folded_value: None,
                    provenance: None,
                });
            } else {
                return Ok(TemplateRef {
                    raw,
                    kind: TemplateRefKind::NodeOutput,
                    path: vec![],
                    node_id: Some(node_id.clone()),
                    field: Some("output".to_string()),
                    required: true,
                    sensitivity: Sensitivity::Low,
                    folded_value: None,
                    provenance: None,
                });
            }
        }

        // Default to variable reference
        Ok(TemplateRef {
            raw,
            kind: TemplateRefKind::Variable,
            path: vec![var_name.to_string()],
            node_id: None,
            field: None,
            required: true,
            sensitivity: Sensitivity::Low,
            folded_value: None,
            provenance: None,
        })
    }

    /// Convert a call argument to a template ref.
    fn arg_to_template_ref(
        &self,
        param_name: &str,
        arg: &CallArg,
        var_to_node: &HashMap<String, String>,
    ) -> TemplateRef {
        let raw = format!("{{{{{}}}}}", param_name);

        match arg {
            CallArg::Variable(var_name) => {
                if let Some(node_id) = var_to_node.get(var_name) {
                    if node_id.starts_with("input.") {
                        TemplateRef {
                            raw,
                            kind: TemplateRefKind::Parameter,
                            path: vec![var_name.clone()],
                            node_id: None,
                            field: None,
                            required: true,
                            sensitivity: Sensitivity::Low,
                            folded_value: None,
                            provenance: Some(Provenance {
                                file: None,
                                line: None,
                                column: None,
                                original_expr: Some(var_name.clone()),
                            }),
                        }
                    } else {
                        TemplateRef {
                            raw,
                            kind: TemplateRefKind::NodeOutput,
                            path: vec![],
                            node_id: Some(node_id.clone()),
                            field: Some("output".to_string()),
                            required: true,
                            sensitivity: Sensitivity::Low,
                            folded_value: None,
                            provenance: Some(Provenance {
                                file: None,
                                line: None,
                                column: None,
                                original_expr: Some(var_name.clone()),
                            }),
                        }
                    }
                } else {
                    TemplateRef {
                        raw,
                        kind: TemplateRefKind::Variable,
                        path: vec![var_name.clone()],
                        node_id: None,
                        field: None,
                        required: true,
                        sensitivity: Sensitivity::Low,
                        folded_value: None,
                        provenance: None,
                    }
                }
            }
            CallArg::Literal(value) => {
                // Literals are immediately folded
                TemplateRef {
                    raw: raw.clone(),
                    kind: TemplateRefKind::Variable,
                    path: vec![param_name.to_string()],
                    node_id: None,
                    field: None,
                    required: true,
                    sensitivity: Sensitivity::Low,
                    folded_value: Some(value.clone()),
                    provenance: Some(Provenance {
                        file: None,
                        line: None,
                        column: None,
                        original_expr: Some(format!("literal: {}", value)),
                    }),
                }
            }
            CallArg::FieldAccess { object, field } => {
                if let Some(node_id) = var_to_node.get(object) {
                    TemplateRef {
                        raw,
                        kind: TemplateRefKind::NodeOutput,
                        path: vec![],
                        node_id: Some(node_id.clone()),
                        field: Some(field.clone()),
                        required: true,
                        sensitivity: Sensitivity::Low,
                        folded_value: None,
                        provenance: Some(Provenance {
                            file: None,
                            line: None,
                            column: None,
                            original_expr: Some(format!("{}.{}", object, field)),
                        }),
                    }
                } else {
                    TemplateRef {
                        raw,
                        kind: TemplateRefKind::Variable,
                        path: vec![format!("{}.{}", object, field)],
                        node_id: None,
                        field: None,
                        required: true,
                        sensitivity: Sensitivity::Low,
                        folded_value: None,
                        provenance: None,
                    }
                }
            }
        }
    }

    /// Try to fold a constant at compile time.
    fn try_fold_constant(&self, template_ref: &TemplateRef) -> Option<String> {
        if template_ref.kind == TemplateRefKind::Constant {
            if let Some(const_name) = template_ref.path.first() {
                return self.config.constants.get(const_name).cloned();
            }
        }
        None
    }

    /// Render a string template to a string (preserving placeholders).
    fn render_template_string(&self, template: &StringTemplate) -> String {
        let mut result = String::new();
        for part in &template.parts {
            match part {
                TemplatePart::Literal(s) => result.push_str(s),
                TemplatePart::Variable(v) => {
                    result.push_str(&format!("{{{{{}}}}}", v));
                }
            }
        }
        result
    }
}

impl Default for Codegen {
    fn default() -> Self {
        Self::new(CodegenConfig::default())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;
    use crate::semantic::SemanticAnalyzer;

    fn compile_flow(source: &str, flow_name: &str) -> Result<Dag, String> {
        let program = Parser::new(source)
            .map_err(|e| format!("{:?}", e))?
            .parse_program()
            .map_err(|e| format!("{:?}", e))?;

        let ctx = SemanticAnalyzer::new()
            .analyze(&program)
            .map_err(|errs| format!("{:?}", errs))?;

        Codegen::default().compile_flow(flow_name, &ctx)
    }

    #[test]
    fn test_simple_flow() {
        let source = r#"
            llm fn classify(text: string) -> string {
                model: "gpt-4o",
                prompt: "Classify sentiment: {{text}}"
            }

            flow analyze(input: string) -> string {
                let result = classify(input);
                return result;
            }
        "#;

        let dag = compile_flow(source, "analyze").unwrap();

        assert_eq!(dag.nodes.len(), 1);
        assert_eq!(dag.nodes[0].id, "result");
        assert_eq!(dag.nodes[0].name, Some("classify".to_string()));
        assert_eq!(dag.nodes[0].model, Some("gpt-4o".to_string()));
        assert!(dag.nodes[0].dependencies.is_empty());
        assert_eq!(dag.nodes[0].template_refs.len(), 1);
        assert_eq!(dag.nodes[0].template_refs[0].kind, TemplateRefKind::Parameter);
    }

    #[test]
    fn test_parallel_flow() {
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

        let dag = compile_flow(source, "analyze").unwrap();

        assert_eq!(dag.nodes.len(), 2);

        // Both nodes should have no dependencies (parallel)
        let summary_node = dag.nodes.iter().find(|n| n.id == "summary").unwrap();
        let entities_node = dag.nodes.iter().find(|n| n.id == "entities").unwrap();

        assert!(summary_node.dependencies.is_empty());
        assert!(entities_node.dependencies.is_empty());
    }

    #[test]
    fn test_chained_flow() {
        let source = r#"
            llm fn summarize(text: string) -> string {
                model: "gpt-4o",
                prompt: "Summarize: {{text}}"
            }

            llm fn analyze_summary(summary: string) -> string {
                model: "gpt-4o",
                prompt: "Analyze this summary: {{summary}}"
            }

            flow pipeline(doc: string) -> string {
                let summary = summarize(doc);
                let analysis = analyze_summary(summary);
                return analysis;
            }
        "#;

        let dag = compile_flow(source, "pipeline").unwrap();

        assert_eq!(dag.nodes.len(), 2);

        let summary_node = dag.nodes.iter().find(|n| n.id == "summary").unwrap();
        let analysis_node = dag.nodes.iter().find(|n| n.id == "analysis").unwrap();

        // summary has no dependencies
        assert!(summary_node.dependencies.is_empty());

        // analysis depends on summary
        assert_eq!(analysis_node.dependencies, vec!["summary"]);
    }

    #[test]
    fn test_dag_json_output() {
        let source = r#"
            llm fn greet(name: string) -> string {
                model: "gpt-4o",
                temperature: 0.7,
                prompt: "Say hello to {{name}}"
            }

            flow hello(user: string) -> string {
                let greeting = greet(user);
                return greeting;
            }
        "#;

        let dag = compile_flow(source, "hello").unwrap();
        let json = dag.to_json().unwrap();

        // Verify JSON structure
        assert!(json.contains("\"schema_version\""));
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("\"nodes\""));
        assert!(json.contains("\"template_refs\""));
        assert!(json.contains("\"prompt_template\""));

        // Parse back
        let parsed: Dag = Dag::from_json(&json).unwrap();
        assert_eq!(parsed.nodes.len(), 1);
    }

    #[test]
    fn test_metadata() {
        let source = r#"
            llm fn echo(msg: string) -> string {
                model: "gpt-4o",
                prompt: "Echo: {{msg}}"
            }

            flow test_flow(input: string) -> string {
                let out = echo(input);
                return out;
            }
        "#;

        let dag = compile_flow(source, "test_flow").unwrap();

        assert_eq!(dag.metadata.flow_name, Some("test_flow".to_string()));
        assert_eq!(dag.metadata.inputs.len(), 1);
        assert_eq!(dag.metadata.inputs[0].name, "input");
        assert_eq!(dag.metadata.inputs[0].type_name, "string");
    }
}
