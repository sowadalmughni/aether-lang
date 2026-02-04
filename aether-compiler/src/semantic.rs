//! Semantic Analysis for Aether
//!
//! This module performs semantic analysis on the parsed AST, including:
//! - Symbol resolution: building symbol tables for types, functions, flows
//! - Type checking: verifying type correctness (basic for MVP)
//! - Dependency analysis: tracking data flow for DAG construction
//!
//! The semantic pass produces a `SemanticContext` that the codegen pass uses
//! to emit DAG JSON with correct dependency information.

use crate::ast::*;
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Semantic Errors
// =============================================================================

#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("Undefined symbol '{name}' at position {span:?}")]
    UndefinedSymbol { name: String, span: Span },

    #[error("Duplicate definition of '{name}' at position {span:?}")]
    DuplicateDefinition { name: String, span: Span },

    #[error("Type mismatch: expected {expected}, found {found} at position {span:?}")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("LLM function '{name}' requires a model specification at position {span:?}")]
    MissingModel { name: String, span: Span },

    #[error("LLM function '{name}' requires a prompt at position {span:?}")]
    MissingPrompt { name: String, span: Span },

    #[error("Invalid template reference '{reference}' at position {span:?}")]
    InvalidTemplateRef { reference: String, span: Span },

    #[error("Circular dependency detected involving '{name}'")]
    CircularDependency { name: String },

    #[error("Flow '{flow_name}' calls undefined function '{fn_name}' at position {span:?}")]
    UndefinedFunction {
        flow_name: String,
        fn_name: String,
        span: Span,
    },
}

pub type SemanticResult<T> = Result<T, Vec<SemanticError>>;

// =============================================================================
// Symbol Types
// =============================================================================

/// Kind of symbol in the symbol table.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    /// An LLM function definition
    LlmFn {
        params: Vec<(String, TypeInfo)>,
        return_type: TypeInfo,
    },
    /// A regular function definition
    Function {
        params: Vec<(String, TypeInfo)>,
        return_type: Option<TypeInfo>,
    },
    /// A flow definition
    Flow {
        params: Vec<(String, TypeInfo)>,
        return_type: TypeInfo,
    },
    /// A struct type
    Struct { fields: Vec<(String, TypeInfo)> },
    /// An enum type
    Enum { variants: Vec<EnumVariantInfo> },
    /// A context type
    Context { fields: Vec<(String, TypeInfo)> },
    /// A type alias
    TypeAlias { target: TypeInfo },
    /// A local variable
    Variable { ty: TypeInfo },
    /// A function parameter
    Parameter { ty: TypeInfo },
}

/// Information about an enum variant.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariantInfo {
    pub name: String,
    pub data: Option<Vec<TypeInfo>>,
}

/// Simplified type information for semantic analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeInfo {
    Named(String),
    Primitive(PrimitiveType),
    List(Box<TypeInfo>),
    Map(Box<TypeInfo>, Box<TypeInfo>),
    Optional(Box<TypeInfo>),
    Unit,
    Unknown,
}

impl TypeInfo {
    pub fn from_ast_type(ty: &Type) -> Self {
        match ty {
            Type::Named { name, .. } => TypeInfo::Named(name.clone()),
            Type::Primitive { kind, .. } => TypeInfo::Primitive(*kind),
            Type::List { element, .. } => {
                TypeInfo::List(Box::new(TypeInfo::from_ast_type(element)))
            }
            Type::Map { key, value, .. } => TypeInfo::Map(
                Box::new(TypeInfo::from_ast_type(key)),
                Box::new(TypeInfo::from_ast_type(value)),
            ),
            Type::Optional { inner, .. } => {
                TypeInfo::Optional(Box::new(TypeInfo::from_ast_type(inner)))
            }
            Type::Unit { .. } => TypeInfo::Unit,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            TypeInfo::Named(name) => name.clone(),
            TypeInfo::Primitive(p) => match p {
                PrimitiveType::String => "string".to_string(),
                PrimitiveType::Int => "int".to_string(),
                PrimitiveType::Float => "float".to_string(),
                PrimitiveType::Bool => "bool".to_string(),
            },
            TypeInfo::List(elem) => format!("list<{}>", elem.to_string()),
            TypeInfo::Map(k, v) => format!("map<{}, {}>", k.to_string(), v.to_string()),
            TypeInfo::Optional(inner) => format!("optional<{}>", inner.to_string()),
            TypeInfo::Unit => "()".to_string(),
            TypeInfo::Unknown => "unknown".to_string(),
        }
    }
}

/// A symbol table entry.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub span: Span,
}

// =============================================================================
// Symbol Table
// =============================================================================

/// Hierarchical symbol table with scopes.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Stack of scopes, with the current scope at the end
    scopes: Vec<HashMap<String, Symbol>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    /// Enter a new scope (e.g., function body).
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Leave the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a symbol in the current scope.
    pub fn define(&mut self, symbol: Symbol) -> Result<(), SemanticError> {
        let scope = self.scopes.last_mut().unwrap();
        if scope.contains_key(&symbol.name) {
            return Err(SemanticError::DuplicateDefinition {
                name: symbol.name.clone(),
                span: symbol.span.clone(),
            });
        }
        scope.insert(symbol.name.clone(), symbol);
        Ok(())
    }

    /// Look up a symbol by name, searching from innermost to outermost scope.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol);
            }
        }
        None
    }

    /// Get all symbols in the global (outermost) scope.
    pub fn global_symbols(&self) -> &HashMap<String, Symbol> {
        &self.scopes[0]
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Semantic Context
// =============================================================================

/// The result of semantic analysis, containing all resolved information.
#[derive(Debug, Clone)]
pub struct SemanticContext {
    /// Global symbol table
    pub symbols: SymbolTable,

    /// Map from LLM function name to its definition info
    pub llm_functions: HashMap<String, LlmFnInfo>,

    /// Map from flow name to its call graph
    pub flow_calls: HashMap<String, Vec<FlowCall>>,

    /// Collected errors (may be empty if analysis succeeded)
    pub errors: Vec<SemanticError>,
}

/// Information about an LLM function for codegen.
#[derive(Debug, Clone)]
pub struct LlmFnInfo {
    pub name: String,
    pub params: Vec<(String, TypeInfo)>,
    pub return_type: TypeInfo,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub system_prompt: Option<StringTemplate>,
    pub user_prompt: Option<StringTemplate>,
    pub prompt: Option<StringTemplate>,
    pub span: Span,
}

/// A call to an LLM function within a flow.
#[derive(Debug, Clone)]
pub struct FlowCall {
    /// The variable this call is assigned to
    pub binding: Option<String>,
    /// The function being called
    pub callee: String,
    /// Is this an LLM function?
    pub is_llm_fn: bool,
    /// Arguments passed (variable names or literals)
    pub args: Vec<CallArg>,
    /// Source span
    pub span: Span,
}

/// An argument to a function call.
#[derive(Debug, Clone)]
pub enum CallArg {
    /// A variable reference
    Variable(String),
    /// A literal value
    Literal(String),
    /// A field access (e.g., result.field)
    FieldAccess { object: String, field: String },
}

// =============================================================================
// Semantic Analyzer
// =============================================================================

/// The semantic analyzer.
pub struct SemanticAnalyzer {
    context: SemanticContext,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            context: SemanticContext {
                symbols: SymbolTable::new(),
                llm_functions: HashMap::new(),
                flow_calls: HashMap::new(),
                errors: Vec::new(),
            },
        }
    }

    /// Analyze a program and return the semantic context.
    pub fn analyze(mut self, program: &Program) -> SemanticResult<SemanticContext> {
        // First pass: collect all type definitions and function signatures
        for item in &program.items {
            if let Err(e) = self.collect_item(item) {
                self.context.errors.push(e);
            }
        }

        // Second pass: analyze function/flow bodies
        for item in &program.items {
            self.analyze_item(item);
        }

        if self.context.errors.is_empty() {
            Ok(self.context)
        } else {
            Err(self.context.errors)
        }
    }

    /// First pass: collect top-level definitions.
    fn collect_item(&mut self, item: &Item) -> Result<(), SemanticError> {
        match item {
            Item::LlmFn(def) => self.collect_llm_fn(def),
            Item::Function(def) => self.collect_function(def),
            Item::Flow(def) => self.collect_flow(def),
            Item::Struct(def) => self.collect_struct(def),
            Item::Enum(def) => self.collect_enum(def),
            Item::Context(def) => self.collect_context(def),
            Item::TypeAlias(def) => self.collect_type_alias(def),
            Item::Test(_) | Item::Import(_) => Ok(()), // Skip for MVP
        }
    }

    fn collect_llm_fn(&mut self, def: &LlmFnDef) -> Result<(), SemanticError> {
        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = TypeInfo::from_ast_type(&def.return_type);

        // Store LLM function info
        self.context.llm_functions.insert(
            def.name.node.clone(),
            LlmFnInfo {
                name: def.name.node.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                model: def.body.model.as_ref().map(|m| m.node.clone()),
                temperature: def.body.temperature.as_ref().map(|t| t.node),
                max_tokens: def.body.max_tokens.as_ref().map(|m| m.node),
                system_prompt: def.body.system_prompt.as_ref().map(|s| s.node.clone()),
                user_prompt: def.body.user_prompt.as_ref().map(|s| s.node.clone()),
                prompt: def.body.prompt.as_ref().map(|s| s.node.clone()),
                span: def.span.clone(),
            },
        );

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::LlmFn {
                params,
                return_type,
            },
            span: def.span.clone(),
        })
    }

    fn collect_function(&mut self, def: &FunctionDef) -> Result<(), SemanticError> {
        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = def.return_type.as_ref().map(TypeInfo::from_ast_type);

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Function {
                params,
                return_type,
            },
            span: def.span.clone(),
        })
    }

    fn collect_flow(&mut self, def: &FlowDef) -> Result<(), SemanticError> {
        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = TypeInfo::from_ast_type(&def.return_type);

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Flow {
                params,
                return_type,
            },
            span: def.span.clone(),
        })
    }

    fn collect_struct(&mut self, def: &StructDef) -> Result<(), SemanticError> {
        let fields: Vec<_> = def
            .fields
            .iter()
            .map(|f| (f.name.node.clone(), TypeInfo::from_ast_type(&f.ty)))
            .collect();

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Struct { fields },
            span: def.span.clone(),
        })
    }

    fn collect_enum(&mut self, def: &EnumDef) -> Result<(), SemanticError> {
        let variants: Vec<_> = def
            .variants
            .iter()
            .map(|v| EnumVariantInfo {
                name: v.name.node.clone(),
                data: v
                    .data
                    .as_ref()
                    .map(|types| types.iter().map(TypeInfo::from_ast_type).collect()),
            })
            .collect();

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Enum { variants },
            span: def.span.clone(),
        })
    }

    fn collect_context(&mut self, def: &ContextDef) -> Result<(), SemanticError> {
        let fields: Vec<_> = def
            .fields
            .iter()
            .map(|f| (f.name.node.clone(), TypeInfo::from_ast_type(&f.ty)))
            .collect();

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Context { fields },
            span: def.span.clone(),
        })
    }

    fn collect_type_alias(&mut self, def: &TypeAliasDef) -> Result<(), SemanticError> {
        let target = TypeInfo::from_ast_type(&def.ty);

        self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::TypeAlias { target },
            span: def.span.clone(),
        })
    }

    /// Second pass: analyze item bodies.
    fn analyze_item(&mut self, item: &Item) {
        match item {
            Item::LlmFn(def) => self.analyze_llm_fn(def),
            Item::Flow(def) => self.analyze_flow(def),
            _ => {} // Other items don't need deep analysis for MVP
        }
    }

    fn analyze_llm_fn(&mut self, def: &LlmFnDef) {
        // Check that model is specified
        if def.body.model.is_none() {
            self.context.errors.push(SemanticError::MissingModel {
                name: def.name.node.clone(),
                span: def.span.clone(),
            });
        }

        // Check that at least one prompt is specified
        if def.body.prompt.is_none()
            && def.body.user_prompt.is_none()
            && def.body.system_prompt.is_none()
        {
            self.context.errors.push(SemanticError::MissingPrompt {
                name: def.name.node.clone(),
                span: def.span.clone(),
            });
        }
    }

    fn analyze_flow(&mut self, def: &FlowDef) {
        let flow_name = def.name.node.clone();
        let mut calls = Vec::new();

        // Push scope for flow parameters and local variables
        self.context.symbols.push_scope();

        // Add parameters to scope
        for param in &def.params {
            let _ = self.context.symbols.define(Symbol {
                name: param.name.node.clone(),
                kind: SymbolKind::Parameter {
                    ty: TypeInfo::from_ast_type(&param.ty),
                },
                span: param.span.clone(),
            });
        }

        // Analyze flow body
        for stmt in &def.body.stmts {
            self.analyze_stmt_for_calls(&flow_name, stmt, &mut calls);
        }

        self.context.symbols.pop_scope();
        self.context.flow_calls.insert(flow_name, calls);
    }

    fn analyze_stmt_for_calls(&mut self, flow_name: &str, stmt: &Stmt, calls: &mut Vec<FlowCall>) {
        match stmt {
            Stmt::Let {
                name, value, span, ..
            } => {
                // Check if value is a function call
                if let Some(call) = self.extract_call(value, Some(name.node.clone()), span) {
                    // Verify the callee exists
                    if self.context.symbols.lookup(&call.callee).is_none() {
                        self.context.errors.push(SemanticError::UndefinedFunction {
                            flow_name: flow_name.to_string(),
                            fn_name: call.callee.clone(),
                            span: span.clone(),
                        });
                    }
                    calls.push(call);
                }

                // Add variable to scope
                let _ = self.context.symbols.define(Symbol {
                    name: name.node.clone(),
                    kind: SymbolKind::Variable {
                        ty: TypeInfo::Unknown, // Type inference not implemented for MVP
                    },
                    span: span.clone(),
                });
            }
            Stmt::Expr { expr, span } => {
                if let Some(call) = self.extract_call(expr, None, span) {
                    if self.context.symbols.lookup(&call.callee).is_none() {
                        self.context.errors.push(SemanticError::UndefinedFunction {
                            flow_name: flow_name.to_string(),
                            fn_name: call.callee.clone(),
                            span: span.clone(),
                        });
                    }
                    calls.push(call);
                }
            }
            Stmt::If {
                then_block,
                else_block,
                ..
            } => {
                for s in &then_block.stmts {
                    self.analyze_stmt_for_calls(flow_name, s, calls);
                }
                if let Some(else_b) = else_block {
                    for s in &else_b.stmts {
                        self.analyze_stmt_for_calls(flow_name, s, calls);
                    }
                }
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                for s in &body.stmts {
                    self.analyze_stmt_for_calls(flow_name, s, calls);
                }
            }
            Stmt::Try { body, catches, .. } => {
                for s in &body.stmts {
                    self.analyze_stmt_for_calls(flow_name, s, calls);
                }
                for catch in catches {
                    for s in &catch.body.stmts {
                        self.analyze_stmt_for_calls(flow_name, s, calls);
                    }
                }
            }
            _ => {}
        }
    }

    fn extract_call(
        &self,
        expr: &Expr,
        binding: Option<String>,
        span: &Span,
    ) -> Option<FlowCall> {
        match expr {
            Expr::Call { func, args, .. } => {
                // Get function name
                let callee = match func.as_ref() {
                    Expr::Ident { name, .. } => name.clone(),
                    _ => return None, // Complex call expressions not supported
                };

                // Check if this is an LLM function
                let is_llm_fn = self.context.llm_functions.contains_key(&callee);

                // Extract arguments
                let call_args: Vec<_> = args.iter().map(|a| self.extract_arg(a)).collect();

                Some(FlowCall {
                    binding,
                    callee,
                    is_llm_fn,
                    args: call_args,
                    span: span.clone(),
                })
            }
            Expr::Await { expr, .. } => {
                // Unwrap await and extract the inner call
                self.extract_call(expr, binding, span)
            }
            _ => None,
        }
    }

    fn extract_arg(&self, expr: &Expr) -> CallArg {
        match expr {
            Expr::Ident { name, .. } => CallArg::Variable(name.clone()),
            Expr::Literal { value, .. } => match value {
                Literal::String(s) => CallArg::Literal(s.clone()),
                Literal::Int(i) => CallArg::Literal(i.to_string()),
                Literal::Float(f) => CallArg::Literal(f.to_string()),
                Literal::Bool(b) => CallArg::Literal(b.to_string()),
            },
            Expr::FieldAccess { object, field, .. } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    CallArg::FieldAccess {
                        object: name.clone(),
                        field: field.node.clone(),
                    }
                } else {
                    CallArg::Literal("<complex>".to_string())
                }
            }
            _ => CallArg::Literal("<expr>".to_string()),
        }
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn analyze(source: &str) -> SemanticResult<SemanticContext> {
        let program = Parser::new(source).unwrap().parse_program().unwrap();
        SemanticAnalyzer::new().analyze(&program)
    }

    #[test]
    fn test_collect_llm_fn() {
        let source = r#"
            llm fn classify(text: string) -> Sentiment {
                model: "gpt-4o",
                prompt: "Classify: {{text}}"
            }
        "#;

        let ctx = analyze(source).unwrap();
        assert!(ctx.symbols.lookup("classify").is_some());
        assert!(ctx.llm_functions.contains_key("classify"));

        let info = &ctx.llm_functions["classify"];
        assert_eq!(info.model, Some("gpt-4o".to_string()));
    }

    #[test]
    fn test_collect_struct_and_enum() {
        let source = r#"
            struct Message { role: string, content: string }
            enum Sentiment { Positive, Negative }
        "#;

        let ctx = analyze(source).unwrap();
        assert!(ctx.symbols.lookup("Message").is_some());
        assert!(ctx.symbols.lookup("Sentiment").is_some());
    }

    #[test]
    fn test_flow_calls() {
        let source = r#"
            llm fn summarize(text: string) -> string {
                model: "gpt-4o",
                prompt: "Summarize: {{text}}"
            }

            flow analyze(doc: string) -> string {
                let summary = summarize(doc);
                return summary;
            }
        "#;

        let ctx = analyze(source).unwrap();
        assert!(ctx.flow_calls.contains_key("analyze"));

        let calls = &ctx.flow_calls["analyze"];
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].callee, "summarize");
        assert_eq!(calls[0].binding, Some("summary".to_string()));
        assert!(calls[0].is_llm_fn);
    }

    #[test]
    fn test_undefined_function_error() {
        let source = r#"
            flow broken(text: string) -> string {
                let result = nonexistent(text);
                return result;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            SemanticError::UndefinedFunction { fn_name, .. } if fn_name == "nonexistent"
        )));
    }

    #[test]
    fn test_missing_model_error() {
        let source = r#"
            llm fn bad(text: string) -> string {
                prompt: "Do something"
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, SemanticError::MissingModel { .. })));
    }

    #[test]
    fn test_enum_with_data() {
        let source = r#"
            enum Result {
                Success,
                Error(string),
                Data(int, string)
            }
        "#;

        let ctx = analyze(source).unwrap();
        let symbol = ctx.symbols.lookup("Result").unwrap();

        if let SymbolKind::Enum { variants } = &symbol.kind {
            assert_eq!(variants.len(), 3);
            assert!(variants[0].data.is_none());
            assert_eq!(variants[1].data.as_ref().unwrap().len(), 1);
            assert_eq!(variants[2].data.as_ref().unwrap().len(), 2);
        } else {
            panic!("Expected enum symbol");
        }
    }
}
