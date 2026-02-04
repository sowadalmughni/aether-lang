//! Semantic Analysis for Aether
//!
//! This module performs semantic analysis on the parsed AST, including:
//! - Symbol resolution: building symbol tables for types, functions, flows
//! - Type checking: verifying type correctness with forward-only inference
//! - Template validation: verifying {{variable}} references in prompts
//! - Dependency analysis: tracking data flow for DAG construction
//!
//! The semantic pass produces a `SemanticContext` that the codegen pass uses
//! to emit DAG JSON with correct dependency information.
//!
//! # Module Organization
//!
//! This module is organized into clear sections that could be extracted into
//! separate files in the future:
//!
//! - `errors` -> semantic/errors.rs
//! - `types` -> semantic/types.rs
//! - `symbols` -> semantic/symbols.rs
//! - `analyze_*` passes -> semantic/analyzer.rs
//!
//! For MVP, we keep everything consolidated but structured for easy extraction.

use crate::ast::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// =============================================================================
// SECTION: Errors
// =============================================================================
// Future extraction target: semantic/errors.rs

/// Maximum number of errors to collect before aborting analysis.
const MAX_ERRORS: usize = 10;

#[derive(Debug, Error, Clone)]
pub enum SemanticError {
    #[error("{message}")]
    Generic {
        message: String,
        span: Span,
    },

    #[error("Undefined symbol '{name}' at line {line}, column {column}{suggestion}")]
    UndefinedSymbol {
        name: String,
        span: Span,
        line: usize,
        column: usize,
        suggestion: String,
    },

    #[error("Duplicate definition of '{name}' at line {line}, column {column}. First defined at line {first_line}.")]
    DuplicateDefinition {
        name: String,
        span: Span,
        line: usize,
        column: usize,
        first_line: usize,
    },

    #[error("Type mismatch: expected '{expected}', found '{found}' at line {line}, column {column}")]
    TypeMismatch {
        expected: String,
        found: String,
        span: Span,
        line: usize,
        column: usize,
    },

    #[error("LLM function '{name}' requires a model specification at line {line}")]
    MissingModel {
        name: String,
        span: Span,
        line: usize,
    },

    #[error("LLM function '{name}' requires a prompt (system, user, or prompt) at line {line}")]
    MissingPrompt {
        name: String,
        span: Span,
        line: usize,
    },

    #[error("Invalid template reference '{{{{ {reference} }}}}' at line {line}. {reason}")]
    InvalidTemplateRef {
        reference: String,
        span: Span,
        line: usize,
        reason: String,
    },

    #[error("Circular dependency detected involving: {nodes}")]
    CircularDependency {
        nodes: String,
        span: Span,
    },

    #[error("Flow '{flow_name}' calls undefined function '{fn_name}' at line {line}, column {column}{suggestion}")]
    UndefinedFunction {
        flow_name: String,
        fn_name: String,
        span: Span,
        line: usize,
        column: usize,
        suggestion: String,
    },

    #[error("Wrong number of arguments for '{fn_name}': expected {expected}, found {found} at line {line}")]
    ArgumentCountMismatch {
        fn_name: String,
        expected: usize,
        found: usize,
        span: Span,
        line: usize,
    },

    #[error("Unknown argument '{arg_name}' for function '{fn_name}' at line {line}")]
    UnknownArgument {
        fn_name: String,
        arg_name: String,
        span: Span,
        line: usize,
    },

    #[error("Missing required argument '{arg_name}' for function '{fn_name}' at line {line}")]
    MissingArgument {
        fn_name: String,
        arg_name: String,
        span: Span,
        line: usize,
    },

    #[error("Duplicate field '{field_name}' in struct '{struct_name}' at line {line}")]
    DuplicateField {
        struct_name: String,
        field_name: String,
        span: Span,
        line: usize,
    },

    #[error("Duplicate variant '{variant_name}' in enum '{enum_name}' at line {line}")]
    DuplicateVariant {
        enum_name: String,
        variant_name: String,
        span: Span,
        line: usize,
    },

    #[error("Unknown field '{field_name}' for struct '{struct_name}' at line {line}")]
    UnknownField {
        struct_name: String,
        field_name: String,
        span: Span,
        line: usize,
    },

    #[error("Unknown variant '{variant_name}' for enum '{enum_name}' at line {line}")]
    UnknownVariant {
        enum_name: String,
        variant_name: String,
        span: Span,
        line: usize,
    },

    #[error("Cannot access field '{field_name}' on type '{type_name}' at line {line}")]
    InvalidFieldAccess {
        type_name: String,
        field_name: String,
        span: Span,
        line: usize,
    },

    #[error("Duplicate parameter '{param_name}' in function '{fn_name}' at line {line}")]
    DuplicateParameter {
        fn_name: String,
        param_name: String,
        span: Span,
        line: usize,
    },
}

pub type SemanticResult<T> = Result<T, Vec<SemanticError>>;

/// Compute Levenshtein distance between two strings for "Did you mean?" suggestions.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

    for i in 0..=a_len {
        matrix[i][0] = i;
    }
    for j in 0..=b_len {
        matrix[0][j] = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[a_len][b_len]
}

/// Find similar names for "Did you mean?" suggestions.
fn suggest_similar(name: &str, candidates: &[&str], max_distance: usize) -> Option<String> {
    let mut best_match = None;
    let mut best_distance = max_distance + 1;

    for candidate in candidates {
        let distance = levenshtein(name, candidate);
        if distance < best_distance && distance <= max_distance {
            best_distance = distance;
            best_match = Some(candidate.to_string());
        }
    }

    best_match.map(|s| format!(". Did you mean '{}'?", s))
}

/// Convert byte offset to (line, column) using source text.
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;

    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }

    (line, col)
}

// =============================================================================
// SECTION: Types
// =============================================================================
// Future extraction target: semantic/types.rs

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
    /// Placeholder for unresolved types - requires annotation
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

    /// Check if two types are compatible for assignment.
    pub fn is_compatible_with(&self, other: &TypeInfo) -> bool {
        match (self, other) {
            // Unknown is compatible with anything (for gradual typing)
            (TypeInfo::Unknown, _) | (_, TypeInfo::Unknown) => true,
            // Same types
            (TypeInfo::Primitive(a), TypeInfo::Primitive(b)) => a == b,
            (TypeInfo::Named(a), TypeInfo::Named(b)) => a == b,
            (TypeInfo::List(a), TypeInfo::List(b)) => a.is_compatible_with(b),
            (TypeInfo::Map(k1, v1), TypeInfo::Map(k2, v2)) => {
                k1.is_compatible_with(k2) && v1.is_compatible_with(v2)
            }
            (TypeInfo::Optional(a), TypeInfo::Optional(b)) => a.is_compatible_with(b),
            // T is compatible with optional<T>
            (t, TypeInfo::Optional(inner)) => t.is_compatible_with(inner),
            (TypeInfo::Unit, TypeInfo::Unit) => true,
            _ => false,
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
// SECTION: Symbols
// =============================================================================
// Future extraction target: semantic/symbols.rs

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
    /// Returns Err with the first definition's span if duplicate.
    pub fn define(&mut self, symbol: Symbol) -> Result<(), Span> {
        let scope = self.scopes.last_mut().unwrap();
        if let Some(existing) = scope.get(&symbol.name) {
            return Err(existing.span.clone());
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

    /// Check if a symbol exists in the current scope only.
    pub fn exists_in_current_scope(&self, name: &str) -> bool {
        self.scopes.last().map_or(false, |s| s.contains_key(name))
    }

    /// Get all symbols in the global (outermost) scope.
    pub fn global_symbols(&self) -> &HashMap<String, Symbol> {
        &self.scopes[0]
    }

    /// Get all symbol names for suggestion purposes.
    pub fn all_names(&self) -> Vec<&str> {
        self.scopes
            .iter()
            .flat_map(|scope| scope.keys().map(|s| s.as_str()))
            .collect()
    }

    /// Get all function/flow names for suggestion purposes.
    pub fn callable_names(&self) -> Vec<&str> {
        self.scopes
            .iter()
            .flat_map(|scope| {
                scope.iter().filter_map(|(name, sym)| {
                    match &sym.kind {
                        SymbolKind::LlmFn { .. }
                        | SymbolKind::Function { .. }
                        | SymbolKind::Flow { .. } => Some(name.as_str()),
                        _ => None,
                    }
                })
            })
            .collect()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SECTION: Semantic Context
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
    /// Validated template references
    pub validated_template_refs: Vec<ValidatedTemplateRef>,
}

/// A validated template reference from prompt analysis.
#[derive(Debug, Clone)]
pub struct ValidatedTemplateRef {
    pub raw: String,
    pub kind: TemplateRefKind,
    pub resolved_type: TypeInfo,
}

/// Kind of template reference.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateRefKind {
    /// Direct parameter reference: {{param}}
    Parameter,
    /// Context reference: {{context.key}}
    Context,
    /// Node output reference: {{node.id.output}}
    NodeOutput,
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
// SECTION: Semantic Analyzer
// =============================================================================
// Future extraction target: semantic/analyzer.rs

/// The semantic analyzer.
pub struct SemanticAnalyzer {
    context: SemanticContext,
    /// Original source for line/column computation
    source: String,
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
            source: String::new(),
        }
    }

    /// Create analyzer with source text for better error reporting.
    pub fn with_source(source: &str) -> Self {
        Self {
            context: SemanticContext {
                symbols: SymbolTable::new(),
                llm_functions: HashMap::new(),
                flow_calls: HashMap::new(),
                errors: Vec::new(),
            },
            source: source.to_string(),
        }
    }

    fn line_col(&self, span: &Span) -> (usize, usize) {
        if self.source.is_empty() {
            (span.start, 0)
        } else {
            offset_to_line_col(&self.source, span.start)
        }
    }

    fn add_error(&mut self, error: SemanticError) {
        if self.context.errors.len() < MAX_ERRORS {
            self.context.errors.push(error);
        }
    }

    fn should_continue(&self) -> bool {
        self.context.errors.len() < MAX_ERRORS
    }

    /// Analyze a program and return the semantic context.
    pub fn analyze(mut self, program: &Program) -> SemanticResult<SemanticContext> {
        // Pass 1: Collect all type definitions and function signatures
        for item in &program.items {
            self.collect_item(item);
            if !self.should_continue() {
                break;
            }
        }

        // Pass 2: Validate struct/enum internals (duplicate fields/variants)
        if self.should_continue() {
            for item in &program.items {
                self.validate_type_internals(item);
                if !self.should_continue() {
                    break;
                }
            }
        }

        // Pass 3: Validate LLM function bodies and template references
        if self.should_continue() {
            for item in &program.items {
                if let Item::LlmFn(def) = item {
                    self.analyze_llm_fn(def);
                }
                if !self.should_continue() {
                    break;
                }
            }
        }

        // Pass 4: Analyze flow bodies with type inference
        if self.should_continue() {
            for item in &program.items {
                if let Item::Flow(def) = item {
                    self.analyze_flow(def);
                }
                if !self.should_continue() {
                    break;
                }
            }
        }

        // Pass 5: Analyze regular function bodies
        if self.should_continue() {
            for item in &program.items {
                if let Item::Function(def) = item {
                    self.analyze_function(def);
                }
                if !self.should_continue() {
                    break;
                }
            }
        }

        if self.context.errors.is_empty() {
            Ok(self.context)
        } else {
            Err(self.context.errors)
        }
    }

    // -------------------------------------------------------------------------
    // Pass 1: Collection
    // -------------------------------------------------------------------------

    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::LlmFn(def) => self.collect_llm_fn(def),
            Item::Function(def) => self.collect_function(def),
            Item::Flow(def) => self.collect_flow(def),
            Item::Struct(def) => self.collect_struct(def),
            Item::Enum(def) => self.collect_enum(def),
            Item::Context(def) => self.collect_context(def),
            Item::TypeAlias(def) => self.collect_type_alias(def),
            Item::Test(_) | Item::Import(_) => {} // Skip for MVP
        }
    }

    fn collect_llm_fn(&mut self, def: &LlmFnDef) {
        // Check for duplicate parameters
        self.check_duplicate_params(&def.name.node, &def.params);

        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = TypeInfo::from_ast_type(&def.return_type);

        // Store LLM function info (without validated refs yet)
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
                validated_template_refs: Vec::new(),
            },
        );

        // Define in symbol table
        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::LlmFn {
                params,
                return_type,
            },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_function(&mut self, def: &FunctionDef) {
        self.check_duplicate_params(&def.name.node, &def.params);

        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = def.return_type.as_ref().map(TypeInfo::from_ast_type);

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Function {
                params,
                return_type,
            },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_flow(&mut self, def: &FlowDef) {
        self.check_duplicate_params(&def.name.node, &def.params);

        let params: Vec<_> = def
            .params
            .iter()
            .map(|p| (p.name.node.clone(), TypeInfo::from_ast_type(&p.ty)))
            .collect();
        let return_type = TypeInfo::from_ast_type(&def.return_type);

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Flow {
                params,
                return_type,
            },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_struct(&mut self, def: &StructDef) {
        let fields: Vec<_> = def
            .fields
            .iter()
            .map(|f| (f.name.node.clone(), TypeInfo::from_ast_type(&f.ty)))
            .collect();

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Struct { fields },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_enum(&mut self, def: &EnumDef) {
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

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Enum { variants },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_context(&mut self, def: &ContextDef) {
        let fields: Vec<_> = def
            .fields
            .iter()
            .map(|f| (f.name.node.clone(), TypeInfo::from_ast_type(&f.ty)))
            .collect();

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::Context { fields },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn collect_type_alias(&mut self, def: &TypeAliasDef) {
        let target = TypeInfo::from_ast_type(&def.ty);

        if let Err(first_span) = self.context.symbols.define(Symbol {
            name: def.name.node.clone(),
            kind: SymbolKind::TypeAlias { target },
            span: def.span.clone(),
        }) {
            let (line, col) = self.line_col(&def.span);
            let (first_line, _) = self.line_col(&first_span);
            self.add_error(SemanticError::DuplicateDefinition {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
                column: col,
                first_line,
            });
        }
    }

    fn check_duplicate_params(&mut self, fn_name: &str, params: &[Param]) {
        let mut seen: HashMap<&str, &Span> = HashMap::new();
        for param in params {
            if let Some(_first_span) = seen.get(param.name.node.as_str()) {
                let (line, _) = self.line_col(&param.span);
                self.add_error(SemanticError::DuplicateParameter {
                    fn_name: fn_name.to_string(),
                    param_name: param.name.node.clone(),
                    span: param.span.clone(),
                    line,
                });
            } else {
                seen.insert(&param.name.node, &param.span);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass 2: Type internal validation
    // -------------------------------------------------------------------------

    fn validate_type_internals(&mut self, item: &Item) {
        match item {
            Item::Struct(def) => {
                let mut seen: HashMap<&str, &Span> = HashMap::new();
                for field in &def.fields {
                    if let Some(_first_span) = seen.get(field.name.node.as_str()) {
                        let (line, _) = self.line_col(&field.span);
                        self.add_error(SemanticError::DuplicateField {
                            struct_name: def.name.node.clone(),
                            field_name: field.name.node.clone(),
                            span: field.span.clone(),
                            line,
                        });
                    } else {
                        seen.insert(&field.name.node, &field.span);
                    }
                }
            }
            Item::Enum(def) => {
                let mut seen: HashMap<&str, &Span> = HashMap::new();
                for variant in &def.variants {
                    if let Some(_first_span) = seen.get(variant.name.node.as_str()) {
                        let (line, _) = self.line_col(&variant.span);
                        self.add_error(SemanticError::DuplicateVariant {
                            enum_name: def.name.node.clone(),
                            variant_name: variant.name.node.clone(),
                            span: variant.span.clone(),
                            line,
                        });
                    } else {
                        seen.insert(&variant.name.node, &variant.span);
                    }
                }
            }
            _ => {}
        }
    }

    // -------------------------------------------------------------------------
    // Pass 3: LLM function analysis
    // -------------------------------------------------------------------------

    fn analyze_llm_fn(&mut self, def: &LlmFnDef) {
        let (line, _) = self.line_col(&def.span);

        // Check that model is specified
        if def.body.model.is_none() {
            self.add_error(SemanticError::MissingModel {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
            });
        }

        // Check that at least one prompt is specified
        if def.body.prompt.is_none()
            && def.body.user_prompt.is_none()
            && def.body.system_prompt.is_none()
        {
            self.add_error(SemanticError::MissingPrompt {
                name: def.name.node.clone(),
                span: def.span.clone(),
                line,
            });
        }

        // Validate template references in all prompts
        let param_names: HashSet<_> = def.params.iter().map(|p| p.name.node.as_str()).collect();

        if let Some(prompt) = &def.body.prompt {
            self.validate_template(&prompt.node, &param_names, &def.name.node, &def.body.span);
        }
        if let Some(prompt) = &def.body.user_prompt {
            self.validate_template(&prompt.node, &param_names, &def.name.node, &def.body.span);
        }
        if let Some(prompt) = &def.body.system_prompt {
            self.validate_template(&prompt.node, &param_names, &def.name.node, &def.body.span);
        }
    }

    fn validate_template(
        &mut self,
        template: &StringTemplate,
        param_names: &HashSet<&str>,
        fn_name: &str,
        span: &Span,
    ) {
        for part in &template.parts {
            if let TemplatePart::Variable(var_name) = part {
                self.validate_template_ref(var_name, param_names, fn_name, span);
            }
        }
    }

    fn validate_template_ref(
        &mut self,
        var_name: &str,
        param_names: &HashSet<&str>,
        fn_name: &str,
        span: &Span,
    ) {
        let (line, _) = self.line_col(span);

        // Check for context.KEY pattern
        if var_name.starts_with("context.") {
            let key = &var_name["context.".len()..];
            // Context references are validated at runtime for now
            // Could add context type checking in future
            if key.is_empty() {
                self.add_error(SemanticError::InvalidTemplateRef {
                    reference: var_name.to_string(),
                    span: span.clone(),
                    line,
                    reason: "Context reference requires a key after 'context.'".to_string(),
                });
            }
            return;
        }

        // Check for node.ID.output pattern
        if var_name.starts_with("node.") {
            let parts: Vec<_> = var_name["node.".len()..].split('.').collect();
            if parts.len() < 2 {
                self.add_error(SemanticError::InvalidTemplateRef {
                    reference: var_name.to_string(),
                    span: span.clone(),
                    line,
                    reason: "Node reference requires format: node.ID.field".to_string(),
                });
            }
            // Node references are validated during flow analysis
            return;
        }

        // Check for const.NAME pattern
        if var_name.starts_with("const.") {
            // Constants are provided at compile time
            return;
        }

        // Must be a parameter reference
        if !param_names.contains(var_name) {
            let suggestion = suggest_similar(
                var_name,
                &param_names.iter().copied().collect::<Vec<_>>(),
                3,
            )
            .unwrap_or_default();

            self.add_error(SemanticError::InvalidTemplateRef {
                reference: var_name.to_string(),
                span: span.clone(),
                line,
                reason: format!(
                    "Unknown parameter '{}' in LLM function '{}'{}",
                    var_name, fn_name, suggestion
                ),
            });
        }
    }

    // -------------------------------------------------------------------------
    // Pass 4: Flow analysis with type inference
    // -------------------------------------------------------------------------

    fn analyze_flow(&mut self, def: &FlowDef) {
        let flow_name = def.name.node.clone();
        let mut calls = Vec::new();

        // Get declared return type for validation
        let declared_return = TypeInfo::from_ast_type(&def.return_type);

        // Create local type environment
        let mut local_types: HashMap<String, TypeInfo> = HashMap::new();

        // Push scope for flow parameters and local variables
        self.context.symbols.push_scope();

        // Add parameters to scope and local types
        for param in &def.params {
            let ty = TypeInfo::from_ast_type(&param.ty);
            local_types.insert(param.name.node.clone(), ty.clone());

            let _ = self.context.symbols.define(Symbol {
                name: param.name.node.clone(),
                kind: SymbolKind::Parameter { ty },
                span: param.span.clone(),
            });
        }

        // Analyze flow body statements
        for stmt in &def.body.stmts {
            self.analyze_flow_stmt(&flow_name, stmt, &mut calls, &mut local_types, &declared_return);
            if !self.should_continue() {
                break;
            }
        }

        self.context.symbols.pop_scope();
        self.context.flow_calls.insert(flow_name, calls);
    }

    fn analyze_flow_stmt(
        &mut self,
        flow_name: &str,
        stmt: &Stmt,
        calls: &mut Vec<FlowCall>,
        local_types: &mut HashMap<String, TypeInfo>,
        declared_return: &TypeInfo,
    ) {
        match stmt {
            Stmt::Let {
                name, ty, value, span, ..
            } => {
                // Infer type from RHS
                let inferred_type = self.infer_expr_type(value, local_types);

                // If explicit type annotation, check compatibility
                let final_type = if let Some(explicit_ty) = ty {
                    let explicit = TypeInfo::from_ast_type(explicit_ty);
                    if !inferred_type.is_compatible_with(&explicit) && inferred_type != TypeInfo::Unknown {
                        let (line, col) = self.line_col(span);
                        self.add_error(SemanticError::TypeMismatch {
                            expected: explicit.to_string(),
                            found: inferred_type.to_string(),
                            span: span.clone(),
                            line,
                            column: col,
                        });
                    }
                    explicit
                } else {
                    inferred_type.clone()
                };

                // Extract function calls
                if let Some(call) = self.extract_call(flow_name, value, Some(name.node.clone()), span, local_types) {
                    calls.push(call);
                }

                // Add variable to local types and scope
                local_types.insert(name.node.clone(), final_type.clone());

                let _ = self.context.symbols.define(Symbol {
                    name: name.node.clone(),
                    kind: SymbolKind::Variable { ty: final_type },
                    span: span.clone(),
                });
            }

            Stmt::Return { value, span } => {
                if let Some(ret_expr) = value {
                    let ret_type = self.infer_expr_type(ret_expr, local_types);
                    if !ret_type.is_compatible_with(declared_return) && ret_type != TypeInfo::Unknown {
                        let (line, col) = self.line_col(span);
                        self.add_error(SemanticError::TypeMismatch {
                            expected: declared_return.to_string(),
                            found: ret_type.to_string(),
                            span: span.clone(),
                            line,
                            column: col,
                        });
                    }
                }
            }

            Stmt::Expr { expr, span } => {
                // Validate expression and extract calls
                let _ = self.infer_expr_type(expr, local_types);
                if let Some(call) = self.extract_call(flow_name, expr, None, span, local_types) {
                    calls.push(call);
                }
            }

            Stmt::If {
                condition,
                then_block,
                else_block,
                ..
            } => {
                // Validate condition is boolean
                let cond_type = self.infer_expr_type(condition, local_types);
                if cond_type != TypeInfo::Primitive(PrimitiveType::Bool) && cond_type != TypeInfo::Unknown {
                    let (line, col) = self.line_col(condition.span());
                    self.add_error(SemanticError::TypeMismatch {
                        expected: "bool".to_string(),
                        found: cond_type.to_string(),
                        span: condition.span().clone(),
                        line,
                        column: col,
                    });
                }

                // Analyze then block
                for s in &then_block.stmts {
                    self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                }

                // Analyze else block
                if let Some(else_b) = else_block {
                    for s in &else_b.stmts {
                        self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                    }
                }
            }

            Stmt::For { var, iter, body, .. } => {
                // Infer iterator type
                let iter_type = self.infer_expr_type(iter, local_types);
                let elem_type = match iter_type {
                    TypeInfo::List(elem) => *elem,
                    _ => TypeInfo::Unknown,
                };

                // Add loop variable
                local_types.insert(var.node.clone(), elem_type);

                for s in &body.stmts {
                    self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                }
            }

            Stmt::While { condition, body, .. } => {
                let _ = self.infer_expr_type(condition, local_types);
                for s in &body.stmts {
                    self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                }
            }

            Stmt::Try { body, catches, .. } => {
                for s in &body.stmts {
                    self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                }
                for catch in catches {
                    for s in &catch.body.stmts {
                        self.analyze_flow_stmt(flow_name, s, calls, local_types, declared_return);
                    }
                }
            }

            Stmt::Assert { condition, .. } => {
                let _ = self.infer_expr_type(condition, local_types);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pass 5: Regular function analysis
    // -------------------------------------------------------------------------

    fn analyze_function(&mut self, def: &FunctionDef) {
        // Similar to flow but simpler for MVP
        self.context.symbols.push_scope();

        for param in &def.params {
            let ty = TypeInfo::from_ast_type(&param.ty);
            let _ = self.context.symbols.define(Symbol {
                name: param.name.node.clone(),
                kind: SymbolKind::Parameter { ty },
                span: param.span.clone(),
            });
        }

        // For MVP, we just validate variable references exist
        // Full function body analysis can be added later

        self.context.symbols.pop_scope();
    }

    // -------------------------------------------------------------------------
    // Type inference
    // -------------------------------------------------------------------------

    fn infer_expr_type(&mut self, expr: &Expr, local_types: &HashMap<String, TypeInfo>) -> TypeInfo {
        match expr {
            Expr::Literal { value, .. } => match value {
                Literal::String(_) => TypeInfo::Primitive(PrimitiveType::String),
                Literal::Int(_) => TypeInfo::Primitive(PrimitiveType::Int),
                Literal::Float(_) => TypeInfo::Primitive(PrimitiveType::Float),
                Literal::Bool(_) => TypeInfo::Primitive(PrimitiveType::Bool),
            },

            Expr::Ident { name, span } => {
                // Check local types first
                if let Some(ty) = local_types.get(name) {
                    return ty.clone();
                }

                // Check symbol table
                if let Some(sym) = self.context.symbols.lookup(name) {
                    match &sym.kind {
                        SymbolKind::Variable { ty } | SymbolKind::Parameter { ty } => ty.clone(),
                        _ => TypeInfo::Named(name.clone()),
                    }
                } else {
                    let (line, col) = self.line_col(span);
                    let candidates = self.context.symbols.all_names();
                    let suggestion = suggest_similar(name, &candidates, 3).unwrap_or_default();

                    self.add_error(SemanticError::UndefinedSymbol {
                        name: name.clone(),
                        span: span.clone(),
                        line,
                        column: col,
                        suggestion,
                    });
                    TypeInfo::Unknown
                }
            }

            Expr::Binary { left, op, right, span } => {
                let left_ty = self.infer_expr_type(left, local_types);
                let right_ty = self.infer_expr_type(right, local_types);

                match op {
                    // Comparison ops return bool
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::Gt
                    | BinaryOp::LtEq | BinaryOp::GtEq => TypeInfo::Primitive(PrimitiveType::Bool),

                    // Logical ops require bool, return bool
                    BinaryOp::And | BinaryOp::Or => {
                        if left_ty != TypeInfo::Primitive(PrimitiveType::Bool) && left_ty != TypeInfo::Unknown {
                            let (line, col) = self.line_col(left.span());
                            self.add_error(SemanticError::TypeMismatch {
                                expected: "bool".to_string(),
                                found: left_ty.to_string(),
                                span: span.clone(),
                                line,
                                column: col,
                            });
                        }
                        TypeInfo::Primitive(PrimitiveType::Bool)
                    }

                    // Arithmetic ops
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                        // String concatenation
                        if left_ty == TypeInfo::Primitive(PrimitiveType::String) {
                            return TypeInfo::Primitive(PrimitiveType::String);
                        }
                        // Numeric operations preserve type
                        if left_ty == TypeInfo::Unknown {
                            right_ty
                        } else {
                            left_ty
                        }
                    }
                }
            }

            Expr::Unary { op, operand, .. } => {
                let operand_ty = self.infer_expr_type(operand, local_types);
                match op {
                    UnaryOp::Not => TypeInfo::Primitive(PrimitiveType::Bool),
                    UnaryOp::Neg => operand_ty,
                }
            }

            Expr::Call { func, args, span } => {
                // Get function name
                let fn_name = match func.as_ref() {
                    Expr::Ident { name, .. } => name.clone(),
                    _ => return TypeInfo::Unknown,
                };

                // Look up function
                if let Some(sym) = self.context.symbols.lookup(&fn_name) {
                    let (params, return_type) = match &sym.kind {
                        SymbolKind::LlmFn { params, return_type } => (params.clone(), return_type.clone()),
                        SymbolKind::Function { params, return_type } => {
                            (params.clone(), return_type.clone().unwrap_or(TypeInfo::Unit))
                        }
                        SymbolKind::Flow { params, return_type } => (params.clone(), return_type.clone()),
                        _ => {
                            let (line, col) = self.line_col(span);
                            self.add_error(SemanticError::Generic {
                                message: format!("'{}' is not a callable function", fn_name),
                                span: span.clone(),
                            });
                            return TypeInfo::Unknown;
                        }
                    };

                    // Check argument count
                    if args.len() != params.len() {
                        let (line, _) = self.line_col(span);
                        self.add_error(SemanticError::ArgumentCountMismatch {
                            fn_name: fn_name.clone(),
                            expected: params.len(),
                            found: args.len(),
                            span: span.clone(),
                            line,
                        });
                    }

                    // Check argument types
                    for (arg, (_param_name, param_ty)) in args.iter().zip(params.iter()) {
                        let arg_ty = self.infer_expr_type(arg, local_types);
                        if !arg_ty.is_compatible_with(param_ty) && arg_ty != TypeInfo::Unknown {
                            let (line, col) = self.line_col(arg.span());
                            self.add_error(SemanticError::TypeMismatch {
                                expected: param_ty.to_string(),
                                found: arg_ty.to_string(),
                                span: arg.span().clone(),
                                line,
                                column: col,
                            });
                        }
                    }

                    return_type
                } else {
                    let (line, col) = self.line_col(span);
                    let candidates = self.context.symbols.callable_names();
                    let suggestion = suggest_similar(&fn_name, &candidates, 3).unwrap_or_default();

                    self.add_error(SemanticError::UndefinedSymbol {
                        name: fn_name,
                        span: span.clone(),
                        line,
                        column: col,
                        suggestion,
                    });
                    TypeInfo::Unknown
                }
            }

            Expr::FieldAccess { object, field, span } => {
                let obj_ty = self.infer_expr_type(object, local_types);

                match &obj_ty {
                    TypeInfo::Named(name) => {
                        // Look up struct definition
                        if let Some(sym) = self.context.symbols.lookup(name) {
                            if let SymbolKind::Struct { fields } = &sym.kind {
                                if let Some((_, ty)) = fields.iter().find(|(n, _)| n == &field.node) {
                                    return ty.clone();
                                } else {
                                    let (line, _) = self.line_col(span);
                                    self.add_error(SemanticError::UnknownField {
                                        struct_name: name.clone(),
                                        field_name: field.node.clone(),
                                        span: span.clone(),
                                        line,
                                    });
                                }
                            }
                        }
                        TypeInfo::Unknown
                    }
                    TypeInfo::Unknown => TypeInfo::Unknown,
                    _ => {
                        let (line, _) = self.line_col(span);
                        self.add_error(SemanticError::InvalidFieldAccess {
                            type_name: obj_ty.to_string(),
                            field_name: field.node.clone(),
                            span: span.clone(),
                            line,
                        });
                        TypeInfo::Unknown
                    }
                }
            }

            Expr::Index { object, index, span } => {
                let obj_ty = self.infer_expr_type(object, local_types);
                let _ = self.infer_expr_type(index, local_types);

                match obj_ty {
                    TypeInfo::List(elem) => *elem,
                    TypeInfo::Map(_, value) => *value,
                    TypeInfo::Unknown => TypeInfo::Unknown,
                    _ => {
                        let (line, _) = self.line_col(span);
                        self.add_error(SemanticError::Generic {
                            message: format!("Cannot index into type '{}'", obj_ty.to_string()),
                            span: span.clone(),
                        });
                        TypeInfo::Unknown
                    }
                }
            }

            Expr::StructLiteral { name, fields: field_inits, span } => {
                // Look up struct definition
                if let Some(sym) = self.context.symbols.lookup(&name.node) {
                    if let SymbolKind::Struct { fields } = &sym.kind {
                        let field_map: HashMap<_, _> = fields.iter().cloned().collect();

                        // Check each field initialization
                        for init in field_inits {
                            if let Some(expected_ty) = field_map.get(&init.name.node) {
                                let actual_ty = self.infer_expr_type(&init.value, local_types);
                                if !actual_ty.is_compatible_with(expected_ty) && actual_ty != TypeInfo::Unknown {
                                    let (line, col) = self.line_col(&init.span);
                                    self.add_error(SemanticError::TypeMismatch {
                                        expected: expected_ty.to_string(),
                                        found: actual_ty.to_string(),
                                        span: init.span.clone(),
                                        line,
                                        column: col,
                                    });
                                }
                            } else {
                                let (line, _) = self.line_col(&init.span);
                                self.add_error(SemanticError::UnknownField {
                                    struct_name: name.node.clone(),
                                    field_name: init.name.node.clone(),
                                    span: init.span.clone(),
                                    line,
                                });
                            }
                        }

                        return TypeInfo::Named(name.node.clone());
                    }
                }

                let (line, col) = self.line_col(span);
                self.add_error(SemanticError::UndefinedSymbol {
                    name: name.node.clone(),
                    span: span.clone(),
                    line,
                    column: col,
                    suggestion: String::new(),
                });
                TypeInfo::Unknown
            }

            Expr::List { elements, .. } => {
                if elements.is_empty() {
                    TypeInfo::List(Box::new(TypeInfo::Unknown))
                } else {
                    let elem_ty = self.infer_expr_type(&elements[0], local_types);
                    TypeInfo::List(Box::new(elem_ty))
                }
            }

            Expr::Map { entries, .. } => {
                if entries.is_empty() {
                    TypeInfo::Map(Box::new(TypeInfo::Unknown), Box::new(TypeInfo::Unknown))
                } else {
                    let key_ty = self.infer_expr_type(&entries[0].key, local_types);
                    let val_ty = self.infer_expr_type(&entries[0].value, local_types);
                    TypeInfo::Map(Box::new(key_ty), Box::new(val_ty))
                }
            }

            Expr::EnumVariant { enum_name, variant, span } => {
                // Validate enum and variant exist
                if let Some(sym) = self.context.symbols.lookup(enum_name) {
                    if let SymbolKind::Enum { variants } = &sym.kind {
                        if !variants.iter().any(|v| &v.name == variant) {
                            let (line, _) = self.line_col(span);
                            self.add_error(SemanticError::UnknownVariant {
                                enum_name: enum_name.clone(),
                                variant_name: variant.clone(),
                                span: span.clone(),
                                line,
                            });
                        }
                    }
                }
                TypeInfo::Named(enum_name.clone())
            }

            Expr::Await { expr, .. } => self.infer_expr_type(expr, local_types),

            Expr::Match { expr, arms, .. } => {
                let _ = self.infer_expr_type(expr, local_types);
                // Return type of first arm (should all be same)
                if let Some(arm) = arms.first() {
                    self.infer_expr_type(&arm.body, local_types)
                } else {
                    TypeInfo::Unknown
                }
            }

            Expr::Paren { expr, .. } => self.infer_expr_type(expr, local_types),

            Expr::MethodCall { receiver, args, .. } => {
                let _ = self.infer_expr_type(receiver, local_types);
                for arg in args {
                    let _ = self.infer_expr_type(arg, local_types);
                }
                // Method return types need more complex analysis
                TypeInfo::Unknown
            }
        }
    }

    // -------------------------------------------------------------------------
    // Call extraction for flow analysis
    // -------------------------------------------------------------------------

    fn extract_call(
        &mut self,
        flow_name: &str,
        expr: &Expr,
        binding: Option<String>,
        span: &Span,
        local_types: &HashMap<String, TypeInfo>,
    ) -> Option<FlowCall> {
        match expr {
            Expr::Call { func, args, .. } => {
                // Get function name
                let callee = match func.as_ref() {
                    Expr::Ident { name, .. } => name.clone(),
                    _ => return None,
                };

                // Check if callee exists
                let is_llm_fn = self.context.llm_functions.contains_key(&callee);
                let exists = self.context.symbols.lookup(&callee).is_some();

                if !exists {
                    let (line, col) = self.line_col(span);
                    let candidates = self.context.symbols.callable_names();
                    let suggestion = suggest_similar(&callee, &candidates, 3).unwrap_or_default();

                    self.add_error(SemanticError::UndefinedFunction {
                        flow_name: flow_name.to_string(),
                        fn_name: callee.clone(),
                        span: span.clone(),
                        line,
                        column: col,
                        suggestion,
                    });
                }

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
            Expr::Await { expr, .. } => self.extract_call(flow_name, expr, binding, span, local_types),
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
        SemanticAnalyzer::with_source(source).analyze(&program)
    }

    // -------------------------------------------------------------------------
    // Symbol table tests
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Error detection tests
    // -------------------------------------------------------------------------

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
    fn test_missing_prompt_error() {
        let source = r#"
            llm fn bad(text: string) -> string {
                model: "gpt-4o"
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, SemanticError::MissingPrompt { .. })));
    }

    #[test]
    fn test_duplicate_definition_error() {
        let source = r#"
            struct Message { content: string }
            struct Message { text: string }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::DuplicateDefinition { name, .. } if name == "Message")
        }));
    }

    #[test]
    fn test_duplicate_field_error() {
        let source = r#"
            struct Bad { name: string, name: int }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::DuplicateField { field_name, .. } if field_name == "name")
        }));
    }

    #[test]
    fn test_duplicate_variant_error() {
        let source = r#"
            enum Bad { Value, Value }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::DuplicateVariant { variant_name, .. } if variant_name == "Value")
        }));
    }

    #[test]
    fn test_duplicate_parameter_error() {
        let source = r#"
            llm fn bad(x: string, x: int) -> string {
                model: "gpt-4o",
                prompt: "test"
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::DuplicateParameter { param_name, .. } if param_name == "x")
        }));
    }

    #[test]
    fn test_invalid_template_ref_error() {
        let source = r#"
            llm fn bad(text: string) -> string {
                model: "gpt-4o",
                prompt: "Process: {{unknown_param}}"
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::InvalidTemplateRef { reference, .. } if reference == "unknown_param")
        }));
    }

    // -------------------------------------------------------------------------
    // Type inference tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_type_inference_literals() {
        let source = r#"
            llm fn echo(msg: string) -> string {
                model: "gpt-4o",
                prompt: "{{msg}}"
            }

            flow test_types(x: string) -> string {
                let s = "hello";
                let n = 42;
                let f = 3.14;
                let b = true;
                return x;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_argument_count_mismatch() {
        let source = r#"
            llm fn greet(name: string) -> string {
                model: "gpt-4o",
                prompt: "Hello {{name}}"
            }

            flow test(x: string) -> string {
                let result = greet(x, x);
                return result;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::ArgumentCountMismatch { expected: 1, found: 2, .. })
        }));
    }

    #[test]
    fn test_return_type_mismatch() {
        let source = r#"
            flow bad(x: string) -> int {
                return x;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::TypeMismatch { expected, found, .. } 
                if expected == "int" && found == "string")
        }));
    }

    #[test]
    fn test_unknown_field_access() {
        let source = r#"
            struct Point { x: int, y: int }

            flow test(p: Point) -> int {
                return p.z;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::UnknownField { field_name, .. } if field_name == "z")
        }));
    }

    #[test]
    fn test_unknown_enum_variant() {
        let source = r#"
            enum Color { Red, Green, Blue }

            flow test() -> Color {
                return Color::Yellow;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| {
            matches!(e, SemanticError::UnknownVariant { variant_name, .. } if variant_name == "Yellow")
        }));
    }

    // -------------------------------------------------------------------------
    // Valid program tests (should not error)
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_llm_fn_parses() {
        let source = r#"
            llm fn classify(text: string) -> string {
                model: "gpt-4o",
                temperature: 0.5,
                max_tokens: 100,
                prompt: "Classify: {{text}}"
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_flow_with_calls() {
        let source = r#"
            llm fn step1(x: string) -> string {
                model: "gpt-4o",
                prompt: "Step 1: {{x}}"
            }

            flow pipeline(input: string) -> string {
                let result = step1(input);
                return result;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enum_with_associated_data() {
        let source = r#"
            enum Result {
                Success,
                Error(string),
                Data(int, string)
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());

        let ctx = result.unwrap();
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

    #[test]
    fn test_struct_literal_validation() {
        let source = r#"
            struct Point { x: int, y: int }

            flow make_point() -> Point {
                return Point { x: 1, y: 2 };
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chained_flow_calls() {
        let source = r#"
            llm fn step1(x: string) -> string {
                model: "gpt-4o",
                prompt: "Step 1: {{x}}"
            }

            llm fn step2(x: string) -> string {
                model: "gpt-4o",
                prompt: "Step 2: {{x}}"
            }

            flow chain(input: string) -> string {
                let a = step1(input);
                let b = step2(a);
                return b;
            }
        "#;

        let result = analyze(source);
        assert!(result.is_ok());

        let ctx = result.unwrap();
        let calls = &ctx.flow_calls["chain"];
        assert_eq!(calls.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Suggestion tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", "abd"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_suggest_similar() {
        let candidates = vec!["classify_sentiment", "extract_entities", "summarize"];
        
        let suggestion = suggest_similar("clasify_sentiment", &candidates, 3);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("classify_sentiment"));

        let suggestion = suggest_similar("xyz", &candidates, 3);
        assert!(suggestion.is_none());
    }
}
