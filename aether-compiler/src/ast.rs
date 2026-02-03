//! Abstract Syntax Tree (AST) for the Aether language
//!
//! This module defines all node types that represent Aether programs after parsing.
//! The AST is designed to be:
//! - Serializable (via serde) for tooling integration
//! - Span-aware for error reporting
//! - Hierarchical for easy traversal

use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Source location information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn from_range(range: Range<usize>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }

    /// Merge two spans into one that covers both
    pub fn merge(&self, other: &Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self::from_range(range)
    }
}

/// A node with attached span information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// =============================================================================
// Top-Level Program
// =============================================================================

/// A complete Aether program/module
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Item>,
}

impl Program {
    pub fn new(items: Vec<Item>) -> Self {
        Self { items }
    }
}

/// Top-level items in a program
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Item {
    /// LLM function definition: `llm fn name(...) -> Type { ... }`
    LlmFn(LlmFnDef),

    /// Regular function definition: `fn name(...) -> Type { ... }`
    Function(FunctionDef),

    /// Flow definition: `flow name(...) -> Type { ... }`
    Flow(FlowDef),

    /// Struct definition: `struct Name { ... }`
    Struct(StructDef),

    /// Enum definition: `enum Name { ... }`
    Enum(EnumDef),

    /// Context definition: `context Name { ... }`
    Context(ContextDef),

    /// Type alias: `type Name = ...`
    TypeAlias(TypeAliasDef),

    /// Test block: `test "name" { ... }`
    Test(TestDef),

    /// Import statement: `import ... from "..."`
    Import(ImportDef),
}

// =============================================================================
// LLM Function Definition
// =============================================================================

/// LLM function: typed interaction with an LLM
///
/// ```aether
/// llm fn classify_sentiment(text: string) -> Sentiment {
///     model: "gpt-4o",
///     temperature: 0.3,
///     prompt: "Classify the sentiment: {{text}}"
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmFnDef {
    pub name: Spanned<String>,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub decorators: Vec<Decorator>,
    pub body: LlmFnBody,
    pub span: Span,
}

/// Body of an LLM function
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmFnBody {
    pub model: Option<Spanned<String>>,
    pub temperature: Option<Spanned<f64>>,
    pub max_tokens: Option<Spanned<u32>>,
    pub system_prompt: Option<Spanned<StringTemplate>>,
    pub user_prompt: Option<Spanned<StringTemplate>>,
    pub prompt: Option<Spanned<StringTemplate>>,
    pub span: Span,
}

/// A string template with interpolation: "Hello {{name}}"
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StringTemplate {
    pub parts: Vec<TemplatePart>,
}

impl StringTemplate {
    pub fn literal(s: String) -> Self {
        Self {
            parts: vec![TemplatePart::Literal(s)],
        }
    }

    pub fn new(parts: Vec<TemplatePart>) -> Self {
        Self { parts }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TemplatePart {
    /// Literal text
    Literal(String),
    /// Variable interpolation: {{var}}
    Variable(String),
}

// =============================================================================
// Regular Function Definition
// =============================================================================

/// Regular function definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: Spanned<String>,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub decorators: Vec<Decorator>,
    pub body: Block,
    pub span: Span,
}

// =============================================================================
// Flow Definition
// =============================================================================

/// Flow: DAG-based workflow orchestration
///
/// ```aether
/// flow analyze_document(doc: string) -> AnalysisResult {
///     let summary = summarize(doc);
///     let entities = extract_entities(doc);
///     return AnalysisResult { summary, entities };
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowDef {
    pub name: Spanned<String>,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub decorators: Vec<Decorator>,
    pub body: Block,
    pub span: Span,
}

// =============================================================================
// Type Definitions
// =============================================================================

/// Struct definition
///
/// ```aether
/// struct Message {
///     role: string,
///     content: string,
///     timestamp: int
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructDef {
    pub name: Spanned<String>,
    pub fields: Vec<Field>,
    pub span: Span,
}

/// Enum definition
///
/// ```aether
/// enum Sentiment { Positive, Neutral, Negative }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumDef {
    pub name: Spanned<String>,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

/// Enum variant (may have associated data in the future)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: Spanned<String>,
    pub span: Span,
}

/// Type alias
///
/// ```aether
/// type Rating = int where 1 <= value <= 5
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAliasDef {
    pub name: Spanned<String>,
    pub ty: Type,
    pub constraint: Option<Expr>,
    pub span: Span,
}

// =============================================================================
// Context Definition
// =============================================================================

/// Context: managed state across interactions
///
/// ```aether
/// context ConversationState {
///     history: list<Message>,
///     user_preferences: map<string, string>,
///     session_id: string
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextDef {
    pub name: Spanned<String>,
    pub fields: Vec<Field>,
    pub span: Span,
}

// =============================================================================
// Test Definition
// =============================================================================

/// Test block
///
/// ```aether
/// test "sentiment_classification" {
///     let result = classify_sentiment("I love this!");
///     assert result == Sentiment::Positive;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestDef {
    pub name: Spanned<String>,
    pub body: Block,
    pub span: Span,
}

// =============================================================================
// Import Definition
// =============================================================================

/// Import statement
///
/// ```aether
/// import { classify, summarize } from "./utils.aether"
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportDef {
    pub names: Vec<ImportName>,
    pub path: Spanned<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportName {
    pub name: Spanned<String>,
    pub alias: Option<Spanned<String>>,
}

// =============================================================================
// Types
// =============================================================================

/// Type representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Type {
    /// Named type: `Sentiment`, `Message`
    Named { name: String, span: Span },

    /// Primitive types: `string`, `int`, `float`, `bool`
    Primitive { kind: PrimitiveType, span: Span },

    /// List type: `list<T>`
    List { element: Box<Type>, span: Span },

    /// Map type: `map<K, V>`
    Map {
        key: Box<Type>,
        value: Box<Type>,
        span: Span,
    },

    /// Optional type: `optional<T>`
    Optional { inner: Box<Type>, span: Span },

    /// Unit type (no value)
    Unit { span: Span },
}

impl Type {
    pub fn span(&self) -> &Span {
        match self {
            Type::Named { span, .. } => span,
            Type::Primitive { span, .. } => span,
            Type::List { span, .. } => span,
            Type::Map { span, .. } => span,
            Type::Optional { span, .. } => span,
            Type::Unit { span } => span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitiveType {
    String,
    Int,
    Float,
    Bool,
}

// =============================================================================
// Common Components
// =============================================================================

/// Function/flow parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: Spanned<String>,
    pub ty: Type,
    pub span: Span,
}

/// Struct/context field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub name: Spanned<String>,
    pub ty: Type,
    pub span: Span,
}

/// Decorator (attribute)
///
/// ```aether
/// @input_guard(pii_detection, jailbreak_detection)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decorator {
    pub name: Spanned<String>,
    pub args: Vec<Expr>,
    pub span: Span,
}

// =============================================================================
// Statements and Expressions
// =============================================================================

/// A block of statements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

/// Statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Stmt {
    /// Let binding: `let x = expr;`
    Let {
        name: Spanned<String>,
        ty: Option<Type>,
        value: Expr,
        span: Span,
    },

    /// Return statement: `return expr;`
    Return { value: Option<Expr>, span: Span },

    /// Expression statement
    Expr { expr: Expr, span: Span },

    /// If statement
    If {
        condition: Expr,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },

    /// For loop: `for item in collection { ... }`
    For {
        var: Spanned<String>,
        iter: Expr,
        body: Block,
        span: Span,
    },

    /// While loop: `while condition { ... }`
    While {
        condition: Expr,
        body: Block,
        span: Span,
    },

    /// Try/catch block
    Try {
        body: Block,
        catches: Vec<CatchClause>,
        span: Span,
    },

    /// Assert statement: `assert condition;`
    Assert {
        condition: Expr,
        message: Option<Expr>,
        span: Span,
    },
}

/// Catch clause in try/catch
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatchClause {
    pub error_type: Option<Type>,
    pub binding: Option<Spanned<String>>,
    pub body: Block,
    pub span: Span,
}

/// Expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Expr {
    /// Literal value
    Literal { value: Literal, span: Span },

    /// Identifier reference
    Ident { name: String, span: Span },

    /// Binary operation: `a + b`
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        span: Span,
    },

    /// Unary operation: `!a`, `-x`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },

    /// Function call: `foo(a, b)`
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },

    /// Method call: `obj.method(args)`
    MethodCall {
        receiver: Box<Expr>,
        method: Spanned<String>,
        args: Vec<Expr>,
        span: Span,
    },

    /// Field access: `obj.field`
    FieldAccess {
        object: Box<Expr>,
        field: Spanned<String>,
        span: Span,
    },

    /// Index access: `arr[i]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },

    /// Struct literal: `Point { x: 1, y: 2 }`
    StructLiteral {
        name: Spanned<String>,
        fields: Vec<FieldInit>,
        span: Span,
    },

    /// List literal: `[1, 2, 3]`
    List { elements: Vec<Expr>, span: Span },

    /// Map literal: `{ "a": 1, "b": 2 }`
    Map { entries: Vec<MapEntry>, span: Span },

    /// Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },

    /// Await expression: `await future`
    Await { expr: Box<Expr>, span: Span },

    /// Enum variant access: `Sentiment::Positive`
    EnumVariant {
        enum_name: String,
        variant: String,
        span: Span,
    },

    /// Parenthesized expression
    Paren { expr: Box<Expr>, span: Span },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::Literal { span, .. } => span,
            Expr::Ident { span, .. } => span,
            Expr::Binary { span, .. } => span,
            Expr::Unary { span, .. } => span,
            Expr::Call { span, .. } => span,
            Expr::MethodCall { span, .. } => span,
            Expr::FieldAccess { span, .. } => span,
            Expr::Index { span, .. } => span,
            Expr::StructLiteral { span, .. } => span,
            Expr::List { span, .. } => span,
            Expr::Map { span, .. } => span,
            Expr::Match { span, .. } => span,
            Expr::Await { span, .. } => span,
            Expr::EnumVariant { span, .. } => span,
            Expr::Paren { span, .. } => span,
        }
    }
}

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Literal {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Field initialization in struct literal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldInit {
    pub name: Spanned<String>,
    pub value: Expr,
    pub span: Span,
}

/// Map entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapEntry {
    pub key: Expr,
    pub value: Expr,
    pub span: Span,
}

/// Match arm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

/// Pattern for matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Pattern {
    /// Wildcard: `_`
    Wildcard { span: Span },

    /// Literal pattern
    Literal { value: Literal, span: Span },

    /// Identifier binding
    Ident { name: String, span: Span },

    /// Enum variant: `Sentiment::Positive`
    EnumVariant {
        enum_name: Option<String>,
        variant: String,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let a = Span::new(0, 10);
        let b = Span::new(5, 20);
        let merged = a.merge(&b);
        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 20);
    }

    #[test]
    fn test_type_serialization() {
        let ty = Type::List {
            element: Box::new(Type::Primitive {
                kind: PrimitiveType::String,
                span: Span::new(0, 6),
            }),
            span: Span::new(0, 13),
        };

        let json = serde_json::to_string(&ty).unwrap();
        assert!(json.contains("\"kind\":\"List\""));
    }

    #[test]
    fn test_program_serialization() {
        let program = Program::new(vec![Item::Enum(EnumDef {
            name: Spanned::new("Sentiment".to_string(), Span::new(5, 14)),
            variants: vec![
                EnumVariant {
                    name: Spanned::new("Positive".to_string(), Span::new(17, 25)),
                    span: Span::new(17, 25),
                },
                EnumVariant {
                    name: Spanned::new("Negative".to_string(), Span::new(27, 35)),
                    span: Span::new(27, 35),
                },
            ],
            span: Span::new(0, 37),
        })]);

        let json = serde_json::to_string_pretty(&program).unwrap();
        assert!(json.contains("\"kind\":\"Enum\""));
        assert!(json.contains("\"Sentiment\""));
    }
}
