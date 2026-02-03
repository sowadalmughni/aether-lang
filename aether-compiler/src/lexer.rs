//! Aether Lexer
//!
//! Tokenizes Aether source code using the `logos` library for efficient lexical analysis.
//!
//! # Token Categories
//!
//! - **Keywords**: `llm`, `fn`, `flow`, `context`, `test`, `struct`, `enum`, etc.
//! - **Operators**: `->`, `:`, `,`, `=`, `==`, `!=`, `<`, `>`, etc.
//! - **Delimiters**: `{`, `}`, `(`, `)`, `[`, `]`
//! - **Literals**: Strings (with `{{var}}` template support), integers, floats, booleans
//! - **Identifiers**: User-defined names

use logos::Logos;
use std::fmt;
use std::ops::Range;

/// A token with its kind, span, and source slice
#[derive(Debug, Clone, PartialEq)]
pub struct Token<'src> {
    pub kind: TokenKind,
    pub span: Range<usize>,
    pub text: &'src str,
}

impl<'src> Token<'src> {
    pub fn new(kind: TokenKind, span: Range<usize>, text: &'src str) -> Self {
        Self { kind, span, text }
    }
}

/// All token types in the Aether language
#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum TokenKind {
    // ==========================================================================
    // Keywords
    // ==========================================================================
    #[token("llm")]
    Llm,

    #[token("fn")]
    Fn,

    #[token("flow")]
    Flow,

    #[token("context")]
    Context,

    #[token("test")]
    Test,

    #[token("struct")]
    Struct,

    #[token("enum")]
    Enum,

    #[token("type")]
    Type,

    #[token("let")]
    Let,

    #[token("const")]
    Const,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("match")]
    Match,

    #[token("for")]
    For,

    #[token("in")]
    In,

    #[token("while")]
    While,

    #[token("return")]
    Return,

    #[token("try")]
    Try,

    #[token("catch")]
    Catch,

    #[token("retry")]
    Retry,

    #[token("fallback")]
    Fallback,

    #[token("assert")]
    Assert,

    #[token("async")]
    Async,

    #[token("await")]
    Await,

    #[token("import")]
    Import,

    #[token("from")]
    From,

    #[token("as")]
    As,

    #[token("pub")]
    Pub,

    #[token("where")]
    Where,

    // ==========================================================================
    // Built-in type keywords
    // ==========================================================================
    #[token("string")]
    StringType,

    #[token("int")]
    IntType,

    #[token("float")]
    FloatType,

    #[token("bool")]
    BoolType,

    #[token("list")]
    ListType,

    #[token("map")]
    MapType,

    #[token("optional")]
    OptionalType,

    // ==========================================================================
    // Boolean literals
    // ==========================================================================
    #[token("true")]
    True,

    #[token("false")]
    False,

    // ==========================================================================
    // LLM-specific keywords
    // ==========================================================================
    #[token("model")]
    Model,

    #[token("prompt")]
    Prompt,

    #[token("system")]
    System,

    #[token("user")]
    User,

    #[token("temperature")]
    Temperature,

    #[token("max_tokens")]
    MaxTokens,

    #[token("golden_dataset")]
    GoldenDataset,

    // ==========================================================================
    // Operators
    // ==========================================================================
    #[token("->")]
    Arrow,

    #[token(":")]
    Colon,

    #[token("::")]
    DoubleColon,

    #[token(",")]
    Comma,

    #[token(".")]
    Dot,

    #[token("=")]
    Eq,

    #[token("==")]
    EqEq,

    #[token("!=")]
    NotEq,

    #[token("<")]
    Lt,

    #[token(">")]
    Gt,

    #[token("<=")]
    LtEq,

    #[token(">=")]
    GtEq,

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("%")]
    Percent,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    #[token("|")]
    Pipe,

    #[token("@")]
    At,

    #[token("?")]
    Question,

    #[token(";")]
    Semicolon,

    // ==========================================================================
    // Delimiters
    // ==========================================================================
    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    // ==========================================================================
    // Template markers (inside strings)
    // ==========================================================================
    #[token("{{")]
    TemplateOpen,

    #[token("}}")]
    TemplateClose,

    // ==========================================================================
    // Literals
    // ==========================================================================
    /// String literal: "..." (handles escape sequences)
    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    /// Integer literal
    #[regex(r"-?[0-9]+", priority = 2)]
    IntLiteral,

    /// Float literal
    #[regex(r"-?[0-9]+\.[0-9]+")]
    FloatLiteral,

    // ==========================================================================
    // Identifiers
    // ==========================================================================
    /// Identifier: starts with letter or underscore, followed by alphanumeric or underscore
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    // ==========================================================================
    // Comments
    // ==========================================================================
    /// Single-line comment
    #[regex(r"//[^\n]*")]
    LineComment,

    /// Multi-line comment (non-nested)
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/")]
    BlockComment,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Llm => write!(f, "llm"),
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::Flow => write!(f, "flow"),
            TokenKind::Context => write!(f, "context"),
            TokenKind::Test => write!(f, "test"),
            TokenKind::Struct => write!(f, "struct"),
            TokenKind::Enum => write!(f, "enum"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Const => write!(f, "const"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::For => write!(f, "for"),
            TokenKind::In => write!(f, "in"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Catch => write!(f, "catch"),
            TokenKind::Retry => write!(f, "retry"),
            TokenKind::Fallback => write!(f, "fallback"),
            TokenKind::Assert => write!(f, "assert"),
            TokenKind::Async => write!(f, "async"),
            TokenKind::Await => write!(f, "await"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::From => write!(f, "from"),
            TokenKind::As => write!(f, "as"),
            TokenKind::Pub => write!(f, "pub"),
            TokenKind::Where => write!(f, "where"),
            TokenKind::StringType => write!(f, "string"),
            TokenKind::IntType => write!(f, "int"),
            TokenKind::FloatType => write!(f, "float"),
            TokenKind::BoolType => write!(f, "bool"),
            TokenKind::ListType => write!(f, "list"),
            TokenKind::MapType => write!(f, "map"),
            TokenKind::OptionalType => write!(f, "optional"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Model => write!(f, "model"),
            TokenKind::Prompt => write!(f, "prompt"),
            TokenKind::System => write!(f, "system"),
            TokenKind::User => write!(f, "user"),
            TokenKind::Temperature => write!(f, "temperature"),
            TokenKind::MaxTokens => write!(f, "max_tokens"),
            TokenKind::GoldenDataset => write!(f, "golden_dataset"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::DoubleColon => write!(f, "::"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Eq => write!(f, "="),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::NotEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::TemplateOpen => write!(f, "{{{{"),
            TokenKind::TemplateClose => write!(f, "}}}}"),
            TokenKind::StringLiteral => write!(f, "string literal"),
            TokenKind::IntLiteral => write!(f, "integer"),
            TokenKind::FloatLiteral => write!(f, "float"),
            TokenKind::Ident => write!(f, "identifier"),
            TokenKind::LineComment => write!(f, "comment"),
            TokenKind::BlockComment => write!(f, "block comment"),
        }
    }
}

/// Lexer for Aether source code
pub struct Lexer<'src> {
    inner: logos::Lexer<'src, TokenKind>,
    source: &'src str,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source code
    pub fn new(source: &'src str) -> Self {
        Self {
            inner: TokenKind::lexer(source),
            source,
        }
    }

    /// Get the original source code
    pub fn source(&self) -> &'src str {
        self.source
    }

    /// Tokenize the entire source, returning all tokens (including errors)
    pub fn tokenize(self) -> Vec<Result<Token<'src>, LexError>> {
        self.collect()
    }

    /// Tokenize, filtering out comments and collecting only valid tokens
    pub fn tokenize_filtered(self) -> Result<Vec<Token<'src>>, LexError> {
        self.filter_map(|result| match result {
            Ok(token) => {
                // Skip comments
                if matches!(token.kind, TokenKind::LineComment | TokenKind::BlockComment) {
                    None
                } else {
                    Some(Ok(token))
                }
            }
            Err(e) => Some(Err(e)),
        })
        .collect()
    }
}

/// Error during lexical analysis
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub span: Range<usize>,
    pub text: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unexpected character(s) '{}' at position {}..{}",
            self.text, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for LexError {}

impl<'src> Iterator for Lexer<'src> {
    type Item = Result<Token<'src>, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        let kind = self.inner.next()?;
        let span = self.inner.span();
        let text = self.inner.slice();

        match kind {
            Ok(kind) => Some(Ok(Token::new(kind, span, text))),
            Err(_) => Some(Err(LexError {
                span,
                text: text.to_string(),
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(source: &str) -> Vec<TokenKind> {
        Lexer::new(source)
            .filter_map(|r| r.ok())
            .filter(|t| !matches!(t.kind, TokenKind::LineComment | TokenKind::BlockComment))
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn test_keywords() {
        assert_eq!(tokenize("llm fn flow"), vec![TokenKind::Llm, TokenKind::Fn, TokenKind::Flow]);
        assert_eq!(tokenize("struct enum type"), vec![TokenKind::Struct, TokenKind::Enum, TokenKind::Type]);
        assert_eq!(tokenize("context test"), vec![TokenKind::Context, TokenKind::Test]);
        assert_eq!(tokenize("if else match"), vec![TokenKind::If, TokenKind::Else, TokenKind::Match]);
        assert_eq!(tokenize("try catch retry fallback"), vec![
            TokenKind::Try, TokenKind::Catch, TokenKind::Retry, TokenKind::Fallback
        ]);
    }

    #[test]
    fn test_type_keywords() {
        assert_eq!(tokenize("string int float bool"), vec![
            TokenKind::StringType, TokenKind::IntType, TokenKind::FloatType, TokenKind::BoolType
        ]);
        assert_eq!(tokenize("list map optional"), vec![
            TokenKind::ListType, TokenKind::MapType, TokenKind::OptionalType
        ]);
    }

    #[test]
    fn test_llm_keywords() {
        assert_eq!(tokenize("model prompt system user"), vec![
            TokenKind::Model, TokenKind::Prompt, TokenKind::System, TokenKind::User
        ]);
        assert_eq!(tokenize("temperature max_tokens"), vec![
            TokenKind::Temperature, TokenKind::MaxTokens
        ]);
    }

    #[test]
    fn test_operators() {
        assert_eq!(tokenize("-> : :: , ."), vec![
            TokenKind::Arrow, TokenKind::Colon, TokenKind::DoubleColon,
            TokenKind::Comma, TokenKind::Dot
        ]);
        assert_eq!(tokenize("= == != < > <= >="), vec![
            TokenKind::Eq, TokenKind::EqEq, TokenKind::NotEq,
            TokenKind::Lt, TokenKind::Gt, TokenKind::LtEq, TokenKind::GtEq
        ]);
        assert_eq!(tokenize("+ - * / %"), vec![
            TokenKind::Plus, TokenKind::Minus, TokenKind::Star,
            TokenKind::Slash, TokenKind::Percent
        ]);
        assert_eq!(tokenize("&& || !"), vec![TokenKind::And, TokenKind::Or, TokenKind::Not]);
    }

    #[test]
    fn test_delimiters() {
        assert_eq!(tokenize("{ } ( ) [ ]"), vec![
            TokenKind::LBrace, TokenKind::RBrace,
            TokenKind::LParen, TokenKind::RParen,
            TokenKind::LBracket, TokenKind::RBracket
        ]);
    }

    #[test]
    fn test_literals() {
        assert_eq!(tokenize("42"), vec![TokenKind::IntLiteral]);
        assert_eq!(tokenize("-123"), vec![TokenKind::IntLiteral]);
        assert_eq!(tokenize("3.14"), vec![TokenKind::FloatLiteral]);
        assert_eq!(tokenize("-0.5"), vec![TokenKind::FloatLiteral]);
        assert_eq!(tokenize("true false"), vec![TokenKind::True, TokenKind::False]);
        assert_eq!(tokenize(r#""hello world""#), vec![TokenKind::StringLiteral]);
        assert_eq!(tokenize(r#""escaped \"quote\"""#), vec![TokenKind::StringLiteral]);
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(tokenize("foo bar_baz _private"), vec![
            TokenKind::Ident, TokenKind::Ident, TokenKind::Ident
        ]);
        assert_eq!(tokenize("camelCase PascalCase snake_case"), vec![
            TokenKind::Ident, TokenKind::Ident, TokenKind::Ident
        ]);
    }

    #[test]
    fn test_template_markers() {
        assert_eq!(tokenize("{{ }}"), vec![TokenKind::TemplateOpen, TokenKind::TemplateClose]);
    }

    #[test]
    fn test_comments() {
        let tokens: Vec<_> = Lexer::new("// this is a comment\nfn")
            .filter_map(|r| r.ok())
            .map(|t| t.kind)
            .collect();
        assert_eq!(tokens, vec![TokenKind::LineComment, TokenKind::Fn]);

        let tokens: Vec<_> = Lexer::new("/* block */ fn")
            .filter_map(|r| r.ok())
            .map(|t| t.kind)
            .collect();
        assert_eq!(tokens, vec![TokenKind::BlockComment, TokenKind::Fn]);
    }

    #[test]
    fn test_llm_fn_signature() {
        let source = r#"llm fn classify(text: string) -> Sentiment"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::Llm,
            TokenKind::Fn,
            TokenKind::Ident, // classify
            TokenKind::LParen,
            TokenKind::Ident, // text
            TokenKind::Colon,
            TokenKind::StringType,
            TokenKind::RParen,
            TokenKind::Arrow,
            TokenKind::Ident, // Sentiment
        ]);
    }

    #[test]
    fn test_struct_definition() {
        let source = r#"struct Message { role: string, content: string }"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::Struct,
            TokenKind::Ident, // Message
            TokenKind::LBrace,
            TokenKind::Ident, // role
            TokenKind::Colon,
            TokenKind::StringType,
            TokenKind::Comma,
            TokenKind::Ident, // content
            TokenKind::Colon,
            TokenKind::StringType,
            TokenKind::RBrace,
        ]);
    }

    #[test]
    fn test_enum_definition() {
        let source = r#"enum Sentiment { Positive, Neutral, Negative }"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::Enum,
            TokenKind::Ident, // Sentiment
            TokenKind::LBrace,
            TokenKind::Ident, // Positive
            TokenKind::Comma,
            TokenKind::Ident, // Neutral
            TokenKind::Comma,
            TokenKind::Ident, // Negative
            TokenKind::RBrace,
        ]);
    }

    #[test]
    fn test_flow_signature() {
        let source = r#"flow analyze(doc: string) -> Result"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::Flow,
            TokenKind::Ident, // analyze
            TokenKind::LParen,
            TokenKind::Ident, // doc
            TokenKind::Colon,
            TokenKind::StringType,
            TokenKind::RParen,
            TokenKind::Arrow,
            TokenKind::Ident, // Result
        ]);
    }

    #[test]
    fn test_llm_fn_body() {
        let source = r#"model: "gpt-4o", temperature: 0.7"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::Model,
            TokenKind::Colon,
            TokenKind::StringLiteral, // "gpt-4o"
            TokenKind::Comma,
            TokenKind::Temperature,
            TokenKind::Colon,
            TokenKind::FloatLiteral, // 0.7
        ]);
    }

    #[test]
    fn test_decorator() {
        let source = r#"@input_guard(pii_detection)"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::At,
            TokenKind::Ident, // input_guard
            TokenKind::LParen,
            TokenKind::Ident, // pii_detection
            TokenKind::RParen,
        ]);
    }

    #[test]
    fn test_generic_types() {
        let source = r#"list<string> map<string, int> optional<Message>"#;
        assert_eq!(tokenize(source), vec![
            TokenKind::ListType,
            TokenKind::Lt,
            TokenKind::StringType,
            TokenKind::Gt,
            TokenKind::MapType,
            TokenKind::Lt,
            TokenKind::StringType,
            TokenKind::Comma,
            TokenKind::IntType,
            TokenKind::Gt,
            TokenKind::OptionalType,
            TokenKind::Lt,
            TokenKind::Ident, // Message
            TokenKind::Gt,
        ]);
    }

    #[test]
    fn test_error_on_invalid_char() {
        let tokens: Vec<_> = Lexer::new("fn $ test").collect();
        assert!(tokens.iter().any(|t| t.is_err()));
    }

    #[test]
    fn test_token_spans() {
        let source = "fn test";
        let tokens: Vec<_> = Lexer::new(source)
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(tokens[0].span, 0..2);
        assert_eq!(tokens[0].text, "fn");
        assert_eq!(tokens[1].span, 3..7);
        assert_eq!(tokens[1].text, "test");
    }
}
