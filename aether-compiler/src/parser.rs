//! Aether Parser
//!
//! Recursive descent parser for the Aether language. Produces an AST from tokens.
//!
//! # Parsing Strategy
//!
//! - Recursive descent with predictive parsing
//! - No backtracking required (Aether grammar is LL(1)-ish)
//! - Produces detailed error messages with span information

use crate::ast::*;
use crate::lexer::{LexError, Lexer, Token, TokenKind};
use thiserror::Error;

/// Parser error with location information
#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found} at {span:?}")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: Span,
    },

    #[error("unexpected end of file: expected {expected}")]
    UnexpectedEof { expected: String },

    #[error("lexer error: {0}")]
    LexError(#[from] LexError),

    #[error("invalid integer literal: {text} at {span:?}")]
    InvalidInt { text: String, span: Span },

    #[error("invalid float literal: {text} at {span:?}")]
    InvalidFloat { text: String, span: Span },

    #[error("invalid string literal at {span:?}")]
    InvalidString { span: Span },
}

/// Result type for parsing operations
pub type ParseResult<T> = Result<T, ParseError>;

/// Aether parser
pub struct Parser<'src> {
    tokens: Vec<Token<'src>>,
    pos: usize,
}

impl<'src> Parser<'src> {
    /// Create a new parser from source code
    pub fn new(source: &'src str) -> ParseResult<Self> {
        let tokens = Lexer::new(source).tokenize_filtered()?;
        Ok(Self { tokens, pos: 0 })
    }

    /// Create a parser from pre-lexed tokens
    pub fn from_tokens(tokens: Vec<Token<'src>>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse the entire program
    pub fn parse_program(&mut self) -> ParseResult<Program> {
        let mut items = Vec::new();

        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }

        Ok(Program::new(items))
    }

    /// Parse a single top-level item
    fn parse_item(&mut self) -> ParseResult<Item> {
        // Handle decorators
        let decorators = self.parse_decorators()?;

        match self.peek_kind() {
            Some(TokenKind::Llm) => {
                let mut llm_fn = self.parse_llm_fn()?;
                llm_fn.decorators = decorators;
                Ok(Item::LlmFn(llm_fn))
            }
            Some(TokenKind::Fn) => {
                let mut func = self.parse_function()?;
                func.decorators = decorators;
                Ok(Item::Function(func))
            }
            Some(TokenKind::Flow) => {
                let mut flow = self.parse_flow()?;
                flow.decorators = decorators;
                Ok(Item::Flow(flow))
            }
            Some(TokenKind::Struct) => Ok(Item::Struct(self.parse_struct()?)),
            Some(TokenKind::Enum) => Ok(Item::Enum(self.parse_enum()?)),
            Some(TokenKind::Context) => Ok(Item::Context(self.parse_context()?)),
            Some(TokenKind::Type) => Ok(Item::TypeAlias(self.parse_type_alias()?)),
            Some(TokenKind::Test) => Ok(Item::Test(self.parse_test()?)),
            Some(TokenKind::Import) => Ok(Item::Import(self.parse_import()?)),
            Some(_) => {
                let tok = self.peek().unwrap();
                Err(ParseError::UnexpectedToken {
                    expected: "item (llm fn, fn, flow, struct, enum, context, type, test, import)"
                        .to_string(),
                    found: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })
            }
            None => Err(ParseError::UnexpectedEof {
                expected: "item".to_string(),
            }),
        }
    }

    // =========================================================================
    // Decorators
    // =========================================================================

    fn parse_decorators(&mut self) -> ParseResult<Vec<Decorator>> {
        let mut decorators = Vec::new();

        while self.check(TokenKind::At) {
            decorators.push(self.parse_decorator()?);
        }

        Ok(decorators)
    }

    fn parse_decorator(&mut self) -> ParseResult<Decorator> {
        let start = self.expect(TokenKind::At)?.span.start;
        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        let args = if self.check(TokenKind::LParen) {
            self.advance();
            let args = self.parse_comma_separated(|p| p.parse_expr())?;
            self.expect(TokenKind::RParen)?;
            args
        } else {
            Vec::new()
        };

        let end = self.prev_span_end();
        Ok(Decorator {
            name,
            args,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // LLM Function
    // =========================================================================

    fn parse_llm_fn(&mut self) -> ParseResult<LlmFnDef> {
        let start = self.expect(TokenKind::Llm)?.span.start;
        self.expect(TokenKind::Fn)?;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        let body = self.parse_llm_fn_body()?;

        let end = self.prev_span_end();

        Ok(LlmFnDef {
            name,
            params,
            return_type,
            decorators: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_llm_fn_body(&mut self) -> ParseResult<LlmFnBody> {
        let start = self.expect(TokenKind::LBrace)?.span.start;

        let mut model = None;
        let mut temperature = None;
        let mut max_tokens = None;
        let mut system_prompt = None;
        let mut user_prompt = None;
        let mut prompt = None;

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            match self.peek_kind() {
                Some(TokenKind::Model) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let val_tok = self.expect(TokenKind::StringLiteral)?;
                    model = Some(Spanned::new(
                        self.parse_string_content(val_tok.text)?,
                        Span::from_range(val_tok.span),
                    ));
                }
                Some(TokenKind::Temperature) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let val = self.parse_number_as_float()?;
                    temperature = Some(val);
                }
                Some(TokenKind::MaxTokens) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let val = self.parse_number_as_int()?;
                    max_tokens = Some(Spanned::new(
                        val.node as u32,
                        val.span,
                    ));
                }
                Some(TokenKind::System) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let template = self.parse_string_template()?;
                    system_prompt = Some(template);
                }
                Some(TokenKind::User) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let template = self.parse_string_template()?;
                    user_prompt = Some(template);
                }
                Some(TokenKind::Prompt) => {
                    self.advance();
                    self.expect(TokenKind::Colon)?;
                    let template = self.parse_string_template()?;
                    prompt = Some(template);
                }
                _ => {
                    // Skip unknown fields with comma
                    self.advance();
                }
            }

            // Optional comma
            self.check_and_consume(TokenKind::Comma);
        }

        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(LlmFnBody {
            model,
            temperature,
            max_tokens,
            system_prompt,
            user_prompt,
            prompt,
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_string_template(&mut self) -> ParseResult<Spanned<StringTemplate>> {
        let tok = self.expect(TokenKind::StringLiteral)?;
        let span = Span::from_range(tok.span.clone());
        let content = self.parse_string_content(tok.text)?;

        // Parse template variables from the content
        let parts = self.parse_template_parts(&content);

        Ok(Spanned::new(StringTemplate::new(parts), span))
    }

    fn parse_template_parts(&self, content: &str) -> Vec<TemplatePart> {
        let mut parts = Vec::new();
        let mut current_literal = String::new();
        let mut chars = content.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'

                // Flush literal
                if !current_literal.is_empty() {
                    parts.push(TemplatePart::Literal(current_literal.clone()));
                    current_literal.clear();
                }

                // Parse variable name
                let mut var_name = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc == '}' {
                        break;
                    }
                    var_name.push(chars.next().unwrap());
                }

                // Consume "}}"
                if chars.next() == Some('}') && chars.peek() == Some(&'}') {
                    chars.next();
                }

                parts.push(TemplatePart::Variable(var_name.trim().to_string()));
            } else {
                current_literal.push(c);
            }
        }

        if !current_literal.is_empty() {
            parts.push(TemplatePart::Literal(current_literal));
        }

        if parts.is_empty() {
            parts.push(TemplatePart::Literal(String::new()));
        }

        parts
    }

    // =========================================================================
    // Regular Function
    // =========================================================================

    fn parse_function(&mut self) -> ParseResult<FunctionDef> {
        let start = self.expect(TokenKind::Fn)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        let params = self.parse_params()?;
        let return_type = if self.check(TokenKind::Arrow) {
            Some(self.parse_return_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let end = self.prev_span_end();

        Ok(FunctionDef {
            name,
            params,
            return_type,
            decorators: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // Flow
    // =========================================================================

    fn parse_flow(&mut self) -> ParseResult<FlowDef> {
        let start = self.expect(TokenKind::Flow)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        let body = self.parse_block()?;

        let end = self.prev_span_end();

        Ok(FlowDef {
            name,
            params,
            return_type,
            decorators: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // Struct
    // =========================================================================

    fn parse_struct(&mut self) -> ParseResult<StructDef> {
        let start = self.expect(TokenKind::Struct)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_fields()?;
        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(StructDef {
            name,
            fields,
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_fields(&mut self) -> ParseResult<Vec<Field>> {
        let mut fields = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let field_start = self.current_span_start();
            let name_tok = self.expect(TokenKind::Ident)?;
            let name = Spanned::new(
                name_tok.text.to_string(),
                Span::from_range(name_tok.span.clone()),
            );

            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let field_end = self.prev_span_end();

            fields.push(Field {
                name,
                ty,
                span: Span::new(field_start, field_end),
            });

            // Optional comma
            self.check_and_consume(TokenKind::Comma);
        }

        Ok(fields)
    }

    // =========================================================================
    // Enum
    // =========================================================================

    fn parse_enum(&mut self) -> ParseResult<EnumDef> {
        let start = self.expect(TokenKind::Enum)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        self.expect(TokenKind::LBrace)?;
        let variants = self.parse_enum_variants()?;
        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(EnumDef {
            name,
            variants,
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_enum_variants(&mut self) -> ParseResult<Vec<EnumVariant>> {
        let mut variants = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let name_tok = self.expect(TokenKind::Ident)?;
            let span = Span::from_range(name_tok.span.clone());

            variants.push(EnumVariant {
                name: Spanned::new(name_tok.text.to_string(), span.clone()),
                span,
            });

            // Optional comma
            self.check_and_consume(TokenKind::Comma);
        }

        Ok(variants)
    }

    // =========================================================================
    // Context
    // =========================================================================

    fn parse_context(&mut self) -> ParseResult<ContextDef> {
        let start = self.expect(TokenKind::Context)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        self.expect(TokenKind::LBrace)?;
        let fields = self.parse_fields()?;
        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(ContextDef {
            name,
            fields,
            span: Span::new(start, end_tok.span.end),
        })
    }

    // =========================================================================
    // Type Alias
    // =========================================================================

    fn parse_type_alias(&mut self) -> ParseResult<TypeAliasDef> {
        let start = self.expect(TokenKind::Type)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span.clone()),
        );

        self.expect(TokenKind::Eq)?;
        let ty = self.parse_type()?;

        // Optional constraint: `where condition`
        let constraint = if self.check(TokenKind::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        let end = self.prev_span_end();

        Ok(TypeAliasDef {
            name,
            ty,
            constraint,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // Test
    // =========================================================================

    fn parse_test(&mut self) -> ParseResult<TestDef> {
        let start = self.expect(TokenKind::Test)?.span.start;

        let name_tok = self.expect(TokenKind::StringLiteral)?;
        let name = Spanned::new(
            self.parse_string_content(name_tok.text)?,
            Span::from_range(name_tok.span),
        );

        let body = self.parse_block()?;
        let end = self.prev_span_end();

        Ok(TestDef {
            name,
            body,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // Import
    // =========================================================================

    fn parse_import(&mut self) -> ParseResult<ImportDef> {
        let start = self.expect(TokenKind::Import)?.span.start;

        // Import names (could be single or braced list)
        let names = if self.check(TokenKind::LBrace) {
            self.advance();
            let names = self.parse_import_names()?;
            self.expect(TokenKind::RBrace)?;
            names
        } else {
            let name_tok = self.expect(TokenKind::Ident)?;
            vec![ImportName {
                name: Spanned::new(
                    name_tok.text.to_string(),
                    Span::from_range(name_tok.span),
                ),
                alias: None,
            }]
        };

        self.expect(TokenKind::From)?;

        let path_tok = self.expect(TokenKind::StringLiteral)?;
        let path = Spanned::new(
            self.parse_string_content(path_tok.text)?,
            Span::from_range(path_tok.span),
        );

        let end = self.prev_span_end();

        Ok(ImportDef {
            names,
            path,
            span: Span::new(start, end),
        })
    }

    fn parse_import_names(&mut self) -> ParseResult<Vec<ImportName>> {
        let mut names = Vec::new();

        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let name_tok = self.expect(TokenKind::Ident)?;
            let name = Spanned::new(
                name_tok.text.to_string(),
                Span::from_range(name_tok.span),
            );

            let alias = if self.check(TokenKind::As) {
                self.advance();
                let alias_tok = self.expect(TokenKind::Ident)?;
                Some(Spanned::new(
                    alias_tok.text.to_string(),
                    Span::from_range(alias_tok.span),
                ))
            } else {
                None
            };

            names.push(ImportName { name, alias });
            self.check_and_consume(TokenKind::Comma);
        }

        Ok(names)
    }

    // =========================================================================
    // Parameters and Types
    // =========================================================================

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();
        while !self.check(TokenKind::RParen) && !self.is_at_end() {
            let param_start = self.current_span_start();
            let name_tok = self.expect(TokenKind::Ident)?;
            let name = Spanned::new(
                name_tok.text.to_string(),
                Span::from_range(name_tok.span.clone()),
            );

            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let param_end = self.prev_span_end();

            params.push(Param {
                name,
                ty,
                span: Span::new(param_start, param_end),
            });

            if !self.check(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_return_type(&mut self) -> ParseResult<Type> {
        self.expect(TokenKind::Arrow)?;
        self.parse_type()
    }

    fn parse_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span_start();

        match self.peek_kind() {
            Some(TokenKind::StringType) => {
                self.advance();
                Ok(Type::Primitive {
                    kind: PrimitiveType::String,
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::IntType) => {
                self.advance();
                Ok(Type::Primitive {
                    kind: PrimitiveType::Int,
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::FloatType) => {
                self.advance();
                Ok(Type::Primitive {
                    kind: PrimitiveType::Float,
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::BoolType) => {
                self.advance();
                Ok(Type::Primitive {
                    kind: PrimitiveType::Bool,
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::ListType) => {
                self.advance();
                self.expect(TokenKind::Lt)?;
                let element = self.parse_type()?;
                self.expect(TokenKind::Gt)?;
                Ok(Type::List {
                    element: Box::new(element),
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::MapType) => {
                self.advance();
                self.expect(TokenKind::Lt)?;
                let key = self.parse_type()?;
                self.expect(TokenKind::Comma)?;
                let value = self.parse_type()?;
                self.expect(TokenKind::Gt)?;
                Ok(Type::Map {
                    key: Box::new(key),
                    value: Box::new(value),
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::OptionalType) => {
                self.advance();
                self.expect(TokenKind::Lt)?;
                let inner = self.parse_type()?;
                self.expect(TokenKind::Gt)?;
                Ok(Type::Optional {
                    inner: Box::new(inner),
                    span: Span::new(start, self.prev_span_end()),
                })
            }
            Some(TokenKind::Ident) => {
                let tok = self.advance().unwrap();
                Ok(Type::Named {
                    name: tok.text.to_string(),
                    span: Span::from_range(tok.span),
                })
            }
            Some(_) => {
                let tok = self.peek().unwrap();
                Err(ParseError::UnexpectedToken {
                    expected: "type".to_string(),
                    found: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })
            }
            None => Err(ParseError::UnexpectedEof {
                expected: "type".to_string(),
            }),
        }
    }

    // =========================================================================
    // Block and Statements
    // =========================================================================

    fn parse_block(&mut self) -> ParseResult<Block> {
        let start = self.expect(TokenKind::LBrace)?.span.start;

        let mut stmts = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            stmts.push(self.parse_stmt()?);
        }

        let end_tok = self.expect(TokenKind::RBrace)?;

        Ok(Block {
            stmts,
            span: Span::new(start, end_tok.span.end),
        })
    }

    fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        match self.peek_kind() {
            Some(TokenKind::Let) => self.parse_let_stmt(),
            Some(TokenKind::Return) => self.parse_return_stmt(),
            Some(TokenKind::If) => self.parse_if_stmt(),
            Some(TokenKind::For) => self.parse_for_stmt(),
            Some(TokenKind::While) => self.parse_while_stmt(),
            Some(TokenKind::Try) => self.parse_try_stmt(),
            Some(TokenKind::Assert) => self.parse_assert_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_let_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::Let)?.span.start;

        let name_tok = self.expect(TokenKind::Ident)?;
        let name = Spanned::new(
            name_tok.text.to_string(),
            Span::from_range(name_tok.span),
        );

        let ty = if self.check(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.expect(TokenKind::Semicolon)?;

        let end = self.prev_span_end();

        Ok(Stmt::Let {
            name,
            ty,
            value,
            span: Span::new(start, end),
        })
    }

    fn parse_return_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::Return)?.span.start;

        let value = if !self.check(TokenKind::Semicolon) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.expect(TokenKind::Semicolon)?;
        let end = self.prev_span_end();

        Ok(Stmt::Return {
            value,
            span: Span::new(start, end),
        })
    }

    fn parse_if_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::If)?.span.start;

        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;

        let else_block = if self.check(TokenKind::Else) {
            self.advance();
            Some(self.parse_block()?)
        } else {
            None
        };

        let end = self.prev_span_end();

        Ok(Stmt::If {
            condition,
            then_block,
            else_block,
            span: Span::new(start, end),
        })
    }

    fn parse_for_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::For)?.span.start;

        let var_tok = self.expect(TokenKind::Ident)?;
        let var = Spanned::new(
            var_tok.text.to_string(),
            Span::from_range(var_tok.span),
        );

        self.expect(TokenKind::In)?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;

        let end = self.prev_span_end();

        Ok(Stmt::For {
            var,
            iter,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_while_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::While)?.span.start;

        let condition = self.parse_expr()?;
        let body = self.parse_block()?;

        let end = self.prev_span_end();

        Ok(Stmt::While {
            condition,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_try_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::Try)?.span.start;

        let body = self.parse_block()?;

        let mut catches = Vec::new();
        while self.check(TokenKind::Catch) {
            catches.push(self.parse_catch_clause()?);
        }

        let end = self.prev_span_end();

        Ok(Stmt::Try {
            body,
            catches,
            span: Span::new(start, end),
        })
    }

    fn parse_catch_clause(&mut self) -> ParseResult<CatchClause> {
        let start = self.expect(TokenKind::Catch)?.span.start;

        // Optional error type and binding
        let (error_type, binding) = if self.check(TokenKind::LParen) {
            self.advance();
            let binding_tok = self.expect(TokenKind::Ident)?;
            let binding = Spanned::new(
                binding_tok.text.to_string(),
                Span::from_range(binding_tok.span),
            );

            let error_type = if self.check(TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            self.expect(TokenKind::RParen)?;
            (error_type, Some(binding))
        } else {
            (None, None)
        };

        let body = self.parse_block()?;
        let end = self.prev_span_end();

        Ok(CatchClause {
            error_type,
            binding,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_assert_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.expect(TokenKind::Assert)?.span.start;

        let condition = self.parse_expr()?;

        let message = if self.check(TokenKind::Comma) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.expect(TokenKind::Semicolon)?;
        let end = self.prev_span_end();

        Ok(Stmt::Assert {
            condition,
            message,
            span: Span::new(start, end),
        })
    }

    fn parse_expr_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.current_span_start();
        let expr = self.parse_expr()?;
        self.expect(TokenKind::Semicolon)?;
        let end = self.prev_span_end();

        Ok(Stmt::Expr {
            expr,
            span: Span::new(start, end),
        })
    }

    // =========================================================================
    // Expressions (Pratt parser style)
    // =========================================================================

    fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_and_expr()?;

        while self.check(TokenKind::Or) {
            self.advance();
            let right = self.parse_and_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_equality_expr()?;

        while self.check(TokenKind::And) {
            self.advance();
            let right = self.parse_equality_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_equality_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_comparison_expr()?;

        while let Some(op) = self.match_equality_op() {
            let right = self.parse_comparison_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn match_equality_op(&mut self) -> Option<BinaryOp> {
        if self.check(TokenKind::EqEq) {
            self.advance();
            Some(BinaryOp::Eq)
        } else if self.check(TokenKind::NotEq) {
            self.advance();
            Some(BinaryOp::NotEq)
        } else {
            None
        }
    }

    fn parse_comparison_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_additive_expr()?;

        while let Some(op) = self.match_comparison_op() {
            let right = self.parse_additive_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn match_comparison_op(&mut self) -> Option<BinaryOp> {
        match self.peek_kind() {
            Some(TokenKind::Lt) => {
                self.advance();
                Some(BinaryOp::Lt)
            }
            Some(TokenKind::Gt) => {
                self.advance();
                Some(BinaryOp::Gt)
            }
            Some(TokenKind::LtEq) => {
                self.advance();
                Some(BinaryOp::LtEq)
            }
            Some(TokenKind::GtEq) => {
                self.advance();
                Some(BinaryOp::GtEq)
            }
            _ => None,
        }
    }

    fn parse_additive_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_multiplicative_expr()?;

        while let Some(op) = self.match_additive_op() {
            let right = self.parse_multiplicative_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn match_additive_op(&mut self) -> Option<BinaryOp> {
        if self.check(TokenKind::Plus) {
            self.advance();
            Some(BinaryOp::Add)
        } else if self.check(TokenKind::Minus) {
            self.advance();
            Some(BinaryOp::Sub)
        } else {
            None
        }
    }

    fn parse_multiplicative_expr(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_unary_expr()?;

        while let Some(op) = self.match_multiplicative_op() {
            let right = self.parse_unary_expr()?;
            let span = left.span().merge(right.span());
            left = Expr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn match_multiplicative_op(&mut self) -> Option<BinaryOp> {
        match self.peek_kind() {
            Some(TokenKind::Star) => {
                self.advance();
                Some(BinaryOp::Mul)
            }
            Some(TokenKind::Slash) => {
                self.advance();
                Some(BinaryOp::Div)
            }
            Some(TokenKind::Percent) => {
                self.advance();
                Some(BinaryOp::Mod)
            }
            _ => None,
        }
    }

    fn parse_unary_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span_start();

        if self.check(TokenKind::Not) {
            self.advance();
            let operand = self.parse_unary_expr()?;
            let span = Span::new(start, operand.span().end);
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(operand),
                span,
            });
        }

        if self.check(TokenKind::Minus) {
            // Check if this is a negative number literal or unary minus
            if let Some(next) = self.peek_next() {
                if matches!(next.kind, TokenKind::IntLiteral | TokenKind::FloatLiteral) {
                    // Let parse_postfix_expr handle negative literals
                    return self.parse_postfix_expr();
                }
            }
            self.advance();
            let operand = self.parse_unary_expr()?;
            let span = Span::new(start, operand.span().end);
            return Ok(Expr::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
                span,
            });
        }

        self.parse_postfix_expr()
    }

    fn parse_postfix_expr(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            if self.check(TokenKind::LParen) {
                // Function call
                self.advance();
                let args = self.parse_comma_separated(|p| p.parse_expr())?;
                self.expect(TokenKind::RParen)?;
                let span = expr.span().merge(&Span::new(
                    self.prev_span_end() - 1,
                    self.prev_span_end(),
                ));
                expr = Expr::Call {
                    func: Box::new(expr),
                    args,
                    span,
                };
            } else if self.check(TokenKind::Dot) {
                self.advance();
                let field_tok = self.expect(TokenKind::Ident)?;
                let field = Spanned::new(
                    field_tok.text.to_string(),
                    Span::from_range(field_tok.span.clone()),
                );

                // Check if it's a method call
                if self.check(TokenKind::LParen) {
                    self.advance();
                    let args = self.parse_comma_separated(|p| p.parse_expr())?;
                    self.expect(TokenKind::RParen)?;
                    let span = expr.span().merge(&Span::new(
                        self.prev_span_end() - 1,
                        self.prev_span_end(),
                    ));
                    expr = Expr::MethodCall {
                        receiver: Box::new(expr),
                        method: field,
                        args,
                        span,
                    };
                } else {
                    let span = expr.span().merge(&field.span);
                    expr = Expr::FieldAccess {
                        object: Box::new(expr),
                        field,
                        span,
                    };
                }
            } else if self.check(TokenKind::LBracket) {
                // Index access
                self.advance();
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket)?;
                let span = expr.span().merge(&Span::new(
                    self.prev_span_end() - 1,
                    self.prev_span_end(),
                ));
                expr = Expr::Index {
                    object: Box::new(expr),
                    index: Box::new(index),
                    span,
                };
            } else if self.check(TokenKind::DoubleColon) {
                // Enum variant: Sentiment::Positive
                self.advance();
                let variant_tok = self.expect(TokenKind::Ident)?;

                // Get the enum name from the previous expression
                if let Expr::Ident { name: enum_name, span: start_span } = &expr {
                    let span = start_span.merge(&Span::from_range(variant_tok.span.clone()));
                    expr = Expr::EnumVariant {
                        enum_name: enum_name.clone(),
                        variant: variant_tok.text.to_string(),
                        span,
                    };
                } else {
                    return Err(ParseError::UnexpectedToken {
                        expected: "enum name before ::".to_string(),
                        found: format!("{:?}", expr),
                        span: expr.span().clone(),
                    });
                }
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> ParseResult<Expr> {
        match self.peek_kind() {
            Some(TokenKind::IntLiteral) => {
                let tok = self.advance().unwrap();
                let value: i64 = tok.text.parse().map_err(|_| ParseError::InvalidInt {
                    text: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })?;
                Ok(Expr::Literal {
                    value: Literal::Int(value),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::FloatLiteral) => {
                let tok = self.advance().unwrap();
                let value: f64 = tok.text.parse().map_err(|_| ParseError::InvalidFloat {
                    text: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })?;
                Ok(Expr::Literal {
                    value: Literal::Float(value),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::StringLiteral) => {
                let tok = self.advance().unwrap();
                let content = self.parse_string_content(tok.text)?;
                Ok(Expr::Literal {
                    value: Literal::String(content),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::True) => {
                let tok = self.advance().unwrap();
                Ok(Expr::Literal {
                    value: Literal::Bool(true),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::False) => {
                let tok = self.advance().unwrap();
                Ok(Expr::Literal {
                    value: Literal::Bool(false),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::Ident) => {
                let tok = self.advance().unwrap();
                let name = tok.text.to_string();
                let span = Span::from_range(tok.span);

                // Check if this is a struct literal
                if self.check(TokenKind::LBrace) {
                    return self.parse_struct_literal(name, span);
                }

                Ok(Expr::Ident { name, span })
            }
            Some(TokenKind::LParen) => {
                let start = self.advance().unwrap().span.start;
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                let end = self.prev_span_end();
                Ok(Expr::Paren {
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Some(TokenKind::LBracket) => self.parse_list_literal(),
            Some(TokenKind::Await) => {
                let start = self.advance().unwrap().span.start;
                let expr = self.parse_unary_expr()?;
                let end = expr.span().end;
                Ok(Expr::Await {
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Some(TokenKind::Match) => self.parse_match_expr(),
            Some(_) => {
                let tok = self.peek().unwrap();
                Err(ParseError::UnexpectedToken {
                    expected: "expression".to_string(),
                    found: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })
            }
            None => Err(ParseError::UnexpectedEof {
                expected: "expression".to_string(),
            }),
        }
    }

    fn parse_struct_literal(&mut self, name: String, name_span: Span) -> ParseResult<Expr> {
        let start = name_span.start;
        self.expect(TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let field_start = self.current_span_start();
            let field_name_tok = self.expect(TokenKind::Ident)?;
            let field_name = Spanned::new(
                field_name_tok.text.to_string(),
                Span::from_range(field_name_tok.span),
            );

            // Check for shorthand: `{ x }` instead of `{ x: x }`
            let value = if self.check(TokenKind::Colon) {
                self.advance();
                self.parse_expr()?
            } else {
                Expr::Ident {
                    name: field_name.node.clone(),
                    span: field_name.span.clone(),
                }
            };

            let field_end = self.prev_span_end();
            fields.push(FieldInit {
                name: field_name,
                value,
                span: Span::new(field_start, field_end),
            });

            self.check_and_consume(TokenKind::Comma);
        }

        self.expect(TokenKind::RBrace)?;
        let end = self.prev_span_end();

        Ok(Expr::StructLiteral {
            name: Spanned::new(name, name_span),
            fields,
            span: Span::new(start, end),
        })
    }

    fn parse_list_literal(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenKind::LBracket)?.span.start;
        let elements = self.parse_comma_separated(|p| p.parse_expr())?;
        self.expect(TokenKind::RBracket)?;
        let end = self.prev_span_end();

        Ok(Expr::List {
            elements,
            span: Span::new(start, end),
        })
    }

    fn parse_match_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(TokenKind::Match)?.span.start;
        let expr = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let arm_start = self.current_span_start();
            let pattern = self.parse_pattern()?;
            self.expect(TokenKind::Arrow)?;
            let body = self.parse_expr()?;
            let arm_end = self.prev_span_end();

            arms.push(MatchArm {
                pattern,
                body,
                span: Span::new(arm_start, arm_end),
            });

            self.check_and_consume(TokenKind::Comma);
        }

        self.expect(TokenKind::RBrace)?;
        let end = self.prev_span_end();

        Ok(Expr::Match {
            expr: Box::new(expr),
            arms,
            span: Span::new(start, end),
        })
    }

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        match self.peek_kind() {
            Some(TokenKind::Ident) => {
                let tok = self.advance().unwrap();
                let name = tok.text.to_string();
                let span = Span::from_range(tok.span);

                // Check for enum variant pattern
                if self.check(TokenKind::DoubleColon) {
                    self.advance();
                    let variant_tok = self.expect(TokenKind::Ident)?;
                    let variant_span = Span::from_range(variant_tok.span);
                    return Ok(Pattern::EnumVariant {
                        enum_name: Some(name),
                        variant: variant_tok.text.to_string(),
                        span: span.merge(&variant_span),
                    });
                }

                // Check if it looks like a variant (starts with uppercase)
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    Ok(Pattern::EnumVariant {
                        enum_name: None,
                        variant: name,
                        span,
                    })
                } else if name == "_" {
                    Ok(Pattern::Wildcard { span })
                } else {
                    Ok(Pattern::Ident { name, span })
                }
            }
            Some(TokenKind::IntLiteral) => {
                let tok = self.advance().unwrap();
                let value: i64 = tok.text.parse().map_err(|_| ParseError::InvalidInt {
                    text: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })?;
                Ok(Pattern::Literal {
                    value: Literal::Int(value),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::StringLiteral) => {
                let tok = self.advance().unwrap();
                let content = self.parse_string_content(tok.text)?;
                Ok(Pattern::Literal {
                    value: Literal::String(content),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::True) => {
                let tok = self.advance().unwrap();
                Ok(Pattern::Literal {
                    value: Literal::Bool(true),
                    span: Span::from_range(tok.span),
                })
            }
            Some(TokenKind::False) => {
                let tok = self.advance().unwrap();
                Ok(Pattern::Literal {
                    value: Literal::Bool(false),
                    span: Span::from_range(tok.span),
                })
            }
            _ => {
                let tok = self.peek();
                match tok {
                    Some(t) => Err(ParseError::UnexpectedToken {
                        expected: "pattern".to_string(),
                        found: t.text.to_string(),
                        span: Span::from_range(t.span.clone()),
                    }),
                    None => Err(ParseError::UnexpectedEof {
                        expected: "pattern".to_string(),
                    }),
                }
            }
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn peek(&self) -> Option<&Token<'src>> {
        self.tokens.get(self.pos)
    }

    fn peek_kind(&self) -> Option<TokenKind> {
        self.peek().map(|t| t.kind)
    }

    fn peek_next(&self) -> Option<&Token<'src>> {
        self.tokens.get(self.pos + 1)
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.peek_kind() == Some(kind)
    }

    fn check_and_consume(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn advance(&mut self) -> Option<Token<'src>> {
        if self.is_at_end() {
            None
        } else {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        }
    }

    fn expect(&mut self, kind: TokenKind) -> ParseResult<Token<'src>> {
        match self.peek() {
            Some(tok) if tok.kind == kind => Ok(self.advance().unwrap()),
            Some(tok) => Err(ParseError::UnexpectedToken {
                expected: kind.to_string(),
                found: tok.text.to_string(),
                span: Span::from_range(tok.span.clone()),
            }),
            None => Err(ParseError::UnexpectedEof {
                expected: kind.to_string(),
            }),
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn current_span_start(&self) -> usize {
        self.peek().map(|t| t.span.start).unwrap_or(0)
    }

    fn prev_span_end(&self) -> usize {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span.end
        } else {
            0
        }
    }

    fn parse_comma_separated<T, F>(&mut self, mut parse_fn: F) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        // Check for empty list (next token closes the list)
        if self.check(TokenKind::RParen) || self.check(TokenKind::RBracket) || self.check(TokenKind::RBrace) {
            return Ok(items);
        }

        items.push(parse_fn(self)?);

        while self.check(TokenKind::Comma) {
            self.advance();
            // Allow trailing comma
            if self.check(TokenKind::RParen) || self.check(TokenKind::RBracket) || self.check(TokenKind::RBrace) {
                break;
            }
            items.push(parse_fn(self)?);
        }

        Ok(items)
    }

    fn parse_string_content(&self, quoted: &str) -> ParseResult<String> {
        // Remove surrounding quotes
        if quoted.len() < 2 {
            return Ok(String::new());
        }

        let inner = &quoted[1..quoted.len() - 1];

        // Unescape the string
        let mut result = String::new();
        let mut chars = inner.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn parse_number_as_float(&mut self) -> ParseResult<Spanned<f64>> {
        match self.peek_kind() {
            Some(TokenKind::FloatLiteral) => {
                let tok = self.advance().unwrap();
                let value: f64 = tok.text.parse().map_err(|_| ParseError::InvalidFloat {
                    text: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })?;
                Ok(Spanned::new(value, Span::from_range(tok.span)))
            }
            Some(TokenKind::IntLiteral) => {
                let tok = self.advance().unwrap();
                let value: f64 = tok.text.parse().map_err(|_| ParseError::InvalidFloat {
                    text: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })?;
                Ok(Spanned::new(value, Span::from_range(tok.span)))
            }
            Some(_) => {
                let tok = self.peek().unwrap();
                Err(ParseError::UnexpectedToken {
                    expected: "number".to_string(),
                    found: tok.text.to_string(),
                    span: Span::from_range(tok.span.clone()),
                })
            }
            None => Err(ParseError::UnexpectedEof {
                expected: "number".to_string(),
            }),
        }
    }

    fn parse_number_as_int(&mut self) -> ParseResult<Spanned<i64>> {
        let tok = self.expect(TokenKind::IntLiteral)?;
        let value: i64 = tok.text.parse().map_err(|_| ParseError::InvalidInt {
            text: tok.text.to_string(),
            span: Span::from_range(tok.span.clone()),
        })?;
        Ok(Spanned::new(value, Span::from_range(tok.span)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> ParseResult<Program> {
        Parser::new(source)?.parse_program()
    }

    #[test]
    fn test_parse_enum() {
        let source = "enum Sentiment { Positive, Neutral, Negative }";
        let program = parse(source).unwrap();

        assert_eq!(program.items.len(), 1);
        if let Item::Enum(e) = &program.items[0] {
            assert_eq!(e.name.node, "Sentiment");
            assert_eq!(e.variants.len(), 3);
            assert_eq!(e.variants[0].name.node, "Positive");
            assert_eq!(e.variants[1].name.node, "Neutral");
            assert_eq!(e.variants[2].name.node, "Negative");
        } else {
            panic!("Expected enum");
        }
    }

    #[test]
    fn test_parse_struct() {
        let source = r#"
            struct Message {
                role: string,
                content: string,
                timestamp: int
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Struct(s) = &program.items[0] {
            assert_eq!(s.name.node, "Message");
            assert_eq!(s.fields.len(), 3);
            assert_eq!(s.fields[0].name.node, "role");
            assert_eq!(s.fields[1].name.node, "content");
            assert_eq!(s.fields[2].name.node, "timestamp");
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_llm_fn() {
        let source = r#"
            llm fn classify_sentiment(text: string) -> Sentiment {
                model: "gpt-4o",
                temperature: 0.3,
                prompt: "Classify the sentiment: {{text}}"
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::LlmFn(f) = &program.items[0] {
            assert_eq!(f.name.node, "classify_sentiment");
            assert_eq!(f.params.len(), 1);
            assert_eq!(f.params[0].name.node, "text");

            assert_eq!(f.body.model.as_ref().unwrap().node, "gpt-4o");
            assert!((f.body.temperature.as_ref().unwrap().node - 0.3).abs() < 0.01);

            let prompt = f.body.prompt.as_ref().unwrap();
            assert_eq!(prompt.node.parts.len(), 2);
        } else {
            panic!("Expected llm fn");
        }
    }

    #[test]
    fn test_parse_flow() {
        let source = r#"
            flow analyze(doc: string) -> Result {
                let summary = summarize(doc);
                return summary;
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Flow(f) = &program.items[0] {
            assert_eq!(f.name.node, "analyze");
            assert_eq!(f.params.len(), 1);
            assert_eq!(f.body.stmts.len(), 2);
        } else {
            panic!("Expected flow");
        }
    }

    #[test]
    fn test_parse_decorator() {
        let source = r#"
            @input_guard(pii_detection)
            llm fn safe_classify(text: string) -> Sentiment {
                model: "gpt-4o",
                prompt: "Classify: {{text}}"
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::LlmFn(f) = &program.items[0] {
            assert_eq!(f.decorators.len(), 1);
            assert_eq!(f.decorators[0].name.node, "input_guard");
        } else {
            panic!("Expected llm fn with decorator");
        }
    }

    #[test]
    fn test_parse_context() {
        let source = r#"
            context ConversationState {
                history: list<Message>,
                session_id: string
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Context(c) = &program.items[0] {
            assert_eq!(c.name.node, "ConversationState");
            assert_eq!(c.fields.len(), 2);
        } else {
            panic!("Expected context");
        }
    }

    #[test]
    fn test_parse_test_block() {
        let source = r#"
            test "sentiment_classification" {
                let result = classify("I love this");
                assert result == Sentiment::Positive;
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Test(t) = &program.items[0] {
            assert_eq!(t.name.node, "sentiment_classification");
            assert_eq!(t.body.stmts.len(), 2);
        } else {
            panic!("Expected test");
        }
    }

    #[test]
    fn test_parse_generic_types() {
        let source = r#"
            struct Container {
                items: list<string>,
                metadata: map<string, int>,
                optional_value: optional<float>
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Struct(s) = &program.items[0] {
            assert!(matches!(s.fields[0].ty, Type::List { .. }));
            assert!(matches!(s.fields[1].ty, Type::Map { .. }));
            assert!(matches!(s.fields[2].ty, Type::Optional { .. }));
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_expressions() {
        let source = r#"
            fn test_expr() {
                let a = 1 + 2 * 3;
                let b = x && y || z;
                let c = foo.bar.baz();
                let d = items[0];
            }
        "#;
        let program = parse(source).unwrap();

        if let Item::Function(f) = &program.items[0] {
            assert_eq!(f.body.stmts.len(), 4);
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_import() {
        let source = r#"import { classify, summarize } from "./utils.aether""#;
        let program = parse(source).unwrap();

        if let Item::Import(i) = &program.items[0] {
            assert_eq!(i.names.len(), 2);
            assert_eq!(i.names[0].name.node, "classify");
            assert_eq!(i.names[1].name.node, "summarize");
            assert_eq!(i.path.node, "./utils.aether");
        } else {
            panic!("Expected import");
        }
    }
}
