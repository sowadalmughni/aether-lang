//! Error message tests
//!
//! These tests verify that the compiler produces helpful error messages
//! for various kinds of malformed input.

use aether_compiler::{ParseError, Parser, SemanticAnalyzer, SemanticError};

fn parse(source: &str) -> Result<aether_compiler::Program, ParseError> {
    Parser::new(source)?.parse_program()
}

fn analyze(source: &str) -> Result<aether_compiler::SemanticContext, Vec<SemanticError>> {
    let program = parse(source).expect("Parse should succeed");
    SemanticAnalyzer::new().analyze(&program)
}

// =============================================================================
// Parser Error Tests
// =============================================================================

#[test]
fn test_missing_closing_brace() {
    let source = r#"
        enum Sentiment { Positive, Neutral
    "#;

    let result = parse(source);
    assert!(result.is_err());
}

#[test]
fn test_missing_type_annotation() {
    let source = r#"
        llm fn classify(text) -> Sentiment {
            model: "gpt-4o",
            prompt: "test"
        }
    "#;

    let result = parse(source);
    assert!(result.is_err());
}

#[test]
fn test_missing_return_type() {
    let source = r#"
        llm fn classify(text: string) {
            model: "gpt-4o",
            prompt: "test"
        }
    "#;

    let result = parse(source);
    assert!(result.is_err());
}

#[test]
fn test_invalid_token() {
    let source = r#"
        llm fn @classify(text: string) -> string {
            model: "gpt-4o",
            prompt: "test"
        }
    "#;

    let result = parse(source);
    assert!(result.is_err());
}

#[test]
fn test_unclosed_string() {
    let source = r#"
        llm fn classify(text: string) -> string {
            model: "gpt-4o,
            prompt: "test"
        }
    "#;

    let result = parse(source);
    assert!(result.is_err());
}

// =============================================================================
// Semantic Error Tests
// =============================================================================

#[test]
fn test_undefined_function_error() {
    let source = r#"
        flow broken(text: string) -> string {
            let result = nonexistent_function(text);
            return result;
        }
    "#;

    let result = analyze(source);
    assert!(result.is_err());

    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| {
        matches!(e, SemanticError::UndefinedFunction { fn_name, .. } if fn_name == "nonexistent_function")
    }));
}

#[test]
fn test_missing_model_error() {
    let source = r#"
        llm fn classify(text: string) -> string {
            prompt: "Classify: {{text}}"
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
        llm fn classify(text: string) -> string {
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

// =============================================================================
// Valid Input Tests (should not error)
// =============================================================================

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

    let result = parse(source);
    assert!(result.is_ok());
}

#[test]
fn test_valid_flow_parses() {
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
fn test_enum_with_associated_data_parses() {
    let source = r#"
        enum Result {
            Success,
            Error(string),
            Data(int, string)
        }
    "#;

    let result = parse(source);
    assert!(result.is_ok());

    let program = result.unwrap();
    if let aether_compiler::Item::Enum(e) = &program.items[0] {
        assert_eq!(e.variants.len(), 3);
        assert!(e.variants[0].data.is_none());
        assert!(e.variants[1].data.is_some());
        assert_eq!(e.variants[1].data.as_ref().unwrap().len(), 1);
        assert_eq!(e.variants[2].data.as_ref().unwrap().len(), 2);
    } else {
        panic!("Expected enum");
    }
}

#[test]
fn test_complex_types_parse() {
    let source = r#"
        struct Container {
            items: list<string>,
            metadata: map<string, int>,
            optional_field: optional<string>
        }
    "#;

    let result = parse(source);
    assert!(result.is_ok());
}

#[test]
fn test_template_interpolation_parses() {
    let source = r#"
        llm fn greet(name: string, ctx: string) -> string {
            model: "gpt-4o",
            system: "You are a helpful assistant",
            prompt: "Hello {{name}}, here is some context: {{ctx}}"
        }
    "#;

    let result = parse(source);
    assert!(result.is_ok());
}
