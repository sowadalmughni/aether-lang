//! Tests for the `run` command URL configuration
//!
//! Tests URL precedence: flag > env > default
//! Tests URL validation for http/https schemes

use std::env;
use url::Url;

/// Default runtime URL (mirrors main.rs constant)
const DEFAULT_RUNTIME_URL: &str = "http://127.0.0.1:3000";

/// Environment variable name (mirrors main.rs constant)
const RUNTIME_URL_ENV: &str = "AETHER_RUNTIME_URL";

/// Resolve runtime URL with precedence: flag > env > default
fn resolve_runtime_url(flag_url: Option<String>) -> Result<Url, String> {
    let url_str = if let Some(url) = flag_url {
        url
    } else if let Ok(env_url) = env::var(RUNTIME_URL_ENV) {
        env_url
    } else {
        DEFAULT_RUNTIME_URL.to_string()
    };

    validate_runtime_url(&url_str)
}

/// Validate that a URL is a valid http or https URL
fn validate_runtime_url(url_str: &str) -> Result<Url, String> {
    let url =
        Url::parse(url_str).map_err(|e| format!("Invalid runtime URL '{}': {}", url_str, e))?;

    match url.scheme() {
        "http" | "https" => Ok(url),
        scheme => Err(format!(
            "Invalid runtime URL scheme '{}'. Must be http or https.",
            scheme
        )),
    }
}

// --- Unit tests for URL validation ---

#[test]
fn test_valid_http_url() {
    let result = validate_runtime_url("http://localhost:3000");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().as_str(), "http://localhost:3000/");
}

#[test]
fn test_valid_https_url() {
    let result = validate_runtime_url("https://api.example.com:8080");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().as_str(), "https://api.example.com:8080/");
}

#[test]
fn test_valid_http_with_path() {
    let result = validate_runtime_url("http://127.0.0.1:3000/api/v1");
    assert!(result.is_ok());
}

#[test]
fn test_invalid_scheme_ftp() {
    let result = validate_runtime_url("ftp://example.com");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Must be http or https"));
}

#[test]
fn test_invalid_scheme_file() {
    let result = validate_runtime_url("file:///path/to/file");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Must be http or https"));
}

#[test]
fn test_invalid_url_format() {
    let result = validate_runtime_url("not-a-valid-url");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid runtime URL"));
}

#[test]
fn test_empty_url() {
    let result = validate_runtime_url("");
    assert!(result.is_err());
}

// --- Unit tests for URL precedence ---

#[test]
fn test_precedence_flag_overrides_all() {
    // Set env var
    env::set_var(RUNTIME_URL_ENV, "http://env-value:9000");

    // Flag should take precedence
    let result = resolve_runtime_url(Some("http://flag-value:8000".to_string()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().host_str(), Some("flag-value"));

    // Clean up
    env::remove_var(RUNTIME_URL_ENV);
}

#[test]
fn test_precedence_env_overrides_default() {
    // Set env var
    env::set_var(RUNTIME_URL_ENV, "http://env-value:9000");

    // No flag, should use env
    let result = resolve_runtime_url(None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().host_str(), Some("env-value"));

    // Clean up
    env::remove_var(RUNTIME_URL_ENV);
}

#[test]
fn test_precedence_default_when_nothing_set() {
    // Make sure env is not set
    env::remove_var(RUNTIME_URL_ENV);

    // No flag, no env, should use default
    let result = resolve_runtime_url(None);
    assert!(result.is_ok());
    let url = result.unwrap();
    assert_eq!(url.host_str(), Some("127.0.0.1"));
    assert_eq!(url.port(), Some(3000));
}

#[test]
fn test_precedence_flag_wins_over_env() {
    env::set_var(RUNTIME_URL_ENV, "http://should-not-use:1111");

    let result = resolve_runtime_url(Some("http://should-use:2222".to_string()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().host_str(), Some("should-use"));

    env::remove_var(RUNTIME_URL_ENV);
}

// --- Integration test: build execute URL ---

#[test]
fn test_build_execute_url() {
    let base = validate_runtime_url("http://localhost:3000").unwrap();
    let execute_url = base.join("/execute").unwrap();
    assert_eq!(execute_url.as_str(), "http://localhost:3000/execute");
}

#[test]
fn test_build_execute_url_with_trailing_slash() {
    let base = validate_runtime_url("http://localhost:3000/").unwrap();
    let execute_url = base.join("/execute").unwrap();
    // URL::join handles trailing slash correctly
    assert_eq!(execute_url.as_str(), "http://localhost:3000/execute");
}

#[test]
fn test_build_execute_url_with_path() {
    let base = validate_runtime_url("http://localhost:3000/api/v1").unwrap();
    let execute_url = base.join("/execute").unwrap();
    // Note: /execute is an absolute path, so it replaces the path
    assert_eq!(execute_url.as_str(), "http://localhost:3000/execute");
}
