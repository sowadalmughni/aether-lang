//! LLM API Integration
//!
//! Provides real LLM API calls (OpenAI, Anthropic) behind the `llm-api` feature flag.
//! When disabled, falls back to mock responses for testing.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, instrument, warn};

// =============================================================================
// Configuration
// =============================================================================

/// LLM Provider configuration
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// OpenAI API key (from OPENAI_API_KEY env var)
    pub openai_api_key: Option<String>,
    /// Anthropic API key (from ANTHROPIC_API_KEY env var)
    pub anthropic_api_key: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// Default model to use
    pub default_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            timeout_secs: 30,
            max_retries: 3,
            default_model: "gpt-4o-mini".to_string(),
        }
    }
}

impl LlmConfig {
    /// Check if any real API is configured
    pub fn has_api_keys(&self) -> bool {
        self.openai_api_key.is_some() || self.anthropic_api_key.is_some()
    }

    /// Get the appropriate provider based on model name
    pub fn provider_for_model(&self, model: &str) -> LlmProvider {
        if model.starts_with("claude") || model.starts_with("anthropic") {
            LlmProvider::Anthropic
        } else if model.starts_with("gpt") || model.starts_with("o1") || model.starts_with("o3") {
            LlmProvider::OpenAI
        } else {
            // Default to OpenAI
            LlmProvider::OpenAI
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlmProvider {
    OpenAI,
    Anthropic,
    Mock,
}

// =============================================================================
// Request/Response Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub prompt: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub model: String,
    pub provider: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub latency_ms: u64,
}

// =============================================================================
// OpenAI API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// =============================================================================
// Anthropic API Types
// =============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    model: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// =============================================================================
// LLM Client Trait
// =============================================================================

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError>;
    fn provider(&self) -> LlmProvider;
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("API request failed: {0}")]
    RequestFailed(String),
    #[error("Rate limited: retry after {0}s")]
    RateLimited(u64),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Timeout after {0}s")]
    Timeout(u64),
    #[error("Provider not configured: {0}")]
    NotConfigured(String),
}

// =============================================================================
// Mock Client (always available)
// =============================================================================

/// Mock LLM client for testing without real API calls
pub struct MockLlmClient {
    latency_ms: u64,
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self { latency_ms: 50 }
    }

    pub fn with_latency(latency_ms: u64) -> Self {
        Self { latency_ms }
    }
}

impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    #[instrument(skip(self))]
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let start = std::time::Instant::now();

        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;

        let input_tokens = (request.prompt.len() / 4) as u32;
        let output_tokens = 50 + (input_tokens / 4);

        let content = format!(
            "[Mock LLM Response]\nModel: {}\nPrompt length: {} chars\nGenerated mock response for testing.",
            request.model,
            request.prompt.len()
        );

        info!(
            model = %request.model,
            input_tokens,
            output_tokens,
            "Mock LLM completion"
        );

        Ok(LlmResponse {
            content,
            model: request.model,
            provider: "mock".to_string(),
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn provider(&self) -> LlmProvider {
        LlmProvider::Mock
    }
}

// =============================================================================
// Real LLM Clients (behind feature flag)
// =============================================================================

#[cfg(feature = "llm-api")]
pub mod real {
    use super::*;
    use reqwest::Client;

    /// OpenAI API client
    pub struct OpenAIClient {
        client: Client,
        api_key: String,
        base_url: String,
    }

    impl OpenAIClient {
        pub fn new(api_key: String) -> Self {
            Self {
                client: Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()
                    .expect("Failed to create HTTP client"),
                api_key,
                base_url: "https://api.openai.com/v1".to_string(),
            }
        }

        pub fn with_base_url(mut self, base_url: String) -> Self {
            self.base_url = base_url;
            self
        }
    }

    #[async_trait]
    impl LlmClient for OpenAIClient {
        #[instrument(skip(self, request), fields(model = %request.model))]
        async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
            let start = std::time::Instant::now();

            let mut messages = Vec::new();

            if let Some(system) = &request.system_prompt {
                messages.push(OpenAIMessage {
                    role: "system".to_string(),
                    content: system.clone(),
                });
            }

            messages.push(OpenAIMessage {
                role: "user".to_string(),
                content: request.prompt.clone(),
            });

            let openai_request = OpenAIRequest {
                model: request.model.clone(),
                messages,
                temperature: request.temperature,
                max_tokens: request.max_tokens,
            };

            let response = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&openai_request)
                .send()
                .await
                .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

            let status = response.status();

            if status == reqwest::StatusCode::UNAUTHORIZED {
                return Err(LlmError::AuthenticationFailed(
                    "Invalid OpenAI API key".to_string(),
                ));
            }

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                // Parse retry-after header if present
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60);
                return Err(LlmError::RateLimited(retry_after));
            }

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(LlmError::RequestFailed(format!(
                    "HTTP {}: {}",
                    status, error_text
                )));
            }

            let openai_response: OpenAIResponse = response
                .json()
                .await
                .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

            let content = openai_response
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))?;

            let latency_ms = start.elapsed().as_millis() as u64;

            info!(
                model = %openai_response.model,
                input_tokens = openai_response.usage.prompt_tokens,
                output_tokens = openai_response.usage.completion_tokens,
                latency_ms,
                "OpenAI completion successful"
            );

            Ok(LlmResponse {
                content,
                model: openai_response.model,
                provider: "openai".to_string(),
                input_tokens: openai_response.usage.prompt_tokens,
                output_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
                latency_ms,
            })
        }

        fn provider(&self) -> LlmProvider {
            LlmProvider::OpenAI
        }
    }

    /// Anthropic API client
    pub struct AnthropicClient {
        client: Client,
        api_key: String,
        base_url: String,
    }

    impl AnthropicClient {
        pub fn new(api_key: String) -> Self {
            Self {
                client: Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()
                    .expect("Failed to create HTTP client"),
                api_key,
                base_url: "https://api.anthropic.com/v1".to_string(),
            }
        }

        pub fn with_base_url(mut self, base_url: String) -> Self {
            self.base_url = base_url;
            self
        }
    }

    #[async_trait]
    impl LlmClient for AnthropicClient {
        #[instrument(skip(self, request), fields(model = %request.model))]
        async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
            let start = std::time::Instant::now();

            let anthropic_request = AnthropicRequest {
                model: request.model.clone(),
                messages: vec![AnthropicMessage {
                    role: "user".to_string(),
                    content: request.prompt.clone(),
                }],
                max_tokens: request.max_tokens.unwrap_or(4096),
                temperature: request.temperature,
                system: request.system_prompt.clone(),
            };

            let response = self
                .client
                .post(format!("{}/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&anthropic_request)
                .send()
                .await
                .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

            let status = response.status();

            if status == reqwest::StatusCode::UNAUTHORIZED {
                return Err(LlmError::AuthenticationFailed(
                    "Invalid Anthropic API key".to_string(),
                ));
            }

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60);
                return Err(LlmError::RateLimited(retry_after));
            }

            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(LlmError::RequestFailed(format!(
                    "HTTP {}: {}",
                    status, error_text
                )));
            }

            let anthropic_response: AnthropicResponse = response
                .json()
                .await
                .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

            let content = anthropic_response
                .content
                .first()
                .map(|c| c.text.clone())
                .ok_or_else(|| LlmError::InvalidResponse("No content in response".to_string()))?;

            let latency_ms = start.elapsed().as_millis() as u64;

            info!(
                model = %anthropic_response.model,
                input_tokens = anthropic_response.usage.input_tokens,
                output_tokens = anthropic_response.usage.output_tokens,
                latency_ms,
                "Anthropic completion successful"
            );

            Ok(LlmResponse {
                content,
                model: anthropic_response.model,
                provider: "anthropic".to_string(),
                input_tokens: anthropic_response.usage.input_tokens,
                output_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens
                    + anthropic_response.usage.output_tokens,
                latency_ms,
            })
        }

        fn provider(&self) -> LlmProvider {
            LlmProvider::Anthropic
        }
    }
}

// =============================================================================
// Client Factory
// =============================================================================

/// Create an LLM client based on configuration and desired provider
pub fn create_client(config: &LlmConfig, model: &str) -> Box<dyn LlmClient> {
    #[cfg(feature = "llm-api")]
    {
        let provider = config.provider_for_model(model);

        match provider {
            LlmProvider::OpenAI => {
                if let Some(api_key) = &config.openai_api_key {
                    info!("Using OpenAI API for model: {}", model);
                    return Box::new(real::OpenAIClient::new(api_key.clone()));
                }
            }
            LlmProvider::Anthropic => {
                if let Some(api_key) = &config.anthropic_api_key {
                    info!("Using Anthropic API for model: {}", model);
                    return Box::new(real::AnthropicClient::new(api_key.clone()));
                }
            }
            LlmProvider::Mock => {}
        }

        warn!(
            "No API key configured for provider {:?}, falling back to mock",
            provider
        );
    }

    #[cfg(not(feature = "llm-api"))]
    {
        info!(
            "llm-api feature not enabled, using mock client for model: {}",
            model
        );
    }

    Box::new(MockLlmClient::new())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client() {
        let client = MockLlmClient::new();

        let request = LlmRequest {
            prompt: "Hello, world!".to_string(),
            model: "gpt-4o-mini".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            system_prompt: None,
        };

        let response = client.complete(request).await.unwrap();

        assert!(response.content.contains("Mock LLM Response"));
        assert_eq!(response.provider, "mock");
        assert!(response.total_tokens > 0);
    }

    #[test]
    fn test_provider_detection() {
        let config = LlmConfig::default();

        assert_eq!(config.provider_for_model("gpt-4o"), LlmProvider::OpenAI);
        assert_eq!(
            config.provider_for_model("gpt-4o-mini"),
            LlmProvider::OpenAI
        );
        assert_eq!(config.provider_for_model("o1-preview"), LlmProvider::OpenAI);
        assert_eq!(
            config.provider_for_model("claude-3-opus"),
            LlmProvider::Anthropic
        );
        assert_eq!(
            config.provider_for_model("claude-3-5-sonnet"),
            LlmProvider::Anthropic
        );
        assert_eq!(
            config.provider_for_model("unknown-model"),
            LlmProvider::OpenAI
        );
    }

    #[test]
    fn test_config_from_env() {
        // This test just verifies the config reads from env without crashing
        let config = LlmConfig::default();
        assert!(config.timeout_secs > 0);
        assert!(config.max_retries > 0);
    }
}
