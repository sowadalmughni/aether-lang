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
    /// Force a specific provider (from AETHER_PROVIDER env var: mock|openai|anthropic)
    pub forced_provider: Option<LlmProvider>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        // Parse AETHER_PROVIDER env var to force a specific provider
        let forced_provider = std::env::var("AETHER_PROVIDER")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "mock" => Some(LlmProvider::Mock),
                "openai" => Some(LlmProvider::OpenAI),
                "anthropic" => Some(LlmProvider::Anthropic),
                _ => {
                    tracing::warn!(
                        "Unknown AETHER_PROVIDER value '{}', valid options: mock, openai, anthropic",
                        s
                    );
                    None
                }
            });

        Self {
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            timeout_secs: 30,
            max_retries: 3,
            default_model: "gpt-4o-mini".to_string(),
            forced_provider,
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
    /// Whether this response was served from cache
    #[serde(default)]
    pub cached: bool,
}

impl LlmResponse {
    /// Create a cached version of this response (0 tokens charged)
    pub fn as_cached(&self) -> Self {
        Self {
            content: self.content.clone(),
            model: self.model.clone(),
            provider: self.provider.clone(),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            latency_ms: 0,
            cached: true,
        }
    }

    /// Mark this response as cached
    pub fn with_cached(mut self, cached: bool) -> Self {
        self.cached = cached;
        self
    }
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
///
/// Features:
/// - Configurable latency to simulate network delays
/// - Configurable responses keyed by prompt hash for deterministic testing
/// - Default fallback response when no match found
pub struct MockLlmClient {
    latency_ms: u64,
    /// Map of prompt hash -> response content for deterministic testing
    responses: std::collections::HashMap<String, String>,
    /// Whether to fail requests (for error testing)
    should_fail: bool,
    /// Number of times to fail before succeeding (for retry testing)
    fail_count: std::sync::atomic::AtomicU32,
    fail_until: u32,
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self {
            latency_ms: 50,
            responses: std::collections::HashMap::new(),
            should_fail: false,
            fail_count: std::sync::atomic::AtomicU32::new(0),
            fail_until: 0,
        }
    }

    pub fn with_latency(latency_ms: u64) -> Self {
        Self {
            latency_ms,
            ..Self::new()
        }
    }

    /// Add a deterministic response for a specific prompt
    pub fn with_response(mut self, prompt: &str, response: &str) -> Self {
        let hash = Self::hash_prompt(prompt);
        self.responses.insert(hash, response.to_string());
        self
    }

    /// Add responses from a map
    pub fn with_responses(mut self, responses: std::collections::HashMap<String, String>) -> Self {
        for (prompt, response) in responses {
            let hash = Self::hash_prompt(&prompt);
            self.responses.insert(hash, response);
        }
        self
    }

    /// Configure to fail all requests (for error testing)
    pub fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    /// Configure to fail N times before succeeding (for retry testing)
    pub fn fail_n_times(mut self, n: u32) -> Self {
        self.fail_until = n;
        self
    }

    /// Hash a prompt for lookup
    fn hash_prompt(prompt: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get response for a prompt (or generate default)
    fn get_response(&self, prompt: &str, model: &str) -> String {
        let hash = Self::hash_prompt(prompt);
        if let Some(response) = self.responses.get(&hash) {
            response.clone()
        } else {
            // Generate deterministic default response
            format!(
                "[Mock Response]\nModel: {}\nPrompt hash: {}\nThis is a deterministic mock response for testing.",
                model,
                &hash[..16]
            )
        }
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

        // Check if we should fail
        if self.should_fail {
            return Err(LlmError::RequestFailed("Mock failure mode enabled".to_string()));
        }

        // Check fail_n_times logic
        if self.fail_until > 0 {
            let count = self.fail_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count < self.fail_until {
                return Err(LlmError::RequestFailed(format!(
                    "Mock failure {}/{} (will succeed after {} more attempts)",
                    count + 1,
                    self.fail_until,
                    self.fail_until - count - 1
                )));
            }
        }

        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;

        let input_tokens = (request.prompt.len() / 4) as u32;
        let output_tokens = 50 + (input_tokens / 4);

        let content = self.get_response(&request.prompt, &request.model);

        info!(
            model = %request.model,
            input_tokens,
            output_tokens,
            latency_ms = self.latency_ms,
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
            cached: false,
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
                cached: false,
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
                cached: false,
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
///
/// Provider selection priority:
/// 1. AETHER_PROVIDER env var (mock|openai|anthropic) - forces specific provider
/// 2. Model name prefix detection (gpt-* -> OpenAI, claude-* -> Anthropic)
/// 3. Falls back to mock if no API keys configured
pub fn create_client(config: &LlmConfig, model: &str) -> Box<dyn LlmClient> {
    // Check for forced provider from AETHER_PROVIDER env var
    if let Some(forced) = config.forced_provider {
        match forced {
            LlmProvider::Mock => {
                info!("AETHER_PROVIDER=mock, using MockLlmClient");
                return Box::new(MockLlmClient::new());
            }
            #[cfg(feature = "llm-api")]
            LlmProvider::OpenAI => {
                if let Some(api_key) = &config.openai_api_key {
                    info!("AETHER_PROVIDER=openai, using OpenAI API for model: {}", model);
                    return Box::new(real::OpenAIClient::new(api_key.clone()));
                } else {
                    warn!("AETHER_PROVIDER=openai but OPENAI_API_KEY not set, falling back to mock");
                    return Box::new(MockLlmClient::new());
                }
            }
            #[cfg(feature = "llm-api")]
            LlmProvider::Anthropic => {
                if let Some(api_key) = &config.anthropic_api_key {
                    info!("AETHER_PROVIDER=anthropic, using Anthropic API for model: {}", model);
                    return Box::new(real::AnthropicClient::new(api_key.clone()));
                } else {
                    warn!("AETHER_PROVIDER=anthropic but ANTHROPIC_API_KEY not set, falling back to mock");
                    return Box::new(MockLlmClient::new());
                }
            }
            #[cfg(not(feature = "llm-api"))]
            LlmProvider::OpenAI | LlmProvider::Anthropic => {
                warn!(
                    "AETHER_PROVIDER={:?} but llm-api feature not enabled, falling back to mock",
                    forced
                );
                return Box::new(MockLlmClient::new());
            }
        }
    }

    // No forced provider - use model-based detection
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

    #[test]
    fn test_forced_provider_parsing() {
        // Test that forced_provider field is properly initialized
        // (actual env var testing requires process isolation)
        let config = LlmConfig {
            openai_api_key: None,
            anthropic_api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            default_model: "gpt-4o-mini".to_string(),
            forced_provider: Some(LlmProvider::Mock),
        };
        assert_eq!(config.forced_provider, Some(LlmProvider::Mock));

        let config_openai = LlmConfig {
            forced_provider: Some(LlmProvider::OpenAI),
            ..config.clone()
        };
        assert_eq!(config_openai.forced_provider, Some(LlmProvider::OpenAI));

        let config_anthropic = LlmConfig {
            forced_provider: Some(LlmProvider::Anthropic),
            ..config.clone()
        };
        assert_eq!(config_anthropic.forced_provider, Some(LlmProvider::Anthropic));
    }

    #[tokio::test]
    async fn test_create_client_forced_mock() {
        let config = LlmConfig {
            openai_api_key: Some("sk-test".to_string()),
            anthropic_api_key: Some("sk-ant-test".to_string()),
            timeout_secs: 30,
            max_retries: 3,
            default_model: "gpt-4o-mini".to_string(),
            forced_provider: Some(LlmProvider::Mock),
        };

        // Even with API keys present, forced mock should return MockLlmClient
        let client = create_client(&config, "gpt-4o");
        assert_eq!(client.provider(), LlmProvider::Mock);
    }
}
