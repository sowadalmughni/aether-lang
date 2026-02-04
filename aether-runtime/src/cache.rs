//! Caching layer for Aether runtime
//!
//! Implements multi-level caching for LLM calls:
//! - Level 1: Exact-match cache (LRU, in-memory) - IMPLEMENTED
//! - Level 2: Semantic cache (planned - vector similarity)
//! - Level 3: Provider prefix cache hints (planned)
//!
//! Cache key is computed from: hash(model + rendered_prompt + temperature + other params)
//! Cache hits return stored response with 0 token cost, flagged as cached: true

use aether_core::DagNode;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::Mutex;
use tracing::{info, instrument};

/// Configuration for the cache layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the exact-match cache
    pub max_entries: usize,
    /// Whether exact-match caching is enabled
    pub exact_match_enabled: bool,
    /// Whether semantic caching is enabled (planned)
    pub semantic_enabled: bool,
    /// Similarity threshold for semantic cache hits (0.0-1.0)
    pub semantic_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            exact_match_enabled: true,
            semantic_enabled: false,
            semantic_threshold: 0.95,
        }
    }
}

/// A cached LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub output: String,
    pub token_cost: u32,
    pub cache_key: String,
    pub cached_at: u64,
    /// Original input/output tokens (for metrics, not charged on cache hit)
    #[serde(default)]
    pub original_input_tokens: u32,
    #[serde(default)]
    pub original_output_tokens: u32,
    /// Model that generated this response
    #[serde(default)]
    pub model: String,
    /// Provider that generated this response
    #[serde(default)]
    pub provider: String,
}

impl CachedResponse {
    /// Create a new cached response
    pub fn new(output: String, token_cost: u32, cache_key: String) -> Self {
        Self {
            output,
            token_cost,
            cache_key,
            cached_at: current_timestamp(),
            original_input_tokens: 0,
            original_output_tokens: 0,
            model: String::new(),
            provider: String::new(),
        }
    }

    /// Builder pattern: set original tokens
    pub fn with_tokens(mut self, input: u32, output: u32) -> Self {
        self.original_input_tokens = input;
        self.original_output_tokens = output;
        self
    }

    /// Builder pattern: set model info
    pub fn with_model(mut self, model: impl Into<String>, provider: impl Into<String>) -> Self {
        self.model = model.into();
        self.provider = provider.into();
        self
    }
}

/// Cache key components for exact-match lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheKey {
    pub prompt: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

impl CacheKey {
    pub fn new(prompt: String, model: String) -> Self {
        Self {
            prompt,
            model,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Create a CacheKey from a DagNode and rendered prompt
    ///
    /// This is the recommended way to create cache keys during DAG execution,
    /// as it captures all relevant parameters for deterministic caching.
    pub fn from_dag_node(node: &DagNode, rendered_prompt: &str) -> Self {
        Self {
            prompt: rendered_prompt.to_string(),
            model: node.model.clone().unwrap_or_else(|| "default".to_string()),
            temperature: node.temperature,
            max_tokens: node.max_tokens,
        }
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Generate a SHA256 hash of the cache key
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.prompt.as_bytes());
        hasher.update(self.model.as_bytes());
        if let Some(temp) = self.temperature {
            hasher.update(temp.to_le_bytes());
        }
        if let Some(max) = self.max_tokens {
            hasher.update(max.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Get the prompt (for debugging/logging)
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Total tokens saved by cache hits
    pub tokens_saved: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Estimated cost savings (rough estimate based on tokens)
    pub fn estimated_savings(&self, cost_per_1k_tokens: f64) -> f64 {
        (self.tokens_saved as f64 / 1000.0) * cost_per_1k_tokens
    }
}

/// Multi-level cache for LLM responses
pub struct LlmCache {
    config: CacheConfig,
    exact_cache: Mutex<LruCache<String, CachedResponse>>,
    stats: Mutex<CacheStats>,
}

impl LlmCache {
    /// Create a new cache with the given configuration
    pub fn new(config: CacheConfig) -> Self {
        let capacity = NonZeroUsize::new(config.max_entries.max(1)).unwrap();
        Self {
            config,
            exact_cache: Mutex::new(LruCache::new(capacity)),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Create a cache with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Look up a cached response by key
    #[instrument(skip(self), fields(cache_key_hash))]
    pub fn get(&self, key: &CacheKey) -> Option<CachedResponse> {
        if !self.config.exact_match_enabled {
            return None;
        }

        let hash = key.hash();
        let mut cache = self.exact_cache.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(response) = cache.get(&hash) {
            stats.hits += 1;
            stats.tokens_saved += (response.original_input_tokens + response.original_output_tokens) as u64;
            info!(
                cache_key_hash = %hash,
                hit_rate = %stats.hit_rate(),
                tokens_saved = stats.tokens_saved,
                "Cache hit"
            );
            Some(response.clone())
        } else {
            stats.misses += 1;
            info!(
                cache_key_hash = %hash,
                hit_rate = %stats.hit_rate(),
                "Cache miss"
            );
            None
        }
    }

    /// Store a response in the cache
    #[instrument(skip(self, response), fields(cache_key_hash))]
    pub fn put(&self, key: &CacheKey, response: CachedResponse) {
        if !self.config.exact_match_enabled {
            return;
        }

        let hash = key.hash();
        let mut cache = self.exact_cache.lock().unwrap();

        // Check if we're about to evict
        if cache.len() >= self.config.max_entries {
            let mut stats = self.stats.lock().unwrap();
            stats.evictions += 1;
        }

        cache.put(hash.clone(), response);
        info!(cache_key_hash = %hash, "Cached response");
    }

    /// Get current cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.exact_cache.lock().unwrap();
        cache.clear();
        info!("Cache cleared");
    }

    /// Get the current number of cached entries
    pub fn len(&self) -> usize {
        self.exact_cache.lock().unwrap().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Get current timestamp in seconds
pub fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_hash() {
        let key1 = CacheKey::new("Hello".to_string(), "gpt-4o".to_string());
        let key2 = CacheKey::new("Hello".to_string(), "gpt-4o".to_string());
        let key3 = CacheKey::new("World".to_string(), "gpt-4o".to_string());

        assert_eq!(key1.hash(), key2.hash());
        assert_ne!(key1.hash(), key3.hash());
    }

    #[test]
    fn test_cache_key_hash_with_params() {
        let key1 = CacheKey::new("Hello".to_string(), "gpt-4o".to_string())
            .with_temperature(0.7);
        let key2 = CacheKey::new("Hello".to_string(), "gpt-4o".to_string())
            .with_temperature(0.7);
        let key3 = CacheKey::new("Hello".to_string(), "gpt-4o".to_string())
            .with_temperature(0.5);

        assert_eq!(key1.hash(), key2.hash());
        assert_ne!(key1.hash(), key3.hash());
    }

    #[test]
    fn test_cache_hit_miss() {
        let cache = LlmCache::with_defaults();
        let key = CacheKey::new("Test prompt".to_string(), "gpt-4o".to_string());

        // Miss on first lookup
        assert!(cache.get(&key).is_none());

        // Store a response
        let response = CachedResponse {
            output: "Test output".to_string(),
            token_cost: 10,
            cache_key: key.hash(),
            cached_at: current_timestamp(),
        };
        cache.put(&key, response.clone());

        // Hit on second lookup
        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().output, "Test output");

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let cache = LlmCache::new(config);

        // Fill cache
        for i in 0..3 {
            let key = CacheKey::new(format!("Prompt {}", i), "gpt-4o".to_string());
            let response = CachedResponse {
                output: format!("Output {}", i),
                token_cost: 10,
                cache_key: key.hash(),
                cached_at: current_timestamp(),
            };
            cache.put(&key, response);
        }

        // Check eviction occurred
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_disabled() {
        let config = CacheConfig {
            exact_match_enabled: false,
            ..Default::default()
        };
        let cache = LlmCache::new(config);
        let key = CacheKey::new("Test".to_string(), "gpt-4o".to_string());

        let response = CachedResponse {
            output: "Output".to_string(),
            token_cost: 10,
            cache_key: key.hash(),
            cached_at: current_timestamp(),
        };

        cache.put(&key, response);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = LlmCache::with_defaults();
        let key = CacheKey::new("Test".to_string(), "gpt-4o".to_string());

        let response = CachedResponse {
            output: "Output".to_string(),
            token_cost: 10,
            cache_key: key.hash(),
            cached_at: current_timestamp(),
        };

        cache.put(&key, response);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }
}
