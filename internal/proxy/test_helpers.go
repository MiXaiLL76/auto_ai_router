package proxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
)

// ============================================================================
// Logger Helpers
// ============================================================================

// ============================================================================
// Infrastructure Helpers (Metrics, TokenManager, ModelManager)
// ============================================================================

// createProxyWithParams is a helper to create a proxy with parameters for testing.
// Used in tests that directly call New() instead of using TestProxyBuilder.
func createProxyWithParams(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics, masterKey string, rl *ratelimit.RPMLimiter, tm *auth.VertexTokenManager, mm *models.Manager, version, commit string) *Proxy {
	return New(&Config{
		Balancer:       bal,
		Logger:         logger,
		MaxBodySizeMB:  maxBodySizeMB,
		RequestTimeout: requestTimeout,
		Metrics:        metrics,
		MasterKey:      masterKey,
		RateLimiter:    rl,
		TokenManager:   tm,
		ModelManager:   mm,
		Version:        version,
		Commit:         commit,
	})
}

// createTestProxyMetrics creates a metrics instance for testing.
func createTestProxyMetrics() *monitoring.Metrics {
	return monitoring.New(false)
}

// createTestTokenManager creates a token manager for testing.
func createTestTokenManager(logger *slog.Logger) *auth.VertexTokenManager {
	return auth.NewVertexTokenManager(logger)
}

// createTestModelManager creates a model manager for testing.
func createTestModelManager(logger *slog.Logger) *models.Manager {
	return models.New(logger, 50, []config.ModelRPMConfig{})
}

// ============================================================================
// Balancer and Rate Limiter Helpers
// ============================================================================

// createTestBalancer creates a balancer with rate limiter for testing.
func createTestBalancer(baseURL string) (*balancer.RoundRobin, *ratelimit.RPMLimiter) {
	rl := ratelimit.New()
	creds := []config.CredentialConfig{
		{
			Name:    "test",
			Type:    config.ProviderTypeProxy,
			BaseURL: baseURL,
			APIKey:  "upstream-key-1",
			RPM:     100,
			TPM:     10000,
		},
		{
			Name:    "test2",
			Type:    config.ProviderTypeProxy,
			BaseURL: baseURL,
			APIKey:  "upstream-key-2",
			RPM:     100,
			TPM:     10000,
		},
	}
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	return balancer.New(creds, f2b, rl), rl
}

// ============================================================================
// Proxy Builder - Main Testing Helper
// ============================================================================

// TestProxyConfig holds configuration for building a test proxy instance.
type TestProxyConfig struct {
	Credentials    []config.CredentialConfig
	Logger         *slog.Logger
	Balancer       *balancer.RoundRobin
	RateLimiter    *ratelimit.RPMLimiter
	Metrics        *monitoring.Metrics
	TokenManager   *auth.VertexTokenManager
	ModelManager   *models.Manager
	MasterKey      string
	MaxBodySizeMB  int
	RequestTimeout time.Duration
	Version        string
	Commit         string
}

// NewTestProxyBuilder creates a builder with default configuration.
func NewTestProxyBuilder() *TestProxyBuilder {
	logger := testhelpers.NewTestLogger()
	return &TestProxyBuilder{
		config: &TestProxyConfig{
			Logger:         logger,
			Metrics:        createTestProxyMetrics(),
			TokenManager:   createTestTokenManager(logger),
			ModelManager:   createTestModelManager(logger),
			MasterKey:      "sk-master",
			MaxBodySizeMB:  1,
			RequestTimeout: 5 * time.Second,
			Version:        "test-version",
			Commit:         "test-commit",
		},
	}
}

// TestProxyBuilder is a fluent builder for creating test proxy instances.
type TestProxyBuilder struct {
	config *TestProxyConfig
}

// WithCredentials sets the credentials for the proxy.
func (b *TestProxyBuilder) WithCredentials(creds ...config.CredentialConfig) *TestProxyBuilder {
	b.config.Credentials = creds
	return b
}

// WithSingleCredential is a convenience method for adding a single credential.
func (b *TestProxyBuilder) WithSingleCredential(name string, credType config.ProviderType, baseURL, apiKey string) *TestProxyBuilder {
	cred := config.CredentialConfig{
		Name:       name,
		Type:       credType,
		BaseURL:    baseURL,
		APIKey:     apiKey,
		RPM:        100,
		TPM:        10000,
		IsFallback: false,
	}
	return b.WithCredentials(cred)
}

// WithPrimaryAndFallback is a convenience method for creating primary + fallback pair.
func (b *TestProxyBuilder) WithPrimaryAndFallback(primaryURL, fallbackURL string) *TestProxyBuilder {
	creds := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "pkey",
			BaseURL:    primaryURL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
		{
			Name:       "fallback",
			Type:       config.ProviderTypeProxy,
			APIKey:     "",
			BaseURL:    fallbackURL,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		},
	}
	return b.WithCredentials(creds...)
}

// WithMultipleFallbacks creates primary + multiple fallbacks.
func (b *TestProxyBuilder) WithMultipleFallbacks(primaryURL string, fallbackURLs ...string) *TestProxyBuilder {
	creds := []config.CredentialConfig{
		{
			Name:       "primary",
			Type:       config.ProviderTypeProxy,
			APIKey:     "pkey",
			BaseURL:    primaryURL,
			RPM:        100,
			TPM:        10000,
			IsFallback: false,
		},
	}
	for i, url := range fallbackURLs {
		creds = append(creds, config.CredentialConfig{
			Name:       "fallback" + string(rune('1'+i)),
			Type:       config.ProviderTypeProxy,
			APIKey:     "",
			BaseURL:    url,
			RPM:        100,
			TPM:        10000,
			IsFallback: true,
		})
	}
	return b.WithCredentials(creds...)
}

// WithMasterKey sets the master API key.
func (b *TestProxyBuilder) WithMasterKey(key string) *TestProxyBuilder {
	b.config.MasterKey = key
	return b
}

// WithRequestTimeout sets the request timeout.
func (b *TestProxyBuilder) WithRequestTimeout(timeout time.Duration) *TestProxyBuilder {
	b.config.RequestTimeout = timeout
	return b
}

// Build creates and returns a Proxy instance with the configured settings.
func (b *TestProxyBuilder) Build() *Proxy {
	// Create rate limiter if not already set
	if b.config.RateLimiter == nil {
		b.config.RateLimiter = ratelimit.New()
	}

	// Add credentials to rate limiter
	for _, cred := range b.config.Credentials {
		b.config.RateLimiter.AddCredential(cred.Name, cred.RPM)
	}

	// Create balancer if not already set
	if b.config.Balancer == nil {
		f2b := fail2ban.New(3, 0, []int{401, 403, 500})
		b.config.Balancer = balancer.New(b.config.Credentials, f2b, b.config.RateLimiter)
	}

	return New(&Config{
		Balancer:       b.config.Balancer,
		Logger:         b.config.Logger,
		MaxBodySizeMB:  b.config.MaxBodySizeMB,
		RequestTimeout: b.config.RequestTimeout,
		Metrics:        b.config.Metrics,
		MasterKey:      b.config.MasterKey,
		RateLimiter:    b.config.RateLimiter,
		TokenManager:   b.config.TokenManager,
		ModelManager:   b.config.ModelManager,
		Version:        b.config.Version,
		Commit:         b.config.Commit,
	})
}

// ============================================================================
// HTTP Mock Server Helper
// ============================================================================

// MockHTTPServer wraps httptest.Server with tracking capabilities.
type MockHTTPServer struct {
	Server      *httptest.Server
	CallCount   int32
	LastRequest *http.Request
	LastBody    []byte
}

// NewMockHTTPServer creates a new mock HTTP server with the given handler.
// The handler is wrapped to track calls and requests.
func NewMockHTTPServer(handler func(w http.ResponseWriter, r *http.Request)) *MockHTTPServer {
	mock := &MockHTTPServer{}

	mock.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&mock.CallCount, 1)
		mock.LastRequest = r

		// Try to read body
		body, err := io.ReadAll(r.Body)
		if err == nil {
			mock.LastBody = body
		}

		handler(w, r)
	}))

	return mock
}

// GetCallCount returns the number of times the server was called.
func (m *MockHTTPServer) GetCallCount() int32 {
	return atomic.LoadInt32(&m.CallCount)
}

// Close closes the mock server.
func (m *MockHTTPServer) Close() {
	m.Server.Close()
}

// ============================================================================
// Common Mock Responses
// ============================================================================

// createMockProxyHealthResponse creates a standard health response for testing.
func createMockProxyHealthResponse() *httputil.ProxyHealthResponse {
	return &httputil.ProxyHealthResponse{
		Status:               "ok",
		CredentialsAvailable: 2,
		CredentialsBanned:    0,
		TotalCredentials:     2,
		Credentials: map[string]httputil.CredentialHealthStats{
			"remote_cred_1": {
				Type:       "openai",
				IsFallback: false,
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 25,
				CurrentTPM: 250,
			},
			"remote_cred_2": {
				Type:       "openai",
				IsFallback: false,
				LimitRPM:   200,
				LimitTPM:   2000,
				CurrentRPM: 20,
				CurrentTPM: 200,
			},
		},
		Models: map[string]httputil.ModelHealthStats{
			"gpt4_cred1": {
				Credential: "remote_cred_1",
				Model:      "gpt-4",
				LimitRPM:   50,
				LimitTPM:   500,
				CurrentRPM: 10,
				CurrentTPM: 100,
			},
			"gpt4_cred2": {
				Credential: "remote_cred_2",
				Model:      "gpt-4",
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 15,
				CurrentTPM: 150,
			},
			"claude_cred1": {
				Credential: "remote_cred_1",
				Model:      "claude-3-opus",
				LimitRPM:   75,
				LimitTPM:   1500,
				CurrentRPM: 5,
				CurrentTPM: 50,
			},
		},
	}
}

// createMockChatCompletionResponse creates a standard chat completion response.
func createMockChatCompletionResponse(id, model, content string) map[string]interface{} {
	return map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     10,
			"completion_tokens": 5,
			"total_tokens":      15,
		},
	}
}

// ============================================================================
// Test Assertions Helpers
// ============================================================================

// AssertRateLimiterStats checks that rate limiter has expected values.
type RateLimiterStatsAssertion struct {
	CredName     string
	ExpectedRPM  int
	ExpectedTPM  int
	ModelID      string
	ExpectedMRPM int
	ExpectedMTPM int
}

// AssertCredentialStats is a helper to verify rate limiter stats.
// Use with t.Run subtests for cleaner test output.
func AssertCredentialStats(t *testing.T, rl *ratelimit.RPMLimiter, assertion RateLimiterStatsAssertion) {
	t.Helper()

	rpm := rl.GetLimitRPM(assertion.CredName)
	if assertion.ExpectedRPM >= 0 && rpm != assertion.ExpectedRPM {
		t.Errorf("Expected RPM %d for %s, got %d", assertion.ExpectedRPM, assertion.CredName, rpm)
	}

	tpm := rl.GetLimitTPM(assertion.CredName)
	if assertion.ExpectedTPM >= 0 && tpm != assertion.ExpectedTPM {
		t.Errorf("Expected TPM %d for %s, got %d", assertion.ExpectedTPM, assertion.CredName, tpm)
	}

	if assertion.ModelID != "" {
		mRPM := rl.GetModelLimitRPM(assertion.CredName, assertion.ModelID)
		if assertion.ExpectedMRPM >= 0 && mRPM != assertion.ExpectedMRPM {
			t.Errorf("Expected model RPM %d for %s/%s, got %d",
				assertion.ExpectedMRPM, assertion.CredName, assertion.ModelID, mRPM)
		}

		mTPM := rl.GetModelLimitTPM(assertion.CredName, assertion.ModelID)
		if assertion.ExpectedMTPM >= 0 && mTPM != assertion.ExpectedMTPM {
			t.Errorf("Expected model TPM %d for %s/%s, got %d",
				assertion.ExpectedMTPM, assertion.CredName, assertion.ModelID, mTPM)
		}
	}
}

// ============================================================================
// Mock Response Builders
// ============================================================================

// ResponseBuilder provides a fluent interface for building HTTP responses in tests.
type ResponseBuilder struct {
	statusCode int
	headers    http.Header
	body       interface{}
}

// NewResponseBuilder creates a new response builder.
func NewResponseBuilder() *ResponseBuilder {
	return &ResponseBuilder{
		statusCode: http.StatusOK,
		headers:    make(http.Header),
	}
}

// WithStatus sets the HTTP status code.
func (rb *ResponseBuilder) WithStatus(code int) *ResponseBuilder {
	rb.statusCode = code
	return rb
}

// WithHeader adds a header to the response.
func (rb *ResponseBuilder) WithHeader(key, value string) *ResponseBuilder {
	rb.headers.Set(key, value)
	return rb
}

// WithJSONBody sets the response body to a JSON-encoded value.
func (rb *ResponseBuilder) WithJSONBody(body interface{}) *ResponseBuilder {
	rb.body = body
	rb.headers.Set("Content-Type", "application/json")
	return rb
}

// Write writes the response to the http.ResponseWriter.
func (rb *ResponseBuilder) Write(w http.ResponseWriter) error {
	for key, values := range rb.headers {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(rb.statusCode)

	if rb.body != nil {
		return json.NewEncoder(w).Encode(rb.body)
	}
	return nil
}

// ============================================================================
// Request Builders for Common Test Scenarios
// ============================================================================

// ChatCompletionRequestBuilder builds a chat completion request for testing.
type ChatCompletionRequestBuilder struct {
	model       string
	messages    []map[string]string
	temperature float64
	maxTokens   int
}

// NewChatCompletionRequest creates a new chat completion request builder.
func NewChatCompletionRequest(model string) *ChatCompletionRequestBuilder {
	return &ChatCompletionRequestBuilder{
		model:       model,
		messages:    []map[string]string{},
		temperature: 0.7,
		maxTokens:   100,
	}
}

// WithMessage adds a message to the request.
func (cb *ChatCompletionRequestBuilder) WithMessage(role, content string) *ChatCompletionRequestBuilder {
	cb.messages = append(cb.messages, map[string]string{
		"role":    role,
		"content": content,
	})
	return cb
}

// WithTemperature sets the temperature.
func (cb *ChatCompletionRequestBuilder) WithTemperature(temp float64) *ChatCompletionRequestBuilder {
	cb.temperature = temp
	return cb
}

// WithMaxTokens sets the max tokens.
func (cb *ChatCompletionRequestBuilder) WithMaxTokens(tokens int) *ChatCompletionRequestBuilder {
	cb.maxTokens = tokens
	return cb
}

// Build returns the request body as a map.
func (cb *ChatCompletionRequestBuilder) Build() map[string]interface{} {
	return map[string]interface{}{
		"model":       cb.model,
		"messages":    cb.messages,
		"temperature": cb.temperature,
		"max_tokens":  cb.maxTokens,
	}
}
