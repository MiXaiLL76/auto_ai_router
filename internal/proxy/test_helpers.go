package proxy

import (
	"log/slog"
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
// Infrastructure Helpers (Metrics, TokenManager, ModelManager)
// ============================================================================

// createProxyWithParams is a helper to create a proxy with parameters for testing.
// Used in tests that directly call New() instead of using TestProxyBuilder.
func createProxyWithParams(bal *balancer.RoundRobin, logger *slog.Logger, maxBodySizeMB int, requestTimeout time.Duration, metrics *monitoring.Metrics, masterKey string, rl *ratelimit.RPMLimiter, tm *auth.VertexTokenManager, mm *models.Manager, version, commit string) *Proxy {
	return New(&Config{
		Balancer:        bal,
		Logger:          logger,
		MaxBodySizeMB:   maxBodySizeMB,
		RequestTimeout:  requestTimeout,
		Metrics:         metrics,
		MasterKey:       masterKey,
		RateLimiter:     rl,
		TokenManager:    tm,
		ModelManager:    mm,
		Version:         version,
		Commit:          commit,
		TiktokenEnabled: true, // matches production default (server.tiktoken_enabled: true)
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
		{ //nolint:gosec // G101: fake test fixture, not a real credential — this file only backs *_test.go callers
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
	Credentials            []config.CredentialConfig
	Logger                 *slog.Logger
	Balancer               *balancer.RoundRobin
	RateLimiter            *ratelimit.RPMLimiter
	Metrics                *monitoring.Metrics
	TokenManager           *auth.VertexTokenManager
	ModelManager           *models.Manager
	MasterKey              string
	MaxBodySizeMB          int
	ResponseBodyMultiplier int
	RequestTimeout         time.Duration
	Version                string
	Commit                 string
	SessionStickyEnabled   bool
	SessionStoreTTL        time.Duration
	MaxProviderRetries     int
	MaxFallbackAttempts    int
	DrainUpstreamOnAbort   bool
	TiktokenEnabled        bool
	ResponseHeaderMode     config.ResponseHeaderMode
	CredentialNameAsTeamID bool
	OrganizationPolicies   *models.OrganizationPolicyRegistry
}

// NewTestProxyBuilder creates a builder with default configuration.
func NewTestProxyBuilder() *TestProxyBuilder {
	logger := testhelpers.NewTestLogger()
	return &TestProxyBuilder{
		config: &TestProxyConfig{
			Logger:          logger,
			Metrics:         createTestProxyMetrics(),
			TokenManager:    createTestTokenManager(logger),
			ModelManager:    createTestModelManager(logger),
			MasterKey:       "master-key",
			MaxBodySizeMB:   10,
			RequestTimeout:  30 * time.Second,
			Version:         "test-version",
			Commit:          "test-commit",
			TiktokenEnabled: true, // matches production default (server.tiktoken_enabled: true)
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

// WithSessionSticky enables session-sticky credential routing with the given TTL.
func (b *TestProxyBuilder) WithSessionSticky(ttl time.Duration) *TestProxyBuilder {
	b.config.SessionStickyEnabled = true
	b.config.SessionStoreTTL = ttl
	return b
}

// WithMaxBodySizeMB sets the max response body size (in MB, before the
// ResponseBodyMultiplier is applied). Use together with WithResponseBodyMultiplier
// to get a small, test-friendly size limit for exercising ErrResponseBodyTooLarge.
func (b *TestProxyBuilder) WithMaxBodySizeMB(mb int) *TestProxyBuilder {
	b.config.MaxBodySizeMB = mb
	return b
}

// WithResponseBodyMultiplier overrides DefaultResponseBodyMultiplier (10) for tests
// that need a precise, small max response body size (MaxBodySizeMB * multiplier).
func (b *TestProxyBuilder) WithResponseBodyMultiplier(multiplier int) *TestProxyBuilder {
	b.config.ResponseBodyMultiplier = multiplier
	return b
}

// WithMaxProviderRetries sets the maximum number of same-type credential retries.
// This mirrors the production Config.MaxProviderRetries (default: 2).
// Use this in tests that need to validate retry/fallback behavior under realistic
// retry counts — without it, maxProviderRetries defaults to 0 and the retry loop
// only runs once, masking bugs in the retry logic.
func (b *TestProxyBuilder) WithMaxProviderRetries(n int) *TestProxyBuilder {
	b.config.MaxProviderRetries = n
	return b
}

// WithDrainUpstreamOnAbort enables or disables upstream draining after client disconnect.
func (b *TestProxyBuilder) WithDrainUpstreamOnAbort(v bool) *TestProxyBuilder {
	b.config.DrainUpstreamOnAbort = v
	return b
}

// WithTiktokenEnabled overrides the default (true) local tiktoken-based prompt/completion
// token fallback estimation toggle. Use WithTiktokenEnabled(false) to test the codepath
// where a provider's own usage is unavailable and no local estimate backfills it.
func (b *TestProxyBuilder) WithTiktokenEnabled(v bool) *TestProxyBuilder {
	b.config.TiktokenEnabled = v
	return b
}

func (b *TestProxyBuilder) WithResponseHeaderMode(mode config.ResponseHeaderMode) *TestProxyBuilder {
	b.config.ResponseHeaderMode = mode
	return b
}

func (b *TestProxyBuilder) WithCredentialNameAsTeamID(enabled bool) *TestProxyBuilder {
	b.config.CredentialNameAsTeamID = enabled
	return b
}

// Build creates and returns a Proxy instance with the configured settings.
func (b *TestProxyBuilder) Build() *Proxy {
	if b.config.RateLimiter == nil {
		b.config.RateLimiter = ratelimit.New()
	}
	for _, cred := range b.config.Credentials {
		b.config.RateLimiter.AddCredential(cred.Name, cred.RPM)
	}
	if b.config.Balancer == nil {
		f2b := fail2ban.New(3, 0, []int{401, 403, 500})
		b.config.Balancer = balancer.New(b.config.Credentials, f2b, b.config.RateLimiter)
	}
	return New(&Config{
		Balancer:               b.config.Balancer,
		Logger:                 b.config.Logger,
		MaxBodySizeMB:          b.config.MaxBodySizeMB,
		ResponseBodyMultiplier: b.config.ResponseBodyMultiplier,
		RequestTimeout:         b.config.RequestTimeout,
		Metrics:                b.config.Metrics,
		MasterKey:              b.config.MasterKey,
		RateLimiter:            b.config.RateLimiter,
		TokenManager:           b.config.TokenManager,
		ModelManager:           b.config.ModelManager,
		Version:                b.config.Version,
		Commit:                 b.config.Commit,
		SessionStickyEnabled:   b.config.SessionStickyEnabled,
		SessionStoreTTL:        b.config.SessionStoreTTL,
		MaxProviderRetries:     b.config.MaxProviderRetries,
		MaxFallbackAttempts:    b.config.MaxFallbackAttempts,
		DrainUpstreamOnAbort:   b.config.DrainUpstreamOnAbort,
		TiktokenEnabled:        b.config.TiktokenEnabled,
		ResponseHeaderMode:     b.config.ResponseHeaderMode,
		CredentialNameAsTeamID: b.config.CredentialNameAsTeamID,
		OrganizationPolicies:   b.config.OrganizationPolicies,
	})
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
				Weight:     3,
				LimitRPM:   100,
				LimitTPM:   1000,
				CurrentRPM: 25,
				CurrentTPM: 250,
			},
			"remote_cred_2": {
				Type:       "openai",
				IsFallback: false,
				Weight:     2,
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
				Weight:     5,
				LimitRPM:   50,
				LimitTPM:   500,
				CurrentRPM: 10,
				CurrentTPM: 100,
			},
			"gpt4_cred2": { //nolint:gosec // G101: fake test fixture, not a real credential — this file only backs *_test.go callers
				Credential: "remote_cred_2",
				Model:      "gpt-4",
				Weight:     7,
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
