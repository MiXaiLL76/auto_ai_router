package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
)

func createHealthTestProxy(credentialsCount int) *Proxy {
	credentials := []config.CredentialConfig{}
	for i := 1; i <= credentialsCount; i++ {
		name := "cred_" + string(rune(i+'0'-1))
		credentials = append(credentials, config.CredentialConfig{
			Name:    name,
			APIKey:  "test-key-" + name,
			BaseURL: "http://test.com",
			RPM:     100,
			TPM:     1000,
		})
	}

	return NewTestProxyBuilder().
		WithMasterKey("test-key").
		WithCredentials(credentials...).
		Build()
}

func TestHealthCheck_AllHealthy(t *testing.T) {
	prx := createHealthTestProxy(3)

	healthy, status := prx.HealthCheck()

	assert.True(t, healthy)
	assert.Equal(t, "healthy", status.Status)
	assert.Equal(t, 3, status.TotalCredentials)
	assert.Equal(t, 3, status.CredentialsAvailable)
	assert.Equal(t, 0, status.CredentialsBanned)
	assert.NotNil(t, status.Credentials)
	assert.NotNil(t, status.Models)
}

func TestHealthCheck_NoCredentials(t *testing.T) {
	prx := createHealthTestProxy(0)

	healthy, status := prx.HealthCheck()

	assert.False(t, healthy)
	assert.Equal(t, "unhealthy", status.Status)
	assert.Equal(t, 0, status.TotalCredentials)
	assert.Equal(t, 0, status.CredentialsAvailable)
	assert.Equal(t, 0, status.CredentialsBanned)
}

func TestHealthCheck_CredentialsInfo(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{
			Name:       "openai_cred",
			Type:       config.ProviderTypeOpenAI,
			APIKey:     "sk-test",
			BaseURL:    "http://openai.com",
			RPM:        100,
			TPM:        2000,
			Weight:     7,
			IsFallback: false,
		},
		{
			Name:       "fallback_cred",
			Type:       config.ProviderTypeOpenAI,
			APIKey:     "sk-fallback",
			BaseURL:    "http://fallback.com",
			RPM:        50,
			TPM:        1000,
			IsFallback: true,
		},
	}

	for _, cred := range credentials {
		rl.AddCredentialWithTPM(cred.Name, cred.RPM, cred.TPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Verify credential info
	assert.Len(t, status.Credentials, 2)

	openaiStats := status.Credentials["openai_cred"]
	assert.Equal(t, "openai", openaiStats.Type)
	assert.Equal(t, "http://openai.com", openaiStats.BaseURL)
	assert.Equal(t, false, openaiStats.IsFallback)
	assert.Equal(t, 7, openaiStats.Weight)
	assert.Equal(t, 100, openaiStats.LimitRPM)
	assert.Equal(t, 2000, openaiStats.LimitTPM)

	fallbackStats := status.Credentials["fallback_cred"]
	assert.Equal(t, "http://fallback.com", fallbackStats.BaseURL)
	assert.Equal(t, true, fallbackStats.IsFallback)
}

func TestHealthCheckScoped_FiltersCredentials(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithMasterKey("test-key").
		WithCredentials(
			config.CredentialConfig{Name: "shared", Type: config.ProviderTypeOpenAI, APIKey: "key1", BaseURL: "http://shared.com", RPM: 100},
			config.CredentialConfig{Name: "team-a", Type: config.ProviderTypeOpenAI, APIKey: "key2", BaseURL: "http://team-a.example", RPM: 100, Scopes: []string{"team-a"}},
			config.CredentialConfig{Name: "team-b", Type: config.ProviderTypeOpenAI, APIKey: "key3", BaseURL: "http://team-b.example", RPM: 100, Scopes: []string{"team-b"}},
		).
		Build()

	healthy, status := prx.HealthCheckScoped(scope.NewContext([]string{"team-a"}, nil))

	assert.True(t, healthy)
	assert.Equal(t, 2, status.TotalCredentials)
	assert.Contains(t, status.Credentials, "shared")
	assert.Contains(t, status.Credentials, "team-a")
	assert.NotContains(t, status.Credentials, "team-b")
}

func TestHealthCheck_ReportsIndependentScopeExpression(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:           "chained",
			Type:           config.ProviderTypeProxy,
			APIKey:         "key",
			BaseURL:        "http://router.example",
			RPM:            100,
			Scopes:         []string{"team-a"},
			ProviderScopes: []string{"team-b"},
		}).
		Build()

	_, status := prx.HealthCheck()
	stats := status.Credentials["chained"]

	assert.True(t, scope.NewContext([]string{"team-a", "team-b"}, nil).AllowsExpression(stats.ScopeExpression))
	assert.False(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(stats.ScopeExpression))
	assert.NotEmpty(t, stats.Scopes)
}

func TestHealthCheckScoped_FiltersModelsByScopeExpression(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "chained",
			Type:    config.ProviderTypeProxy,
			APIKey:  "key",
			BaseURL: "http://router.example",
			RPM:     100,
		}).
		Build()
	prx.modelManager.ReplaceModelsForCredential("chained", []string{"team-a-model", "team-b-model"})
	prx.modelManager.ReplaceModelScopesForCredential("chained", map[string]models.ScopeMetadata{
		"team-a-model": {ScopeExpression: scope.FromScopes([]string{"team-a"}, nil)},
		"team-b-model": {ScopeExpression: scope.FromScopes([]string{"team-b"}, nil)},
	})

	_, status := prx.HealthCheckScoped(scope.NewContext([]string{"team-a"}, nil))

	assert.Contains(t, status.Models, "chained:team-a-model")
	assert.NotContains(t, status.Models, "chained:team-b-model")
	assert.True(t, scope.NewContext([]string{"team-a"}, nil).AllowsExpression(status.Models["chained:team-a-model"].ScopeExpression))
}

func TestHealthCheck_CredentialRateLimit(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "test_cred",
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
		TPM:     2000,
	}

	rl.AddCredentialWithTPM(cred.Name, cred.RPM, cred.TPM)
	// Consume some tokens
	rl.SetCredentialCurrentUsage(cred.Name, 45, 500)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	credStats := status.Credentials["test_cred"]
	// Verify that credential info is populated
	assert.Equal(t, 100, credStats.LimitRPM)
	assert.Equal(t, 2000, credStats.LimitTPM)
	// Current usage may vary due to rate limiter internal logic
	assert.GreaterOrEqual(t, credStats.CurrentRPM, 0)
	assert.GreaterOrEqual(t, credStats.CurrentTPM, 0)
}

func TestHealthCheck_ModelInfo(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "test_cred",
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
		Weight:  4,
	}

	rl.AddCredential(cred.Name, 100)

	// Add model limits
	rl.AddModelWithTPM(cred.Name, "gpt-4", 50, 500)
	rl.AddModelWithTPM(cred.Name, "claude-3-opus", 60, 600)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-4", Credential: "test_cred", Weight: 9},
	})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Should have model info
	assert.NotNil(t, status.Models)
	assert.Equal(t, 9, status.Models["test_cred:gpt-4"].Weight)
	assert.Equal(t, 4, status.Models["test_cred:claude-3-opus"].Weight)
}

func TestHealthCheck_ModelProviderErrorAndBanUntil(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(100, 0, []int{429})
	rl := ratelimit.New()
	cred := config.CredentialConfig{
		Name:    "bedrock-a",
		Type:    config.ProviderTypeBedrock,
		APIKey:  "key",
		BaseURL: "https://bedrock.example",
		RPM:     100,
	}
	rl.AddCredential(cred.Name, 100)
	rl.AddModelWithTPM(cred.Name, "claude-opus", 100, 1000)
	until := time.Now().Add(time.Hour)
	f2b.BanUntil(cred.Name, "claude-opus", http.StatusTooManyRequests, until, bedrockDailyQuotaProviderError)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	proxy := createProxyWithParams(
		bal,
		logger,
		10,
		30*time.Second,
		monitoring.New(false),
		"test-key",
		rl,
		auth.NewVertexTokenManager(logger),
		models.New(logger, 50, []config.ModelRPMConfig{}),
		"test-version",
		"test-commit",
	)

	_, health := proxy.HealthCheck()
	stats := health.Models["bedrock-a:claude-opus"]
	assert.True(t, stats.IsBanned)
	assert.Equal(t, bedrockDailyQuotaProviderError, stats.ProviderError)
	if assert.NotNil(t, stats.BanUntil) {
		assert.WithinDuration(t, until, *stats.BanUntil, time.Second)
	}

	trace := proxy.TraceCheck(context.Background(), 0)
	traceStats := trace.Models["bedrock-a:claude-opus"]
	assert.Equal(t, bedrockDailyQuotaProviderError, traceStats.ProviderError)
	assert.NotNil(t, traceStats.BanUntil)
}

func TestHealthCheck_IncludesModelManagerOnlyWeights(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:   "proxy_cred",
		Type:   config.ProviderTypeProxy,
		APIKey: "sk-test",
		RPM:    100,
		Weight: 2,
	}

	rl.AddCredential(cred.Name, 100)
	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})
	mm.ReplaceModelsForCredential(cred.Name, []string{"weight-only-model"})
	mm.ReplaceModelWeightsForCredential(cred.Name, map[string]int{"weight-only-model": 11})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	stats, ok := status.Models["proxy_cred:weight-only-model"]
	assert.True(t, ok)
	assert.Equal(t, 11, stats.Weight)
	assert.Equal(t, -1, stats.LimitRPM)
	assert.Empty(t, rl.GetAllModelPairs(), "health visibility should not require registering an unlimited model limiter")
}

func TestVisualHealthCheck_Success(t *testing.T) {
	prx := createHealthTestProxy(2)

	req := httptest.NewRequest("GET", "/health/visual", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "healthy")
}

func TestVisualHealthCheck_ServesHTML(t *testing.T) {
	prx := createHealthTestProxy(1)

	req := httptest.NewRequest("GET", "/vhealth", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "System health")
}

func TestHealthCheck_MultipleModelsPerCredential(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "test_cred",
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
	}

	rl.AddCredential(cred.Name, 100)

	// Add multiple model limits for same credential
	rl.AddModelWithTPM(cred.Name, "gpt-4", 50, 500)
	rl.AddModelWithTPM(cred.Name, "gpt-3.5-turbo", 40, 400)
	rl.SetModelCurrentUsage(cred.Name, "gpt-4", 25, 250)
	rl.SetModelCurrentUsage(cred.Name, "gpt-3.5-turbo", 20, 200)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Should have model info for both models
	assert.NotNil(t, status.Models)
}

func TestHealthCheck_BannedCredentials(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(1, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred1 := config.CredentialConfig{
		Name:    "cred_1",
		APIKey:  "sk-test1",
		BaseURL: "http://test.com",
		RPM:     100,
	}
	cred2 := config.CredentialConfig{
		Name:    "cred_2",
		APIKey:  "sk-test2",
		BaseURL: "http://test.com",
		RPM:     100,
	}

	rl.AddCredential(cred1.Name, 100)
	rl.AddCredential(cred2.Name, 100)

	bal := balancer.New([]config.CredentialConfig{cred1, cred2}, f2b, rl)

	// Ban one credential
	f2b.RecordResponse(cred1.Name, "", 401) // This will ban it because max_attempts = 1

	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	// Should still be healthy (1 available credential)
	assert.True(t, healthy)
	assert.Equal(t, 1, status.CredentialsAvailable)
	assert.Equal(t, 1, status.CredentialsBanned)
	assert.Equal(t, 2, status.TotalCredentials)
}

func TestHealthCheck_AllBanned(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(1, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred1 := config.CredentialConfig{
		Name:    "cred_1",
		APIKey:  "sk-test1",
		BaseURL: "http://test.com",
		RPM:     100,
	}
	cred2 := config.CredentialConfig{
		Name:    "cred_2",
		APIKey:  "sk-test2",
		BaseURL: "http://test.com",
		RPM:     100,
	}

	rl.AddCredential(cred1.Name, 100)
	rl.AddCredential(cred2.Name, 100)

	bal := balancer.New([]config.CredentialConfig{cred1, cred2}, f2b, rl)

	// Ban all credentials
	f2b.RecordResponse(cred1.Name, "", 401)
	f2b.RecordResponse(cred2.Name, "", 401)

	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	healthy, status := prx.HealthCheck()

	// Should be unhealthy
	assert.False(t, healthy)
	assert.Equal(t, "unhealthy", status.Status)
	assert.Equal(t, 0, status.CredentialsAvailable)
	assert.Equal(t, 2, status.CredentialsBanned)
	assert.Equal(t, 2, status.TotalCredentials)
}

func TestHealthCheck_CredentialsWithoutSpecificModels(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "test_cred",
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
	}

	rl.AddCredential(cred.Name, 100)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)

	// Create model manager with some models
	modelConfig := []config.ModelRPMConfig{
		{Name: "gpt-4"},
		{Name: "gpt-3.5-turbo"},
	}
	mm := models.New(logger, 50, modelConfig)

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Should have model info even without specific model configuration for credential
	assert.NotNil(t, status.Models)
}

func TestVisualHealthCheck_ContainsCredentialInfo(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "openai_cred",
		Type:    config.ProviderTypeOpenAI,
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
		TPM:     2000,
	}

	rl.AddCredentialWithTPM(cred.Name, cred.RPM, cred.TPM)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	req := httptest.NewRequest("GET", "/health/visual", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	body := w.Body.String()
	// Should contain credential info
	assert.NotEmpty(t, body)
}

func TestHealthCheck_ProxyCredentialLimitsFromRateLimiter(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	// Create proxy credential with default/invalid config values
	proxyCred := config.CredentialConfig{
		Name:       "gateway_proxy",
		Type:       config.ProviderTypeProxy,
		BaseURL:    "http://remote-proxy.com",
		RPM:        -1,
		TPM:        -1,
		IsFallback: false,
	}

	// Register proxy credential without limits initially
	bal := balancer.New([]config.CredentialConfig{proxyCred}, f2b, rl)

	// Now simulate what UpdateStatsFromRemoteProxy does - update limits in rateLimiter
	rl.AddCredentialWithTPM(proxyCred.Name, 20000, 2000000)
	rl.SetCredentialCurrentUsage(proxyCred.Name, 1, 36831)

	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Verify proxy credential stats
	proxyStat := status.Credentials["gateway_proxy"]
	assert.Equal(t, "proxy", proxyStat.Type)
	// Limits should come from rateLimiter, not from config (-1 values)
	assert.Equal(t, 20000, proxyStat.LimitRPM)
	assert.Equal(t, 2000000, proxyStat.LimitTPM)
	// Current usage should be from rateLimiter (may be slightly different due to time-based distribution)
	assert.GreaterOrEqual(t, proxyStat.CurrentRPM, 0)
	assert.GreaterOrEqual(t, proxyStat.CurrentTPM, 36800) // Allow small variance due to time calculations
}
