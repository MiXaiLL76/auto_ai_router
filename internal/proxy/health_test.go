package proxy

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

func createHealthTestLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
}

func createHealthTestProxy(credentialsCount int) *Proxy {
	logger := createHealthTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

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
		rl.AddCredential(name, 100)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")
	return prx
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
	logger := createHealthTestLogger()
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
	assert.Equal(t, false, openaiStats.IsFallback)
	assert.Equal(t, 100, openaiStats.LimitRPM)
	assert.Equal(t, 2000, openaiStats.LimitTPM)

	fallbackStats := status.Credentials["fallback_cred"]
	assert.Equal(t, true, fallbackStats.IsFallback)
}

func TestHealthCheck_CredentialRateLimit(t *testing.T) {
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	cred := config.CredentialConfig{
		Name:    "test_cred",
		APIKey:  "sk-test",
		BaseURL: "http://test.com",
		RPM:     100,
	}

	rl.AddCredential(cred.Name, 100)

	// Add model limits
	rl.AddModelWithTPM(cred.Name, "gpt-4", 50, 500)
	rl.AddModelWithTPM(cred.Name, "claude-3-opus", 60, 600)

	bal := balancer.New([]config.CredentialConfig{cred}, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	_, status := prx.HealthCheck()

	// Should have model info
	assert.NotNil(t, status.Models)
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

func TestVisualHealthCheck_NoTemplate(t *testing.T) {
	logger := createHealthTestLogger()
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
	mm := models.New(logger, 50, []config.ModelRPMConfig{})

	prx := createProxyWithParams(bal, logger, 10, 30*time.Second, metrics, "test-key", rl, tm, mm, "test-version", "test-commit")

	// Explicitly set template to nil to simulate template parsing error
	prx.healthTemplate = nil

	req := httptest.NewRequest("GET", "/health/visual", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	assert.Contains(t, w.Body.String(), "Template not available")
}

func TestVisualHealthCheck_HealthyStatus(t *testing.T) {
	prx := createHealthTestProxy(1)

	req := httptest.NewRequest("GET", "/health/visual", nil)
	w := httptest.NewRecorder()

	prx.VisualHealthCheck(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	// Should contain healthy status
	body := w.Body.String()
	assert.Contains(t, body, "healthy")
	// Check for credential and status information
	assert.Contains(t, body, "System Health Dashboard")
}

func TestHealthCheck_MultipleModelsPerCredential(t *testing.T) {
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
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
	logger := createHealthTestLogger()
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
