package router

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/auth"
	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/scope"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type unavailableScopeDB struct {
	*litellmdb.NoopManager
}

func (unavailableScopeDB) IsEnabled() bool { return true }

func (unavailableScopeDB) IsHealthy() bool { return false }

type routerAuthTestDB struct {
	litellmdb.Manager
	tokens map[string]*dbmodels.TokenInfo
}

func (m *routerAuthTestDB) IsEnabled() bool { return true }
func (m *routerAuthTestDB) IsHealthy() bool { return true }
func (m *routerAuthTestDB) ValidateToken(_ context.Context, rawToken string) (*dbmodels.TokenInfo, error) {
	info := m.tokens[rawToken]
	if info == nil {
		return nil, litellmdb.ErrTokenNotFound
	}
	clone := *info
	clone.Models = append([]string(nil), info.Models...)
	clone.UserModels = append([]string(nil), info.UserModels...)
	clone.TeamModels = append([]string(nil), info.TeamModels...)
	clone.TeamMemberModels = append([]string(nil), info.TeamMemberModels...)
	if err := clone.Validate(""); err != nil {
		return nil, err
	}
	return &clone, nil
}

func newIPv4Server(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()

	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("tcp4 listener unavailable in test environment: %v", err)
	}
	server := &httptest.Server{
		Listener: listener,
		Config:   &http.Server{Handler: handler},
	}
	server.Start()
	return server
}

func createTestProxy() *proxy.Proxy {
	return createTestProxyWithStrictACL(false)
}

func createTestProxyWithStrictACL(strictAllTeamModelsACL bool) *proxy.Proxy {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
		{Name: "test2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tokenManager := auth.NewVertexTokenManager(logger)

	return proxy.New(&proxy.Config{
		Balancer:               bal,
		Logger:                 logger,
		MaxBodySizeMB:          10,
		RequestTimeout:         30 * time.Second,
		MaxIdleConns:           200,
		MaxIdleConnsPerHost:    20,
		IdleConnTimeout:        120 * time.Second,
		Metrics:                metrics,
		MasterKey:              "test-master-key",
		RateLimiter:            rl,
		TokenManager:           tokenManager,
		ModelManager:           createTestModelManager(),
		Version:                "test-version",
		Commit:                 "test-commit",
		StrictAllTeamModelsACL: strictAllTeamModelsACL,
	})
}

// createTestModelManager creates a test model manager instance (disabled - no static models)
func createTestModelManager() *models.Manager {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	return models.New(logger, 100, []config.ModelRPMConfig{})
}

// createEnabledTestModelManager creates an enabled model manager with static models
func createEnabledTestModelManager() *models.Manager {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	staticModels := []config.ModelRPMConfig{{Name: "test-model", RPM: 100, TPM: 100000}}
	return models.New(logger, 100, staticModels)
}

// createProxyWithConfig creates a test proxy with custom credentials
func createProxyWithConfig(credentials []config.CredentialConfig, bannedCreds []string) *proxy.Proxy {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(1, 0, []int{500})
	rl := ratelimit.New()

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)

	// Ban specified credentials
	for _, credName := range bannedCreds {
		f2b.RecordResponse(credName, "", 500)
	}

	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	return proxy.New(&proxy.Config{
		Balancer:            bal,
		Logger:              logger,
		MaxBodySizeMB:       10,
		RequestTimeout:      30 * time.Second,
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     120 * time.Second,
		Metrics:             metrics,
		MasterKey:           "test-key",
		RateLimiter:         rl,
		TokenManager:        tm,
		ModelManager:        createTestModelManager(),
		Version:             "test-version",
		Commit:              "test-commit",
	})
}

// createProxyWithMockServer creates a proxy configured with a mock server URL
func createProxyWithMockServer(mockServerURL string) *proxy.Proxy {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServerURL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	tm := auth.NewVertexTokenManager(logger)
	return proxy.New(&proxy.Config{
		Balancer:            bal,
		Logger:              logger,
		MaxBodySizeMB:       10,
		RequestTimeout:      30 * time.Second,
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     120 * time.Second,
		Metrics:             metrics,
		MasterKey:           "test-key",
		RateLimiter:         rl,
		TokenManager:        tm,
		ModelManager:        createTestModelManager(),
		Version:             "test-version",
		Commit:              "test-commit",
	})
}

func TestNew(t *testing.T) {
	prx := createTestProxy()
	modelManager := createTestModelManager()
	monConfig := testhelpers.NewTestMonitoringConfig("/health", false, "")
	logger := testhelpers.NewTestLogger()

	r := New(nil, modelManager, monConfig, logger, nil)

	assert.NotNil(t, r)
	assert.Equal(t, "/health", r.monitoringConfig.HealthCheckPath)
	assert.Equal(t, modelManager, r.modelManager)

	monConfig2 := testhelpers.NewTestMonitoringConfig("/status", false, "")
	r2 := New(prx, nil, monConfig2, logger, nil)
	assert.NotNil(t, r2)
	assert.Equal(t, "/status", r2.monitoringConfig.HealthCheckPath)
}

func TestServeHTTP_HealthCheck(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "healthy", response["status"])
}

func TestServeHTTP_HealthCheck_Unhealthy(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}
	prx := createProxyWithConfig(credentials, []string{"test1"})
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "unhealthy", response["status"])
}

func TestServeHTTP_HealthCheck_NoProviderRouteIsUnavailable(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "no-route", RPM: 100, ProviderScopeExpression: scope.FalseExpression()},
	}
	prx := createProxyWithConfig(credentials, nil)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
}

func TestServeHTTP_HealthCheck_ScopedViewDoesNotDriveStatusCode(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "team-a", APIKey: "key1", BaseURL: "http://team-a.example", RPM: 100, Scopes: []string{"team-a"}},
	}
	prx := createProxyWithConfig(credentials, nil)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "unhealthy", response["status"])
	assert.Equal(t, float64(0), response["total_credentials"])
}

func TestServeHTTP_HealthCheck_UnverifiableTokenFallsBackToPublic(t *testing.T) {
	credentials := []config.CredentialConfig{
		{Name: "team-a", APIKey: "key1", BaseURL: "http://team-a.example", RPM: 100, Scopes: []string{"team-a"}},
	}
	prx := createProxyWithConfig(credentials, nil)
	prx.LiteLLMDB = unavailableScopeDB{NoopManager: litellmdb.NewNoopManager()}
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set("Authorization", "Bearer stale-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "unhealthy", response["status"])
	assert.Equal(t, float64(0), response["total_credentials"])
}

func TestServeHTTP_V1Models_UnverifiableTokenRemainsUnauthorized(t *testing.T) {
	prx := createTestProxy()
	prx.LiteLLMDB = unavailableScopeDB{NoopManager: litellmdb.NewNoopManager()}
	router := New(prx, createEnabledTestModelManager(), testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer stale-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestServeHTTP_V1Models_Enabled(t *testing.T) {
	modelManager := createEnabledTestModelManager()

	prx := createTestProxy()
	router := New(prx, modelManager, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-master-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var response models.ModelsResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	// Empty models is OK for this test, just verifying the endpoint works
}

func TestServeHTTP_V1Models_Disabled(t *testing.T) {
	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": "proxied"})
	}))
	defer mockServer.Close()

	prx := createProxyWithMockServer(mockServer.URL)
	modelManager := createTestModelManager() // disabled (no static models)
	router := New(prx, modelManager, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Should proxy the request instead of handling locally
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServeHTTP_V1Models_NilManager(t *testing.T) {
	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": "proxied"})
	}))
	defer mockServer.Close()

	prx := createProxyWithMockServer(mockServer.URL)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Should proxy the request when model manager is nil
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServeHTTP_ProxyRequest(t *testing.T) {
	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "ok"})
	}))
	defer mockServer.Close()

	prx := createProxyWithMockServer(mockServer.URL)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	tests := []struct {
		name string
		path string
		body string
	}{
		{"chat completions", "/v1/chat/completions", `{"model":"test-model","messages":[{"role":"user","content":"test"}]}`},
		{"completions", "/v1/completions", `{"model":"test-model","prompt":"test"}`},
		{"embeddings", "/v1/embeddings", `{"model":"test-model","input":"test"}`},
		{"images", "/v1/images/generations", `{"model":"test-model","prompt":"test"}`},
		{"image edits", "/v1/images/edits", `{"model":"test-model"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(tt.body)
			req := httptest.NewRequest("POST", tt.path, strings.NewReader(string(body)))
			req.Header.Set("Authorization", "Bearer test-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusOK, w.Code)
		})
	}
}

func TestServeHTTP_NotFound(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	tests := []struct {
		name string
		path string
	}{
		{"root path", "/"},
		{"api path", "/api/test"},
		{"random path", "/random"},
		{"v2 path", "/v2/chat"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", tt.path, nil)
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusNotFound, w.Code)
		})
	}
}

func TestServeHTTPAddsSecurityHeadersToLocalErrors(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodPost, "/not-found", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
	assert.Equal(t, "frame-ancestors 'none'", w.Header().Get("Content-Security-Policy"))
	assert.Equal(t, "DENY", w.Header().Get("X-Frame-Options"))
	assert.Equal(t, "nosniff", w.Header().Get("X-Content-Type-Options"))
}

func TestHandleHealth(t *testing.T) {
	tests := []struct {
		name           string
		bannedCreds    []string
		expectedStatus int
	}{
		{
			name:           "healthy - all available",
			bannedCreds:    []string{},
			expectedStatus: http.StatusOK,
		},
		{
			name:           "unhealthy - all banned",
			bannedCreds:    []string{"test1", "test2"},
			expectedStatus: http.StatusServiceUnavailable,
		},
		{
			name:           "healthy - partially available",
			bannedCreds:    []string{"test1"},
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			credentials := []config.CredentialConfig{
				{Name: "test1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
				{Name: "test2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
			}
			prx := createProxyWithConfig(credentials, tt.bannedCreds)
			router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

			req := httptest.NewRequest("GET", "/health", nil)
			w := httptest.NewRecorder()

			router.handleHealth(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)
			assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

			var response map[string]interface{}
			err := json.Unmarshal(w.Body.Bytes(), &response)
			assert.NoError(t, err)

			if tt.expectedStatus == http.StatusOK {
				assert.Equal(t, "healthy", response["status"])
			} else {
				assert.Equal(t, "unhealthy", response["status"])
			}
		})
	}
}

func TestHandleModels(t *testing.T) {
	modelManager := createEnabledTestModelManager()
	prx := createTestProxy()

	router := New(prx, modelManager, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-master-key")
	w := httptest.NewRecorder()

	router.handleModels(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

	var response models.ModelsResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "list", response.Object)
	// Models list might be empty if not fetched, which is OK
}

func TestServeHTTPV1ModelsAuthAndModelACLPolicy(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	modelManager := models.New(logger, 100, []config.ModelRPMConfig{
		{Name: "z-backend", RPM: 100},
		{Name: "a-backend", RPM: 100},
	})
	modelManager.SetModelAliases(map[string]string{
		"openai/z-public":  "z-backend",
		"openai/a-public":  "a-backend",
		"openai/a-premium": "a-backend",
	})
	modelManager.SetClientModelIDs([]string{"openai/a-public", "openai/z-public"})
	catalogCredentials := []config.CredentialConfig{{Name: "catalog", Type: config.ProviderTypeOpenAI}}
	modelManager.SetCredentials(catalogCredentials)
	modelManager.LoadModelsFromConfig(catalogCredentials)
	prx := createTestProxyWithStrictACL(true)
	blocked := true
	prx.LiteLLMDB = &routerAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"unrestricted-key": {Token: "unrestricted-hash"},
		// The key itself grants both IDs, while its parent scopes grant only the
		// public model. The internal routing target must not be advertised.
		"restricted-key": {
			Token:      "restricted-hash",
			Models:     []string{"openai/a-public", "a-backend"},
			TeamID:     "team-alt",
			TeamModels: []string{"openai/a-public"},
		},
		"blocked-team-key": {
			Token:       "blocked-team-hash",
			TeamID:      "team",
			TeamBlocked: &blocked,
		},
		"no-default-user-key": {
			Token:      "no-default-user-hash",
			Models:     []string{"openai/a-public"},
			UserID:     "personal-user",
			UserModels: []string{dbmodels.NoDefaultModels, "openai/a-public"},
		},
		"dangling-team-key": {
			Token:        "dangling-team-hash",
			Models:       []string{"openai/a-public"},
			TeamID:       "deleted-team",
			TeamDangling: true,
		},
		"wildcard-key": {
			Token:  "wildcard-hash",
			Models: []string{"openai/a-*"},
		},
		"regex-looking-key": {
			Token:  "regex-looking-hash",
			Models: []string{"openai/a.public*"},
		},
	}}
	router := New(prx, modelManager, testhelpers.NewTestMonitoringConfig("/health", false, ""), logger, nil)

	request := func(headers map[string]string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		for key, value := range headers {
			req.Header.Set(key, value)
		}
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w
	}
	modelIDs := func(t *testing.T, w *httptest.ResponseRecorder) []string {
		t.Helper()
		var response models.ModelsResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
		ids := make([]string, 0, len(response.Data))
		for _, model := range response.Data {
			ids = append(ids, model.ID)
		}
		return ids
	}

	missing := request(nil)
	assert.Equal(t, http.StatusUnauthorized, missing.Code)
	assert.Contains(t, missing.Body.String(), `"type":"authentication_error"`)

	invalid := request(map[string]string{"Authorization": "Bearer invalid-key"})
	assert.Equal(t, http.StatusUnauthorized, invalid.Code)
	for _, key := range []string{"blocked-team-key"} {
		blockedResponse := request(map[string]string{"Authorization": "Bearer " + key})
		assert.Equal(t, http.StatusForbidden, blockedResponse.Code)
	}

	restricted := request(map[string]string{"Authorization": "Bearer restricted-key"})
	require.Equal(t, http.StatusOK, restricted.Code)
	assert.Equal(t, []string{"openai/a-public"}, modelIDs(t, restricted))

	noDefault := request(map[string]string{"Authorization": "Bearer no-default-user-key"})
	require.Equal(t, http.StatusOK, noDefault.Code)
	assert.Empty(t, modelIDs(t, noDefault))

	danglingTeam := request(map[string]string{"Authorization": "Bearer dangling-team-key"})
	require.Equal(t, http.StatusOK, danglingTeam.Code)
	assert.Empty(t, modelIDs(t, danglingTeam))

	wildcard := request(map[string]string{"Authorization": "Bearer wildcard-key"})
	require.Equal(t, http.StatusOK, wildcard.Code)
	assert.Equal(t, []string{"openai/a-public"}, modelIDs(t, wildcard),
		"an unknown short backend must not inherit openai/* from its transport credential")

	regexLooking := request(map[string]string{"Authorization": "Bearer regex-looking-key"})
	require.Equal(t, http.StatusOK, regexLooking.Code)
	assert.Empty(t, modelIDs(t, regexLooking))

	unrestricted := request(map[string]string{"x-api-key": "unrestricted-key"})
	require.Equal(t, http.StatusOK, unrestricted.Code)
	assert.Equal(t,
		[]string{"openai/a-public", "openai/z-public"},
		modelIDs(t, unrestricted),
	)

	master := request(map[string]string{"Authorization": "Bearer test-master-key"})
	require.Equal(t, http.StatusOK, master.Code)
	assert.Equal(t, modelIDs(t, unrestricted), modelIDs(t, master))

	restrictedGroupsReq := httptest.NewRequest(http.MethodGet, "/v1/models?include_model_access_groups=true", nil)
	restrictedGroupsReq.Header.Set("Authorization", "Bearer restricted-key")
	restrictedGroups := httptest.NewRecorder()
	router.ServeHTTP(restrictedGroups, restrictedGroupsReq)
	require.Equal(t, http.StatusOK, restrictedGroups.Code)
	assert.Equal(t, []string{"openai/a-public"}, modelIDs(t, restrictedGroups))

	unrestrictedGroupsReq := httptest.NewRequest(http.MethodGet, "/v1/models?include_model_access_groups=true", nil)
	unrestrictedGroupsReq.Header.Set("x-api-key", "unrestricted-key")
	unrestrictedGroups := httptest.NewRecorder()
	router.ServeHTTP(unrestrictedGroups, unrestrictedGroupsReq)
	require.Equal(t, http.StatusOK, unrestrictedGroups.Code)
	assert.Equal(t,
		[]string{"openai/a-public", "openai/z-public"},
		modelIDs(t, unrestrictedGroups),
	)

	compatibilityProxy := createTestProxy()
	compatibilityProxy.LiteLLMDB = prx.LiteLLMDB
	compatibilityRouter := New(
		compatibilityProxy,
		modelManager,
		testhelpers.NewTestMonitoringConfig("/health", false, ""),
		logger,
		nil,
	)
	compatibilityRequest := func(key string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.Header.Set("Authorization", "Bearer "+key)
		w := httptest.NewRecorder()
		compatibilityRouter.ServeHTTP(w, req)
		return w
	}
	allModels := []string{"openai/a-public", "openai/z-public"}
	for _, key := range []string{
		"restricted-key",
		"no-default-user-key",
		"dangling-team-key",
		"wildcard-key",
		"regex-looking-key",
	} {
		response := compatibilityRequest(key)
		require.Equal(t, http.StatusOK, response.Code)
		assert.Equal(t, allModels, modelIDs(t, response))
	}
	assert.Equal(t, http.StatusForbidden, compatibilityRequest("blocked-team-key").Code)
}

func TestServeHTTPPublicPreflightDoesNotEnableWildcardCORS(t *testing.T) {
	router := New(nil, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://client.example.invalid")
	req.Header.Set("Access-Control-Request-Method", http.MethodPost)
	req.Header.Set("Access-Control-Request-Headers", "content-type,x-api-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	assert.Equal(t, http.MethodPost, w.Header().Get("Allow"))
	assert.Empty(t, w.Header().Get("Access-Control-Allow-Origin"))
	assert.Empty(t, w.Header().Get("Access-Control-Allow-Headers"))
	assert.Empty(t, w.Header().Get("Access-Control-Allow-Methods"))
}

func TestServeHTTPWebSocketUpgradeIsCaseInsensitive(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	server := newIPv4Server(t, router)
	defer server.Close()

	conn, err := net.Dial("tcp", strings.TrimPrefix(server.URL, "http://"))
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()

	request := "GET /v1/responses HTTP/1.1\r\n" +
		"Host: test\r\n" +
		"Connection: Upgrade\r\n" +
		"Upgrade: WebSocket\r\n" +
		"Sec-WebSocket-Version: 13\r\n" +
		"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" +
		"Authorization: Bearer test-master-key\r\n\r\n"
	_, err = io.WriteString(conn, request)
	require.NoError(t, err)

	response, err := http.ReadResponse(bufio.NewReader(conn), &http.Request{Method: http.MethodGet})
	require.NoError(t, err)
	defer func() { _ = response.Body.Close() }()
	assert.Equal(t, http.StatusSwitchingProtocols, response.StatusCode)
}

func TestServeHTTPRejectsUnsupportedMethodsBeforeAuth(t *testing.T) {
	router := New(nil, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	chatReq := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	chat := httptest.NewRecorder()
	router.ServeHTTP(chat, chatReq)
	assert.Equal(t, http.StatusMethodNotAllowed, chat.Code)
	assert.Equal(t, http.MethodPost, chat.Header().Get("Allow"))
	assert.Equal(t, "application/json", chat.Header().Get("Content-Type"))
	assert.JSONEq(t, `{"detail":"Method Not Allowed"}`, chat.Body.String())

	modelsReq := httptest.NewRequest(http.MethodPost, "/v1/models", nil)
	modelsResult := httptest.NewRecorder()
	router.ServeHTTP(modelsResult, modelsReq)
	assert.Equal(t, http.StatusMethodNotAllowed, modelsResult.Code)
	assert.Equal(t, http.MethodGet, modelsResult.Header().Get("Allow"))

	// Native Anthropic Messages is intentionally outside the configured public
	// surface; adding CORS must not make it look supported.
	messagesReq := httptest.NewRequest(http.MethodOptions, "/v1/messages", nil)
	messagesReq.Header.Set("Origin", "https://client.example.invalid")
	messagesResult := httptest.NewRecorder()
	router.ServeHTTP(messagesResult, messagesReq)
	assert.Equal(t, http.StatusNotFound, messagesResult.Code)
	assert.Empty(t, messagesResult.Header().Get("Access-Control-Allow-Origin"))
}

func TestServeHTTPV1ModelsWithNilProxyFailsClosed(t *testing.T) {
	router := New(nil, createEnabledTestModelManager(), testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer should-not-be-accepted")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), `"type":"server_error"`)
}

func TestHandleVisualHealth(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/vhealth", nil)
	w := httptest.NewRecorder()

	router.handleVisualHealth(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
	assert.NotEmpty(t, w.Body.String())
	// Should return HTML content
	assert.Contains(t, w.Body.String(), "html")
}

func TestServeHTTP_VisualHealth(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/vhealth", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
}

func TestServeHTTP_StreamingRequestNotLogged(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a mock proxy that returns a 500 error
	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer mockServer.Close()

	prx := createProxyWithMockServer(mockServer.URL)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", true, tmpDir+"/errors.log"), testhelpers.NewTestLogger(), nil)

	// Test: Streaming request should NOT be logged even if status is 500
	streamingBody := []byte(`{"stream":true,"model":"test-model","messages":[{"role":"user","content":"test"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(streamingBody)))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Streaming request should still be processed (500 from mock)
	assert.Equal(t, http.StatusInternalServerError, w.Code)

	// But log file should be empty (streaming requests are not logged)
	logPath := tmpDir + "/errors.log"
	content, err := os.ReadFile(logPath)
	if err == nil {
		// File exists but should be empty
		assert.Empty(t, content, "Streaming requests should not be logged")
	}
	// If file doesn't exist, that's also expected (no logging)
}

func TestServeHTTP_NonStreamingErrorIsLogged(t *testing.T) {
	tmpDir := t.TempDir()
	logPath := tmpDir + "/errors.log"

	// Create a mock proxy that returns a 400 error
	mockServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad request"))
	}))
	defer mockServer.Close()

	prx := createProxyWithMockServer(mockServer.URL)
	router := New(prx, nil, testhelpers.NewTestMonitoringConfig("/health", true, logPath), testhelpers.NewTestLogger(), nil)

	// Test: Non-streaming request SHOULD be logged when status is error
	nonStreamingBody := []byte(`{"stream": false, "model": "test-model"}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(nonStreamingBody)))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Non-streaming request should be processed (400 from mock)
	assert.Equal(t, http.StatusBadRequest, w.Code)

	// Log file should contain the error
	content, err := os.ReadFile(logPath)
	assert.NoError(t, err, "Log file should exist")
	assert.NotEmpty(t, content, "Non-streaming error should be logged")

	// Verify log format
	var entry ErrorLogEntry
	err = json.Unmarshal(content, &entry)
	assert.NoError(t, err, "Log file should contain valid JSON")
	assert.Equal(t, http.StatusBadRequest, entry.Status)
}
