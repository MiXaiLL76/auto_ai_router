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
	tokens       map[string]*dbmodels.TokenInfo
	spendLogging bool
}

func (m *routerAuthTestDB) IsEnabled() bool           { return true }
func (m *routerAuthTestDB) IsHealthy() bool           { return true }
func (m *routerAuthTestDB) SpendLoggingEnabled() bool { return m.spendLogging }
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
	manager := createTestModelManager()
	for i := range credentials {
		manager.AddModel(credentials[i].Name, "test-model")
	}

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
		ModelManager:           manager,
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
	manager := createTestModelManager()
	manager.AddModel("test1", "test-model")
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
		ModelManager:        manager,
		Version:             "test-version",
		Commit:              "test-commit",
	})
}

func createProxyWithModelManager(modelManager *models.Manager, credentials []config.CredentialConfig, strict bool) *proxy.Proxy {
	logger := testhelpers.NewTestLogger()
	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	for i := range credentials {
		rl.AddCredential(credentials[i].Name, credentials[i].RPM)
	}
	return proxy.New(&proxy.Config{
		Balancer:               balancer.New(credentials, f2b, rl),
		Logger:                 logger,
		MaxBodySizeMB:          10,
		RequestTimeout:         30 * time.Second,
		Metrics:                monitoring.New(false),
		MasterKey:              "test-master-key",
		RateLimiter:            rl,
		TokenManager:           auth.NewVertexTokenManager(logger),
		ModelManager:           modelManager,
		StrictAllTeamModelsACL: strict,
	})
}

func TestNew(t *testing.T) {
	prx := createTestProxy()
	monConfig := testhelpers.NewTestMonitoringConfig("/health", false, "")
	logger := testhelpers.NewTestLogger()

	r := New(nil, monConfig, logger, nil)

	assert.NotNil(t, r)
	assert.Equal(t, "/health", r.monitoringConfig.HealthCheckPath)
	monConfig2 := testhelpers.NewTestMonitoringConfig("/status", false, "")
	r2 := New(prx, monConfig2, logger, nil)
	assert.NotNil(t, r2)
	assert.Equal(t, "/status", r2.monitoringConfig.HealthCheckPath)
}

func TestServeHTTP_HealthCheck(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer stale-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestServeHTTP_V1Models_Enabled(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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

func TestCanonicalPublicPath(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{"/chat/completions", "/v1/chat/completions"},
		{"/chat/completions/", "/v1/chat/completions"},
		{"/completions", "/v1/completions"},
		{"/embeddings", "/v1/embeddings"},
		{"/image/generations", "/v1/images/generations"},
		{"/images/generations", "/v1/images/generations"},
		{"/images/edits", "/v1/images/edits"},
		{"/messages", "/v1/messages"},
		{"/models", "/v1/models"},
		{"/responses", "/v1/responses"},
		{"/responses/resp_123", "/v1/responses/resp_123"},
		{"/responses/resp_123/", "/v1/responses/resp_123/"},
		{"/v1/chat/completions", "/v1/chat/completions"},
		{"/health", "/health"},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			assert.Equal(t, tt.want, canonicalPublicPath(tt.path))
		})
	}
}

func TestServeHTTPLegacyChatCompletionsAlias(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "value", r.URL.Query().Get("query"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer upstream.Close()

	prx := createProxyWithMockServer(upstream.URL)
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodPost, "/chat/completions?query=value", strings.NewReader(`{
		"model":"test-model",
		"messages":[{"role":"user","content":"test"}]
	}`))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	result := httptest.NewRecorder()

	router.ServeHTTP(result, req)

	require.Equal(t, http.StatusOK, result.Code)
}

func TestServeHTTP_Messages(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		var body map[string]interface{}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		messages := body["messages"].([]interface{})
		require.Len(t, messages, 2)
		assert.Equal(t, "system", messages[0].(map[string]interface{})["role"])
		assert.Equal(t, "user", messages[1].(map[string]interface{})["role"])

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"test-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
		}`)
	}))
	defer upstream.Close()

	prx := createProxyWithMockServer(upstream.URL)
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"test-model",
		"max_tokens":64,
		"system":"Be concise",
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	result := httptest.NewRecorder()

	router.ServeHTTP(result, req)

	require.Equal(t, http.StatusOK, result.Code)
	assert.Equal(t, "application/json", result.Header().Get("Content-Type"))
	var response map[string]interface{}
	require.NoError(t, json.Unmarshal(result.Body.Bytes(), &response))
	assert.Equal(t, "message", response["type"])
	assert.Equal(t, "assistant", response["role"])
	assert.Equal(t, "end_turn", response["stop_reason"])
	assert.Equal(t, "hello", response["content"].([]interface{})[0].(map[string]interface{})["text"])
}

func TestServeHTTP_MessagesStreaming(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"test-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	prx := createProxyWithMockServer(upstream.URL)
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"test-model",
		"max_tokens":64,
		"stream":true,
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	result := httptest.NewRecorder()

	router.ServeHTTP(result, req)

	require.Equal(t, http.StatusOK, result.Code)
	assert.Contains(t, result.Header().Get("Content-Type"), "text/event-stream")
	assert.Contains(t, result.Body.String(), "event: message_start")
	assert.Contains(t, result.Body.String(), `"text":"hello","type":"text_delta"`)
	assert.Contains(t, result.Body.String(), `"stop_reason":"end_turn"`)
	assert.Contains(t, result.Body.String(), "event: message_stop")
}

func TestServeHTTP_MessagesUsesAnthropicErrorShape(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{
		"model":"test-model",
		"max_tokens":64,
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Content-Type", "application/json")
	result := httptest.NewRecorder()

	router.ServeHTTP(result, req)

	require.Equal(t, http.StatusUnauthorized, result.Code)
	assert.JSONEq(t, `{
		"type":"error",
		"error":{"type":"authentication_error","message":"Missing Authorization header"}
	}`, result.Body.String())
}

func TestServeHTTP_NotFound(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
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
			router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	prx := createTestProxy()

	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	prx := createProxyWithModelManager(modelManager, catalogCredentials, true)
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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), logger, nil)

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
	assert.Equal(t,
		[]string{"a-backend", "openai/a-premium", "openai/a-public", "openai/z-public", "z-backend"},
		modelIDs(t, master),
	)

	// GET /v1/models honours a key's explicit allowlist regardless of
	// strict_all_team_models_acl: the listing must never advertise a model the
	// caller was not granted, even where inference admission stays permissive.
	compatibilityProxy := createProxyWithModelManager(modelManager, catalogCredentials, false)
	compatibilityProxy.LiteLLMDB = prx.LiteLLMDB
	compatibilityRouter := New(
		compatibilityProxy,
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
	compatibilityExpected := map[string][]string{
		"restricted-key":      {"openai/a-public", "openai/z-public"},
		"no-default-user-key": {"openai/a-public", "openai/z-public"},
		"dangling-team-key":   {"openai/a-public", "openai/z-public"},
		"wildcard-key":        {"openai/a-public", "openai/z-public"},
		"regex-looking-key":   {"openai/a-public", "openai/z-public"},
	}
	for key, expected := range compatibilityExpected {
		response := compatibilityRequest(key)
		require.Equal(t, http.StatusOK, response.Code)
		assert.Equal(t, expected, modelIDs(t, response), "key %s", key)
	}
	assert.Equal(t, http.StatusForbidden, compatibilityRequest("blocked-team-key").Code)
}

func TestServeHTTPV1ModelsHidesUnpricedModels(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	modelManager := models.New(logger, 100, []config.ModelRPMConfig{
		{Name: "priced-backend", RPM: 100},
		{Name: "unpriced-backend", RPM: 100},
	})
	modelManager.SetModelAliases(map[string]string{
		"openai/priced":   "priced-backend",
		"openai/unpriced": "unpriced-backend",
	})
	modelManager.SetClientModelIDs([]string{"openai/priced", "openai/unpriced"})
	catalogCredentials := []config.CredentialConfig{{Name: "catalog", Type: config.ProviderTypeOpenAI}}
	modelManager.SetCredentials(catalogCredentials)
	modelManager.LoadModelsFromConfig(catalogCredentials)

	registry := models.NewModelPriceRegistry()
	registry.ReplaceFilePrices(map[string]*models.ModelPrice{
		"openai/priced":  {InputCostPerToken: 0.001},
		"priced-backend": {InputCostPerToken: 0.001},
	})

	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	tokenManager := auth.NewVertexTokenManager(logger)
	defer tokenManager.Stop()
	prx := proxy.New(&proxy.Config{
		Balancer:       balancer.New(catalogCredentials, f2b, rl),
		Logger:         logger,
		MaxBodySizeMB:  10,
		RequestTimeout: 30 * time.Second,
		Metrics:        monitoring.New(false),
		MasterKey:      "test-master-key",
		RateLimiter:    rl,
		TokenManager:   tokenManager,
		ModelManager:   modelManager,
		PriceRegistry:  registry,
	})
	// routerAuthTestDB reports IsEnabled()==true and no SpendLoggingEnabled, so
	// postgres spend tracking is on — the gate that makes unpriced models fatal.
	prx.LiteLLMDB = &routerAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{}, spendLogging: true}
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), logger, nil)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-master-key")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	var response models.ModelsResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	ids := make([]string, 0, len(response.Data))
	for _, model := range response.Data {
		ids = append(ids, model.ID)
	}
	assert.Equal(t, []string{"openai/priced", "priced-backend"}, ids,
		"a model without a resolvable price must not be advertised while spend tracking is on")
}

func TestServeHTTPV1ModelsOrganizationPolicyUsesMappedACLTarget(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	credential := config.CredentialConfig{Name: "catalog", Type: config.ProviderTypeOpenAI, RPM: 100}
	modelManager := models.New(logger, 100, []config.ModelRPMConfig{{Name: "route-b", Credential: credential.Name}})
	modelManager.SetCredentials([]config.CredentialConfig{credential})
	modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	pricePath := t.TempDir() + "/prices.json"
	require.NoError(t, os.WriteFile(pricePath, []byte(`{"public/shared":{"input_cost_per_token":0.001}}`), 0600))
	policies, err := models.LoadOrganizationPolicies([]config.OrganizationPolicyConfig{{
		OrganizationID:  "org-1",
		PriceProfileID:  "profile-1",
		ModelPricesLink: pricePath,
		ModelMappings:   map[string]string{"public/shared": "route-b"},
	}}, modelManager, models.OrganizationPolicyLoadOptions{
		LiteLLMDBEnabled: true, LiteLLMDBRequired: true, DisableSpendLogsWrite: false,
	})
	require.NoError(t, err)

	f2b := fail2ban.New(3, 0, []int{401, 403, 500})
	rl := ratelimit.New()
	rl.AddCredential(credential.Name, credential.RPM)
	tokenManager := auth.NewVertexTokenManager(logger)
	defer tokenManager.Stop()
	prx := proxy.New(&proxy.Config{
		Balancer:               balancer.New([]config.CredentialConfig{credential}, f2b, rl),
		Logger:                 logger,
		MaxBodySizeMB:          10,
		RequestTimeout:         30 * time.Second,
		Metrics:                monitoring.New(false),
		MasterKey:              "test-master-key",
		RateLimiter:            rl,
		TokenManager:           tokenManager,
		ModelManager:           modelManager,
		StrictAllTeamModelsACL: true,
		OrganizationPolicies:   policies,
	})
	prx.LiteLLMDB = &routerAuthTestDB{tokens: map[string]*dbmodels.TokenInfo{
		"organization-key": {
			Token: "organization-hash", DirectOrganizationID: "org-1", OrganizationID: "org-1",
			Models: []string{"route-b"},
		},
	}}
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), logger, nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer organization-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var response models.ModelsResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	ids := make([]string, 0, len(response.Data))
	for _, model := range response.Data {
		ids = append(ids, model.ID)
	}
	assert.Contains(t, ids, "public/shared")
}

func TestServeHTTPPublicPreflightDoesNotEnableWildcardCORS(t *testing.T) {
	router := New(nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
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
	router := New(nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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

	messagesReq := httptest.NewRequest(http.MethodOptions, "/v1/messages", nil)
	messagesReq.Header.Set("Origin", "https://client.example.invalid")
	messagesResult := httptest.NewRecorder()
	router.ServeHTTP(messagesResult, messagesReq)
	assert.Equal(t, http.StatusMethodNotAllowed, messagesResult.Code)
	assert.Equal(t, http.MethodPost, messagesResult.Header().Get("Allow"))
	assert.Empty(t, messagesResult.Header().Get("Access-Control-Allow-Origin"))
}

func TestServeHTTPV1ModelsWithNilProxyFailsClosed(t *testing.T) {
	router := New(nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer should-not-be-accepted")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Contains(t, w.Body.String(), `"type":"server_error"`)
}

func TestHandleVisualHealth(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", true, tmpDir+"/errors.log"), testhelpers.NewTestLogger(), nil)

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
	router := New(prx, testhelpers.NewTestMonitoringConfig("/health", true, logPath), testhelpers.NewTestLogger(), nil)

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
