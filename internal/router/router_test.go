package router

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/balancer"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
)

// createTestProxy creates a test proxy instance
func createTestProxy() *proxy.Proxy {
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

	return proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-master-key", rl)
}

// createTestModelManager creates a test model manager instance
func createTestModelManager(enabled bool) *models.Manager {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	return models.New(logger, enabled, 100, "/tmp/models_test.yaml")
}

func TestNew(t *testing.T) {
	prx := createTestProxy()
	modelManager := createTestModelManager(false)

	r := New(nil, "/health", modelManager)

	assert.NotNil(t, r)
	assert.Equal(t, "/health", r.healthCheckPath)
	assert.Equal(t, modelManager, r.modelManager)

	r2 := New(prx, "/status", nil)
	assert.NotNil(t, r2)
	assert.Equal(t, "/status", r2.healthCheckPath)
}

func TestServeHTTP_HealthCheck(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, "/health", nil)

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
	// Create a proxy with all credentials banned
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(1, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	// Ban the credential
	f2b.RecordResponse("test1", 500)

	metrics := monitoring.New(false)
	prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

	router := New(prx, "/health", nil)

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

func TestServeHTTP_V1Models_Enabled(t *testing.T) {
	// Create a model manager with enabled=true and manually add models
	modelManager := createTestModelManager(true)

	prx := createTestProxy()
	router := New(prx, "/health", modelManager)

	req := httptest.NewRequest("GET", "/v1/models", nil)
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
	// Create mock server to verify proxy behavior
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": "proxied"})
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

	modelManager := createTestModelManager(false) // disabled
	router := New(prx, "/health", modelManager)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Should proxy the request instead of handling locally
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServeHTTP_V1Models_NilManager(t *testing.T) {
	// Create mock server to verify proxy behavior
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": "proxied"})
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

	router := New(prx, "/health", nil)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Should proxy the request when model manager is nil
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServeHTTP_V1Models_PostMethod(t *testing.T) {
	// Create mock server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		w.WriteHeader(http.StatusOK)
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

	modelManager := createTestModelManager(true)
	router := New(prx, "/health", modelManager)

	req := httptest.NewRequest("POST", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	// Should proxy POST requests even if model manager is enabled
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestServeHTTP_ProxyRequest(t *testing.T) {
	// Create mock server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"result": "ok"})
	}))
	defer mockServer.Close()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	f2b := fail2ban.New(3, 0, []int{500})
	rl := ratelimit.New()

	credentials := []config.CredentialConfig{
		{Name: "test1", APIKey: "key1", BaseURL: mockServer.URL, RPM: 100},
	}

	for _, cred := range credentials {
		rl.AddCredential(cred.Name, cred.RPM)
	}

	bal := balancer.New(credentials, f2b, rl)
	metrics := monitoring.New(false)
	prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

	router := New(prx, "/health", nil)

	tests := []struct {
		name string
		path string
	}{
		{"chat completions", "/v1/chat/completions"},
		{"completions", "/v1/completions"},
		{"embeddings", "/v1/embeddings"},
		{"images", "/v1/images/generations"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", tt.path, nil)
			req.Header.Set("Authorization", "Bearer test-key")
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusOK, w.Code)
		})
	}
}

func TestServeHTTP_NotFound(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, "/health", nil)

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
			logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
			f2b := fail2ban.New(1, 0, []int{500})
			rl := ratelimit.New()

			credentials := []config.CredentialConfig{
				{Name: "test1", APIKey: "key1", BaseURL: "http://test1.com", RPM: 100},
				{Name: "test2", APIKey: "key2", BaseURL: "http://test2.com", RPM: 100},
			}

			for _, cred := range credentials {
				rl.AddCredential(cred.Name, cred.RPM)
			}

			bal := balancer.New(credentials, f2b, rl)

			// Ban specified credentials
			for _, credName := range tt.bannedCreds {
				f2b.RecordResponse(credName, 500)
			}

			metrics := monitoring.New(false)
			prx := proxy.New(bal, logger, 10, 30*time.Second, metrics, "test-key", rl)

			router := New(prx, "/health", nil)

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
	modelManager := createTestModelManager(true)
	prx := createTestProxy()

	router := New(prx, "/health", modelManager)

	req := httptest.NewRequest("GET", "/v1/models", nil)
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

func TestHandleVisualHealth(t *testing.T) {
	prx := createTestProxy()
	router := New(prx, "/health", nil)

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
	router := New(prx, "/health", nil)

	req := httptest.NewRequest("GET", "/vhealth", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/html; charset=utf-8", w.Header().Get("Content-Type"))
}
