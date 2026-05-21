package models

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/httputil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestGetRemoteModels_Caching(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	// Track number of requests and change response based on request count
	requestCount := 0

	// Create test server that returns different data on first vs subsequent requests
	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			requestCount++
			w.Header().Set("Content-Type", "application/json")
			modelID := "model-a"
			if requestCount > 1 {
				modelID = "model-b"
			}
			_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"upstream-primary": {Type: "openai", IsFallback: false},
				},
				Models: map[string]httputil.ModelHealthStats{
					"m1": {Credential: "upstream-primary", Model: modelID},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	// Create manager with empty static models
	m := New(logger, 100, []config.ModelRPMConfig{})

	// Set up credential for the test server
	cred := &config.CredentialConfig{
		Name:    "proxy-1",
		Type:    config.ProviderTypeProxy,
		BaseURL: server.URL,
	}

	// --- Test 1: First call should fetch from remote server ---
	models1 := m.GetRemoteModels(cred)
	assert.Equal(t, 1, len(models1), "First call should return 1 model")
	assert.Equal(t, "model-a", models1[0].ID, "First call should return model-a")
	assert.Equal(t, 1, requestCount, "First call should make exactly 1 HTTP request")

	// --- Test 2: Second call should use cache (before TTL expires) ---
	models2 := m.GetRemoteModels(cred)
	assert.Equal(t, 1, len(models2), "Second call should return 1 model")
	assert.Equal(t, "model-a", models2[0].ID, "Second call should return cached model-a (not the new model-b)")
	assert.Equal(t, 1, requestCount, "Second call should NOT make HTTP request (using cache)")

	// --- Test 3: After cache expiration, should fetch new data from server ---
	// Set cache expiration to very short TTL to avoid long test waits
	m.cacheExpiration = 1 * time.Millisecond

	// Clear the current cache entry to allow testing TTL expiration
	m.mu.Lock()
	if cached, ok := m.remoteModelsCache[cred.Name]; ok {
		// Set expiration to past time to force cache miss
		cached.expiresAt = time.Now().UTC().Add(-1 * time.Millisecond)
		m.remoteModelsCache[cred.Name] = cached
	}
	m.mu.Unlock()

	// Small delay to ensure cache is considered expired
	time.Sleep(5 * time.Millisecond)

	// Third call should fetch new data
	models3 := m.GetRemoteModels(cred)
	assert.Equal(t, 1, len(models3), "Third call should return 1 model")
	assert.Equal(t, "model-b", models3[0].ID, "Third call should return new model-b from server (cache expired)")
	assert.Equal(t, 2, requestCount, "Third call should make new HTTP request after cache expiration")

	// --- Test 4: Fourth call should cache the new result ---
	models4 := m.GetRemoteModels(cred)
	assert.Equal(t, 1, len(models4), "Fourth call should return 1 model")
	assert.Equal(t, "model-b", models4[0].ID, "Fourth call should return cached model-b")
	assert.Equal(t, 2, requestCount, "Fourth call should NOT make HTTP request (using cache)")
}

func TestGetRemoteModels_CachingMultipleCredentials(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	requestCountProxy1 := 0
	requestCountProxy2 := 0

	// Create test server for proxy-1
	server1 := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			requestCountProxy1++
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"upstream-primary": {Type: "openai", IsFallback: false},
				},
				Models: map[string]httputil.ModelHealthStats{
					"m1": {Credential: "upstream-primary", Model: "proxy1-model"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server1.Close()

	// Create test server for proxy-2
	server2 := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			requestCountProxy2++
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"upstream-primary": {Type: "openai", IsFallback: false},
				},
				Models: map[string]httputil.ModelHealthStats{
					"m1": {Credential: "upstream-primary", Model: "proxy2-model"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server2.Close()

	m := New(logger, 100, []config.ModelRPMConfig{})

	cred1 := &config.CredentialConfig{
		Name:    "proxy-1",
		Type:    config.ProviderTypeProxy,
		BaseURL: server1.URL,
	}

	cred2 := &config.CredentialConfig{
		Name:    "proxy-2",
		Type:    config.ProviderTypeProxy,
		BaseURL: server2.URL,
	}

	// Fetch from proxy-1
	models1a := m.GetRemoteModels(cred1)
	assert.Equal(t, "proxy1-model", models1a[0].ID)
	assert.Equal(t, 1, requestCountProxy1)
	assert.Equal(t, 0, requestCountProxy2)

	// Fetch from proxy-2
	models2a := m.GetRemoteModels(cred2)
	assert.Equal(t, "proxy2-model", models2a[0].ID)
	assert.Equal(t, 1, requestCountProxy1)
	assert.Equal(t, 1, requestCountProxy2)

	// Second fetch from proxy-1 should use cache
	models1b := m.GetRemoteModels(cred1)
	assert.Equal(t, "proxy1-model", models1b[0].ID)
	assert.Equal(t, 1, requestCountProxy1, "Should still be 1 - using cache")
	assert.Equal(t, 1, requestCountProxy2)

	// Second fetch from proxy-2 should use cache
	models2b := m.GetRemoteModels(cred2)
	assert.Equal(t, "proxy2-model", models2b[0].ID)
	assert.Equal(t, 1, requestCountProxy1)
	assert.Equal(t, 1, requestCountProxy2, "Should still be 1 - using cache")
}

func TestGetRemoteModelsWithError_FiltersRemoteHealthByFallbackParity(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&httputil.ProxyHealthResponse{
				Credentials: map[string]httputil.CredentialHealthStats{
					"primary-upstream":  {Type: "openai", IsFallback: false},
					"fallback-upstream": {Type: "openai", IsFallback: true},
				},
				Models: map[string]httputil.ModelHealthStats{
					"m1": {Credential: "primary-upstream", Model: "primary-only"},
					"m2": {Credential: "fallback-upstream", Model: "fallback-only"},
					"m3": {Credential: "primary-upstream", Model: "shared-model"},
					"m4": {Credential: "fallback-upstream", Model: "shared-model"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	m := New(logger, 100, []config.ModelRPMConfig{})

	primaryModels, err := m.GetRemoteModelsWithError(context.Background(), &config.CredentialConfig{
		Name:       "proxy-primary",
		Type:       config.ProviderTypeProxy,
		BaseURL:    server.URL,
		IsFallback: false,
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"primary-only", "shared-model"}, modelIDs(primaryModels))

	fallbackModels, err := m.GetRemoteModelsWithError(context.Background(), &config.CredentialConfig{
		Name:       "proxy-fallback",
		Type:       config.ProviderTypeProxy,
		BaseURL:    server.URL,
		IsFallback: true,
	})
	require.NoError(t, err)
	// Fallback gateway includes ALL upstream credentials (both primary and fallback),
	// so it sees all models: primary-only, fallback-only, and shared-model (deduplicated).
	assert.ElementsMatch(t, []string{"fallback-only", "primary-only", "shared-model"}, modelIDs(fallbackModels))
}

func TestGetRemoteModelsWithError_FallsBackToV1ModelsWhenHealthUnavailable(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			http.NotFound(w, r)
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(ModelsResponse{
				Object: "list",
				Data: []Model{
					{ID: "fallback-model", Object: "model", OwnedBy: "proxy"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	m := New(logger, 100, []config.ModelRPMConfig{})

	models, err := m.GetRemoteModelsWithError(context.Background(), &config.CredentialConfig{
		Name:    "proxy-1",
		Type:    config.ProviderTypeProxy,
		BaseURL: server.URL,
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"fallback-model"}, modelIDs(models))
}

func TestGetRemoteModelsWithError_FallsBackToV1ModelsWhenHealthLacksMetadata(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			// Simulate older/non-AAR proxy returning unrelated JSON shape.
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(ModelsResponse{
				Object: "list",
				Data: []Model{
					{ID: "ignored-health-model", Object: "model"},
				},
			})
		case "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(ModelsResponse{
				Object: "list",
				Data: []Model{
					{ID: "real-model", Object: "model", OwnedBy: "proxy"},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	m := New(logger, 100, []config.ModelRPMConfig{})

	models, err := m.GetRemoteModelsWithError(context.Background(), &config.CredentialConfig{
		Name:    "proxy-1",
		Type:    config.ProviderTypeProxy,
		BaseURL: server.URL,
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"real-model"}, modelIDs(models))
}

func modelIDs(models []Model) []string {
	result := make([]string, 0, len(models))
	for _, model := range models {
		result = append(result, model.ID)
	}
	return result
}
