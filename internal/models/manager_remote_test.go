package models

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestGetRemoteModels_Caching(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))

	// Track number of requests and change response based on request count
	requestCount := 0

	// Create test server that returns different data on first vs subsequent requests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++

		w.Header().Set("Content-Type", "application/json")

		if requestCount == 1 {
			// First request returns model-a
			resp := ModelsResponse{
				Object: "list",
				Data: []Model{
					{ID: "model-a", Object: "model", OwnedBy: "test-provider"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			// Subsequent requests return model-b (simulating server-side change)
			resp := ModelsResponse{
				Object: "list",
				Data: []Model{
					{ID: "model-b", Object: "model", OwnedBy: "test-provider"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
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
		cached.expiresAt = time.Now().Add(-1 * time.Millisecond)
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
	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCountProxy1++
		w.Header().Set("Content-Type", "application/json")
		resp := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "proxy1-model", Object: "model", OwnedBy: "proxy1"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server1.Close()

	// Create test server for proxy-2
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCountProxy2++
		w.Header().Set("Content-Type", "application/json")
		resp := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "proxy2-model", Object: "model", OwnedBy: "proxy2"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
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
