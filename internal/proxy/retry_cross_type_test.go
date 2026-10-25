package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/models"
)

// A retry reaches any credential serving the model, AIR or direct, in both directions.
func TestProxyRequestRetryCrossesAIRAndDirectCredentials(t *testing.T) {
	for _, tc := range []struct {
		name             string
		failType, okType config.ProviderType
	}{
		{"air fails, direct serves", config.ProviderTypeAIR, config.ProviderTypeOpenAI},
		{"direct fails, air serves", config.ProviderTypeOpenAI, config.ProviderTypeAIR},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var failed, served atomic.Int32
			failing := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				failed.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = io.WriteString(w, `{"error":{"message":"unavailable","type":"server_error"}}`)
			}))
			defer failing.Close()
			serving := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				served.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, requestHeadersCases[0].response)
			}))
			defer serving.Close()

			prx := NewTestProxyBuilder().WithCredentials(
				config.CredentialConfig{Name: "first", Type: tc.failType, BaseURL: failing.URL, APIKey: "k1", RPM: 100, TPM: 10000},
				config.CredentialConfig{Name: "second", Type: tc.okType, BaseURL: serving.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000},
			).WithMaxProviderRetries(2).Build()
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Equal(t, int32(1), failed.Load())
			assert.Equal(t, int32(1), served.Load())
		})
	}
}

// A request from a peer router whose direct credential failed must not hop to
// another router: routers listing each other would bounce it back and forth.
func TestProxyRequestRetryFromPeerDoesNotHopDirectToAIR(t *testing.T) {
	var airHits atomic.Int32
	failing := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"unavailable","type":"server_error"}}`)
	}))
	defer failing.Close()
	air := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		airHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, requestHeadersCases[0].response)
	}))
	defer air.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "direct", Type: config.ProviderTypeOpenAI, BaseURL: failing.URL, APIKey: "k1", RPM: 100, TPM: 10000},
		config.CredentialConfig{Name: "peer", Type: config.ProviderTypeAIR, BaseURL: air.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000},
	).WithMaxProviderRetries(2).Build()
	req := newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body)
	req.Header.Set(HeaderAIRProxyClient, "1")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusServiceUnavailable, w.Code, w.Body.String())
	assert.Equal(t, int32(0), airHits.Load())
}

// The credential name a failed peer router reported must not stick to the
// spend log of the credential that served the retry.
func TestProxyRequestRetryDropsFailedPeerCredentialName(t *testing.T) {
	peer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Credential-Name", "peer-failed-cred")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"unavailable","type":"server_error"}}`)
	}))
	defer peer.Close()
	serving := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, requestHeadersCases[0].response)
	}))
	defer serving.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "peer", Type: config.ProviderTypeAIR, BaseURL: peer.URL, APIKey: "k1", RPM: 100, TPM: 10000},
		config.CredentialConfig{Name: "direct", Type: config.ProviderTypeOpenAI, BaseURL: serving.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000},
	).WithMaxProviderRetries(2).Build()
	db := &nativeWSTestDB{clientAuthTestDB: &clientAuthTestDB{}, entries: make(chan *dbmodels.SpendLogEntry, 1)}
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "gpt-4", &models.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	select {
	case entry := <-db.entries:
		assert.Equal(t, "direct:gpt-4", entry.ModelID)
	case <-time.After(5 * time.Second):
		t.Fatal("no spend log entry")
	}
}
