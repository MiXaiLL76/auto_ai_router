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
	"github.com/mixaill76/auto_ai_router/internal/scope"
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

// When the response written comes from an earlier peer attempt (a later one had
// no response at all), its peer credential name is kept for the spend log.
func TestProxyRequestRetryKeepsPeerCredentialNameOfWrittenResponse(t *testing.T) {
	limited := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Credential-Name", "peer-limited-cred")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`)
	}))
	defer limited.Close()
	down := newIPv4Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	down.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "peer-a", Type: config.ProviderTypeAIR, BaseURL: limited.URL, APIKey: "k1", RPM: 100, TPM: 10000},
		config.CredentialConfig{Name: "peer-b", Type: config.ProviderTypeAIR, BaseURL: down.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000},
	).WithMaxProviderRetries(2).Build()
	db := &nativeWSTestDB{clientAuthTestDB: &clientAuthTestDB{}, entries: make(chan *dbmodels.SpendLogEntry, 1)}
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "gpt-4", &models.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

	require.Equal(t, http.StatusTooManyRequests, w.Code, w.Body.String())
	select {
	case entry := <-db.entries:
		assert.Equal(t, "peer-limited-cred:gpt-4", entry.ModelID)
	case <-time.After(5 * time.Second):
		t.Fatal("no spend log entry")
	}
}

func TestProxyRequestRetryKeepsEarlierHTTPResponseAcrossCredentialTypes(t *testing.T) {
	for _, tc := range []struct {
		name                string
		firstType, lastType config.ProviderType
		status, wantStatus  int
	}{
		{"air to direct, 429", config.ProviderTypeAIR, config.ProviderTypeOpenAI, http.StatusTooManyRequests, http.StatusTooManyRequests},
		{"direct to air, 429", config.ProviderTypeOpenAI, config.ProviderTypeAIR, http.StatusTooManyRequests, http.StatusTooManyRequests},
		{"air to direct, 503", config.ProviderTypeAIR, config.ProviderTypeOpenAI, http.StatusServiceUnavailable, http.StatusServiceUnavailable},
		{"direct to air, 503", config.ProviderTypeOpenAI, config.ProviderTypeAIR, http.StatusServiceUnavailable, http.StatusServiceUnavailable},
		{"air to direct, 401", config.ProviderTypeAIR, config.ProviderTypeOpenAI, http.StatusUnauthorized, http.StatusBadGateway},
		{"direct to air, 401", config.ProviderTypeOpenAI, config.ProviderTypeAIR, http.StatusUnauthorized, http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var firstHits atomic.Int32
			first := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				firstHits.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "17")
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"error":{"message":"original upstream error","type":"server_error"}}`)
			}))
			defer first.Close()
			down := newIPv4Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
			down.Close()

			prx := NewTestProxyBuilder().WithCredentials(
				config.CredentialConfig{Name: "first", Type: tc.firstType, BaseURL: first.URL, APIKey: "k1", RPM: 100, TPM: 10000},
				config.CredentialConfig{Name: "last", Type: tc.lastType, BaseURL: down.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000},
			).WithMaxProviderRetries(2).Build()
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", requestHeadersCases[0].body))

			require.Equal(t, tc.wantStatus, w.Code, w.Body.String())
			assert.Equal(t, int32(1), firstHits.Load())
			assert.Contains(t, w.Body.String(), `"error"`)
			if tc.status == http.StatusTooManyRequests {
				assert.Equal(t, "17", w.Header().Get("Retry-After"))
			}
		})
	}
}

func TestSimulatedStreamRetryDoesNotSelectAIRForDirectSend(t *testing.T) {
	direct := config.CredentialConfig{Name: "direct", Type: config.ProviderTypeOpenAI, BaseURL: "http://direct.invalid", APIKey: "k1", RPM: 100, TPM: 10000}
	air := config.CredentialConfig{Name: "air", Type: config.ProviderTypeAIR, BaseURL: "http://air.invalid", APIKey: "k2", Priority: 1, RPM: 100, TPM: 10000}
	prx := NewTestProxyBuilder().WithCredentials(direct, air).Build()
	tried := map[string]bool{"direct": true}
	_, _, ready, _ := prx.nextRetryCredential(
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		&RequestLogContext{Scope: scope.PublicContext()}, "gpt-4", &direct,
		&orchestratedRequest{baseBody: []byte(requestHeadersCases[0].body), baseProxyBody: []byte(requestHeadersCases[0].body), basePath: "/v1/chat/completions"},
		false, 1, tried, map[string]bool{"direct": true},
	)
	assert.False(t, ready)
	assert.False(t, tried["air"], "AIR must not be consumed by a direct-only retry loop")
}
