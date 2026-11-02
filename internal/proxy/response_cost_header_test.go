package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func costHeaderProxy(t *testing.T, upstreamURL string) *Proxy {
	t.Helper()
	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "c", Type: config.ProviderTypeOpenAI, BaseURL: upstreamURL,
		APIKey: "k", RPM: 100, TPM: 100000,
	}).WithMasterKey("master-key").Build()
	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"gpt-cost": {InputCostPerToken: 1, OutputCostPerToken: 2},
	})
	prx.priceRegistry = registry
	prx.LiteLLMDB = &stubLiteLLMManager{}
	return prx
}

// A non-streaming request whose body is written after the cost is computed
// carries X-Litellm-Response-Cost with the client-billed total.
func TestResponseCostHeaderOnNonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","model":"gpt-cost","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer upstream.Close()

	prx := costHeaderProxy(t, upstream.URL)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-cost","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	// 10 input * 1 + 5 output * 2 = 20.
	assert.Equal(t, "20", w.Header().Get(HeaderResponseCost))
}

// A provider's own cost header must not reach the client; only the router's own
// computed cost does.
func TestResponseCostHeaderOverridesProviderValue(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Litellm-Response-Cost", "999") // provider's, must not leak
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","model":"gpt-cost","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	prx := costHeaderProxy(t, upstream.URL)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-cost","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "3", w.Header().Get(HeaderResponseCost), "must be the router's own cost, not the provider's")
}

// A plain streaming response cannot carry the header: its headers leave before
// the usage that prices it has arrived.
func TestResponseCostHeaderAbsentOnStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"c","object":"chat.completion.chunk","model":"gpt-cost","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"c","object":"chat.completion.chunk","model":"gpt-cost","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	prx := costHeaderProxy(t, upstream.URL)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-cost","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Empty(t, w.Header().Get(HeaderResponseCost))
}
