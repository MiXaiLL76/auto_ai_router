package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_CacheBillingLogsLiteLLMSpend(t *testing.T) {
	tests := []struct {
		name                  string
		proxyUsageFormat      config.ProxyUsageFormat
		upstreamAudioTokens   int
		expectedLoggedAudioIn int
	}{
		{
			name:                  "raw OpenAI-compatible proxy subtracts cached audio",
			upstreamAudioTokens:   100,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "normalized proxy keeps non-cached audio",
			proxyUsageFormat:      config.ProxyUsageFormatNormalized,
			upstreamAudioTokens:   60,
			expectedLoggedAudioIn: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newCacheBillingUpstream(t, false, tt.upstreamAudioTokens)
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := newCacheBillingProxy(t, upstream.URL, tt.proxyUsageFormat, dbStub)

			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
				"model": "gpt-cache",
				"messages": [{"role": "user", "content": "hello"}]
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			require.JSONEq(t, `{"role":"assistant","content":"ok"}`, extractMessage(t, w.Body.String()))
			require.Len(t, dbStub.loggedEntries, 1)

			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 200, entry.PromptTokens)
			assert.Equal(t, 50, entry.CompletionTokens)
			assert.Equal(t, 250, entry.TotalTokens)
			assert.Equal(t, 1090.0, entry.Spend)

			metadata := decodeMetadata(t, entry.Metadata)
			usageObject := metadata["usage_object"].(map[string]interface{})
			promptDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
			assert.Equal(t, float64(tt.expectedLoggedAudioIn), promptDetails["audio_tokens"])
			assert.Equal(t, float64(80), promptDetails["cached_tokens"])
			assert.Equal(t, float64(40), promptDetails["cached_audio_tokens"])
			assert.Equal(t, float64(30), promptDetails["cache_creation_tokens"])

			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.Equal(t, float64(30), costBreakdown["cached_input_cost"])
			assert.Equal(t, float64(150), costBreakdown["cache_creation_cost"])
			assert.Equal(t, float64(1090), costBreakdown["total_cost"])
		})
	}
}

func TestProxyRequest_StreamingCacheBillingLogsLiteLLMSpend(t *testing.T) {
	upstream := newCacheBillingUpstream(t, true, 100)
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := newCacheBillingProxy(t, upstream.URL, "", dbStub)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model": "gpt-cache",
		"stream": true,
		"messages": [{"role": "user", "content": "hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "data: [DONE]")
	require.Len(t, dbStub.loggedEntries, 1)

	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 200, entry.PromptTokens)
	assert.Equal(t, 50, entry.CompletionTokens)
	assert.Equal(t, 250, entry.TotalTokens)
	assert.Equal(t, 1090.0, entry.Spend)

	metadata := decodeMetadata(t, entry.Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	promptDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(60), promptDetails["audio_tokens"])
	assert.Equal(t, float64(80), promptDetails["cached_tokens"])
	assert.Equal(t, float64(40), promptDetails["cached_audio_tokens"])
}

func newCacheBillingProxy(t *testing.T, upstreamURL string, usageFormat config.ProxyUsageFormat, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()

	cred := config.CredentialConfig{
		Name:             "proxy-cache",
		Type:             config.ProviderTypeProxy,
		BaseURL:          upstreamURL,
		APIKey:           "upstream-key",
		RPM:              100,
		TPM:              10000,
		ProxyUsageFormat: usageFormat,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(cred).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub

	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"gpt-cache": {
			InputCostPerToken:                   1,
			OutputCostPerToken:                  2,
			InputCostPerAudioToken:              10,
			CacheReadInputTokenCost:             0.5,
			CacheReadInputAudioTokenCost:        0.25,
			CacheCreationInputTokenCost:         3,
			CacheCreationInputTokenCostAbove1hr: 6,
			OutputCostPerReasoningToken:         20,
		},
	})
	prx.priceRegistry = registry

	return prx
}

func newCacheBillingUpstream(t *testing.T, stream bool, audioTokens int) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"id":"chatcmpl-cache","object":"chat.completion.chunk","created":1,"model":"gpt-cache","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + cacheBillingResponseJSON(audioTokens) + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}

		_, _ = w.Write([]byte(cacheBillingResponseJSON(audioTokens)))
	}))
}

func cacheBillingResponseJSON(audioTokens int) string {
	return `{"id":"chatcmpl-cache","object":"chat.completion","created":1,"model":"gpt-cache","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":50,"total_tokens":250,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":` + jsonInt(audioTokens) + `,"cache_creation_tokens":30,"cache_creation_token_details":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":20}},"completion_tokens_details":{"reasoning_tokens":10}}}`
}

func jsonInt(value int) string {
	b, _ := json.Marshal(value)
	return string(b)
}

func decodeMetadata(t *testing.T, raw string) map[string]interface{} {
	t.Helper()

	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(raw), &metadata))
	return metadata
}

func extractMessage(t *testing.T, raw string) string {
	t.Helper()

	var resp struct {
		Choices []struct {
			Message json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal([]byte(raw), &resp))
	require.Len(t, resp.Choices, 1)
	return string(resp.Choices[0].Message)
}
