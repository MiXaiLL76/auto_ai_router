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
		upstreamUsageHeader   string
		upstreamAudioTokens   int
		expectedUsageHeader   string
		expectedLoggedAudioIn int
	}{
		{
			name:                  "raw OpenAI-compatible proxy subtracts cached audio",
			upstreamAudioTokens:   100,
			expectedUsageHeader:   aarUsageAudioTokensIncludeCached,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR-normalized proxy header keeps non-cached audio without credential config",
			upstreamUsageHeader:   aarUsageAudioTokensExcludeCached,
			upstreamAudioTokens:   60,
			expectedUsageHeader:   aarUsageAudioTokensExcludeCached,
			expectedLoggedAudioIn: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newCacheBillingUpstream(t, false, tt.upstreamAudioTokens, tt.upstreamUsageHeader)
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := newCacheBillingProxy(t, upstream.URL, dbStub)

			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
				"model": "gpt-cache",
				"messages": [{"role": "user", "content": "hello"}]
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			assert.Equal(t, tt.expectedUsageHeader, w.Header().Get(HeaderAARUsageAudioTokens))
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
	tests := []struct {
		name                  string
		upstreamUsageHeader   string
		upstreamAudioTokens   int
		expectedUsageHeader   string
		expectedLoggedAudioIn int
	}{
		{
			name:                  "raw OpenAI-compatible proxy subtracts cached audio",
			upstreamAudioTokens:   100,
			expectedUsageHeader:   aarUsageAudioTokensIncludeCached,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR-normalized proxy header keeps non-cached audio without credential config",
			upstreamUsageHeader:   aarUsageAudioTokensExcludeCached,
			upstreamAudioTokens:   60,
			expectedUsageHeader:   aarUsageAudioTokensExcludeCached,
			expectedLoggedAudioIn: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newCacheBillingUpstream(t, true, tt.upstreamAudioTokens, tt.upstreamUsageHeader)
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := newCacheBillingProxy(t, upstream.URL, dbStub)

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
			assert.Equal(t, tt.expectedUsageHeader, w.Header().Get(HeaderAARUsageAudioTokens))
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
			assert.Equal(t, float64(tt.expectedLoggedAudioIn), promptDetails["audio_tokens"])
			assert.Equal(t, float64(80), promptDetails["cached_tokens"])
			assert.Equal(t, float64(40), promptDetails["cached_audio_tokens"])
		})
	}
}

func TestProxyRequest_AIRCredentialCacheBillingUsesUsageContract(t *testing.T) {
	tests := []struct {
		name                  string
		upstreamUsageHeader   string
		upstreamAudioTokens   int
		expectedLoggedAudioIn int
	}{
		{
			name:                  "AIR upstream says audio includes cached audio",
			upstreamUsageHeader:   aarUsageAudioTokensIncludeCached,
			upstreamAudioTokens:   100,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR upstream says audio excludes cached audio",
			upstreamUsageHeader:   aarUsageAudioTokensExcludeCached,
			upstreamAudioTokens:   60,
			expectedLoggedAudioIn: 60,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newCacheBillingUpstream(t, false, tt.upstreamAudioTokens, tt.upstreamUsageHeader)
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := newCacheBillingProxyWithType(t, config.ProviderTypeAIR, upstream.URL, dbStub)

			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
				"model": "gpt-cache",
				"messages": [{"role": "user", "content": "hello"}]
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			assert.Equal(t, tt.upstreamUsageHeader, w.Header().Get(HeaderAARUsageAudioTokens))
			require.Len(t, dbStub.loggedEntries, 1)

			metadata := decodeMetadata(t, dbStub.loggedEntries[0].Metadata)
			usageObject := metadata["usage_object"].(map[string]interface{})
			promptDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
			assert.Equal(t, float64(tt.expectedLoggedAudioIn), promptDetails["audio_tokens"])
			assert.Equal(t, float64(80), promptDetails["cached_tokens"])
			assert.Equal(t, float64(40), promptDetails["cached_audio_tokens"])
		})
	}
}

func TestProxyRequest_ConvertedResponsesEmitsNormalizedUsageHeaderAndLogsSpend(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(cacheBillingResponseJSON(100)))
	}))
	defer upstream.Close()

	passthroughResponses := false
	dbStub := &stubLiteLLMManager{}
	builder := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "openai-cache",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key")
	builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-cache", PassthroughResponses: &passthroughResponses},
	})
	prx := builder.Build()
	prx.LiteLLMDB = dbStub
	installCacheBillingPrices(prx)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model": "gpt-cache",
		"input": "hello"
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, aarUsageAudioTokensExcludeCached, w.Header().Get(HeaderAARUsageAudioTokens))

	var response struct {
		Usage struct {
			InputTokensDetails struct {
				AudioTokens       int `json:"audio_tokens"`
				CachedTokens      int `json:"cached_tokens"`
				CachedAudioTokens int `json:"cached_audio_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, 60, response.Usage.InputTokensDetails.AudioTokens)
	assert.Equal(t, 80, response.Usage.InputTokensDetails.CachedTokens)
	assert.Equal(t, 40, response.Usage.InputTokensDetails.CachedAudioTokens)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 200, entry.PromptTokens)
	assert.Equal(t, 50, entry.CompletionTokens)
	assert.Equal(t, 1090.0, entry.Spend)

	metadata := decodeMetadata(t, entry.Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	inputDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(60), inputDetails["audio_tokens"])
	assert.Equal(t, float64(80), inputDetails["cached_tokens"])
	assert.Equal(t, float64(40), inputDetails["cached_audio_tokens"])
}

func newCacheBillingProxy(t *testing.T, upstreamURL string, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()

	return newCacheBillingProxyWithType(t, config.ProviderTypeProxy, upstreamURL, dbStub)
}

func newCacheBillingProxyWithType(t *testing.T, providerType config.ProviderType, upstreamURL string, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()

	cred := config.CredentialConfig{
		Name:    "proxy-cache",
		Type:    providerType,
		BaseURL: upstreamURL,
		APIKey:  "upstream-key",
		RPM:     100,
		TPM:     10000,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(cred).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	installCacheBillingPrices(prx)

	return prx
}

func installCacheBillingPrices(prx *Proxy) {
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
}

func newCacheBillingUpstream(t *testing.T, stream bool, audioTokens int, usageHeader string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		if usageHeader != "" {
			w.Header().Set(HeaderAARUsageAudioTokens, usageHeader)
		}
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
