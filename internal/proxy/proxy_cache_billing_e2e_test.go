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
			expectedUsageHeader:   airUsageAudioTokensIncludeCached,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR-normalized proxy header keeps non-cached audio without credential config",
			upstreamUsageHeader:   airUsageAudioTokensExcludeCached,
			upstreamAudioTokens:   60,
			expectedUsageHeader:   airUsageAudioTokensExcludeCached,
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
			assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
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

func TestProxyRequest_QwenUsageIsNormalizedForResponseAndSpend(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-qwen","model":"qwen3.7-plus","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":202,"total_tokens":212,"completion_tokens_details":{"text_tokens":0,"reasoning_tokens":194}}}`))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	credential := config.CredentialConfig{
		Name: "qwen", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
		APIKey: "upstream-key", RPM: 100, TPM: 10000,
	}
	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "qwen3.7-plus", Credential: "qwen"},
	})
	builder.config.ModelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx := builder.Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"qwen3.7-plus": {
			InputCostPerToken:           1,
			OutputCostPerToken:          2,
			OutputCostPerReasoningToken: 20,
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model": "qwen3.7-plus",
		"messages": [{"role": "user", "content": "hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var response map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	details := response["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), details["text_tokens"])
	require.Len(t, dbStub.loggedEntries, 1)
	metadata := decodeMetadata(t, dbStub.loggedEntries[0].Metadata)
	loggedDetails := metadata["usage_object"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), loggedDetails["text_tokens"])
	assert.Equal(t, float64(194), loggedDetails["reasoning_tokens"])
	assert.Equal(t, 3906.0, dbStub.loggedEntries[0].Spend)
}

func TestProxyRequest_StreamingQwenUsageIsNormalizedForResponseAndSpend(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-qwen\",\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-qwen\",\"model\":\"qwen3.7-plus\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":202,\"total_tokens\":212,\"completion_tokens_details\":{\"text_tokens\":202,\"reasoning_tokens\":194}}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	credential := config.CredentialConfig{
		Name: "qwen", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
		APIKey: "upstream-key", RPM: 100, TPM: 10000,
	}
	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "qwen3.7-plus", Credential: "qwen"},
	})
	builder.config.ModelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx := builder.Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"qwen3.7-plus": {
			InputCostPerToken:           1,
			OutputCostPerToken:          2,
			OutputCostPerReasoningToken: 20,
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model": "qwen3.7-plus",
		"stream": true,
		"stream_options": {"include_usage": true},
		"messages": [{"role": "user", "content": "hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"text_tokens":8`)
	assert.NotContains(t, w.Body.String(), `"text_tokens":202`)
	require.Len(t, dbStub.loggedEntries, 1)
	metadata := decodeMetadata(t, dbStub.loggedEntries[0].Metadata)
	loggedDetails := metadata["usage_object"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), loggedDetails["text_tokens"])
	assert.Equal(t, float64(194), loggedDetails["reasoning_tokens"])
	assert.Equal(t, 3906.0, dbStub.loggedEntries[0].Spend)
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
			expectedUsageHeader:   airUsageAudioTokensIncludeCached,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR-normalized proxy header keeps non-cached audio without credential config",
			upstreamUsageHeader:   airUsageAudioTokensExcludeCached,
			upstreamAudioTokens:   60,
			expectedUsageHeader:   airUsageAudioTokensExcludeCached,
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
			assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
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
			upstreamUsageHeader:   airUsageAudioTokensIncludeCached,
			upstreamAudioTokens:   100,
			expectedLoggedAudioIn: 60,
		},
		{
			name:                  "AIR upstream says audio excludes cached audio",
			upstreamUsageHeader:   airUsageAudioTokensExcludeCached,
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
			assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
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
		{Name: "gpt-cache", Credential: "openai-cache", PassthroughResponses: &passthroughResponses},
	})
	builder.config.ModelManager.LoadModelsFromConfig(builder.config.Credentials)
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
	assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))

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

func TestProxyRequest_PassthroughResponsesPreservesCacheWriteTokens(t *testing.T) {
	const upstreamBody = `{
		"id": "resp_passthrough",
		"object": "response",
		"created_at": 1,
		"status": "completed",
		"model": "gpt-cache",
		"output": [{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
		"usage": {
			"input_tokens": 200,
			"output_tokens": 50,
			"total_tokens": 250,
			"input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 120},
			"output_tokens_details": {"reasoning_tokens": 0}
		}
	}`

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(upstreamBody))
	}))
	defer upstream.Close()

	passthroughResponses := true
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
		{Name: "gpt-cache", Credential: "openai-cache", PassthroughResponses: &passthroughResponses},
	})
	builder.config.ModelManager.LoadModelsFromConfig(builder.config.Credentials)
	prx := builder.Build()
	prx.LiteLLMDB = dbStub
	installCacheBillingPrices(prx)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"gpt-cache","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	var response struct {
		Usage struct {
			InputTokensDetails struct {
				CachedTokens     int `json:"cached_tokens"`
				CacheWriteTokens int `json:"cache_write_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, 120, response.Usage.InputTokensDetails.CacheWriteTokens,
		"cache_write_tokens must survive the passthrough re-serialization")

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]

	metadata := decodeMetadata(t, entry.Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	inputDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(120), inputDetails["cache_creation_tokens"],
		"cache writes must reach the spend log")

	assert.Equal(t, 540.0, entry.Spend)
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
			w.Header().Set(HeaderAIRUsageAudioTokens, usageHeader)
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
