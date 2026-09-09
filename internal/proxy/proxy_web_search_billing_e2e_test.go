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

func TestProxyRequest_WebSearchEnabledButUnusedDoesNotBillToolCost(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-web-search",
			"object": "chat.completion",
			"created": 1,
			"model": "gpt-web-search",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "ok"},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 4,
				"total_tokens": 14
			}
		}`))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "openai-web-search",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.ReplaceFilePrices(map[string]*pricing.ModelPrice{
		"gpt-web-search": {
			InputCostPerToken:  1,
			OutputCostPerToken: 2,
			SearchContextCostPerQuery: map[string]float64{
				"search_context_size_low":    3,
				"search_context_size_medium": 5,
				"search_context_size_high":   7,
			},
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model": "gpt-web-search",
		"messages": [{"role": "user", "content": "latest news?"}],
		"web_search_options": {"search_context_size": "high"}
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.JSONEq(t, `{"role":"assistant","content":"ok"}`, extractMessage(t, w.Body.String()))
	require.Len(t, dbStub.loggedEntries, 1)

	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 10, entry.PromptTokens)
	assert.Equal(t, 4, entry.CompletionTokens)
	assert.Equal(t, 14, entry.TotalTokens)
	assert.Equal(t, 18.0, entry.Spend) // 10*1 + 4*2; request enablement is not usage

	metadata := decodeMetadata(t, entry.Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
	assert.Equal(t, float64(0), serverToolUse["web_search_requests"])
	assert.Nil(t, serverToolUse["web_search_context_size"])

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(0), costBreakdown["tool_usage_cost"])
	assert.Equal(t, float64(0), costBreakdown["web_search_cost"])
	assert.Equal(t, float64(18), costBreakdown["total_cost"])
}

func TestProxyRequest_ResponsesWebSearchCallsOverrideRequestFallback(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "resp_web_search",
			"object": "response",
			"status": "completed",
			"model": "gpt-web-search",
			"output": [
				{"type": "web_search_call", "id": "ws_1", "status": "completed"},
				{"type": "web_search_call", "id": "ws_2", "status": "completed"},
				{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
			],
			"usage": {
				"input_tokens": 10,
				"output_tokens": 4,
				"total_tokens": 14,
				"input_tokens_details": {"cached_tokens": 0},
				"output_tokens_details": {"reasoning_tokens": 0}
			}
		}`))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "openai-responses-web-search",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.ReplaceFilePrices(map[string]*pricing.ModelPrice{
		"gpt-web-search": {
			InputCostPerToken:  1,
			OutputCostPerToken: 2,
			SearchContextCostPerQuery: map[string]float64{
				"search_context_size_low":    3,
				"search_context_size_medium": 5,
				"search_context_size_high":   7,
			},
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model": "gpt-web-search",
		"input": "latest news?",
		"tools": [{"type": "web_search_preview", "search_context_size": "low"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)

	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 10, entry.PromptTokens)
	assert.Equal(t, 4, entry.CompletionTokens)
	assert.Equal(t, 14, entry.TotalTokens)
	assert.Equal(t, 24.0, entry.Spend) // 10*1 + 4*2 + 2 low-context web searches * 3

	metadata := decodeMetadata(t, entry.Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
	assert.Equal(t, float64(2), serverToolUse["web_search_requests"])
	assert.Equal(t, "low", serverToolUse["web_search_context_size"])

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(6), costBreakdown["tool_usage_cost"])
	assert.Equal(t, float64(6), costBreakdown["web_search_cost"])
	assert.Equal(t, float64(24), costBreakdown["total_cost"])
}

func TestProxyRequest_StreamingWebSearchBillingUsesCompletedResponse(t *testing.T) {
	tests := []struct {
		name              string
		output            string
		billingUnit       string
		wantRequests      int
		wantWebSearchCost float64
	}{
		{
			name:              "tool enabled but unused",
			output:            `[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[]}]`,
			billingUnit:       "per_query",
			wantRequests:      0,
			wantWebSearchCost: 0,
		},
		{
			name:              "one completed call",
			output:            `[{"type":"web_search_call","id":"ws_1","status":"completed"}]`,
			billingUnit:       "per_query",
			wantRequests:      1,
			wantWebSearchCost: 3,
		},
		{
			name:              "two completed calls per query",
			output:            `[{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"web_search_call","id":"ws_2","status":"completed"}]`,
			billingUnit:       "per_query",
			wantRequests:      2,
			wantWebSearchCost: 6,
		},
		{
			name:              "two completed calls per prompt",
			output:            `[{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"web_search_call","id":"ws_2","status":"completed"}]`,
			billingUnit:       "per_prompt",
			wantRequests:      2,
			wantWebSearchCost: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("event: response.completed\ndata: " +
					`{"type":"response.completed","response":{"id":"resp_stream","object":"response","status":"completed","model":"gpt-web-search","output":` +
					tt.output +
					`,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}}}` +
					"\n\ndata: [DONE]\n\n"))
			}))
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "openai-web-search",
					Type:    config.ProviderTypeOpenAI,
					BaseURL: upstream.URL,
					APIKey:  "upstream-key",
					RPM:     100,
					TPM:     10000,
				}).
				WithMasterKey("master-key").
				Build()
			prx.LiteLLMDB = dbStub
			registry := pricing.NewModelPriceRegistry()
			registry.ReplaceFilePrices(map[string]*pricing.ModelPrice{
				"gpt-web-search": {
					InputCostPerToken:    1,
					OutputCostPerToken:   2,
					WebSearchBillingUnit: tt.billingUnit,
					SearchContextCostPerQuery: map[string]float64{
						"search_context_size_low": 3,
					},
				},
			})
			prx.priceRegistry = registry

			req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
				"model": "gpt-web-search",
				"input": "latest news?",
				"stream": true,
				"tools": [{"type": "web_search_preview", "search_context_size": "low"}]
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			metadata := decodeMetadata(t, entry.Metadata)
			usageObject := metadata["usage_object"].(map[string]interface{})
			serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
			assert.Equal(t, float64(tt.wantRequests), serverToolUse["web_search_requests"])
			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.Equal(t, tt.wantWebSearchCost, costBreakdown["web_search_cost"])
		})
	}
}

func TestProxyRequest_InterruptedStreamingSearchIsNotBilledWithoutConfirmation(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"event: response.web_search_call.in_progress\n" +
				`data: {"type":"response.web_search_call.in_progress","item_id":"ws_unconfirmed","output_index":0}` +
				"\n\n",
		))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "openai-web-search",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.ReplaceFilePrices(map[string]*pricing.ModelPrice{
		"gpt-web-search": {
			SearchContextCostPerQuery: map[string]float64{"search_context_size_medium": 5},
		},
	})
	prx.priceRegistry = registry

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model": "gpt-web-search",
		"input": "latest news?",
		"stream": true,
		"tools": [{"type": "web_search_preview"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	metadata := decodeMetadata(t, dbStub.loggedEntries[0].Metadata)
	usageObject := metadata["usage_object"].(map[string]interface{})
	serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
	assert.Equal(t, float64(0), serverToolUse["web_search_requests"])
	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(0), costBreakdown["web_search_cost"])
}
