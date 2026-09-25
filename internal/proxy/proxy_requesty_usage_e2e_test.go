package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newBillingTestProxy(t *testing.T, credType config.ProviderType, upstreamURL string, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name: "upstream", Type: credType,
			BaseURL: upstreamURL, APIKey: "upstream-key", RPM: 100, TPM: 100000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	prx.responseCompat = compatlitellm.New()
	registry := pricing.NewModelPriceRegistry()
	registry.Update(map[string]*pricing.ModelPrice{
		"claude-test": {
			InputCostPerToken:                   1,
			OutputCostPerToken:                  2,
			CacheCreationInputTokenCost:         3,
			CacheCreationInputTokenCostAbove1hr: 5,
			WebSearchBillingUnit:                "per_query",
			SearchContextCostPerQuery: map[string]float64{
				"search_context_size_low":    7,
				"search_context_size_medium": 7,
				"search_context_size_high":   7,
			},
		},
	})
	prx.priceRegistry = registry
	return prx
}

func serveUpstream(t *testing.T, path, contentType, response string) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, path, r.URL.Path)
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(response))
	}))
}

func postAsMaster(t *testing.T, prx *Proxy, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

func newRequestyTest(t *testing.T, contentType, response string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	upstream := serveUpstream(t, "/v1/chat/completions", contentType, response)
	t.Cleanup(upstream.Close)
	dbStub := &stubLiteLLMManager{}
	return newBillingTestProxy(t, config.ProviderTypeOpenAI, upstream.URL, dbStub), dbStub
}

func TestProxyRequest_RequestyCacheWritesBilledByTTL(t *testing.T) {
	const usage5m = `{"prompt_tokens":130,"completion_tokens":10,"total_tokens":140,` +
		`"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":100,"caching_token_details":{"caching_1h_tokens":0,"caching_5m_tokens":100}}}`
	const usage1h = `{"prompt_tokens":130,"completion_tokens":10,"total_tokens":140,` +
		`"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":100,"caching_token_details":{"caching_1h_tokens":100,"caching_5m_tokens":0}}}`

	nonStream := func(usage string) string {
		return `{"id":"rqsty-cmpl-1","object":"chat.completion","created":1,"model":"claude-test",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":` + usage + `}`
	}
	stream := func(usage string) string {
		return "data: " + `{"id":"rqsty-cmpl-2","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}` +
			"\n\ndata: " + `{"id":"rqsty-cmpl-2","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` +
			"\n\ndata: " + `{"id":"rqsty-cmpl-2","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{}}],"usage":` + usage + `}` +
			"\n\ndata: [DONE]\n\n"
	}

	tests := []struct {
		name          string
		contentType   string
		response      string
		request       string
		wantWriteCost float64
	}{
		{
			name:          "non-stream 5m",
			contentType:   "application/json",
			response:      nonStream(usage5m),
			request:       `{"model":"claude-test","messages":[{"role":"user","content":"hi"}]}`,
			wantWriteCost: 100 * 3,
		},
		{
			name:          "non-stream 1h",
			contentType:   "application/json",
			response:      nonStream(usage1h),
			request:       `{"model":"claude-test","messages":[{"role":"user","content":"hi"}]}`,
			wantWriteCost: 100 * 5,
		},
		{
			name:          "stream 5m",
			contentType:   "text/event-stream",
			response:      stream(usage5m),
			request:       `{"model":"claude-test","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`,
			wantWriteCost: 100 * 3,
		},
		{
			name:          "stream 1h",
			contentType:   "text/event-stream",
			response:      stream(usage1h),
			request:       `{"model":"claude-test","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`,
			wantWriteCost: 100 * 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx, dbStub := newRequestyTest(t, tt.contentType, tt.response)

			w := postAsMaster(t, prx, "/v1/chat/completions", tt.request)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, 130, entry.PromptTokens)
			assert.Equal(t, 10, entry.CompletionTokens)

			metadata := decodeMetadata(t, entry.Metadata)
			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.Equal(t, tt.wantWriteCost, costBreakdown["cache_creation_cost"])
			assert.Equal(t, 30+10*2+tt.wantWriteCost, entry.Spend)
		})
	}
}

func TestProxyRequest_RequestyServerWebSearchToolCallIsStrippedAndBilled(t *testing.T) {
	prx, dbStub := newRequestyTest(t, "application/json",
		`{"id":"rqsty-cmpl-3","object":"chat.completion","created":1,"model":"claude-test","choices":[{"index":0,"finish_reason":"tool_calls","message":{`+
			`"role":"assistant","content":"Reykjavik has 139,804 people.",`+
			`"annotations":[{"type":"url_citation","tool_use_id":"srvtoolu_01A","query":"reykjavik population","url_citation":{"url":"https://example.com","title":"t","start_index":0,"end_index":0}}],`+
			`"tool_calls":[{"id":"srvtoolu_01A","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"reykjavik population\"}"}},`+
			`{"id":"srvtoolu_01B","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"reykjavik 2026\"}"}}]}}],`+
			`"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":0}}}`)

	w := postAsMaster(t, prx, "/v1/chat/completions", `{"model":"claude-test","tools":[{"type":"web_search"}],"messages":[{"role":"user","content":"population of Reykjavik?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var response struct {
		Choices []struct {
			FinishReason string                 `json:"finish_reason"`
			Message      map[string]interface{} `json:"message"`
		} `json:"choices"`
		Usage map[string]interface{} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	require.Len(t, response.Choices, 1)
	assert.Equal(t, "stop", response.Choices[0].FinishReason)
	assert.NotContains(t, response.Choices[0].Message, "tool_calls")
	assert.Equal(t, "Reykjavik has 139,804 people.", response.Choices[0].Message["content"])
	assert.Contains(t, response.Choices[0].Message, "annotations")
	assert.Equal(t, float64(2), response.Usage["server_tool_use"].(map[string]interface{})["web_search_requests"])

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	metadata := decodeMetadata(t, entry.Metadata)
	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(2*7), costBreakdown["web_search_cost"])
	assert.Equal(t, float64(100+10*2+2*7), entry.Spend)
}

func TestProxyRequest_RequestyKeepsClientToolCallsNextToServerSearch(t *testing.T) {
	prx, _ := newRequestyTest(t, "application/json",
		`{"id":"rqsty-cmpl-4","object":"chat.completion","created":1,"model":"claude-test","choices":[{"index":0,"finish_reason":"tool_calls","message":{`+
			`"role":"assistant","content":null,`+
			`"tool_calls":[{"id":"srvtoolu_01A","type":"function","function":{"name":"web_search","arguments":"{}"}},`+
			`{"id":"toolu_01C","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}}]}}],`+
			`"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}`)

	w := postAsMaster(t, prx, "/v1/chat/completions", `{"model":"claude-test","tools":[{"type":"web_search"},{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"weather?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var response struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ToolCalls []struct {
					ID string `json:"id"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	require.Len(t, response.Choices, 1)
	assert.Equal(t, "tool_calls", response.Choices[0].FinishReason)
	require.Len(t, response.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "toolu_01C", response.Choices[0].Message.ToolCalls[0].ID)
}

func TestProxyRequest_RequestyStreamCitationBillsWebSearch(t *testing.T) {
	prx, dbStub := newRequestyTest(t, "text/event-stream",
		"data: "+`{"id":"rqsty-cmpl-5","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`+
			"\n\ndata: "+`{"id":"rqsty-cmpl-5","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{"content":"Krasznahorkai won in 2025.",`+
			`"annotations":[{"type":"url_citation","tool_use_id":"","query":"","url_citation":{"url":"https://example.com","title":"t","start_index":0,"end_index":0}}]}}]}`+
			"\n\ndata: "+`{"id":"rqsty-cmpl-5","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+
			"\n\ndata: "+`{"id":"rqsty-cmpl-5","object":"chat.completion.chunk","created":1,"model":"claude-test","choices":[{"index":0,"delta":{}}],`+
			`"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":0}}}`+
			"\n\ndata: [DONE]\n\n")

	w := postAsMaster(t, prx, "/v1/chat/completions", `{"model":"claude-test","stream":true,"stream_options":{"include_usage":true},"tools":[{"type":"web_search"}],"messages":[{"role":"user","content":"nobel?"}]}`)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 100, entry.PromptTokens)
	assert.Equal(t, 10, entry.CompletionTokens)
	metadata := decodeMetadata(t, entry.Metadata)
	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(7), costBreakdown["web_search_cost"])
	assert.Equal(t, float64(100+10*2+7), entry.Spend)
}
