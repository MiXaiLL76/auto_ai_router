package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Z.AI reports a built-in web search only through a top-level "web_search"
// results array, and returns it only when web_search.search_result is true.
// The router always asks for it so the search can be billed, and hides it
// again from clients that did not ask.
func TestProxyRequest_ZAIWebSearchResultsBilling(t *testing.T) {
	const results = `[{"title":"a","link":"https://a.example","refer":"ref_1"},{"title":"b","link":"https://b.example","refer":"ref_2"}]`
	const jsonResponse = `{"id":"zai-1","object":"chat.completion","created":1,"model":"glm-5.3-flashx",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14},"web_search":` + results + `}`
	const providerFieldResponse = `{"id":"zai-4","object":"chat.completion","created":1,"model":"glm-5.3-flashx","provider":"Z.AI",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14},"web_search":` + results + `}`
	const noResultsResponse = `{"id":"zai-3","object":"chat.completion","created":1,"model":"glm-5.3-flashx",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`
	const streamResponse = "data: " +
		`{"id":"zai-2","object":"chat.completion.chunk","created":1,"model":"glm-5.3-flashx","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}` +
		"\n\ndata: " +
		`{"id":"zai-2","object":"chat.completion.chunk","created":1,"model":"glm-5.3-flashx","choices":[{"index":0,"delta":{"content":""}}],"web_search":` + results + `}` +
		"\n\ndata: " +
		`{"id":"zai-2","object":"chat.completion.chunk","created":1,"model":"glm-5.3-flashx","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}` +
		"\n\ndata: [DONE]\n\n"
	const cutStreamResponse = "data: " +
		`{"id":"zai-5","object":"chat.completion.chunk","created":1,"model":"glm-5.3-flashx","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}` +
		"\n\ndata: " +
		`{"id":"zai-5","object":"chat.completion.chunk","created":1,"model":"glm-5.3-flashx","choices":[{"index":0,"delta":{"content":""}}],"web_search":` + results + `}` +
		"\n\n"

	const hidden = `{"enable":true,"search_engine":"search-prime"}`
	const requested = `{"enable":true,"search_engine":"search-prime","search_result":true}`

	tests := []struct {
		name                string
		credType            config.ProviderType
		searchOptions       string
		stream              bool
		response            string
		wantResultsInClient bool
		wantRequests        int
		wantEstimatedTokens bool
	}{
		{name: "direct, results not requested", credType: config.ProviderTypeOpenAI, searchOptions: hidden, response: jsonResponse, wantRequests: 1},
		{name: "direct stream, results not requested", credType: config.ProviderTypeOpenAI, searchOptions: hidden, stream: true, response: streamResponse, wantRequests: 1},
		{name: "direct, results requested", credType: config.ProviderTypeOpenAI, searchOptions: requested, response: jsonResponse, wantResultsInClient: true, wantRequests: 1},
		{name: "direct stream, results requested", credType: config.ProviderTypeOpenAI, searchOptions: requested, stream: true, response: streamResponse, wantResultsInClient: true, wantRequests: 1},
		{name: "remote router, results not requested", credType: config.ProviderTypeProxy, searchOptions: hidden, response: jsonResponse, wantRequests: 1},
		{name: "remote router stream, results not requested", credType: config.ProviderTypeProxy, searchOptions: hidden, stream: true, response: streamResponse, wantRequests: 1},
		{name: "direct, sanitized response envelope", credType: config.ProviderTypeOpenAI, searchOptions: hidden, response: providerFieldResponse, wantRequests: 1},
		{name: "remote router stream cut before usage", credType: config.ProviderTypeProxy, searchOptions: hidden, stream: true, response: cutStreamResponse, wantRequests: 1, wantEstimatedTokens: true},
		{name: "provider skipped the search", credType: config.ProviderTypeOpenAI, searchOptions: hidden, response: noResultsResponse, wantRequests: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/v1/chat/completions", r.URL.Path)
				raw, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				var forwarded struct {
					Tools []struct {
						Type      string         `json:"type"`
						WebSearch map[string]any `json:"web_search"`
					} `json:"tools"`
				}
				require.NoError(t, json.Unmarshal(raw, &forwarded), string(raw))
				require.Len(t, forwarded.Tools, 1)
				assert.Equal(t, true, forwarded.Tools[0].WebSearch["search_result"], "upstream must always be asked for the results")
				assert.Equal(t, "search-prime", forwarded.Tools[0].WebSearch["search_engine"])

				if tt.stream {
					w.Header().Set("Content-Type", "text/event-stream")
				} else {
					w.Header().Set("Content-Type", "application/json")
				}
				_, _ = w.Write([]byte(tt.response))
			}))
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "zai",
					Type:    tt.credType,
					BaseURL: upstream.URL,
					APIKey:  "upstream-key",
					RPM:     100,
					TPM:     10000,
				}).
				WithMasterKey("master-key").
				Build()
			prx.LiteLLMDB = dbStub
			registry := pricing.NewModelPriceRegistry()
			registry.Update(map[string]*pricing.ModelPrice{
				"glm-5.3-flashx": {
					InputCostPerToken:    1,
					OutputCostPerToken:   2,
					WebSearchBillingUnit: "per_query",
					SearchContextCostPerQuery: map[string]float64{
						"search_context_size_low":    5,
						"search_context_size_medium": 5,
						"search_context_size_high":   5,
					},
				},
			})
			prx.priceRegistry = registry

			request := `{"model":"glm-5.3-flashx","messages":[{"role":"user","content":"latest news?"}],` +
				`"tools":[{"type":"web_search","web_search":` + tt.searchOptions + `}]`
			if tt.stream {
				request += `,"stream":true,"stream_options":{"include_usage":true}`
			}
			request += `}`
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(request))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			body := w.Body.String()
			assert.Contains(t, body, `"content":"`)
			if tt.wantResultsInClient {
				assert.Contains(t, compactJSONFragments(t, body), `"refer":"ref_2"`)
			} else {
				assert.NotContains(t, body, `"web_search"`)
			}

			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			if tt.wantEstimatedTokens {
				assert.Positive(t, entry.PromptTokens)
				assert.Positive(t, entry.CompletionTokens)
			} else {
				assert.Equal(t, 10, entry.PromptTokens)
				assert.Equal(t, 4, entry.CompletionTokens)
			}

			metadata := decodeMetadata(t, entry.Metadata)
			usageObject := metadata["usage_object"].(map[string]interface{})
			serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
			assert.Equal(t, float64(tt.wantRequests), serverToolUse["web_search_requests"])

			wantSearchCost := float64(tt.wantRequests) * 5
			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.Equal(t, wantSearchCost, costBreakdown["web_search_cost"])
			assert.Equal(t, float64(entry.PromptTokens+entry.CompletionTokens*2)+wantSearchCost, entry.Spend)
		})
	}
}

func TestStripWebSearchResults(t *testing.T) {
	stripped, ok := stripWebSearchResults([]byte(`{"id":"1","choices":[{"message":{"content":"\"web_search\""}}],"web_search":[{"refer":"ref_1"}]}`))
	require.True(t, ok)
	assert.JSONEq(t, `{"id":"1","choices":[{"message":{"content":"\"web_search\""}}]}`, string(stripped))

	for _, body := range []string{
		`{"id":"1","choices":[{"message":{"content":"\"web_search\""}}]}`,
		`{"id":"1"}`,
		`not json "web_search"`,
	} {
		got, ok := stripWebSearchResults([]byte(body))
		assert.False(t, ok, body)
		assert.Equal(t, body, string(got))
	}
}

func TestWebSearchResultsStripReader(t *testing.T) {
	longContent := strings.Repeat("x", 70*1024)
	stream := "event: message\r\n" +
		`data: {"id":"1","choices":[{"delta":{"content":"` + longContent + `"}}]}` + "\r\n\r\n" +
		`data:{"id":"1","choices":[{"delta":{"content":""}}],"web_search":[{"refer":"ref_1"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	var removed []string

	out, err := io.ReadAll(newWebSearchResultsStripReader(strings.NewReader(stream), func(payload []byte) {
		removed = append(removed, string(payload))
	}))

	require.NoError(t, err)
	want := "event: message\r\n" +
		`data: {"id":"1","choices":[{"delta":{"content":"` + longContent + `"}}]}` + "\r\n\r\n" +
		`data:{"choices":[{"delta":{"content":""}}],"id":"1"}` + "\n\n" +
		"data: [DONE]\n\n"
	assert.Equal(t, want, string(out))
	assert.Equal(t, []string{`{"id":"1","choices":[{"delta":{"content":""}}],"web_search":[{"refer":"ref_1"}]}`}, removed)
}

func TestWebSearchResultsHiddenFromClient(t *testing.T) {
	body := []byte(`{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":true}}]}`)
	assert.True(t, webSearchResultsHiddenFromClient("/v1/chat/completions", body))
	assert.True(t, webSearchResultsHiddenFromClient("/chat/completions", body))
	assert.False(t, webSearchResultsHiddenFromClient("/v1/responses", body))
	assert.False(t, webSearchResultsHiddenFromClient("/v1/chat/completions",
		[]byte(`{"model":"glm","tools":[{"type":"web_search","web_search":{"enable":true,"search_result":true}}]}`)))
}
