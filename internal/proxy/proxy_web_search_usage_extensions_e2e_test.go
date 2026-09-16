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

// Some OpenAI-compatible providers report built-in web search executions in
// usage extensions (x_tools / x_details on the Responses API, plugins.search
// on chat-style usage) instead of server_tool_use. Each response below
// carries more than one representation of the same searches; billing must
// pick one of them, never their sum.
func TestProxyRequest_WebSearchUsageExtensionsBilling(t *testing.T) {
	const completedOutput = `[` +
		`{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"q"}},` +
		`{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}` +
		`]`

	tests := []struct {
		name          string
		path          string
		request       string
		contentType   string
		response      string
		wantRequests  int
		wantInBody    []string
		wantPrompt    int
		wantCompleted int
	}{
		{
			name:        "responses x_tools and x_details with one output item",
			path:        "/v1/responses",
			request:     `{"model":"search-model","input":"latest news?","tools":[{"type":"web_search"}]}`,
			contentType: "application/json",
			response: `{"id":"resp_1","object":"response","status":"completed","model":"search-model","output":` + completedOutput +
				`,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":1},` +
				`"x_details":[{"input_tokens":10,"output_tokens":4,"total_tokens":14,"plugins":{"web_search":{"count":1}},"x_billing_type":"response_api"}],` +
				`"x_tools":{"web_search":{"count":1}}}}`,
			wantRequests: 1,
			// The typed re-encode of a passthrough response must keep the
			// extensions, otherwise a downstream router hop cannot bill them.
			wantInBody:    []string{`"x_tools":{"web_search":{"count":1}}`, `"x_billing_type":"response_api"`},
			wantPrompt:    10,
			wantCompleted: 4,
		},
		{
			name:        "responses x_tools count wins over output items",
			path:        "/v1/responses",
			request:     `{"model":"search-model","input":"latest news?","tools":[{"type":"web_search"}]}`,
			contentType: "application/json",
			response: `{"id":"resp_2","object":"response","status":"completed","model":"search-model","output":` + completedOutput +
				`,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"x_tools":{"web_search":{"count":2}}}}`,
			wantRequests:  2,
			wantInBody:    []string{`"x_tools":{"web_search":{"count":2}}`},
			wantPrompt:    10,
			wantCompleted: 4,
		},
		{
			name:        "responses x_details only",
			path:        "/v1/responses",
			request:     `{"model":"search-model","input":"latest news?","tools":[{"type":"web_search"}]}`,
			contentType: "application/json",
			response: `{"id":"resp_3","object":"response","status":"completed","model":"search-model","output":` + completedOutput +
				`,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,"x_details":[{"plugins":{"web_search":{"count":3}},"x_billing_type":"response_api"}]}}`,
			wantRequests:  3,
			wantPrompt:    10,
			wantCompleted: 4,
		},
		{
			name:        "responses stream completed event",
			path:        "/v1/responses",
			request:     `{"model":"search-model","input":"latest news?","stream":true,"tools":[{"type":"web_search"}]}`,
			contentType: "text/event-stream",
			response: "event: response.output_item.done\ndata: " +
				`{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"q"}}}` +
				"\n\nevent: response.completed\ndata: " +
				`{"type":"response.completed","response":{"id":"resp_4","object":"response","status":"completed","model":"search-model","output":` + completedOutput +
				`,"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14,` +
				`"x_details":[{"plugins":{"web_search":{"count":1}},"x_billing_type":"response_api"}],"x_tools":{"web_search":{"count":1}}}}}` +
				"\n\ndata: [DONE]\n\n",
			wantRequests:  1,
			wantInBody:    []string{`"x_tools":{"web_search":{"count":1}}`},
			wantPrompt:    10,
			wantCompleted: 4,
		},
		{
			name:        "chat plugins.search",
			path:        "/v1/chat/completions",
			request:     `{"model":"search-model","messages":[{"role":"user","content":"latest news?"}],"enable_search":true}`,
			contentType: "application/json",
			response: `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"search-model",` +
				`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
				`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"plugins":{"search":{"count":1,"strategy":"agent"}}}}`,
			wantRequests:  1,
			wantInBody:    []string{`"plugins":{"search":{"count":1,"strategy":"agent"}}`},
			wantPrompt:    10,
			wantCompleted: 4,
		},
		{
			name:        "chat stream plugins.search",
			path:        "/v1/chat/completions",
			request:     `{"model":"search-model","messages":[{"role":"user","content":"latest news?"}],"enable_search":true,"stream":true,"stream_options":{"include_usage":true}}`,
			contentType: "text/event-stream",
			response: "data: " +
				`{"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"search-model","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}` +
				"\n\ndata: " +
				`{"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"search-model","choices":[],` +
				`"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"plugins":{"search":{"count":1,"strategy":"agent"}}}}` +
				"\n\ndata: [DONE]\n\n",
			wantRequests:  1,
			wantInBody:    []string{`"plugins":{"search":{"count":1,"strategy":"agent"}}`},
			wantPrompt:    10,
			wantCompleted: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, tt.path, r.URL.Path)
				w.Header().Set("Content-Type", tt.contentType)
				_, _ = w.Write([]byte(tt.response))
			}))
			defer upstream.Close()

			dbStub := &stubLiteLLMManager{}
			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "openai-compatible-search",
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
			registry.Update(map[string]*pricing.ModelPrice{
				"search-model": {
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

			req := httptest.NewRequest("POST", tt.path, strings.NewReader(tt.request))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			body := w.Body.String()
			for _, fragment := range tt.wantInBody {
				assert.Contains(t, compactJSONFragments(t, body), fragment)
			}

			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, tt.wantPrompt, entry.PromptTokens)
			assert.Equal(t, tt.wantCompleted, entry.CompletionTokens)

			metadata := decodeMetadata(t, entry.Metadata)
			usageObject := metadata["usage_object"].(map[string]interface{})
			serverToolUse := usageObject["server_tool_use"].(map[string]interface{})
			assert.Equal(t, float64(tt.wantRequests), serverToolUse["web_search_requests"])

			wantSearchCost := float64(tt.wantRequests) * 5
			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.Equal(t, wantSearchCost, costBreakdown["web_search_cost"])
			assert.Equal(t, float64(tt.wantPrompt)+float64(tt.wantCompleted)*2+wantSearchCost, entry.Spend)
		})
	}
}

// compactJSONFragments re-encodes every JSON document found in body (a plain
// JSON response or the data lines of an SSE stream) without insignificant
// whitespace and with sorted keys, so assertions can match fragments
// regardless of how the proxy re-encoded them.
func compactJSONFragments(t *testing.T, body string) string {
	t.Helper()
	var docs []string
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "data:"))
		if line == "" || line[0] != '{' {
			continue
		}
		var decoded any
		if err := json.Unmarshal([]byte(line), &decoded); err != nil {
			continue
		}
		encoded, err := json.Marshal(decoded)
		require.NoError(t, err)
		docs = append(docs, string(encoded))
	}
	return strings.Join(docs, "\n")
}
