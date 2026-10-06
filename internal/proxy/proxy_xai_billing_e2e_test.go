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

// xaiTestPrices is a grok-4.7 row using every xAI billing field and the
// image model its image_generation tool uses.
const xaiTestPrices = `{
  "grok-4.7": {
    "input_cost_per_token": 0.0000026,
    "output_cost_per_token": 0.0000078,
    "cache_read_input_token_cost": 0.00000065,
    "input_cost_per_token_above_200k_tokens": 0.0000052,
    "output_cost_per_token_above_200k_tokens": 0.0000156,
    "cache_read_input_token_cost_above_200k_tokens": 0.0000013,
    "search_context_cost_per_query": {"search_context_size_low": 0.0065, "search_context_size_medium": 0.0065, "search_context_size_high": 0.0065},
    "web_search_billing_unit": "per_query",
    "long_context_pricing_mode": "full_request_200k_inclusive",
    "reasoning_tokens_accounting": "auto",
    "reasoning_tokens_additive": true,
    "tool_cost_per_call": {"code_execution": 0.0065, "attachment_search": 0.013, "collections_search": 0.00325},
    "x_search_cost_per_post": 0.0065,
    "x_search_cost_per_profile": 0.013,
    "image_generation_tool_model": "grok-imagine-image-2.0"
  },
  "grok-imagine-image-2.0": {
    "output_cost_per_image": 0.104,
    "input_cost_per_image": 0.013,
    "image_request_defaults": {"resolution": "1k", "quality": "auto"},
    "output_cost_per_image_tiers": [
      {"operation": "edit", "when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.078},
      {"when": {"resolution": "1k", "quality": "auto"}, "output_cost_per_image": 0.052}
    ]
  }
}`

// newXAIBillingTestProxy routes every model to a single OpenAI-compatible
// credential pointed at upstream (the xAI or the Requesty route), billing
// against xaiTestPrices.
func newXAIBillingTestProxy(t *testing.T, upstream *httptest.Server, credentialName string) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	var prices map[string]*pricing.ModelPrice
	require.NoError(t, json.Unmarshal([]byte(xaiTestPrices), &prices))

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    credentialName,
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     100_000_000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	registry := pricing.NewModelPriceRegistry()
	registry.Update(prices)
	prx.priceRegistry = registry
	return prx, dbStub
}

func serveXAIBillingRequest(t *testing.T, prx *Proxy, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	return w
}

// The same request served by xAI (reasoning on top of completion_tokens) and
// by Requesty (reasoning inside completion_tokens) must cost the client the
// same, below and from the 200k long-context threshold on.
func TestProxyRequest_XAIChatBillingMatchesAcrossRoutes(t *testing.T) {
	tests := []struct {
		name               string
		credential         string
		usage              string
		wantSpend          float64
		wantAccounting     string
		wantProviderCost   float64
		wantPromptTokens   int
		wantCompletionToks int
	}{
		{
			name:       "xai direct",
			credential: "xai-direct",
			usage: `{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1350,
				"prompt_tokens_details":{"text_tokens":1000,"cached_tokens":200},
				"completion_tokens_details":{"reasoning_tokens":300},"num_sources_used":0,"cost_in_usd_ticks":37756000}`,
			wantSpend:          800*0.0000026 + 200*0.00000065 + 350*0.0000078,
			wantAccounting:     "additive",
			wantProviderCost:   0.0037756,
			wantPromptTokens:   1000,
			wantCompletionToks: 50,
		},
		{
			name:       "requesty fallback",
			credential: "requesty-fallback",
			usage: `{"prompt_tokens":1000,"completion_tokens":350,"total_tokens":1350,
				"prompt_tokens_details":{"cached_tokens":200},
				"completion_tokens_details":{"reasoning_tokens":300},"cost":0.0042}`,
			wantSpend:          800*0.0000026 + 200*0.00000065 + 350*0.0000078,
			wantAccounting:     "included",
			wantProviderCost:   0.0042,
			wantPromptTokens:   1000,
			wantCompletionToks: 350,
		},
		{
			name:       "xai at exactly 200k prompt tokens",
			credential: "xai-direct",
			usage: `{"prompt_tokens":200000,"completion_tokens":10,"total_tokens":200110,
				"prompt_tokens_details":{"cached_tokens":100000},"completion_tokens_details":{"reasoning_tokens":100}}`,
			wantSpend:          100000*0.0000052 + 100000*0.0000013 + 110*0.0000156,
			wantAccounting:     "additive",
			wantPromptTokens:   200000,
			wantCompletionToks: 10,
		},
		{
			name:       "requesty at 199999 prompt tokens",
			credential: "requesty-fallback",
			usage: `{"prompt_tokens":199999,"completion_tokens":110,"total_tokens":200109,
				"prompt_tokens_details":{"cached_tokens":100000},"completion_tokens_details":{"reasoning_tokens":100}}`,
			wantSpend:          99999*0.0000026 + 100000*0.00000065 + 110*0.0000078,
			wantAccounting:     "included",
			wantPromptTokens:   199999,
			wantCompletionToks: 110,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/v1/chat/completions", r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","model":"grok-4.7","choices":[{"index":0,` +
					`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":` + tt.usage + `}`))
			}))
			defer upstream.Close()
			prx, dbStub := newXAIBillingTestProxy(t, upstream, tt.credential)

			serveXAIBillingRequest(t, prx, "/v1/chat/completions", `{"model":"grok-4.7","messages":[{"role":"user","content":"hi"}]}`)

			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			assert.Equal(t, tt.wantPromptTokens, entry.PromptTokens)
			assert.Equal(t, tt.wantCompletionToks, entry.CompletionTokens)
			assert.InDelta(t, tt.wantSpend, entry.Spend, 1e-12)
			metadata := decodeMetadata(t, entry.Metadata)
			assert.Equal(t, tt.wantAccounting, metadata["reasoning_tokens_accounting"])
			if tt.wantProviderCost > 0 {
				assert.InDelta(t, tt.wantProviderCost, metadata["provider_reported_cost"], 1e-12,
					"logged for reconciliation...")
			} else {
				assert.NotContains(t, metadata, "provider_reported_cost")
			}
			costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
			assert.InDelta(t, tt.wantSpend, costBreakdown["total_cost"], 1e-12, "...but never added to the price")
		})
	}
}

const xaiResponsesWithToolsBody = `{"id":"resp_tools","object":"response","status":"completed","model":"grok-4.7",
	"output":[
		{"type":"web_search_call","id":"ws_1","status":"completed"},
		{"type":"web_search_call","id":"ws_2","status":"completed"},
		{"type":"web_search_call","id":"ws_3","status":"failed"},
		{"type":"x_search_call","id":"xs_1","status":"completed"},
		{"type":"code_interpreter_call","id":"ci_1","status":"completed"},
		{"type":"file_search_call","id":"fs_1","status":"completed"},
		{"type":"image_generation_call","id":"ig_1","status":"completed","prompt":"a corgi","result":"/9j/4AAQ"},
		{"type":"image_generation_call","id":"ie_2","status":"completed","prompt":"make it blue","result":"/9j/4AAQ"},
		{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done",
			"annotations":[{"type":"url_citation","url":"https://x.com/a/status/1"}]}]}],
	"usage":{"input_tokens":5000,"input_tokens_details":{"cached_tokens":1000},"output_tokens":200,
		"output_tokens_details":{"reasoning_tokens":800},"total_tokens":6000,"num_sources_used":0,"num_server_side_tools_used":9,
		"cost_in_usd_ticks":1234567890,"cost_in_nano_usd":123456789,
		"server_side_tool_usage_details":{"web_search_calls":2,"x_search_calls":1,"x_posts_fetched":44,"x_users_fetched":3,
			"code_interpreter_calls":1,"file_search_calls":3,"mcp_calls":1,"document_search_calls":2,"image_generation_calls":2}}}`

// xaiResponsesToolCost is the tool bill for xaiResponsesWithToolsBody.
const xaiResponsesToolCost = 2*0.0065 + // web search, successful calls only
	44*0.0065 + 3*0.013 + // X Search: fetched posts and profiles
	1*0.0065 + // code execution
	2*0.013 + // attachment search (document_search_calls)
	3*0.00325 + // collections search (file_search_calls)
	0.052 + // generated image, grok-imagine-image-2.0 default tier
	0.078 + 0.013 // edited image: edit tier plus its source image

const xaiResponsesTokenCost = 4000*0.0000026 + 1000*0.00000065 + (200+800)*0.0000078

func TestProxyRequest_XAIResponsesServerSideToolBilling(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path, "xAI server-side tools only exist on the Responses API")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(xaiResponsesWithToolsBody))
	}))
	defer upstream.Close()
	prx, dbStub := newXAIBillingTestProxy(t, upstream, "xai-direct")

	w := serveXAIBillingRequest(t, prx, "/v1/responses", `{"model":"grok-4.7","input":"research this",
		"tools":[{"type":"web_search"},{"type":"x_search"},{"type":"code_interpreter"},{"type":"file_search"},{"type":"image_generation"}]}`)

	body := compactJSONFragments(t, w.Body.String())
	assert.Contains(t, body, `"server_side_tool_usage_details":{`, "the client keeps xAI's own tool usage")
	assert.NotContains(t, body, "cost_in_usd_ticks", "the purchase cost never reaches the client")
	assert.NotContains(t, body, "cost_in_nano_usd")

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.InDelta(t, xaiResponsesTokenCost+xaiResponsesToolCost, entry.Spend, 1e-12)

	metadata := decodeMetadata(t, entry.Metadata)
	serverToolUse := metadata["usage_object"].(map[string]interface{})["server_tool_use"].(map[string]interface{})
	assert.Equal(t, float64(2), serverToolUse["web_search_requests"])
	assert.Equal(t, float64(1), serverToolUse["x_search_calls"])
	assert.Equal(t, float64(44), serverToolUse["x_posts_fetched"])
	assert.Equal(t, float64(3), serverToolUse["x_users_fetched"])
	assert.Equal(t, float64(1), serverToolUse["code_execution_calls"])
	assert.Equal(t, float64(2), serverToolUse["attachment_search_calls"])
	assert.Equal(t, float64(3), serverToolUse["collections_search_calls"])
	assert.Equal(t, float64(1), serverToolUse["mcp_calls"])
	assert.Equal(t, float64(1), serverToolUse["image_generation_calls"])
	assert.Equal(t, float64(1), serverToolUse["image_edit_calls"])

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.InDelta(t, 2*0.0065, costBreakdown["web_search_cost"], 1e-12)
	assert.InDelta(t, 44*0.0065+3*0.013, costBreakdown["x_search_cost"], 1e-12)
	assert.InDelta(t, 0.0065, costBreakdown["code_execution_cost"], 1e-12)
	assert.InDelta(t, 2*0.013, costBreakdown["attachment_search_cost"], 1e-12)
	assert.InDelta(t, 3*0.00325, costBreakdown["collections_search_cost"], 1e-12)
	assert.InDelta(t, 0.052+0.078+0.013, costBreakdown["image_generation_tool_cost"], 1e-12)
	assert.InDelta(t, xaiResponsesToolCost, costBreakdown["tool_usage_cost"], 1e-12)
	assert.InDelta(t, xaiResponsesTokenCost+xaiResponsesToolCost, costBreakdown["total_cost"], 1e-12)
	assert.InDelta(t, 0.123456789, metadata["provider_reported_cost"], 1e-12)
}

func TestProxyRequest_XAIResponsesStreamToolBillingCountedOnce(t *testing.T) {
	for _, terminal := range []string{"response.completed", "response.incomplete"} {
		t.Run(terminal, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				items := []string{
					`{"type":"web_search_call","id":"ws_1","status":"completed"}`,
					`{"type":"web_search_call","id":"ws_2","status":"completed"}`,
					`{"type":"x_search_call","id":"xs_1","status":"completed"}`,
				}
				var stream strings.Builder
				for i, item := range items {
					stream.WriteString("event: response.output_item.done\ndata: " +
						`{"type":"response.output_item.done","output_index":` + string(rune('0'+i)) + `,"item":` + item + "}\n\n")
				}
				stream.WriteString("event: " + terminal + "\ndata: " +
					`{"type":"` + terminal + `","response":{"id":"resp_s","object":"response","status":"completed","model":"grok-4.7",` +
					`"output":[` + strings.Join(items, ",") + `],` +
					`"usage":{"input_tokens":300,"output_tokens":20,"output_tokens_details":{"reasoning_tokens":30},"total_tokens":350,` +
					`"cost_in_usd_ticks":5000000,` +
					`"server_side_tool_usage_details":{"web_search_calls":1,"x_search_calls":1,"x_posts_fetched":10,"x_users_fetched":0}}}}` +
					"\n\ndata: [DONE]\n\n")
				_, _ = w.Write([]byte(stream.String()))
			}))
			defer upstream.Close()
			prx, dbStub := newXAIBillingTestProxy(t, upstream, "xai-direct")

			serveXAIBillingRequest(t, prx, "/v1/responses", `{"model":"grok-4.7","input":"news?","stream":true,
				"tools":[{"type":"web_search"},{"type":"x_search"}]}`)

			require.Len(t, dbStub.loggedEntries, 1)
			entry := dbStub.loggedEntries[0]
			metadata := decodeMetadata(t, entry.Metadata)
			serverToolUse := metadata["usage_object"].(map[string]interface{})["server_tool_use"].(map[string]interface{})
			assert.Equal(t, float64(1), serverToolUse["web_search_requests"],
				"the terminal usage counter wins over the two streamed web_search_call items")
			assert.Equal(t, float64(10), serverToolUse["x_posts_fetched"])
			wantTokens := 300*0.0000026 + (20+30)*0.0000078
			wantTools := 0.0065 + 10*0.0065
			assert.InDelta(t, wantTokens+wantTools, entry.Spend, 1e-12)
			assert.InDelta(t, 0.0005, metadata["provider_reported_cost"], 1e-12)
		})
	}
}

// xAI repeats the cumulative usage object on every chat stream chunk; the
// final figures are billed once, not summed per chunk.
func TestProxyRequest_XAIChatStreamCumulativeUsageNotSummed(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunk := func(delta, usage string) string {
			return `data: {"id":"c1","object":"chat.completion.chunk","model":"grok-4.7","choices":[{"index":0,"delta":` +
				delta + `}],"usage":` + usage + "}\n\n"
		}
		_, _ = w.Write([]byte(
			chunk(`{"role":"assistant","content":""}`,
				`{"prompt_tokens":100,"completion_tokens":0,"total_tokens":150,"completion_tokens_details":{"reasoning_tokens":50},`+
					`"server_side_tool_usage_details":{"document_search_calls":1}}`) +
				chunk(`{"content":"Hel"}`,
					`{"prompt_tokens":100,"completion_tokens":5,"total_tokens":355,"completion_tokens_details":{"reasoning_tokens":250},`+
						`"server_side_tool_usage_details":{"document_search_calls":2}}`) +
				`data: {"id":"c1","object":"chat.completion.chunk","model":"grok-4.7","choices":[],` +
				`"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":360,"completion_tokens_details":{"reasoning_tokens":250},` +
				`"server_side_tool_usage_details":{"document_search_calls":2},"cost_in_usd_ticks":20000000}}` + "\n\n" +
				"data: [DONE]\n\n"))
	}))
	defer upstream.Close()
	prx, dbStub := newXAIBillingTestProxy(t, upstream, "xai-direct")

	serveXAIBillingRequest(t, prx, "/v1/chat/completions",
		`{"model":"grok-4.7","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"summarize"},{"type":"file","file":{"file_id":"file-1"}}]}]}`)

	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 100, entry.PromptTokens)
	assert.Equal(t, 10, entry.CompletionTokens)
	metadata := decodeMetadata(t, entry.Metadata)
	assert.Equal(t, "additive", metadata["reasoning_tokens_accounting"])
	serverToolUse := metadata["usage_object"].(map[string]interface{})["server_tool_use"].(map[string]interface{})
	assert.Equal(t, float64(2), serverToolUse["attachment_search_calls"])
	assert.InDelta(t, 100*0.0000026+(10+250)*0.0000078+2*0.013, entry.Spend, 1e-12)
	assert.InDelta(t, 0.002, metadata["provider_reported_cost"], 1e-12)
}
