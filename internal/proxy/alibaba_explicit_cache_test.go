package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// alibabaExplicitCacheResponsesUsage is the Responses API usage Alibaba
// answers an explicit cache read with: the mode ("ephemeral") is named only
// per billing line in x_details, not in input_tokens_details. One line, so it
// can be a single SSE data line too.
const alibabaExplicitCacheResponsesUsage = `{"input_tokens": 200, "output_tokens": 50, "total_tokens": 250, "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens_details": {}}, "output_tokens_details": {"reasoning_tokens": 0}, "x_details": [{"input_tokens": 200, "output_tokens": 50, "total_tokens": 250, "prompt_tokens_details": {"cached_tokens": 80, "cache_type": "ephemeral", "cache_creation_input_tokens": 0, "cache_write_tokens": 0}, "x_billing_type": "response_api"}]}`

// alibabaExplicitCacheProxy is alibabaQwen4Proxy with an explicit cache read
// priced at half the implicit one, as Alibaba prices it.
func alibabaExplicitCacheProxy(t *testing.T, upstreamURL string, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()

	prx := alibabaQwen4Proxy(t, upstreamURL, dbStub)
	setTestModelPrice(prx, "qwen4-max", &pricing.ModelPrice{
		InputCostPerToken:               1,
		OutputCostPerToken:              2,
		CacheReadInputTokenCost:         0.5,
		ExplicitCacheReadInputTokenCost: 0.25,
	})
	return prx
}

// assertAlibabaExplicitCacheRead checks the spend log of a request that read
// 80 tokens from the explicit cache: they are billed at the explicit tariff
// (20 = 80 x 0.25), not the implicit one (40), and the mode reaches the log.
func assertAlibabaExplicitCacheRead(t *testing.T, entrySpend float64, metadata map[string]interface{}) {
	t.Helper()

	usageObject := metadata["usage_object"].(map[string]interface{})
	inputDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, "ephemeral", inputDetails["cache_type"])

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(20), costBreakdown["explicit_cache_read_cost"])
	assert.Equal(t, float64(0), costBreakdown["cached_input_cost"])
	// 120 regular input x1 + 20 explicit cache reads + 50 output x2
	assert.Equal(t, float64(240), costBreakdown["total_cost"])
	assert.InDelta(t, 240, entrySpend, 1e-9)
}

// TestProxyRequest_AlibabaResponsesExplicitCacheRead: a /v1/responses
// request to Alibaba whose answer names the explicit cache only in x_details.
// The cached read must be billed at the explicit tariff.
func TestProxyRequest_AlibabaResponsesExplicitCacheRead(t *testing.T) {
	const upstreamBody = `{
		"id": "resp_alibaba_explicit",
		"object": "response",
		"created_at": 1,
		"status": "completed",
		"model": "qwen4-max",
		"output": [{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
		"usage": ` + alibabaExplicitCacheResponsesUsage + `
	}`

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(upstreamBody))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := alibabaExplicitCacheProxy(t, upstream.URL, dbStub)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"qwen4-max","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assertAlibabaExplicitCacheRead(t, entry.Spend, decodeMetadata(t, entry.Metadata))
}

// TestProxyRequest_AlibabaResponsesStreamExplicitCacheRead: the same answer
// streamed (the terminal event's usage), which the router reads on its own
// path.
func TestProxyRequest_AlibabaResponsesStreamExplicitCacheRead(t *testing.T) {
	sse := "event: response.completed\ndata: " +
		`{"type":"response.completed","response":{"id":"resp_alibaba_explicit","object":"response","status":"completed","model":"qwen4-max",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],` +
		`"usage":` + alibabaExplicitCacheResponsesUsage + `}}` +
		"\n\ndata: [DONE]\n\n"

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := alibabaExplicitCacheProxy(t, upstream.URL, dbStub)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"qwen4-max","input":"hello","stream":true}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assertAlibabaExplicitCacheRead(t, entry.Spend, decodeMetadata(t, entry.Metadata))
}
