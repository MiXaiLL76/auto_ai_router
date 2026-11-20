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

// alibabaQwen4ResponsesUsage is the Responses API usage Alibaba's Qwen4
// models answer with: the explicit cache write is split in
// cache_write_tokens_details into a five-minute and a one-hour part
// (ttl_5m_tokens / ttl_1h_tokens) — the two tiers Anthropic calls 5m and 1h,
// each billed at its own rate. Alibaba's input_tokens is the whole input,
// the cached and the written parts included. One line, so it can be a single
// SSE data line too.
const alibabaQwen4ResponsesUsage = `{"input_tokens": 200, "output_tokens": 50, "total_tokens": 250, "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 120, "cache_write_tokens_details": {"ttl_5m_tokens": 40, "ttl_1h_tokens": 80}}, "output_tokens_details": {"reasoning_tokens": 0}}`

// alibabaQwen4Price is qwen4-max's price row: the one-hour cache write is
// priced above the five-minute one, which is why the split has to be parsed —
// billed as one tier the write costs less than what Alibaba charges for it.
func alibabaQwen4Price() *pricing.ModelPrice {
	return &pricing.ModelPrice{
		InputCostPerToken:                   1,
		OutputCostPerToken:                  2,
		CacheReadInputTokenCost:             0.5,
		CacheCreationInputTokenCost:         3,
		CacheCreationInputTokenCostAbove1hr: 6,
	}
}

// alibabaQwen4Proxy builds a proxy whose openai credential (Alibaba's are of
// that type, and its passthrough_responses is on by default) serves the model
// qwen4-max.
func alibabaQwen4Proxy(t *testing.T, upstreamURL string, dbStub *stubLiteLLMManager) *Proxy {
	t.Helper()

	builder := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "alibaba-qwen4",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstreamURL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     5_000_000,
		}).
		WithMasterKey("master-key")
	builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "qwen4-max", Credential: "alibaba-qwen4"},
	})
	builder.config.ModelManager.LoadModelsFromConfig(builder.config.Credentials)
	prx := builder.Build()
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "qwen4-max", alibabaQwen4Price())
	return prx
}

// assertAlibabaQwen4CacheTTLSplit checks the spend log of a request that
// wrote 40 tokens to the five-minute cache and 80 to the one-hour one:
// both tiers have to be there, and the one-hour part has to be billed at the
// price above the five-minute one (600 = 40 x 3 + 80 x 6; as one tier the
// write would cost 360 and the hour-long part would be sold below cost).
func assertAlibabaQwen4CacheTTLSplit(t *testing.T, entrySpend float64, metadata map[string]interface{}) {
	t.Helper()

	usageObject := metadata["usage_object"].(map[string]interface{})
	inputDetails := usageObject["prompt_tokens_details"].(map[string]interface{})
	assert.Equal(t, float64(120), inputDetails["cache_creation_tokens"])

	creationDetails, ok := inputDetails["cache_creation_token_details"].(map[string]interface{})
	require.True(t, ok, "the TTL split must reach the spend log")
	assert.Equal(t, float64(40), creationDetails["ephemeral_5m_input_tokens"])
	assert.Equal(t, float64(80), creationDetails["ephemeral_1h_input_tokens"])

	costBreakdown := metadata["cost_breakdown"].(map[string]interface{})
	assert.Equal(t, float64(600), costBreakdown["cache_creation_cost"])
	// 0 regular input (200 - 80 cached - 120 written) + 40 cache reads x0.5
	// + 600 cache writes + 50 output x2
	assert.Equal(t, float64(740), costBreakdown["total_cost"])
	assert.InDelta(t, 740, entrySpend, 1e-9)
}

// TestProxyRequest_AlibabaQwen4ResponsesCacheTTLSplit: a /v1/responses
// request to Alibaba's qwen4-max, whose answer carries the explicit cache
// write split into a five-minute and a one-hour part. The one-hour part must
// be billed at the price above the five-minute one.
func TestProxyRequest_AlibabaQwen4ResponsesCacheTTLSplit(t *testing.T) {
	const upstreamBody = `{
		"id": "resp_alibaba_qwen4",
		"object": "response",
		"created_at": 1,
		"status": "completed",
		"model": "qwen4-max",
		"output": [{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
		"usage": ` + alibabaQwen4ResponsesUsage + `
	}`

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(upstreamBody))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := alibabaQwen4Proxy(t, upstream.URL, dbStub)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"qwen4-max","input":"hello"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 200, entry.PromptTokens)
	assert.Equal(t, 50, entry.CompletionTokens)

	assertAlibabaQwen4CacheTTLSplit(t, entry.Spend, decodeMetadata(t, entry.Metadata))
}

// TestProxyRequest_AlibabaQwen4ResponsesStreamCacheTTLSplit: the same split
// in a streamed answer (the terminal event's usage), which the router reads
// on its own path.
func TestProxyRequest_AlibabaQwen4ResponsesStreamCacheTTLSplit(t *testing.T) {
	sse := "event: response.completed\ndata: " +
		`{"type":"response.completed","response":{"id":"resp_alibaba_qwen4","object":"response","status":"completed","model":"qwen4-max",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],` +
		`"usage":` + alibabaQwen4ResponsesUsage + `}}` +
		"\n\ndata: [DONE]\n\n"

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	t.Cleanup(upstream.Close)

	dbStub := &stubLiteLLMManager{}
	prx := alibabaQwen4Proxy(t, upstream.URL, dbStub)

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"qwen4-max","input":"hello","stream":true}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	assert.Equal(t, 200, entry.PromptTokens)
	assert.Equal(t, 50, entry.CompletionTokens)

	assertAlibabaQwen4CacheTTLSplit(t, entry.Spend, decodeMetadata(t, entry.Metadata))
}
