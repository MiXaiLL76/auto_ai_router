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

// TestProxyRequest_ResponsesIncompleteEventStillBillsRealUsage reproduces a
// real production bug: DashScope (and potentially other providers) sends full
// usage in the terminal "response.incomplete" event when the model stops
// early (e.g. max_output_tokens reached), not only in "response.completed".
// handlePassthroughResponsesStreaming used to only capture usage on
// "response.completed", so an incomplete response fell through to
// finalizeStreamingLog's local-tiktoken-estimate fallback for PromptTokens
// instead of the real, much larger provider-reported value — silently
// undercounting both the rate limiter and the billed spend.
//
// The mock "response.incomplete" payload is deliberately padded past 8 KB
// (streamBufPool's fixed read-buffer size in streamToClient), so the SSE
// line genuinely arrives across multiple onChunk calls, each holding only a
// raw byte-level fragment. This matters: without the fix,
// completedEventPayload is never captured, and finalizeStreamingLog's
// type-agnostic fallback extractor runs on lastRawChunk — the last such
// fragment, which is NOT valid/parseable JSON on its own. Only the properly
// reassembled (partialSSELine-stitched) full line, captured into
// completedEventPayload by the "response.completed" || "response.incomplete"
// check, parses successfully. A small (<8 KB) mock payload would accidentally
// mask the bug: sanitizingSSEReader's internal bufio.Reader buffers a whole
// short line before streamToClient's Read() loop ever sees it, so
// lastRawChunk would already hold the complete, valid line regardless of the
// fix — this was confirmed by first writing this test without padding and
// watching it pass even with the fix reverted.
//
// The request body is also deliberately a short, few-token prompt while the
// upstream reports a huge real input_tokens count: if usage extraction ever
// falls back to estimating tokens from the request body again, PromptTokens
// comes back tiny instead of matching the upstream value — failing
// unambiguously either way.
func TestProxyRequest_ResponsesIncompleteEventStillBillsRealUsage(t *testing.T) {
	const (
		realInputTokens  = 150000
		realOutputTokens = 30
		realTotalTokens  = realInputTokens + realOutputTokens
	)

	// Padding lives in a field the decoder doesn't map (silently ignored by
	// json.Unmarshal), placed BEFORE "usage" so the fragment(s) preceding the
	// real usage object are pushed past the 8 KB read-buffer boundary.
	padding := strings.Repeat("x", 16*1024)

	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		data := "event: response.incomplete\ndata: " +
			`{"type":"response.incomplete","response":{"id":"resp_incomplete","object":"response",` +
			`"status":"incomplete","model":"gpt-incomplete-usage",` +
			`"incomplete_details":{"reason":"max_output_tokens"},` +
			`"_padding":"` + padding + `",` +
			`"output":[{"type":"message","id":"msg_1","status":"incomplete","role":"assistant","content":[]}],` +
			`"usage":{"input_tokens":150000,"output_tokens":30,"total_tokens":150030}}}` +
			"\n\ndata: [DONE]\n\n"
		_, _ = w.Write([]byte(data))
	}))
	defer upstream.Close()

	dbStub := &stubLiteLLMManager{}
	prx := NewTestProxyBuilder().
		WithCredentials(config.CredentialConfig{
			Name:    "openai-incomplete-usage",
			Type:    config.ProviderTypeOpenAI,
			BaseURL: upstream.URL,
			APIKey:  "upstream-key",
			RPM:     100,
			TPM:     10000,
		}).
		WithMasterKey("master-key").
		Build()
	prx.LiteLLMDB = dbStub
	setTestModelPrice(prx, "gpt-incomplete-usage", &pricing.ModelPrice{
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	})

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model": "gpt-incomplete-usage",
		"input": "hi",
		"stream": true
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Len(t, dbStub.loggedEntries, 1)

	entry := dbStub.loggedEntries[0]
	assert.Equal(t, realInputTokens, entry.PromptTokens,
		"PromptTokens must come from the provider's response.incomplete usage, not a tiktoken estimate of the short request body")
	assert.Equal(t, realOutputTokens, entry.CompletionTokens)
	assert.Equal(t, realTotalTokens, entry.TotalTokens)
	assert.Equal(t, float64(realInputTokens)*1+float64(realOutputTokens)*2, entry.Spend)
}
