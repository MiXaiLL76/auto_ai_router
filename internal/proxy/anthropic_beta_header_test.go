package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_SendsAnthropicBetaAsHeader(t *testing.T) {
	type capturedRequest struct {
		header string
		body   []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{header: r.Header.Get("anthropic-beta"), body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-7",
			"content":[{"type":"thinking","thinking":"reasoning","signature":"sig"},{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("comet", config.ProviderTypeCometAPI, upstream.URL, "upstream-key").
		Build()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"claude-opus-4.7",
		"messages":[{"role":"user","content":"think"}],
		"reasoning_effort":"high",
		"max_completion_tokens":2048
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-beta", "interleaved-thinking-2025-05-14")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	assert.Equal(t, "interleaved-thinking-2025-05-14,effort-2025-11-24", outbound.header)
	var body map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &body))
	assert.NotContains(t, body, "anthropic_beta")
	assert.Equal(t, "adaptive", body["thinking"].(map[string]any)["type"])
	assert.Equal(t, "summarized", body["thinking"].(map[string]any)["display"])
	assert.Equal(t, "high", body["output_config"].(map[string]any)["effort"])
	assert.Contains(t, w.Body.String(), `"reasoning_content":"reasoning"`)
}

// TestProxyRequest_AIRChain_StripsAnthropicBetaFromForwardedBody reproduces a
// chain-router setup: the client talks to a main node whose credential for this
// model is proxy-like (type "air"/"proxy", e.g. an AIR peer or a sub-router that
// itself terminates at the real Anthropic API). A client sending a native
// /v1/messages request with "anthropic_beta" as a body field (instead of the
// "anthropic-beta" header) must not have that field forwarded verbatim in the
// JSON body — some downstream Anthropic-compatible servers reject unknown body
// fields with "anthropic_beta: Extra inputs are not permitted". It must be
// moved to the anthropic-beta header, exactly as already happens for
// cred.Type == anthropic/cometapi/proman.
func TestProxyRequest_AIRChain_StripsAnthropicBetaFromForwardedBody(t *testing.T) {
	type capturedRequest struct {
		header string
		body   []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{header: r.Header.Get("anthropic-beta"), body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-7",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("air-peer", config.ProviderTypeAIR, upstream.URL, "upstream-key").
		Build()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"anthropic_beta":["prompt-caching-2024-07-31"]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	assert.Equal(t, "prompt-caching-2024-07-31", outbound.header)
	var body map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &body))
	assert.NotContains(t, body, "anthropic_beta")
}

// TestProxyRequest_DirectAnthropic_MessagesBodyBetaField probes the non-chained
// case: a credential of type "anthropic" talked to directly (no proxy/air hop),
// with the client sending a native /v1/messages request that has "anthropic_beta"
// as a top-level BODY field (rather than the "anthropic-beta" header). Unlike the
// proxy-like chain case, this path is NOT a raw passthrough: prepareRequestForCredential
// converts /v1/messages -> chat -> Anthropic (MessagesToChat + OpenAIToAnthropic)
// for any non-proxy-like credential. This test checks what that round-trip does
// with the field: does it survive (in body or header), or does it get silently
// dropped by the intermediate OpenAI-chat representation?
func TestProxyRequest_DirectAnthropic_MessagesBodyBetaField(t *testing.T) {
	type capturedRequest struct {
		header string
		body   []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{header: r.Header.Get("anthropic-beta"), body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-7",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("anthropic-direct", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
		Build()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"claude-opus-4.7",
		"max_tokens":100,
		"messages":[{"role":"user","content":"hi"}],
		"anthropic_beta":["prompt-caching-2024-07-31"]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	var body map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &body))

	// Current (verified) behavior: the /v1/messages -> chat -> Anthropic round-trip
	// (MessagesToChat copies unknown top-level keys as-is; OpenAIToAnthropic only
	// reads betas back out of a nested "extra_body" object) silently drops a
	// body-level "anthropic_beta" field entirely — it reaches neither the outbound
	// header nor the outbound body. This differs from the proxy/air chain case
	// (see TestProxyRequest_AIRChain_StripsAnthropicBetaFromForwardedBody), where
	// the same field survives unchanged in the body and can trigger a strict
	// upstream's "anthropic_beta: Extra inputs are not permitted" rejection.
	// So: no chain -> feature silently lost, not an upstream error.
	assert.Empty(t, outbound.header)
	assert.NotContains(t, body, "anthropic_beta")
}
