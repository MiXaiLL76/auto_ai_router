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

// TestProxyRequest_SendsReasoningBetaAsHeaderForAnthropicFamily covers /v1/chat/completions
// requests converted to Anthropic's wire format for a real Anthropic-compatible backend
// (Anthropic, ProMan): the real Anthropic API rejects "anthropic_beta" as a body field
// ("anthropic_beta: Extra inputs are not permitted"), so the synthesized effort-2025-11-24
// beta — like any anthropic_beta — must always travel via the anthropic-beta HTTP header,
// never left in the body.
func TestProxyRequest_SendsReasoningBetaAsHeaderForAnthropicFamily(t *testing.T) {
	for _, providerType := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeProMan} {
		t.Run(string(providerType), func(t *testing.T) {
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
				WithSingleCredential("provider", providerType, upstream.URL, "upstream-key").
				Build()
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
				"model":"claude-opus-4.7",
				"messages":[{"role":"user","content":"think"}],
				"reasoning_effort":"high",
				"max_completion_tokens":2048
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			proxy.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			outbound := <-captured
			assert.Equal(t, "effort-2025-11-24", outbound.header)
			var body map[string]any
			require.NoError(t, json.Unmarshal(outbound.body, &body))
			assert.NotContains(t, body, "anthropic_beta")
		})
	}
}

func TestProxyRequest_ForwardsMessagesBetaAsHeader(t *testing.T) {
	for _, providerType := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeAIR} {
		t.Run(string(providerType), func(t *testing.T) {
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
				WithSingleCredential("provider", providerType, upstream.URL, "upstream-key").
				Build()
			req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
				"model":"claude-opus-4.7",
				"max_tokens":100,
				"messages":[{"role":"user","content":"hi"}],
				"thinking":{"type":"adaptive"},
				"output_config":{"effort":"high"},
				"anthropic_beta":["prompt-caching-2024-07-31"]
			}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("anthropic-beta", "client-beta")
			w := httptest.NewRecorder()

			proxy.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code)
			outbound := <-captured
			var body map[string]any
			require.NoError(t, json.Unmarshal(outbound.body, &body))
			if providerType == config.ProviderTypeAnthropic {
				// Anthropic is passthrough-eligible by default (IsPassthroughMessagesForProvider):
				// the request is forwarded natively, so the client's own body-level
				// anthropic_beta survives (merged into the header) alongside the synthesized
				// effort beta, instead of being discarded by the Messages->Chat->Messages
				// round trip. The real Anthropic API rejects "anthropic_beta" as a body field
				// outright, so ExtractBetaHeader must strip it from the body either way.
				assert.Equal(t, "client-beta,prompt-caching-2024-07-31,effort-2025-11-24", outbound.header)
			} else {
				assert.Equal(t, "client-beta,prompt-caching-2024-07-31", outbound.header)
			}
			assert.NotContains(t, body, "anthropic_beta")
		})
	}
}
