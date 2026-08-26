package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_CometAPIOpenAIProtocol_UsesChatCompletionsWire verifies that a
// cometapi credential with OpenAIProtocol=true is routed through the plain
// OpenAI-compatible path: request body passes through unconverted, the
// upstream URL is <base_url>/v1/chat/completions (not /v1/messages), and auth
// uses a bare "Authorization: Bearer" header with no Anthropic-only headers.
func TestProxyRequest_CometAPIOpenAIProtocol_UsesChatCompletionsWire(t *testing.T) {
	type capturedRequest struct {
		path             string
		authHeader       string
		xAPIKeyHeader    string
		anthropicVersion string
		body             []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{
			path:             r.URL.Path,
			authHeader:       r.Header.Get("Authorization"),
			xAPIKeyHeader:    r.Header.Get("X-Api-Key"),
			anthropicVersion: r.Header.Get("anthropic-version"),
			body:             body,
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-test","object":"chat.completion","model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
		}`))
	}))
	defer upstream.Close()

	cred := config.CredentialConfig{
		Name:           "comet-openai",
		Type:           config.ProviderTypeCometAPI,
		BaseURL:        upstream.URL,
		APIKey:         "upstream-key",
		OpenAIProtocol: true,
		RPM:            100,
		TPM:            10000,
	}
	proxy := NewTestProxyBuilder().WithCredentials(cred).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-4o",
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	assert.Equal(t, "/v1/chat/completions", outbound.path)
	assert.Equal(t, "Bearer upstream-key", outbound.authHeader)
	assert.Empty(t, outbound.xAPIKeyHeader)
	assert.Empty(t, outbound.anthropicVersion)

	var body map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &body))
	// OpenAI-protocol passthrough must not translate to Anthropic's shape
	// (which uses a top-level "max_tokens" and a "system" field, not present here).
	assert.NotContains(t, body, "anthropic_version")
	assert.Contains(t, body, "messages")
}

// TestProxyRequest_CometAPIDefault_StillUsesAnthropicWire is the control case:
// without OpenAIProtocol, cometapi keeps using /v1/messages with Anthropic-style
// auth headers, unaffected by the new opt-in field.
func TestProxyRequest_CometAPIDefault_StillUsesAnthropicWire(t *testing.T) {
	type capturedRequest struct {
		path             string
		xAPIKeyHeader    string
		anthropicVersion string
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		captured <- capturedRequest{
			path:             r.URL.Path,
			xAPIKeyHeader:    r.Header.Get("X-Api-Key"),
			anthropicVersion: r.Header.Get("anthropic-version"),
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-7",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("comet", config.ProviderTypeCometAPI, upstream.URL, "upstream-key").
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"claude-opus-4.7",
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	assert.Equal(t, "/v1/messages", outbound.path)
	assert.Equal(t, "upstream-key", outbound.xAPIKeyHeader)
	assert.Equal(t, "2023-06-01", outbound.anthropicVersion)
}

// TestHandleProviderStreaming_CometAPIOpenAIProtocol_SkipsAnthropicTransform
// verifies the streaming dispatch: with OpenAIProtocol=true, an already
// OpenAI-shaped SSE chunk must reach the client unchanged (passthrough path),
// not through handleAnthropicCompatibleStreaming's Anthropic-event parser,
// which knows nothing about a "choices"/"delta" shaped chunk and would drop it.
func TestHandleProviderStreaming_CometAPIOpenAIProtocol_SkipsAnthropicTransform(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := &config.CredentialConfig{Name: "comet-openai", Type: config.ProviderTypeCometAPI, OpenAIProtocol: true}
	rawStream := `data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n" + "data: [DONE]\n\n"
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(rawStream)),
	}
	logCtx := &RequestLogContext{
		RequestID:   "comet-openai-proto-stream",
		StartTime:   time.Now().UTC(),
		Credential:  cred,
		ModelID:     "gpt-4o",
		RealModelID: "gpt-4o",
	}
	w := httptest.NewRecorder()

	_ = prx.handleProviderStreaming(w, resp, cred, "gpt-4o", "gpt-4o", logCtx)

	assert.Contains(t, w.Body.String(), `"content":"hi"`)
}

// TestHandleProviderStreaming_CometAPIDefault_UsesAnthropicTransform is the
// control case: without OpenAIProtocol, cometapi keeps parsing upstream SSE
// as Anthropic-shaped events and transforming them into OpenAI chunks.
func TestHandleProviderStreaming_CometAPIDefault_UsesAnthropicTransform(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := &config.CredentialConfig{Name: "comet", Type: config.ProviderTypeCometAPI}
	anthropicEvents := []string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"usage":{"input_tokens":5}}}`,
		`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
		`event: message_delta` + "\n" + `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
	}
	rawStream := strings.Join(anthropicEvents, "\n\n") + "\n\n"
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(rawStream)),
	}
	logCtx := &RequestLogContext{
		RequestID:   "comet-default-stream",
		StartTime:   time.Now().UTC(),
		Credential:  cred,
		ModelID:     "claude-haiku-4.5",
		RealModelID: "claude-haiku-4.5",
	}
	w := httptest.NewRecorder()

	_ = prx.handleProviderStreaming(w, resp, cred, "claude-haiku-4.5", "claude-haiku-4.5", logCtx)

	assert.Contains(t, w.Body.String(), `"content":"hi"`)
	assert.Contains(t, w.Body.String(), "chat.completion.chunk")
}
