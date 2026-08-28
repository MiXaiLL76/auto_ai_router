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

// TestProxyRequest_CometAPIGoogleProtocol_UsesGenerateContentWire verifies that a
// cometapi credential with GoogleProtocol=true is routed through the Google
// GenAI-compatible path: the OpenAI request body is converted to Gemini's
// contents/generationConfig shape, the upstream URL is
// <base_url>/v1beta/models/<model>:generateContent (not /v1/messages), auth
// travels as the x-goog-api-key header, and the Gemini response is converted
// back to OpenAI Chat Completions shape.
func TestProxyRequest_CometAPIGoogleProtocol_UsesGenerateContentWire(t *testing.T) {
	type capturedRequest struct {
		path             string
		xGoogAPIKey      string
		authHeader       string
		anthropicVersion string
		body             []byte
	}
	captured := make(chan capturedRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- capturedRequest{
			path:             r.URL.Path,
			xGoogAPIKey:      r.Header.Get("x-goog-api-key"),
			authHeader:       r.Header.Get("Authorization"),
			anthropicVersion: r.Header.Get("anthropic-version"),
			body:             body,
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}
		}`))
	}))
	defer upstream.Close()

	cred := config.CredentialConfig{
		Name:           "comet-google",
		Type:           config.ProviderTypeCometAPI,
		BaseURL:        upstream.URL,
		APIKey:         "upstream-key",
		GoogleProtocol: true,
		RPM:            100,
		TPM:            10000,
	}
	proxy := NewTestProxyBuilder().WithCredentials(cred).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gemini-3.1-flash",
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	outbound := <-captured
	assert.Equal(t, "/v1beta/models/gemini-3.1-flash:generateContent", outbound.path)
	assert.Equal(t, "upstream-key", outbound.xGoogAPIKey)
	assert.Empty(t, outbound.authHeader)
	assert.Empty(t, outbound.anthropicVersion)

	var reqBody map[string]any
	require.NoError(t, json.Unmarshal(outbound.body, &reqBody))
	// Google GenAI shape, not OpenAI ("messages") nor Anthropic ("system").
	assert.Contains(t, reqBody, "contents")
	assert.NotContains(t, reqBody, "messages")

	var respBody map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &respBody))
	assert.Equal(t, "chat.completion", respBody["object"])
	assert.Contains(t, w.Body.String(), `"content":"hi"`)
}

// TestProxyRequest_CometAPIGoogleProtocol_MessagesNotPassedThrough verifies that
// an incoming /v1/messages request for a google_proto credential is NOT forwarded
// as raw Anthropic Messages JSON (which the Gemini endpoint cannot parse) — it is
// converted to Gemini's generateContent shape like any other request.
func TestProxyRequest_CometAPIGoogleProtocol_MessagesNotPassedThrough(t *testing.T) {
	captured := make(chan []byte, 1)
	capturedPath := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- body
		capturedPath <- r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}
		}`))
	}))
	defer upstream.Close()

	cred := config.CredentialConfig{
		Name:           "comet-google",
		Type:           config.ProviderTypeCometAPI,
		BaseURL:        upstream.URL,
		APIKey:         "upstream-key",
		GoogleProtocol: true,
		RPM:            100,
		TPM:            10000,
	}
	proxy := NewTestProxyBuilder().WithCredentials(cred).Build()

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{
		"model":"gemini-3.1-flash",
		"max_tokens":64,
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "/v1beta/models/gemini-3.1-flash:generateContent", <-capturedPath)

	var reqBody map[string]any
	require.NoError(t, json.Unmarshal(<-captured, &reqBody))
	assert.Contains(t, reqBody, "contents")
	assert.NotContains(t, reqBody, "messages")
}

// TestHandleProviderStreaming_CometAPIGoogleProtocol_UsesVertexTransform verifies
// the streaming dispatch: with GoogleProtocol=true, upstream SSE is parsed as
// Gemini-shaped events and transformed into OpenAI chunks (handleVertexStreaming),
// not handleAnthropicCompatibleStreaming.
func TestHandleProviderStreaming_CometAPIGoogleProtocol_UsesVertexTransform(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := &config.CredentialConfig{Name: "comet-google", Type: config.ProviderTypeCometAPI, GoogleProtocol: true}
	rawStream := `data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}]}` + "\n\n" +
		`data: {"candidates":[{"content":{"parts":[{"text":""}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7}}` + "\n\n"
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(rawStream)),
	}
	logCtx := &RequestLogContext{
		RequestID:   "comet-google-proto-stream",
		StartTime:   time.Now().UTC(),
		Credential:  cred,
		ModelID:     "gemini-3.1-flash",
		RealModelID: "gemini-3.1-flash",
	}
	w := httptest.NewRecorder()

	_ = prx.handleProviderStreaming(w, resp, cred, "gemini-3.1-flash", "gemini-3.1-flash", logCtx)

	assert.Contains(t, w.Body.String(), `"content":"hi"`)
	assert.Contains(t, w.Body.String(), "chat.completion.chunk")
}
