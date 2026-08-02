package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractMetadataFromBodyStripsClientControlledServiceTier(t *testing.T) {
	tests := []struct {
		name              string
		body              string
		wantStream        bool
		wantSessionID     string
		wantStreamOptions bool
	}{
		{
			name: "non-streaming chat top level",
			body: `{"model":"gpt-4","messages":[],"service_tier":"priority"}`,
		},
		{
			name:              "streaming chat top level",
			body:              `{"model":"gpt-4","messages":[],"stream":true,"service_tier":"priority"}`,
			wantStream:        true,
			wantStreamOptions: true,
		},
		{
			name: "non-streaming responses",
			body: `{"model":"gpt-4","input":"hello","service_tier":"priority"}`,
		},
		{
			name:       "streaming responses does not gain stream options",
			body:       `{"model":"gpt-4","input":"hello","stream":true,"service_tier":"priority"}`,
			wantStream: true,
		},
		{
			name:          "extra body",
			body:          `{"model":"gpt-4","messages":[],"extra_body":{"service_tier":"priority","litellm_session_id":"session-1"}}`,
			wantSessionID: "session-1",
		},
		{
			name:          "top level and extra body",
			body:          `{"model":"gpt-4","messages":[],"service_tier":null,"extra_body":{"service_tier":[],"litellm_session_id":"session-1"}}`,
			wantSessionID: "session-1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, stream, sessionID, sanitized, err := extractMetadataFromBody([]byte(tt.body), "application/json")
			require.NoError(t, err)
			assert.Equal(t, "gpt-4", model)
			assert.Equal(t, tt.wantStream, stream)
			assert.Equal(t, tt.wantSessionID, sessionID)

			var body map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(sanitized, &body))
			assert.NotContains(t, body, "service_tier")
			if extraRaw, exists := body["extra_body"]; exists {
				var extra map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(extraRaw, &extra))
				assert.NotContains(t, extra, "service_tier")
				if tt.wantSessionID != "" {
					assert.JSONEq(t, `"`+tt.wantSessionID+`"`, string(extra["litellm_session_id"]))
				}
			}

			var streamOptions map[string]bool
			streamOptionsRaw, hasStreamOptions := body["stream_options"]
			assert.Equal(t, tt.wantStreamOptions, hasStreamOptions)
			if hasStreamOptions {
				require.NoError(t, json.Unmarshal(streamOptionsRaw, &streamOptions))
				assert.True(t, streamOptions["include_usage"])
			}
		})
	}
}

func TestStripClientControlledServiceTierAcceptsEveryJSONType(t *testing.T) {
	values := []string{`"priority"`, `null`, `123`, `{}`, `[]`}
	for _, value := range values {
		t.Run(value, func(t *testing.T) {
			body := []byte(`{"model":"gpt-4","messages":[],"service_tier":` + value + `,"extra_body":{"service_tier":` + value + `,"keep":true}}`)
			_, _, _, sanitized, err := extractMetadataFromBody(body, "application/json")
			require.NoError(t, err)
			var parsed map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(sanitized, &parsed))
			assert.NotContains(t, parsed, "service_tier")
			var extra map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(parsed["extra_body"], &extra))
			assert.NotContains(t, extra, "service_tier")
			assert.Equal(t, "true", string(extra["keep"]))
		})
	}
}

func TestServiceTierSanitizationPreservesUnrelatedData(t *testing.T) {
	body := []byte(`{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"{\"service_tier\":\"priority\"}"}],
		"metadata":{"service_tier":"customer-tag"},
		"tools":[{"type":"function","function":{"parameters":{"properties":{"service_tier":{"type":"string"}}}}}],
		"extra_body":{"service_tier":"priority","litellm_session_id":"session-1","provider_number":9223372036854775806},
		"seed":9223372036854775807,
		"service_tier":"priority"
	}`)

	_, _, sessionID, sanitized, err := extractMetadataFromBody(body, "application/json")
	require.NoError(t, err)
	assert.Equal(t, "session-1", sessionID)

	var parsed map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(sanitized, &parsed))
	assert.NotContains(t, parsed, "service_tier")
	assert.Equal(t, "9223372036854775807", string(parsed["seed"]))
	assert.JSONEq(t, `{"service_tier":"customer-tag"}`, string(parsed["metadata"]))
	assert.Contains(t, string(parsed["messages"]), `\"service_tier\"`)
	assert.Contains(t, string(parsed["tools"]), `"service_tier"`)

	var extra map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(parsed["extra_body"], &extra))
	assert.NotContains(t, extra, "service_tier")
	assert.Equal(t, "9223372036854775806", string(extra["provider_number"]))
}

func TestExtractMetadataFromBodyDoesNotRebuildUnchangedNonStreamingBody(t *testing.T) {
	body := []byte("{ \"model\" : \"gpt-4\", \"messages\" : [], \"seed\" : 9223372036854775807 }")
	_, stream, _, got, err := extractMetadataFromBody(body, "application/json")
	require.NoError(t, err)
	assert.False(t, stream)
	assert.Equal(t, body, got)
}

func TestProxyRequestNeverForwardsClientControlledServiceTier(t *testing.T) {
	tests := []struct {
		name     string
		provider config.ProviderType
		path     string
		body     string
	}{
		{name: "proxy chat", provider: config.ProviderTypeProxy, path: "/v1/chat/completions", body: `{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"service_tier":"priority","extra_body":{"service_tier":"flex","keep":"yes"}}`},
		{name: "AIR chat", provider: config.ProviderTypeAIR, path: "/v1/chat/completions", body: `{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"service_tier":"priority"}`},
		{name: "direct OpenAI chat", provider: config.ProviderTypeOpenAI, path: "/v1/chat/completions", body: `{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"service_tier":"priority"}`},
		{name: "native responses passthrough", provider: config.ProviderTypeOpenAI, path: "/v1/responses", body: `{"model":"gpt-4","input":"test","service_tier":"priority"}`},
		{name: "Anthropic messages proxy", provider: config.ProviderTypeProxy, path: "/v1/messages", body: `{"model":"gpt-4","max_tokens":16,"messages":[{"role":"user","content":"test"}],"service_tier":"priority"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var outbound map[string]json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, tt.path, r.URL.Path)
				require.NoError(t, json.NewDecoder(r.Body).Decode(&outbound))
				w.Header().Set("Content-Type", "application/json")
				if tt.path == "/v1/responses" {
					_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","status":"completed","model":"gpt-4","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
					return
				}
				if tt.path == "/v1/messages" {
					_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"gpt-4","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
					return
				}
				_, _ = io.WriteString(w, `{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("upstream", tt.provider, upstream.URL, "upstream-key").
				WithMasterKey("master-key").
				Build()
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()

			prx.ProxyRequest(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			assert.NotContains(t, outbound, "service_tier")
			assert.Contains(t, outbound, "model")
			if tt.path == "/v1/chat/completions" {
				assert.Contains(t, outbound, "messages")
			}
			if extraRaw, exists := outbound["extra_body"]; exists {
				var extra map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(extraRaw, &extra))
				assert.NotContains(t, extra, "service_tier")
				assert.JSONEq(t, `"yes"`, string(extra["keep"]))
			}
		})
	}
}

func TestFallbackInheritsSanitizedServiceTierBody(t *testing.T) {
	received := make(chan map[string]json.RawMessage, 2)
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		received <- body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"rate limited","type":"rate_limit_error"}}`)
	}))
	defer primary.Close()
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback(primary.URL, fallback.URL).
		WithMasterKey("master-key").
		Build()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"service_tier":"priority","extra_body":{"service_tier":"flex","keep":true}}`,
	))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
	for range 2 {
		body := <-received
		assert.NotContains(t, body, "service_tier")
		var extra map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(body["extra_body"], &extra))
		assert.NotContains(t, extra, "service_tier")
		assert.Equal(t, "true", string(extra["keep"]))
	}
}

func TestResponsesWebSocketNeverForwardsClientControlledServiceTier(t *testing.T) {
	outbound := make(chan map[string]json.RawMessage, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		outbound <- body
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-4\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\ndata: [DONE]\n\n")
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("upstream", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMasterKey("master-key").
		Build()
	server := httptest.NewServer(http.HandlerFunc(prx.HandleWebSocketResponses))
	defer server.Close()

	header := http.Header{"Authorization": []string{"Bearer master-key"}}
	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(server.URL, "http")+"/v1/responses", header)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, conn.Close()) })
	require.NoError(t, conn.WriteJSON(map[string]interface{}{
		"type":         "response.create",
		"model":        "gpt-4",
		"input":        "test",
		"service_tier": "priority",
	}))

	select {
	case body := <-outbound:
		assert.NotContains(t, body, "service_tier")
		assert.Contains(t, body, "input")
	case <-time.After(3 * time.Second):
		t.Fatal("upstream did not receive the WebSocket request")
	}
}
