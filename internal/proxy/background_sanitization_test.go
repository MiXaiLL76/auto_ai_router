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

func TestSanitizeJSONRequestBodyStripsBackgroundMode(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		wantStream     bool
		wantBackground string // top-level value left in the body, "" when absent
	}{
		{
			name: "non-streaming responses",
			body: `{"model":"gpt-4","input":"hello","background":true}`,
		},
		{
			name:       "streaming responses",
			body:       `{"model":"gpt-4","input":"hello","stream":true,"background":true}`,
			wantStream: true,
		},
		{
			name: "chat completions",
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}],"background":true}`,
		},
		{
			name: "extra body",
			body: `{"model":"gpt-4","input":"hello","extra_body":{"background":true,"keep":1}}`,
		},
		{
			name:           "false at the top level does not hide true in extra body",
			body:           `{"model":"gpt-4","input":"hello","background":false,"extra_body":{"background":true,"keep":1}}`,
			wantBackground: "false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := sanitizeAndExtractRequestBody([]byte(tt.body), "application/json", false)
			require.NoError(t, err)
			assert.True(t, result.Changed)
			assert.Equal(t, "gpt-4", result.ModelID)
			assert.Equal(t, tt.wantStream, result.Streaming)

			var body map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(result.Body, &body))
			assert.Equal(t, tt.wantBackground, string(body["background"]))
			if extraRaw, exists := body["extra_body"]; exists {
				assert.JSONEq(t, `{"keep":1}`, string(extraRaw))
			}
		})
	}
}

func TestBackgroundModeSanitizationKeepsOtherValues(t *testing.T) {
	// "transparent" is the Images API's own string background parameter.
	values := []string{`false`, `null`, `"transparent"`, `"true"`, `1`, `{}`}
	for _, value := range values {
		t.Run(value, func(t *testing.T) {
			body := []byte(`{"model":"gpt-image-2","prompt":"a cat","background":` + value + `,"extra_body":{"background":` + value + `}}`)
			result, err := sanitizeAndExtractRequestBody(body, "application/json", false)
			require.NoError(t, err)
			assert.False(t, result.Changed)
			assert.Equal(t, body, result.Body)
		})
	}
}

func TestSanitizeMultipartRequestBodyKeepsImageBackground(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, "air-background-boundary",
		multipartTestPart{disposition: `form-data; name="model"`, data: []byte("gpt-image-2")},
		multipartTestPart{disposition: `form-data; name="background"`, data: []byte("transparent")},
	)

	result, err := sanitizeAndExtractRequestBody(body, contentType, false)

	require.NoError(t, err)
	_, parts := parseMultipartTestBody(t, result.Body, contentType)
	require.Equal(t, []string{"model", "background"}, multipartPartNames(parts))
	assert.Equal(t, "transparent", string(parts[1].data))
}

func TestProxyRequestNeverForwardsBackgroundMode(t *testing.T) {
	tests := []struct {
		name     string
		provider config.ProviderType
		body     string
		stream   bool
	}{
		{name: "OpenAI responses passthrough", provider: config.ProviderTypeOpenAI, body: `{"model":"gpt-4","input":"test","background":true}`},
		{name: "OpenAI streaming responses passthrough", provider: config.ProviderTypeOpenAI, body: `{"model":"gpt-4","input":"test","stream":true,"background":true}`, stream: true},
		{name: "AIR responses", provider: config.ProviderTypeAIR, body: `{"model":"gpt-4","input":"test","extra_body":{"background":true,"keep":"yes"}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var outbound map[string]json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, "/v1/responses", r.URL.Path)
				require.NoError(t, json.NewDecoder(r.Body).Decode(&outbound))
				if tt.stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-4\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\ndata: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"resp_1","object":"response","status":"completed","model":"gpt-4","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("upstream", tt.provider, upstream.URL, "upstream-key").
				WithMasterKey("master-key").
				Build()
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(tt.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()

			prx.ProxyRequest(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			require.NotNil(t, outbound, "upstream was not called")
			assert.NotContains(t, outbound, "background")
			if extraRaw, exists := outbound["extra_body"]; exists {
				var extra map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(extraRaw, &extra))
				assert.NotContains(t, extra, "background")
				assert.JSONEq(t, `"yes"`, string(extra["keep"]))
			}
		})
	}
}
