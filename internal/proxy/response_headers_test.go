package proxy

import (
	"compress/gzip"
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

func TestProxyRequest_ResponseHeaderAllowlist(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		credType   config.ProviderType
	}{
		{name: "direct success", statusCode: http.StatusOK, credType: config.ProviderTypeOpenAI},
		{name: "relay success", statusCode: http.StatusOK, credType: config.ProviderTypeProxy},
		{name: "relay error", statusCode: http.StatusTooManyRequests, credType: config.ProviderTypeProxy},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				setProviderResponseHeaders(w.Header())
				w.WriteHeader(tt.statusCode)
				if tt.statusCode == http.StatusOK {
					_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("response-id", "gpt-4", "ok"))
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
			}))
			defer upstream.Close()

			prx := NewTestProxyBuilder().
				WithSingleCredential("provider", tt.credType, upstream.URL, "upstream-key").
				WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
				Build()
			req := newHeaderPolicyRequest(false)
			w := httptest.NewRecorder()

			prx.ProxyRequest(w, req)

			assert.Equal(t, tt.statusCode, w.Code)
			assertPublicProviderHeaders(t, w.Header())
		})
	}
}

func TestProxyRequest_StreamingResponseHeaderAllowlist(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setProviderResponseHeaders(w.Header())
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeProxy, upstream.URL, "upstream-key").
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build()
	req := newHeaderPolicyRequest(true)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Contains(t, w.Body.String(), "data:")
	assertPublicProviderHeaders(t, w.Header())
}

func TestProxyRequest_ResponseHeaderAllowlistWithCompression(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setProviderResponseHeaders(w.Header())
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("compressed-id", "gpt-4", strings.Repeat("content", 100)))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeProxy, upstream.URL, "upstream-key").
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build()
	req := newHeaderPolicyRequest(false)
	req.Header.Set("Accept-Encoding", "gzip")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "gzip", w.Header().Get("Content-Encoding"))
	assertPublicProviderHeaders(t, w.Header())
	reader, err := gzip.NewReader(w.Body)
	require.NoError(t, err)
	decompressed, err := io.ReadAll(reader)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	assert.Contains(t, string(decompressed), "compressed-id")
}

func TestProxyRequest_FallbackResponseHeaderAllowlist(t *testing.T) {
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "rate_limit"})
	}))
	defer primary.Close()

	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setProviderResponseHeaders(w.Header())
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("fallback-id", "gpt-4", "ok"))
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback(primary.URL, fallback.URL).
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build()
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, newHeaderPolicyRequest(false))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "fallback-id")
	assertPublicProviderHeaders(t, w.Header())
}

func newHeaderPolicyRequest(stream bool) *http.Request {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}`
	if stream {
		body = `{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"stream":true}`
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

func setProviderResponseHeaders(header http.Header) {
	header.Set("Content-Type", "application/json")
	header.Set("Cache-Control", "no-cache")
	header.Set("Retry-After", "30")
	header.Set("Server", "provider")
	header.Set("Set-Cookie", "session=secret")
	header.Set("X-Amzn-Requestid", "amazon-request-id")
	header.Set("X-Litellm-Model-Id", "model-id")
	header.Set("Llm_provider-Api-Base", "https://provider.example")
	header.Set("X-Future-Provider", "future-value")
}

func assertPublicProviderHeaders(t *testing.T, header http.Header) {
	t.Helper()
	require.Equal(t, "no-cache", header.Get("Cache-Control"))
	require.Equal(t, "30", header.Get("Retry-After"))
	assert.Empty(t, header.Get("Server"))
	assert.Empty(t, header.Get("Set-Cookie"))
	assert.Empty(t, header.Get("X-Amzn-Requestid"))
	assert.Empty(t, header.Get("X-Litellm-Model-Id"))
	assert.Empty(t, header.Get("Llm_provider-Api-Base"))
	assert.Empty(t, header.Get("X-Future-Provider"))
	assert.Empty(t, header.Get("X-Credential-Name"))
}
