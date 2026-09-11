package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_AnthropicDoesNotForwardClientOrigin(t *testing.T) {
	for _, path := range []string{"/v1/messages", "/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			captured := make(chan http.Header, 1)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				captured <- r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				if r.Header.Get("Origin") != "" {
					w.WriteHeader(http.StatusUnauthorized)
					_, _ = w.Write([]byte(`{"error":{"type":"authentication_error","message":"CORS requests must set 'anthropic-dangerous-direct-browser-access' header"}}`))
					return
				}
				_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
			}))
			defer upstream.Close()

			proxy := NewTestProxyBuilder().
				WithSingleCredential("anthropic", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
				Build()
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hello"}],"max_tokens":16}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Origin", "https://client.example.invalid")
			req.Header.Set("anthropic-beta", "test-beta")
			w := httptest.NewRecorder()

			proxy.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			outbound := <-captured
			assert.Empty(t, outbound.Get("Origin"))
			assert.Empty(t, outbound.Get("anthropic-dangerous-direct-browser-access"))
			assert.Equal(t, "upstream-key", outbound.Get("X-Api-Key"))
			assert.Equal(t, "test-beta", outbound.Get("anthropic-beta"))
			assert.Equal(t, "https://client.example.invalid", req.Header.Get("Origin"))
		})
	}
}
