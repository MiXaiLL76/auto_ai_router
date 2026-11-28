package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCredentialResponseHeadersOnWire(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		status                    int
		stream, compat, allowlist bool
	}{
		{"json", 200, false, false, false},
		{"sse", 200, true, false, true},
		{"compat-json", 200, false, true, true},
		{"compat-sse", 200, true, true, false},
		{"provider-error", 429, false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Empty(t, r.Header.Get("X-Pool"))
				w.Header().Add("X-Pool", "upstream")
				w.Header().Add("X-Pool", "second")
				if tc.stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				if tc.status >= 400 {
					_, _ = io.WriteString(w, `{"error":{"message":"rate limit exceeded"}}`)
					return
				}
				_, _ = io.WriteString(w, requestHeadersCases[0].response)
			}))
			defer upstream.Close()
			prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{Name: "internal-name", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "key", RPM: -1, TPM: -1, ResponseHeaders: map[string]string{"x-pool": "public"}}).Build()
			if tc.compat {
				prx.responseCompat = compatlitellm.New()
			}
			if tc.allowlist {
				prx.responseHeaderMode = config.ResponseHeaderModeAllowlist
			}
			body := requestHeadersCases[0].body
			if tc.stream {
				body = `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`
			}
			w := httptest.NewRecorder()
			req := newRequestHeadersClientRequest("/v1/chat/completions", body)
			prx.ProxyRequest(w, req)
			require.Equal(t, tc.status, w.Code, w.Body.String())
			require.Equal(t, []string{"public"}, w.Result().Header.Values("X-Pool"))
			if tc.stream {
				require.Contains(t, w.Result().Header.Get("Content-Type"), "text/event-stream")
			}
			w = httptest.NewRecorder()
			req = newRequestHeadersClientRequest("/v1/chat/completions", body)
			req.Header.Del("Authorization")
			prx.ProxyRequest(w, req)
			require.GreaterOrEqual(t, w.Code, 400)
			require.Empty(t, w.Result().Header.Values("X-Pool"))
		})
	}
}
