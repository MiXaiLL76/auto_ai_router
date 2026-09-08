package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequestUsesCredentialProxy(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeOpenAI, config.ProviderTypeProxy, config.ProviderTypeAIR} {
		t.Run(string(provider), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "http://upstream.invalid/v1/chat/completions", r.URL.String())
				assert.Equal(t, "Bearer provider-key", r.Header.Get("Authorization"))
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"proxied","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
			}))
			defer server.Close()
			prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
				Name: "test", Type: provider, BaseURL: "http://upstream.invalid", APIKey: "provider-key", ProxyURL: server.URL, RPM: -1, TPM: -1,
			}).Build()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			prx.ProxyRequest(recorder, req)
			require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
			require.Contains(t, recorder.Body.String(), "proxied")
		})
	}
}
