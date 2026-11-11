package proxy

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/require"
)

func TestProxyURLLogsRedactQueryOnSuccessAndError(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("api_key"); got != "query-secret" {
			t.Errorf("upstream lost original api_key: %q", got)
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	cred := &config.CredentialConfig{Name: "relay", Type: config.ProviderTypeProxy, BaseURL: upstream.URL}
	prx := NewTestProxyBuilder().Build()
	var logs bytes.Buffer
	prx.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions?api_key=query-secret&model=gpt", strings.NewReader(`{"model":"gpt"}`))
	logCtx := testLogCtx(t)
	response, err := prx.executeProxyRequest(request, cred, "gpt", []byte(`{"model":"gpt"}`), time.Now(), logCtx)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, response.StatusCode)
	require.Contains(t, logs.String(), "Proxy request forwarded")
	require.NotContains(t, logs.String(), "query-secret")
	require.Contains(t, logs.String(), "REDACTED")

	upstream.Close()
	logs.Reset()
	_, err = prx.executeProxyRequest(request, cred, "gpt", []byte(`{"model":"gpt"}`), time.Now(), logCtx)
	require.Error(t, err)
	require.Contains(t, logs.String(), "Failed to proxy request")
	require.NotContains(t, logs.String(), "query-secret")
}
