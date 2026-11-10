package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	goccyjson "github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractSessionID_PromptCacheKeyBeforeClientID(t *testing.T) {
	var body map[string]goccyjson.RawMessage
	require.NoError(t, goccyjson.Unmarshal([]byte(`{"user":"u","safety_identifier":"s","prompt_cache_key":"conv-1"}`), &body))
	assert.Equal(t, "conv-1", extractSessionIDFromRawBody(body))
	assert.Equal(t, "s", extractClientIDFromRawBody(body))
}

func TestExtractSessionID_MetadataSessionIDPriority(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"metadata only", `{"metadata":{"session_id":"metadata-session"}}`, "metadata-session"},
		{"prompt cache key wins", `{"prompt_cache_key":"cache-session","metadata":{"session_id":"metadata-session"}}`, "cache-session"},
		{"top-level session wins", `{"session_id":"body-session","metadata":{"session_id":"metadata-session"}}`, "body-session"},
		{"empty metadata session", `{"metadata":{"session_id":""}}`, ""},
		{"non-string metadata session", `{"metadata":{"session_id":123}}`, ""},
		{"malformed metadata shape", `{"metadata":["metadata-session"]}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body map[string]goccyjson.RawMessage
			require.NoError(t, goccyjson.Unmarshal([]byte(tt.body), &body))
			assert.Equal(t, tt.want, extractSessionIDFromRawBody(body))
		})
	}
}

func TestProxyRequest_MetadataSessionIDBeatsHeaderAndUser(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(chatCompletionOKBody("id-1", "ok")))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(testProxyCredential("cred1", upstream.URL, false)).
		WithSessionSticky(5 * time.Minute).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"metadata":{"session_id":"metadata-session"},"user":"client-1"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Session-Id", "header-session")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	_, boundToMetadata := prx.sessionStore.Get("metadata-session", "gpt-4")
	_, boundToHeader := prx.sessionStore.Get("header-session", "gpt-4")
	_, boundToUser := prx.sessionStore.Get("client-1", "gpt-4")
	assert.True(t, boundToMetadata)
	assert.False(t, boundToHeader)
	assert.False(t, boundToUser)
}

func TestProxyRequest_SessionHeaderBeatsUser(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(chatCompletionOKBody("id-1", "ok")))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(testProxyCredential("cred1", upstream.URL, false)).
		WithSessionSticky(5 * time.Minute).
		Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"user":"client-1"}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Session-Id", "conv-1")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	_, boundToHeader := prx.sessionStore.Get("conv-1", "gpt-4")
	_, boundToUser := prx.sessionStore.Get("client-1", "gpt-4")
	assert.True(t, boundToHeader, "binding must use the session header")
	assert.False(t, boundToUser, "user must not win over the session header")
}
