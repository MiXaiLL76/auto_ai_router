package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestGetHopByHopHeaders(t *testing.T) {
	headers := GetHopByHopHeaders()

	// Should contain all 8 RFC 7230 hop-by-hop headers
	expectedHeaders := []string{
		"Connection",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"TE",
		"Trailer",
		"Transfer-Encoding",
		"Upgrade",
	}

	assert.Len(t, headers, len(expectedHeaders))
	for _, h := range expectedHeaders {
		assert.True(t, headers[h], "should contain %s", h)
	}

	// Verify it returns a copy (modifying it doesn't affect the original)
	headers["X-Custom"] = true
	original := GetHopByHopHeaders()
	_, hasCustom := original["X-Custom"]
	assert.False(t, hasCustom, "modifying returned map should not affect the original")
}

func TestCopyResponseHeaders_PassthroughByDefault(t *testing.T) {
	src := http.Header{
		"Content-Type":       {"application/json"},
		"X-Provider-Request": {"provider-request-id"},
		"Content-Length":     {"100"},
		"Content-Encoding":   {"gzip"},
		"Connection":         {"close"},
		"X-Credential-Name":  {"internal"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().Build().copyResponseHeaders(w, src, false)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Equal(t, "provider-request-id", w.Header().Get("X-Provider-Request"))
	assert.Empty(t, w.Header().Get("Content-Length"))
	assert.Empty(t, w.Header().Get("Content-Encoding"))
	assert.Empty(t, w.Header().Get("Connection"))
	assert.Empty(t, w.Header().Get("X-Credential-Name"))
}

func TestCopyResponseHeaders_Allowlist(t *testing.T) {
	src := http.Header{
		"Cache-Control":         {"no-cache"},
		"Content-Disposition":   {"attachment"},
		"Content-Range":         {"bytes 0-9/10"},
		"Content-Type":          {"application/json"},
		"ETag":                  {`"revision"`},
		"Last-Modified":         {"Sun, 19 Jul 2026 10:00:00 GMT"},
		"Location":              {"/v1/files/result"},
		"Retry-After":           {"30", "60"},
		"Server":                {"provider"},
		"Set-Cookie":            {"session=secret"},
		"X-Amzn-Requestid":      {"amazon-request-id"},
		"X-Litellm-Model-Id":    {"model-id"},
		"Llm_provider-Api-Base": {"https://provider.example"},
		"X-Future-Provider":     {"future-value"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		copyResponseHeaders(w, src, false)

	assert.Equal(t, "no-cache", w.Header().Get("Cache-Control"))
	assert.Equal(t, "attachment", w.Header().Get("Content-Disposition"))
	assert.Equal(t, "bytes 0-9/10", w.Header().Get("Content-Range"))
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Equal(t, "Sun, 19 Jul 2026 10:00:00 GMT", w.Header().Get("Last-Modified"))
	assert.Equal(t, "/v1/files/result", w.Header().Get("Location"))
	assert.Equal(t, []string{"30", "60"}, w.Header().Values("Retry-After"))
	assert.Empty(t, w.Header().Get("Server"))
	assert.Empty(t, w.Header().Get("Set-Cookie"))
	assert.Empty(t, w.Header().Get("X-Amzn-Requestid"))
	assert.Empty(t, w.Header().Get("X-Litellm-Model-Id"))
	assert.Empty(t, w.Header().Get("Llm_provider-Api-Base"))
	assert.Empty(t, w.Header().Get("X-Future-Provider"))
}

func TestCopyResponseHeaders_RemovesValidators(t *testing.T) {
	src := http.Header{
		"Content-Type":  {"application/json"},
		"ETag":          {`"revision"`},
		"Content-Range": {"bytes 0-9/10"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		copyResponseHeaders(w, src, false)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Equal(t, "bytes 0-9/10", w.Header().Get("Content-Range"))
}

func TestSetCredentialResponseHeader_Allowlist(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set("X-Credential-Name", "stale")
	logCtx := &RequestLogContext{IsProxyRequest: true, ActualCredentialName: "provider"}

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		setCredentialResponseHeader(w, logCtx, "")

	assert.Empty(t, w.Header().Get("X-Credential-Name"))
}

func TestSetCredentialResponseHeader_Passthrough(t *testing.T) {
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{IsProxyRequest: true, ActualCredentialName: "provider"}

	NewTestProxyBuilder().Build().setCredentialResponseHeader(w, logCtx, "")

	assert.Equal(t, "provider", w.Header().Get("X-Credential-Name"))
}
