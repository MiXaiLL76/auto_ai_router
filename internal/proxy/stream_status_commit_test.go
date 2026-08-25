package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// statusTrackingWriter records every WriteHeader call (unlike
// httptest.ResponseRecorder, whose Code field defaults to 200 whether
// WriteHeader was ever called or not — indistinguishable from an implicit
// 200, which is exactly the bug these tests guard against).
type statusTrackingWriter struct {
	header      http.Header
	body        strings.Builder
	statusCalls []int
}

func (w *statusTrackingWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *statusTrackingWriter) Write(p []byte) (int, error) {
	return w.body.Write(p)
}

func (w *statusTrackingWriter) WriteHeader(statusCode int) {
	w.statusCalls = append(w.statusCalls, statusCode)
}

func (w *statusTrackingWriter) Flush() {}

// TestStreamToClient_NonStandard2xxStatusIsCommitted is a regression test:
// streamToClient used to rely entirely on Go's implicit WriteHeader(200) on
// first Write() to commit the response, so a genuine non-200 2xx upstream
// status (e.g. 201/202/204) was silently downgraded to 200 for the client.
func TestStreamToClient_NonStandard2xxStatusIsCommitted(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := &statusTrackingWriter{}
	reader := strings.NewReader(`data: {"choices":[{"delta":{"content":"hi"}}]}` + "\n\n")

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusAccepted, nil, nil, nil)
	require.NoError(t, err)

	require.Len(t, w.statusCalls, 1, "streamToClient must explicitly commit the real upstream status exactly once")
	assert.Equal(t, http.StatusAccepted, w.statusCalls[0])
}

// TestStreamToClient_NonStandard2xxStatusIsCommittedOnImmediateEOF covers the
// other commit path: an upstream that returns a non-200 2xx and then an empty
// body (immediate EOF, nothing ever buffered past the initial commit gate).
func TestStreamToClient_NonStandard2xxStatusIsCommittedOnImmediateEOF(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := &statusTrackingWriter{}
	reader := strings.NewReader("")

	err := prx.streamToClient(context.Background(), w, reader, "cred1", "gpt-4o", "/v1/chat/completions", http.StatusNoContent, nil, nil, nil)
	require.NoError(t, err)

	require.Len(t, w.statusCalls, 1)
	assert.Equal(t, http.StatusNoContent, w.statusCalls[0])
}

// nonFlushingWriter deliberately does NOT implement http.Flusher, exercising
// writeProxyStreamingResponseWithTokens's non-flushing io.Copy fallback path.
type nonFlushingWriter struct {
	header      http.Header
	body        strings.Builder
	statusCalls []int
}

func (w *nonFlushingWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *nonFlushingWriter) Write(p []byte) (int, error) {
	return w.body.Write(p)
}

func (w *nonFlushingWriter) WriteHeader(statusCode int) {
	w.statusCalls = append(w.statusCalls, statusCode)
}

// TestWriteProxyStreamingResponseWithTokens_NonFlushingFallbackCommitsStatus
// is a regression test: the non-flushing io.Copy fallback branch used to
// never call WriteHeader at all, so Go implicitly sent 200 on the first
// Write() regardless of the real upstream/error status.
func TestWriteProxyStreamingResponseWithTokens_NonFlushingFallbackCommitsStatus(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	w := &nonFlushingWriter{}

	proxyResp := &ProxyResponse{
		StatusCode:  http.StatusTooManyRequests,
		Headers:     http.Header{"Content-Type": {"application/json"}},
		StreamBody:  io.NopCloser(strings.NewReader(`{"error":{"message":"rate limited"}}`)),
		IsStreaming: true,
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	credential := &config.CredentialConfig{Name: "cred1", Type: config.ProviderTypeOpenAI, BaseURL: "https://api.invalid"}
	logCtx := &RequestLogContext{Request: request, Credential: credential}

	_, err := prx.writeProxyStreamingResponseWithTokens(w, proxyResp, request, credential, "gpt-4o", "gpt-4o", logCtx)
	require.NoError(t, err)

	require.Len(t, w.statusCalls, 1, "non-flushing fallback must explicitly commit the real upstream status")
	assert.Equal(t, http.StatusTooManyRequests, w.statusCalls[0])
}
