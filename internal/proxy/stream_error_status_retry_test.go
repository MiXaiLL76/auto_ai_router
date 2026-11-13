package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider may answer an HTTP error under an event-stream Content-Type --
// Google's SSE endpoints (Vertex's streamGenerateContent?alt=sse) return the
// error JSON that way for early 4xx/5xx. Before this, such a response was
// classified as streaming and hit the "Cannot retry streaming responses" break,
// so a force_stream_for_ttft request (whose upstream is exactly that SSE
// endpoint) returned the provider's first 400 straight to the client with no
// retry on the next credential.
func TestHTTPErrorWithEventStreamContentTypeRetriesOnNextCredential(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := sseStatusServer(t, &primaryHits, http.StatusBadRequest,
		`{"error":{"code":400,"message":"Invalid request","status":"INVALID_ARGUMENT"}}`)
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits, streamOKChunks)
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load(), "the first credential must be tried")
	assert.Equal(t, int32(1), secondaryHits.Load(), "a 4xx under event-stream must retry on the next credential")
	assert.Contains(t, w.Body.String(), "hello")
}

// The retried-away error must not leak to the client, and the classification
// must still read the provider's body (the point of buffering it).
func TestHTTPErrorWithEventStreamContentTypeIsClassifiedNotRelayed(t *testing.T) {
	var hits atomic.Int32
	upstream := sseStatusServer(t, &hits, http.StatusBadRequest,
		`{"error":{"code":400,"message":"Invalid request","status":"INVALID_ARGUMENT"}}`)
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "only", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "only-key",
		RPM: 100, TPM: 100000,
	}).WithMaxProviderRetries(3).Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	assert.Equal(t, int32(1), hits.Load())
	assert.Contains(t, w.Header().Get("Content-Type"), "application/json",
		"an error must reach the client as JSON, never as the provider's event-stream framing")
}

// A 4xx the retry policy treats as credential-independent (a bad-request marker)
// is not retried, but is still relayed as a classified JSON error rather than
// as a raw event stream.
func TestHTTPErrorWithEventStreamContentTypeNonRetryableStaysError(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := sseStatusServer(t, &primaryHits, http.StatusBadRequest,
		`{"error":{"message":"Penalty is not enabled for this model"}}`)
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits, streamOKChunks)
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(0), secondaryHits.Load(), "a request-fault 400 is not retried (it fails identically everywhere)")
}

// A healthy streaming response (2xx under event-stream) must still be treated
// as a stream and forwarded, unchanged by this fix.
func TestSuccessfulEventStreamStillStreams(t *testing.T) {
	var hits atomic.Int32
	upstream := sseServer(t, &hits, streamOKChunks)
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "only", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "only-key",
		RPM: 100, TPM: 100000,
	}).Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Header().Get("Content-Type"), "text/event-stream")
	assert.Contains(t, w.Body.String(), "hello")
}

// sseStatusServer answers with an event-stream Content-Type but a non-2xx
// status, the shape Google's SSE endpoints use for early errors.
func sseStatusServer(t *testing.T, hits *atomic.Int32, status int, body string) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}
