package proxy

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	streamStartRateLimitEvent = "data: {\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Request failed\"},\"output\":[]},\"sequence_number\":0,\"type\":\"response.failed\"}\n\n"
	streamOKChunks            = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
	streamChatRequestBody = `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`
)

func sseServer(t *testing.T, hits *atomic.Int32, body string) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

func twoCredentialProxy(primaryURL, secondaryURL string) *Proxy {
	return NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: primaryURL, APIKey: "primary-key",
			RPM: 100, TPM: 100000,
		},
		config.CredentialConfig{
			Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: secondaryURL, APIKey: "secondary-key",
			Priority: 1, RPM: 100, TPM: 100000,
		},
	).WithMaxProviderRetries(3).Build()
}

func TestStreamStartingWithRateLimitEventRetriesOnNextCredential(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := sseServer(t, &primaryHits, streamStartRateLimitEvent)
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits, streamOKChunks)
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(1), secondaryHits.Load())
	assert.Contains(t, w.Body.String(), "hello")
	assert.NotContains(t, w.Body.String(), "rate_limit_exceeded")
}

func TestResponsesStreamStartingWithRateLimitEventRetriesOnNextCredential(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := sseServer(t, &primaryHits, streamStartRateLimitEvent)
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits,
		"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_2\",\"status\":\"in_progress\",\"output\":[]}}\n\n"+
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_2\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/responses", `{"model":"gpt-4","stream":true,"input":"hi"}`))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(1), secondaryHits.Load())
	assert.Contains(t, w.Body.String(), "response.completed")
	assert.NotContains(t, w.Body.String(), "rate_limit_exceeded")
}

func TestStreamStartingWithRateLimitEventOnLastCredentialReturns429(t *testing.T) {
	var hits atomic.Int32
	upstream := sseServer(t, &hits, streamStartRateLimitEvent)
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(config.CredentialConfig{
		Name: "only", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "only-key",
		RPM: 100, TPM: 100000,
	}).WithMaxProviderRetries(3).Build()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	require.Equal(t, http.StatusTooManyRequests, w.Code, w.Body.String())
	assert.Equal(t, int32(1), hits.Load())
	assert.Contains(t, w.Header().Get("Content-Type"), "application/json")
	assert.NotEmpty(t, w.Header().Get("Retry-After"))
}

func TestHealthyStreamIsForwardedUnchangedAfterPeek(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	primary := sseServer(t, &primaryHits, streamOKChunks)
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits, streamOKChunks)
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(0), secondaryHits.Load())
	assert.Contains(t, w.Body.String(), "hello")
	assert.Contains(t, w.Body.String(), "[DONE]")
}

func TestStreamErrorAfterFirstEventIsNotRetried(t *testing.T) {
	var primaryHits, secondaryHits atomic.Int32
	firstChunk := strings.SplitAfter(streamOKChunks, "\n\n")[0]
	// The first frame must reach the router as its own read so it is forwarded to
	// the client before the error arrives.
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, firstChunk)
		w.(http.Flusher).Flush()
		time.Sleep(100 * time.Millisecond)
		_, _ = io.WriteString(w, "data: {\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"later\"}}\n\n")
	}))
	defer primary.Close()
	secondary := sseServer(t, &secondaryHits, streamOKChunks)
	defer secondary.Close()

	prx := twoCredentialProxy(primary.URL, secondary.URL)
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, newRequestHeadersClientRequest("/v1/chat/completions", streamChatRequestBody))

	assert.Equal(t, int32(1), primaryHits.Load())
	assert.Equal(t, int32(0), secondaryHits.Load(), "an error after content was forwarded must not trigger a retry")
	assert.Contains(t, w.Body.String(), "hello")
}

type chunkedReadCloser struct {
	chunks [][]byte
	err    error
	closed bool
}

func (c *chunkedReadCloser) Read(p []byte) (int, error) {
	if len(c.chunks) == 0 {
		return 0, c.err
	}
	n := copy(p, c.chunks[0])
	c.chunks[0] = c.chunks[0][n:]
	if len(c.chunks[0]) == 0 {
		c.chunks = c.chunks[1:]
	}
	return n, nil
}

func (c *chunkedReadCloser) Close() error {
	c.closed = true
	return nil
}

func TestPeekStreamStartErrorReplaysBytesAndReadError(t *testing.T) {
	boom := errors.New("upstream reset")
	frame := "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"
	body := &chunkedReadCloser{chunks: [][]byte{[]byte(frame[:10]), []byte(frame[10:]), []byte("data: tail\n\n")}, err: boom}
	resp := &http.Response{Body: body}

	require.Empty(t, peekStreamStartError(resp))

	got, err := io.ReadAll(resp.Body)
	assert.ErrorIs(t, err, boom)
	assert.Equal(t, frame+"data: tail\n\n", string(got))
	require.NoError(t, resp.Body.Close())
	assert.True(t, body.closed)
}

func TestPeekStreamStartErrorDetectsFragmentedErrorFrame(t *testing.T) {
	body := &chunkedReadCloser{chunks: [][]byte{[]byte(streamStartRateLimitEvent[:40]), []byte(streamStartRateLimitEvent[40:])}, err: io.EOF}
	payload := peekStreamStartError(&http.Response{Body: body})
	assert.Contains(t, payload, "rate_limit_exceeded")
	assert.Equal(t, http.StatusTooManyRequests, statusCodeFromProviderStreamError(payload))
}
