package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// erroringTransport always fails a RoundTrip with the given error, never
// producing a response. Used to simulate the transport-level doErr path in
// sendToCredential (p.client.Do returning an error), as opposed to the
// HTTP-response-level mocks most other tests use.
type erroringTransport struct {
	err error
}

func (t erroringTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	return nil, t.err
}

// TestAttemptTimeout_HeaderTimeout_Bans verifies the core claim this feature
// makes: an attempt that never got a response at all (simulated here as a
// context.DeadlineExceeded transport error, the same classification Go's own
// ResponseHeaderTimeout produces) on a real streaming request bans the
// credential for this model, and the client sees 408 once no credential is
// left to retry.
func TestAttemptTimeout_HeaderTimeout_Bans(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("dead-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		Build()
	prx.client.Transport = erroringTransport{err: context.DeadlineExceeded}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.True(t, prx.balancer.IsBanned("dead-cred", "gpt-4"),
		"a response-header-timeout-shaped error on a streaming attempt must ban the credential for this model")
}

// TestAttemptTimeout_HTTP2ConnectionLost_DoesNotBan is the regression test for
// the review's bug 2: Go's HTTP/2 idle-ping teardown ("http2: client
// connection lost") is classified as a timeout by isTimeoutError (so the
// request still comes back as 408, matching existing behavior), but it means
// a pooled connection went silent and got pinged out -- not that the
// credential itself is unresponsive right now -- and must not trigger the
// attempt-timeout ban.
func TestAttemptTimeout_HTTP2ConnectionLost_DoesNotBan(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("flaky-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		Build()
	prx.client.Transport = erroringTransport{err: errors.New("http2: client connection lost")}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.False(t, prx.balancer.IsBanned("flaky-cred", "gpt-4"),
		"an HTTP/2 idle-ping teardown is a pool-hygiene event, not evidence the credential is dead -- must not ban")
}

// TestAttemptTimeout_NonStreaming_HeaderTimeout_DoesNotBan guards the
// useFastHeaderClient gate itself: a header-timeout-shaped error on a
// non-streaming attempt (image generation here) must not ban, because that
// attempt was never sent with the short ResponseHeaderTimeout in the first
// place -- a slow-but-healthy non-streaming provider must not look identical
// to a dead one.
func TestAttemptTimeout_NonStreaming_HeaderTimeout_DoesNotBan(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("image-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		Build()
	prx.client.Transport = erroringTransport{err: context.DeadlineExceeded}

	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations",
		strings.NewReader(`{"model":"gpt-image-1","prompt":"a cat","n":1}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.False(t, prx.balancer.IsBanned("image-cred", "gpt-image-1"),
		"image generation never gets the short header timeout, so a slow response must not ban the credential")
}
