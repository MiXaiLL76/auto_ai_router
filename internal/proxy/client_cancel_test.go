package proxy

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// clientCanceledTransport simulates what Go's real http.Transport does once
// the request's own context is already canceled by the time RoundTrip is
// called: it fails immediately with that context's error, without ever
// reaching the network. Counting invocations lets the tests below assert the
// retry loop didn't waste an attempt on a second/fallback credential once
// the client is already gone.
type clientCanceledTransport struct {
	calls *atomic.Int32
}

func (tr clientCanceledTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	tr.calls.Add(1)
	return nil, r.Context().Err()
}

func newCanceledClientRequest(t *testing.T) *http.Request {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // simulate the client having already disconnected before AIR talks to any provider
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

// TestProxyRequestClientCanceled_DirectProvider covers the direct-provider
// retry loop (proxy.go's `for attempt := 0; attempt <= p.maxProviderRetries`
// loop and its "resp == nil" tail): when the client has already disconnected
// before AIR ever reaches a provider, the request must be classified as
// StatusClientClosedRequest (499)/ErrorOriginClientCanceled, not retried
// against a second credential, and not recorded as a 502 against anyone.
func TestProxyRequestClientCanceled_DirectProvider(t *testing.T) {
	var calls atomic.Int32
	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "primary", Type: config.ProviderTypeOpenAI, BaseURL: "http://primary.invalid", APIKey: "k1", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "secondary", Type: config.ProviderTypeOpenAI, BaseURL: "http://secondary.invalid", APIKey: "k2", RPM: 100, TPM: 10000},
		).
		Build()
	prx.client.Transport = clientCanceledTransport{calls: &calls}

	req := newCanceledClientRequest(t)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, StatusClientClosedRequest, w.Code)
	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, "client_closed_request", response.Error.Type)
	assert.Equal(t, int32(1), calls.Load(), "must not retry a second credential once the client is already gone")
}

// TestProxyRequestClientCanceled_ProxyCredential covers the AIR-to-AIR
// proxy-credential path (executeProxyRequest / forwardToProxy and its
// "lastProxyErr != nil && proxyResp == nil" tail): same expectation, but for
// a credential of type "proxy", which also has a same-type retry loop plus a
// separate fallback-proxy attempt that must both be skipped.
func TestProxyRequestClientCanceled_ProxyCredential(t *testing.T) {
	var calls atomic.Int32
	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.invalid", "http://fallback.invalid").
		Build()
	prx.client.Transport = clientCanceledTransport{calls: &calls}

	req := newCanceledClientRequest(t)
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	assert.Equal(t, StatusClientClosedRequest, w.Code)
	var response APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	assert.Equal(t, "client_closed_request", response.Error.Type)
	assert.Equal(t, int32(1), calls.Load(), "must not attempt the fallback proxy once the client is already gone")
}

// TestIsClientCanceledTransportError_RequiresBothSignals is a focused unit
// test for the helper itself, pinning down the two false-positive cases the
// combined check exists to avoid (see its doc comment):
//   - a bare context.Canceled from a transport/mock that has nothing to do
//     with r's own context (e.g. client_error_messages_test.go's "transport
//     error" case) must NOT be classified as a client cancellation;
//   - a genuine, unrelated transport error (e.g. connection refused)
//     occurring after the client happens to have already disconnected for
//     an unrelated reason must NOT be masked as a client cancellation either
//     -- that would hide a real outage from fail2ban/error-rate accounting.
func TestIsClientCanceledTransportError_RequiresBothSignals(t *testing.T) {
	liveReq := httptest.NewRequest(http.MethodGet, "/", nil)
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	canceledReq := httptest.NewRequest(http.MethodGet, "/", nil).WithContext(canceledCtx)

	assert.False(t, isClientCanceledTransportError(liveReq, context.Canceled),
		"a live request context must never be classified as client-canceled, regardless of the error")
	assert.False(t, isClientCanceledTransportError(canceledReq, errConnRefusedForTest),
		"a genuine transport error must not be masked as client-canceled just because the client also disconnected")
	assert.True(t, isClientCanceledTransportError(canceledReq, context.Canceled),
		"a canceled request context whose attempt failed with context.Canceled is exactly the case this exists to catch")
}

var errConnRefusedForTest = &fakeConnRefusedError{}

// fakeConnRefusedError stands in for a genuine transport failure (e.g.
// net.OpError wrapping ECONNREFUSED) that is unambiguously not
// context.Canceled.
type fakeConnRefusedError struct{}

func (e *fakeConnRefusedError) Error() string { return "connection refused" }
