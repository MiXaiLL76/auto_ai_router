package proxy

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

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

// netTimeoutError is Go's httpError shape: a plain error that also implements
// net.Error with Timeout() == true. The Transport wraps it in a *url.Error
// before handing it back to the caller.
type netTimeoutError struct {
	msg string
}

func (e *netTimeoutError) Error() string   { return e.msg }
func (e *netTimeoutError) Timeout() bool   { return true }
func (e *netTimeoutError) Temporary() bool { return true }

// headerTimeoutError reproduces what the Transport hands back when
// ResponseHeaderTimeout fires: the request was written, the provider never
// answered with headers, and the transport gave up waiting for them.
func headerTimeoutError() error {
	return &url.Error{
		Op:  "Post",
		URL: "https://api.example.com/v1/chat/completions",
		Err: &netTimeoutError{msg: "net/http: timeout awaiting response headers"},
	}
}

// http2LostError reproduces Go's HTTP/2 idle-ping teardown: a pooled
// connection went silent and was closed. Deliberately also a net.Error with
// Timeout() == true, so the check must key on the error being a response-header
// timeout specifically -- a broad "any net.Error timeout" check would ban here.
func http2LostError() error {
	return &url.Error{
		Op:  "Post",
		URL: "https://api.example.com/v1/chat/completions",
		Err: &netTimeoutError{msg: "http2: client connection lost"},
	}
}

// streamingChatRequest is a streaming chat request against the given model.
func streamingChatRequest(model string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"`+model+`","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

// TestAttemptTimeout_HeaderTimeout_Bans verifies the core claim this feature
// makes: an attempt that never got a response at all (the error Go's own
// ResponseHeaderTimeout produces) on a real streaming request bans the
// credential for this model, and the client sees 408 once no credential is
// left to retry.
func TestAttemptTimeout_HeaderTimeout_Bans(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("dead-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		WithResponseHeaderTimeout(4 * time.Second).
		Build()
	prx.client.Transport = erroringTransport{err: headerTimeoutError()}
	prx.streamingClientForTest(nil).Transport = erroringTransport{err: headerTimeoutError()}

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))

	assert.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.True(t, prx.balancer.IsBanned("dead-cred", "gpt-4"),
		"a response-header-timeout on a streaming attempt must ban the credential for this model")
}

// TestAttemptTimeout_FeatureDisabled_NoBan is the regression test for the
// review's finding that the ban fired even with the feature switched off:
// with response_header_timeout unset the transport falls back to
// request_timeout, so a timeout means "the provider needed longer than that",
// not "the provider never answered" — banning on it is a false positive that
// would hit a healthy but slow credential.
func TestAttemptTimeout_FeatureDisabled_NoBan(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("slow-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		WithRequestTimeout(30 * time.Second).
		Build()
	// No WithResponseHeaderTimeout: the feature is off, as it is by default.
	prx.client.Transport = erroringTransport{err: headerTimeoutError()}
	prx.streamingClientForTest(nil).Transport = erroringTransport{err: headerTimeoutError()}

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))

	assert.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.False(t, prx.balancer.IsBanned("slow-cred", "gpt-4"),
		"with response_header_timeout unset a timeout is the general request_timeout — a slow provider must not be banned")
}

// TestAttemptTimeout_HTTP2ConnectionLost_DoesNotBan is the regression test for
// the review's bug 2: Go's HTTP/2 idle-ping teardown ("http2: client
// connection lost") is classified as a timeout by isTimeoutError (so the
// request still comes back as 408, matching existing behavior), but it means
// a pooled connection went silent and got pinged out — not that the
// credential itself is unresponsive right now — and must not trigger the
// attempt-timeout ban.
func TestAttemptTimeout_HTTP2ConnectionLost_DoesNotBan(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("flaky-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		WithResponseHeaderTimeout(4 * time.Second).
		Build()
	prx.client.Transport = erroringTransport{err: http2LostError()}
	prx.streamingClientForTest(nil).Transport = erroringTransport{err: http2LostError()}

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))

	assert.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.False(t, prx.balancer.IsBanned("flaky-cred", "gpt-4"),
		"an HTTP/2 idle-ping teardown is a pool-hygiene event, not evidence the credential is dead -- must not ban")
}

// TestAttemptTimeout_DialTimeout_DoesNotBan guards the narrowness of
// isAttemptTimeoutError: a dial timeout also implements net.Error with
// Timeout() == true, but it says the connection was never established — not
// that the provider went silent — so it must not ban the credential.
func TestAttemptTimeout_DialTimeout_DoesNotBan(t *testing.T) {
	prx := NewTestProxyBuilder().
		WithSingleCredential("unreachable-cred", config.ProviderTypeOpenAI, "https://api.example.com", "test-key").
		WithResponseHeaderTimeout(4 * time.Second).
		Build()
	dialErr := &url.Error{
		Op:  "Post",
		URL: "https://api.example.com/v1/chat/completions",
		Err: &netTimeoutError{msg: "dial tcp 10.255.255.1:443: i/o timeout"},
	}
	prx.client.Transport = erroringTransport{err: dialErr}
	prx.streamingClientForTest(nil).Transport = erroringTransport{err: dialErr}

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))

	assert.False(t, prx.balancer.IsBanned("unreachable-cred", "gpt-4"),
		"a dial timeout means the connection was never established — it says nothing about the credential's health")
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
		WithResponseHeaderTimeout(4 * time.Second).
		Build()
	prx.client.Transport = erroringTransport{err: headerTimeoutError()}
	prx.streamingClientForTest(nil).Transport = erroringTransport{err: headerTimeoutError()}

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

// TestAttemptTimeout_RealSlowProvider_Bans is the end-to-end check that the
// feature is wired to the transport at all: a provider that accepts the
// connection but never sends headers must trip the configured budget on the
// streaming client and ban the credential. The mock-transport tests above
// cover the classification; this one covers the wiring.
func TestAttemptTimeout_RealSlowProvider_Bans(t *testing.T) {
	// A server that accepts the request and never answers it.
	silent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	}))
	defer silent.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("silent-cred", config.ProviderTypeOpenAI, silent.URL, "test-key").
		WithResponseHeaderTimeout(300 * time.Millisecond).
		WithRequestTimeout(30 * time.Second).
		Build()

	w := httptest.NewRecorder()
	start := time.Now()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))
	elapsed := time.Since(start)

	require.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.Less(t, elapsed, 5*time.Second,
		"the attempt must give up on the header budget quickly, not wait out request_timeout")
	assert.True(t, prx.balancer.IsBanned("silent-cred", "gpt-4"),
		"a provider that never sends headers must be banned for this model")
}

// TestAttemptClient_SharesClientWhenDisabled is the regression test for the
// 2026-11-06 gpt-6-luna incident: client and the per-timeout streaming
// clients were built as two (then: N) completely separate *http.Client (two
// separate Transports, two separate idle-connection pools), one handed only
// half of MaxIdleConns/MaxIdleConnsPerHost. force_stream_for_ttft routes
// nearly all chat traffic through the streaming path, so splitting its pool
// cut real idle-connection capacity for busy models fleet-wide the moment
// this feature merged -- entirely independent of whether
// ResponseHeaderTimeout ever actually fired anywhere. With no budget
// configured for a credential (its own override, and the server-level
// default, both unset) there is no reason for a second pool at all:
// attemptClient must hand back the literal same *http.Client as p.client,
// not a look-alike copy.
func TestAttemptClient_SharesClientWhenDisabled(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	client, enabled := prx.attemptClient(nil, true)
	assert.Same(t, prx.client, client,
		"with no response_header_timeout configured, attemptClient must return the exact same *http.Client as p.client -- a separate pool here silently cuts real idle-connection capacity for no benefit")
	assert.False(t, enabled, "the attempt-timeout ban must not be enabled when no budget is configured")
}

// TestAttemptClient_DistinctWhenEnabled is the flip side: once a budget is
// actually configured (the server-level default here), attemptClient must
// hand back a distinct client with its own ResponseHeaderTimeout -- Go only
// allows a different budget via a separate Transport, so it must NOT be
// collapsed to the same object as p.client.
func TestAttemptClient_DistinctWhenEnabled(t *testing.T) {
	prx := NewTestProxyBuilder().WithResponseHeaderTimeout(4 * time.Second).Build()
	client, enabled := prx.attemptClient(nil, true)
	assert.NotSame(t, prx.client, client,
		"with a response_header_timeout configured, attemptClient needs a distinct Transport to carry it")
	assert.True(t, enabled, "the attempt-timeout ban must be enabled once a budget is configured")
}

// TestAttemptClient_CredentialOverride_IndependentOfServerDefault is the
// regression test for the per-credential redesign's own main claim: a
// credential's own response_header_timeout takes effect even when the
// server-level default is off, which is exactly the state the fleet is in
// today after the 2026-11-06 revert -- everything stays unlimited except the
// specific credentials an operator opts in by hand.
func TestAttemptClient_CredentialOverride_IndependentOfServerDefault(t *testing.T) {
	prx := NewTestProxyBuilder().Build() // server default: off (0)
	cred := &config.CredentialConfig{Name: "opted-in", ResponseHeaderTimeout: 4 * time.Second}

	client, enabled := prx.attemptClient(cred, true)

	assert.NotSame(t, prx.client, client,
		"a credential's own override must get its own Transport even though the server default is off")
	assert.True(t, enabled, "a credential with its own override must be ban-eligible regardless of the server default")
}

// TestAttemptClient_CredentialOptOut_OverridesServerDefault is the flip side:
// a negative override must disable the budget for that one credential even
// while the server default is positive for everyone else -- the "A negative
// duration ... explicitly opts this credential out" promise on
// config.CredentialConfig.ResponseHeaderTimeout's own doc comment.
func TestAttemptClient_CredentialOptOut_OverridesServerDefault(t *testing.T) {
	prx := NewTestProxyBuilder().WithResponseHeaderTimeout(4 * time.Second).Build()
	cred := &config.CredentialConfig{Name: "opted-out", ResponseHeaderTimeout: -1 * time.Second}

	client, enabled := prx.attemptClient(cred, true)

	assert.Same(t, prx.client, client,
		"a negative override must opt this credential out even while the server default is positive")
	assert.False(t, enabled, "an opted-out credential must never be ban-eligible on this path")
}

// TestAttemptClient_DistinctCredentials_GetDistinctPools checks the other
// direction from TestAttemptClient_SharesClientWhenDisabled: two credentials
// with two different overrides must not collapse onto the same client (Go
// enforces one ResponseHeaderTimeout per Transport), but asking for the same
// credential's client twice must return the literal same instance both
// times (attemptClient's LoadOrStore cache), not build a fresh pool per call.
func TestAttemptClient_DistinctCredentials_GetDistinctPools(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	fast := &config.CredentialConfig{Name: "fast", ResponseHeaderTimeout: 2 * time.Second}
	slow := &config.CredentialConfig{Name: "slow", ResponseHeaderTimeout: 6 * time.Second}

	fastClient, fastEnabled := prx.attemptClient(fast, true)
	slowClient, slowEnabled := prx.attemptClient(slow, true)
	fastClientAgain, _ := prx.attemptClient(fast, true)

	assert.True(t, fastEnabled)
	assert.True(t, slowEnabled)
	assert.NotSame(t, fastClient, slowClient, "two different timeout values must never share a Transport/pool")
	assert.NotSame(t, prx.client, fastClient)
	assert.NotSame(t, prx.client, slowClient)
	assert.Same(t, fastClient, fastClientAgain,
		"the same timeout value must reuse its cached client, not build a new pool on every attempt")
}

// TestAttemptTimeout_PerCredentialOverride_OnlyOverriddenCredentialBans is
// the end-to-end version of the three unit tests above: in one real fleet
// with two credentials behind the same dead provider, a response-header
// timeout must ban only the credential that actually had a budget in
// effect -- the other one falls through to request_timeout and must come
// back healthy, exactly the mixed state "давай предусмотрим per credential
// вариант... у остальных оставим пока без лимита" asks for.
func TestAttemptTimeout_PerCredentialOverride_OnlyOverriddenCredentialBans(t *testing.T) {
	// A server that accepts the request and never answers it, for every attempt.
	silent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(10 * time.Second):
		}
	}))
	defer silent.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name: "fast-budget", Type: config.ProviderTypeOpenAI, BaseURL: silent.URL, APIKey: "test-key",
				RPM: 100, TPM: 10000, IsFallback: false,
				ResponseHeaderTimeout: 300 * time.Millisecond,
			},
			config.CredentialConfig{
				// No ResponseHeaderTimeout: inherits the server-level default,
				// which WithResponseHeaderTimeout is deliberately not called
				// here to configure, matching every credential's state today.
				Name: "plain-cred", Type: config.ProviderTypeOpenAI, BaseURL: silent.URL, APIKey: "test-key",
				RPM: 100, TPM: 10000, IsFallback: true,
			},
		).
		WithRequestTimeout(1 * time.Second).
		Build()

	w := httptest.NewRecorder()
	start := time.Now()
	prx.ProxyRequest(w, streamingChatRequest("gpt-4"))
	elapsed := time.Since(start)

	require.Equal(t, http.StatusRequestTimeout, w.Code, "body: %s", w.Body.String())
	assert.Less(t, elapsed, 3*time.Second,
		"fast-budget must give up on its own short budget, not wait out plain-cred's full request_timeout twice")
	assert.True(t, prx.balancer.IsBanned("fast-budget", "gpt-4"),
		"fast-budget has its own response_header_timeout -- a provider that never sends headers must ban it")
	assert.False(t, prx.balancer.IsBanned("plain-cred", "gpt-4"),
		"plain-cred has no override and the server default is off -- it must fall back to request_timeout (Client.Timeout, a different error entirely) and never be banned")
}
