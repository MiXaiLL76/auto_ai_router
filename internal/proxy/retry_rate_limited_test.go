package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_CascadeAllCredentialsRateLimited_Returns429 reproduces the
// production incident behind this fix: a model served by a single credential
// whose RPM limit is exhausted.
//
// The chain, as it happened in production (glm-4.6v, one credential, rpm 10):
//
//  1. The credential is selected -- which records its RPM slot.
//  2. The upstream fails at the transport level, so no response is obtained.
//  3. The cascade looks for another credential; there is none, and the only
//     candidate is now over its limit.
//
// The balancer distinguishes "no candidate exists" (ErrNoCredentialsAvailable)
// from "every candidate is rate limited" (ErrRateLimitExceeded), but
// nextRetryCredential dropped that distinction, so the request ended as a
// generic 502 "All provider attempts failed". A 429 with a Retry-After was
// expected: without one the client has no way to pace its retries, which is
// how a rate-limited model turns into a request storm.
func TestProxyRequest_CascadeAllCredentialsRateLimited_Returns429(t *testing.T) {
	var upstreamHits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits.Add(1)
		// Transport-level failure: no response at all, so the router has
		// nothing to relay and must look for another credential.
		hijackAndClose(w)
	}))
	defer upstream.Close()

	// One credential for the model, with an RPM of 1: selecting it consumes the
	// only slot, so the cascade that follows finds it saturated.
	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "limited-cred", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 1, TPM: 100000,
		},
	).WithMaxProviderRetries(2).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Positive(t, upstreamHits.Load(), "the upstream must have been tried")
	assert.Equal(t, http.StatusTooManyRequests, rec.Code,
		"a model whose every credential is rate limited must answer 429, not 502: body: %s", rec.Body.String())
	assert.NotEmpty(t, rec.Header().Get("Retry-After"),
		"a 429 must tell the client when to come back")
	assert.NotContains(t, rec.Body.String(), "All provider attempts failed")
}

// hijackAndClose drops the connection without a response -- the transport
// failure the cascade treats as "this attempt produced nothing".
func hijackAndClose(w http.ResponseWriter) {
	if hj, ok := w.(http.Hijacker); ok {
		if conn, _, err := hj.Hijack(); err == nil {
			_ = conn.Close()
		}
	}
}

// TestProxyRequest_ForceStream_CascadeAllCredentialsRateLimited_Returns429 is
// the force_stream_for_ttft variant of the same scenario: the client asked for
// a non-streaming response, so the router streams upstream and reassembles --
// and the same cascade runs out of rate-limited credentials. The simulated
// path must answer 429 too, not the 502 it produced before.
func TestProxyRequest_ForceStream_CascadeAllCredentialsRateLimited_Returns429(t *testing.T) {
	var hits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		// Transport-level failure: no response at all.
		hijackAndClose(w)
	}))
	defer upstream.Close()

	// One credential, RPM 1: selecting it consumes the only slot.
	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "limited-cred", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 1, TPM: 100000,
		},
	).WithForceStreamForTTFT(true).WithMaxProviderRetries(2).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusTooManyRequests, rec.Code,
		"the simulated path must answer 429 for a rate-limited cascade: body: %s", rec.Body.String())
	assert.NotEmpty(t, rec.Header().Get("Retry-After"))
}

// TestProxyRequest_StructurallyExcludedCredentialRateLimited_Stays502 is the
// counterweight to the 429 fix: a credential that was never dispatched to must
// not turn a genuine transport failure into a false "all credentials rate
// limited".
//
// The exclusion set the balancer scans is not only "credentials this chain
// tried" -- it also carries credentials that were ruled out structurally
// before the first attempt (a reasoning_only credential on a request without
// reasoning, an organization-policy denylist entry, a system_one split). One of
// those being over its rate limit because of unrelated traffic says nothing
// about why this request failed. Scanning them would report 429 for a request
// that actually failed on the transport -- exactly the confusion the new
// error origin was introduced to prevent.
func TestProxyRequest_StructurallyExcludedCredentialRateLimited_Stays502(t *testing.T) {
	var realHits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		realHits.Add(1)
		// Transport-level failure on the only credential that was actually used.
		hijackAndClose(w)
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "reasoning-cred", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k1", RPM: 1, TPM: 100000, ReasoningOnly: true,
		},
		config.CredentialConfig{
			Name: "real-cred", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k2", RPM: 5, TPM: 100000,
		},
	).WithMaxProviderRetries(2).Build()

	// Exhaust the reasoning-only credential's only RPM slot before the request.
	prx.rateLimiter.TryAllowAll("reasoning-cred", "m")

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, int32(1), realHits.Load(), "only the real credential is a candidate")
	assert.Equal(t, http.StatusBadGateway, rec.Code,
		"a structurally excluded credential's rate limit must not mask a transport failure: body: %s", rec.Body.String())
}

// TestProxyRequest_ProviderRateLimitedThenTransportFailure_Returns429 reproduces
// the production incident behind this fix: a model served by several Azure
// credentials, where the upstream accounts are over their own token rate
// limit.
//
// The chain, as it happened in production (gpt-5-mini, seven credentials):
//
//  1. Three credentials answer 429 -- the upstream account is over its token
//     rate limit, which the router retries on the next credential.
//  2. The remaining four fail on the transport (the egress answered CONNECT
//     with 503), so no response is left to relay.
//
// The outcome was a 502 "All provider attempts failed", which blames the
// provider for being broken. It was really a rate limit: the model answers
// normally once the upstream's window rolls over, and the client needs a
// Retry-After to pace its retries -- without one it hammers the pool.
func TestProxyRequest_ProviderRateLimitedThenTransportFailure_Returns429(t *testing.T) {
	var hits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) <= 3 {
			// The upstream account is over its own rate limit.
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","message":"exceeded token rate limit"}}`))
			return
		}
		// Transport-level failure: no response at all.
		hijackAndClose(w)
	}))
	defer upstream.Close()

	// Enough credentials for the cascade to run through the rate-limited ones
	// and reach the transport failures.
	creds := make([]config.CredentialConfig, 0, 7)
	for i := range 7 {
		creds = append(creds, config.CredentialConfig{
			Name: "azure-cred-" + string(rune('a'+i)), Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 100, TPM: 100000,
		})
	}
	prx := NewTestProxyBuilder().WithCredentials(creds...).WithMaxProviderRetries(6).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusTooManyRequests, rec.Code,
		"a model whose upstream accounts are rate limited must answer 429, not 502: body: %s", rec.Body.String())
	assert.NotEmpty(t, rec.Header().Get("Retry-After"),
		"a 429 must tell the client when to come back")
	assert.NotContains(t, rec.Body.String(), "All provider attempts failed")
}

// TestProxyRequest_TransportFailureOnly_Stays502 is the counterweight to the
// 429 fix: a request whose credentials all failed on the transport, with no
// provider 429 anywhere, is a genuine outage and must keep its 502. Without
// this the fix would report every exhausted cascade as a rate limit.
func TestProxyRequest_TransportFailureOnly_Stays502(t *testing.T) {
	var hits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		hijackAndClose(w)
	}))
	defer upstream.Close()

	creds := make([]config.CredentialConfig, 0, 3)
	for i := range 3 {
		creds = append(creds, config.CredentialConfig{
			Name: "azure-cred-" + string(rune('a'+i)), Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 100, TPM: 100000,
		})
	}
	prx := NewTestProxyBuilder().WithCredentials(creds...).WithMaxProviderRetries(3).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusBadGateway, rec.Code,
		"a pure transport failure is an outage and must stay a 502: body: %s", rec.Body.String())
	assert.NotContains(t, rec.Body.String(), "Rate limit exceeded",
		"a pure transport failure must not be reported as a rate limit")
}
