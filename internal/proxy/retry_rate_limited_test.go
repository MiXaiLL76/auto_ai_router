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
