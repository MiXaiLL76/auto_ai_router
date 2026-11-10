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

// azureContentFilterBody is what Azure answers when its content filter
// refuses the request. Reproduced from production request
// cf33c6845d7f7f3e051a4c1e95a9e889 (2026-11-10, model gpt-5-mini, four
// Azure credentials): the refusal arrives as a 400 whose message names no
// content at all, and whose code is the provider's own "invalid_prompt".
//
// The wording matters: it is a refusal dressed as a provider-side failure, so
// a classifier that looks for content-policy markers finds none, and the
// request is replayed on every credential in the pool -- the client got a 502
// after 7 attempts and 95 seconds.
const azureContentFilterBody = `{"error": {"message": "The model was unable to complete inference due to an internal error."}}`

// TestProxyRequest_AzureContentFilterIs400Not502 pins the whole chain for
// the production incident: a content-policy refusal must reach the client as
// a 400 with the content-policy message, decided on the FIRST credential.
//
// Before the fix the refusal was classified as a generic error: it stayed
// retryable across the pool, every credential refused it the same way, and
// the client got a 502 "All provider attempts failed" after 95 seconds.
func TestProxyRequest_AzureContentFilterIs400Not502(t *testing.T) {
	var upstreamHits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(azureContentFilterBody))
	}))
	defer upstream.Close()

	// A second credential that WOULD answer: it must not be reached -- every
	// credential refuses the same content, so replaying only multiplies the
	// delay (measured in production: 4 attempts and 55s for one blocked
	// prompt).
	var secondHits atomic.Int32
	healthy := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer healthy.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "azure-filter", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
		config.CredentialConfig{
			Name: "azure-healthy", Type: config.ProviderTypeOpenAI, BaseURL: healthy.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(3).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code,
		"a content-policy refusal must be a 400, not a 502: body: %s", rec.Body.String())
	assert.Contains(t, rec.Body.String(), "content_policy_violation",
		"the client must be told the content was refused: %s", rec.Body.String())
	assert.NotContains(t, rec.Body.String(), "unable to complete inference",
		"the provider's own text must not reach the client")
	assert.EqualValues(t, 1, upstreamHits.Load(),
		"the refusal must be decided on the first credential")
	assert.EqualValues(t, 0, secondHits.Load(),
		"a refusal must not be replayed on the next credential")
}

// TestProxyRequest_AzureContentFilterIs400InSimulatedPath is the same check
// through the forced-stream path (force_stream_for_ttft), which is how a
// non-streaming client is served in production -- the router asks the
// provider for a stream and reassembles the answer.
func TestProxyRequest_AzureContentFilterIs400InSimulatedPath(t *testing.T) {
	var upstreamHits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(azureContentFilterBody))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "azure-filter", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(3).WithForceStreamForTTFT(true).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code,
		"a content-policy refusal must be a 400 in the forced-stream path too: body: %s", rec.Body.String())
	assert.Contains(t, rec.Body.String(), "content_policy_violation")
}

// TestProxyRequest_ProviderRejectionSurvivesLaterTransportFailure pins the
// other half of the production incident: the provider answered 400 (a
// request-side refusal), and the NEXT credential failed on the transport.
//
// The provider's 400 is the truth about why the request could not be served:
// every credential would refuse it the same way. It used to be lost -- the
// retry loop cleared the response when it moved on, so the request ended as
// a 502 "All provider attempts failed" that blamed the provider for being
// broken and told the client to retry something that cannot succeed.
//
// This is the same defect class as the provider rate limit (a 429 that
// survived a later transport failure -- see #356), generalized from 429 to
// every 4xx.
func TestProxyRequest_ProviderRejectionSurvivesLaterTransportFailure(t *testing.T) {
	// The first credential answers 400: the provider's own verdict on the
	// request.
	var firstHits atomic.Int32
	first := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		firstHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Unknown parameter: 'foo'."}}`))
	}))
	defer first.Close()

	// The second credential dies on the transport: no response at all.
	var secondHits atomic.Int32
	second := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondHits.Add(1)
		hijackAndClose(w)
	}))
	defer second.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "cred-a", Type: config.ProviderTypeOpenAI, BaseURL: first.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
		config.CredentialConfig{
			Name: "cred-b", Type: config.ProviderTypeOpenAI, BaseURL: second.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(3).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code,
		"the provider's own 400 must survive a later transport failure, not become a 502: body: %s",
		rec.Body.String())
	assert.NotContains(t, rec.Body.String(), "Bad Gateway",
		"a 502 would blame the provider for being broken")
	assert.Positive(t, firstHits.Load())
	assert.Positive(t, secondHits.Load())
}

// TestProxyRequest_ProviderRejectionSurvivesLaterTransportFailureInSimulatedPath
// is the same check through the forced-stream path.
func TestProxyRequest_ProviderRejectionSurvivesLaterTransportFailureInSimulatedPath(t *testing.T) {
	var firstHits atomic.Int32
	first := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		firstHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Unknown parameter: 'foo'."}}`))
	}))
	defer first.Close()

	var secondHits atomic.Int32
	second := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondHits.Add(1)
		hijackAndClose(w)
	}))
	defer second.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "cred-a", Type: config.ProviderTypeOpenAI, BaseURL: first.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
		config.CredentialConfig{
			Name: "cred-b", Type: config.ProviderTypeOpenAI, BaseURL: second.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(3).WithForceStreamForTTFT(true).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code,
		"the provider's own 400 must survive a later transport failure in the forced-stream path too: body: %s",
		rec.Body.String())
}

// TestProxyRequest_ProviderRateLimitStillWins pins that the 429 case keeps
// its own treatment: a rate limit needs a Retry-After, which a plain status
// relay does not give, so it must not be folded into the generic 4xx path.
func TestProxyRequest_ProviderRateLimitStillWins(t *testing.T) {
	var firstHits atomic.Int32
	first := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		firstHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error": {"message": "Rate limit exceeded"}}`))
	}))
	defer first.Close()

	var secondHits atomic.Int32
	second := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondHits.Add(1)
		hijackAndClose(w)
	}))
	defer second.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "cred-a", Type: config.ProviderTypeOpenAI, BaseURL: first.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
		config.CredentialConfig{
			Name: "cred-b", Type: config.ProviderTypeOpenAI, BaseURL: second.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(3).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusTooManyRequests, rec.Code,
		"a provider rate limit must stay a 429: body: %s", rec.Body.String())
	assert.NotEmpty(t, rec.Header().Get("Retry-After"),
		"a 429 must carry a Retry-After so the client paces its retries")
}

// TestProxyRequest_AllTransportFailuresKeepTheir502 is the counterpart: a
// request whose credentials ALL failed on the transport, with no provider
// status anywhere, keeps its 502 -- it is a genuine outage, and the new
// classification must not swallow it.
func TestProxyRequest_AllTransportFailuresKeepTheir502(t *testing.T) {
	var hits atomic.Int32
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		hijackAndClose(w)
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{
			Name: "cred-a", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL,
			APIKey: "k", RPM: 1000, TPM: 1000000,
		},
	).WithMaxProviderRetries(2).Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	assert.Equal(t, http.StatusBadGateway, rec.Code,
		"a genuine outage must stay a 502: body: %s", rec.Body.String())
	assert.Positive(t, hits.Load())
}
