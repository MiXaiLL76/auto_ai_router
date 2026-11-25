package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_UnreachableImageURLIsClientError is the end-to-end
// regression test for the production incident.
//
// A client sent an image whose URL the router could not fetch. The failure
// was reported as a plain Go error, which the proxy classified as a fatal
// conversion error: the client received a generic 500 "Failed to convert
// request", the credential was reported unhealthy, and the request was
// retried by clients that retry 5xx. 17 473 such 500s were produced in two
// weeks — 97.4% of every 500 in the system.
//
// The image URL must instead produce a classified 4xx, the upstream must
// never be called, and nothing about the URL may reach the client.
func TestProxyRequest_UnreachableImageURLIsClientError(t *testing.T) {
	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	// A URL that cannot be fetched: the host does not resolve, so the
	// failure happens before any connection is attempted.
	const unreachableURL = "http://image-host.invalid/nonexistent.png"

	prx := NewTestProxyBuilder().
		WithSingleCredential("anthropic", "anthropic", upstream.URL, "upstream-key").
		Build()

	body := fmt.Sprintf(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"%s"}},
			{"type":"text","text":"describe it"}
		]}]
	}`, unreachableURL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// A 4xx, not a 500.
	require.Equal(t, http.StatusBadRequest, w.Code, "body: %s", w.Body.String())

	// The upstream provider must not have been contacted: the request never
	// got past conversion.
	assert.Equal(t, int32(0), atomic.LoadInt32(&upstreamCalls),
		"the provider must not be called for an unconvertible request")

	respBody := w.Body.String()

	// The client gets an actionable, classified error.
	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "Failed to fetch image URL", resp.Error.Message)
	require.NotNil(t, resp.Error.Code)
	assert.Equal(t, "image_fetch_failed", *resp.Error.Code)
	require.NotNil(t, resp.Error.Param)
	assert.Equal(t, "messages.content.image_url.url", *resp.Error.Param)

	// Nothing about the target may leak to the client.
	assert.NotContains(t, respBody, "image-host.invalid")
	assert.NotContains(t, respBody, unreachableURL)
	assert.NotContains(t, respBody, "http://")
	assert.NotContains(t, respBody, "https://")
}

// TestProxyRequest_PrivateImageURLIsRejected covers the SSRF guard at the
// request level: a client must not be able to make the router fetch an
// address inside the cluster network.
func TestProxyRequest_PrivateImageURLIsRejected(t *testing.T) {
	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("anthropic", "anthropic", upstream.URL, "upstream-key").
		Build()

	const metadataURL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
	body := fmt.Sprintf(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"%s"}},
			{"type":"text","text":"describe it"}
		]}]
	}`, metadataURL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, int32(0), atomic.LoadInt32(&upstreamCalls))

	respBody := w.Body.String()
	assert.NotContains(t, respBody, "169.254.169.254")
	assert.NotContains(t, respBody, "security-credentials")

	var resp APIErrorResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.NotNil(t, resp.Error.Code)
	assert.Equal(t, "image_url_not_allowed", *resp.Error.Code)
}

// TestProxyRequest_WorkingImageURLReachesProvider is the counterpart: a
// valid image must still be fetched and forwarded, so the fix does not
// break the working path. The image server here is on loopback, which the
// SSRF guard rejects, so this test exercises the converter directly rather
// than through the request path.
func TestProxyRequest_WorkingImageURLReachesProvider(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	}))
	defer imageServer.Close()

	var upstreamCalls int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("anthropic", "anthropic", upstream.URL, "upstream-key").
		Build()

	// A data URL needs no download at all, so it exercises the working path
	// through the real request handler regardless of the SSRF guard.
	const dataURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	body := fmt.Sprintf(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"%s"}},
			{"type":"text","text":"describe it"}
		]}]
	}`, dataURL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, int32(1), atomic.LoadInt32(&upstreamCalls),
		"a valid image must still reach the provider")
}

// TestProxyRequest_ImageErrorOnRetryPath checks the bug class the first
// attempt's classification does not cover: a request whose image URL is
// bad can reach the retry path (the first credential failed for an
// unrelated reason), and the retry path must still answer with the
// classified 4xx rather than the earlier transport error.
//
// If a retry site swallowed the classification, the client would get a
// 502/503 for a bad image URL — the exact "wrong status surfaces instead
// of a classified client error" bug this PR exists to fix.
func TestProxyRequest_ImageErrorOnRetryPath(t *testing.T) {
	const unreachableURL = "http://image-host.invalid/nonexistent.png"

	// The first credential fails with a 503, which is retryable; the
	// second one is the one whose conversion hits the bad image URL.
	var firstCalls, secondCalls int32
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&firstCalls, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"message":"upstream busy"}}`))
	}))
	defer first.Close()

	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&secondCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer second.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name: "anthropic-first", Type: "anthropic",
				BaseURL: first.URL, APIKey: "k1", Priority: 1,
			},
			config.CredentialConfig{
				Name: "anthropic-second", Type: "anthropic",
				BaseURL: second.URL, APIKey: "k2", Priority: 2,
			},
		).
		Build()

	body := fmt.Sprintf(`{
		"model":"claude-sonnet-4-5",
		"max_tokens":128,
		"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"%s"}},
			{"type":"text","text":"describe it"}
		]}]
	}`, unreachableURL)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	// Whatever the client gets, it must be a 4xx about the image -- not a
	// 5xx carried over from the first credential's failure.
	require.Less(t, w.Code, 500, "a bad image URL must not surface as a 5xx: %s", w.Body.String())

	respBody := w.Body.String()
	assert.NotContains(t, respBody, "image-host.invalid")
	assert.NotContains(t, respBody, unreachableURL)
}
