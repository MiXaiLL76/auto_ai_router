package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider that refuses content on safety grounds and answers HTTP 200 with
// the failure inside the body -- reproduced from production (OpenRouter
// forwarding Alibaba's own error, model qwen/qwen3.7-plus):
//
//	{"error":{"code":502,"message":"Output data may contain inappropriate content.",
//	          "metadata":{"error_type":"provider_unavailable"}}}
//
// The provider's own code is a NUMBER, which used to fail the decode of the
// whole error object and take the message with it -- so the refusal was
// invisible and the client got a blanket 502 after the request had been
// replayed across every credential.
const upstreamContentFilterBody = `{"id":"gen-1","object":"chat.completion.chunk","created":1794222366,` +
	`"model":"qwen/qwen3.7-plus","provider":"Alibaba","choices":[],` +
	`"error":{"code":502,"message":"Upstream error from Alibaba: Output data may contain inappropriate content.",` +
	`"metadata":{"error_type":"provider_unavailable"}}}`

// TestProxyRequest_ContentFilterRefusalIs400Not502 is the end-to-end check:
// a fake upstream that answers 200 with the refusal body must reach the
// client as a 400 with the content-policy message -- not as a 502, and not
// as a success with the refusal body passed through.
func TestProxyRequest_ContentFilterRefusalIs400Not502(t *testing.T) {
	var upstreamCalls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(upstreamContentFilterBody))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openrouter", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMaxProviderRetries(0).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen/qwen3.7-plus","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code,
		"a content-policy refusal must be a 400, not a 502: %s", w.Body.String())

	var decoded struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &decoded))
	assert.Equal(t, "The prompt was blocked by the provider's content policy", decoded.Error.Message)
	assert.Equal(t, "content_policy_violation", decoded.Error.Code)
	assert.NotContains(t, w.Body.String(), "Alibaba",
		"the provider's own text must not reach the client")
	assert.NotContains(t, w.Body.String(), "provider_unavailable")
}

// TestProxyRequest_ContentFilterRefusalIsNotRetried pins the other half: the
// refusal must not be replayed across the pool. Every credential refuses the
// same content, so a retry only multiplies the delay (measured in production:
// 4 attempts and 55s for one blocked prompt).
func TestProxyRequest_ContentFilterRefusalIsNotRetried(t *testing.T) {
	var upstreamCalls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(upstreamContentFilterBody))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openrouter", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMaxProviderRetries(3).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen/qwen3.7-plus","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
	assert.Equal(t, 1, upstreamCalls,
		"a content-policy refusal must be decided on the first credential")
}

// TestProxyRequest_ProviderFailureStays502 is the counterpart: a provider
// that merely failed keeps its 5xx, so the request stays retryable and the
// client is not told its content was refused.
func TestProxyRequest_ProviderFailureStays502(t *testing.T) {
	var upstreamCalls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"gen-2","object":"chat.completion.chunk","choices":[],` +
			`"error":{"code":502,"message":"Upstream error from Darkbloom: inference generation failed",` +
			`"metadata":{"error_type":"provider_unavailable"}}}`))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openrouter", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMaxProviderRetries(0).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen/qwen3.7-plus","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	proxy.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadGateway, w.Code,
		"a provider failure must stay a 5xx: %s", w.Body.String())
	fmt.Printf("provider failure -> %d, upstream calls %d\n", w.Code, upstreamCalls)
}

// TestProxyRequest_ContentFilterRefusalIs400InLitellmCompat is the same
// check through the response_compatibility: litellm path -- the mode the
// header fix (#334) had to work around, where the compatibility layer's own
// transform would otherwise strip the cost/refusal handling.
func TestProxyRequest_ContentFilterRefusalIs400InLitellmCompat(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(upstreamContentFilterBody))
	}))
	defer upstream.Close()

	proxy := NewTestProxyBuilder().
		WithSingleCredential("openrouter", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMaxProviderRetries(0).
		Build()
	proxy.responseCompat = compatlitellm.New()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"qwen/qwen3.7-plus","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	proxy.WithResponseCompatibility(w, req, proxy.proxyRequest)

	require.Equal(t, http.StatusBadRequest, w.Code,
		"a content-policy refusal must be a 400 in compatibility mode too: %s", w.Body.String())
	assert.Contains(t, w.Body.String(), "content_policy_violation")
	assert.NotContains(t, w.Body.String(), "Alibaba")
}
