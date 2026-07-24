package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestGetHopByHopHeaders(t *testing.T) {
	headers := GetHopByHopHeaders()

	// Should contain all 8 RFC 7230 hop-by-hop headers
	expectedHeaders := []string{
		"Connection",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"TE",
		"Trailer",
		"Transfer-Encoding",
		"Upgrade",
	}

	assert.Len(t, headers, len(expectedHeaders))
	for _, h := range expectedHeaders {
		assert.True(t, headers[h], "should contain %s", h)
	}

	// Verify it returns a copy (modifying it doesn't affect the original)
	headers["X-Custom"] = true
	original := GetHopByHopHeaders()
	_, hasCustom := original["X-Custom"]
	assert.False(t, hasCustom, "modifying returned map should not affect the original")
}

func TestCopyRequestHeadersStripsInternalUsageContractHeader(t *testing.T) {
	src := httptestRequestWithHeaders(map[string]string{
		HeaderAIRUsageAudioTokens: "exclude-cached",
		HeaderAIRProxyClient:      "1",
		"X-Regular":               "ok",
	})
	dst, err := http.NewRequest(http.MethodPost, "http://upstream.example/v1/chat/completions", nil)
	assert.NoError(t, err)

	copyRequestHeaders(dst, src, "")

	assert.Empty(t, dst.Header.Get(HeaderAIRUsageAudioTokens))
	assert.Empty(t, dst.Header.Get(HeaderAIRProxyClient))
	assert.Equal(t, "ok", dst.Header.Get("X-Regular"))
}

func httptestRequestWithHeaders(headers map[string]string) *http.Request {
	req, _ := http.NewRequest(http.MethodPost, "http://router.example/v1/chat/completions", nil)
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	return req
}

func TestCopyResponseHeadersProManStripsInternalProviderHeaders(t *testing.T) {
	src := http.Header{
		"Content-Type":                           []string{"application/json"},
		"X-Litellm-Version":                      []string{"1.92.0"},
		"X-Litellm-Response-Cost":                []string{"0.001"},
		"Llm_provider-Anthropic-Organization-Id": []string{"org_hidden"},
		"X-Powered-By":                           []string{"LiteLLM"},
		"Server":                                 []string{"uvicorn"},
		"Request-Id":                             []string{"req_proxied"},
		"X-Ratelimit-Limit-Requests":             []string{"100"},
		"Anthropic-Ratelimit-Tokens-Limit":       []string{"10000"},
		"X-Amzn-Requestid":                       []string{"bedrock_req"},
		"X-Credential-Name":                      []string{"anthropic-promanYT-01"},
	}
	cred := &config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan}
	w := httptest.NewRecorder()

	copyResponseHeaders(w, src, cred)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	for _, header := range []string{
		"X-Litellm-Version",
		"X-Litellm-Response-Cost",
		"Llm_provider-Anthropic-Organization-Id",
		"X-Powered-By",
		"Server",
		"Request-Id",
		"X-Ratelimit-Limit-Requests",
		"Anthropic-Ratelimit-Tokens-Limit",
		"X-Amzn-Requestid",
		"X-Credential-Name",
	} {
		assert.Empty(t, w.Header().Get(header), "header %s must not reach clients", header)
	}
}

func TestCopyResponseHeadersRegularCredentialKeepsNonStructuralHeaders(t *testing.T) {
	src := http.Header{
		"Content-Type":      []string{"application/json"},
		"X-Litellm-Version": []string{"debug-upstream"},
		"Server":            []string{"provider-server"},
	}
	cred := &config.CredentialConfig{Name: "anthropic-promanYT-01", Type: config.ProviderTypeAnthropic}
	w := httptest.NewRecorder()

	copyResponseHeaders(w, src, cred)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Equal(t, "debug-upstream", w.Header().Get("X-Litellm-Version"))
	assert.Equal(t, "provider-server", w.Header().Get("Server"))
}
