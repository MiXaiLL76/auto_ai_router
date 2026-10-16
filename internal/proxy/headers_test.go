package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestCopyResponseHeaders_PassthroughByDefault(t *testing.T) {
	src := http.Header{
		"Content-Type":       {"application/json"},
		"X-Provider-Request": {"provider-request-id"},
		"Content-Length":     {"100"},
		"Content-Encoding":   {"gzip"},
		"Connection":         {"close"},
		"X-Credential-Name":  {"internal"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().Build().copyResponseHeaders(w, src, nil, false)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Equal(t, "provider-request-id", w.Header().Get("X-Provider-Request"))
	assert.Empty(t, w.Header().Get("Content-Length"))
	assert.Empty(t, w.Header().Get("Content-Encoding"))
	assert.Empty(t, w.Header().Get("Connection"))
	assert.Empty(t, w.Header().Get("X-Credential-Name"))
}

func TestCopyResponseHeaders_Allowlist(t *testing.T) {
	src := http.Header{
		"Cache-Control":         {"no-cache"},
		"Content-Disposition":   {"attachment"},
		"Content-Range":         {"bytes 0-9/10"},
		"Content-Type":          {"application/json"},
		"ETag":                  {`"revision"`},
		"Last-Modified":         {"Sun, 19 Jul 2026 10:00:00 GMT"},
		"Location":              {"/v1/files/result"},
		"Retry-After":           {"30", "60"},
		"Server":                {"provider"},
		"Set-Cookie":            {"session=secret"},
		"X-Amzn-Requestid":      {"amazon-request-id"},
		"X-Litellm-Model-Id":    {"model-id"},
		"Llm_provider-Api-Base": {"https://provider.example"},
		"X-Future-Provider":     {"future-value"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		copyResponseHeaders(w, src, nil, false)

	assert.Equal(t, "no-cache", w.Header().Get("Cache-Control"))
	assert.Equal(t, "attachment", w.Header().Get("Content-Disposition"))
	assert.Equal(t, "bytes 0-9/10", w.Header().Get("Content-Range"))
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Equal(t, "Sun, 19 Jul 2026 10:00:00 GMT", w.Header().Get("Last-Modified"))
	assert.Equal(t, "/v1/files/result", w.Header().Get("Location"))
	assert.Equal(t, []string{"30", "60"}, w.Header().Values("Retry-After"))
	assert.Empty(t, w.Header().Get("Server"))
	assert.Empty(t, w.Header().Get("Set-Cookie"))
	assert.Empty(t, w.Header().Get("X-Amzn-Requestid"))
	assert.Empty(t, w.Header().Get("X-Litellm-Model-Id"))
	assert.Empty(t, w.Header().Get("Llm_provider-Api-Base"))
	assert.Empty(t, w.Header().Get("X-Future-Provider"))
}

func TestCopyResponseHeaders_RemovesValidators(t *testing.T) {
	src := http.Header{
		"Content-Type":  {"application/json"},
		"ETag":          {`"revision"`},
		"Content-Range": {"bytes 0-9/10"},
	}
	w := httptest.NewRecorder()

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		copyResponseHeaders(w, src, nil, false)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Equal(t, "bytes 0-9/10", w.Header().Get("Content-Range"))
}

func TestSetCredentialResponseHeader_Allowlist(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set("X-Credential-Name", "stale")
	logCtx := &RequestLogContext{IsProxyRequest: true, ActualCredentialName: "provider"}

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		setCredentialResponseHeader(w, logCtx, "")

	assert.Empty(t, w.Header().Get("X-Credential-Name"))
}

func TestSetCredentialResponseHeader_Passthrough(t *testing.T) {
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{IsProxyRequest: true, ActualCredentialName: "provider"}

	NewTestProxyBuilder().Build().setCredentialResponseHeader(w, logCtx, "")

	assert.Equal(t, "provider", w.Header().Get("X-Credential-Name"))
}

func TestCopyRequestHeadersStripsInternalUsageContractHeader(t *testing.T) {
	src := httptestRequestWithHeaders(map[string]string{
		HeaderAIRUsageAudioTokens:  "exclude-cached",
		HeaderAIRProxyClient:       "1",
		HeaderLegacyAIRProxyClient: "1",
		"X-Regular":                "ok",
	})
	dst, err := http.NewRequest(http.MethodPost, "http://upstream.example/v1/chat/completions", nil)
	assert.NoError(t, err)

	copyRequestHeaders(dst, src, "")

	assert.Empty(t, dst.Header.Get(HeaderAIRUsageAudioTokens))
	assert.Empty(t, dst.Header.Get(HeaderAIRProxyClient))
	assert.Empty(t, dst.Header.Get(HeaderLegacyAIRProxyClient))
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

	NewTestProxyBuilder().Build().copyResponseHeaders(w, src, cred, false)

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

	NewTestProxyBuilder().Build().copyResponseHeaders(w, src, cred, false)

	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Equal(t, "debug-upstream", w.Header().Get("X-Litellm-Version"))
	assert.Equal(t, "provider-server", w.Header().Get("Server"))
}

// TestAIRHopReceivesSessionIDHeader covers #266: the body session id is stripped at
// ingress, so a type: air hop must get it as Session-Id to keep session-sticky
// routing on the downstream instance. Plain proxy credentials are left alone.
func TestAIRHopReceivesSessionIDHeader(t *testing.T) {
	for _, tc := range []struct {
		credType   config.ProviderType
		body       string
		wantHeader string
	}{
		{config.ProviderTypeAIR, `{"model":"route-a","session_id":"sess-1","messages":[{"role":"user","content":"hi"}]}`, "sess-1"},
		{config.ProviderTypeAIR, `{"model":"route-a","extra_body":{"litellm_session_id":"sess-2"},"messages":[{"role":"user","content":"hi"}]}`, "sess-2"},
		{config.ProviderTypeAIR, `{"model":"route-a","messages":[{"role":"user","content":"hi"}]}`, ""},
		{config.ProviderTypeProxy, `{"model":"route-a","session_id":"sess-1","messages":[{"role":"user","content":"hi"}]}`, ""},
	} {
		t.Run(string(tc.credType)+"/"+tc.wantHeader, func(t *testing.T) {
			var gotHeader string
			var gotBody map[string]interface{}
			upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotHeader = r.Header.Get(HeaderSessionID)
				_ = json.NewDecoder(r.Body).Decode(&gotBody)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(createMockChatCompletionResponse("chatcmpl-1", "route-a", "ok"))
			}))
			defer upstream.Close()

			cred := proxyCred("downstream", upstream.URL, 1)
			cred.Type = tc.credType
			prx := NewTestProxyBuilder().WithCredentials(cred).WithMasterKey("master-key").Build()
			registerTestModel(prx, cred.Name, "route-a")

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			prx.ProxyRequest(w, req)

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Equal(t, tc.wantHeader, gotHeader)
			assert.NotContains(t, gotBody, "session_id")
		})
	}
}
