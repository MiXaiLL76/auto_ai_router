package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricingmodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestShouldRetryWithFallback_RateLimitError(t *testing.T) {
	shouldRetry, reason := ShouldRetryWithFallback(http.StatusTooManyRequests, []byte("rate limited"))

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonRateLimit, reason)
}

func TestShouldRetryWithFallback_ServerErrors(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"500 Internal Server Error", http.StatusInternalServerError},
		{"501 Not Implemented", http.StatusNotImplemented},
		{"502 Bad Gateway", http.StatusBadGateway},
		{"503 Service Unavailable", http.StatusServiceUnavailable},
		{"504 Gateway Timeout", http.StatusGatewayTimeout},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("server error"))

			assert.True(t, shouldRetry)
			assert.Equal(t, RetryReasonServerErr, reason)
		})
	}
}

func TestShouldRetryWithFallback_AuthErrors(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"401 Unauthorized", http.StatusUnauthorized},
		{"403 Forbidden", http.StatusForbidden},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("unauthorized"))

			assert.True(t, shouldRetry)
			assert.Equal(t, RetryReasonAuthErr, reason)
		})
	}
}

func TestShouldRetryWithFallback_PaymentRequired(t *testing.T) {
	shouldRetry, reason := ShouldRetryWithFallback(http.StatusPaymentRequired, []byte("quota exceeded"))

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonPaymentErr, reason)
}

func TestShouldRetryWithFallback_NonRetryableStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
	}{
		{"200 OK", http.StatusOK},
		{"201 Created", http.StatusCreated},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(tt.statusCode, []byte("test"))

			assert.False(t, shouldRetry)
			assert.Equal(t, RetryReason(""), reason)
		})
	}
}

func TestShouldRetryWithFallback_BadRequest(t *testing.T) {
	// 400 Bad Request is retried — a different credential may not produce the same error
	shouldRetry, reason := ShouldRetryWithFallback(http.StatusBadRequest, []byte("bad request"))

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestShouldRetryWithFallback_NotFound(t *testing.T) {
	shouldRetry, reason := ShouldRetryWithFallback(http.StatusNotFound, []byte("not found"))

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestShouldRetryWithFallback_ContentPolicyViolation(t *testing.T) {
	// Content policy violations are NOT retried — they are provider-specific business logic errors
	tests := []struct {
		name     string
		respBody string
	}{
		{"content policy violation", "content policy violation"},
		{"Content Policy violation uppercase", "Content Policy violation"},
		{"CONTENT POLICY", "CONTENT POLICY"},
		{"content management policy", "content management policy"},
		{"policy violation", "policy violation"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(
				http.StatusInternalServerError,
				[]byte(tt.respBody),
			)

			assert.False(t, shouldRetry)
			assert.Equal(t, RetryReason(""), reason)
		})
	}
}

func TestShouldRetryWithFallback_ModelErrors(t *testing.T) {
	tests := []struct {
		name     string
		respBody string
	}{
		{"model not found", "model not found"},
		{"Model Not Found uppercase", "Model Not Found"},
		{"model does not exist", "model does not exist"},
		{"Model Does Not Exist", "Model Does Not Exist"},
		{"unsupported model", "unsupported model"},
		{"UNSUPPORTED MODEL", "UNSUPPORTED MODEL"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldRetry, reason := ShouldRetryWithFallback(
				http.StatusInternalServerError,
				[]byte(tt.respBody),
			)

			assert.True(t, shouldRetry)
			assert.Equal(t, RetryReasonServerErr, reason)
		})
	}
}

func TestShouldRetryWithFallback_RetryableInfrastructureError(t *testing.T) {
	// Regular infrastructure errors should be retried
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusServiceUnavailable,
		[]byte("service temporarily unavailable"),
	)

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestShouldRetryWithFallback_RateLimitWithContentPolicy(t *testing.T) {
	// Content policy in the body suppresses retry regardless of status code
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusTooManyRequests,
		[]byte("content policy violation during rate limit"),
	)

	assert.False(t, shouldRetry)
	assert.Equal(t, RetryReason(""), reason)
}

func TestShouldRetryWithFallback_EmptyResponseBody(t *testing.T) {
	// Empty body should be treated as retryable for retryable status codes
	shouldRetry, reason := ShouldRetryWithFallback(
		http.StatusInternalServerError,
		[]byte(""),
	)

	assert.True(t, shouldRetry)
	assert.Equal(t, RetryReasonServerErr, reason)
}

func TestIsRetryableContent_ContentPolicyViolation(t *testing.T) {
	// Content policy strings are treated as non-retryable (provider-specific business logic)
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{"content policy lowercase", "content policy violation", false},
		{"content policy uppercase", "CONTENT POLICY VIOLATION", false},
		{"content policy mixed", "Content Policy Violation", false},
		{"content management policy", "content management policy violation", false},
		{"policy violation", "policy violation detected", false},
		{"no violation", "server error", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableContent([]byte(tt.content))
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsRetryableContent_ModelErrors(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{"model not found", "model not found", true},
		{"Model Not Found uppercase", "MODEL NOT FOUND", true},
		{"model does not exist", "model does not exist", true},
		{"Model Does Not Exist", "MODEL DOES NOT EXIST", true},
		{"unsupported model", "unsupported model gpt-4", true},
		{"Unsupported Model", "UNSUPPORTED MODEL", true},
		{"other error", "validation error", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableContent([]byte(tt.content))
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRetryReasonConstants(t *testing.T) {
	// Verify retry reason constants are defined
	assert.Equal(t, RetryReason("rate_limit"), RetryReasonRateLimit)
	assert.Equal(t, RetryReason("server_error"), RetryReasonServerErr)
	assert.Equal(t, RetryReason("auth_error"), RetryReasonAuthErr)
	assert.Equal(t, RetryReason("network_error"), RetryReasonNetErr)
}

func TestTryFallbackProxy_Success(t *testing.T) {
	// Track number of calls to fallback server
	var fallbackCalls int32

	// Create fallback server mock that returns 200 OK
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		_ = testhelpers.NewResponseBuilder().
			WithStatus(http.StatusOK).
			WithJSONBody(createMockChatCompletionResponse(
				"chatcmpl-test-fallback",
				"gpt-4",
				"fallback ok",
			)).
			Write(w)
	}))
	defer fallbackServer.Close()

	// Build proxy with primary + fallback credentials
	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.local", fallbackServer.URL).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{
		"model": "gpt-4",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Test message for fallback",
			},
		},
	}
	bodyBytes, err := json.Marshal(requestBody)
	require.NoError(t, err, "Failed to marshal request body")

	// Create HTTP request to proxy endpoint
	req := httptest.NewRequest(
		"POST",
		"/v1/chat/completions",
		strings.NewReader(string(bodyBytes)),
	)

	// Set required headers
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",              // modelID
		"primary",            // originalCredName
		http.StatusOK,        // originalStatus
		RetryReasonRateLimit, // originalReason
		bodyBytes,            // body
		time.Now().UTC(),     // start
		nil,                  // logCtx
	)

	// Assertions
	assert.True(t, success, "TryFallbackProxy should return success=true")
	assert.Empty(t, reason, "TryFallbackProxy should return empty reason on success")

	// Verify fallback server was called
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallbackCalls), "Fallback server should be called exactly once")

	// Check response recorder
	assert.Equal(t, http.StatusOK, w.Code, "Response status code should be 200 OK")
	assert.NotEmpty(t, w.Body.String(), "Response body should not be empty")

	// Verify response contains expected data
	var respData map[string]interface{}
	err = json.Unmarshal(w.Body.Bytes(), &respData)
	require.NoError(t, err, "Failed to unmarshal response")
	assert.Equal(t, "chatcmpl-test-fallback", respData["id"])
	assert.Equal(t, "gpt-4", respData["model"])
}

// TestTryFallbackProxy_ExhaustedSetsRetryAfterFromBan verifies that a real
// upstream 429 relayed after the fallback chain is exhausted still carries a
// Retry-After hint derived from the shortest active ban, not just the
// self-generated "no credentials available" 429 in selectCredentialForModel.
func TestTryFallbackProxy_ExhaustedSetsRetryAfterFromBan(t *testing.T) {
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.local", fallbackServer.URL).
		Build()

	// The original credential is banned by the time the fallback chain
	// exhausts; MinRemainingBanForModel should still find it (exclude=nil).
	prx.balancer.BanUntil("primary", "gpt-4", http.StatusTooManyRequests, time.Now().Add(3*time.Second), "test-ban")

	bodyBytes := []byte(`{"model":"gpt-4","messages":[]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	success, _ := prx.TryFallbackProxy(
		w, req, "gpt-4", "primary", http.StatusTooManyRequests, RetryReasonRateLimit,
		bodyBytes, time.Now().UTC(), nil,
	)

	assert.True(t, success)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)

	retryAfter := w.Header().Get("Retry-After")
	require.NotEmpty(t, retryAfter, "expected Retry-After derived from the active ban on the exhausted fallback response")
	seconds, err := strconv.Atoi(retryAfter)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, seconds, 1)
	assert.LessOrEqual(t, seconds, 3)
}

// TestTryFallbackProxy_ExhaustedSetsRetryAfter_EvenWithoutActiveBan verifies
// the core guarantee: a 429 relayed to the client after the fallback chain
// is exhausted must always carry a Retry-After header, even when zero prior
// failures mean fail2ban has no active ban to report a precise ETA from.
func TestTryFallbackProxy_ExhaustedSetsRetryAfter_EvenWithoutActiveBan(t *testing.T) {
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.local", fallbackServer.URL).
		Build()

	// Deliberately no BanUntil/RecordResponse call — this credential has
	// never failed before and is not, and has never been, banned.

	bodyBytes := []byte(`{"model":"gpt-4","messages":[]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	success, _ := prx.TryFallbackProxy(
		w, req, "gpt-4", "primary", http.StatusTooManyRequests, RetryReasonRateLimit,
		bodyBytes, time.Now().UTC(), nil,
	)

	assert.True(t, success)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)

	retryAfter := w.Header().Get("Retry-After")
	require.NotEmpty(t, retryAfter, "a 429 reaching the client must always carry a Retry-After header, even with no active ban")
	seconds, err := strconv.Atoi(retryAfter)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, seconds, 1)
}

// TestTryFallbackProxy_ExhaustedPreservesUpstreamRetryAfter verifies that a
// Retry-After the upstream itself sent is never overridden by our own
// ban-derived guess.
func TestTryFallbackProxy_ExhaustedPreservesUpstreamRetryAfter(t *testing.T) {
	fallbackServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "42")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer fallbackServer.Close()

	prx := NewTestProxyBuilder().
		WithPrimaryAndFallback("http://primary.local", fallbackServer.URL).
		Build()
	prx.balancer.BanUntil("primary", "gpt-4", http.StatusTooManyRequests, time.Now().Add(3*time.Second), "test-ban")

	bodyBytes := []byte(`{"model":"gpt-4","messages":[]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	_, _ = prx.TryFallbackProxy(
		w, req, "gpt-4", "primary", http.StatusTooManyRequests, RetryReasonRateLimit,
		bodyBytes, time.Now().UTC(), nil,
	)

	assert.Equal(t, "42", w.Header().Get("Retry-After"), "must not override the upstream's own Retry-After")
}

func TestWriteFallbackResponseUsesAIRUsageContractForNonStreaming(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	body := []byte(`{"id":"chatcmpl-fallback","object":"chat.completion","model":"gpt-4o-audio","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":200,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":60}}}`)
	proxyResp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":            []string{"application/json"},
			HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
		},
		Body: body,
	}
	fallbackCred := &config.CredentialConfig{
		Name:    "fallback-air",
		Type:    config.ProviderTypeAIR,
		BaseURL: "http://fallback-air.local",
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o-audio"}`))
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{RequestID: "req-fallback", Logged: true}

	success, reason := prx.writeFallbackResponse(
		w, req, proxyResp, fallbackCred, "gpt-4o-audio", "primary", logCtx, time.Now().UTC(),
	)

	require.True(t, success)
	assert.Empty(t, reason)
	require.NotNil(t, logCtx.TokenUsage)
	assert.Equal(t, 200, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, 80, logCtx.TokenUsage.CachedInputTokens)
	assert.Equal(t, 40, logCtx.TokenUsage.CachedAudioInputTokens)
	assert.Equal(t, 60, logCtx.TokenUsage.AudioInputTokens)
}

func TestWriteFallbackResponseUsesAIRUsageContractForStreaming(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	stream := `data: {"choices":[],"usage":{"prompt_tokens":200,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":80,"cached_audio_tokens":40,"audio_tokens":60}}}` + "\n\n" +
		"data: [DONE]\n\n"
	proxyResp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":            []string{"text/event-stream"},
			HeaderAIRUsageAudioTokens: []string{airUsageAudioTokensExcludeCached},
		},
		StreamBody:  io.NopCloser(strings.NewReader(stream)),
		IsStreaming: true,
	}
	fallbackCred := &config.CredentialConfig{
		Name:    "fallback-air",
		Type:    config.ProviderTypeAIR,
		BaseURL: "http://fallback-air.local",
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o-audio"}`))
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{RequestID: "req-fallback-stream", Logged: true}

	success, reason := prx.writeFallbackResponse(
		w, req, proxyResp, fallbackCred, "gpt-4o-audio", "primary", logCtx, time.Now().UTC(),
	)

	require.True(t, success)
	assert.Empty(t, reason)
	require.NotNil(t, logCtx.TokenUsage)
	assert.Equal(t, 200, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, 80, logCtx.TokenUsage.CachedInputTokens)
	assert.Equal(t, 40, logCtx.TokenUsage.CachedAudioInputTokens)
	assert.Equal(t, 60, logCtx.TokenUsage.AudioInputTokens)
}

func TestTryFallbackProxy_NoFallbackAvailable(t *testing.T) {
	// Build proxy with only primary credential (no fallback)
	prx := NewTestProxyBuilder().
		WithSingleCredential(
			"primary",
			config.ProviderTypeProxy,
			"http://primary.local",
			"pkey",
		).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{
		"model": "gpt-4",
	}
	bodyBytes, _ := json.Marshal(requestBody)

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))
	req.Header.Set("Authorization", "Bearer master-key")

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy (should fail to find fallback)
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",
		"primary",
		http.StatusOK,
		RetryReasonRateLimit,
		bodyBytes,
		time.Now().UTC(),
		nil,
	)

	// Assertions
	assert.False(t, success, "TryFallbackProxy should return success=false when no fallback available")
	assert.Equal(t, "no_fallback_available", reason, "Should return no_fallback_available reason")
}

func TestFallbackStreamingSpendIsFinalizedOnce(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	db := &stubLiteLLMManager{}
	prx.LiteLLMDB = db
	setTestModelPrice(prx, "gpt-4", &pricingmodels.ModelPrice{
		InputCostPerToken: 0.000001, OutputCostPerToken: 0.000002,
	})

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	credential := &config.CredentialConfig{
		Name:    "fallback",
		Type:    config.ProviderTypeProxy,
		BaseURL: "http://fallback.invalid",
	}
	logCtx := &RequestLogContext{
		RequestID:  "fallback-stream-once",
		StartTime:  time.Now().UTC(),
		Request:    request,
		Token:      "client-key",
		ModelID:    "gpt-4",
		Credential: credential,
	}
	response := &ProxyResponse{
		StatusCode:  http.StatusOK,
		Headers:     http.Header{"Content-Type": {"text/event-stream"}},
		StreamBody:  io.NopCloser(strings.NewReader("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")),
		IsStreaming: true,
	}
	recorder := httptest.NewRecorder()

	handled, reason := prx.writeFallbackResponse(
		recorder,
		request,
		response,
		credential,
		"gpt-4",
		"primary",
		logCtx,
		time.Now().UTC(),
	)
	require.True(t, handled)
	require.Empty(t, reason)

	if !logCtx.Logged {
		require.NoError(t, prx.logSpendToLiteLLMDB(logCtx))
	}

	assert.True(t, logCtx.Logged)
	assert.Len(t, db.loggedEntries, 1)
}

func TestFormatTriedCreds(t *testing.T) {
	tests := []struct {
		name    string
		input   map[string]bool
		checkFn func(t *testing.T, result string)
	}{
		{
			name:  "nil slice returns none",
			input: nil,
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "none", result)
			},
		},
		{
			name:  "empty map returns none",
			input: map[string]bool{},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "none", result)
			},
		},
		{
			name:  "single entry",
			input: map[string]bool{"cred-a": true},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "[[cred-a]]", result)
			},
		},
		{
			name:  "multiple entries",
			input: map[string]bool{"cred-a": true, "cred-b": true},
			checkFn: func(t *testing.T, result string) {
				// Map iteration order is non-deterministic, so check both possibilities
				assert.True(t,
					result == "[[cred-a cred-b]]" || result == "[[cred-b cred-a]]",
					"unexpected result: %s", result)
			},
		},
		{
			name:  "entry with false value is excluded",
			input: map[string]bool{"cred-a": true, "cred-b": false},
			checkFn: func(t *testing.T, result string) {
				assert.Equal(t, "[[cred-a]]", result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatTriedCreds(tt.input)
			tt.checkFn(t, result)
		})
	}
}

func TestTryFallbackProxy_SameCredentialAsOriginal(t *testing.T) {
	// Build proxy with single credential marked as both primary and fallback (edge case)
	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{
				Name:       "primary",
				Type:       config.ProviderTypeProxy,
				APIKey:     "pkey",
				BaseURL:    "http://primary.local",
				RPM:        100,
				TPM:        10000,
				IsFallback: true, // Edge case: marked as fallback
			},
		).
		Build()

	// Prepare request body
	requestBody := map[string]interface{}{"model": "gpt-4"}
	bodyBytes, _ := json.Marshal(requestBody)

	// Create HTTP request
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(bodyBytes)))

	// Create response recorder
	w := httptest.NewRecorder()

	// Call TryFallbackProxy (should detect same credential)
	success, reason := prx.TryFallbackProxy(
		w,
		req,
		"gpt-4",
		"primary",
		http.StatusOK,
		RetryReasonRateLimit,
		bodyBytes,
		time.Now().UTC(),
		nil,
	)

	// Assertions
	assert.False(t, success, "TryFallbackProxy should return success=false when fallback is same credential")
	assert.Equal(t, "fallback_is_same_credential", reason, "Should return fallback_is_same_credential reason")
}

func TestShouldRetryWithFallback_DeterministicBadRequest(t *testing.T) {
	// 400s that fault the request itself fail identically on every credential, so
	// retrying them only multiplies one client mistake into one upstream error per
	// account in the rotation.
	bodies := []string{
		`{"error":{"code":400,"message":"Penalty is not enabled for this model","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"Thinking level MINIMAL is not supported for this model.","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"Thinking level is unsupported: THINKING_LEVEL_MINIMAL","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"Unsupported MIME type: application/octet-stream","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"* GenerateContentRequest.contents[2].parts[0].data: required oneof field 'data' must have one initialized field","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"Unable to submit request because it has a maxOutputTokens value of 100000 but the supported range is from 1 (inclusive) to 65537 (exclusive)."}}`,
	}
	for _, body := range bodies {
		t.Run(body[:60], func(t *testing.T) {
			shouldRetry, _ := ShouldRetryWithFallback(http.StatusBadRequest, []byte(body))
			if shouldRetry {
				t.Errorf("expected no retry for deterministic bad request, got retry")
			}
		})
	}
}

func TestShouldRetryWithFallback_CredentialSpecific400StillRetries(t *testing.T) {
	// The opposite case must keep working: a 400 that can differ between accounts
	// is exactly what moving to the next credential fixes.
	bodies := []string{
		`{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"model not found","status":"INVALID_ARGUMENT"}}`,
		`{"error":{"code":400,"message":"User location is not supported for the API use.","status":"FAILED_PRECONDITION"}}`,
	}
	for _, body := range bodies {
		t.Run(body[:50], func(t *testing.T) {
			shouldRetry, _ := ShouldRetryWithFallback(http.StatusBadRequest, []byte(body))
			if !shouldRetry {
				t.Errorf("expected retry for credential-specific 400, got none")
			}
		})
	}
}

func TestShouldRetryWithFallback_MarkersOnlyApplyTo400(t *testing.T) {
	// The markers describe bad requests; the same text arriving with a 5xx is a
	// server fault and must stay retryable.
	body := []byte(`{"error":{"message":"Penalty is not enabled for this model"}}`)
	shouldRetry, _ := ShouldRetryWithFallback(http.StatusInternalServerError, body)
	if !shouldRetry {
		t.Errorf("expected retry for 500 regardless of body text")
	}
}
