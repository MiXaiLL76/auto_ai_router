package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const anthropicDefaultMaxTokensTestValue = 32000

// claudeUpstream starts a fake Anthropic-wire upstream that answers with a one-line
// message and hands every request body it receives to the returned channel.
func claudeUpstream(t *testing.T) (*httptest.Server, <-chan map[string]any) {
	t.Helper()
	captured := make(chan map[string]any, 4)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		captured <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-5-5",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	}))
	t.Cleanup(upstream.Close)
	return upstream, captured
}

// sentMaxTokens sends one request through a proxy with a single credential and returns
// the max_tokens the upstream received.
func sentMaxTokens(t *testing.T, providerType config.ProviderType, defaultMaxTokens int, path, requestBody string) any {
	t.Helper()
	upstream, captured := claudeUpstream(t)
	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", providerType, upstream.URL, "upstream-key").
		Build()
	prx.anthropicDefaultMaxTokens = defaultMaxTokens
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(requestBody))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	select {
	case body := <-captured:
		return body["max_tokens"]
	default:
		t.Fatal("the upstream received no request")
		return nil
	}
}

func TestProxyRequest_AnthropicDefaultMaxTokens_ChatCompletions(t *testing.T) {
	build := func(model, extra string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":"hi"}]` + extra + `}`
	}

	tests := []struct {
		name     string
		provider config.ProviderType
		model    string
		extra    string
		want     float64
	}{
		{"Claude without a limit gets the default", config.ProviderTypeAnthropic, "claude-opus-5-5", "", anthropicDefaultMaxTokensTestValue},
		{"the client's max_tokens wins", config.ProviderTypeAnthropic, "claude-opus-5-5", `,"max_tokens":100`, 100},
		{"the client's max_completion_tokens wins", config.ProviderTypeAnthropic, "claude-opus-5-5", `,"max_completion_tokens":200`, 200},
		{"null max_tokens counts as no limit", config.ProviderTypeAnthropic, "claude-opus-5-5", `,"max_tokens":null`, anthropicDefaultMaxTokensTestValue},
		{"null in both spellings counts as no limit", config.ProviderTypeAnthropic, "claude-opus-5-5", `,"max_tokens":null,"max_completion_tokens":null`, anthropicDefaultMaxTokensTestValue},
		{"a numeric limit next to a null one is kept", config.ProviderTypeAnthropic, "claude-opus-5-5", `,"max_tokens":null,"max_completion_tokens":300`, 300},
		{"Claude behind CometAPI gets the default", config.ProviderTypeCometAPI, "claude-opus-5-5", "", anthropicDefaultMaxTokensTestValue},
		{"Claude behind ProMan gets the default", config.ProviderTypeProMan, "claude-sonnet-5-5", "", anthropicDefaultMaxTokensTestValue},
		{"a model that is not Claude keeps the converter fallback", config.ProviderTypeCometAPI, "gemini-3.5-flash", "", 4096},
		{"an old Claude ID keeps the converter fallback", config.ProviderTypeAnthropic, "claude-3-haiku-20240307", "", 4096},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sentMaxTokens(t, tt.provider, anthropicDefaultMaxTokensTestValue, "/v1/chat/completions", build(tt.model, tt.extra))
			assert.EqualValues(t, tt.want, got)
		})
	}

	t.Run("no configured value uses the built-in default", func(t *testing.T) {
		got := sentMaxTokens(t, config.ProviderTypeAnthropic, 0, "/v1/chat/completions", build("claude-opus-5-5", ""))
		assert.EqualValues(t, config.AnthropicDefaultMaxTokens, got)
	})

	t.Run("a configured value is used", func(t *testing.T) {
		got := sentMaxTokens(t, config.ProviderTypeAnthropic, 16000, "/v1/chat/completions", build("claude-opus-5-5", ""))
		assert.EqualValues(t, 16000, got)
	})
}

func TestProxyRequest_AnthropicDefaultMaxTokens_Responses(t *testing.T) {
	tests := []struct {
		name  string
		extra string
		want  float64
	}{
		{"no limit gets the default", "", anthropicDefaultMaxTokensTestValue},
		{"the client's max_output_tokens wins", `,"max_output_tokens":300`, 300},
		{"null max_output_tokens counts as no limit", `,"max_output_tokens":null`, anthropicDefaultMaxTokensTestValue},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sentMaxTokens(t, config.ProviderTypeAnthropic, anthropicDefaultMaxTokensTestValue, "/v1/responses",
				`{"model":"claude-opus-5-5","input":"hi"`+tt.extra+`}`)
			assert.EqualValues(t, tt.want, got)
		})
	}
}

// A /v1/messages request carries its own max_tokens (the field is required by the
// native API) and is forwarded as it is.
func TestProxyRequest_AnthropicDefaultMaxTokens_MessagesUntouched(t *testing.T) {
	got := sentMaxTokens(t, config.ProviderTypeAnthropic, anthropicDefaultMaxTokensTestValue, "/v1/messages",
		`{"model":"claude-opus-5-5","max_tokens":77,"messages":[{"role":"user","content":"hi"}]}`)
	assert.EqualValues(t, 77, got)
}

func TestApplyAnthropicDefaultMaxTokens(t *testing.T) {
	const body = `{"model":"m","messages":[]}`
	prx := NewTestProxyBuilder().Build()
	prx.anthropicDefaultMaxTokens = anthropicDefaultMaxTokensTestValue

	tests := []struct {
		name     string
		provider config.ProviderType
		model    string
		prepared orchestratedRequest
		want     bool
	}{
		{"Anthropic", config.ProviderTypeAnthropic, "claude-opus-5-5", orchestratedRequest{}, true},
		{"CometAPI", config.ProviderTypeCometAPI, "claude-opus-5-5", orchestratedRequest{}, true},
		{"ProMan", config.ProviderTypeProMan, "claude-opus-5-5", orchestratedRequest{}, true},
		{"Bedrock Claude", config.ProviderTypeBedrock, "anthropic.claude-opus-4-7-v1:0", orchestratedRequest{}, true},
		{"Bedrock model that is not Claude", config.ProviderTypeBedrock, "amazon.nova-pro-v1:0", orchestratedRequest{}, false},
		{"relayed to another router", config.ProviderTypeAIR, "claude-opus-5-5", orchestratedRequest{}, false},
		{"OpenAI credential", config.ProviderTypeOpenAI, "claude-opus-5-5", orchestratedRequest{}, false},
		{"messages passthrough", config.ProviderTypeAnthropic, "claude-opus-5-5", orchestratedRequest{passthroughMessages: true}, false},
		{"legacy completions", config.ProviderTypeAnthropic, "claude-opus-5-5", orchestratedRequest{completionsPath: true}, false},
		{"model that is not Claude", config.ProviderTypeAnthropic, "gemini-3.5-flash", orchestratedRequest{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cred := &config.CredentialConfig{Name: "c", Type: tt.provider}
			out := prx.applyAnthropicDefaultMaxTokens(cred, &tt.prepared, tt.model, []byte(body))
			if !tt.want {
				assert.JSONEq(t, body, string(out))
				return
			}
			var got map[string]any
			require.NoError(t, json.Unmarshal(out, &got))
			assert.EqualValues(t, anthropicDefaultMaxTokensTestValue, got["max_tokens"])
		})
	}

	t.Run("native Responses body gets max_output_tokens", func(t *testing.T) {
		cred := &config.CredentialConfig{Name: "c", Type: config.ProviderTypeAnthropic}
		out := prx.applyAnthropicDefaultMaxTokens(cred, &orchestratedRequest{nativeResponses: true}, "claude-opus-5-5", []byte(`{"model":"m","input":"hi"}`))
		var got map[string]any
		require.NoError(t, json.Unmarshal(out, &got))
		assert.EqualValues(t, anthropicDefaultMaxTokensTestValue, got["max_output_tokens"])
		assert.NotContains(t, got, "max_tokens")
	})
}

// The key budget check runs on the body orchestrateRequest hands over, before the
// request is converted: a request without a limit must still be estimated as before
// (default_estimated_completion_tokens), while the upstream receives the default.
func TestAnthropicDefaultMaxTokens_BudgetEstimateKeepsFallback(t *testing.T) {
	upstream, captured := claudeUpstream(t)
	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
		Build()
	prx.anthropicDefaultMaxTokens = anthropicDefaultMaxTokensTestValue
	const requestBody = `{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(requestBody))
	req.Header.Set("Authorization", "Bearer master-key")
	prepared, ok := prx.orchestrateRequest(httptest.NewRecorder(), req, &RequestLogContext{})
	require.True(t, ok)

	var seenByBudget map[string]any
	require.NoError(t, json.Unmarshal(prepared.body, &seenByBudget))
	assert.NotContains(t, seenByBudget, "max_tokens", "the budget check must not see the router's default")
	assert.Equal(t, 1000, prx.estimateCompletionTokens(prepared.body))

	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(requestBody))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.EqualValues(t, anthropicDefaultMaxTokensTestValue, (<-captured)["max_tokens"])
}

// A request replayed on the next credential after an upstream error goes through the
// same step, so the retry carries the default too.
func TestAnthropicDefaultMaxTokens_AppliedOnRetry(t *testing.T) {
	var calls int32
	bodies := make(chan map[string]any, 4)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"boom"}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-5-5",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}
		}`))
	})
	first := newIPv4Server(t, handler)
	defer first.Close()
	second := newIPv4Server(t, handler)
	defer second.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "anthropic-1", Type: config.ProviderTypeAnthropic, APIKey: "k1", BaseURL: first.URL, RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "anthropic-2", Type: config.ProviderTypeAnthropic, APIKey: "k2", BaseURL: second.URL, RPM: 100, TPM: 10000},
		).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()
	prx.anthropicDefaultMaxTokens = anthropicDefaultMaxTokensTestValue

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	require.EqualValues(t, 2, atomic.LoadInt32(&calls), "the first attempt must fail and be replayed once")
	for i := 0; i < 2; i++ {
		assert.EqualValues(t, anthropicDefaultMaxTokensTestValue, (<-bodies)["max_tokens"], "attempt %d", i+1)
	}
}

// After a direct Claude attempt fails and the request goes to another router, that
// router gets the body the client sent, not the default this router added for its own
// upstream call: the receiving router applies its own value.
func TestAnthropicDefaultMaxTokens_ProxyFallbackBodyUntouched(t *testing.T) {
	primaryBodies := make(chan map[string]any, 2)
	primary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		primaryBodies <- body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"boom"}}`))
	}))
	defer primary.Close()
	fallbackBodies := make(chan map[string]any, 2)
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		fallbackBodies <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, APIKey: "k1", BaseURL: primary.URL, RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "relay", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, RPM: 100, TPM: 10000, IsFallback: true},
		).
		WithMasterKey("master-key").
		Build()
	prx.anthropicDefaultMaxTokens = anthropicDefaultMaxTokensTestValue

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.EqualValues(t, anthropicDefaultMaxTokensTestValue, (<-primaryBodies)["max_tokens"], "the direct attempt carries the default")
	relayed := <-fallbackBodies
	assert.Equal(t, "claude-opus-5-5", relayed["model"])
	assert.NotContains(t, relayed, "max_tokens", "the relayed body must not carry the default as if the client had set it")
}
