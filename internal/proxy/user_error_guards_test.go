package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetryPolicy_VLLMDoesNotRetry400ByDefault(t *testing.T) {
	rp := newRetryPolicy(config.RetryConfig{})
	vllm := &config.CredentialConfig{Name: "vllm-a", Type: config.ProviderTypeVLLM}
	openaiCred := &config.CredentialConfig{Name: "oa", Type: config.ProviderTypeOpenAI}
	body := []byte(`{"object":"error","message":"Unexpected reasoning effort minimal. Supported types are xhigh (default), medium, and low.","type":"BadRequestError","code":400}`)

	retry, _ := rp.shouldRetry(vllm, http.StatusBadRequest, body)
	assert.False(t, retry, "a vLLM 400 fails the same on every replica")

	retry, reason := rp.shouldRetry(vllm, http.StatusServiceUnavailable, nil)
	assert.True(t, retry)
	assert.Equal(t, RetryReasonServerErr, reason)

	retry, reason = rp.shouldRetry(vllm, http.StatusTooManyRequests, nil)
	assert.True(t, retry)
	assert.Equal(t, RetryReasonRateLimit, reason)

	retry, _ = rp.shouldRetry(openaiCred, http.StatusBadRequest, body)
	assert.True(t, retry, "other providers keep retrying 400 unless a marker matches")
}

func TestRetryPolicy_Overrides(t *testing.T) {
	rp := newRetryPolicy(config.RetryConfig{
		StatusCodes: []int{429, 500, 502, 503},
		ProviderOverrides: map[config.ProviderType]config.RetryOverrideConfig{
			config.ProviderTypeVLLM: {StatusCodes: []int{400, 503}},
		},
		CredentialOverrides: map[string]config.RetryOverrideConfig{
			"never": {StatusCodes: []int{}},
		},
	})
	openaiCred := &config.CredentialConfig{Name: "oa", Type: config.ProviderTypeOpenAI}
	vllm := &config.CredentialConfig{Name: "vllm-a", Type: config.ProviderTypeVLLM}
	never := &config.CredentialConfig{Name: "never", Type: config.ProviderTypeVLLM}

	retry, _ := rp.shouldRetry(openaiCred, http.StatusBadRequest, nil)
	assert.False(t, retry, "400 is not in the configured global set")
	retry, _ = rp.shouldRetry(openaiCred, http.StatusBadGateway, nil)
	assert.True(t, retry)

	retry, _ = rp.shouldRetry(vllm, http.StatusBadRequest, nil)
	assert.True(t, retry, "a configured provider override replaces the built-in vllm one")
	retry, _ = rp.shouldRetry(vllm, http.StatusBadGateway, nil)
	assert.False(t, retry, "the provider override replaces the global set, not adds to it")

	retry, _ = rp.shouldRetry(never, http.StatusServiceUnavailable, nil)
	assert.False(t, retry, "an empty credential override disables retries")

	retry, _ = rp.shouldRetry(nil, http.StatusBadGateway, nil)
	assert.True(t, retry, "no credential falls back to the global set")
}

func TestRetryPolicy_ConfiguredMarkers(t *testing.T) {
	rp := newRetryPolicy(config.RetryConfig{
		NonRetryableMarkers: []string{"Prompt Blocked"},
		BadRequestMarkers:   []string{"maximum context length"},
	})
	cred := &config.CredentialConfig{Name: "oa", Type: config.ProviderTypeOpenAI}

	retry, _ := rp.shouldRetry(cred, http.StatusInternalServerError, []byte(`{"error":"prompt blocked by filter"}`))
	assert.False(t, retry, "non-retryable markers apply to any status, case-insensitively")

	retry, _ = rp.shouldRetry(cred, http.StatusBadRequest, []byte(`This model's maximum context length is 262144 tokens`))
	assert.False(t, retry)
	retry, _ = rp.shouldRetry(cred, http.StatusInternalServerError, []byte(`This model's maximum context length is 262144 tokens`))
	assert.True(t, retry, "bad-request markers only apply to 400")

	retry, _ = rp.shouldRetry(cred, http.StatusBadRequest, []byte(`Penalty is not enabled for this model`))
	assert.False(t, retry, "built-in markers stay active next to configured ones")
	retry, _ = rp.shouldRetry(cred, http.StatusBadRequest, []byte(`content policy violation`))
	assert.False(t, retry)
}

// countingVLLM is a fake vLLM that answers every request with a fixed status/body
// and remembers what it was sent.
type countingVLLM struct {
	mu     sync.Mutex
	hits   int
	bodies []map[string]any
}

func newCountingVLLM(t *testing.T, c *countingVLLM, status int, respBody string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		c.mu.Lock()
		c.hits++
		c.bodies = append(c.bodies, body)
		c.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(respBody))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (c *countingVLLM) snapshot() (int, []map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, append([]map[string]any(nil), c.bodies...)
}

// newTwoReplicaVLLMProxy builds a proxy with two vLLM replicas of one model.
func newTwoReplicaVLLMProxy(t *testing.T, urlA, urlB string, effortMap *config.ReasoningEffortMap, retry config.RetryConfig) *Proxy {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	creds := []config.CredentialConfig{
		{Name: "vllm-a", Type: config.ProviderTypeVLLM, BaseURL: urlA, RPM: 100, TPM: 1000000},
		{Name: "vllm-b", Type: config.ProviderTypeVLLM, BaseURL: urlB, RPM: 100, TPM: 1000000},
	}
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "qwen-flash", Credential: "vllm-a", RPM: -1, TPM: -1, ReasoningEffortMap: effortMap},
		{Name: "qwen-flash", Credential: "vllm-b", RPM: -1, TPM: -1, ReasoningEffortMap: effortMap},
	})
	manager.LoadModelsFromConfig(creds)

	builder := NewTestProxyBuilder().WithCredentials(creds...).WithMasterKey("master-key").
		WithMaxProviderRetries(2).WithRetry(retry)
	builder.config.ModelManager = manager
	return builder.Build()
}

func postChat(t *testing.T, prx *Proxy, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", stringsReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

const vllmBadRequestBody = `{"object":"error","message":"'max_tokens' or 'max_completion_tokens' is too large: 9000000.","type":"BadRequestError","param":null,"code":400}`

const vllmOKBody = `{"id":"chatcmpl-1","object":"chat.completion","model":"qwen-flash","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`

func TestVLLM_BadRequestIsNotReplayedOnOtherReplicas(t *testing.T) {
	a, b := &countingVLLM{}, &countingVLLM{}
	srvA := newCountingVLLM(t, a, http.StatusBadRequest, vllmBadRequestBody)
	srvB := newCountingVLLM(t, b, http.StatusBadRequest, vllmBadRequestBody)
	prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, nil, config.RetryConfig{})

	w := postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":9000000}`)

	assert.Equal(t, http.StatusBadRequest, w.Code, "the client gets the upstream 400")
	assert.Contains(t, w.Body.String(), "max_completion_tokens")
	hitsA, _ := a.snapshot()
	hitsB, _ := b.snapshot()
	assert.Equal(t, 1, hitsA+hitsB, "a vLLM 400 must reach exactly one replica")
}

func TestVLLM_ServerErrorStillRetriesOnOtherReplica(t *testing.T) {
	a, b := &countingVLLM{}, &countingVLLM{}
	srvA := newCountingVLLM(t, a, http.StatusServiceUnavailable, `{"error":"overloaded"}`)
	srvB := newCountingVLLM(t, b, http.StatusServiceUnavailable, `{"error":"overloaded"}`)
	prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, nil, config.RetryConfig{})

	postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}]}`)

	hitsA, _ := a.snapshot()
	hitsB, _ := b.snapshot()
	assert.Equal(t, 1, hitsA)
	assert.Equal(t, 1, hitsB, "a 5xx is still replayed on the next replica")
}

func TestVLLM_ConfiguredOverrideRestores400Retry(t *testing.T) {
	a, b := &countingVLLM{}, &countingVLLM{}
	srvA := newCountingVLLM(t, a, http.StatusBadRequest, vllmBadRequestBody)
	srvB := newCountingVLLM(t, b, http.StatusBadRequest, vllmBadRequestBody)
	prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, nil, config.RetryConfig{
		ProviderOverrides: map[config.ProviderType]config.RetryOverrideConfig{
			config.ProviderTypeVLLM: {StatusCodes: []int{400}},
		},
	})

	postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}]}`)

	hitsA, _ := a.snapshot()
	hitsB, _ := b.snapshot()
	assert.Equal(t, 2, hitsA+hitsB)
}

func TestVLLM_EmptyToolsAreDropped(t *testing.T) {
	a, b := &countingVLLM{}, &countingVLLM{}
	srvA := newCountingVLLM(t, a, http.StatusOK, vllmOKBody)
	srvB := newCountingVLLM(t, b, http.StatusOK, vllmOKBody)
	prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, nil, config.RetryConfig{})

	w := postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":true}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, bodiesA := a.snapshot()
	_, bodiesB := b.snapshot()
	bodies := append(bodiesA, bodiesB...)
	require.Len(t, bodies, 1)
	assert.NotContains(t, bodies[0], "tools")
	assert.NotContains(t, bodies[0], "tool_choice")
	assert.NotContains(t, bodies[0], "parallel_tool_calls")
}

func TestVLLM_ReasoningEffortIsMapped(t *testing.T) {
	effortMap, err := config.NewReasoningEffortMap(map[string]string{
		"minimal": "low", "high": "medium", "max": "xhigh", "default": "low",
	})
	require.NoError(t, err)

	tests := []struct {
		name string
		body string
		path []string
		want string
	}{
		{"mapped", `"reasoning_effort":"minimal"`, []string{"reasoning_effort"}, "low"},
		{"mapped high", `"reasoning_effort":"HIGH"`, []string{"reasoning_effort"}, "medium"},
		{"accepted value kept", `"reasoning_effort":"xhigh"`, []string{"reasoning_effort"}, "xhigh"},
		{"unknown -> default", `"reasoning_effort":"turbo"`, []string{"reasoning_effort"}, "low"},
		{"chat_template_kwargs", `"chat_template_kwargs":{"reasoning_effort":"max","enable_thinking":true}`, []string{"chat_template_kwargs", "reasoning_effort"}, "xhigh"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := &countingVLLM{}, &countingVLLM{}
			srvA := newCountingVLLM(t, a, http.StatusOK, vllmOKBody)
			srvB := newCountingVLLM(t, b, http.StatusOK, vllmOKBody)
			prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, &effortMap, config.RetryConfig{})

			w := postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}],`+tt.body+`}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())

			_, bodiesA := a.snapshot()
			_, bodiesB := b.snapshot()
			bodies := append(bodiesA, bodiesB...)
			require.Len(t, bodies, 1)
			var got any = bodies[0]
			for _, key := range tt.path {
				got = got.(map[string]any)[key]
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestVLLM_ReasoningEffortUntouchedWithoutMap(t *testing.T) {
	a, b := &countingVLLM{}, &countingVLLM{}
	srvA := newCountingVLLM(t, a, http.StatusOK, vllmOKBody)
	srvB := newCountingVLLM(t, b, http.StatusOK, vllmOKBody)
	prx := newTwoReplicaVLLMProxy(t, srvA.URL, srvB.URL, nil, config.RetryConfig{})

	w := postChat(t, prx, `{"model":"qwen-flash","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"minimal"}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, bodiesA := a.snapshot()
	_, bodiesB := b.snapshot()
	bodies := append(bodiesA, bodiesB...)
	require.Len(t, bodies, 1)
	assert.Equal(t, "minimal", bodies[0]["reasoning_effort"])
}
