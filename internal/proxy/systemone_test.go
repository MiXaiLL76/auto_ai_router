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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/mixaill76/auto_ai_router/internal/models"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const systemOneTestBody = `{
	"model": "typesafe/jev-1.14",
	"state": {"ticket": "refund request", "history": [1, 2.5, null]},
	"questions": [
		{"id": "q1", "type": "noul", "text": "Is it a refund?", "instructions": "Answer yes or no", "criteria": ["explicit", {"weight": 0.5}]},
		{"id": "q2", "type": "score", "text": "How urgent?"}
	],
	"provider": {"order": ["x"]},
	"tools": []
}`

const systemOneTestReply = `{"id":"so_1","model":"jev-1.14.0","answers":[{"id":"q1","type":"noul","answer":"yes","probabilities":{"yes":0.91,"no":0.09},"confidence":0.91},{"id":"q2","type":"score","score":0.73,"probabilities":null,"legend":{"0":"calm","1":"urgent"},"refusal":null}],"usage":{"input_tokens":120,"output_tokens":3}}`

type systemOneUpstream struct {
	server *httptest.Server
	calls  atomic.Int32
	path   atomic.Value
	body   atomic.Value
}

func newSystemOneUpstream(t *testing.T, status int, reply string) *systemOneUpstream {
	t.Helper()
	u := &systemOneUpstream{}
	u.server = newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.calls.Add(1)
		u.path.Store(r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		u.body.Store(string(body))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(u.server.Close)
	return u
}

func (u *systemOneUpstream) receivedBody(t *testing.T) map[string]any {
	t.Helper()
	raw, _ := u.body.Load().(string)
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &body))
	return body
}

// newSystemOneTestProxy builds a proxy with a System One primary (priority 1),
// a System One fallback (priority 999, its own upstream model id) and an
// ordinary chat credential, retrying the primary only on 402, 429 and 5xx.
func newSystemOneTestProxy(t *testing.T, primaryURL, fallbackURL, chatURL string) *Proxy {
	t.Helper()
	credentials := []config.CredentialConfig{
		{Name: "typesafe", Type: config.ProviderTypeOpenAI, BaseURL: primaryURL, APIKey: "p", RPM: 100, TPM: 100000, Priority: 1, SystemOne: true},
		{Name: "requesty-typesafe", Type: config.ProviderTypeOpenAI, BaseURL: fallbackURL + "/v1", APIKey: "f", RPM: -1, TPM: -1, Priority: 999, SystemOne: true},
		{Name: "chat", Type: config.ProviderTypeOpenAI, BaseURL: chatURL, APIKey: "c", RPM: 100},
	}
	retryCodes := append([]int{402, 429}, statusRange(500, 599)...)
	builder := NewTestProxyBuilder().
		WithCredentials(credentials...).
		WithMasterKey("master-key").
		WithMaxProviderRetries(2).
		WithRetry(config.RetryConfig{CredentialOverrides: map[string]config.RetryOverrideConfig{
			"typesafe":          {StatusCodes: retryCodes},
			"requesty-typesafe": {StatusCodes: retryCodes},
		}})
	builder.config.ModelManager = models.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "typesafe/jev-1.14", Model: "jev-1.14.0", Credential: "typesafe"},
		{Name: "typesafe/jev-1.14", Model: "typesafe/jev-1.14.0", Credential: "requesty-typesafe"},
		{Name: "gpt-chat", Credential: "chat"},
	})
	builder.config.ModelManager.LoadModelsFromConfig(credentials)
	prx := builder.Build()
	prx.rateLimiter.AddModel("typesafe", "typesafe/jev-1.14", -1)
	prx.rateLimiter.AddModel("requesty-typesafe", "typesafe/jev-1.14", -1)
	return prx
}

func statusRange(from, to int) []int {
	codes := make([]int, 0, to-from+1)
	for code := from; code <= to; code++ {
		codes = append(codes, code)
	}
	return codes
}

func systemOneRequest(path, body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestSystemOne_ForwardsBodyAndKeepsReply(t *testing.T) {
	primary := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
	prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), primary.calls.Load())
	assert.Zero(t, fallback.calls.Load())
	assert.Zero(t, chat.calls.Load())
	assert.Equal(t, "/v1/systemone", primary.path.Load())

	// Only the model id changes on the way up; no chat fields are added or dropped.
	sent := primary.receivedBody(t)
	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(systemOneTestBody), &want))
	want["model"] = "jev-1.14.0"
	assert.Equal(t, want, sent)

	// The reply keeps every field (nulls and fractional scores included) and
	// names the public model.
	var got map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	var wantReply map[string]any
	require.NoError(t, json.Unmarshal([]byte(systemOneTestReply), &wantReply))
	wantReply["model"] = "typesafe/jev-1.14"
	assert.Equal(t, wantReply, got)
	assert.Contains(t, w.Body.String(), `"score":0.73`)

	// No total_tokens in the reply: input + output feed the credential and model TPM.
	assert.Equal(t, 123, prx.rateLimiter.GetCurrentTPM("typesafe"))
	assert.Equal(t, 123, prx.rateLimiter.GetCurrentModelTPM("typesafe", "typesafe/jev-1.14"))
}

func TestSystemOne_FallsBackOnRetryableStatus(t *testing.T) {
	for _, status := range []int{http.StatusPaymentRequired, http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable, 529} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			primary := newSystemOneUpstream(t, status, `{"error":{"message":"upstream trouble"}}`)
			fallback := newSystemOneUpstream(t, http.StatusOK, strings.Replace(systemOneTestReply, `"jev-1.14.0"`, `"typesafe/jev-1.14.0"`, 1))
			chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
			prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

			w := httptest.NewRecorder()
			prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Equal(t, int32(1), primary.calls.Load())
			assert.Equal(t, int32(1), fallback.calls.Load())
			assert.Zero(t, chat.calls.Load())
			assert.Equal(t, "/v1/systemone", fallback.path.Load())
			assert.Equal(t, "typesafe/jev-1.14.0", fallback.receivedBody(t)["model"])
			assert.Contains(t, w.Body.String(), `"model":"typesafe/jev-1.14"`)
			assert.Equal(t, 123, prx.rateLimiter.GetCurrentTPM("requesty-typesafe"))
			assert.Zero(t, prx.rateLimiter.GetCurrentTPM("typesafe"))
		})
	}
}

func TestSystemOne_FallsBackOnNetworkError(t *testing.T) {
	dead := newIPv4Server(t, http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close()
	fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
	prx := newSystemOneTestProxy(t, deadURL, fallback.server.URL, chat.server.URL)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), fallback.calls.Load())
}

func TestSystemOne_LiteLLMCompatibilityLeavesReplyAlone(t *testing.T) {
	primary := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
	prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)
	prx.responseCompat = compatlitellm.New()

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Contains(t, w.Body.String(), `"probabilities":null`, "nulls survive")
	assert.Contains(t, w.Body.String(), `"usage":{"input_tokens":120,"output_tokens":3}`)
}

func TestSystemOne_ClientErrorsAreNotRetried(t *testing.T) {
	for _, status := range []int{http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnsupportedMediaType, http.StatusUnprocessableEntity} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			primary := newSystemOneUpstream(t, status, `{"error":{"message":"question exceeds 32768 tokens"}}`)
			fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
			chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
			prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

			w := httptest.NewRecorder()
			prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

			assert.Equal(t, status, w.Code)
			assert.Equal(t, int32(1), primary.calls.Load())
			assert.Zero(t, fallback.calls.Load())
		})
	}
}

func TestSystemOne_RefusalInsideOKIsNotRetried(t *testing.T) {
	refusal := `{"model":"jev-1.14.0","answers":[{"id":"q1","type":"refusal","refusal":"cannot judge"}],"usage":{"input_tokens":10,"output_tokens":0}}`
	primary := newSystemOneUpstream(t, http.StatusOK, refusal)
	fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
	prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, systemOneRequest("/v1/systemone", systemOneTestBody))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"type":"refusal"`)
	assert.Zero(t, fallback.calls.Load())
}

func TestSystemOne_RoutingRejections(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
		want string
	}{
		{"stream on System One", "/v1/systemone", strings.Replace(systemOneTestBody, `"tools": []`, `"stream": true`, 1), "stream is not supported"},
		{"chat model on System One", "/v1/systemone", strings.Replace(systemOneTestBody, `"typesafe/jev-1.14"`, `"gpt-chat"`, 1), "does not support the System One API"},
		{"System One model on chat", "/v1/chat/completions", `{"model":"typesafe/jev-1.14","messages":[{"role":"user","content":"hi"}]}`, "only available via /v1/systemone"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			primary := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
			fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
			chat := newSystemOneUpstream(t, http.StatusOK, `{}`)
			prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

			w := httptest.NewRecorder()
			prx.ProxyRequest(w, systemOneRequest(tt.path, tt.body))

			assert.Equal(t, http.StatusBadRequest, w.Code)
			assert.Contains(t, w.Body.String(), tt.want)
			assert.Zero(t, primary.calls.Load()+fallback.calls.Load()+chat.calls.Load())
		})
	}
}

func TestSystemOne_ChatStillServedByOrdinaryCredential(t *testing.T) {
	primary := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	fallback := newSystemOneUpstream(t, http.StatusOK, systemOneTestReply)
	chat := newSystemOneUpstream(t, http.StatusOK, `{"id":"c","object":"chat.completion","model":"gpt-chat","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	prx := newSystemOneTestProxy(t, primary.server.URL, fallback.server.URL, chat.server.URL)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, systemOneRequest("/v1/chat/completions", `{"model":"gpt-chat","messages":[{"role":"user","content":"hi"}]}`))

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, int32(1), chat.calls.Load())
	assert.Zero(t, primary.calls.Load()+fallback.calls.Load())
}

func TestSystemOneUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	systemOne := &RequestLogContext{IsSystemOne: true}

	tokens, usage := prx.systemOneUsage(t.Context(), systemOne, "c", http.StatusOK, nil, "m", 0,
		&converter.TokenUsage{PromptTokens: 40, CompletionTokens: 2})
	assert.Equal(t, 42, tokens, "input + output when total_tokens is absent")
	assert.Equal(t, 42, usage.Total())

	tokens, _ = prx.systemOneUsage(t.Context(), systemOne, "c", http.StatusOK, nil, "m", 50,
		&converter.TokenUsage{PromptTokens: 40, CompletionTokens: 2})
	assert.Equal(t, 50, tokens, "a reported total_tokens wins")

	tokens, usage = prx.systemOneUsage(t.Context(), systemOne, "c", http.StatusOK, []byte(systemOneTestBody), "m", 0, nil)
	require.NotNil(t, usage, "a reply without usage is not a free call")
	assert.Positive(t, usage.PromptTokens)
	assert.Equal(t, usage.PromptTokens, tokens)
	assert.Equal(t, "estimated", systemOne.UsageSource)

	armed := &RequestLogContext{IsSystemOne: true}
	prx.setPromptTokensEstimate(armed, []byte(systemOneTestBody), "m")
	tokens, usage = prx.systemOneUsage(t.Context(), armed, "c", http.StatusOK, nil, "m", 0, nil)
	require.NotNil(t, usage, "without a body the armed estimate is billed")
	assert.Equal(t, estimatePromptTokensForModel([]byte(systemOneTestBody), "m"), usage.PromptTokens)
	assert.Equal(t, usage.PromptTokens, tokens)
	assert.Equal(t, "estimated", armed.UsageSource)

	chat := &RequestLogContext{}
	tokens, usage = prx.systemOneUsage(t.Context(), chat, "c", http.StatusOK, []byte(systemOneTestBody), "m", 0, nil)
	assert.Zero(t, tokens)
	assert.Nil(t, usage, "other APIs keep their behavior")

	tokens, usage = prx.systemOneUsage(t.Context(), &RequestLogContext{IsSystemOne: true}, "c", http.StatusBadGateway, []byte(systemOneTestBody), "m", 0, nil)
	assert.Zero(t, tokens)
	assert.Nil(t, usage, "errors are not billed")
}

// TestSystemOne_ProxyFallbackWithoutUsageBillsEstimate: a reply relayed by a
// proxy-like fallback without any usage is billed on the prompt estimate, not
// as a free call. writeFallbackResponse has no request body at hand, so it
// relies on the estimate TryFallbackProxy arms.
func TestSystemOne_ProxyFallbackWithoutUsageBillsEstimate(t *testing.T) {
	fallback := newSystemOneUpstream(t, http.StatusOK, `{"id":"so_1","model":"jev-1.14.0","answers":[{"id":"q1","type":"noul","answer":"yes"}]}`)
	prx := NewTestProxyBuilder().WithPrimaryAndFallback("http://primary.local", fallback.server.URL).Build()
	logCtx := &RequestLogContext{RequestID: "so-fallback", IsSystemOne: true, Scope: scope.PublicContext(), Logged: true}
	w := httptest.NewRecorder()

	success, reason := prx.TryFallbackProxy(w, systemOneRequest("/v1/systemone", systemOneTestBody), "typesafe/jev-1.14", "primary",
		http.StatusServiceUnavailable, RetryReasonServerErr, []byte(systemOneTestBody), time.Now().UTC(), logCtx)

	require.True(t, success, reason)
	require.Equal(t, int32(1), fallback.calls.Load())
	want := estimatePromptTokensForModel([]byte(systemOneTestBody), "typesafe/jev-1.14")
	require.Positive(t, want)
	require.NotNil(t, logCtx.TokenUsage, "a reply without usage is not a free call")
	assert.Equal(t, want, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, "estimated", logCtx.UsageSource)
	assert.Equal(t, want, prx.rateLimiter.GetCurrentTPM("fallback"))
}

func TestEstimatePromptTokens_SystemOneGrowsWithInputAndQuestions(t *testing.T) {
	question := `{"id":"q","type":"choice","text":"Which department handles this ticket?","instructions":"Pick one of the options","criteria":["billing","support","sales"]}`
	body := func(state string, questions int) []byte {
		qs := make([]string, questions)
		for i := range qs {
			qs[i] = question
		}
		return []byte(`{"model":"typesafe/jev-1.14","state":` + state + `,"questions":[` + strings.Join(qs, ",") + `]}`)
	}
	short := `"a customer writes about a refund"`
	long := `"` + strings.Repeat("a customer writes about a refund and a late delivery ", 50) + `"`

	one := estimatePromptTokens(body(short, 1))
	assert.Greater(t, one, 20, "nested instructions and criteria are counted")
	assert.Greater(t, estimatePromptTokens(body(short, 5)), one)
	assert.Greater(t, estimatePromptTokens(body(long, 1)), one)

	objectState := estimatePromptTokens(body(`{"ticket":{"subject":"refund","lines":["late","damaged"]}}`, 1))
	arrayState := estimatePromptTokens(body(`["refund","late","damaged"]`, 1))
	questionsOnly := estimatePromptTokens([]byte(`{"model":"m","questions":[` + question + `]}`))
	assert.Greater(t, objectState, questionsOnly)
	assert.Greater(t, arrayState, questionsOnly)
}
