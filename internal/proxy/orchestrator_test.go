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
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOrchestrateRequest_ResponsesAPIStreaming(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeAnthropic, "http://test.local", "upstream-key").
		WithMasterKey("master-key").
		Build()
	prx.logger = logger

	body := `{"model":"Xpt-5","input":"Hello","stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{}

	prepared, ok := prx.orchestrateRequest(w, req, logCtx)
	require.True(t, ok)
	require.NotNil(t, prepared)

	require.True(t, prepared.isResponsesAPI)
	require.True(t, prepared.streaming)
	require.True(t, prepared.convertedResp)
	require.Equal(t, "/v1/chat/completions", prepared.request.URL.Path)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.body, &raw))

	_, hasInput := raw["input"]
	require.False(t, hasInput, "input should be removed after conversion")

	_, hasMessages := raw["messages"]
	require.True(t, hasMessages, "messages should be present after conversion")

	streamOptions, ok := raw["stream_options"].(map[string]interface{})
	require.True(t, ok, "stream_options should be present")
	require.Equal(t, true, streamOptions["include_usage"])
}

func TestOrchestrateRequest_RecordsReasoningRequestMetadata(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		body         string
		requested    bool
		source       string
		thinkingMode string
		endpoint     string
	}{
		{
			name:         "chat completions",
			path:         "/v1/chat/completions",
			body:         `{"model":"claude-opus-4-8","messages":[{"role":"user","content":"test"}],"reasoning_effort":"high"}`,
			requested:    true,
			source:       "reasoning_effort",
			thinkingMode: thinkingModeUnspecified,
			endpoint:     "/v1/chat/completions",
		},
		{
			name:         "responses",
			path:         "/v1/responses",
			body:         `{"model":"claude-opus-4-8","input":"test","reasoning":{"effort":"high"}}`,
			requested:    true,
			source:       "reasoning",
			thinkingMode: thinkingModeUnspecified,
			endpoint:     "/v1/responses",
		},
		{
			name:         "messages",
			path:         "/v1/messages",
			body:         `{"model":"claude-opus-4-8","max_tokens":4096,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"test"}]}`,
			requested:    true,
			source:       "thinking",
			thinkingMode: thinkingModeAdaptive,
			endpoint:     "/v1/messages",
		},
		{
			name:         "messages disabled",
			path:         "/v1/messages",
			body:         `{"model":"claude-opus-5","max_tokens":4096,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"test"}]}`,
			thinkingMode: thinkingModeDisabled,
			endpoint:     "/v1/messages",
		},
		{
			name:         "messages unspecified",
			path:         "/v1/messages",
			body:         `{"model":"claude-opus-5","max_tokens":4096,"messages":[{"role":"user","content":"test"}]}`,
			thinkingMode: thinkingModeUnspecified,
			endpoint:     "/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().
				WithSingleCredential("test", config.ProviderTypeAnthropic, "http://test.local", "upstream-key").
				WithMasterKey("master-key").
				Build()
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Authorization", "Bearer master-key")
			logCtx := &RequestLogContext{}

			prepared, ok := prx.orchestrateRequest(httptest.NewRecorder(), req, logCtx)

			require.True(t, ok)
			require.NotNil(t, prepared)
			assert.Equal(t, tt.requested, logCtx.ReasoningRequested)
			assert.Equal(t, tt.source, logCtx.ReasoningSource)
			assert.Equal(t, tt.thinkingMode, logCtx.ThinkingMode)
			assert.Equal(t, tt.endpoint, logCtx.RequestEndpoint)
		})
	}
}

func TestOrchestrateRequest_ResponsesAPI_PassthroughForOpenAI(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeOpenAI, "http://test.local", "upstream-key").
		WithMasterKey("master-key").
		Build()
	prx.logger = logger

	body := `{"model":"qwen-5","input":[{"role":"user","content":[{"type":"input_file","file_id":"file-abc"},{"type":"input_text","text":"Hello"}]}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{}

	prepared, ok := prx.orchestrateRequest(w, req, logCtx)
	require.True(t, ok)
	require.NotNil(t, prepared)

	require.True(t, prepared.isResponsesAPI)
	require.False(t, prepared.convertedResp)
	require.True(t, prepared.passthroughResponses)
	require.Equal(t, "/v1/responses", prepared.request.URL.Path)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.body, &raw))

	_, hasInput := raw["input"]
	require.True(t, hasInput, "input should remain for native passthrough")
	require.Contains(t, string(prepared.body), `"file_id":"file-abc"`)
	_, hasMessages := raw["messages"]
	require.False(t, hasMessages, "messages should not be injected for native passthrough")
}

func TestOrchestrateRequest_ResponsesAPI_ConvertedForOpenAIWhenPassthroughDisabled(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	passthroughResponses := false
	builder := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeOpenAI, "http://test.local", "upstream-key").
		WithMasterKey("master-key")
	modelManager := models.New(logger, 50, []config.ModelRPMConfig{
		{
			Name:                 "qwen-5",
			Credential:           "test",
			PassthroughResponses: &passthroughResponses,
		},
	})
	modelManager.LoadModelsFromConfig(builder.config.Credentials)
	builder.config.ModelManager = modelManager
	prx := builder.Build()
	prx.logger = logger

	body := `{"model":"qwen-5","input":"Hello","stream":false}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{}

	prepared, ok := prx.orchestrateRequest(w, req, logCtx)
	require.True(t, ok)
	require.NotNil(t, prepared)

	require.True(t, prepared.isResponsesAPI)
	require.True(t, prepared.convertedResp)
	require.False(t, prepared.passthroughResponses)
	require.Equal(t, "/v1/chat/completions", prepared.request.URL.Path)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.body, &raw))

	_, hasInput := raw["input"]
	require.False(t, hasInput, "input should be removed after conversion")
	_, hasMessages := raw["messages"]
	require.True(t, hasMessages, "messages should be present after conversion")
}

func TestPrepareRequestForCredential_UsesCredentialSpecificRealModel(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	cheap := config.CredentialConfig{Name: "cheapgpt", Type: config.ProviderTypeAnthropic, APIKey: "key", BaseURL: "http://cheapgpt.local", RPM: 100}
	grant := config.CredentialConfig{Name: "grant", Type: config.ProviderTypeBedrock, APIKey: "key2", BaseURL: "http://grant.local", RPM: 100}
	mm := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: "claude", Model: "anthropic/claude-sonnet", Credential: cheap.Name},
		{Name: "claude", Model: "global.anthropic.claude-sonnet-v1:0", Credential: grant.Name},
	})
	mm.LoadModelsFromConfig([]config.CredentialConfig{cheap, grant})

	builder := NewTestProxyBuilder().WithCredentials(cheap, grant)
	builder.config.ModelManager = mm
	prx := builder.Build()

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	body := []byte(`{"model":"claude","messages":[]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"claude",
		"claude",
		"/v1/chat/completions",
		false,
		&grant,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	require.Equal(t, "global.anthropic.claude-sonnet-v1:0", prepared.realModelID)
	require.Contains(t, string(prepared.body), `"model":"global.anthropic.claude-sonnet-v1:0"`)
}

func TestSelectCredentialForModelMarksDirectSpendLogComplete(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	kafka := &stubKafkaManager{enabled: true}
	prx.kafkaLog = kafka
	setTestModelPrice(prx, "gpt-4o-mini", &models.ModelPrice{})
	logCtx := testLogCtx(t)
	logCtx.Credential = nil

	credential, ok := prx.selectCredentialForModel(
		httptest.NewRecorder(),
		"missing-model",
		"",
		"",
		nil,
		logCtx,
	)

	require.False(t, ok)
	require.Nil(t, credential)
	require.True(t, logCtx.Logged)
	require.Len(t, kafka.events, 1)
}

// TestSelectCredentialForModelLogsZeroCostRowWhenPriceUnavailable guards
// against a regression of PR #96/#115 follow-up review TODO 2: the
// "no credentials available" (429) branch logs an audit row before any
// provider is contacted, so logCtx.ModelPrice is never pre-resolved. With
// logSpendToLiteLLMDB now failing closed on an unresolved price, that audit
// row used to be silently dropped whenever the rejected model had no price
// entry. Since no usage was ever incurred here, the row must still be
// written (with zero cost) instead of disappearing.
func TestSelectCredentialForModelLogsZeroCostRowWhenPriceUnavailable(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	kafka := &stubKafkaManager{enabled: true}
	prx.kafkaLog = kafka
	prx.priceRegistry = models.NewModelPriceRegistry() // no price for "missing-model"
	logCtx := testLogCtx(t)
	logCtx.ModelID = "missing-model"
	logCtx.TokenUsage = nil // no provider was ever contacted
	logCtx.Credential = nil

	credential, ok := prx.selectCredentialForModel(
		httptest.NewRecorder(),
		"missing-model",
		"",
		"",
		nil,
		logCtx,
	)

	require.False(t, ok)
	require.Nil(t, credential)
	require.True(t, logCtx.Logged)
	require.Len(t, kafka.events, 1)
	assert.Equal(t, 0.0, kafka.events[0].TotalCost)
}

// TestSelectCredentialForModelSetsRetryAfterFromBan verifies the synthesized
// 429 ("no credentials available") carries a Retry-After header computed
// from the shortest remaining fail2ban ban for the requested model, instead
// of omitting the header entirely.
func TestSelectCredentialForModelSetsRetryAfterFromBan(t *testing.T) {
	cred := config.CredentialConfig{
		Name:    "banned-cred",
		Type:    config.ProviderTypeOpenAI,
		BaseURL: "http://openai.local",
		APIKey:  "key",
		RPM:     -1,
	}
	prx := NewTestProxyBuilder().WithCredentials(cred).Build()
	setTestModelPrice(prx, "banned-model", &models.ModelPrice{})

	// Ban the only credential serving this model for 3s, bypassing the
	// attempt-threshold rules (same mechanism used for provider-supplied
	// quota-retry hints) so the ban has a known, finite remaining duration.
	prx.balancer.BanUntil("banned-cred", "banned-model", http.StatusTooManyRequests, time.Now().Add(3*time.Second), "test-ban")

	logCtx := testLogCtx(t)
	logCtx.Credential = nil

	w := httptest.NewRecorder()
	credential, ok := prx.selectCredentialForModel(
		w,
		"banned-model",
		"",
		"",
		nil,
		logCtx,
	)

	require.False(t, ok)
	require.Nil(t, credential)
	require.Equal(t, http.StatusTooManyRequests, w.Code)

	retryAfter := w.Header().Get("Retry-After")
	require.NotEmpty(t, retryAfter, "expected Retry-After header derived from the active ban")
	seconds, err := strconv.Atoi(retryAfter)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, seconds, 1)
	assert.LessOrEqual(t, seconds, 3)
}

// TestSelectCredentialForModelSetsRetryAfter_EvenWithoutActiveBan verifies
// the core guarantee: the synthesized "no credentials available" 429 must
// always carry a Retry-After header, even when there is no credential at
// all for the model (so no active ban exists to derive a precise ETA from).
func TestSelectCredentialForModelSetsRetryAfter_EvenWithoutActiveBan(t *testing.T) {
	prx := NewTestProxyBuilder().Build() // no credentials configured at all
	setTestModelPrice(prx, "nonexistent-model", &models.ModelPrice{})

	logCtx := testLogCtx(t)
	logCtx.Credential = nil

	w := httptest.NewRecorder()
	credential, ok := prx.selectCredentialForModel(
		w,
		"nonexistent-model",
		"",
		"",
		nil,
		logCtx,
	)

	require.False(t, ok)
	require.Nil(t, credential)
	require.Equal(t, http.StatusTooManyRequests, w.Code)

	retryAfter := w.Header().Get("Retry-After")
	require.NotEmpty(t, retryAfter, "a 429 reaching the client must always carry a Retry-After header, even with no active ban")
	seconds, err := strconv.Atoi(retryAfter)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, seconds, 1)
}

func TestPrepareRequestForCredential_ProxyBodyKeepsOriginalParams(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, APIKey: "key", BaseURL: "http://openai.local", RPM: 100}
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	body := []byte(`{"model":"o3-mini","messages":[],"max_tokens":100,"temperature":0.7}`)
	proxyBody := []byte(`{"model":"gpt-alias","messages":[],"max_tokens":100,"temperature":0.7}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		proxyBody,
		"gpt-alias",
		"o3-mini",
		"/v1/chat/completions",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)

	var direct map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.body, &direct))
	require.Equal(t, "o3-mini", direct["model"])
	require.Contains(t, direct, "max_completion_tokens")
	require.NotContains(t, direct, "max_tokens")

	var forwarded map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.proxyBody, &forwarded))
	require.Equal(t, "gpt-alias", forwarded["model"])
	require.Contains(t, forwarded, "max_tokens")
	require.NotContains(t, forwarded, "max_completion_tokens")
	require.Contains(t, forwarded, "temperature")
}

func TestPrepareRequestForCredential_MessagesKeepsOriginalProxyRequest(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, APIKey: "key", BaseURL: "http://openai.local", RPM: 100}
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	body := []byte(`{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)
	proxyBody := []byte(`{"model":"claude-alias","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		proxyBody,
		"claude-alias",
		"claude-sonnet",
		"/v1/messages",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	require.Equal(t, "/v1/chat/completions", prepared.path)
	require.Equal(t, "/v1/messages", prepared.proxyPath)
	require.JSONEq(t, string(proxyBody), string(prepared.proxyBody))
	require.Contains(t, string(prepared.body), `"model":"claude-sonnet"`)
	require.True(t, prepared.convertedMessages)
}

func TestPrepareRequestForCredential_MessagesProxyLikeCredentialKeepsNativeFormat(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := config.CredentialConfig{Name: "air-peer", Type: config.ProviderTypeAIR, APIKey: "key", BaseURL: "http://air-peer.local", RPM: 100}
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	body := []byte(`{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)
	proxyBody := []byte(`{"model":"claude-alias","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		proxyBody,
		"claude-alias",
		"claude-sonnet",
		"/v1/messages",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	// Proxy-like credentials (AIR-to-AIR chaining) must receive the original
	// Anthropic-shaped request/path unchanged: the downstream peer does its own
	// routing/conversion, same as the Responses API passthrough contract.
	require.Equal(t, "/v1/messages", prepared.path)
	require.Equal(t, "/v1/messages", prepared.proxyPath)
	require.JSONEq(t, string(body), string(prepared.body))
	require.JSONEq(t, string(proxyBody), string(prepared.proxyBody))
	require.False(t, prepared.convertedMessages)
}

func TestPrepareRequestForCredential_MessagesPassthroughForAnthropic(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, APIKey: "key", BaseURL: "http://anthropic.local", RPM: 100}
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	// A document block referencing a provider-hosted file_id: MessagesToChat rejects this
	// shape outright (no Chat Completions equivalent), but a real Anthropic Files API caller
	// can use it — passthrough must forward it untouched instead of erroring.
	body := []byte(`{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_abc"}}]}]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"claude-alias",
		"claude-sonnet",
		"/v1/messages",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	require.True(t, prepared.passthroughMessages)
	require.True(t, prepared.convertedMessages)
	require.Equal(t, "/v1/messages", prepared.path)
	require.Contains(t, string(prepared.body), `"file_id":"file_abc"`)
}

func TestPrepareRequestForCredential_MessagesConvertedForCometAPIOpenAIProtocol(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	cred := config.CredentialConfig{Name: "comet", Type: config.ProviderTypeCometAPI, OpenAIProtocol: true, APIKey: "key", BaseURL: "http://comet.local", RPM: 100}
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	body := []byte(`{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"claude-alias",
		"claude-sonnet",
		"/v1/messages",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	require.False(t, prepared.passthroughMessages)
	require.True(t, prepared.convertedMessages)
	require.Equal(t, "/v1/chat/completions", prepared.path)
}

func TestPrepareRequestForCredential_MessagesPassthroughDisabledByModelOverride(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	cred := config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, APIKey: "key", BaseURL: "http://anthropic.local", RPM: 100}
	disabled := false
	mm := models.New(logger, 50, []config.ModelRPMConfig{
		{Name: "claude-alias", PassthroughMessages: &disabled},
	})
	mm.LoadModelsFromConfig([]config.CredentialConfig{cred})

	builder := NewTestProxyBuilder().WithCredentials(cred)
	builder.config.ModelManager = mm
	prx := builder.Build()

	req := httptest.NewRequest("POST", "/v1/messages", nil)
	body := []byte(`{"model":"claude-sonnet","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`)

	prepared, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"claude-alias",
		"claude-sonnet",
		"/v1/messages",
		false,
		&cred,
		false,
		false,
		false,
	)

	require.NoError(t, err)
	require.False(t, prepared.passthroughMessages)
	require.True(t, prepared.convertedMessages)
	require.Equal(t, "/v1/chat/completions", prepared.path)
}

func TestPrepareRequestForCredential_ResponsesRecomputesProviderMode(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	openaiCred := config.CredentialConfig{Name: "openai", Type: config.ProviderTypeOpenAI, APIKey: "key", BaseURL: "http://openai.local", RPM: 100}
	anthropicCred := config.CredentialConfig{Name: "anthropic", Type: config.ProviderTypeAnthropic, APIKey: "key2", BaseURL: "http://anthropic.local", RPM: 100}

	builder := NewTestProxyBuilder().WithCredentials(openaiCred, anthropicCred)
	builder.config.ModelManager = models.New(logger, 50, []config.ModelRPMConfig{})
	prx := builder.Build()

	req := httptest.NewRequest("POST", "/v1/responses", nil)
	body := []byte(`{"model":"qwen-5","input":"Hello","stream":false}`)

	openaiReq, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"qwen-5",
		"qwen-5",
		"/v1/responses",
		false,
		&openaiCred,
		true,
		false,
		false,
	)
	require.NoError(t, err)
	require.True(t, openaiReq.passthroughResponses)
	require.False(t, openaiReq.convertedResp)
	require.Equal(t, "/v1/responses", openaiReq.path)
	require.Contains(t, string(openaiReq.body), `"input"`)

	anthropicReq, err := prx.prepareRequestForCredential(
		req,
		body,
		body,
		"qwen-5",
		"qwen-5",
		"/v1/responses",
		false,
		&anthropicCred,
		true,
		false,
		false,
	)
	require.NoError(t, err)
	require.True(t, anthropicReq.convertedResp)
	require.False(t, anthropicReq.passthroughResponses)
	require.Equal(t, "/v1/chat/completions", anthropicReq.path)
	require.Equal(t, "/v1/responses", anthropicReq.proxyPath)
	require.Contains(t, string(anthropicReq.body), `"messages"`)
	require.NotContains(t, string(anthropicReq.body), `"input"`)
	// proxyBody must be converted too, not just body: TryFallbackProxy forwards
	// proxyBody to fallback credentials, and a Chat-Completions-only fallback
	// receiving the original Responses-shaped "input" body rejects it outright
	// ("specify prompt or messages") instead of the converted request the
	// primary attempt used.
	require.Contains(t, string(anthropicReq.proxyBody), `"messages"`)
	require.NotContains(t, string(anthropicReq.proxyBody), `"input"`)
}

func TestProxyRequest_ResponsesRetryRecomputesProviderMode(t *testing.T) {
	var openaiCalls int32
	openaiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&openaiCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer openaiSrv.Close()

	var anthropicCalls int32
	var anthropicPath string
	var anthropicBody []byte
	anthropicSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&anthropicCalls, 1)
		anthropicPath = r.URL.Path
		anthropicBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_1",
			"type":"message",
			"role":"assistant",
			"model":"qwen-5",
			"stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer anthropicSrv.Close()

	openaiCred := config.CredentialConfig{
		Name:    "openai",
		Type:    config.ProviderTypeOpenAI,
		APIKey:  "key",
		BaseURL: openaiSrv.URL,
		RPM:     100, Priority: 10,
	}
	anthropicCred := config.CredentialConfig{
		Name:    "anthropic",
		Type:    config.ProviderTypeAnthropic,
		APIKey:  "key2",
		BaseURL: anthropicSrv.URL,
		RPM:     100, Priority: 20,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(openaiCred, anthropicCred).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		Build()

	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"qwen-5","input":"Hello","stream":false}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, int32(1), atomic.LoadInt32(&openaiCalls))
	require.Equal(t, int32(1), atomic.LoadInt32(&anthropicCalls))
	require.Equal(t, "/v1/messages", anthropicPath)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(anthropicBody, &raw))
	require.Contains(t, raw, "messages")
	require.NotContains(t, raw, "input")
}

func TestProxyRequest_UnsupportedProManRequestRoutesToNextPrimary(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	var nextPrimaryCalls int32
	nextPrimary := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&nextPrimaryCalls, 1)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer next-key", r.Header.Get("Authorization"))
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		assert.Contains(t, string(body), `"server_tool_use"`)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-next",
			"object": "chat.completion",
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": "next primary"},
			}},
		})
	}))
	defer nextPrimary.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan, BaseURL: promanUpstream.URL, APIKey: "proman-key", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "next-primary", Type: config.ProviderTypeProxy, BaseURL: nextPrimary.URL, APIKey: "next-key", RPM: 100, TPM: 10000},
		).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "next primary")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
	assert.Equal(t, int32(1), atomic.LoadInt32(&nextPrimaryCalls))
}

func TestProxyRequest_UnsupportedProManRequestRoutesToFallbackProxy(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	var fallbackCalls int32
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-fallback",
			"object": "chat.completion",
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": "fallback ok"},
			}},
		})
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan, BaseURL: promanUpstream.URL, APIKey: "proman-key", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "fallback", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, APIKey: "fallback-key", RPM: 100, TPM: 10000, Priority: config.FallbackPriorityGroup},
		).
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "fallback ok")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
	assert.Equal(t, int32(1), atomic.LoadInt32(&fallbackCalls))
}

// TestProxyRequest_UnsupportedProManRequestFallbackBlockedWhenPriceUnavailable
// guards against a regression of PR #96/#115 follow-up review TODO 1:
// applyCredentialCompatibilityRouting's TryFallbackProxy branch used to
// forward the request to the fallback credential and return before
// enforceBudgetAndRateLimits' price gate ever ran, so an unpriced model could
// reach a real provider with zero billing trace. The price gate must now be
// checked before the fallback proxy is actually contacted.
func TestProxyRequest_UnsupportedProManRequestFallbackBlockedWhenPriceUnavailable(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	var fallbackCalls int32
	fallback := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fallbackCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":     "chatcmpl-fallback",
			"object": "chat.completion",
			"choices": []map[string]any{{
				"message": map[string]string{"role": "assistant", "content": "fallback ok"},
			}},
		})
	}))
	defer fallback.Close()

	prx := NewTestProxyBuilder().
		WithCredentials(
			config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan, BaseURL: promanUpstream.URL, APIKey: "proman-key", RPM: 100, TPM: 10000},
			config.CredentialConfig{Name: "fallback", Type: config.ProviderTypeProxy, BaseURL: fallback.URL, APIKey: "fallback-key", RPM: 100, TPM: 10000, Priority: config.FallbackPriorityGroup},
		).
		Build()
	kafka := &stubKafkaManager{enabled: true}
	prx.kafkaLog = kafka
	prx.priceRegistry = models.NewModelPriceRegistry() // no price for claude-sonnet-4-6

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
	assert.Equal(t, int32(0), atomic.LoadInt32(&fallbackCalls))

	// The rejection must still leave an audit trail — not just an app log —
	// so a forgotten price-registry entry is visible in spend logs/Kafka,
	// not just silently swallowed.
	require.Len(t, kafka.events, 1)
	assert.Equal(t, "failure", kafka.events[0].Status)
	assert.Equal(t, http.StatusServiceUnavailable, kafka.events[0].HTTPStatus)
	assert.Zero(t, kafka.events[0].TotalCost)
	assert.Contains(t, kafka.events[0].ErrorMessage, "model pricing unavailable")
}

func TestProxyRequest_UnsupportedProManRequestWithoutFallbackReturnsLocalError(t *testing.T) {
	var promanCalls int32
	promanUpstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&promanCalls, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer promanUpstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("proman", config.ProviderTypeProMan, promanUpstream.URL, "proman-key").
		Build()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"claude-sonnet-4-6","messages":[{"role":"assistant","content":[{"type":"server_tool_use","name":"web_search"}]}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), unsupportedCredentialRequestMessage)
	assert.NotContains(t, strings.ToLower(w.Body.String()), "proman")
	assert.Equal(t, int32(0), atomic.LoadInt32(&promanCalls))
}
