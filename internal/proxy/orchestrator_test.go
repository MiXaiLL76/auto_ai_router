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
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
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

func TestOrchestrateRequest_ResponsesAPI_PassthroughForOpenAI(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeOpenAI, "http://test.local", "upstream-key").
		WithMasterKey("master-key").
		Build()
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
	require.False(t, prepared.convertedResp)
	require.True(t, prepared.passthroughResponses)
	require.Equal(t, "/v1/responses", prepared.request.URL.Path)

	var raw map[string]interface{}
	require.NoError(t, json.Unmarshal(prepared.body, &raw))

	_, hasInput := raw["input"]
	require.True(t, hasInput, "input should remain for native passthrough")
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
	require.Contains(t, string(anthropicReq.proxyBody), `"input"`)
	require.NotContains(t, string(anthropicReq.proxyBody), `"messages"`)
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
		Name:             "openai",
		Type:             config.ProviderTypeOpenAI,
		APIKey:           "key",
		BaseURL:          openaiSrv.URL,
		RPM:              100,
		FallbackPriority: 10,
	}
	anthropicCred := config.CredentialConfig{
		Name:             "anthropic",
		Type:             config.ProviderTypeAnthropic,
		APIKey:           "key2",
		BaseURL:          anthropicSrv.URL,
		RPM:              100,
		FallbackPriority: 20,
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
