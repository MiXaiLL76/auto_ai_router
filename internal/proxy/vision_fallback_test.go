package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	routermodels "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	visionTestImage = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA"
	visionTestOwl   = "A brown owl sits on a birch branch at night."
)

// visionUpstream is a fake vLLM serving a text-only model (glm) and a vision model
// (qwen-vl). It records every request body in arrival order.
type visionUpstream struct {
	mu         sync.Mutex
	bodies     []map[string]any
	paths      []string
	failVision bool
	visionBody string // raw 200 body returned for the vision model when set
}

func (u *visionUpstream) snapshot() ([]string, []map[string]any) {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]string(nil), u.paths...), append([]map[string]any(nil), u.bodies...)
}

func newVisionUpstream(t *testing.T, u *visionUpstream) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		u.mu.Lock()
		u.bodies = append(u.bodies, body)
		u.paths = append(u.paths, r.URL.Path)
		failVision, visionBody := u.failVision, u.visionBody
		u.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/responses" {
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"glm","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
			return
		}
		answer := "ok"
		if body["model"] == "qwen-vl" {
			if failVision {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"bad image ` + strings.Repeat("x", 400) + `","type":"invalid_request_error"}}`))
				return
			}
			if visionBody != "" {
				_, _ = w.Write([]byte(visionBody))
				return
			}
			answer = visionTestOwl
		}
		resp, _ := json.Marshal(map[string]any{
			"id": "chatcmpl-1", "object": "chat.completion", "model": body["model"],
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": answer}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
		_, _ = w.Write(resp)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func newVisionProxy(t *testing.T, upstreamURL string, fallback config.VisionFallbackConfig) (*Proxy, *organizationPolicyTestDB) {
	t.Helper()
	logger := testhelpers.NewTestLogger()
	credential := config.CredentialConfig{
		Name: "vllm-node", Type: config.ProviderTypeVLLM, BaseURL: upstreamURL, RPM: 100, TPM: 1000000,
	}
	openaiCredential := config.CredentialConfig{
		Name: "openai-node", Type: config.ProviderTypeOpenAI, APIKey: "sk-test", BaseURL: upstreamURL, RPM: 100, TPM: 1000000,
	}
	creds := []config.CredentialConfig{credential, openaiCredential}
	noVision, vision := false, true
	manager := routermodels.New(logger, 100, []config.ModelRPMConfig{
		{Name: "glm", Credential: credential.Name, RPM: -1, TPM: -1, SupportsVision: &noVision},
		{Name: "qwen-vl", Credential: credential.Name, RPM: -1, TPM: -1, SupportsVision: &vision},
		{Name: "gpt-oss", Credential: credential.Name, RPM: -1, TPM: -1},
		{Name: "glm-alias", Model: "zai/glm-5", Credential: credential.Name, RPM: -1, TPM: -1, SupportsVision: &noVision},
		// Not vLLM-only: the flag must be ignored.
		{Name: "openai-text", Credential: openaiCredential.Name, RPM: -1, TPM: -1, SupportsVision: &noVision},
		{Name: "mixed", Credential: credential.Name, RPM: -1, TPM: -1, SupportsVision: &noVision},
		{Name: "mixed", Credential: openaiCredential.Name, RPM: -1, TPM: -1, SupportsVision: &noVision},
	})
	manager.LoadModelsFromConfig(creds)
	manager.SetCredentials(creds)

	prices := routermodels.NewModelPriceRegistry()
	prices.MergeDB(map[string]*routermodels.ModelPrice{
		"glm":         {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"qwen-vl":     {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"gpt-oss":     {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"openai-text": {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"mixed":       {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"glm-alias":   {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
	})

	builder := NewTestProxyBuilder().WithCredentials(creds...).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	builder.config.VisionFallback = fallback
	prx := builder.Build()
	db := newVLLMTestDB()
	prx.LiteLLMDB = db
	prx.priceRegistry = prices
	return prx, db
}

func describeFallback() config.VisionFallbackConfig {
	return config.VisionFallbackConfig{DescribeModel: "qwen-vl"}
}

func sendVisionRequest(t *testing.T, prx *Proxy, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, stringsReader(body))
	req.Header.Set("Authorization", "Bearer token")
	req.Header.Set("X-OpenWebUI-User-Email", "user@example.com")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	return w
}

// messageParts returns the content parts of messages[i] of an upstream chat body.
func messageParts(t *testing.T, body map[string]any, i int) []any {
	t.Helper()
	messages, ok := body["messages"].([]any)
	require.True(t, ok, "messages: %v", body)
	require.Greater(t, len(messages), i)
	parts, ok := messages[i].(map[string]any)["content"].([]any)
	require.True(t, ok, "message %d content is not an array: %v", i, messages[i])
	return parts
}

func partText(t *testing.T, part any) string {
	t.Helper()
	m, ok := part.(map[string]any)
	require.True(t, ok)
	text, _ := m["text"].(string)
	return text
}

// First turn with an image: the image is described by the vision model and the text-only
// model receives the description instead of the image. Both calls are billed to the key.
func TestVisionFallback_DescribesCurrentTurnImage(t *testing.T) {
	u := &visionUpstream{}
	prx, db := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","temperature":0.2,"messages":[
		{"role":"user","content":"привет"},
		{"role":"assistant","content":"Привет! Чем могу помочь?"},
		{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "described", w.Header().Get(HeaderVisionFallback))

	paths, bodies := u.snapshot()
	require.Len(t, bodies, 2, "one describe call, then the real request")

	describe := bodies[0]
	assert.Equal(t, "/v1/chat/completions", paths[0])
	assert.Equal(t, "qwen-vl", describe["model"])
	assert.Equal(t, false, describe["stream"])
	describeParts := messageParts(t, describe, 1)
	require.Len(t, describeParts, 2)
	assert.Contains(t, partText(t, describeParts[0]), "что на картинке?", "the user's question is passed as context")
	assert.Equal(t, visionTestImage, describeParts[1].(map[string]any)["image_url"].(map[string]any)["url"])

	main := bodies[1]
	assert.Equal(t, "glm", main["model"])
	assert.EqualValues(t, 0.2, main["temperature"], "other request fields survive the rewrite")
	parts := messageParts(t, main, 2)
	require.Len(t, parts, 2)
	assert.Equal(t, "что на картинке?", partText(t, parts[0]))
	assert.Equal(t, "text", parts[1].(map[string]any)["type"])
	assert.Contains(t, partText(t, parts[1]), visionTestOwl)
	assert.Contains(t, partText(t, parts[1]), "described by qwen-vl")
	assert.NotContains(t, main["messages"].([]any)[2].(map[string]any)["content"], "image_url")

	require.Len(t, db.logs, 2, "the describe call is a separate billed request")
	models := []string{db.logs[0].Model, db.logs[1].Model}
	assert.ElementsMatch(t, []string{"qwen-vl", "glm"}, models)
	for _, log := range db.logs {
		assert.Equal(t, "user@example.com", log.EndUser, "the describe call is billed to the same end user")
		assert.Equal(t, "success", log.Status)
	}
}

// A later turn: the image is in the history, the assistant already answered about it.
// It becomes a placeholder and no describe call is made.
func TestVisionFallback_HistoryImageBecomesPlaceholder(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":"привет"},
		{"role":"assistant","content":"Привет! Чем могу помочь?"},
		{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]},
		{"role":"assistant","content":"На картинке изображена сова."},
		{"role":"user","content":"спасибо"}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "stripped", w.Header().Get(HeaderVisionFallback))

	_, bodies := u.snapshot()
	require.Len(t, bodies, 1, "no describe call for history images")
	parts := messageParts(t, bodies[0], 2)
	assert.Equal(t, "[image from an earlier turn omitted]", partText(t, parts[1]))
}

// A tool result carrying a screenshot after the assistant's tool call is part of the
// current turn and is described.
func TestVisionFallback_DescribesToolResultImage(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":"сделай скриншот"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"screenshot","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	assert.Contains(t, partText(t, messageParts(t, bodies[1], 2)[0]), visionTestOwl)
}

func TestVisionFallback_RejectMode(t *testing.T) {
	u := &visionUpstream{}
	prx, db := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{Mode: config.VisionFallbackReject})

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "glm does not support image inputs")
	_, bodies := u.snapshot()
	assert.Empty(t, bodies, "rejected before any upstream call")
	assert.Empty(t, db.logs)
}

func TestVisionFallback_StripMode(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{Mode: config.VisionFallbackStrip})

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"text","text":"что это?"},{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "stripped", w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	require.Len(t, bodies, 1)
	assert.Equal(t, "[image omitted: the model cannot see images]", partText(t, messageParts(t, bodies[0], 0)[1]))
}

// Vision models and models that never declared supports_vision get images unchanged.
func TestVisionFallback_OtherModelsUntouched(t *testing.T) {
	for _, model := range []string{"qwen-vl", "gpt-oss"} {
		t.Run(model, func(t *testing.T) {
			u := &visionUpstream{}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
			w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"`+model+`","messages":[
				{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Empty(t, w.Header().Get(HeaderVisionFallback))
			_, bodies := u.snapshot()
			require.Len(t, bodies, 1)
			assert.Equal(t, "image_url", messageParts(t, bodies[0], 0)[0].(map[string]any)["type"])
		})
	}
}

// Vision fallback is vLLM-only: a supports_vision: false model reachable through any
// non-vLLM credential is never rewritten or rejected, even in reject mode.
func TestVisionFallback_IgnoresNonVLLMModels(t *testing.T) {
	for _, model := range []string{"openai-text", "mixed"} {
		t.Run(model, func(t *testing.T) {
			u := &visionUpstream{}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{Mode: config.VisionFallbackReject})
			w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"`+model+`","messages":[
				{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			assert.Empty(t, w.Header().Get(HeaderVisionFallback))
			_, bodies := u.snapshot()
			require.Len(t, bodies, 1)
			assert.Equal(t, "image_url", messageParts(t, bodies[0], 0)[0].(map[string]any)["type"], "image forwarded unchanged")
		})
	}
}

// A failed describe call does not fail the request: the image becomes a placeholder.
func TestVisionFallback_DescribeFailureKeepsRequest(t *testing.T) {
	u := &visionUpstream{failVision: true}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "stripped", w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	last := bodies[len(bodies)-1]
	assert.Equal(t, "glm", last["model"])
	assert.Equal(t, "[image omitted: the image could not be described]", partText(t, messageParts(t, last, 0)[0]))
}

// describe_model misconfigured as a text-only model: the inner call strips instead of
// recursing, and the request still completes.
func TestVisionFallback_NoRecursionWhenDescribeModelLacksVision(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{DescribeModel: "glm"})

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	_, bodies := u.snapshot()
	assert.Len(t, bodies, 2, "one inner call (stripped) and the real request")
}

// /v1/messages (Anthropic shape) with a base64 image block: converted to Chat for vLLM,
// the image is described first.
func TestVisionFallback_MessagesAPI(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/messages", `{"model":"glm","max_tokens":100,"messages":[
		{"role":"user","content":[{"type":"text","text":"что на картинке?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	describeParts := messageParts(t, bodies[0], 1)
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgo=", describeParts[1].(map[string]any)["image_url"].(map[string]any)["url"])
	assert.Contains(t, mustJSON(t, bodies[1]["messages"]), visionTestOwl)
	assert.NotContains(t, mustJSON(t, bodies[1]["messages"]), "iVBORw0KGgo=")
}

// /v1/responses is passed through to vLLM natively; input_image parts are rewritten
// to input_text.
func TestVisionFallback_ResponsesAPI(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/responses", `{"model":"glm","store":false,"input":[
		{"role":"user","content":[{"type":"input_text","text":"что на картинке?"},{"type":"input_image","image_url":"`+visionTestImage+`"}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "described", w.Header().Get(HeaderVisionFallback))

	paths, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	assert.Equal(t, "/v1/responses", paths[1])
	input := bodies[1]["input"].([]any)
	parts := input[0].(map[string]any)["content"].([]any)
	assert.Equal(t, "input_text", parts[1].(map[string]any)["type"])
	assert.Contains(t, partText(t, parts[1]), visionTestOwl)
}

func TestCollectVisionImages_ResponsesTurnBoundary(t *testing.T) {
	root, err := decodeVisionBody([]byte(`{"input":[
		{"role":"user","content":[{"type":"input_image","image_url":"data:old"}]},
		{"type":"function_call","call_id":"c1","name":"shot","arguments":"{}"},
		{"type":"function_call_output","call_id":"c1","output":[{"type":"input_image","image_url":"data:new"},{"type":"input_image","file_id":"file-1"}]}]}`))
	require.NoError(t, err)
	refs, _ := collectVisionImages(root, visionFormatResponses)
	require.Len(t, refs, 3)
	assert.False(t, refs[0].current)
	assert.True(t, refs[1].current)
	assert.Equal(t, "data:new", refs[1].url)
	assert.True(t, refs[2].current)
	assert.Empty(t, refs[2].url, "file_id images cannot be described")
}

func TestCollectVisionImages_MessagesToolResult(t *testing.T) {
	root, err := decodeVisionBody([]byte(`{"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"shot","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"image","source":{"type":"url","url":"https://x/a.png"}}]},{"type":"text","text":"что тут?"}]}]}`))
	require.NoError(t, err)
	refs, question := collectVisionImages(root, visionFormatMessages)
	require.Len(t, refs, 1)
	assert.True(t, refs[0].current)
	assert.Equal(t, "https://x/a.png", refs[0].url)
	assert.Equal(t, "что тут?", question)
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return strings.ReplaceAll(string(b), `=`, "=")
}

// Only the first max_images images of the turn are described.
func TestVisionFallback_MaxImages(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{DescribeModel: "qwen-vl", MaxImages: 1})

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}},{"type":"image_url","image_url":"`+visionTestImage+`"}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())

	_, bodies := u.snapshot()
	require.Len(t, bodies, 2, "one describe call only")
	parts := messageParts(t, bodies[1], 0)
	assert.Contains(t, partText(t, parts[0]), visionTestOwl)
	assert.Contains(t, partText(t, parts[1]), "too many images")
}

// A describe answer that is not a chat completion, or is empty, leaves a placeholder.
func TestVisionFallback_BadDescribeResponses(t *testing.T) {
	for name, body := range map[string]string{
		"not json":      `not json`,
		"no choices":    `{"choices":[]}`,
		"empty content": `{"choices":[{"message":{"content":"   "}}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			u := &visionUpstream{visionBody: body}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
			w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm","messages":[
				{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
			require.Equal(t, http.StatusOK, w.Code, w.Body.String())
			_, bodies := u.snapshot()
			assert.Equal(t, "[image omitted: the image could not be described]", partText(t, messageParts(t, bodies[len(bodies)-1], 0)[0]))
		})
	}
}

// A model whose real upstream name differs from its router name keeps the real name
// in the rewritten body.
func TestVisionFallback_RealModelNameKept(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm-alias","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	_, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	assert.Equal(t, "zai/glm-5", bodies[1]["model"])
	assert.Contains(t, partText(t, messageParts(t, bodies[1], 0)[0]), "because glm-alias cannot see images")
}

func TestVisionFallback_RejectOnResponsesAndMessages(t *testing.T) {
	for path, body := range map[string]string{
		"/v1/responses": `{"model":"glm","input":[{"role":"user","content":[{"type":"input_image","image_url":"` + visionTestImage + `"}]}]}`,
		"/v1/messages":  `{"model":"glm","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://x/a.png"}}]}]}`,
	} {
		t.Run(path, func(t *testing.T) {
			u := &visionUpstream{}
			prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{Mode: config.VisionFallbackReject})
			w := sendVisionRequest(t, prx, path, body)
			assert.Equal(t, http.StatusBadRequest, w.Code, w.Body.String())
			_, bodies := u.snapshot()
			assert.Empty(t, bodies)
		})
	}
}

func TestApplyVisionFallback_EdgeCases(t *testing.T) {
	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, config.VisionFallbackConfig{Mode: config.VisionFallbackStrip})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	t.Run("invalid json mentioning image", func(t *testing.T) {
		body := []byte(`{"model":"glm","messages":"image`)
		res := prx.applyVisionFallback(req, body, "glm", visionFormatChat)
		assert.Equal(t, body, res.body)
		assert.Empty(t, res.outcome)
	})
	t.Run("the word image without image parts", func(t *testing.T) {
		body := []byte(`{"model":"glm","messages":[{"role":"user","content":"draw an image"}]}`)
		res := prx.applyVisionFallback(req, body, "glm", visionFormatChat)
		assert.Equal(t, body, res.body)
		assert.False(t, res.rejected)
	})
	t.Run("no model manager", func(t *testing.T) {
		bare := &Proxy{}
		body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}`)
		assert.Equal(t, body, bare.applyVisionFallback(req, body, "glm", visionFormatChat).body)
	})
	t.Run("unset mode rejects", func(t *testing.T) {
		unset := *prx
		unset.visionFallback.Mode = ""
		body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}`)
		assert.True(t, unset.applyVisionFallback(req, body, "glm", visionFormatChat).rejected)
	})
	t.Run("model without credentials", func(t *testing.T) {
		assert.False(t, prx.servedOnlyByVLLM("no-such-model"))
	})
	t.Run("numbers survive the rewrite", func(t *testing.T) {
		body := []byte(`{"seed":12345678901234567890,"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}`)
		res := prx.applyVisionFallback(req, body, "glm", visionFormatChat)
		assert.Equal(t, "stripped", res.outcome)
		assert.Contains(t, string(res.body), `"seed":12345678901234567890`)
	})
}

// If JSON encoding fails, the describe call is skipped and the body is left unchanged.
func TestApplyVisionFallback_EncodeFailure(t *testing.T) {
	orig := marshalVisionJSON
	marshalVisionJSON = func(any) ([]byte, error) { return nil, assert.AnError }
	t.Cleanup(func() { marshalVisionJSON = orig })

	u := &visionUpstream{}
	prx, _ := newVisionProxy(t, newVisionUpstream(t, u).URL, describeFallback())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}`)
	res := prx.applyVisionFallback(req, body, "glm", visionFormatChat)
	assert.Equal(t, body, res.body)
	assert.Empty(t, res.outcome)
	_, bodies := u.snapshot()
	assert.Empty(t, bodies, "no describe call went out")
}

func TestCollectVisionImages_Robustness(t *testing.T) {
	long := strings.Repeat("я", maxVisionQuestionChars+10) + "END"
	root, err := decodeVisionBody([]byte(`{"messages":[
		"junk",
		{"role":"user","content":null},
		{"role":"user","content":["junk",{"type":"text","text":"` + long + `"},{"type":"text","text":42},{"type":"image_url","image_url":42}]}]}`))
	require.NoError(t, err)
	refs, question := collectVisionImages(root, visionFormatChat)
	require.Len(t, refs, 1)
	assert.Empty(t, refs[0].url, "an image_url of an unknown shape has no usable URL")
	assert.Equal(t, maxVisionQuestionChars, len([]rune(question)))
	assert.True(t, strings.HasSuffix(question, "END"), "the latest text is kept")

	empty, _ := collectVisionImages(map[string]any{}, visionFormatChat)
	assert.Empty(t, empty)
}

func TestVisionImageURL(t *testing.T) {
	cases := []struct {
		name    string
		part    string
		format  visionBodyFormat
		url     string
		isImage bool
	}{
		{"chat object", `{"type":"image_url","image_url":{"url":"u1"}}`, visionFormatChat, "u1", true},
		{"chat string", `{"type":"image_url","image_url":"u2"}`, visionFormatChat, "u2", true},
		{"chat text", `{"type":"text","text":"t"}`, visionFormatChat, "", false},
		{"responses string", `{"type":"input_image","image_url":"u3"}`, visionFormatResponses, "u3", true},
		{"responses object", `{"type":"input_image","image_url":{"url":"u4"}}`, visionFormatResponses, "u4", true},
		{"responses file_id", `{"type":"input_image","file_id":"f"}`, visionFormatResponses, "", true},
		{"responses chat-shaped part", `{"type":"image_url","image_url":"u"}`, visionFormatResponses, "", false},
		{"messages base64", `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}`, visionFormatMessages, "data:image/png;base64,AA", true},
		{"messages base64 without data", `{"type":"image","source":{"type":"base64","media_type":"image/png"}}`, visionFormatMessages, "", true},
		{"messages url", `{"type":"image","source":{"type":"url","url":"https://x"}}`, visionFormatMessages, "https://x", true},
		{"messages file", `{"type":"image","source":{"type":"file","file_id":"f"}}`, visionFormatMessages, "", true},
		{"messages text", `{"type":"text","text":"t"}`, visionFormatMessages, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var part map[string]any
			require.NoError(t, json.Unmarshal([]byte(tc.part), &part))
			url, isImage := visionImageURL(part, tc.format)
			assert.Equal(t, tc.url, url)
			assert.Equal(t, tc.isImage, isImage)
		})
	}
}

func TestVisionItemText(t *testing.T) {
	assert.Equal(t, []string{"a"}, visionItemText("a"))
	assert.Equal(t, []string{"b", "c"}, visionItemText([]any{
		"junk",
		map[string]any{"type": "text", "text": "b"},
		map[string]any{"type": "input_text", "text": "c"},
		map[string]any{"type": "image_url"},
	}))
	assert.Nil(t, visionItemText(nil))
	assert.Nil(t, visionItemText(42))
}

func TestTruncateForLog(t *testing.T) {
	assert.Equal(t, "abc", truncateForLog("abc", 5))
	assert.Equal(t, "ab...", truncateForLog("abcdef", 2))
}

// A model with a global real name (models[].model without a credential): the rewritten
// body keeps the real name, and the proxy body is re-derived with the router name.
func TestVisionFallback_GlobalRealModelName(t *testing.T) {
	u := &visionUpstream{}
	upstream := newVisionUpstream(t, u)
	credential := config.CredentialConfig{Name: "vllm-only", Type: config.ProviderTypeVLLM, BaseURL: upstream.URL, RPM: 100, TPM: 1000000}
	noVision, vision := false, true
	manager := routermodels.New(testhelpers.NewTestLogger(), 100, []config.ModelRPMConfig{
		{Name: "glm-global", Model: "zai/glm-global", RPM: -1, TPM: -1, SupportsVision: &noVision},
		{Name: "qwen-vl", Credential: credential.Name, RPM: -1, TPM: -1, SupportsVision: &vision},
	})
	manager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	manager.SetCredentials([]config.CredentialConfig{credential})
	prices := routermodels.NewModelPriceRegistry()
	prices.MergeDB(map[string]*routermodels.ModelPrice{
		"glm-global": {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
		"qwen-vl":    {InputCostPerToken: 1e-7, OutputCostPerToken: 1e-6},
	})
	builder := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key")
	builder.config.ModelManager = manager
	builder.config.VisionFallback = describeFallback()
	prx := builder.Build()
	prx.LiteLLMDB = newVLLMTestDB()
	prx.priceRegistry = prices

	w := sendVisionRequest(t, prx, "/v1/chat/completions", `{"model":"glm-global","messages":[
		{"role":"user","content":[{"type":"image_url","image_url":{"url":"`+visionTestImage+`"}}]}]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	assert.Equal(t, "described", w.Header().Get(HeaderVisionFallback))
	_, bodies := u.snapshot()
	require.Len(t, bodies, 2)
	assert.Equal(t, "zai/glm-global", bodies[1]["model"])
	assert.Contains(t, partText(t, messageParts(t, bodies[1], 0)[0]), visionTestOwl)

	// The proxy body (sent to AIR/proxy peers) carries the router name.
	base, proxyBody := []byte(`{"model":"zai/glm-global","messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}`), []byte(nil)
	prx.visionFallback.Mode = config.VisionFallbackStrip
	ok := prx.applyVisionFallbackToBase(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		&RequestLogContext{}, &base, &proxyBody, "glm-global", "zai/glm-global", visionFormatChat)
	require.True(t, ok)
	assert.Contains(t, string(base), `"model":"zai/glm-global"`)
	assert.Contains(t, string(proxyBody), `"model":"glm-global"`)
	assert.NotContains(t, string(proxyBody), "image_url")
}

func TestMarshalVisionJSON(t *testing.T) {
	out, err := marshalVisionJSON(map[string]any{"text": "<a & b>"})
	require.NoError(t, err)
	assert.Equal(t, `{"text":"<a & b>"}`, string(out), "no HTML escaping, no trailing newline")

	_, err = marshalVisionJSON(make(chan int))
	assert.Error(t, err)
}
