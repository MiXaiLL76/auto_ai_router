package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const textFormatTestRequest = `{"model":"%s","input":"Who wrote Eugene Onegin?","store":true,` +
	`"text":{"format":{"type":"json_schema","name":"person","strict":true,` +
	`"schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}}`

var textFormatTestFormat = map[string]any{
	"type": "json_schema", "name": "person", "strict": true,
	"schema": map[string]any{
		"type":       "object",
		"properties": map[string]any{"name": map[string]any{"type": "string"}},
		"required":   []any{"name"},
	},
}

// sendTextFormatRequest sends one /v1/responses request through prx and
// returns the decoded client response.
func sendTextFormatRequest(t *testing.T, prx *Proxy, model string) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(fmt.Sprintf(textFormatTestRequest, model)))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	var resp map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	return resp
}

// jsonUpstream starts an upstream that answers every request with body.
func jsonUpstream(t *testing.T, body string) *httptest.Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(upstream.Close)
	return upstream
}

// Claude gets the schema as native Structured Outputs, and the client and the
// stored copy see its own text.format echoed back.
func TestProxyRequest_ResponsesTextFormat_Claude(t *testing.T) {
	upstream, captured := claudeUpstream(t)
	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeAnthropic, upstream.URL, "upstream-key").
		Build()
	store := &terminalEventResponseStore{}
	prx.responseStore = store

	resp := sendTextFormatRequest(t, prx, "claude-haiku-5-5")

	var sent map[string]any
	select {
	case sent = <-captured:
	default:
		t.Fatal("the upstream received no request")
	}
	format := sent["output_config"].(map[string]any)["format"].(map[string]any)
	assert.Equal(t, "json_schema", format["type"])
	assert.NotContains(t, sent, "text")
	assert.Equal(t, map[string]any{"format": textFormatTestFormat}, resp["text"])
	require.Len(t, store.saved, 1)
	assert.Equal(t, map[string]any{"format": textFormatTestFormat}, store.saved[0].Text)
}

// A response the router builds from a Chat Completions reply echoes text.format too.
func TestProxyRequest_ResponsesTextFormat_ConvertedFromChat(t *testing.T) {
	upstream := jsonUpstream(t, `{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"glm-5",
		"choices":[{"index":0,"message":{"role":"assistant","content":"{\"name\":\"Pushkin\"}"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	passthroughResponses := false
	builder := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeOpenAI, upstream.URL, "upstream-key")
	modelManager := models.New(builder.config.Logger, 50, []config.ModelRPMConfig{
		{Name: "glm-5", Credential: "provider", PassthroughResponses: &passthroughResponses},
	})
	modelManager.LoadModelsFromConfig(builder.config.Credentials)
	builder.config.ModelManager = modelManager
	prx := builder.Build()

	resp := sendTextFormatRequest(t, prx, "glm-5")

	assert.Equal(t, "response", resp["object"])
	assert.Equal(t, map[string]any{"format": textFormatTestFormat}, resp["text"])
}

// A provider's own Responses reply already carries text, with the provider's
// defaults (verbosity): passthrough must leave it as the provider sent it.
func TestProxyRequest_ResponsesTextFormat_PassthroughKeepsProviderText(t *testing.T) {
	upstream := jsonUpstream(t, `{"id":"resp_1","object":"response","created_at":1700000000,"model":"gpt-4o","status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant",
			"content":[{"type":"output_text","text":"{\"name\":\"Pushkin\"}","annotations":[]}]}],
		"text":{"format":{"type":"json_schema","name":"person","strict":true,"schema":{"type":"object"}},"verbosity":"medium"},
		"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`)
	prx := NewTestProxyBuilder().
		WithSingleCredential("provider", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		Build()

	resp := sendTextFormatRequest(t, prx, "gpt-4o")

	text := resp["text"].(map[string]any)
	assert.Equal(t, "medium", text["verbosity"])
	assert.Equal(t, map[string]any{"type": "object"}, text["format"].(map[string]any)["schema"])
}
