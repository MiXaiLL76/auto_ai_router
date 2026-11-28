package proxy

import (
	"encoding/json"
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

const (
	chatCompletionReply = `{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"m",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`
	responsesReply = `{"id":"resp_1","object":"response","created_at":1700000000,"model":"m","status":"completed",` +
		`"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant",` +
		`"content":[{"type":"output_text","text":"ok","annotations":[]}]}],` +
		`"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}`
	geminiReply = `{"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":6}}`
)

// pdfUpstream is the request AIR sent upstream.
type pdfUpstream struct {
	path string
	body []byte
}

func newPDFUpstream(t *testing.T, reply string) (<-chan pdfUpstream, string) {
	t.Helper()
	captured := make(chan pdfUpstream, 1)
	server := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- pdfUpstream{path: r.URL.Path, body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(server.Close)
	return captured, server.URL
}

func sendPDFRequest(t *testing.T, prx *Proxy, path, body string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, req)
	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
}

// firstContent returns the content parts of the first message (Chat) or input
// item (Responses) of a captured body.
func firstContent(t *testing.T, raw []byte, key string) []any {
	t.Helper()
	var body map[string]any
	require.NoError(t, json.Unmarshal(raw, &body))
	items, ok := body[key].([]any)
	require.True(t, ok, "%s: %v", key, body)
	require.NotEmpty(t, items)
	content, ok := items[0].(map[string]any)["content"].([]any)
	require.True(t, ok, "content: %v", items[0])
	return content
}

func TestProxyRequest_MessagesPDFToOpenAIChat(t *testing.T) {
	captured, url := newPDFUpstream(t, chatCompletionReply)
	prx := NewTestProxyBuilder().WithSingleCredential("openai", config.ProviderTypeOpenAI, url, "key").Build()

	sendPDFRequest(t, prx, "/v1/messages", `{"model":"gpt-4.1-mini","max_tokens":64,"messages":[{"role":"user","content":[
		{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="},"cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"What is in it?"}]}]}`)
	upstream := <-captured

	assert.Equal(t, "/v1/chat/completions", upstream.path)
	assert.Equal(t, map[string]any{
		"type": "file",
		"file": map[string]any{"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "document.pdf"},
	}, firstContent(t, upstream.body, "messages")[0])
}

func TestProxyRequest_MessagesPDFURLToGemini(t *testing.T) {
	captured, url := newPDFUpstream(t, geminiReply)
	prx := NewTestProxyBuilder().WithSingleCredential("gemini", config.ProviderTypeGemini, url, "key").Build()

	sendPDFRequest(t, prx, "/v1/messages", `{"model":"gemini-3.8-flash","max_tokens":64,"messages":[{"role":"user","content":[
		{"type":"document","source":{"type":"url","url":"https://example.com/report"}},
		{"type":"text","text":"What is in it?"}]}]}`)
	upstream := <-captured

	var body map[string]any
	require.NoError(t, json.Unmarshal(upstream.body, &body))
	contents := body["contents"].([]any)
	parts := contents[0].(map[string]any)["parts"].([]any)
	assert.Equal(t, map[string]any{"mimeType": "application/pdf", "fileUri": "https://example.com/report"},
		parts[0].(map[string]any)["fileData"])
}

func TestProxyRequest_ResponsesOnlyModelChatPDF(t *testing.T) {
	captured, url := newPDFUpstream(t, responsesReply)
	credential := config.CredentialConfig{Name: "openai_main", Type: config.ProviderTypeOpenAI, BaseURL: url, APIKey: "key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).Build()
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-5-pro", ResponsesOnly: true, Credential: "openai_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})

	sendPDFRequest(t, prx, "/v1/chat/completions", `{"model":"gpt-5-pro","messages":[{"role":"user","content":[
		{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}`)
	upstream := <-captured

	assert.Equal(t, "/v1/responses", upstream.path)
	assert.Equal(t, map[string]any{
		"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0=", "filename": "document.pdf",
	}, firstContent(t, upstream.body, "input")[0])
}

func TestProxyRequest_ResponsesConvertedToChatPDF(t *testing.T) {
	captured, url := newPDFUpstream(t, chatCompletionReply)
	credential := config.CredentialConfig{Name: "openai_main", Type: config.ProviderTypeOpenAI, BaseURL: url, APIKey: "key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).Build()
	passthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-4.1-mini", PassthroughResponses: &passthrough, Credential: "openai_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})

	sendPDFRequest(t, prx, "/v1/responses", `{"model":"gpt-4.1-mini","input":[{"role":"user","content":[
		{"type":"input_file","file_data":"data:application/pdf;base64,JVBERi0=","filename":"report.pdf"}]}]}`)
	upstream := <-captured

	assert.Equal(t, "/v1/chat/completions", upstream.path)
	assert.Equal(t, map[string]any{
		"type": "file",
		"file": map[string]any{"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "report.pdf"},
	}, firstContent(t, upstream.body, "messages")[0])
}
