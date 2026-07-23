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
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProxyRequest_OpenAIPassthroughResponseUsesAliasModel(t *testing.T) {
	const (
		aliasModel = "gpt-5.2-chat"
		realModel  = "gpt-chat-latest"
	)

	var upstreamBody []byte
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		upstreamBody, err = io.ReadAll(r.Body)
		require.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-test",
			"object":"chat.completion",
			"created":1784570000,
			"model":"gpt-chat-latest",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":25869,"completion_tokens":318,"total_tokens":26187}
		}`))
	}))
	defer upstream.Close()

	mm := models.New(testhelpers.NewTestLogger(), 50, []config.ModelRPMConfig{
		{Name: aliasModel, Model: realModel},
	})
	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").
		WithMasterKey("master-key").
		Build()
	prx.modelManager = mm

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-5.2-chat",
		"messages":[{"role":"user","content":"hello"}]
	}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code)

	var sent map[string]any
	require.NoError(t, json.Unmarshal(upstreamBody, &sent))
	assert.Equal(t, realModel, sent["model"], "upstream must receive the provider-facing model")

	var got map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	assert.Equal(t, aliasModel, got["model"], "client/LiteLLM must see the billed alias model")
}

func TestOpenAIModelAliasReader_RewritesSSEModelLines(t *testing.T) {
	stream := strings.NewReader(
		"data: {\"id\":\"chatcmpl-test\",\"model\":\"gpt-chat-latest\",\"choices\":[]}\n\n" +
			"data: {\"id\":\"chatcmpl-test\",\"model\": \"gpt-chat-latest\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n" +
			"data: [DONE]\n\n",
	)

	out, err := io.ReadAll(newOpenAIModelAliasReader(stream, "gpt-chat-latest", "gpt-5.2-chat"))
	require.NoError(t, err)

	body := string(out)
	assert.Contains(t, body, `"model":"gpt-5.2-chat"`)
	assert.Contains(t, body, `"model": "gpt-5.2-chat"`)
	assert.NotContains(t, body, `"model":"gpt-chat-latest"`)
	assert.Contains(t, body, "data: [DONE]")
}
