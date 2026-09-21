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

// responsesOnlyUpstreamTransport simulates an OpenAI-compatible upstream that
// only accepts /v1/responses (a genuine real-world shape for some OpenAI
// reasoning-tier deployments): it asserts the request AIR actually sent was
// converted to Responses API shape, and replies with a Responses API
// response body.
type responsesOnlyUpstreamTransport struct {
	t *testing.T
}

func (tr responsesOnlyUpstreamTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	assert.Equal(tr.t, "/v1/responses", r.URL.Path,
		"AIR must call the provider's native /v1/responses, not /v1/chat/completions, for a responses_only model")

	body, err := io.ReadAll(r.Body)
	require.NoError(tr.t, err)

	var reqBody map[string]interface{}
	require.NoError(tr.t, json.Unmarshal(body, &reqBody))
	assert.Contains(tr.t, reqBody, "input", "request body must be Responses-API-shaped")
	assert.NotContains(tr.t, reqBody, "messages", "messages must not leak into a Responses API request")
	assert.Equal(tr.t, float64(64), reqBody["max_output_tokens"])

	respBody := `{
		"id":"resp_test1","object":"response","created_at":1700000000,"model":"gpt-5-pro","status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","role":"assistant",
			"content":[{"type":"output_text","text":"4","annotations":[]}]}],
		"usage":{"input_tokens":9,"output_tokens":1,"total_tokens":10}
	}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(respBody)),
		Request:    r,
	}, nil
}

func TestProxyRequest_ResponsesOnlyModel_ChatCompletionsEndpoint(t *testing.T) {
	credential := config.CredentialConfig{Name: "openai_main", Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com", APIKey: "provider-key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).Build()
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-5-pro", ResponsesOnly: true, Credential: "openai_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx.client.Transport = responsesOnlyUpstreamTransport{t: t}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-pro","max_tokens":64,"messages":[{"role":"user","content":"2+2?"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "chat.completion", resp["object"])
	choices := resp["choices"].([]interface{})
	require.Len(t, choices, 1)
	choice := choices[0].(map[string]interface{})
	assert.Equal(t, "stop", choice["finish_reason"])
	message := choice["message"].(map[string]interface{})
	assert.Equal(t, "4", message["content"])
	assert.Equal(t, "assistant", message["role"])
}
