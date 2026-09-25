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

// failedStatusResponsesTransport simulates the Responses API's async-style error shape:
// an outer HTTP 200 whose body carries "status":"failed" and an embedded error, no
// output items at all.
type failedStatusResponsesTransport struct{}

func (failedStatusResponsesTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	respBody := `{
		"id":"resp_failed1","object":"response","created_at":1700000000,"model":"gpt-5-pro",
		"status":"failed","error":{"type":"server_error","message":"something broke upstream"},
		"output":[],
		"usage":{"input_tokens":9,"output_tokens":0,"total_tokens":9}
	}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(respBody)),
		Request:    r,
	}, nil
}

// TestProxyRequest_ResponsesOnlyModel_FailedStatusBecomesErrorResponse covers review
// finding #7: response_to_chat.go's status=="failed" fallback (surfacing the embedded
// error as message content when there are no output items) looked like it might never
// actually run, since statusCodeFromProviderBodyError already remaps an outer-200
// status:"failed" body to a real error status (proxy.go, before ResponseToChat's own
// conversion runs) on every path that calls it. This locks in the actually-observed,
// end-to-end behavior through the full responses_only pipeline: the client must get a
// masked JSON error response at the remapped status (500 for a "server_error" signal),
// never a fake chat.completion "success" with the failure text stuffed into content.
func TestProxyRequest_ResponsesOnlyModel_FailedStatusBecomesErrorResponse(t *testing.T) {
	credential := config.CredentialConfig{Name: "openai_main", Type: config.ProviderTypeOpenAI, BaseURL: "https://api.openai.com", APIKey: "provider-key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).Build()
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{Name: "gpt-5-pro", ResponsesOnly: true, Credential: "openai_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx.client.Transport = failedStatusResponsesTransport{}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-5-pro","max_tokens":64,"messages":[{"role":"user","content":"2+2?"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusInternalServerError, w.Code, "body: %s", w.Body.String())
	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	_, isChatCompletion := resp["object"]
	assert.False(t, isChatCompletion, "must be a masked error body, never a fake chat.completion success")
	assert.Contains(t, resp, "error")
}

// nonOpenAIWithResponsesOnlyTransport simulates an Anthropic-wire upstream. If AIR ever
// sent a Responses-shaped body here (the bug responses_only's provider-type guard
// exists to prevent), the transport itself would fail the request: real Anthropic
// expects "system"/native Messages fields, not Responses API's "input"/
// "max_output_tokens".
type nonOpenAIWithResponsesOnlyTransport struct {
	t *testing.T
}

func (tr nonOpenAIWithResponsesOnlyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	assert.Equal(tr.t, "/v1/messages", r.URL.Path,
		"a non-OpenAI/vLLM credential must never be routed to /v1/responses, regardless of responses_only")

	body, err := io.ReadAll(r.Body)
	require.NoError(tr.t, err)
	var reqBody map[string]interface{}
	require.NoError(tr.t, json.Unmarshal(body, &reqBody))
	assert.NotContains(tr.t, reqBody, "input", "must not be Responses-API-shaped")
	assert.Contains(tr.t, reqBody, "messages", "must still be Anthropic Messages API-shaped")

	respBody := `{
		"id":"msg_test1","type":"message","role":"assistant","model":"claude-opus-4-5",
		"content":[{"type":"text","text":"4"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":9,"output_tokens":1}
	}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(respBody)),
		Request:    r,
	}, nil
}

// TestProxyRequest_ResponsesOnlyModel_NonOpenAICredentialIgnoresFlag covers the review
// finding that responses_only was only ever gated on !cred.IsProxyLike(): a model
// misconfigured (or a DB model_info.mode:"responses" bound) with responses_only:true on
// a non-OpenAI/vLLM credential must be served through the credential's normal
// conversion path, not sent as a Responses-shaped body to a converter/URL builder that
// doesn't understand it.
func TestProxyRequest_ResponsesOnlyModel_NonOpenAICredentialIgnoresFlag(t *testing.T) {
	credential := config.CredentialConfig{Name: "anthropic_main", Type: config.ProviderTypeAnthropic, BaseURL: "https://api.anthropic.com", APIKey: "provider-key", RPM: -1, TPM: -1}
	prx := NewTestProxyBuilder().WithCredentials(credential).Build()
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		// Misconfigured on purpose: responses_only on an Anthropic credential.
		{Name: "claude-opus-4.5", ResponsesOnly: true, Credential: "anthropic_main", RPM: -1, TPM: -1},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
	prx.client.Transport = nonOpenAIWithResponsesOnlyTransport{t: t}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"claude-opus-4.5","max_tokens":64,"messages":[{"role":"user","content":"2+2?"}]}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	prx.ProxyRequest(w, req)

	require.Equal(t, http.StatusOK, w.Code, "body: %s", w.Body.String())
	var resp map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "chat.completion", resp["object"])
}
