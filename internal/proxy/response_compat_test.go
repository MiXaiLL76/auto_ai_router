package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponseCompatibilityWriterTransformsInternalError(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.RequestID = "request-1"
	info.RequestedModel = "public-model"

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusBadRequest)
	_, err := writer.Write([]byte(`{"error":{"message":"bad request","type":"invalid_request_error","param":null,"code":null}}`))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "request-1", recorder.Header().Get("x-litellm-call-id"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
	assert.Equal(t, "400", body["error"].(map[string]any)["code"])
}

func TestResponseCompatibilityWriterTransformsStream(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.RequestedModel = "public-model"

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.WriteHeader(http.StatusOK)
	_, err := writer.Write([]byte("data: {\"model\":\"provider-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), `"model":"public-model"`)
	assert.Equal(t, 1, strings.Count(recorder.Body.String(), "data: [DONE]"))
}

func TestProxyRequestLiteLLMCompatibility(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Provider-Request-Id", "provider-1")
		_, _ = w.Write([]byte(`{
			"id":"upstream-id",
			"created":1,
			"model":"provider-model",
			"choices":[{"index":7,"message":{"role":null,"content":"hello"},"finish_reason":null}]
		}`))
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", "openai", upstream.URL, "upstream-key").
		Build()
	prx.responseCompat = compatlitellm.New()

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public-model","messages":[{"role":"user","content":"hello"}]}`,
	))
	request.Header.Set("Authorization", "Bearer master-key")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, request)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "provider-1", recorder.Header().Get("llm_provider-x-provider-request-id"))
	assert.NotEmpty(t, recorder.Header().Get("x-litellm-call-id"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
	assert.Equal(t, "public-model", body["model"])
	choice := body["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, float64(0), choice["index"])
	assert.Equal(t, "stop", choice["finish_reason"])
}
