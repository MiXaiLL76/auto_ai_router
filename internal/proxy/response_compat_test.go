package proxy

import (
	"context"
	"encoding/json"
	"errors"
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

func TestResponseCompatibilityWriterDoesNotCompleteFailedStream(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.RequestedModel = "public-model"

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.WriteHeader(http.StatusOK)

	streamErr := errors.New("upstream stream failed")
	reader := &failingStreamReader{
		body: []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n"),
		err:  streamErr,
	}
	prx := NewTestProxyBuilder().Build()
	err := prx.streamToClient(
		context.Background(),
		writer,
		reader,
		"credential",
		"model",
		"/v1/chat/completions",
		nil,
		nil,
		nil,
	)
	require.ErrorIs(t, err, streamErr)
	require.ErrorIs(t, writer.Close(), streamErr)
	assert.Contains(t, recorder.Body.String(), "partial")
	assert.NotContains(t, recorder.Body.String(), "finish_reason")
	assert.NotContains(t, recorder.Body.String(), "data: [DONE]")
}

func TestResponseCompatibilityWriterPreservesInitialHeaders(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, _ = withResponseCompatRequest(request)

	recorder := httptest.NewRecorder()
	recorder.Header().Set("Content-Security-Policy", "frame-ancestors 'none'")
	recorder.Header().Set("X-Frame-Options", "DENY")
	recorder.Header().Set("X-Content-Type-Options", "nosniff")
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "application/json")
	_, err := writer.Write([]byte(`{
		"model":"provider-model",
		"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]
	}`))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Equal(t, "frame-ancestors 'none'", recorder.Header().Get("Content-Security-Policy"))
	assert.Equal(t, "DENY", recorder.Header().Get("X-Frame-Options"))
	assert.Equal(t, "nosniff", recorder.Header().Get("X-Content-Type-Options"))
	assert.Empty(t, recorder.Header().Get("llm_provider-content-security-policy"))
	assert.Empty(t, recorder.Header().Get("llm_provider-x-frame-options"))
	assert.Empty(t, recorder.Header().Get("llm_provider-x-content-type-options"))
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

type failingStreamReader struct {
	body []byte
	err  error
	read bool
}

func (r *failingStreamReader) Read(target []byte) (int, error) {
	if r.read {
		return 0, r.err
	}
	r.read = true
	return copy(target, r.body), r.err
}
