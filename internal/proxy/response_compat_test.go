package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
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
	assert.Empty(t, recorder.Header().Get("x-litellm-call-id"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
	// A 400 now goes through the same signal-based classifier as native mode
	// (see upstreamerror.ClassifyBadRequest); this body's text doesn't match
	// any known pattern, so it lands on the same generic default native mode
	// would also produce for it — "invalid_request", not the old hardcoded
	// "400" — and the raw "bad request" text must never leak into the
	// client-facing message either way.
	assert.Equal(t, "invalid_request", body["error"].(map[string]any)["code"])
	assert.Equal(t, "Invalid request", body["error"].(map[string]any)["message"])
	assert.NotContains(t, recorder.Body.String(), "bad request")
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

func TestResponseCompatibilityWriterDetectsStreamWithWrongContentType(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.RequestedModel = "anthropic/claude-opus-4.6"
	info.Streaming = true

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusOK)
	_, err := writer.Write([]byte("data: {\"model\":\"provider-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "text/event-stream", recorder.Header().Get("Content-Type"))
	assert.Contains(t, recorder.Body.String(), `"model":"anthropic/claude-opus-4.6"`)
	assert.NotContains(t, recorder.Body.String(), "Unable to get json response")
	assert.Equal(t, 1, strings.Count(recorder.Body.String(), "data: [DONE]"))
}

func TestLiteLLMRequestCompatibilityAddsEmbeddingEncodingFormat(t *testing.T) {
	body := applyLiteLLMRequestCompatibility(
		"/v1/embeddings",
		[]byte(`{"input":"AIR shadow fixture","model":"text-embedding-3-small"}`),
	)

	var request map[string]any
	require.NoError(t, json.Unmarshal(body, &request))
	assert.Equal(t, "float", request["encoding_format"])

	explicit := []byte(`{"input":"fixture","model":"text-embedding-3-small","encoding_format":"base64"}`)
	assert.Equal(t, explicit, applyLiteLLMRequestCompatibility("/v1/embeddings", explicit))
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
		http.StatusOK, nil, nil, nil,
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
	assert.Empty(t, recorder.Header().Get("llm_provider-x-provider-request-id"))
	assert.Empty(t, recorder.Header().Get("x-litellm-call-id"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
	assert.Equal(t, "public-model", body["model"])
	choice := body["choices"].([]any)[0].(map[string]any)
	assert.Equal(t, float64(0), choice["index"])
	assert.Equal(t, "stop", choice["finish_reason"])
}

func TestProxyRequestLiteLLMStreamingCompatibility(t *testing.T) {
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		frames := []string{
			`data: {"id":"","created":0,"model":"provider-model","object":"","choices":[],"prompt_filter_results":[{"prompt_index":0}]}` + "\n\n",
			`data: {"id":"provider-id","created":10,"model":"provider-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"","refusal":null,"role":"assistant"},"finish_reason":null,"logprobs":null}],"usage":null}` + "\n\n",
			`data: {"id":"provider-id","created":10,"model":"provider-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"stream"},"finish_reason":null,"logprobs":null}],"usage":null}` + "\n\n",
			`data: {"id":"provider-id","created":10,"model":"provider-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"-ok"},"finish_reason":null,"logprobs":null}],"usage":null}` + "\n\n",
			`data: {"id":"provider-id","created":10,"model":"provider-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}],"usage":null}` + "\n\n",
			`data: {"id":"","created":10,"model":"public-model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{}}],"usage":{"completion_tokens":0,"prompt_tokens":0,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":0}}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, frame := range frames {
			_, _ = io.WriteString(w, frame)
			w.(http.Flusher).Flush()
		}
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("openai", "openai", upstream.URL, "upstream-key").
		Build()
	prx.responseCompat = compatlitellm.New()

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"public-model","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}`,
	))
	request.Header.Set("Authorization", "Bearer master-key")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	prx.ProxyRequest(recorder, request)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), `"content":"stream"`)
	assert.Contains(t, recorder.Body.String(), `"content":"-ok"`)
	assert.Contains(t, recorder.Body.String(), `"finish_reason":"stop"`)
}

// The compatibility writer snapshots the target headers in its constructor and
// hands everything else to the LiteLLM transformer as "provider" headers.
// The cost headers are computed only after the request finishes, so they are not
// in that snapshot: TransformHeaders used to drop them, and a client served with
// response_compatibility: litellm never received X-Total-Cost. They must be
// preserved verbatim, while a provider's own header is still filtered.
func TestResponseCompatibilityWriterPreservesCostHeaders(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, _ = withResponseCompatRequest(request)

	recorder := httptest.NewRecorder()
	recorder.Header().Set("Content-Security-Policy", "frame-ancestors 'none'")
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)

	// Provider headers land in the writer's header map after construction
	// (copyResponseHeaders writes to Header()).
	writer.Header().Set("X-Provider-Request-Id", "provider-1")
	// Cost headers, set after the constructor snapshot.
	writer.Header().Set(HeaderResponseCost, "0.0002")
	writer.Header().Set(HeaderCostBreakdown, `{"input_cost":0.0001,"total_cost":0.0002}`)
	writer.Header().Set("Content-Type", "application/json")

	_, err := writer.Write([]byte(`{"model":"provider-model","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}`))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Equal(t, "0.0002", recorder.Header().Get(HeaderResponseCost))
	assert.Equal(t, `{"input_cost":0.0001,"total_cost":0.0002}`, recorder.Header().Get(HeaderCostBreakdown))
	assert.Equal(t, "frame-ancestors 'none'", recorder.Header().Get("Content-Security-Policy"))
	assert.Empty(t, recorder.Header().Get("X-Provider-Request-Id"),
		"provider headers must still be filtered in compatibility mode")
	assert.Empty(t, recorder.Header().Get("llm_provider-x-provider-request-id"))
}

// The streaming path splits headers too. A plain stream's headers leave before
// the cost is known, so the cost headers are absent there by design; the test
// pins that splitHeaders leaves provider headers filtered on this path as well.
func TestResponseCompatibilityWriterPreservesCostHeadersOnStream(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.Streaming = true

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("X-Provider-Request-Id", "provider-1")
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.WriteHeader(http.StatusOK)

	_, err := writer.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	assert.Empty(t, recorder.Header().Get("X-Provider-Request-Id"))
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

// TestResponseCompatibilityWriterPreservesContentPolicyError pins that a
// content-policy 400 -- the body the vertex prompt-block fix makes the router
// produce -- survives litellm-compat mode with both its message and its code
// intact. Compat mode re-runs ClassifyBadRequest on the router's own body
// (normalizeError), so this also proves that classification round-trips rather
// than degrading to the generic "Invalid request"/"invalid_request".
func TestResponseCompatibilityWriterPreservesContentPolicyError(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request, info := withResponseCompatRequest(request)
	info.RequestID = "request-content-policy"

	recorder := httptest.NewRecorder()
	writer := newResponseCompatibilityWriter(recorder, compatlitellm.New(), request)
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(http.StatusBadRequest)
	_, err := writer.Write([]byte(`{"error":{"message":"The prompt was blocked by the provider's content policy","type":"invalid_request_error","param":null,"code":"content_policy_violation"},"request_id":"abc123"}`))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	require.Equal(t, http.StatusBadRequest, recorder.Code)
	var body map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &body))
	errObj := body["error"].(map[string]any)
	assert.Equal(t, "content_policy_violation", errObj["code"])
	assert.Equal(t, "The prompt was blocked by the provider's content policy", errObj["message"])
	assert.Equal(t, "invalid_request_error", errObj["type"])
}

// TestProxyRequest_CompatMode_PromptBlockReturns400 is the end-to-end guard for
// the production configuration (response_compatibility: litellm): a vertex
// prompt block must reach the client as a 400 content-policy error, decided on
// the first credential, with the compat wrapper in the path.
func TestProxyRequest_CompatMode_PromptBlockReturns400(t *testing.T) {
	var hits atomic.Int32
	blocked := `data: {"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"},"usageMetadata":{"promptTokenCount":6,"totalTokenCount":6}}` + "\n\n"
	upstream := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, blocked+"data: [DONE]\n\n")
	}))
	defer upstream.Close()

	prx := NewTestProxyBuilder().WithCredentials(
		config.CredentialConfig{Name: "g1", Type: config.ProviderTypeGemini, BaseURL: upstream.URL, APIKey: "k1", RPM: 100, TPM: 100000},
		config.CredentialConfig{Name: "g2", Type: config.ProviderTypeGemini, BaseURL: upstream.URL, APIKey: "k2", Priority: 1, RPM: 100, TPM: 100000},
	).WithForceStreamForTTFT(true).WithMaxProviderRetries(5).Build()
	prx.responseCompat = compatlitellm.New()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"blocked"}],"stream":false}`))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	prx.WithResponseCompatibility(w, req, prx.proxyRequest)

	require.Equal(t, http.StatusBadRequest, w.Code, "body: %s", w.Body.String())
	assert.Equal(t, int32(1), hits.Load(), "a content-policy refusal must not be retried across credentials")
	assert.Contains(t, w.Body.String(), "content_policy_violation")
	assert.Contains(t, w.Body.String(), "content policy")
}
