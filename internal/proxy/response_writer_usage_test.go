package proxy

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriteProxyResponseNormalizesQwenUsageBeforeCompression(t *testing.T) {
	originalBody := []byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1370,"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417,"provider_detail":"kept"}},"provider_metadata":{"trace":"kept"}}`)
	resp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":   []string{"application/json"},
			"ETag":           []string{`"upstream-body"`},
			"Content-Digest": []string{"sha-256=:invalid-after-normalization=:"},
			"X-Provider":     []string{"kept"},
		},
		Body: append([]byte(nil), originalBody...),
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Accept-Encoding", "gzip")
	w := httptest.NewRecorder()

	NewTestProxyBuilder().Build().writeProxyResponse(w, resp, req, &config.CredentialConfig{Name: "test"}, "qwen/qwen3.6-35b-a3b")

	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, "gzip", w.Header().Get("Content-Encoding"))
	assert.Equal(t, strconv.Itoa(w.Body.Len()), w.Header().Get("Content-Length"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Empty(t, w.Header().Get("Content-Digest"))
	assert.Equal(t, "kept", w.Header().Get("X-Provider"))
	assert.Equal(t, originalBody, resp.Body, "diagnostic upstream body must remain unchanged")

	zr, err := gzip.NewReader(w.Body)
	require.NoError(t, err)
	decoded, err := io.ReadAll(zr)
	require.NoError(t, err)
	require.NoError(t, zr.Close())

	var payload struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens        int `json:"completion_tokens"`
			CompletionTokensDetails struct {
				TextTokens      int    `json:"text_tokens"`
				ReasoningTokens int    `json:"reasoning_tokens"`
				ProviderDetail  string `json:"provider_detail"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
		ProviderMetadata struct {
			Trace string `json:"trace"`
		} `json:"provider_metadata"`
	}
	require.NoError(t, json.Unmarshal(decoded, &payload))
	assert.Equal(t, "chatcmpl-1", payload.ID)
	require.Len(t, payload.Choices, 1)
	assert.Equal(t, "ok", payload.Choices[0].Message.Content)
	assert.Equal(t, 4774, payload.Usage.CompletionTokens)
	assert.Equal(t, 357, payload.Usage.CompletionTokensDetails.TextTokens)
	assert.Equal(t, 4417, payload.Usage.CompletionTokensDetails.ReasoningTokens)
	assert.Equal(t, "kept", payload.Usage.CompletionTokensDetails.ProviderDetail)
	assert.Equal(t, "kept", payload.ProviderMetadata.Trace)
}

func TestWriteProxyResponseDoesNotNormalizeError(t *testing.T) {
	originalBody := []byte(`{"error":{"message":"upstream failed"},"usage":{"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417}}}`)
	resp := &ProxyResponse{
		StatusCode: http.StatusBadGateway,
		Headers:    http.Header{"ETag": []string{`"unchanged-error-body"`}},
		Body:       originalBody,
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	NewTestProxyBuilder().Build().writeProxyResponse(w, resp, req, &config.CredentialConfig{Name: "test"}, "qwen3.6-35b-a3b")

	assert.Equal(t, http.StatusBadGateway, w.Code)
	assert.Equal(t, originalBody, w.Body.Bytes())
	assert.Equal(t, `"unchanged-error-body"`, w.Header().Get("ETag"))
}

func TestWriteProxyResponseProManMasksErrorAndStripsHeaders(t *testing.T) {
	rawBody := []byte(`{"error":{"message":"litellm.BadRequestError: Received Model Group=anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a Available Model Group Fallbacks=None"}}`)
	resp := &ProxyResponse{
		StatusCode: http.StatusBadRequest,
		Headers: http.Header{
			"Content-Type":      []string{"application/json"},
			"X-Litellm-Version": []string{"1.92.0"},
			"Llm_provider-Base": []string{"anthropic-direct"},
			"Server":            []string{"uvicorn"},
			"ETag":              []string{`"raw-error"`},
		},
		Body: rawBody,
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	cred := &config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan}

	NewTestProxyBuilder().Build().writeProxyResponse(w, resp, req, cred, "claude-haiku-4.5")

	require.Equal(t, http.StatusBadRequest, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	assert.Empty(t, w.Header().Get("X-Litellm-Version"))
	assert.Empty(t, w.Header().Get("Llm_provider-Base"))
	assert.Empty(t, w.Header().Get("Server"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.NotContains(t, w.Body.String(), "litellm")
	assert.NotContains(t, w.Body.String(), "anthropic-direct-client")
	assert.Contains(t, w.Body.String(), "Upstream provider error")
	assert.Equal(t, rawBody, resp.Body, "raw upstream body should remain available to internal logging")
}

func TestWriteProxyResponseProManSanitizesSuccessBody(t *testing.T) {
	resp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":      []string{"application/json"},
			"X-Litellm-Version": []string{"1.92.0"},
			"ETag":              []string{`"raw-success"`},
		},
		Body: []byte(`{"id":"chatcmpl-1","model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a","provider_specific_fields":{"trace":"hidden"},"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	cred := &config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan}

	NewTestProxyBuilder().Build().writeProxyResponse(w, resp, req, cred, "claude-haiku-4.5")

	require.Equal(t, http.StatusOK, w.Code)
	assert.Empty(t, w.Header().Get("X-Litellm-Version"))
	assert.Empty(t, w.Header().Get("ETag"))
	assert.Contains(t, w.Body.String(), `"model":"claude-haiku-4.5"`)
	assert.NotContains(t, w.Body.String(), "provider_specific_fields")
	assert.NotContains(t, w.Body.String(), "anthropic-direct-client")
}

func TestSanitizeUpstreamJSONBodyRemovesLiteLLMSurface(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-1",
		"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a",
		"provider_specific_fields":{"trace":"hidden"},
		"caller":"litellm",
		"choices":[{"message":{"content":"ok"},"provider_specific_fields":{"router":"hidden"}}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)

	sanitized, changed := sanitizeUpstreamJSONBody(raw, "claude-haiku-4.5")

	require.True(t, changed)
	body := string(sanitized)
	assert.NotContains(t, body, "provider_specific_fields")
	assert.NotContains(t, body, "caller")
	assert.NotContains(t, body, "anthropic-direct-client")
	assert.NotContains(t, body, "litellm")

	var decoded map[string]any
	require.NoError(t, json.Unmarshal(sanitized, &decoded))
	assert.Equal(t, "claude-haiku-4.5", decoded["model"])
	assert.Contains(t, body, `"usage"`)
}

func TestClientResponseBodyForProManMasksErrors(t *testing.T) {
	cred := &config.CredentialConfig{Name: "proman", Type: config.ProviderTypeProMan}
	raw := []byte(`{"error":{"message":"litellm.BadRequestError: Received Model Group=anthropic/claude/anthropic-direct-client-0dce8b1a Available Model Group Fallbacks=None"}}`)

	body, changed := clientResponseBodyForCredential(400, raw, cred, "claude-haiku-4.5")

	require.True(t, changed)
	assert.NotContains(t, string(body), "litellm")
	assert.NotContains(t, string(body), "anthropic-direct-client")
	assert.NotContains(t, string(body), "Model Group")
	assert.Contains(t, string(body), "Upstream provider error")
}

func TestSanitizingSSEReadCloserRemovesInternalFields(t *testing.T) {
	stream := strings.Join([]string{
		"event: response.output_text.delta",
		`data: {"type":"response.completed","response":{"model":"anthropic/claude-haiku-4-5-20251001/anthropic-direct-client-0dce8b1a","provider_specific_fields":{"trace":"hidden"},"output":[]}}`,
		`data: {"type":"error","error":{"type":"invalid_request_error","message":"litellm.BadRequestError: Received Model Group=anthropic/claude/anthropic-direct-client-0dce8b1a Available Model Group Fallbacks=None"}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	rc := newSanitizingSSEReadCloser(io.NopCloser(strings.NewReader(stream)), "claude-haiku-4.5")

	out, err := io.ReadAll(rc)
	require.NoError(t, err)

	assert.Contains(t, string(out), "event: response.output_text.delta")
	assert.Contains(t, string(out), `"model":"claude-haiku-4.5"`)
	assert.Contains(t, string(out), "Upstream provider error")
	assert.NotContains(t, string(out), "provider_specific_fields")
	assert.NotContains(t, string(out), "anthropic-direct-client")
	assert.NotContains(t, string(out), "Model Group")
	assert.Contains(t, string(out), "data: [DONE]")
}

func TestWriteProxyStreamingResponseNormalizesQwenUsage(t *testing.T) {
	stream := "event: message\r\n" +
		`data: {"choices":[],"usage":{"prompt_tokens":1370,"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417,"provider_detail":"kept"}}}` + "\r\n\r\n" +
		"data: [DONE]\r\n\r\n"
	resp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type": []string{"text/event-stream"},
			"Digest":       []string{"sha-256=:invalid-after-normalization=:"},
		},
		StreamBody:  io.NopCloser(strings.NewReader(stream)),
		IsStreaming: true,
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	usage, err := NewTestProxyBuilder().Build().writeProxyStreamingResponseWithTokens(
		w,
		resp,
		req,
		&config.CredentialConfig{Name: "test"},
		"gateway/qwen3.6-35b-a3b-20260415",
		"qwen3.6-35b-a3b",
		nil,
	)

	require.NoError(t, err)
	require.NotNil(t, usage)
	assert.Empty(t, w.Header().Get("Digest"))
	assert.Equal(t, 1370, usage.PromptTokens)
	assert.Equal(t, 4774, usage.CompletionTokens)
	assert.Equal(t, 4417, usage.ReasoningTokens)
	assert.Contains(t, w.Body.String(), "event: message\r\n")
	assert.Contains(t, w.Body.String(), `"text_tokens":357`)
	assert.Contains(t, w.Body.String(), `"reasoning_tokens":4417`)
	assert.Contains(t, w.Body.String(), `"provider_detail":"kept"`)
	assert.Contains(t, w.Body.String(), "data: [DONE]\r\n\r\n")
}

func TestWriteProxyStreamingResponseQwenDrainCapturesUsage(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		chunks := []string{
			`data: {"choices":[{"delta":{"content":"answer"}}]}` + "\n\n",
			`data: {"choices":[],"usage":{"prompt_tokens":1370,"completion_tokens":4774,"completion_tokens_details":{"text_tokens":4774,"reasoning_tokens":4417}}}` + "\n\n",
			"data: [DONE]\n\n",
		}
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "key1").
		WithRequestTimeout(5 * time.Second).
		WithDrainUpstreamOnAbort(true).
		Build()
	upstreamResp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = upstreamResp.Body.Close() }()

	proxyResp := &ProxyResponse{
		StatusCode:  http.StatusOK,
		Headers:     upstreamResp.Header,
		StreamBody:  upstreamResp.Body,
		IsStreaming: true,
	}
	w := newFailAfterNBytesWriter(10)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	usage, err := prx.writeProxyStreamingResponseWithTokens(w, proxyResp, req, &config.CredentialConfig{Name: "test"}, "qwen3.6-35b-a3b", "qwen3.6-35b-a3b", nil)

	require.Error(t, err)
	require.NotNil(t, usage)
	assert.Equal(t, 1370, usage.PromptTokens)
	assert.Equal(t, 4774, usage.CompletionTokens)
	assert.Equal(t, 4417, usage.ReasoningTokens)
}
