package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeSuccessfulResponseModelPreservesLargeNumbers(t *testing.T) {
	body := []byte(`{"model":"backend","sequence":9007199254740993,"nested":{"value":9223372036854775807}}`)

	normalized := normalizeSuccessfulResponseModel(body, "/v1/responses", "public")

	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(normalized, &fields))
	assert.JSONEq(t, `"public"`, string(fields["model"]))
	assert.Equal(t, "9007199254740993", string(fields["sequence"]))
	assert.Contains(t, string(fields["nested"]), "9223372036854775807")
}

func TestNormalizeSuccessfulResponseModelStreamPreservesLargeNumbers(t *testing.T) {
	stream := "data: {\"model\":\"backend\",\"sequence\":9007199254740993}\n\n"
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{Request: request, PublicModelID: "public"}

	normalized, err := io.ReadAll(normalizeSuccessfulResponseModelStream(
		strings.NewReader(stream),
		http.StatusOK,
		logCtx,
		"backend",
	))

	require.NoError(t, err)
	assert.Contains(t, string(normalized), `"model":"public"`)
	assert.Contains(t, string(normalized), `"sequence":9007199254740993`)
}

func TestNormalizeSuccessfulResponseModelStreamPreservesNestedModelForLiteLLM(t *testing.T) {
	stream := "data: {\"model\":\"backend\",\"response\":{\"model\":\"backend\"}}\n\n"
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	request, _ = withResponseCompatRequest(request)
	logCtx := &RequestLogContext{Request: request, PublicModelID: "public"}

	result, err := io.ReadAll(normalizeSuccessfulResponseModelStream(
		strings.NewReader(stream),
		http.StatusOK,
		logCtx,
		"backend",
	))

	require.NoError(t, err)
	assert.JSONEq(t,
		`{"model":"public","response":{"model":"backend"}}`,
		strings.TrimSpace(strings.TrimPrefix(string(result), "data:")),
	)
}

func TestNormalizeSuccessfulResponseModelStreamBoundsOversizedLines(t *testing.T) {
	stream := "data: {\"model\":\"backend\",\"payload\":\"" +
		strings.Repeat("x", maxSSEModelRewriteLineBytes) +
		"\"}\n\n"
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{Request: request, PublicModelID: "public"}

	result, err := io.ReadAll(normalizeSuccessfulResponseModelStream(
		strings.NewReader(stream),
		http.StatusOK,
		logCtx,
		"backend",
	))

	require.NoError(t, err)
	assert.Equal(t, stream, string(result))
}

func TestNormalizeSuccessfulResponseModelStreamPreservesErrorEventsAndFraming(t *testing.T) {
	stream := "data: {\"model\":\"backend\",\"sequence\":1}\r\n\r\n" +
		"data: {\"error\":{\"message\":\"failed\"},\"model\":\"backend\"}\r\n\r\n" +
		"data: {\"type\":\"response.failed\",\"response\":{\"model\":\"backend\"}}\r\n\r\n" +
		"data: {\"type\":\"response.error\",\"model\":\"backend\"}\r\n\r\n" +
		"data: [DONE]\r\n\r\n"
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	logCtx := &RequestLogContext{Request: request, PublicModelID: "public"}

	result, err := io.ReadAll(normalizeSuccessfulResponseModelStream(
		strings.NewReader(stream),
		http.StatusOK,
		logCtx,
		"backend",
	))

	require.NoError(t, err)
	lines := strings.Split(string(result), "\r\n")
	assert.Contains(t, lines[0], `"model":"public"`)
	assert.Equal(t, `data: {"error":{"message":"failed"},"model":"backend"}`, lines[2])
	assert.Equal(t, `data: {"type":"response.failed","response":{"model":"backend"}}`, lines[4])
	assert.Equal(t, `data: {"type":"response.error","model":"backend"}`, lines[6])
	assert.Equal(t, "data: [DONE]", lines[8])
}

func TestFinalizeStreamingLogKeepsExplicitZeroProviderUsage(t *testing.T) {
	prx := NewTestProxyBuilder().Build()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{
		Request:                request,
		promptTokensEstimateFn: func() int { return 123 },
	}
	lastChunk := []byte(`data: {"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0}}`)

	prx.finalizeStreamingLog(logCtx, 456, lastChunk, "openai", http.StatusOK, false)

	require.NotNil(t, logCtx.TokenUsage)
	assert.Zero(t, logCtx.TokenUsage.PromptTokens)
	assert.Zero(t, logCtx.TokenUsage.CompletionTokens)
	assert.Equal(t, "provider", logCtx.UsageSource)
}
