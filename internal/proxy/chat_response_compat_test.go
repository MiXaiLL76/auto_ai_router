package proxy

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
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

func TestNormalizeSuccessfulResponseModelStreamRewritesImageSizedLines(t *testing.T) {
	imageURL := "data:image/png;base64," + strings.Repeat("x", 2*1024*1024)
	stream := "data: {\"model\":\"canonical\",\"choices\":[{\"delta\":{\"images\":[{\"image_url\":{\"url\":" +
		strconv.Quote(imageURL) + "}}]}}]}\n\n"
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{Request: request, PublicModelID: "google/gemini-3-pro-image-preview"}

	result, err := io.ReadAll(normalizeSuccessfulResponseModelStream(
		strings.NewReader(stream),
		http.StatusOK,
		logCtx,
		"canonical",
	))

	require.NoError(t, err)
	line := strings.TrimSpace(strings.TrimPrefix(string(result), "data:"))
	var payload struct {
		Model   string `json:"model"`
		Choices []struct {
			Delta struct {
				Images []struct {
					ImageURL struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"images"`
			} `json:"delta"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal([]byte(line), &payload))
	assert.Equal(t, "google/gemini-3-pro-image-preview", payload.Model)
	require.Len(t, payload.Choices, 1)
	require.Len(t, payload.Choices[0].Delta.Images, 1)
	assert.Equal(t, imageURL, payload.Choices[0].Delta.Images[0].ImageURL.URL)
}

func TestResponseModelStreamReaderBoundsOversizedLines(t *testing.T) {
	stream := "data: {\"model\":\"backend\",\"payload\":\"" + strings.Repeat("x", 2048) + "\"}\n\n"
	reader := &responseModelStreamReader{
		source:       bufio.NewReader(strings.NewReader(stream)),
		publicModel:  "public",
		maxLineBytes: 1024,
	}

	result, err := io.ReadAll(reader)

	require.NoError(t, err)
	assert.Equal(t, stream, string(result))
}

func TestNormalizeSuccessfulResponseModel_RewritesExistingImageModelOnly(t *testing.T) {
	withModel := []byte(`{"created":1,"data":[],"model":"canonical"}`)
	withoutModel := []byte(`{"created":1,"data":[]}`)

	assert.JSONEq(t,
		`{"created":1,"data":[],"model":"google/gemini-3-pro-image-preview"}`,
		string(normalizeSuccessfulResponseModel(withModel, "/v1/images/generations", "google/gemini-3-pro-image-preview")),
	)
	assert.Equal(t, withoutModel,
		normalizeSuccessfulResponseModel(withoutModel, "/v1/images/generations", "openai/gpt-image-1"),
	)
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
