package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/require"
)

func TestStreamingUsageInLargeImageFrame(t *testing.T) {
	prx := NewTestProxyBuilder().WithSingleCredential("test", config.ProviderTypeProxy, "http://unused", "unused").Build()
	image := "data:image/png;base64," + strings.Repeat("A", 2*1024*1024)
	body := `data: {"choices":[{"delta":{"images":[{"image_url":{"url":"` + image + `"}}]}}],"usage":{"prompt_tokens":14,"completion_tokens":1301,"total_tokens":1315,"completion_tokens_details":{"image_tokens":1290}}}` + "\n\ndata: [DONE]\n\n"
	resp := &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(body))}
	logCtx := &RequestLogContext{RequestID: "image-usage", Credential: &config.CredentialConfig{Name: "test", Type: config.ProviderTypeProxy}}
	recorder := httptest.NewRecorder()
	require.NoError(t, prx.handleStreamingWithTokens(recorder, resp, "test", "google/gemini-2.5-flash-image", logCtx))
	require.Contains(t, recorder.Body.String(), image)
	require.NotNil(t, logCtx.TokenUsage)
	require.Equal(t, 14, logCtx.TokenUsage.PromptTokens)
	require.Equal(t, 1301, logCtx.TokenUsage.CompletionTokens)
	require.Equal(t, 1290, logCtx.TokenUsage.OutputImageTokens)
	require.Equal(t, "provider", logCtx.UsageSource)
	cost := models.CalculateTokenCosts(logCtx.TokenUsage, &models.ModelPrice{
		InputCostPerToken: 0.0000005124, OutputCostPerToken: 0.00000427,
		OutputCostPerImageToken: 0.00005124, CacheReadInputTokensFree: true,
	})
	require.InDelta(t, 0.0661537436, cost.TotalCost, 1e-12)
}

func TestStreamUsageLinesSplitAtEveryBoundary(t *testing.T) {
	line := []byte(`data: {"usage":{"prompt_tokens":14,"completion_tokens":1301,"total_tokens":1315,"completion_tokens_details":{"image_tokens":1290}}}` + "\r\n")
	for split := 1; split < len(line); split++ {
		var capture streamUsageLines
		var usage *StreamUsageInfo
		consume := func(frame []byte) {
			usage = (&openAIStreamUsageExtractor{}).ExtractUsage(frame)
		}
		capture.Observe(line[:split], consume)
		require.Nil(t, usage)
		capture.Observe(line[split:], consume)
		require.NotNil(t, usage, "split %d", split)
		require.Equal(t, 1290, usage.OutputImageTokens)
	}
}

func TestProxyHopStreamingUsageInLargeImageFrame(t *testing.T) {
	for _, mode := range []string{"complete", "drain after disconnect", "EOF without newline"} {
		t.Run(mode, func(t *testing.T) {
			prx := NewTestProxyBuilder().WithDrainUpstreamOnAbort(true).Build()
			image := "data:image/png;base64," + strings.Repeat("A", 2*1024*1024)
			body := `data: {"choices":[{"delta":{"images":[{"image_url":{"url":"` + image + `"}}]}}],"usage":{"prompt_tokens":14,"completion_tokens":1301,"total_tokens":1315,"completion_tokens_details":{"image_tokens":1290}}}` + "\n\ndata: [DONE]\n\n"
			if mode == "EOF without newline" {
				body = strings.TrimSuffix(body, "\n\ndata: [DONE]\n\n")
			}
			resp := &ProxyResponse{StatusCode: 200, Headers: http.Header{"Content-Type": []string{"text/event-stream"}}, StreamBody: io.NopCloser(strings.NewReader(body))}
			logCtx := &RequestLogContext{RequestID: "image-usage", Credential: &config.CredentialConfig{Name: "test", Type: config.ProviderTypeAIR}}
			recorder := httptest.NewRecorder()
			var writer http.ResponseWriter = recorder
			if mode == "drain after disconnect" {
				writer = newFailAfterNBytesWriter(10)
			}
			usage, err := prx.writeProxyStreamingResponseWithTokens(writer, resp, httptest.NewRequest("POST", "/v1/chat/completions", nil), logCtx.Credential, "google/gemini-2.5-flash-image", "google/gemini-2.5-flash-image", logCtx)
			if mode == "drain after disconnect" {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				require.Contains(t, recorder.Body.String(), image)
			}
			require.NotNil(t, usage)
			require.Equal(t, 14, usage.PromptTokens)
			require.Equal(t, 1301, usage.CompletionTokens)
			require.Equal(t, 1290, usage.OutputImageTokens)
			require.Equal(t, "provider", logCtx.UsageSource)
			cost := models.CalculateTokenCosts(usage, &models.ModelPrice{
				InputCostPerToken: 0.0000005124, OutputCostPerToken: 0.00000427,
				OutputCostPerImageToken: 0.00005124, CacheReadInputTokensFree: true,
			})
			require.InDelta(t, 0.0661537436, cost.TotalCost, 1e-12)
		})
	}
}

func TestStreamUsageLinesFinalize(t *testing.T) {
	var capture streamUsageLines
	var lines []string
	consume := func(line []byte) { lines = append(lines, string(line)) }
	capture.Observe([]byte("data: first\n\ndata: last"), consume)
	capture.Finalize(consume)
	require.Equal(t, []string{"data: first\n", "\n", "data: last"}, lines)
}

func TestStreamUsageLinesBoundAndRecovery(t *testing.T) {
	capture := streamUsageLines{maxLineBytes: 64}
	var lines []string
	consume := func(line []byte) { lines = append(lines, string(line)) }
	chunk := []byte(strings.Repeat("A", 8))
	for size := 0; size <= capture.maxLineBytes; size += len(chunk) {
		capture.Observe(chunk, consume)
	}
	require.True(t, capture.discard)
	require.Nil(t, capture.pending)
	capture.Observe([]byte("tail\ndata: next\n"), consume)
	require.Equal(t, []string{"data: next\n"}, lines)
	require.False(t, capture.discard)
}
