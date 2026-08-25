package proxy

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriteProxyStreamingResponseWithTokensDetectsFragmentedProviderTerminalErrors(t *testing.T) {
	tests := []struct {
		name       string
		chunks     []string
		wantMarker string
		wantStatus int
		masked     bool
	}{
		{
			name: "openai error object",
			chunks: []string{
				`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"partial"}}]}` + "\n\n",
				`data: {"err`,
				`or":{"message":"provider exploded","type":"server_error"}}` + "\n",
				"\n",
				"data: [DONE]\n\n",
			},
			wantMarker: "provider exploded",
			wantStatus: http.StatusOK,
			masked:     true,
		},
		{
			name: "anthropic error event with CRLF framing",
			chunks: []string{
				"event: err",
				"or\r\n",
				`data: {"ty`,
				`pe":"error","error":{"type":"overloaded_error","message":"anthropic overloaded"}}` + "\r\n\r\n",
			},
			wantMarker: "anthropic overloaded",
			wantStatus: http.StatusServiceUnavailable,
			masked:     true,
		},
		{
			name: "responses failed event",
			chunks: []string{
				`data: {"type":"response.`,
				`failed","response":{"id":"resp_1","status":"failed","error":{"code":"rate_limit_exceeded","message":"Request failed"}}}` + "\n",
				"\n",
			},
			wantMarker: "rate_limit_exceeded",
			wantStatus: http.StatusTooManyRequests,
			masked:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().Build()
			streamBody := strings.Join(tt.chunks, "")
			proxyResp := &ProxyResponse{
				StatusCode: http.StatusOK,
				Headers: http.Header{
					"Content-Type": {"text/event-stream"},
				},
				StreamBody:  &fragmentedStreamReadCloser{chunks: stringChunks(tt.chunks)},
				IsStreaming: true,
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			credential := &config.CredentialConfig{
				Name:    "proxy-upstream",
				Type:    config.ProviderTypeProxy,
				BaseURL: "https://proxy.invalid",
			}
			logCtx := &RequestLogContext{
				RequestID:   "event-terminal-error",
				StartTime:   time.Now().UTC(),
				Request:     request,
				Status:      "unknown",
				Credential:  credential,
				ModelID:     "gpt-4o-mini",
				RealModelID: "gpt-4o-mini",
				TargetURL:   credential.BaseURL,
			}
			w := httptest.NewRecorder()

			_, err := prx.writeProxyStreamingResponseWithTokens(
				w,
				proxyResp,
				request,
				credential,
				logCtx.ModelID,
				logCtx.RealModelID,
				logCtx,
			)

			var terminalErr proxyProviderStreamError
			require.ErrorAs(t, err, &terminalErr)
			assert.Equal(t, tt.wantStatus, w.Code)
			if tt.masked {
				if tt.wantStatus == http.StatusTooManyRequests {
					assert.Contains(t, w.Body.String(), "Rate limit exceeded")
					assert.Contains(t, w.Body.String(), "rate_limit_error")
				} else {
					assert.Contains(t, w.Body.String(), "Request failed")
				}
				assert.NotContains(t, w.Body.String(), tt.wantMarker)
			} else {
				assert.Equal(t, streamBody, w.Body.String())
			}
			assert.Equal(t, "failure", logCtx.Status)
			assert.Equal(t, tt.wantStatus, logCtx.HTTPStatus)
			assert.Equal(t, "stream_error", logCtx.StreamOutcome)
			assert.Contains(t, logCtx.ErrorMsg, tt.wantMarker)

		})
	}
}

func TestWriteProxyStreamingResponseWithTokensKeepsStatusAfterStreamCommitted(t *testing.T) {
	chunks := []string{
		`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"partial"}}]}` + "\n\n",
		`data: {"error":{"message":"provider exploded","type":"server_error"}}` + "\n\n",
	}
	prx := NewTestProxyBuilder().Build()
	proxyResp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type": {"text/event-stream"},
		},
		StreamBody:  &fragmentedStreamReadCloser{chunks: stringChunks(chunks)},
		IsStreaming: true,
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{
		RequestID:   "event-terminal-error-after-content",
		StartTime:   time.Now().UTC(),
		Request:     request,
		Status:      "unknown",
		Credential:  &config.CredentialConfig{Name: "proxy-upstream"},
		ModelID:     "gpt-4o-mini",
		RealModelID: "gpt-4o-mini",
	}
	w := httptest.NewRecorder()

	_, err := prx.writeProxyStreamingResponseWithTokens(
		w,
		proxyResp,
		request,
		logCtx.Credential,
		logCtx.ModelID,
		logCtx.RealModelID,
		logCtx,
	)

	var terminalErr proxyProviderStreamError
	require.ErrorAs(t, err, &terminalErr)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "partial")
	assert.Contains(t, w.Body.String(), "Request failed")
	assert.Equal(t, http.StatusOK, logCtx.HTTPStatus)
}

func TestWriteProxyStreamingResponseWithTokensKeepsNormalFragmentedStreamSuccessful(t *testing.T) {
	chunks := []string{
		`data: {"id":"chatcmpl-normal","choices":[{"delta":{"con`,
		`tent":"all good"}}]}` + "\n",
		"\n",
		`data: {"id":"chatcmpl-normal","choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n",
		"data: [DONE]\n\n",
	}
	prx := NewTestProxyBuilder().Build()
	proxyResp := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type": {"text/event-stream"},
		},
		StreamBody:  &fragmentedStreamReadCloser{chunks: stringChunks(chunks)},
		IsStreaming: true,
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	logCtx := &RequestLogContext{}
	w := httptest.NewRecorder()

	_, err := prx.writeProxyStreamingResponseWithTokens(
		w,
		proxyResp,
		request,
		&config.CredentialConfig{Name: "proxy-upstream"},
		"gpt-4o-mini",
		"gpt-4o-mini",
		logCtx,
	)

	require.NoError(t, err)
	assert.Equal(t, "completed", logCtx.StreamOutcome)
	assert.Empty(t, logCtx.ErrorMsg)
	assert.Equal(t, strings.Join(chunks, ""), w.Body.String())
}

func TestProxyStreamErrorCaptureFinalizesFragmentedBareJSON(t *testing.T) {
	capture := &proxyStreamErrorCapture{}
	assert.Empty(t, capture.Observe([]byte(`{"err`)))
	assert.Empty(t, capture.Observe([]byte(`or":{"message":"bare failure"}}`)))
	assert.Contains(t, capture.Finalize(), "bare failure")
}

func TestStatusCodeFromProviderBodyError(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
		wantStatus int
		wantOK     bool
	}{
		{
			name:       "top level rate limit error",
			statusCode: http.StatusOK,
			body:       `{"error":{"code":"rate_limit_exceeded","message":"Request failed"}}`,
			wantStatus: http.StatusTooManyRequests,
			wantOK:     true,
		},
		{
			name:       "failed responses object",
			statusCode: http.StatusOK,
			body:       `{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"Request failed"}}`,
			wantStatus: http.StatusInternalServerError,
			wantOK:     true,
		},
		{
			name:       "nested failed response event body",
			statusCode: http.StatusOK,
			body:       `{"type":"response.failed","response":{"status":"failed","error":{"code":"rate_limit_exceeded"}}}`,
			wantStatus: http.StatusTooManyRequests,
			wantOK:     true,
		},
		{
			name:       "normal success",
			statusCode: http.StatusOK,
			body:       `{"id":"chatcmpl-1","choices":[{"message":{"content":"ok"}}]}`,
			wantOK:     false,
		},
		{
			name:       "real error status not remapped",
			statusCode: http.StatusBadGateway,
			body:       `{"error":{"code":"rate_limit_exceeded"}}`,
			wantOK:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotStatus, gotOK := statusCodeFromProviderBodyError(tt.statusCode, []byte(tt.body))

			assert.Equal(t, tt.wantOK, gotOK)
			if tt.wantOK {
				assert.Equal(t, tt.wantStatus, gotStatus)
			}
		})
	}
}

type fragmentedStreamReadCloser struct {
	chunks [][]byte
	index  int
}

func (r *fragmentedStreamReadCloser) Read(dst []byte) (int, error) {
	if r.index >= len(r.chunks) {
		return 0, io.EOF
	}
	chunk := r.chunks[r.index]
	r.index++
	if len(chunk) > len(dst) {
		return 0, errors.New("test stream chunk exceeds destination buffer")
	}
	return copy(dst, chunk), nil
}

func (r *fragmentedStreamReadCloser) Close() error { return nil }

func stringChunks(chunks []string) [][]byte {
	result := make([][]byte, len(chunks))
	for i, chunk := range chunks {
		result[i] = []byte(chunk)
	}
	return result
}
