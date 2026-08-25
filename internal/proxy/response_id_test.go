package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCaptureClientResponseID(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{name: "chat completion", payload: `{"id":"chatcmpl-123","choices":[]}`, want: "chatcmpl-123"},
		{name: "responses event", payload: `{"type":"response.created","response":{"id":"resp_123"}}`, want: "resp_123"},
		{name: "messages event", payload: `{"type":"message_start","message":{"id":"msg_123"}}`, want: "msg_123"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logCtx := &RequestLogContext{RequestID: "internal-123"}
			logCtx.captureClientResponseID([]byte(tt.payload))
			assert.Equal(t, tt.want, logCtx.ClientResponseID)
			assert.Equal(t, tt.want, logCtx.spendRequestID())
			assert.Equal(t, "internal-123", logCtx.RequestID)
		})
	}
}

func TestClientResponseIDScannerHandlesSplitSSE(t *testing.T) {
	logCtx := &RequestLogContext{RequestID: "internal-123"}
	var scanner clientResponseIDScanner

	scanner.observe(logCtx, []byte("event: response.created\ndata: {\"type\":\"response.created\",\"res"))
	assert.Empty(t, logCtx.ClientResponseID)
	scanner.observe(logCtx, []byte("ponse\":{\"id\":\"resp_stream_123\"}}\n\n"))

	assert.Equal(t, "resp_stream_123", logCtx.ClientResponseID)
	assert.Equal(t, "resp_stream_123", logCtx.spendRequestID())
}

func TestSpendRequestIDFallsBackToInternalID(t *testing.T) {
	logCtx := &RequestLogContext{RequestID: "internal-123"}
	assert.Equal(t, "internal-123", logCtx.spendRequestID())
}

func TestStreamToClientCapturesResponseID(t *testing.T) {
	logCtx := &RequestLogContext{RequestID: "internal-123"}
	recorder := httptest.NewRecorder()
	payload := "data: {\"id\":\"chatcmpl-stream-123\",\"choices\":[]}\n\ndata: [DONE]\n\n"

	err := NewTestProxyBuilder().Build().streamToClient(
		context.Background(), recorder, strings.NewReader(payload), "cred", "model",
		"/v1/chat/completions", http.StatusOK, nil, nil, logCtx,
	)

	require.NoError(t, err)
	assert.Equal(t, payload, recorder.Body.String())
	assert.Equal(t, "chatcmpl-stream-123", logCtx.ClientResponseID)
}

func TestWriteProxyResponseCapturesResponseID(t *testing.T) {
	logCtx := &RequestLogContext{RequestID: "internal-123"}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	response := &ProxyResponse{
		StatusCode: http.StatusOK,
		Headers:    http.Header{"Content-Type": []string{"application/json"}},
		Body:       []byte(`{"id":"chatcmpl-response-123","choices":[]}`),
	}

	NewTestProxyBuilder().Build().writeProxyResponse(
		recorder, response, request, &config.CredentialConfig{Name: "cred"}, "model", logCtx,
	)

	assert.Equal(t, "chatcmpl-response-123", logCtx.ClientResponseID)
}
