package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type clientErrorTransport struct {
	status int
	body   string
}

func (tr clientErrorTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if tr.status == 0 {
		return nil, context.Canceled
	}
	return &http.Response{
		StatusCode: tr.status,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(tr.body)),
		Request:    r,
	}, nil
}

func TestProxyRequestClientErrorMessages(t *testing.T) {
	for _, mode := range []string{"native", "litellm"} {
		for _, tt := range []struct {
			name   string
			status int
			body   string
		}{
			{"prompt length", http.StatusBadRequest, `{"error":{"message":"prompt is too long: 1685418 tokens \u003e 1000000 maximum","type":"invalid_request_error"},"request_id":"req_provider","type":"error"}`},
			{"internal server error", http.StatusInternalServerError, `{"error":{"message":"Post http://air-ru01/v1/responses failed"}}`},
			{"bad gateway", http.StatusBadGateway, `{"error":{"message":"Proxy forward error: Post http://air-ru01/v1/responses: context canceled"}}`},
			{"transport error", 0, ""},
		} {
			t.Run(mode+"/"+tt.name, func(t *testing.T) {
				prx := NewTestProxyBuilder().
					WithSingleCredential("gateway", config.ProviderTypeProxy, "http://air-ru01", "upstream-key").
					Build()
				prx.client.Transport = clientErrorTransport{status: tt.status, body: tt.body}
				if mode == "litellm" {
					prx.responseCompat = compatlitellm.New()
				}

				req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-4","input":"hello"}`))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)

				wantStatus := tt.status
				if wantStatus == 0 {
					wantStatus = http.StatusBadGateway
				}
				require.Equal(t, wantStatus, w.Code)
				var response APIErrorResponse
				require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
				for _, detail := range []string{"air-ru01", "http://", "context canceled", "upstream-key", "req_provider"} {
					assert.NotContains(t, w.Body.String(), detail)
				}
				if tt.status == http.StatusBadRequest {
					assert.Equal(t, "Context length exceeded", response.Error.Message)
					assert.Equal(t, "invalid_request_error", response.Error.Type)
					require.NotNil(t, response.Error.Code)
					assert.Equal(t, "context_length_exceeded", *response.Error.Code)
					require.NotNil(t, response.Error.Param)
					assert.Equal(t, "prompt", *response.Error.Param)
				} else {
					assert.Contains(t, []string{"Request failed", "Bad Gateway"}, response.Error.Message)
				}
			})
		}
	}
}
