package proxy

import (
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

func TestStreamingHandlersDetectFragmentedTerminalErrorsAcrossReads(t *testing.T) {
	terminalChunks := []string{
		"event: error\n",
		`data: {"ty`,
		`pe":"error","error":{"type":"overloaded_error","message":"fragmented terminal failure"}}` + "\n\n",
	}
	normalOutput := `data: {"id":"chatcmpl-normalized","choices":[{"delta":{"content":"normalized"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	outputErrorChunks := []string{
		`data: {"type":"response.`,
		`failed","response":{"id":"resp-output-failed","status":"failed","error":{"code":"rate_limit_exceeded","message":"Request failed"}}}` + "\n\n",
	}

	tests := []struct {
		name       string
		run        func(*Proxy, http.ResponseWriter, *http.Response, *RequestLogContext) error
		input      []string
		wantBody   string
		wantMarker string
		wantStatus int
		masked     bool
	}{
		{
			name: "direct passthrough observes raw provider frames",
			run: func(p *Proxy, w http.ResponseWriter, resp *http.Response, logCtx *RequestLogContext) error {
				return p.handleStreamingWithTokens(w, resp, "direct-openai", "gpt-4o-mini", logCtx)
			},
			input:      terminalChunks,
			wantBody:   strings.Join(terminalChunks, ""),
			wantMarker: "fragmented terminal failure",
			wantStatus: http.StatusServiceUnavailable,
			masked:     true,
		},
		{
			name: "transformed stream retains raw provider failure after normalization",
			run: func(p *Proxy, w http.ResponseWriter, resp *http.Response, logCtx *RequestLogContext) error {
				transformer := func(r io.Reader, _ string, output io.Writer) error {
					if _, err := io.Copy(io.Discard, r); err != nil {
						return err
					}
					_, err := io.WriteString(output, normalOutput)
					return err
				}
				return p.handleTransformedStreaming(w, resp, "direct-anthropic", "claude", "Anthropic", transformer, logCtx)
			},
			input:      terminalChunks,
			wantBody:   normalOutput,
			wantMarker: "fragmented terminal failure",
			wantStatus: http.StatusOK,
		},
		{
			name: "transformed stream observes fragmented emitted failure",
			run: func(p *Proxy, w http.ResponseWriter, resp *http.Response, logCtx *RequestLogContext) error {
				transformer := func(r io.Reader, _ string, output io.Writer) error {
					if _, err := io.Copy(io.Discard, r); err != nil {
						return err
					}
					for _, chunk := range outputErrorChunks {
						if _, err := io.WriteString(output, chunk); err != nil {
							return err
						}
					}
					return nil
				}
				return p.handleTransformedStreaming(w, resp, "direct-converted", "gpt-4o-mini", "converted", transformer, logCtx)
			},
			input:      []string{"data: [DONE]\n\n"},
			wantBody:   strings.Join(outputErrorChunks, ""),
			wantMarker: "rate_limit_exceeded",
			wantStatus: http.StatusTooManyRequests,
			masked:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prx := NewTestProxyBuilder().Build()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			credential := &config.CredentialConfig{Name: "terminal-provider", Type: config.ProviderTypeOpenAI, BaseURL: "https://provider.invalid"}
			logCtx := &RequestLogContext{
				RequestID:   "terminal-handler",
				StartTime:   time.Now().UTC(),
				Request:     request,
				Credential:  credential,
				ModelID:     "gpt-4o-mini",
				RealModelID: "gpt-4o-mini",
				TargetURL:   credential.BaseURL,
			}
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       &fragmentedStreamReadCloser{chunks: stringChunks(tt.input)},
				Request:    request,
			}
			w := httptest.NewRecorder()

			err := tt.run(prx, w, resp, logCtx)

			var terminalErr proxyProviderStreamError
			require.ErrorAs(t, err, &terminalErr)
			if tt.masked {
				if tt.wantStatus == http.StatusTooManyRequests {
					assert.Contains(t, w.Body.String(), "Rate limit exceeded")
					assert.Contains(t, w.Body.String(), "rate_limit_error")
				} else {
					assert.Contains(t, w.Body.String(), "Request failed")
				}
				assert.NotContains(t, w.Body.String(), tt.wantMarker)
			} else {
				assert.Equal(t, tt.wantBody, w.Body.String())
			}
			assert.Equal(t, "failure", logCtx.Status)
			assert.Equal(t, tt.wantStatus, logCtx.HTTPStatus)
			assert.Equal(t, "stream_error", logCtx.StreamOutcome)
			assert.Contains(t, logCtx.ErrorMsg, tt.wantMarker)
		})
	}
}

// TestHandleAnthropicCompatibleStreamingMasksInStreamErrorForAllCredentialTypes
// guards against the leak fixed alongside PR #111's review: an Anthropic-format
// in-stream `error` event's raw message is folded by the converter into a plain
// delta.content string, which the output-side sanitizer in handleTransformedStreaming
// never inspects (it only masks structural error.message keys). The input-side
// sanitizer in handleAnthropicCompatibleStreaming must run unconditionally — not
// just for ProMan — since it's the only pass that still sees the message under its
// original error.message key, before conversion moves it into content.
func TestHandleAnthropicCompatibleStreamingMasksInStreamErrorForAllCredentialTypes(t *testing.T) {
	rawStream := "event: error\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"internal: credential anthropic-direct-client-0dce8b1a exhausted, litellm_call_id=abc123"}}` + "\n\n"

	for _, credType := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeCometAPI, config.ProviderTypeProMan} {
		t.Run(string(credType), func(t *testing.T) {
			prx := NewTestProxyBuilder().Build()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			cred := &config.CredentialConfig{Name: "cred", Type: credType}
			logCtx := &RequestLogContext{
				RequestID:   "anthropic-error-leak",
				StartTime:   time.Now().UTC(),
				Request:     request,
				Credential:  cred,
				ModelID:     "claude-haiku-4.5",
				RealModelID: "claude-haiku-4.5",
			}
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(rawStream)),
			}
			w := httptest.NewRecorder()

			_ = prx.handleAnthropicCompatibleStreaming(w, resp, cred, "claude-haiku-4.5", "claude-haiku-4.5", credType, "label", logCtx)

			assert.NotContains(t, w.Body.String(), "anthropic-direct-client")
			assert.NotContains(t, w.Body.String(), "litellm_call_id")
			assert.Contains(t, w.Body.String(), "Request failed")
		})
	}
}
