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
		`failed","response":{"id":"resp-output-failed","status":"failed"}}` + "\n\n",
	}

	tests := []struct {
		name       string
		run        func(*Proxy, http.ResponseWriter, *http.Response, *RequestLogContext) error
		input      []string
		wantBody   string
		wantMarker string
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
			wantMarker: "response.failed",
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
				assert.Contains(t, w.Body.String(), "Request failed")
				assert.NotContains(t, w.Body.String(), tt.wantMarker)
			} else {
				assert.Equal(t, tt.wantBody, w.Body.String())
			}
			assert.Equal(t, "failure", logCtx.Status)
			assert.Equal(t, http.StatusOK, logCtx.HTTPStatus)
			assert.Equal(t, "stream_error", logCtx.StreamOutcome)
			assert.Contains(t, logCtx.ErrorMsg, tt.wantMarker)
		})
	}
}
