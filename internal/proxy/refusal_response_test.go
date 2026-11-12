package proxy

import (
	"fmt"
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

func TestProxyMessagesRefusalPrivacy(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeCometAPI, config.ProviderTypeProxy, config.ProviderTypeAIR} {
		for _, mode := range []string{"json", "malformed-json", "multiline-stream", "large-stream"} {
			t.Run(fmt.Sprintf("%s/%s", provider, mode), func(t *testing.T) {
				stream := strings.HasSuffix(mode, "stream")
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					details := `{"category":"reasoning_extraction","explanation":"fixture-secret","fallback_credit_token":"fixture-credit"}`
					if mode == "large-stream" {
						details = strings.Replace(details, "fixture-secret", "fixture-secret"+strings.Repeat("x", 1024*1024+1), 1)
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fixture\",\"content\":[],\"usage\":{\"input_tokens\":17,\"output_tokens\":0}}}\n\n")
						separator := ""
						if mode == "multiline-stream" {
							separator = "\ndata: "
						}
						_, _ = fmt.Fprintf(w, "event: message_delta\ndata: {\"type\":\"message_delta\",%s\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":%s},\"usage\":{\"output_tokens\":0}}\n\n", separator, details)
						_, _ = fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
						return
					}
					w.Header().Set("Content-Type", "application/json")
					if mode == "malformed-json" {
						_, _ = fmt.Fprint(w, `{"stop_reason":"refusal","stop_details":{"explanation":"fixture-secret`)
						return
					}
					_, _ = fmt.Fprintf(w, `{"id":"msg_fixture","type":"message","role":"assistant","content":[],"stop_reason":"refusal","stop_details":%s,"usage":{"input_tokens":17,"output_tokens":0}}`, details)
				}))
				defer upstream.Close()
				prx := NewTestProxyBuilder().WithSingleCredential("privacy-fixture", provider, upstream.URL, "fixture-key").Build()
				req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(fmt.Sprintf(`{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"stream":%t}`, stream)))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)
				assert.NotContains(t, w.Body.String(), "fixture-secret")
				assert.NotContains(t, w.Body.String(), "fixture-credit")
				if mode == "malformed-json" {
					assert.Equal(t, http.StatusBadGateway, w.Code, w.Body.String())
				} else {
					require.Equal(t, http.StatusOK, w.Code, w.Body.String())
					assert.Contains(t, w.Body.String(), `"stop_reason":"refusal"`)
					assert.Contains(t, w.Body.String(), "reasoning_extraction")
				}
			})
		}
	}
}

func TestProxyMessagesAbortClosesProvider(t *testing.T) {
	for range 5 {
		reader, writer := io.Pipe()
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer func() { _ = writer.Close() }()
			for range 100 {
				_, err := fmt.Fprint(writer, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"category\":\"reasoning_extraction\",\"explanation\":\"fixture-secret\"}}}\n\n")
				if err != nil {
					return
				}
			}
		}()
		prx := NewTestProxyBuilder().Build()
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
		response := &ProxyResponse{StatusCode: 200, Headers: http.Header{"Content-Type": {"text/event-stream"}}, StreamBody: reader, IsStreaming: true}
		_, err := prx.writeProxyStreamingResponseWithTokens(newFailAfterNBytesWriter(10), response, req,
			&config.CredentialConfig{Name: "privacy-fixture", Type: config.ProviderTypeProxy}, "claude-sonnet-4.5", "claude-sonnet-4.5", &RequestLogContext{})
		require.Error(t, err)
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("provider reader was not closed after the client disconnected")
		}
	}
}
