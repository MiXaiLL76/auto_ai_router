package proxy

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Exercise the real proxy response/stream dispatch, including native Messages
// passthrough. A classifier refusal is terminal model output, not a transport
// failure to retry or a reason to ban the credential.
func TestProxyRequest_AnthropicRefusal(t *testing.T) {
	for _, provider := range []config.ProviderType{config.ProviderTypeAnthropic, config.ProviderTypeCometAPI} {
		for _, endpoint := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s%s/stream=%t", provider, endpoint, stream), func(t *testing.T) {
					var attempts atomic.Int64
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						attempts.Add(1)
						assert.Equal(t, "/v1/messages", r.URL.Path)
						const details = `{"type":"refusal","category":"reasoning_extraction","explanation":"Request declined."}`
						if stream {
							w.Header().Set("Content-Type", "text/event-stream")
							for _, event := range []string{
								`{"type":"message_start","message":{"id":"msg_fixture","type":"message","role":"assistant","content":[],"usage":{"input_tokens":17,"output_tokens":0}}}`,
								`{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":` + details + `},"usage":{"output_tokens":0}}`,
								`{"type":"message_stop"}`,
							} {
								_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
							}
							return
						}
						w.Header().Set("Content-Type", "application/json")
						_, _ = fmt.Fprint(w, `{"id":"msg_fixture","type":"message","role":"assistant","content":[],"stop_reason":"refusal","stop_details":`+details+`,"usage":{"input_tokens":17,"output_tokens":0}}`)
					}))
					defer upstream.Close()
					prx := NewTestProxyBuilder().WithSingleCredential("refusal-fixture", provider, upstream.URL, "fixture-key").Build()
					db := &stubLiteLLMManager{}
					prx.LiteLLMDB = db
					setTestModelPrice(prx, "claude-sonnet-4.5", &pricing.ModelPrice{InputCostPerToken: 1, OutputCostPerToken: 2})
					input := `"messages":[{"role":"user","content":"hi"}],"max_tokens":100`
					if endpoint == "/v1/responses" {
						input = `"input":"hi","max_output_tokens":100`
					}
					body := fmt.Sprintf(`{"model":"claude-sonnet-4.5",%s,"stream":%t}`, input, stream)
					req := httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(body))
					req.Header.Set("Authorization", "Bearer master-key")
					req.Header.Set("Content-Type", "application/json")
					w := httptest.NewRecorder()
					prx.ProxyRequest(w, req)
					require.Equal(t, http.StatusOK, w.Code, w.Body.String())
					assert.Equal(t, int64(1), attempts.Load(), "refusals must not be retried")
					assert.Contains(t, w.Body.String(), "Request declined.")
					assert.Contains(t, w.Body.String(), "reasoning_extraction")
					switch endpoint {
					case "/v1/chat/completions":
						assert.Contains(t, w.Body.String(), `"finish_reason":"content_filter"`)
						assert.Contains(t, w.Body.String(), `"refusal":"Request declined.`)
					case "/v1/responses":
						assert.Contains(t, w.Body.String(), `"type":"refusal"`)
						assert.Contains(t, w.Body.String(), `"status":"incomplete"`)
						if stream {
							assert.Contains(t, w.Body.String(), `"type":"response.refusal.delta"`)
							assert.Contains(t, w.Body.String(), `"type":"response.incomplete"`)
						}
					case "/v1/messages":
						assert.Contains(t, w.Body.String(), `"stop_reason":"refusal"`)
						assert.Contains(t, w.Body.String(), `"stop_details"`)
					}
					require.Len(t, db.loggedEntries, 1)
					assert.Equal(t, 17, db.loggedEntries[0].PromptTokens)
					if endpoint != "/v1/messages" {
						assert.Zero(t, db.loggedEntries[0].CompletionTokens)
					}
				})
			}
		}
	}
}
