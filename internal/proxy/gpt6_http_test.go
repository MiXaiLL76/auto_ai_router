package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGPT6ResponsesHTTP(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, effort := range []string{"max", "pro"} {
			for _, toolFields := range []string{
				`"tools":[{"type":"function","async":true,"function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":"required"`,
				`"tools":[],"additional_tools":[{"type":"function","name":"lookup","async":true}],"tool_choice":{"type":"function","name":"lookup"}`,
			} {
				t.Run(fmt.Sprintf("stream=%t/effort=%s/%s", streaming, effort, toolFields), func(t *testing.T) {
					upstreamBody := make(chan map[string]json.RawMessage, 1)
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						assert.Equal(t, "/v1/responses", r.URL.Path)
						var body map[string]json.RawMessage
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
							w.WriteHeader(400)
							return
						}
						upstreamBody <- body
						response := `{"id":"resp_gpt6","object":"response","model":"gpt-6-astra","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"GPT6_OK"}]}],"usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":10},"output_tokens_details":{"reasoning_tokens":5}}}`
						if streaming {
							w.Header().Set("Content-Type", "text/event-stream")
							_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\ndata: [DONE]\n\n", response)
						} else {
							w.Header().Set("Content-Type", "application/json")
							_, _ = io.WriteString(w, response)
						}
					}))
					defer upstream.Close()
					prx := NewTestProxyBuilder().WithSingleCredential("upstream", config.ProviderTypeOpenAI, upstream.URL, "upstream-key").WithMasterKey("master-key").Build()
					dbStub := &stubLiteLLMManager{}
					prx.LiteLLMDB = dbStub
					prx.priceRegistry = models.NewModelPriceRegistry()
					prx.priceRegistry.Update(map[string]*models.ModelPrice{"gpt-6-astra": {
						InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadInputTokenCost: 0.1, CacheCreationInputTokenCost: 1.25,
					}})
					input := `[{"role":"user","content":"test"},{"type":"configuration_update","reasoning":{"effort":"high"}},{"type":"prompt_cache_breakpoint"}]`
					body := fmt.Sprintf(`{"model":"gpt-6-astra","input":%s,"stream":%t,"max_output_tokens":256,"reasoning":{"effort":%q},"temperature":0.5,"top_p":0.9,"top_logprobs":2,"include":["message.output_text.logprobs","reasoning.encrypted_content"],"prompt_cache_options":{"ttl":"30m","mode":"explicit"},%s}`, input, streaming, effort, toolFields)
					req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
					req.Header.Set("Authorization", "Bearer master-key")
					req.Header.Set("Content-Type", "application/json")
					recorder := httptest.NewRecorder()
					prx.ProxyRequest(recorder, req)
					require.Equal(t, 200, recorder.Code, recorder.Body.String())
					require.NotEmpty(t, upstreamBody)
					got := <-upstreamBody
					assert.JSONEq(t, input, string(got["input"]))
					assert.JSONEq(t, `{"effort":`+fmt.Sprintf("%q", effort)+`}`, string(got["reasoning"]))
					assert.JSONEq(t, `{"ttl":"30m","mode":"explicit"}`, string(got["prompt_cache_options"]))
					assert.Equal(t, "256", string(got["max_output_tokens"]))
					for _, key := range []string{"temperature", "top_p", "top_logprobs"} {
						assert.NotContains(t, got, key)
					}
					assert.JSONEq(t, `["reasoning.encrypted_content"]`, string(got["include"]))
					assert.Contains(t, got, "tool_choice")
					if tools, ok := got["tools"]; ok {
						assert.JSONEq(t, `[{"type":"function","name":"lookup","async":true,"parameters":{"type":"object"}}]`, string(tools))
					} else {
						assert.JSONEq(t, `[{"type":"function","name":"lookup","async":true}]`, string(got["additional_tools"]))
					}
					assert.Contains(t, recorder.Body.String(), "GPT6_OK")
					require.Len(t, dbStub.loggedEntries, 1)
					entry := dbStub.loggedEntries[0]
					assert.Equal(t, 100, entry.PromptTokens)
					assert.Equal(t, 20, entry.CompletionTokens)
					assert.InDelta(t, 106.5, entry.Spend, 1e-9)
				})
			}
		}
	}
}

func TestGPT6HTTPAliasUsesUpstreamModelRules(t *testing.T) {
	for _, path := range []string{"/v1/responses", "/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			upstreamBody := make(chan map[string]json.RawMessage, 1)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, path, r.URL.Path)
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				upstreamBody <- body
				w.Header().Set("Content-Type", "application/json")
				if path == "/v1/responses" {
					_, _ = io.WriteString(w, `{"id":"resp_alias","object":"response","model":"gpt-6-astra","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
				} else {
					_, _ = io.WriteString(w, `{"id":"chat_alias","object":"chat.completion","model":"gpt-6-astra","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
				}
			}))
			defer upstream.Close()
			credential := config.CredentialConfig{Name: "upstream", Type: config.ProviderTypeOpenAI, BaseURL: upstream.URL, APIKey: "upstream-key", RPM: 100, TPM: 10000}
			prx := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
			prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{Name: "astra", Model: "gpt-6-astra", Credential: "upstream"}})
			prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{credential})
			body := `{"model":"astra","input":"test","max_output_tokens":300,"temperature":0.5,"top_p":0.9,"top_logprobs":2}`
			if path == "/v1/chat/completions" {
				body = `{"model":"astra","messages":[{"role":"user","content":"test"}],"max_tokens":200,"reasoning_effort":"max","temperature":0.5,"top_p":0.9,"top_logprobs":2,"logprobs":true}`
			}
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			prx.ProxyRequest(recorder, req)
			require.Equal(t, 200, recorder.Code, recorder.Body.String())
			require.NotEmpty(t, upstreamBody)
			got := <-upstreamBody
			assert.JSONEq(t, `"gpt-6-astra"`, string(got["model"]))
			for _, key := range []string{"temperature", "top_p", "top_logprobs", "logprobs", "max_tokens"} {
				assert.NotContains(t, got, key)
			}
			if path == "/v1/chat/completions" {
				assert.Equal(t, "200", string(got["max_completion_tokens"]))
				assert.JSONEq(t, `"max"`, string(got["reasoning_effort"]))
			} else {
				assert.Equal(t, "300", string(got["max_output_tokens"]))
			}
		})
	}
}
