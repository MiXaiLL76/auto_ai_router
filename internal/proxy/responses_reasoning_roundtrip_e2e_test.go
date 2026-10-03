package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	pricing "github.com/mixaill76/auto_ai_router/internal/models"
	compatlitellm "github.com/mixaill76/auto_ai_router/internal/responsecompat/litellm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProxyRequest_ConvertedResponsesRoundTripsReasoning drives a two-turn
// thinking-mode tool loop through /v1/responses against a Chat-Completions-only
// upstream (DeepSeek-style, passthrough_responses: false), the way a Responses
// client such as Codex does: turn one's reasoning must reach the client, and
// when the client echoes the output back with the tool result, the upstream
// must get it as reasoning_content on the assistant tool-call message —
// DeepSeek rejects a thinking-mode tool-call turn without it.
func TestProxyRequest_ConvertedResponsesRoundTripsReasoning(t *testing.T) {
	const reasoning = "Need the weather tool for Paris."

	for _, tt := range []struct {
		name   string
		stream bool
	}{
		{name: "non-stream", stream: false},
		{name: "stream", stream: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var mu sync.Mutex
			var upstreamBodies []map[string]interface{}

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				var body map[string]interface{}
				require.NoError(t, json.Unmarshal(raw, &body))
				mu.Lock()
				upstreamBodies = append(upstreamBodies, body)
				turn := len(upstreamBodies)
				mu.Unlock()

				if turn == 1 {
					// Turn one: reasoning + a tool call. The stream uses the
					// "reasoning" spelling (OpenRouter/vLLM), the JSON body
					// "reasoning_content" (DeepSeek).
					if tt.stream {
						w.Header().Set("Content-Type", "text/event-stream")
						for _, chunk := range []string{
							`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Need the weather "},"finish_reason":null}]}`,
							`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"reasoning":"tool for Paris."},"finish_reason":null}]}`,
							`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}`,
							`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`,
						} {
							_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
						}
						_, _ = w.Write([]byte("data: [DONE]\n\n"))
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","created":1,"model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"` + reasoning + `","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`))
					return
				}

				// Turn two: final answer.
				if tt.stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(`data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"Sunny."},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":2,"total_tokens":32}}` + "\n\n"))
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"c2","object":"chat.completion","created":1,"model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"Sunny."},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":2,"total_tokens":32}}`))
			}))
			defer upstream.Close()

			passthroughResponses := false
			builder := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{
					Name:    "deepseek-direct",
					Type:    config.ProviderTypeOpenAI,
					BaseURL: upstream.URL + "/v1",
					APIKey:  "upstream-key",
					RPM:     100,
					TPM:     -1,
				}).
				WithMasterKey("master-key")
			builder.config.ModelManager = pricing.New(builder.config.Logger, 50, []config.ModelRPMConfig{
				{Name: "deepseek-flash", Credential: "deepseek-direct", PassthroughResponses: &passthroughResponses},
			})
			builder.config.ModelManager.LoadModelsFromConfig(builder.config.Credentials)
			prx := builder.Build()
			prx.responseCompat = compatlitellm.New()

			send := func(body map[string]interface{}) *httptest.ResponseRecorder {
				if tt.stream {
					body["stream"] = true
				}
				raw, err := json.Marshal(body)
				require.NoError(t, err)
				req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(raw)))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)
				require.Equal(t, http.StatusOK, w.Code, w.Body.String())
				return w
			}

			userTurn := map[string]interface{}{"role": "user", "content": "Weather in Paris?"}
			tools := []interface{}{map[string]interface{}{
				"type": "function", "name": "get_weather",
				"parameters": map[string]interface{}{"type": "object"},
			}}

			// Turn one.
			w := send(map[string]interface{}{
				"model": "deepseek-flash", "store": false, "tools": tools,
				"input": []interface{}{userTurn},
			})
			output := responsesOutput(t, w.Body.String(), tt.stream)
			if tt.stream {
				assert.Contains(t, w.Body.String(), `"type":"response.reasoning_summary_text.delta"`,
					"reasoning must stream live to the client")
			}
			require.Len(t, output, 2, "reasoning + function_call")
			assert.Equal(t, "reasoning", output[0]["type"])
			summary := output[0]["summary"].([]interface{})
			require.Len(t, summary, 1)
			assert.Equal(t, reasoning, summary[0].(map[string]interface{})["text"])
			assert.Equal(t, "function_call", output[1]["type"])

			// Turn two: the client echoes turn one's output plus the tool result.
			input := []interface{}{userTurn}
			for _, item := range output {
				input = append(input, item)
			}
			input = append(input, map[string]interface{}{"type": "function_call_output", "call_id": "call_1", "output": "Sunny"})
			send(map[string]interface{}{
				"model": "deepseek-flash", "store": false, "tools": tools,
				"input": input,
			})

			mu.Lock()
			defer mu.Unlock()
			require.Len(t, upstreamBodies, 2)
			messages := upstreamBodies[1]["messages"].([]interface{})
			require.Len(t, messages, 3, "user, assistant tool-call turn, tool result")
			assistant := messages[1].(map[string]interface{})
			assert.Equal(t, "assistant", assistant["role"])
			assert.Equal(t, reasoning, assistant["reasoning_content"])
			toolCalls := assistant["tool_calls"].([]interface{})
			require.Len(t, toolCalls, 1)
			assert.Equal(t, "call_1", toolCalls[0].(map[string]interface{})["id"])
			assert.Equal(t, "tool", messages[2].(map[string]interface{})["role"])

			raw, err := json.Marshal(messages)
			require.NoError(t, err)
			assert.NotContains(t, string(raw), "[Reasoning]")
		})
	}
}

// responsesOutput extracts the output items of a /v1/responses reply: the
// body itself, or the response.completed event of a stream.
func responsesOutput(t *testing.T, body string, stream bool) []map[string]interface{} {
	t.Helper()
	var response struct {
		Output []map[string]interface{} `json:"output"`
	}
	if !stream {
		require.NoError(t, json.Unmarshal([]byte(body), &response), body)
		return response.Output
	}
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || !strings.Contains(data, `"type":"response.completed"`) {
			continue
		}
		var event struct {
			Response json.RawMessage `json:"response"`
		}
		require.NoError(t, json.Unmarshal([]byte(data), &event))
		require.NoError(t, json.Unmarshal(event.Response, &response))
		return response.Output
	}
	require.Fail(t, "no response.completed event", body)
	return nil
}
