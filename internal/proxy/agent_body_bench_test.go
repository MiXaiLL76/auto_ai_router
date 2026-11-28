package proxy

// What the router does with a large agent-shaped request body (~1.2 MB,
// 338 messages, 15 tools) on each client format, through a proxy/AIR
// credential and through a direct provider. The numbers behind REVIEW_JSON.md:
//
//	go test ./internal/proxy/ -run '^$' -bench 'AgentBody/' -benchtime 60x
//	AGENT_BODY_RETAINED=1 go test ./internal/proxy/ -run TestAgentBodyRetained -v

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

func agentTools() []map[string]any {
	tools := make([]map[string]any, 0, 15)
	for i := 0; i < 15; i++ {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        fmt.Sprintf("tool_%d", i),
				"description": strings.Repeat("Reads a file from the workspace and returns it. ", 12),
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":  map[string]any{"type": "string", "description": "absolute path"},
						"limit": map[string]any{"type": "integer"},
					},
					"required": []string{"path"},
				},
			},
		})
	}
	return tools
}

func filler(i, n int) string {
	line := fmt.Sprintf("func handler%d(w http.ResponseWriter, r *http.Request) {\n\tif err := \"quoted\\tvalue\"; err != nil { return }\n}\n", i)
	return strings.Repeat(line, n/len(line)+1)[:n]
}

func agentChatBody(model string) []byte {
	msgs := []map[string]any{{"role": "system", "content": filler(0, 20000)}}
	for i := 0; len(msgs) < 338; i++ {
		id := fmt.Sprintf("call_%06d", i)
		msgs = append(msgs,
			map[string]any{"role": "assistant", "content": "Let me look at that file.", "tool_calls": []map[string]any{{
				"id": id, "type": "function",
				"function": map[string]any{"name": "tool_1", "arguments": fmt.Sprintf(`{"path":"/src/file_%d.go","limit":200}`, i)},
			}}},
			map[string]any{"role": "tool", "tool_call_id": id, "content": filler(i, 6500)},
		)
		if i%10 == 0 {
			msgs = append(msgs, map[string]any{"role": "user", "content": filler(i, 800)})
		}
	}
	body, _ := json.Marshal(map[string]any{
		"model": model, "stream": true, "messages": msgs, "tools": agentTools(),
		"stream_options": map[string]any{"include_usage": true}, "max_tokens": 4096,
	})
	return body
}

func agentMessagesBody(model string) []byte {
	msgs := []map[string]any{}
	for i := 0; len(msgs) < 338; i++ {
		id := fmt.Sprintf("toolu_%06d", i)
		msgs = append(msgs,
			map[string]any{"role": "assistant", "content": []map[string]any{
				{"type": "text", "text": "Let me look at that file."},
				{"type": "tool_use", "id": id, "name": "tool_1", "input": map[string]any{"path": fmt.Sprintf("/src/file_%d.go", i), "limit": 200}},
			}},
			map[string]any{"role": "user", "content": []map[string]any{
				{"type": "tool_result", "tool_use_id": id, "content": filler(i, 6500)},
			}},
		)
	}
	msgs[len(msgs)-1]["content"].([]map[string]any)[0]["cache_control"] = map[string]any{"type": "ephemeral"}
	tools := make([]map[string]any, 0, 15)
	for _, t := range agentTools() {
		fn := t["function"].(map[string]any)
		tools = append(tools, map[string]any{"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]})
	}
	body, _ := json.Marshal(map[string]any{
		"model": model, "stream": true, "max_tokens": 4096, "messages": msgs, "tools": tools,
		"system": []map[string]any{{"type": "text", "text": filler(0, 20000), "cache_control": map[string]any{"type": "ephemeral"}}},
	})
	return body
}

func agentResponsesBody(model string) []byte {
	input := []map[string]any{{"role": "user", "content": filler(0, 800)}}
	for i := 0; len(input) < 338; i++ {
		id := fmt.Sprintf("call_%06d", i)
		input = append(input,
			map[string]any{"type": "reasoning", "id": fmt.Sprintf("rs_%06d", i), "summary": []any{}, "encrypted_content": filler(i, 600)},
			map[string]any{"type": "function_call", "call_id": id, "name": "tool_1", "arguments": fmt.Sprintf(`{"path":"/src/file_%d.go"}`, i)},
			map[string]any{"type": "function_call_output", "call_id": id, "output": filler(i, 9500)},
		)
	}
	tools := make([]map[string]any, 0, 15)
	for _, t := range agentTools() {
		fn := t["function"].(map[string]any)
		tools = append(tools, map[string]any{"type": "function", "name": fn["name"], "description": fn["description"], "parameters": fn["parameters"]})
	}
	body, _ := json.Marshal(map[string]any{
		"model": model, "stream": true, "input": input, "tools": tools, "instructions": filler(0, 20000), "store": false,
	})
	return body
}

const agentChatSSE = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
	"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":292000,\"completion_tokens\":1,\"total_tokens\":292001}}\n\n" +
	"data: [DONE]\n\n"

const agentMessagesSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"usage\":{\"input_tokens\":292000,\"output_tokens\":1}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

const agentResponsesSSE = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"m\",\"output\":[]}}\n\n" +
	"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"ok\"}\n\n" +
	"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"m\",\"output\":[],\"usage\":{\"input_tokens\":292000,\"output_tokens\":1,\"total_tokens\":292001}}}\n\n"

type agentCase struct {
	name, path, sse string
	body            func(string) []byte
	credType        config.ProviderType
}

func agentCases() []agentCase {
	return []agentCase{
		{"chat_via_proxy", "/v1/chat/completions", agentChatSSE, agentChatBody, config.ProviderTypeProxy},
		{"chat_direct_openai", "/v1/chat/completions", agentChatSSE, agentChatBody, config.ProviderTypeOpenAI},
		{"messages_via_proxy", "/v1/messages", agentMessagesSSE, agentMessagesBody, config.ProviderTypeProxy},
		{"messages_direct_anthropic", "/v1/messages", agentMessagesSSE, agentMessagesBody, config.ProviderTypeAnthropic},
		{"responses_via_proxy", "/v1/responses", agentResponsesSSE, agentResponsesBody, config.ProviderTypeProxy},
		{"responses_direct_openai", "/v1/responses", agentResponsesSSE, agentResponsesBody, config.ProviderTypeOpenAI},
	}
}

func BenchmarkAgentBody(b *testing.B) {
	for _, tc := range agentCases() {
		b.Run(tc.name, func(b *testing.B) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, tc.sse)
			}))
			defer upstream.Close()
			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{Name: "c", Type: tc.credType, BaseURL: upstream.URL, APIKey: "master-key", RPM: 100000000, TPM: 0}).
				WithMaxBodySizeMB(35).
				Build()
			body := tc.body("gpt-4o")
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				req := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewReader(body))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)
				if w.Code != http.StatusOK {
					b.Fatalf("status %d: %.300s", w.Code, w.Body.String())
				}
			}
		})
	}
}

// Reference points: what one pass over the same body costs.
func BenchmarkAgentBodyReference(b *testing.B) {
	body := agentChatBody("gpt-4o")
	b.Run("stdlib_unmarshal_any", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var v map[string]any
			_ = json.Unmarshal(body, &v)
		}
	})
	b.Run("stdlib_unmarshal_rawmessage_top", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var v map[string]json.RawMessage
			_ = json.Unmarshal(body, &v)
		}
	})
	b.Run("stdlib_valid", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = json.Valid(body)
		}
	})
	b.Run("copy", func(b *testing.B) {
		b.SetBytes(int64(len(body)))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = bytes.Clone(body)
		}
	})
}

// TestAgentBodyRetained: how many bytes stay live per in-flight request while
// the upstream stream is open (i.e. while the client is reading the answer).
func TestAgentBodyRetained(t *testing.T) {
	if os.Getenv("AGENT_BODY_RETAINED") == "" {
		t.Skip("measurement, not a check: set AGENT_BODY_RETAINED=1 to run")
	}
	const inflight = 40
	for _, tc := range agentCases() {
		t.Run(tc.name, func(t *testing.T) {
			release := make(chan struct{})
			started := make(chan struct{}, inflight)
			first, _, _ := strings.Cut(tc.sse, "\n\n")
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, first+"\n\n")
				w.(http.Flusher).Flush()
				started <- struct{}{}
				<-release
				_, _ = io.WriteString(w, tc.sse[len(first)+2:])
			}))
			defer upstream.Close()
			prx := NewTestProxyBuilder().
				WithCredentials(config.CredentialConfig{Name: "c", Type: tc.credType, BaseURL: upstream.URL, APIKey: "master-key", RPM: 100000000}).
				WithMaxBodySizeMB(35).
				Build()
			body := tc.body("gpt-4o")

			var before, during runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			var wg sync.WaitGroup
			for i := 0; i < inflight; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					// Each request gets its own reader over the shared body, like a
					// server would after reading the socket (the read buffer itself
					// is the router's own io.ReadAll copy).
					req := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewReader(body))
					req.Header.Set("Authorization", "Bearer master-key")
					req.Header.Set("Content-Type", "application/json")
					prx.ProxyRequest(httptest.NewRecorder(), req)
				}()
			}
			for i := 0; i < inflight; i++ {
				<-started
			}
			time.Sleep(300 * time.Millisecond)
			runtime.GC()
			runtime.GC()
			runtime.ReadMemStats(&during)
			close(release)
			wg.Wait()
			per := float64(during.HeapAlloc-before.HeapAlloc) / inflight / (1 << 20)
			t.Logf("RETAINED %-28s body=%.2f MB  live per in-flight request=%.2f MB (%.1fx body)",
				tc.name, float64(len(body))/(1<<20), per, per*(1<<20)/float64(len(body)))
		})
	}
}
