package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
)

// relayProvider answers /v1/chat/completions with a stream that says "hello",
// then waits for the client to have seen it before writing the rest.
//
// A buffering router reads the whole stream before it relays a byte, so the
// wait times out; a relaying one releases the provider at once. That is what
// tells the two apart -- the answer alone cannot, because the buffered loop
// replays it verbatim when the model asks for no tool.
func relayProvider(t *testing.T) (*httptest.Server, *firstByteRecorder, *bool) {
	t.Helper()
	sawFirst := make(chan struct{})
	var relayed bool
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		if r.URL.Path != "/v1/chat/completions" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"hello"}}]}` + "\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		select {
		case <-sawFirst:
			mu.Lock()
			relayed = true
			mu.Unlock()
		case <-time.After(1500 * time.Millisecond):
		}
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"world"}}]}` + "\n\n"))
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	rec := &firstByteRecorder{onFirst: func() {
		select {
		case <-sawFirst:
		default:
			close(sawFirst)
		}
	}}
	return server, rec, &relayed
}

// TestGateRelaysAStreamWithNothingOfOurs covers the false side of the gate on
// both dispatch paths and both adapters: when the request offers no tool the
// router could run, the stream must reach the client as it arrives.
//
// The /v1/responses path had no test at all before this one, though it carries
// its own copy of the decision.
func TestGateRelaysAStreamWithNothingOfOurs(t *testing.T) {
	cases := []struct {
		name     string
		path     string
		provider string
		body     string
	}{
		{
			name:     "deepseek chat, no tools",
			path:     "/v1/chat/completions",
			provider: "deepseek",
			body:     `{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}],"stream":true}`,
		},
		{
			name:     "deepseek responses, no tools",
			path:     "/v1/responses",
			provider: "deepseek",
			body:     `{"model":"deepseek-flash","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"stream":true}`,
		},
		{
			name:     "deepseek chat, a client's own tool",
			path:     "/v1/chat/completions",
			provider: "deepseek",
			body:     `{"model":"deepseek-flash","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather"}}],"stream":true}`,
		},
		{
			name:     "moonshot chat, no tools",
			path:     "/v1/chat/completions",
			provider: "moonshot",
			body:     `{"model":"kimi-k2","messages":[{"role":"user","content":"hi"}],"stream":true}`,
		},
		{
			name:     "moonshot responses, no tools",
			path:     "/v1/responses",
			provider: "moonshot",
			body:     `{"model":"kimi-k2","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"stream":true}`,
		},
		{
			name:     "moonshot chat, a client's own tool",
			path:     "/v1/chat/completions",
			provider: "moonshot",
			body:     `{"model":"kimi-k2","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather"}}],"stream":true}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server, rec, relayed := relayProvider(t)
			defer server.Close()

			var prx *Proxy
			switch tc.provider {
			case "deepseek":
				fake := &deepSeekFake{baseURL: server.URL}
				prx, _ = deepSeekTestProxy(t, fake)
			case "moonshot":
				fake := &moonshotFake{baseURL: server.URL}
				prx, _ = moonshotTestProxy(t, fake)
			}

			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			prx.ProxyRequest(rec, req)

			if !*relayed {
				t.Fatalf("the client got nothing until the answer was finished: the stream was buffered")
			}
			if !strings.Contains(rec.body.String(), "hello") || !strings.Contains(rec.body.String(), "world") {
				t.Fatalf("the answer did not come through: %s", rec.body.String())
			}
		})
	}
}

// TestGateRunsTheLoopForEveryBuiltInTool covers the true side for the built-in
// tools that had no streaming test: the DeepSeek fetch and the dated
// web_search_preview. A gate that missed either would show the client the
// router's function call instead of an answer.
func TestGateRunsTheLoopForEveryBuiltInTool(t *testing.T) {
	cases := []struct {
		name      string
		builtin   string
		toolName  string
		toolPath  string
		toolReply string
		arguments string
		wantTool  func(f *deepSeekFake) int
	}{
		{
			name:      "web_fetch",
			builtin:   "web_fetch",
			toolName:  deepseektools.FetchToolName,
			toolPath:  "/v1/tools/fetch",
			toolReply: `{"title":"Example","content":"page"}`,
			arguments: `{"url":"https://example.com/a"}`,
			wantTool:  func(f *deepSeekFake) int { return f.fetchCalls },
		},
		{
			name:      "web_search_preview",
			builtin:   "web_search_preview",
			toolName:  deepseektools.SearchToolName,
			toolPath:  "/v1/tools/search",
			toolReply: `{"results":[{"title":"DeepSeek V4","url":"https://example.com/v4"}]}`,
			arguments: `{"query":"deepseek v4","count":3}`,
			wantTool:  func(f *deepSeekFake) int { return f.searchCalls },
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &deepSeekFake{}
			var toolMu sync.Mutex
			var toolCalls int
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.ReadAll(r.Body)
				switch r.URL.Path {
				case tc.toolPath:
					toolMu.Lock()
					toolCalls++
					if tc.toolPath == "/v1/tools/fetch" {
						fake.mu.Lock()
						fake.fetchCalls++
						fake.mu.Unlock()
					} else {
						fake.mu.Lock()
						fake.searchCalls++
						fake.mu.Unlock()
					}
					toolMu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(tc.toolReply))
				case "/v1/chat/completions":
					fake.mu.Lock()
					fake.chatCalls++
					call := fake.chatCalls
					fake.mu.Unlock()
					w.Header().Set("Content-Type", "text/event-stream")
					if call == 1 {
						_, _ = w.Write([]byte("data: " + fmt.Sprintf(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":%q,"arguments":%q}}]}}]}`, tc.toolName, tc.arguments) + "\n\n"))
						_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
						_, _ = w.Write([]byte("data: [DONE]\n\n"))
						return
					}
					_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"Here it is."}}]}` + "\n\n"))
					_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}}` + "\n\n"))
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			fake.baseURL = server.URL

			prx, _ := deepSeekTestProxy(t, fake)

			body := fmt.Sprintf(`{"model":"deepseek-flash","messages":[{"role":"user","content":"go"}],"tools":[{"type":%q}],"stream":true}`, tc.builtin)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			req.Header.Set("Authorization", "Bearer master-key")
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			prx.ProxyRequest(rec, req)

			if got := tc.wantTool(fake); got != 1 {
				t.Errorf("the router made %d tool calls, want 1 (the loop did not run)", got)
			}
			if fake.chatCalls != 2 {
				t.Errorf("the model was called %d times, want 2", fake.chatCalls)
			}
			out := rec.Body.String()
			if !strings.Contains(out, "Here it is.") {
				t.Errorf("the client did not get the answer: %s", out)
			}
			if strings.Contains(out, tc.toolName) {
				t.Errorf("the client saw the router's function tool: %s", out)
			}
		})
	}
}

// TestRelayProviderIsAPlainJSONStream is a guard on the helper above: if the
// SSE it emits stopped being parseable, every relay test would fail for the
// wrong reason.
func TestRelayProviderIsAPlainJSONStream(t *testing.T) {
	server, _, _ := relayProvider(t)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("the fake did not answer: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	raw, _ := io.ReadAll(resp.Body)
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(strings.SplitN(string(raw), "\n\n", 2)[0], "data: ")), &chunk); err != nil {
		t.Fatalf("the first chunk is not JSON: %v (%q)", err, raw)
	}
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content != "hello" {
		t.Fatalf("the first chunk does not say hello: %q", raw)
	}
}
