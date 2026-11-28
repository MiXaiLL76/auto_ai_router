package proxy

// Corpus-driven checks for the request pipeline. AIR_BODY_CORPUS names a
// directory of request bodies (chat_*.json, messages_*.json,
// responses_*.json) kept outside the repository; without it everything here
// is skipped. See REVIEW_JSON.md for how the corpus is built and used.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

type corpusBody struct {
	name string
	kind string // chat | messages | responses
	body []byte
}

func loadCorpus(tb testing.TB) []corpusBody {
	dir := os.Getenv("AIR_BODY_CORPUS")
	if dir == "" {
		tb.Skip("set AIR_BODY_CORPUS to a directory of request bodies")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		tb.Fatal(err)
	}
	var out []corpusBody
	for _, e := range entries {
		kind, _, ok := strings.Cut(e.Name(), "_")
		if !ok || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		body, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			tb.Fatal(err)
		}
		out = append(out, corpusBody{name: strings.TrimSuffix(e.Name(), ".json"), kind: kind, body: body})
	}
	return out
}

// corpusVariants derives requests that exercise the optional branches of the
// request pipeline from one real body. The original bytes are always the
// first variant, untouched.
func corpusVariants(tb testing.TB, c corpusBody) []corpusBody {
	variants := []corpusBody{c}
	edit := func(suffix string, fn func(m map[string]any)) {
		var m map[string]any
		if err := json.Unmarshal(c.body, &m); err != nil {
			tb.Fatal(err)
		}
		fn(m)
		b, err := json.Marshal(m)
		if err != nil {
			tb.Fatal(err)
		}
		variants = append(variants, corpusBody{name: c.name + "+" + suffix, kind: c.kind, body: b})
	}
	edit("nostream", func(m map[string]any) { delete(m, "stream"); delete(m, "stream_options") })
	edit("billing", func(m map[string]any) {
		m["service_tier"] = "priority"
		m["session_id"] = "sess-1"
		m["litellm_session_id"] = "sess-2"
		m["extra_body"] = map[string]any{"service_tier": "flex", "thinking": map[string]any{"type": "enabled"}}
	})
	edit("notools", func(m map[string]any) { m["tools"] = []any{}; m["tool_choice"] = "auto" })
	switch c.kind {
	case "chat":
		edit("usageopts", func(m map[string]any) {
			m["stream_options"] = map[string]any{"include_usage": false, "include_obfuscation": false}
		})
		edit("nousageopts", func(m map[string]any) { delete(m, "stream_options") })
		edit("websearch", func(m map[string]any) {
			m["web_search_options"] = map[string]any{"search_context_size": "high"}
			m["tools"] = append(m["tools"].([]any), map[string]any{"type": "web_search_preview"})
		})
		edit("params", func(m map[string]any) {
			m["max_completion_tokens"] = 123
			m["logprobs"] = true
			m["parallel_tool_calls"] = false
			m["response_format"] = map[string]any{"type": "json_object"}
			m["metadata"] = map[string]any{"k": "v"}
			m["seed"] = 12345678901234567890.0
		})
	case "responses":
		edit("store", func(m map[string]any) {
			m["store"] = true
			m["metadata"] = map[string]any{"k": "v"}
			m["ttl"] = 60
		})
		edit("shapes", func(m map[string]any) {
			in := m["input"].([]any)
			in = append(in,
				map[string]any{"type": "compaction", "id": "cmp_1", "encrypted_content": "summary text"},
				map[string]any{"type": "compaction", "id": "cmp_2"},
				map[string]any{"type": "reasoning", "id": "rs_x", "content": []any{map[string]any{"type": "reasoning_text", "text": "thought"}}},
				map[string]any{"type": "reasoning", "id": "rs_y", "encrypted_content": "enc"},
				map[string]any{"type": "reasoning", "id": "rs_z"},
			)
			m["input"] = in
			m["instructions"] = []any{map[string]any{"role": "system", "content": "be brief"}, map[string]any{"role": "developer", "content": "really"}}
			m["reasoning"] = map[string]any{"effort": "none"}
			m["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "nested", "parameters": map[string]any{"type": "object"}}}, map[string]any{"type": "web_search"}}
		})
		edit("stringinput", func(m map[string]any) { m["input"] = "just a question"; m["previous_response_id"] = "resp_unknown" })
	case "messages":
		edit("nomarkers", func(m map[string]any) {
			b, _ := json.Marshal(m)
			b = bytes.ReplaceAll(b, []byte(`,"cache_control":{"type":"ephemeral"}`), nil)
			b = bytes.ReplaceAll(b, []byte(`,"cache_control":{"ttl":"1h","type":"ephemeral"}`), nil)
			clear(m)
			_ = json.Unmarshal(b, &m)
		})
		edit("stringsystem", func(m map[string]any) {
			m["system"] = "You are helpful."
			m["thinking"] = map[string]any{"type": "adaptive"}
			m["output_config"] = map[string]any{"effort": "high"}
			m["betas"] = []any{"context-1m-2025-08-07"}
			m["anthropic_beta"] = []any{"interleaved-thinking-2025-05-14"}
		})
	}
	return variants
}

var corpusPaths = map[string]string{"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}

// corpusTargets are the provider types every corpus body is routed to: the
// proxy/AIR hop, and each wire dialect a direct credential can speak.
var corpusTargets = []config.ProviderType{
	config.ProviderTypeProxy,
	config.ProviderTypeOpenAI,
	config.ProviderTypeAnthropic,
	config.ProviderTypeGemini,
}

// TestDumpUpstreamRequests records, for every corpus body and variant on
// every route, exactly what the router sends upstream (path, selected
// headers, body) and the status it answers the client with. Run it on two
// trees with AIR_UPSTREAM_DUMP pointing at two directories and compare them:
// the requests must be the same JSON.
func TestDumpUpstreamRequests(t *testing.T) {
	out := os.Getenv("AIR_UPSTREAM_DUMP")
	if out == "" {
		t.Skip("set AIR_UPSTREAM_DUMP (and AIR_BODY_CORPUS) to record upstream requests")
	}
	if err := os.MkdirAll(out, 0o750); err != nil {
		t.Fatal(err)
	}
	for _, target := range corpusTargets {
		for _, base := range loadCorpus(t) {
			for _, c := range corpusVariants(t, base) {
				var mu sync.Mutex
				var seen []map[string]any
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					got, _ := io.ReadAll(r.Body)
					mu.Lock()
					seen = append(seen, map[string]any{
						"method": r.Method, "path": r.URL.Path, "query": r.URL.RawQuery,
						"anthropic-beta": r.Header.Values("anthropic-beta"), "content-type": r.Header.Get("Content-Type"),
						"body": string(got),
					})
					mu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusTeapot) // not retried: one attempt per request
					_, _ = io.WriteString(w, `{"error":{"message":"recorded","type":"invalid_request_error"}}`)
				}))
				prx := NewTestProxyBuilder().
					WithCredentials(config.CredentialConfig{Name: "c", Type: target, BaseURL: upstream.URL, APIKey: "master-key", RPM: 100000000}).
					WithMaxBodySizeMB(35).
					Build()
				req := httptest.NewRequest(http.MethodPost, corpusPaths[c.kind], bytes.NewReader(c.body))
				req.Header.Set("Authorization", "Bearer master-key")
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				prx.ProxyRequest(w, req)
				upstream.Close()
				record, _ := json.Marshal(map[string]any{"client_status": w.Code, "upstream": seen})
				name := string(target) + "__" + strings.ReplaceAll(c.name, "+", "__") + ".json"
				if err := os.WriteFile(filepath.Join(out, name), record, 0o600); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}

// corpusSSE is a minimal successful stream per upstream dialect.
func corpusSSE(target config.ProviderType, kind string) string {
	switch target {
	case config.ProviderTypeAnthropic:
		return agentMessagesSSE
	case config.ProviderTypeGemini:
		return "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":1,\"totalTokenCount\":11},\"modelVersion\":\"m\"}\n\n"
	case config.ProviderTypeOpenAI:
		if kind == "responses" {
			return agentResponsesSSE
		}
		return agentChatSSE
	}
	// proxy/AIR: the hop answers in the client's own format.
	switch kind {
	case "messages":
		return agentMessagesSSE
	case "responses":
		return agentResponsesSSE
	}
	return agentChatSSE
}

// inMemoryUpstream answers every upstream request from memory: it reads the
// request body to the end, as a provider would, and returns a fixed stream.
// With it a benchmark measures what the router itself does with a request,
// not the loopback network between the router and a mock.
type inMemoryUpstream struct{ sse string }

func (u inMemoryUpstream) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		_, _ = io.Copy(io.Discard, req.Body)
		_ = req.Body.Close()
	}
	return &http.Response{
		Status: "200 OK", StatusCode: http.StatusOK, Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
		Header:  http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:    io.NopCloser(strings.NewReader(u.sse)),
		Request: req,
	}, nil
}

// BenchmarkCorpusBody times one request per route on real session bodies:
// every client format through the proxy/AIR hop and through each dialect a
// direct credential speaks. The upstream is in memory (inMemoryUpstream), so
// the figures are the router's own CPU and allocations per request; set
// AIR_BENCH_TCP=1 to put a real loopback HTTP server behind it instead.
func BenchmarkCorpusBody(b *testing.B) {
	only := os.Getenv("AIR_BODY_CORPUS_ONLY") // e.g. "_0" to time one session
	if only == "" {
		only = "_0"
	}
	for _, c := range loadCorpus(b) {
		if !strings.HasSuffix(c.name, only) {
			continue
		}
		for _, target := range corpusTargets {
			b.Run(c.kind+"_to_"+string(target), func(b *testing.B) {
				sse := corpusSSE(target, c.kind)
				baseURL := "http://upstream.invalid"
				if os.Getenv("AIR_BENCH_TCP") != "" {
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						_, _ = io.Copy(io.Discard, r.Body)
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, sse)
					}))
					defer upstream.Close()
					baseURL = upstream.URL
				}
				prx := NewTestProxyBuilder().
					WithCredentials(config.CredentialConfig{Name: "c", Type: target, BaseURL: baseURL, APIKey: "master-key", RPM: 100000000}).
					WithMaxBodySizeMB(35).
					Build()
				if os.Getenv("AIR_BENCH_TCP") == "" {
					prx.client.Transport = inMemoryUpstream{sse: sse}
				}
				b.SetBytes(int64(len(c.body)))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					req := httptest.NewRequest(http.MethodPost, corpusPaths[c.kind], bytes.NewReader(c.body))
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
}

// BenchmarkChatBySize is the ordinary chat request of the load test (a system
// prompt and a document, short answer) at its three sizes -- about 1, 8 and
// 30 thousand tokens -- through the proxy/AIR hop and to an OpenAI-wire
// provider. It needs no corpus. Most traffic looks like this, so a change
// that helps the multi-megabyte agent bodies must not cost anything here.
func BenchmarkChatBySize(b *testing.B) {
	for _, size := range []struct {
		name  string
		bytes int
	}{{"1k_tokens_4KB", 4 << 10}, {"8k_tokens_35KB", 35 << 10}, {"30k_tokens_134KB", 134 << 10}} {
		doc := strings.Repeat("Section 4.2: the supplier shall deliver within thirty (30) days.\nПункт 4.3: оплата производится в рублях.\n", size.bytes/120+1)[:size.bytes]
		body, err := json.Marshal(map[string]any{
			"model": "gpt-4o", "stream": true, "temperature": 0.2, "max_tokens": 300,
			"messages": []map[string]any{
				{"role": "system", "content": "You are a careful assistant. Answer from the document only."},
				{"role": "user", "content": doc + "\n\nWhat is the delivery term?"},
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		for _, target := range []config.ProviderType{config.ProviderTypeProxy, config.ProviderTypeOpenAI} {
			b.Run(size.name+"_to_"+string(target), func(b *testing.B) {
				prx := NewTestProxyBuilder().
					WithCredentials(config.CredentialConfig{Name: "c", Type: target, BaseURL: "http://upstream.invalid", APIKey: "master-key", RPM: 100000000}).
					WithMaxBodySizeMB(35).
					Build()
				prx.client.Transport = inMemoryUpstream{sse: agentChatSSE}
				b.SetBytes(int64(len(body)))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
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
}
