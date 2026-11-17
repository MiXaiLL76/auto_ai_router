package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// TestMoonshotExamplesArePrinted writes the request the router sends to
// Moonshot, the client's answer and the spend row, so the documentation's
// examples are the real ones rather than a hand-written guess.
//
// Run it with -v to see them:
//
//	go test ./internal/proxy/ -run TestMoonshotExamplesArePrinted -v
func TestMoonshotExamplesArePrinted(t *testing.T) {
	var toolRequest map[string]any
	var toolPath string
	var toolAuth string
	var chatRequests []map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			toolPath = r.URL.Path
			toolAuth = r.Header.Get("Authorization")
			_ = json.Unmarshal(body, &toolRequest)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"search_results":[
				{"title":"Moonshot AI releases Kimi K2","url":"https://moonshot.ai/news/kimi-k2",
				 "snippet":"Moonshot AI has released the Kimi K2 model.","content":"Kimi K2 is Moonshot AI's ..."}
			]}`))
		case "/v1/chat/completions":
			var parsed map[string]any
			_ = json.Unmarshal(body, &parsed)
			chatRequests = append(chatRequests, parsed)
			w.Header().Set("Content-Type", "application/json")
			if len(chatRequests) == 1 {
				_, _ = fmt.Fprintf(w, `{
					"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k3.1-pro",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":%q,"arguments":"{\"text_query\":\"kimi k2 release\",\"limit\":5,\"timeout_seconds\":30,\"include_content\":true}"}}]}}],
					"usage":{"prompt_tokens":412,"completion_tokens":37,"total_tokens":449}
				}`, moonshottools.SearchToolName)
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k3.1-pro",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant",
					"content":"Moonshot AI released Kimi K2 on 2025-07-11."}}],
				"usage":{"prompt_tokens":1587,"completion_tokens":64,"total_tokens":1651}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	fake := &moonshotFake{baseURL: server.URL}
	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k3.1-pro",
		"messages": [{"role": "user", "content": "When was Kimi K2 released?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	t.Logf("=== the client's request ===\n%s", body)
	t.Logf("=== the client's raw answer (status %d) ===\n%s", rec.Code, rec.Body.String())
	t.Logf("=== the router's call to Moonshot ===\nPOST %s\nAuthorization: %s\n%s",
		toolPath, toolAuth, moonshotMustJSON(t, toolRequest))
	t.Logf("=== the router's second call to the model (the tool result goes back) ===\n%s",
		moonshotMustJSON(t, chatRequests[1]))
	t.Logf("=== the client's answer ===\n%s", rec.Body.String())
	for _, entry := range dbStub.loggedEntries {
		t.Logf("=== the spend row ===\nSpend: %v\nPromptTokens: %d\nCompletionTokens: %d\nMetadata: %s",
			entry.Spend, entry.PromptTokens, entry.CompletionTokens, entry.Metadata)
	}
}

// moonshotMustJSON renders a value as indented JSON, or "<unparsable>" on failure.
func moonshotMustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return "<unparsable>"
	}
	return string(encoded)
}
