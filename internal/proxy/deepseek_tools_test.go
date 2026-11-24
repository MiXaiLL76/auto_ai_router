package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/deepseektools"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/require"
)

// deepSeekFake is a fake DeepSeek: it serves the chat endpoint (answering
// with a tool call the first time and with the answer the second) and the
// two tool endpoints.
type deepSeekFake struct {
	mu sync.Mutex
	// chatCalls counts the chat requests the fake served.
	chatCalls int
	// searchCalls and fetchCalls count the tool calls.
	searchCalls int
	fetchCalls  int
	// lastChatBody is the body of the last chat request.
	lastChatBody map[string]json.RawMessage
	// searchResults is what the search endpoint answers.
	searchResults []map[string]any
	// fetchContent is what the fetch endpoint answers.
	fetchContent string
	// searchStatus overrides the search endpoint's status code.
	searchStatus int
	// baseURL is the fake's address; the test sets it after starting the
	// server.
	baseURL string
}

func (f *deepSeekFake) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			f.mu.Lock()
			f.searchCalls++
			status := f.searchStatus
			results := f.searchResults
			f.mu.Unlock()
			if status != 0 {
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":{"message":"rate limited","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/tools/fetch":
			f.mu.Lock()
			f.fetchCalls++
			content := f.fetchContent
			f.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"title": "Example", "content": content})
		case "/v1/chat/completions":
			f.mu.Lock()
			f.chatCalls++
			call := f.chatCalls
			var parsed map[string]json.RawMessage
			_ = json.Unmarshal(body, &parsed)
			f.lastChatBody = parsed
			f.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			// A request without the router's tools is answered at once: the
			// fake only asks for a tool when the request asked for one.
			if !strings.Contains(string(body), deepseektools.SearchToolName) &&
				!strings.Contains(string(body), deepseektools.FetchToolName) {
				_, _ = w.Write([]byte(`{
					"id":"chatcmpl-plain","object":"chat.completion","model":"deepseek-flash",
					"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hi!"}}],
					"usage":{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}
				}`))
				return
			}
			if call == 1 {
				// The first answer asks for the router's search tool.
				_, _ = w.Write([]byte(`{
					"id":"chatcmpl-1","object":"chat.completion","model":"deepseek-flash",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + deepseektools.SearchToolName + `","arguments":"{\"query\":\"deepseek v4\",\"count\":3}"}}]}}],
					"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}
				}`))
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"deepseek-flash",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"DeepSeek V4 was released."}}],
				"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
}

// toolCountsFromMetadata reads the tool counters the spend row carries in its
// metadata: the usage object the router wrote.
func toolCountsFromMetadata(t *testing.T, metadata string) (webSearch int, webFetch int) {
	t.Helper()
	var envelope struct {
		UsageObject struct {
			PromptTokensDetails struct {
				WebSearchRequests int `json:"web_search_requests"`
				WebFetchRequests  int `json:"web_fetch_requests"`
			} `json:"prompt_tokens_details"`
			ServerToolUse struct {
				WebSearchRequests int `json:"web_search_requests"`
				WebFetchRequests  int `json:"web_fetch_requests"`
			} `json:"server_tool_use"`
		} `json:"usage_object"`
	}
	if err := json.Unmarshal([]byte(metadata), &envelope); err != nil {
		t.Fatalf("decoding the spend metadata: %v", err)
	}
	webSearch = envelope.UsageObject.ServerToolUse.WebSearchRequests
	if webSearch == 0 {
		webSearch = envelope.UsageObject.PromptTokensDetails.WebSearchRequests
	}
	webFetch = envelope.UsageObject.ServerToolUse.WebFetchRequests
	if webFetch == 0 {
		webFetch = envelope.UsageObject.PromptTokensDetails.WebFetchRequests
	}
	return webSearch, webFetch
}

// deepSeekTestProxy builds a proxy whose single credential points at the
// fake DeepSeek.
func deepSeekTestProxy(t *testing.T, fake *deepSeekFake) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	credential := config.CredentialConfig{
		Name:          "deepseek",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       fake.serverURL(),
		APIKey:        "ds-key",
		RPM:           100,
		TPM:           100000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	// DeepSeek has no native /v1/responses endpoint, so a Responses request
	// is converted to Chat Completions for it -- the same shape the adapter
	// works in.
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name:                 "deepseek-flash",
		Model:                "deepseek-flash",
		Credential:           "deepseek",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "deepseek", Type: config.ProviderTypeOpenAI, APIKey: "ds-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
		CacheReadInputTokenCost: 1e-7,
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low":    0.000975,
			"search_context_size_medium": 0.000975,
			"search_context_size_high":   0.000975,
		},
		WebSearchBillingUnit: "per_query",
		ToolCostPerCall:      map[string]float64{"fetch": 0.000975},
	}})
	return prx, dbStub
}

// serverURL returns the fake's base URL (set by the test before use).
func (f *deepSeekFake) serverURL() string { return f.baseURL }

// TestDeepSeekToolLoopRunsSearchAndBillsIt is the end-to-end check of the
// router-run tool loop: the model asks for a search, the router runs it, the
// result goes back, the model answers -- and the request is billed for the
// search and for the tokens of both steps.
func TestDeepSeekToolLoopRunsSearchAndBillsIt(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title":   "DeepSeek V4 released",
			"url":     "https://example.com/v4",
			"summary": "DeepSeek has released the V4 model.",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decoding the response: %v", err)
	}
	if len(response.Choices) == 0 || !strings.Contains(response.Choices[0].Message.Content, "DeepSeek V4") {
		t.Errorf("the final answer is missing: %s", rec.Body.String())
	}

	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}

	// The second chat request must carry the tool result.
	if fake.lastChatBody == nil {
		t.Fatal("no chat body was captured")
	}
	rawMessages, ok := fake.lastChatBody["messages"]
	if !ok {
		t.Fatal("the second request has no messages")
	}
	var messages []map[string]any
	if err := json.Unmarshal(rawMessages, &messages); err != nil {
		t.Fatalf("decoding the messages: %v", err)
	}
	if len(messages) < 3 {
		t.Fatalf("the conversation has %d messages, want at least 3: %v", len(messages), messages)
	}
	toolMessage := messages[len(messages)-1]
	if toolMessage["role"] != "tool" {
		t.Errorf("the last message is %v, want a tool message", toolMessage["role"])
	}
	if toolMessage["tool_call_id"] != "call_1" {
		t.Errorf("the tool message carries the call id %v, want call_1", toolMessage["tool_call_id"])
	}
	content, _ := toolMessage["content"].(string)
	if !strings.Contains(content, "DeepSeek V4 released") {
		t.Errorf("the tool result does not carry the search results: %s", content)
	}

	// The spend row: the search is billed, and the tokens of both steps.
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if searches, _ := toolCountsFromMetadata(t, entry.Metadata); searches != 1 {
		t.Errorf("the web search count = %d, want 1", searches)
	}
	// Both steps' tokens: 100+200 prompt, 10+20 completion.
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps summed)", entry.PromptTokens)
	}
	if entry.CompletionTokens != 30 {
		t.Errorf("CompletionTokens = %d, want 30 (both steps summed)", entry.CompletionTokens)
	}
	// The search costs 0.000975, the tokens 300*1e-6 + 30*2e-6.
	wantSpend := 300*1e-6 + 30*2e-6 + 0.000975
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestDeepSeekToolLoopWithoutToolsIsUntouched checks that a request without
// the built-in tools is served normally: the adapter must not engage.
func TestDeepSeekToolLoopWithoutToolsIsUntouched(t *testing.T) {
	fake := &deepSeekFake{}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{"model":"deepseek-flash","messages":[{"role":"user","content":"Say hi."}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 0 {
		t.Errorf("the router made %d search calls, want 0", fake.searchCalls)
	}
	if fake.chatCalls != 1 {
		t.Errorf("the model was called %d times, want 1", fake.chatCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	if searches, _ := toolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata); searches != 0 {
		t.Errorf("the web search count = %d, want 0", searches)
	}
}

// TestDeepSeekEmptySearchIsNotBilled checks the billing rule: a search that
// returned nothing is not a billable event.
func TestDeepSeekEmptySearchIsNotBilled(t *testing.T) {
	fake := &deepSeekFake{searchResults: []map[string]any{}}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if searches, _ := toolCountsFromMetadata(t, entry.Metadata); searches != 0 {
		t.Errorf("the web search count = %d, want 0 (an empty search is not billable)", searches)
	}
	// Only the tokens are billed.
	wantSpend := 300*1e-6 + 30*2e-6
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v (tokens only)", entry.Spend, wantSpend)
	}
}

// TestDeepSeekSearchErrorIsNotBilled checks that a failed search costs
// nothing.
func TestDeepSeekSearchErrorIsNotBilled(t *testing.T) {
	fake := &deepSeekFake{searchStatus: http.StatusTooManyRequests}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if searches, _ := toolCountsFromMetadata(t, entry.Metadata); searches != 0 {
		t.Errorf("the web search count = %d, want 0 (a failed search is not billable)", searches)
	}
}

// TestDeepSeekResponsesEndpointRunsTheLoop checks the same loop on the
// /v1/responses endpoint.
func TestDeepSeekResponsesEndpointRunsTheLoop(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"input": [{"role": "user", "content": [{"type": "input_text", "text": "What is new in DeepSeek V4?"}]}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	if searches, _ := toolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata); searches != 1 {
		t.Errorf("the web search count = %d, want 1", searches)
		t.Logf("metadata: %s", dbStub.loggedEntries[0].Metadata)
	}
}

// TestDeepSeekFetchIsBilled checks the fetch tool's own price: a fetch that
// returned content is billed at tool_cost_per_call.fetch.
func TestDeepSeekFetchIsBilled(t *testing.T) {
	fake := &deepSeekFake{
		fetchContent: "# Example\n\nThe page content.",
	}
	// The fake asks for the fetch tool on the first turn.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/fetch":
			fake.mu.Lock()
			fake.fetchCalls++
			content := fake.fetchContent
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"title": "Example", "content": content})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = fmt.Fprintf(w, `{
					"id":"chatcmpl-1","object":"chat.completion","model":"deepseek-flash",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[{"id":"call_f","type":"function","function":{"name":%q,"arguments":"{\"url\":\"https://example.com/a\"}"}}]}}],
					"usage":{"prompt_tokens":50,"completion_tokens":5,"total_tokens":55}
				}`, deepseektools.FetchToolName)
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"deepseek-flash",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Here is the page."}}],
				"usage":{"prompt_tokens":60,"completion_tokens":10,"total_tokens":70}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is on this page?"}],
		"tools": [{"type": "web_fetch"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.fetchCalls != 1 {
		t.Errorf("the router made %d fetch calls, want 1", fake.fetchCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if _, fetches := toolCountsFromMetadata(t, entry.Metadata); fetches != 1 {
		t.Errorf("the web fetch count = %d, want 1", fetches)
	}
	// The fetch costs 0.000975, the tokens 110*1e-6 + 15*2e-6.
	wantSpend := 110*1e-6 + 15*2e-6 + 0.000975
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestDeepSeekToolLoopWithV1BaseURL checks the credential whose base URL
// already ends with /v1: the tool endpoints live under /v1, so the path must
// not be doubled.
func TestDeepSeekToolLoopWithV1BaseURL(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL + "/v1"

	prx, dbStub := deepSeekTestProxy(t, fake)
	// The fake serves /v1/chat/completions; with the /v1 base the router
	// asks for /v1/chat/completions too, and the tools for /v1/tools/search.
	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1 (the tools path must not be doubled)", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	if searches, _ := toolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata); searches != 1 {
		t.Errorf("the web search count = %d, want 1", searches)
	}
}

// TestDeepSeekStreamingToolLoopRunsSearch checks the streaming variant: the
// model asks for a search mid-stream, the router runs it, and the client
// gets the final answer's stream -- not the tool call.
func TestDeepSeekStreamingToolLoopRunsSearch(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			fake.mu.Lock()
			fake.searchCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			var parsed map[string]json.RawMessage
			_ = json.Unmarshal(body, &parsed)
			fake.lastChatBody = parsed
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				// The first stream asks for the search tool.
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"` + deepseektools.SearchToolName + `","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"DeepSeek V4 was released."}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}
	out := rec.Body.String()
	if !strings.Contains(out, "DeepSeek V4 was released.") {
		t.Errorf("the client did not get the answer: %s", out)
	}
	// The client must not see the tool call: it declared a web_search, not
	// the router's function.
	if strings.Contains(out, deepseektools.SearchToolName) {
		t.Errorf("the client saw the router's function tool: %s", out)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if searches, _ := toolCountsFromMetadata(t, entry.Metadata); searches != 1 {
		t.Errorf("the web search count = %d, want 1", searches)
	}
	// Both steps' tokens.
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps summed)", entry.PromptTokens)
	}
}

// TestDeepSeekForceStreamToolLoopRunsSearch covers the path a non-streaming
// client actually takes when the router forces stream:true upstream: the
// provider's SSE is reassembled into the Chat JSON, and a tool call in it
// must still be run by the router.
func TestDeepSeekForceStreamToolLoopRunsSearch(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			fake.mu.Lock()
			fake.searchCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			var parsed map[string]json.RawMessage
			_ = json.Unmarshal(body, &parsed)
			fake.lastChatBody = parsed
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				// The first stream asks for the search tool.
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"` + deepseektools.SearchToolName + `","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"DeepSeek V4 was released."}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)
	prx.forceStreamForTTFT = true

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}
	out := rec.Body.String()
	if !strings.Contains(out, "DeepSeek V4 was released.") {
		t.Errorf("the client did not get the answer: %s", out)
	}
	if strings.Contains(out, deepseektools.SearchToolName) {
		t.Errorf("the client saw the router's function tool: %s", out)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if searches, _ := toolCountsFromMetadata(t, entry.Metadata); searches != 1 {
		t.Errorf("the web search count = %d, want 1", searches)
	}
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps summed)", entry.PromptTokens)
	}
}

// TestDeepSeekEmptySearchIsNotBilledInSpendRow проверяет правило "пустые
// результаты — $0": строка spend получает нулевой счётчик поисков.
func TestDeepSeekEmptySearchIsNotBilledInSpendRow(t *testing.T) {
	fake := &deepSeekFake{searchResults: []map[string]any{}}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)
	body := `{"model":"deepseek-flash","messages":[{"role":"user","content":"q"}],"tools":[{"type":"web_search"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, 200, rec.Code, rec.Body.String())
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	searches, _ := toolCountsFromMetadata(t, entry.Metadata)
	require.Equal(t, 0, searches, "an empty search was billed")
	// The request still cost its tokens.
	require.Greater(t, entry.Spend, 0.0)
}

// TestDeepSeekKeepsReasoningAndToolHistory checks that the loop carries the
// model's own state across the steps: its reasoning_content and its tool
// calls, so the second request is a continuation of the first rather than a
// fresh conversation.
func TestDeepSeekKeepsReasoningAndToolHistory(t *testing.T) {
	fake := &deepSeekFake{searchResults: []map[string]any{{"title": "T", "url": "https://e.com"}}}
	// Первый ответ несёт reasoning_content вместе с вызовом инструмента.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"results":[{"title":"T","url":"https://e.com"}]}`))
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			var parsed map[string]json.RawMessage
			_ = json.Unmarshal(body, &parsed)
			fake.lastChatBody = parsed
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = w.Write([]byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","reasoning_content":"I need to search for that","tool_calls":[{"id":"c1","type":"function","function":{"name":"deepseek_web_search","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}`))
				return
			}
			_, _ = w.Write([]byte(`{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, _ := deepSeekTestProxy(t, fake)
	body := `{"model":"deepseek-flash","messages":[{"role":"user","content":"q"}],"tools":[{"type":"web_search"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)
	require.Equal(t, 200, rec.Code, rec.Body.String())

	require.NotNil(t, fake.lastChatBody)
	rawMessages, ok := fake.lastChatBody["messages"]
	require.True(t, ok, "the second request has no messages")
	var messages []map[string]any
	require.NoError(t, json.Unmarshal(rawMessages, &messages))
	require.Len(t, messages, 3, "the conversation is not the user's, the tool call and its result")

	// Ассистентский ход сохраняет вызов инструмента с его id.
	assistant := messages[1]
	require.Equal(t, "assistant", assistant["role"])
	calls, ok := assistant["tool_calls"].([]any)
	require.True(t, ok, "the assistant's tool calls were dropped")
	require.Len(t, calls, 1)
	first := calls[0].(map[string]any)
	require.Equal(t, "c1", first["id"], "the tool call's id was lost")

	// Результат инструмента привязан к тому же id.
	tool := messages[2]
	require.Equal(t, "tool", tool["role"])
	require.Equal(t, "c1", tool["tool_call_id"], "the tool result lost the call id")
}

// TestDeepSeekKeepsReasoningContent checks that the model's own reasoning
// reaches the next request: it is part of the conversation the loop
// continues, not something the router drops.
func TestDeepSeekKeepsReasoningContent(t *testing.T) {
	fake := &deepSeekFake{searchResults: []map[string]any{{"title": "T", "url": "https://e.com"}}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"results":[{"title":"T","url":"https://e.com"}]}`))
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			var parsed map[string]json.RawMessage
			_ = json.Unmarshal(body, &parsed)
			fake.lastChatBody = parsed
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = w.Write([]byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","reasoning_content":"I need to search for that","tool_calls":[{"id":"c1","type":"function","function":{"name":"deepseek_web_search","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}`))
				return
			}
			_, _ = w.Write([]byte(`{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"done"}}]}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, _ := deepSeekTestProxy(t, fake)
	body := `{"model":"deepseek-flash","messages":[{"role":"user","content":"q"}],"tools":[{"type":"web_search"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)
	require.Equal(t, 200, rec.Code, rec.Body.String())

	require.NotNil(t, fake.lastChatBody)
	rawMessages, ok := fake.lastChatBody["messages"]
	require.True(t, ok)
	var messages []map[string]any
	require.NoError(t, json.Unmarshal(rawMessages, &messages))
	require.Len(t, messages, 3)
	require.Equal(t, "I need to search for that", messages[1]["reasoning_content"],
		"the model's reasoning did not reach the next request")
}

// TestDeepSeekSuccessfulToolIsBilledWhenTheModelFails checks the rule
// "учитывать успешный инструмент и при последующей ошибке LLM": the search
// that ran and returned results is billed even when the model that follows it
// fails.
func TestDeepSeekSuccessfulToolIsBilledWhenTheModelFails(t *testing.T) {
	fake := &deepSeekFake{searchResults: []map[string]any{{"title": "T", "url": "https://e.com"}}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"results":[{"title":"T","url":"https://e.com"}]}`))
		case "/v1/chat/completions":
			_ = body
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = w.Write([]byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"deepseek_web_search","arguments":"{\"query\":\"q\"}"}}]}}]}`))
				return
			}
			// Второй шаг падает: провайдер отвечает ошибкой.
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"upstream failure","type":"server_error"}}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)
	body := `{"model":"deepseek-flash","messages":[{"role":"user","content":"q"}],"tools":[{"type":"web_search"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Len(t, dbStub.loggedEntries, 1, "the failed request wrote no spend row")
	entry := dbStub.loggedEntries[0]
	searches, _ := toolCountsFromMetadata(t, entry.Metadata)
	require.Equal(t, 1, searches, "the successful search was not billed after the model failed")
}

// TestDeepSeekStreamingWithAClientToolCallSendsOnce checks that a stream
// whose answer is the CLIENT's own tool call is not mistaken for the model
// asking for one of the router's tools.
//
// The check that decides whether to loop must count only the router's own
// tools: a coding agent's read_file or edit is the client's business, and
// the router must relay it as it is. Counting it made the router send the
// very same request to the provider a second time -- the client never
// noticed (it got the second answer's stream), but the request was billed
// twice: the prompt was counted twice, and the difference landed entirely in
// the uncached input, the expensive part.
func TestDeepSeekStreamingWithAClientToolCallSendsOnce(t *testing.T) {
	fake := &deepSeekFake{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			fake.mu.Lock()
			fake.searchCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			// The model's answer is the CLIENT's own tool call -- a coding
			// agent's read_file. The router must relay it, not run it and
			// not ask the model again.
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"main.go\"}"}}]}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "Read main.go"}],
		"tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// The provider was asked exactly once: the client's own tool call is
	// not the router's business.
	if fake.chatCalls != 1 {
		t.Errorf("the provider was asked %d times, want 1 (a client tool call is not the router's tool)", fake.chatCalls)
	}
	if fake.searchCalls != 0 {
		t.Errorf("the router made %d search calls, want 0", fake.searchCalls)
	}
	// The client's stream came through as it is.
	if !strings.Contains(rec.Body.String(), "read_file") {
		t.Errorf("the client did not get its own tool call: %s", rec.Body.String())
	}
	// And the request was billed once.
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if entry.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100 (the one send's own)", entry.PromptTokens)
	}
	if entry.CompletionTokens != 10 {
		t.Errorf("CompletionTokens = %d, want 10 (the one send's own)", entry.CompletionTokens)
	}
}

// TestDeepSeekLoopSumsTheCachedAndReasoningTokens checks that a genuine
// loop bills every step's cached and reasoning tokens, not only the last
// step's.
//
// The loop's steps are separate upstream requests, and the client is charged
// for all of them. The cached tokens are the cheap part of the input, and
// the reasoning tokens are output: leaving the last step's figure alone
// under-bills a real search loop.
func TestDeepSeekLoopSumsTheCachedAndReasoningTokens(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			fake.mu.Lock()
			fake.searchCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"deepseek_web_search","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":40},"completion_tokens_details":{"reasoning_tokens":7}}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"DeepSeek V4 was released."}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220,"prompt_tokens_details":{"cached_tokens":60},"completion_tokens_details":{"reasoning_tokens":9}}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.chatCalls != 2 {
		t.Fatalf("the model was called %d times, want 2", fake.chatCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps)", entry.PromptTokens)
	}
	if entry.CompletionTokens != 30 {
		t.Errorf("CompletionTokens = %d, want 30 (both steps)", entry.CompletionTokens)
	}
	// The cached tokens of both steps: 40 + 60, and the reasoning tokens of
	// both steps: 7 + 9. They are in the metadata (the spend row's columns
	// carry the token totals only).
	var envelope struct {
		UsageObject struct {
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
			CompletionTokensDetails struct {
				ReasoningTokens int `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage_object"`
	}
	if err := json.Unmarshal([]byte(entry.Metadata), &envelope); err != nil {
		t.Fatalf("decoding the spend metadata: %v", err)
	}
	if envelope.UsageObject.PromptTokensDetails.CachedTokens != 100 {
		t.Errorf("the cached tokens = %d, want 100 (both steps)",
			envelope.UsageObject.PromptTokensDetails.CachedTokens)
	}
	if envelope.UsageObject.CompletionTokensDetails.ReasoningTokens != 16 {
		t.Errorf("the reasoning tokens = %d, want 16 (both steps)",
			envelope.UsageObject.CompletionTokensDetails.ReasoningTokens)
	}
}

// TestDeepSeekStreamingMidLoopFailureDoesNotRelayAClosedBody checks what
// the client gets when a step of the loop fails after a successful first
// step.
//
// The loop read the provider's stream whole (and closed it) before it ran
// the tool, so the fallback that relayed the original response would have
// read from a closed body: the client would get a 200 with an empty stream
// and the log would say "invalid Read on closed Body" instead of the real
// failure.
//
// The client must get the error, not an empty stream.
func TestDeepSeekStreamingMidLoopFailureDoesNotRelayAClosedBody(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search":
			fake.mu.Lock()
			fake.searchCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				// The first stream asks for the router's search.
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"deepseek_web_search","arguments":"{\"query\":\"deepseek v4\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			// The continuation fails: the upstream is gone.
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":{"message":"upstream is down"}}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	out := rec.Body.String()
	// The client must not get an empty stream that ends without an answer:
	// it would look like a successful request with no content.
	if strings.Contains(out, "invalid Read on closed Body") {
		t.Errorf("the client was served a closed body: %s", out)
	}
	if rec.Code == http.StatusOK && !strings.Contains(out, "finish_reason") &&
		!strings.Contains(out, "error") {
		t.Errorf("the client got a 200 with no answer and no error: %s", out)
	}
	// It gets the failure, and it is the real one.
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 (the upstream failed after the tool ran): %s", rec.Code, out)
	}
	if !strings.Contains(out, "error") {
		t.Errorf("the client got no error: %s", out)
	}
	// The search that already ran is still billed.
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	searches, _ := toolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if searches != 1 {
		t.Errorf("the web search count = %d, want 1 (the search already ran)", searches)
	}
}

// TestDeepSeekStreamWithoutOurToolIsRelayedAsItArrives checks that a streaming
// request which offers none of the router's tools is relayed to the client as
// it arrives, not buffered.
//
// The router-run loop exists to catch a mid-stream call of one of the router's
// tools, and it reads the whole answer to see one. On a request that offers no
// such tool the model cannot make that call, so the buffering bought nothing
// and cost the client its stream: the first byte arrived only once the last was
// generated. A client that gives up on a slow first token then tore down a
// request that had been answering all along.
//
// The fake provider writes one chunk and then waits for the client to see it.
// If the router relays, the write has already reached the client and the
// provider is released at once; if it buffers, the wait times out, which is
// the bug.
func TestDeepSeekStreamWithoutOurToolIsRelayedAsItArrives(t *testing.T) {
	fake := &deepSeekFake{}
	clientSawFirstChunk := make(chan struct{})
	var relayed bool
	var relayedMu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		if r.URL.Path != "/v1/chat/completions" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		fake.mu.Lock()
		fake.chatCalls++
		fake.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		// The first fragment of the answer.
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"hello"}}]}` + "\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		// Did the client see it before the answer is finished?
		select {
		case <-clientSawFirstChunk:
			relayedMu.Lock()
			relayed = true
			relayedMu.Unlock()
		case <-time.After(1500 * time.Millisecond):
		}
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"world"}}]}` + "\n\n"))
		_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, _ := deepSeekTestProxy(t, fake)

	// A request with a plain user message and no tools at all: the router has
	// nothing of its own to run, so it must not buffer.
	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "hi"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")

	rec := &firstByteRecorder{onFirst: func() {
		select {
		case <-clientSawFirstChunk:
		default:
			close(clientSawFirstChunk)
		}
	}}
	prx.ProxyRequest(rec, req)

	relayedMu.Lock()
	got := relayed
	relayedMu.Unlock()
	if !got {
		t.Fatalf("the client did not get the first fragment before the answer was generated: the stream was buffered")
	}
	if !strings.Contains(rec.body.String(), "hello") || !strings.Contains(rec.body.String(), "world") {
		t.Fatalf("the answer did not come through: %s", rec.body.String())
	}
	if fake.chatCalls != 1 {
		t.Errorf("the provider was asked %d times, want 1", fake.chatCalls)
	}
}

// TestDeepSeekStreamWithOurToolStillRunsTheLoop is the other side of the gate:
// a request that does declare one of the router's tools still needs the
// buffered loop, because the model may ask for it mid-stream.
func TestDeepSeekStreamWithOurToolStillRunsTheLoop(t *testing.T) {
	fake := &deepSeekFake{
		searchResults: []map[string]any{{
			"title": "DeepSeek V4 released",
			"url":   "https://example.com/v4",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, _ := deepSeekTestProxy(t, fake)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "What is new in DeepSeek V4?"}],
		"tools": [{"type": "web_search"}],
		"stream": true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	out := rec.Body.String()
	if fake.searchCalls != 1 {
		t.Errorf("the router made %d search calls, want 1", fake.searchCalls)
	}
	if fake.chatCalls != 2 {
		t.Errorf("the model was called %d times, want 2", fake.chatCalls)
	}
	if !strings.Contains(out, "DeepSeek V4 was released.") {
		t.Errorf("the client did not get the answer: %s", out)
	}
	if strings.Contains(out, deepseektools.SearchToolName) {
		t.Errorf("the client saw the router's function tool: %s", out)
	}
}

// firstByteRecorder is an http.ResponseWriter that reports the first Write, so
// a test can tell whether a relayed stream reached the client before the
// provider finished.
type firstByteRecorder struct {
	header  http.Header
	status  int
	body    bytes.Buffer
	onFirst func()
	fired   bool
}

func (r *firstByteRecorder) Header() http.Header {
	if r.header == nil {
		r.header = http.Header{}
	}
	return r.header
}

func (r *firstByteRecorder) WriteHeader(status int) { r.status = status }

func (r *firstByteRecorder) Write(p []byte) (int, error) {
	if !r.fired {
		r.fired = true
		if r.onFirst != nil {
			r.onFirst()
		}
	}
	return r.body.Write(p)
}

func (r *firstByteRecorder) Flush() {}
