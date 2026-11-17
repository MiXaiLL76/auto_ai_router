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

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/mixaill76/auto_ai_router/internal/moonshottools"
)

// moonshotFake is a fake Moonshot: it serves the chat endpoint (answering
// with a tool call the first time and with the answer the second) and the
// three tool endpoints.
type moonshotFake struct {
	mu sync.Mutex
	// chatCalls counts the chat requests the fake served.
	chatCalls int
	// searchCalls, searchProCalls and fetchCalls count the tool calls.
	searchCalls    int
	searchProCalls int
	fetchCalls     int
	// lastChatBody is the body of the last chat request.
	lastChatBody map[string]json.RawMessage
	// searchResults is what the search endpoints answer.
	searchResults []map[string]any
	// fetchMarkdown is what the fetch endpoint answers.
	fetchMarkdown string
	// fetchError is the provider's own error code, when it refuses.
	fetchError string
	// searchStatus overrides the search endpoint's status code.
	searchStatus int
	// baseURL is the fake's address; the test sets it after starting the
	// server.
	baseURL string
}

// serverURL returns the fake's base URL (set by the test before use).
func (f *moonshotFake) serverURL() string { return f.baseURL }

// toolCall is what the fake answers with on the first turn: a tool call for
// the named tool with the given arguments.
func (f *moonshotFake) toolCall(name, arguments string) string {
	return fmt.Sprintf(`{
		"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k2",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
			"tool_calls":[{"id":"call_1","type":"function","function":{"name":%q,"arguments":%q}}]}}],
		"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}
	}`, name, arguments)
}

// handler serves the chat endpoint and the three tool endpoints.
func (f *moonshotFake) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search", "/v1/tools/search_pro":
			f.mu.Lock()
			if r.URL.Path == "/v1/tools/search" {
				f.searchCalls++
			} else {
				f.searchProCalls++
			}
			status := f.searchStatus
			results := f.searchResults
			f.mu.Unlock()
			if status != 0 {
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"error":{"message":"rate limited","type":"invalid_request_error"}}`))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"search_results": results})
		case "/v1/tools/fetch":
			f.mu.Lock()
			f.fetchCalls++
			markdown := f.fetchMarkdown
			fetchError := f.fetchError
			f.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"title": "Example", "markdown": markdown, "error": fetchError,
			})
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
			if !strings.Contains(string(body), moonshottools.SearchToolName) &&
				!strings.Contains(string(body), moonshottools.SearchProToolName) &&
				!strings.Contains(string(body), moonshottools.FetchToolName) {
				_, _ = w.Write([]byte(`{
					"id":"chatcmpl-plain","object":"chat.completion","model":"kimi-k2",
					"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Hi!"}}],
					"usage":{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}
				}`))
				return
			}
			if call == 1 {
				// The first answer asks for the Basic search.
				_, _ = w.Write([]byte(f.toolCall(moonshottools.SearchToolName, `{"text_query":"kimi k2","limit":3}`)))
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k2",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Kimi K2 was released."}}],
				"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
}

// moonshotToolCountsFromMetadata reads the router's own tool counters from
// the spend row's metadata.
func moonshotToolCountsFromMetadata(t *testing.T, metadata string) (search int, searchPro int, fetch int) {
	t.Helper()
	var envelope struct {
		UsageObject struct {
			ServerToolUse struct {
				MoonshotSearchRequests    int `json:"moonshot_search_requests"`
				MoonshotSearchProRequests int `json:"moonshot_search_pro_requests"`
				MoonshotFetchRequests     int `json:"moonshot_fetch_requests"`
			} `json:"server_tool_use"`
		} `json:"usage_object"`
	}
	if err := json.Unmarshal([]byte(metadata), &envelope); err != nil {
		t.Fatalf("decoding the spend metadata: %v", err)
	}
	return envelope.UsageObject.ServerToolUse.MoonshotSearchRequests,
		envelope.UsageObject.ServerToolUse.MoonshotSearchProRequests,
		envelope.UsageObject.ServerToolUse.MoonshotFetchRequests
}

// moonshotTestProxy builds a proxy whose single credential points at the
// fake Moonshot.
func moonshotTestProxy(t *testing.T, fake *moonshotFake) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	credential := config.CredentialConfig{
		Name:          "moonshot",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       fake.serverURL(),
		APIKey:        "ms-key",
		RPM:           100,
		TPM:           100000,
		MoonshotTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	// Moonshot has no native /v1/responses endpoint, so a Responses request
	// is converted to Chat Completions for it -- the same shape the adapter
	// works in.
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{
		{
			Name:                 "kimi-k2",
			Model:                "kimi-k2",
			Credential:           "moonshot",
			PassthroughResponses: &noPassthrough,
		},
		{
			Name:                 "kimi-k3.1-pro",
			Model:                "kimi-k3.1-pro",
			Credential:           "moonshot",
			PassthroughResponses: &noPassthrough,
		},
	})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "moonshot", Type: config.ProviderTypeOpenAI, APIKey: "ms-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.SetMoonshotToolPrices(&models.ToolPricesProvider{
		SearchCostPerCall:    0.013,
		SearchProCostPerCall: 0.013,
		FetchCostPerCall:     0.013,
		Rate:                 1.3,
	})
	prx.priceRegistry.Update(map[string]*models.ModelPrice{
		"kimi-k2": {
			InputCostPerToken:  1e-6,
			OutputCostPerToken: 2e-6,
		},
		"kimi-k3.1-pro": {
			InputCostPerToken:  1e-6,
			OutputCostPerToken: 2e-6,
		},
	})
	return prx, dbStub
}

// TestMoonshotToolLoopRunsSearchAndBillsIt is the end-to-end check of the
// router-run tool loop: the model asks for a search, the router runs it, the
// result goes back, the model answers -- and the request is billed for the
// search and for the tokens of both steps.
func TestMoonshotToolLoopRunsSearchAndBillsIt(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title":   "Kimi K2 released",
			"url":     "https://example.com/k2",
			"snippet": "Moonshot has released the K2 model.",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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
	if len(response.Choices) == 0 || !strings.Contains(response.Choices[0].Message.Content, "Kimi K2") {
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
	if !strings.Contains(content, "Kimi K2 released") {
		t.Errorf("the tool result does not carry the search results: %s", content)
	}

	// The spend row: the search is billed, and the tokens of both steps.
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	search, searchPro, fetch := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
	if searchPro != 0 || fetch != 0 {
		t.Errorf("the other tool counts = %d/%d, want 0/0", searchPro, fetch)
	}
	// Both steps' tokens: 100+200 prompt, 10+20 completion.
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps summed)", entry.PromptTokens)
	}
	if entry.CompletionTokens != 30 {
		t.Errorf("CompletionTokens = %d, want 30 (both steps summed)", entry.CompletionTokens)
	}
	// The search costs 0.013, the tokens 300*1e-6 + 30*2e-6.
	wantSpend := 300*1e-6 + 30*2e-6 + 0.013
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestMoonshotToolLoopWithoutToolsIsUntouched checks that a request without
// the built-in tools is served normally: the adapter must not engage.
func TestMoonshotToolLoopWithoutToolsIsUntouched(t *testing.T) {
	fake := &moonshotFake{}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{"model":"kimi-k2","messages":[{"role":"user","content":"Say hi."}]}`
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
	search, searchPro, fetch := moonshotToolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if search != 0 || searchPro != 0 || fetch != 0 {
		t.Errorf("the tool counts = %d/%d/%d, want 0/0/0", search, searchPro, fetch)
	}
}

// TestMoonshotEmptySearchIsNotBilled checks the billing rule: a search that
// returned nothing is not a billable event.
func TestMoonshotEmptySearchIsNotBilled(t *testing.T) {
	fake := &moonshotFake{searchResults: []map[string]any{}}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
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
	search, _, _ := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if search != 0 {
		t.Errorf("the moonshot search count = %d, want 0 (an empty search is not billable)", search)
	}
	// Only the tokens are billed.
	wantSpend := 300*1e-6 + 30*2e-6
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v (tokens only)", entry.Spend, wantSpend)
	}
}

// TestMoonshotFailedSearchIsNotBilled checks that a failed search costs
// nothing: the provider's error goes to the model, the client is not charged
// for the tool.
func TestMoonshotFailedSearchIsNotBilled(t *testing.T) {
	for _, status := range []int{
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusForbidden,
		http.StatusPaymentRequired,
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
	} {
		t.Run(fmt.Sprintf("status %d", status), func(t *testing.T) {
			fake := &moonshotFake{searchStatus: status}
			server := httptest.NewServer(fake.handler())
			defer server.Close()
			fake.baseURL = server.URL

			prx, dbStub := moonshotTestProxy(t, fake)

			body := `{
				"model": "kimi-k2",
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
			search, _, _ := moonshotToolCountsFromMetadata(t, entry.Metadata)
			if search != 0 {
				t.Errorf("the moonshot search count = %d, want 0 (a failed search is not billable)", search)
			}
			wantSpend := 300*1e-6 + 30*2e-6
			if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("Spend = %v, want %v (tokens only)", entry.Spend, wantSpend)
			}
		})
	}
}

// TestMoonshotFetchIsBilled checks the fetch tool's own price: a fetch that
// returned content is billed at fetch_cost_per_call.
func TestMoonshotFetchIsBilled(t *testing.T) {
	fake := &moonshotFake{fetchMarkdown: "# Example\n\nThe page content."}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/fetch":
			fake.mu.Lock()
			fake.fetchCalls++
			markdown := fake.fetchMarkdown
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"title": "Example", "markdown": markdown})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = fmt.Fprintf(w, `{
					"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k2",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[{"id":"call_f","type":"function","function":{"name":%q,"arguments":"{\"url\":\"https://example.com/a\"}"}}]}}],
					"usage":{"prompt_tokens":50,"completion_tokens":5,"total_tokens":55}
				}`, moonshottools.FetchToolName)
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k2",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Here is the page."}}],
				"usage":{"prompt_tokens":60,"completion_tokens":10,"total_tokens":70}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
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
	_, _, fetch := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if fetch != 1 {
		t.Errorf("the moonshot fetch count = %d, want 1", fetch)
	}
	// The fetch costs 0.013, the tokens 110*1e-6 + 15*2e-6.
	wantSpend := 110*1e-6 + 15*2e-6 + 0.013
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestMoonshotFetchWithoutContentIsNotBilled checks the fetch's own billing
// rule: only a fetch that produced page content is charged. An empty page
// and the provider's own "markdown_not_found" answer both serve nothing.
func TestMoonshotFetchWithoutContentIsNotBilled(t *testing.T) {
	cases := []struct {
		name     string
		markdown string
		fetchErr string
	}{
		{name: "empty markdown", markdown: ""},
		{name: "blank markdown", markdown: "   \n\t "},
		{name: "provider error", markdown: "", fetchErr: "markdown_not_found"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &moonshotFake{fetchMarkdown: tc.markdown, fetchError: tc.fetchErr}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.ReadAll(r.Body)
				switch r.URL.Path {
				case "/v1/tools/fetch":
					fake.mu.Lock()
					fake.fetchCalls++
					markdown := fake.fetchMarkdown
					fetchError := fake.fetchError
					fake.mu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]any{
						"title": "Example", "markdown": markdown, "error": fetchError,
					})
				case "/v1/chat/completions":
					fake.mu.Lock()
					fake.chatCalls++
					call := fake.chatCalls
					fake.mu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					if call == 1 {
						_, _ = fmt.Fprintf(w, `{
							"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k2",
							"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
								"tool_calls":[{"id":"call_f","type":"function","function":{"name":%q,"arguments":"{\"url\":\"https://example.com/a\"}"}}]}}],
							"usage":{"prompt_tokens":50,"completion_tokens":5,"total_tokens":55}
						}`, moonshottools.FetchToolName)
						return
					}
					_, _ = w.Write([]byte(`{
						"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k2",
						"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Nothing there."}}],
						"usage":{"prompt_tokens":60,"completion_tokens":10,"total_tokens":70}
					}`))
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()
			fake.baseURL = server.URL

			prx, dbStub := moonshotTestProxy(t, fake)

			body := `{
				"model": "kimi-k2",
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
			if len(dbStub.loggedEntries) != 1 {
				t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
			}
			entry := dbStub.loggedEntries[0]
			_, _, fetch := moonshotToolCountsFromMetadata(t, entry.Metadata)
			if fetch != 0 {
				t.Errorf("the moonshot fetch count = %d, want 0 (a fetch without content is not billable)", fetch)
			}
			wantSpend := 110*1e-6 + 15*2e-6
			if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("Spend = %v, want %v (tokens only)", entry.Spend, wantSpend)
			}
		})
	}
}

// TestMoonshotSearchProIsBilled checks the Pro search's own counter: it is
// counted apart from the Basic one, because the two are priced apart and the
// operation's type must be known from the call the router actually made.
func TestMoonshotSearchProIsBilled(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		switch r.URL.Path {
		case "/v1/tools/search_pro":
			fake.mu.Lock()
			fake.searchProCalls++
			results := fake.searchResults
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"search_results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				_, _ = fmt.Fprintf(w, `{
					"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k2",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[{"id":"call_p","type":"function","function":{"name":%q,"arguments":"{\"text_query\":\"kimi k2\",\"sites\":[\"arxiv.org\"],\"time_window\":\"week\"}"}}]}}],
					"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}
				}`, moonshottools.SearchProToolName)
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k2",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Found it."}}],
				"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "Find the Kimi K2 paper."}],
		"tools": [{"type": "web_search_pro"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchProCalls != 1 {
		t.Errorf("the router made %d search_pro calls, want 1", fake.searchProCalls)
	}
	if fake.searchCalls != 0 {
		t.Errorf("the router made %d basic search calls, want 0", fake.searchCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	search, searchPro, fetch := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if searchPro != 1 {
		t.Errorf("the moonshot search_pro count = %d, want 1", searchPro)
	}
	if search != 0 || fetch != 0 {
		t.Errorf("the other tool counts = %d/%d, want 0/0", search, fetch)
	}
	wantSpend := 300*1e-6 + 30*2e-6 + 0.013
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestMoonshotResponsesEndpointRunsTheLoop checks the same loop on the
// /v1/responses endpoint.
func TestMoonshotResponsesEndpointRunsTheLoop(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"input": [{"role": "user", "content": [{"type": "input_text", "text": "What is new in Kimi K2?"}]}],
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
	search, _, _ := moonshotToolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
}

// TestMoonshotToolLoopWithV1BaseURL checks the credential whose base URL
// already ends with /v1: the tool endpoints live under /v1, so the path must
// not be doubled.
func TestMoonshotToolLoopWithV1BaseURL(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL + "/v1"

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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
	search, _, _ := moonshotToolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
}

// TestMoonshotCredentialWithoutTheOptInIsNotServed checks the opt-in: a
// credential that did not ask for the router-run tools must not have them
// run, even when the request asks for the built-in search.
func TestMoonshotCredentialWithoutTheOptInIsNotServed(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	credential := config.CredentialConfig{
		Name:    "moonshot",
		Type:    config.ProviderTypeOpenAI,
		BaseURL: fake.serverURL(),
		APIKey:  "ms-key",
		RPM:     100,
		TPM:     100000,
		// No MoonshotTools: the adapter must not engage.
	}
	prx := NewTestProxyBuilder().WithCredentials(credential).WithMasterKey("master-key").Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	noPassthrough := false
	prx.modelManager = models.New(prx.logger, 50, []config.ModelRPMConfig{{
		Name: "kimi-k2", Model: "kimi-k2", Credential: "moonshot",
		PassthroughResponses: &noPassthrough,
	}})
	prx.modelManager.LoadModelsFromConfig([]config.CredentialConfig{{
		Name: "moonshot", Type: config.ProviderTypeOpenAI, APIKey: "ms-key",
	}})
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"kimi-k2": {
		InputCostPerToken: 1e-6, OutputCostPerToken: 2e-6,
	}})

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if fake.searchCalls != 0 {
		t.Errorf("the router made %d search calls, want 0 (the credential did not opt in)", fake.searchCalls)
	}
	if fake.chatCalls != 1 {
		t.Errorf("the model was called %d times, want 1", fake.chatCalls)
	}
}

// TestMoonshotClientFunctionToolsAreNotRun checks the rule that the router
// does not run a client's own tools: only the built-in search tools the
// adapter replaced are executed.
func TestMoonshotClientFunctionToolsAreNotRun(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
		"tools": [
			{"type": "web_search"},
			{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
		]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// The client's own function tool survives the rewrite and is served by the
	// upstream: the router does not run it.
	if fake.lastChatBody == nil {
		t.Fatal("no chat body was captured")
	}
	rawTools, ok := fake.lastChatBody["tools"]
	if !ok {
		t.Fatal("the second request has no tools")
	}
	if !strings.Contains(string(rawTools), `"lookup"`) {
		t.Errorf("the client's function tool was dropped: %s", rawTools)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	search, _, _ := moonshotToolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
}

// TestMoonshotToolLoopIsNotChargedTwiceForOneCall checks that a single
// successful call is charged once: the loop's own bookkeeping must not
// double-count it when the same tool call id is seen again.
func TestMoonshotToolLoopIsNotChargedTwiceForOneCall(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	// The model asks for the search twice in the same step, with the same
	// call id: the second one is the same call seen again, so it must not be
	// charged twice.
	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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
	search, _, _ := moonshotToolCountsFromMetadata(t, dbStub.loggedEntries[0].Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1 (one call, one charge)", search)
	}
	wantSpend := 300*1e-6 + 30*2e-6 + 0.013
	if diff := dbStub.loggedEntries[0].Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", dbStub.loggedEntries[0].Spend, wantSpend)
	}
}

// TestMoonshotStreamingToolLoopRunsSearch checks the streaming variant: the
// model asks for a search mid-stream, the router runs it, and the client gets
// the final answer's stream -- not the tool call.
func TestMoonshotStreamingToolLoopRunsSearch(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
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
			_ = json.NewEncoder(w).Encode(map[string]any{"search_results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				// The first stream asks for the search tool.
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"` + moonshottools.SearchToolName + `","arguments":"{\"text_query\":\"kimi k2\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"Kimi K2 was released."}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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
	if !strings.Contains(out, "Kimi K2 was released.") {
		t.Errorf("the client did not get the answer: %s", out)
	}
	// The client must not see the tool call: it declared a web_search, not
	// the router's function.
	if strings.Contains(out, moonshottools.SearchToolName) {
		t.Errorf("the client saw the router's function tool: %s", out)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	search, _, _ := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
	// Both steps' tokens.
	if entry.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300 (both steps summed)", entry.PromptTokens)
	}
}

// TestMoonshotForceStreamToolLoopRunsSearch covers the path a non-streaming
// client actually takes when the router forces stream:true upstream: the
// provider's SSE is reassembled into the Chat JSON, and a tool call in it
// must still be run by the router.
func TestMoonshotForceStreamToolLoopRunsSearch(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
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
			_ = json.NewEncoder(w).Encode(map[string]any{"search_results": results})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			if call == 1 {
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_s1","type":"function","function":{"name":"` + moonshottools.SearchToolName + `","arguments":"{\"text_query\":\"kimi k2\"}"}}]}}]}` + "\n\n"))
				_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}` + "\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
				return
			}
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{"content":"Kimi K2 was released."}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}}` + "\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)
	prx.forceStreamForTTFT = true

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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
	if !strings.Contains(out, "Kimi K2 was released.") {
		t.Errorf("the client did not get the answer: %s", out)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	search, _, _ := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if search != 1 {
		t.Errorf("the moonshot search count = %d, want 1", search)
	}
}

// TestMoonshotSeveralToolCallsInOneStep checks that the router runs every
// tool the model asked for in one step, and bills each of them.
func TestMoonshotSeveralToolCallsInOneStep(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
		fetchMarkdown: "# Example\n\nThe page content.",
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
			_ = json.NewEncoder(w).Encode(map[string]any{"search_results": results})
		case "/v1/tools/fetch":
			fake.mu.Lock()
			fake.fetchCalls++
			markdown := fake.fetchMarkdown
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"title": "Example", "markdown": markdown})
		case "/v1/chat/completions":
			fake.mu.Lock()
			fake.chatCalls++
			call := fake.chatCalls
			fake.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			if call == 1 {
				// The first answer asks for the search and the fetch at once.
				_, _ = fmt.Fprintf(w, `{
					"id":"chatcmpl-1","object":"chat.completion","model":"kimi-k2",
					"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
						"tool_calls":[
							{"id":"call_s","type":"function","function":{"name":%q,"arguments":"{\"text_query\":\"kimi k2\"}"}},
							{"id":"call_f","type":"function","function":{"name":%q,"arguments":"{\"url\":\"https://example.com/a\"}"}}
						]}}],
					"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}
				}`, moonshottools.SearchToolName, moonshottools.FetchToolName)
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","object":"chat.completion","model":"kimi-k2",
				"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"Here is everything."}}],
				"usage":{"prompt_tokens":200,"completion_tokens":20,"total_tokens":220}
			}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2, and what is on this page?"}],
		"tools": [{"type": "web_search"}, {"type": "web_fetch"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if fake.searchCalls != 1 || fake.fetchCalls != 1 {
		t.Errorf("the router made %d search and %d fetch calls, want 1 and 1",
			fake.searchCalls, fake.fetchCalls)
	}
	if len(dbStub.loggedEntries) != 1 {
		t.Fatalf("got %d spend rows, want 1", len(dbStub.loggedEntries))
	}
	entry := dbStub.loggedEntries[0]
	search, _, fetch := moonshotToolCountsFromMetadata(t, entry.Metadata)
	if search != 1 || fetch != 1 {
		t.Errorf("the tool counts = %d/%d, want 1/1", search, fetch)
	}
	// Two tools at $0.013 each, plus the tokens of both steps.
	wantSpend := 300*1e-6 + 30*2e-6 + 2*0.013
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}

// TestMoonshotBasicProAndFetchCostTheSame checks the tariff rule: all three
// tools cost the same final client price ($0.013), so one Basic search, one
// Pro search and one fetch cost $0.039 between them, plus the tokens.
func TestMoonshotBasicProAndFetchCostTheSame(t *testing.T) {
	prices := models.ToolPricesProvider{
		SearchCostPerCall:    0.013,
		SearchProCostPerCall: 0.013,
		FetchCostPerCall:     0.013,
		Rate:                 1.3,
	}
	if prices.SearchCostPerCall != prices.SearchProCostPerCall ||
		prices.SearchCostPerCall != prices.FetchCostPerCall {
		t.Fatalf("the three tool prices differ: %+v", prices)
	}
	const wantTotal = 0.039
	got := prices.SearchCostPerCall + prices.SearchProCostPerCall + prices.FetchCostPerCall
	if diff := got - wantTotal; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("the three tools cost %v together, want %v", got, wantTotal)
	}
}

// TestMoonshotCallsAreNotCountedAsTheBuiltInTools checks that the router's
// own calls are counted only as Moonshot's: the built-in tools' counters
// belong to the model, and counting the router's calls in both would charge
// the client twice -- once by Moonshot's tariff and once by the model's.
func TestMoonshotCallsAreNotCountedAsTheBuiltInTools(t *testing.T) {
	fake := &moonshotFake{
		searchResults: []map[string]any{{
			"title": "Kimi K2 released",
			"url":   "https://example.com/k2",
		}},
	}
	server := httptest.NewServer(fake.handler())
	defer server.Close()
	fake.baseURL = server.URL

	prx, dbStub := moonshotTestProxy(t, fake)

	body := `{
		"model": "kimi-k2",
		"messages": [{"role": "user", "content": "What is new in Kimi K2?"}],
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

	// The built-in tools' counters belong to the model: the provider never
	// reported a built-in search here, so they must stay at zero.
	var envelope struct {
		UsageObject struct {
			ServerToolUse struct {
				WebSearchRequests      int `json:"web_search_requests"`
				WebFetchRequests       int `json:"web_fetch_requests"`
				MoonshotSearchRequests int `json:"moonshot_search_requests"`
				MoonshotFetchRequests  int `json:"moonshot_fetch_requests"`
			} `json:"server_tool_use"`
		} `json:"usage_object"`
	}
	if err := json.Unmarshal([]byte(entry.Metadata), &envelope); err != nil {
		t.Fatalf("decoding the spend metadata: %v", err)
	}
	use := envelope.UsageObject.ServerToolUse
	if use.WebSearchRequests != 0 {
		t.Errorf("web_search_requests = %d, want 0 (the router's search is Moonshot's, not the model's)",
			use.WebSearchRequests)
	}
	if use.WebFetchRequests != 0 {
		t.Errorf("web_fetch_requests = %d, want 0", use.WebFetchRequests)
	}
	if use.MoonshotSearchRequests != 1 {
		t.Errorf("moonshot_search_requests = %d, want 1", use.MoonshotSearchRequests)
	}

	// The price is the tool's own: one search at $0.013, and nothing for the
	// built-in web search.
	wantSpend := 300*1e-6 + 30*2e-6 + 0.013
	if diff := entry.Spend - wantSpend; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Spend = %v, want %v", entry.Spend, wantSpend)
	}
}
