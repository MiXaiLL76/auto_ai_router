package moonshottools

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// capturedCall is what the fake upstream saw.
type capturedCall struct {
	method string
	path   string
	auth   string
	body   []byte
}

// newCapturingUpstream serves one endpoint and records the call it got.
func newCapturingUpstream(t *testing.T, captured *capturedCall, response string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured.method = r.Method
		captured.path = r.URL.Path
		captured.auth = r.Header.Get("Authorization")
		captured.body = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(response))
	}))
}

// TestSearchCallsTheDocumentedEndpoint checks that a search goes to the
// endpoint Moonshot documents, with the documented method, key and fields.
func TestSearchCallsTheDocumentedEndpoint(t *testing.T) {
	for _, base := range []string{
		"https://api.moonshot.ai",
		"https://api.moonshot.ai/v1",
	} {
		t.Run(base, func(t *testing.T) {
			captured := &capturedCall{}
			server := newCapturingUpstream(t, captured,
				`{"search_results":[{"title":"T","url":"https://e.com","snippet":"s"}]}`)
			defer server.Close()

			client := NewClient(ClientConfig{
				BaseURL: server.URL + strings.TrimPrefix(base, "https://api.moonshot.ai"),
				APIKey:  "ms-secret-key",
				Timeout: 5 * time.Second,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			status, result, _, err := client.Search(ctx, SearchRequest{
				TextQuery:      "kimi k2",
				Limit:          3,
				TimeoutSeconds: 20,
				IncludeContent: true,
			})
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if status != http.StatusOK {
				t.Errorf("status = %d, want 200", status)
			}
			if len(result.Results) != 1 || result.Results[0].URL != "https://e.com" {
				t.Fatalf("unexpected result: %+v", result)
			}

			if captured.method != http.MethodPost {
				t.Errorf("method = %s, want POST", captured.method)
			}
			// The endpoint the vendor documents, whatever the base URL's shape.
			if captured.path != "/v1/tools/search" {
				t.Errorf("path = %q, want %q", captured.path, "/v1/tools/search")
			}
			if captured.auth != "Bearer ms-secret-key" {
				t.Errorf("Authorization = %q, want the credential's bearer key", captured.auth)
			}

			// The fields of the documented request schema, and only those.
			var sent map[string]any
			if err := json.Unmarshal(captured.body, &sent); err != nil {
				t.Fatalf("decoding the request body: %v", err)
			}
			want := map[string]any{
				"text_query":      "kimi k2",
				"limit":           float64(3),
				"timeout_seconds": float64(20),
				"include_content": true,
			}
			for name, value := range want {
				if sent[name] != value {
					t.Errorf("the request's %q = %v, want %v", name, sent[name], value)
				}
			}
			if len(sent) != len(want) {
				t.Errorf("the request carries %d fields, want %d: %s", len(sent), len(want), captured.body)
			}
		})
	}
}

// TestSearchProCallsTheDocumentedEndpoint checks the Pro search the same way,
// including the fields only it has: the sites and the time window.
func TestSearchProCallsTheDocumentedEndpoint(t *testing.T) {
	for _, base := range []string{
		"https://api.moonshot.ai",
		"https://api.moonshot.ai/v1",
	} {
		t.Run(base, func(t *testing.T) {
			captured := &capturedCall{}
			server := newCapturingUpstream(t, captured,
				`{"search_results":[{"title":"T","url":"https://e.com"}]}`)
			defer server.Close()

			client := NewClient(ClientConfig{
				BaseURL: server.URL + strings.TrimPrefix(base, "https://api.moonshot.ai"),
				APIKey:  "ms-secret-key",
				Timeout: 5 * time.Second,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			status, result, _, err := client.SearchPro(ctx, SearchProRequest{
				TextQuery:      "kimi k2",
				Limit:          5,
				TimeoutSeconds: 15,
				IncludeContent: true,
				Sites:          []string{"arxiv.org", "github.com"},
				TimeWindow:     "week",
			})
			if err != nil {
				t.Fatalf("SearchPro: %v", err)
			}
			if status != http.StatusOK {
				t.Errorf("status = %d, want 200", status)
			}
			if len(result.Results) != 1 {
				t.Fatalf("unexpected result: %+v", result)
			}
			if captured.method != http.MethodPost {
				t.Errorf("method = %s, want POST", captured.method)
			}
			if captured.path != "/v1/tools/search_pro" {
				t.Errorf("path = %q, want %q", captured.path, "/v1/tools/search_pro")
			}
			if captured.auth != "Bearer ms-secret-key" {
				t.Errorf("Authorization = %q, want the credential's bearer key", captured.auth)
			}

			var sent map[string]any
			if err := json.Unmarshal(captured.body, &sent); err != nil {
				t.Fatalf("decoding the request body: %v", err)
			}
			want := map[string]any{
				"text_query":      "kimi k2",
				"limit":           float64(5),
				"timeout_seconds": float64(15),
				"include_content": true,
				"time_window":     "week",
			}
			for name, value := range want {
				if sent[name] != value {
					t.Errorf("the request's %q = %v, want %v", name, sent[name], value)
				}
			}
			sites, ok := sent["sites"].([]any)
			if !ok || len(sites) != 2 {
				t.Fatalf("the request's sites = %v, want the two sites", sent["sites"])
			}
			if len(sent) != len(want)+1 {
				t.Errorf("the request carries %d fields, want %d: %s", len(sent), len(want)+1, captured.body)
			}
		})
	}
}

// TestFetchCallsTheDocumentedEndpoint checks the fetch endpoint the same way.
func TestFetchCallsTheDocumentedEndpoint(t *testing.T) {
	for _, base := range []string{
		"https://api.moonshot.ai",
		"https://api.moonshot.ai/v1",
	} {
		t.Run(base, func(t *testing.T) {
			captured := &capturedCall{}
			server := newCapturingUpstream(t, captured, `{"title":"Example","markdown":"# body"}`)
			defer server.Close()

			client := NewClient(ClientConfig{
				BaseURL: server.URL + strings.TrimPrefix(base, "https://api.moonshot.ai"),
				APIKey:  "ms-secret-key",
				Timeout: 5 * time.Second,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			status, result, _, err := client.Fetch(ctx, FetchRequest{URL: "https://example.com"})
			if err != nil {
				t.Fatalf("Fetch: %v", err)
			}
			if status != http.StatusOK {
				t.Errorf("status = %d, want 200", status)
			}
			if result.Markdown != "# body" {
				t.Fatalf("unexpected result: %+v", result)
			}
			if captured.method != http.MethodPost {
				t.Errorf("method = %s, want POST", captured.method)
			}
			if captured.path != "/v1/tools/fetch" {
				t.Errorf("path = %q, want %q", captured.path, "/v1/tools/fetch")
			}
			if captured.auth != "Bearer ms-secret-key" {
				t.Errorf("Authorization = %q, want the credential's bearer key", captured.auth)
			}
			var sent map[string]any
			if err := json.Unmarshal(captured.body, &sent); err != nil {
				t.Fatalf("decoding the request body: %v", err)
			}
			if len(sent) != 1 || sent["url"] != "https://example.com" {
				t.Errorf("the request carries %v, want only the url", sent)
			}
		})
	}
}

// TestProviderErrorFormatIsPreserved checks the other half of the vendor's
// contract: the provider's own error format reaches the caller unchanged, so
// the model is told what the provider said.
func TestProviderErrorFormatIsPreserved(t *testing.T) {
	providerError := `{"error":{"message":"text_query is required","type":"invalid_request_error"}}`
	captured := &capturedCall{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(providerError))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "k", Timeout: 5 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	status, _, raw, err := client.Search(ctx, SearchRequest{TextQuery: "q"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if status != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", status)
	}
	if string(raw) != providerError {
		t.Errorf("the provider's error was rewritten:\n got %s\nwant %s", raw, providerError)
	}
	if captured.path != "/v1/tools/search" {
		t.Errorf("path = %q, want %q", captured.path, "/v1/tools/search")
	}
}

// TestCallerAuthorizationNeverReachesMoonshot checks the isolation rule: the
// header is built from scratch, so the client's own credentials cannot leak
// to the provider.
func TestCallerAuthorizationNeverReachesMoonshot(t *testing.T) {
	captured := &capturedCall{}
	server := newCapturingUpstream(t, captured, `{"search_results":[]}`)
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "ms-key", Timeout: 5 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if _, _, _, err := client.Search(ctx, SearchRequest{TextQuery: "q"}); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if strings.Contains(string(captured.body), "client-key") {
		t.Errorf("the request body carries the caller's key: %s", captured.body)
	}
	if captured.auth != "Bearer ms-key" {
		t.Errorf("Authorization = %q, want only the credential's key", captured.auth)
	}
}

// TestFetchErrorCodeIsRead checks that the provider's own error code (the
// "markdown_not_found" answer) reaches the caller: it decides whether the
// fetch was billable.
func TestFetchErrorCodeIsRead(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"title":"","markdown":"","error":"markdown_not_found"}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "k", Timeout: 5 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	status, result, _, err := client.Fetch(ctx, FetchRequest{URL: "https://example.com"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if status != http.StatusOK {
		t.Errorf("status = %d, want 200 (the provider answered)", status)
	}
	if result.Error != "markdown_not_found" {
		t.Errorf("the provider's error code = %q, want markdown_not_found", result.Error)
	}
	if result.Markdown != "" {
		t.Errorf("the markdown = %q, want it empty", result.Markdown)
	}
}

// TestTimeoutIsReportedAsAnError checks that a provider that never answers
// fails the call instead of hanging: the caller must be able to hand the
// model an error and go on.
func TestTimeoutIsReportedAsAnError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "k", Timeout: 100 * time.Millisecond})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	started := time.Now()
	status, _, _, err := client.Search(ctx, SearchRequest{TextQuery: "q"})
	if err == nil {
		t.Fatalf("a timed-out search must be an error, got status %d", status)
	}
	if time.Since(started) > time.Second {
		t.Errorf("the call took %s, want it to give up at the timeout", time.Since(started))
	}
}
