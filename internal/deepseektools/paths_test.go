package deepseektools

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
// endpoint DeepSeek documents, with the documented method, key and fields.
func TestSearchCallsTheDocumentedEndpoint(t *testing.T) {
	for _, base := range []string{
		"https://api.deepseek.com",
		"https://api.deepseek.com/v1",
	} {
		t.Run(base, func(t *testing.T) {
			captured := &capturedCall{}
			server := newCapturingUpstream(t, captured, `{"results":[{"title":"T","url":"https://e.com"}]}`)
			defer server.Close()

			client := NewClient(ClientConfig{
				BaseURL: server.URL + strings.TrimPrefix(base, "https://api.deepseek.com"),
				APIKey:  "ds-secret-key",
				Timeout: 5 * time.Second,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			_, result, _, err := client.Search(ctx, SearchRequest{
				Query:        "deepseek v4",
				Count:        3,
				OriQuestion:  "what is new",
				Localization: "203.0.113.7",
				NeedContent:  true,
			})
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if result == nil || len(result.Results) != 1 {
				t.Fatalf("unexpected result: %+v", result)
			}

			if captured.method != http.MethodPost {
				t.Errorf("method = %s, want POST", captured.method)
			}
			// The endpoint the vendor documents, whatever the base URL's shape.
			if captured.path != "/v1/tools/search" {
				t.Errorf("path = %q, want %q", captured.path, "/v1/tools/search")
			}
			if captured.auth != "Bearer ds-secret-key" {
				t.Errorf("Authorization = %q, want the credential's bearer key", captured.auth)
			}

			// The fields of the documented request schema, and only those.
			var sent map[string]any
			if err := json.Unmarshal(captured.body, &sent); err != nil {
				t.Fatalf("decoding the request body: %v", err)
			}
			want := map[string]any{
				"query":        "deepseek v4",
				"count":        float64(3),
				"ori_question": "what is new",
				"localization": "203.0.113.7",
				"need_content": true,
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

// TestFetchCallsTheDocumentedEndpoint checks the fetch endpoint the same way.
func TestFetchCallsTheDocumentedEndpoint(t *testing.T) {
	for _, base := range []string{
		"https://api.deepseek.com",
		"https://api.deepseek.com/v1",
	} {
		t.Run(base, func(t *testing.T) {
			captured := &capturedCall{}
			server := newCapturingUpstream(t, captured, `{"title":"Example","content":"body"}`)
			defer server.Close()

			client := NewClient(ClientConfig{
				BaseURL: server.URL + strings.TrimPrefix(base, "https://api.deepseek.com"),
				APIKey:  "ds-secret-key",
				Timeout: 5 * time.Second,
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			_, result, _, err := client.Fetch(ctx, FetchRequest{URL: "https://example.com"})
			if err != nil {
				t.Fatalf("Fetch: %v", err)
			}
			if result == nil || result.Content != "body" {
				t.Fatalf("unexpected result: %+v", result)
			}
			if captured.method != http.MethodPost {
				t.Errorf("method = %s, want POST", captured.method)
			}
			if captured.path != "/v1/tools/fetch" {
				t.Errorf("path = %q, want %q", captured.path, "/v1/tools/fetch")
			}
			if captured.auth != "Bearer ds-secret-key" {
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
	providerError := `{"error":{"message":"'count' must be between 1 and 50","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}`
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

	status, _, raw, err := client.Search(ctx, SearchRequest{Query: "q", Count: 100})
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
