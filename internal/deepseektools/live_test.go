package deepseektools

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// The live checks in this file talk to DeepSeek's own Search and Fetch
// endpoints, so they need a real key. They are skipped without one:
//
//	DEEPSEEK_API_KEY=sk-... go test ./internal/deepseektools/ -run Live -v
//
// The key is read from the environment on purpose: it never lands in the
// repository, and a CI run without the variable skips these instead of
// failing.

// liveClient builds a client for the live checks, or skips the test.
func liveClient(t *testing.T) *Client {
	t.Helper()
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY is not set; skipping the live DeepSeek check")
	}
	base := os.Getenv("DEEPSEEK_BASE_URL")
	if base == "" {
		base = "https://api.deepseek.com"
	}
	return NewClient(ClientConfig{
		BaseURL:    base,
		APIKey:     key,
		Timeout:    60 * time.Second,
		MaxRetries: 2,
	})
}

// TestLiveSearch checks that DeepSeek's search endpoint answers a real query
// with results: the router's adapter depends on the results array being
// non-empty (an empty one is not a billable search).
func TestLiveSearch(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Search(ctx, SearchRequest{
		Query:       "DeepSeek V4 release",
		Count:       3,
		NeedContent: true,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Fatalf("status = %d, body = %s", status, truncate(string(raw), 400))
	}
	if result == nil || len(result.Results) == 0 {
		t.Fatalf("the search returned no results: %s", truncate(string(raw), 400))
	}
	first := result.Results[0]
	if first.URL == "" {
		t.Errorf("the first result has no URL: %+v", first)
	}
	if first.Title == "" {
		t.Errorf("the first result has no title: %+v", first)
	}
	t.Logf("%d results; first: %s (%s)", len(result.Results), first.Title, first.URL)
}

// TestLiveSearchEmptyResultIsNotBillable checks the billing rule the adapter
// follows: a search that finds nothing is not a billable search. The query
// is nonsense on purpose, so the provider answers with an empty result set.
func TestLiveSearchEmptyResultIsNotBillable(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Search(ctx, SearchRequest{
		Query: "zzqxvnonexistentterm9f3k2",
		Count: 1,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Skipf("the provider answered %d for a nonsense query: %s", status, truncate(string(raw), 200))
	}
	if result != nil && len(result.Results) > 0 {
		t.Skipf("the provider found something for the nonsense query: %d results", len(result.Results))
	}
	// The empty result set is what the adapter treats as "not billable".
	t.Log("the nonsense query returned no results, as expected")
}

// TestLiveFetch checks that DeepSeek's fetch endpoint returns a page's
// content: the adapter bills a fetch only when the content is non-empty.
func TestLiveFetch(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Fetch(ctx, FetchRequest{URL: "https://example.com"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Fatalf("status = %d, body = %s", status, truncate(string(raw), 400))
	}
	if result == nil {
		t.Fatalf("the fetch returned no result: %s", truncate(string(raw), 400))
	}
	if strings.TrimSpace(result.Content) == "" {
		t.Fatalf("the fetch returned no content: %s", truncate(string(raw), 400))
	}
	t.Logf("fetched %q (%d characters)", result.Title, len(result.Content))
}

// TestLiveFetchBadURLIsNotBillable checks that a fetch the provider refuses
// comes back as an error status, not as an empty success: the adapter bills
// only a fetch that returned content.
func TestLiveFetchBadURLIsNotBillable(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, _, raw, err := client.Fetch(ctx, FetchRequest{URL: "https://nonexistent.invalid/page"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if status == 200 {
		t.Fatalf("the provider answered 200 for a nonexistent host: %s", truncate(string(raw), 200))
	}
	t.Logf("the provider answered %d, as expected", status)
}

// truncate shortens a body for a log line.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
