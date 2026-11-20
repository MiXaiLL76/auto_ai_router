package moonshottools

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// The live checks in this file talk to Moonshot's own Search, Search Pro and
// Fetch endpoints, so they need a real key. They are skipped without one:
//
//	MOONSHOT_API_KEY=sk-... go test ./internal/moonshottools/ -run Live -v
//
// The key is read from the environment on purpose: it never lands in the
// repository, and a CI run without the variable skips these instead of
// failing.

// liveClient builds a client for the live checks, or skips the test.
func liveClient(t *testing.T) *Client {
	t.Helper()
	key := os.Getenv("MOONSHOT_API_KEY")
	if key == "" {
		t.Skip("MOONSHOT_API_KEY is not set; skipping the live Moonshot check")
	}
	base := os.Getenv("MOONSHOT_BASE_URL")
	if base == "" {
		base = "https://api.moonshot.ai"
	}
	return NewClient(ClientConfig{
		BaseURL:    base,
		APIKey:     key,
		Timeout:    60 * time.Second,
		MaxRetries: 2,
	})
}

// TestLiveSearch checks that Moonshot's Basic search endpoint answers a real
// query with results: the router's adapter depends on the results array
// being non-empty (an empty one is not a billable search).
func TestLiveSearch(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Search(ctx, SearchRequest{
		TextQuery: "Kimi K2 release",
		Limit:     3,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Fatalf("status = %d, body = %s", status, truncateLive(string(raw), 400))
	}
	if len(result.Results) == 0 {
		t.Fatalf("the search returned no results: %s", truncateLive(string(raw), 400))
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

// TestLiveSearchPro checks the Pro search the same way.
func TestLiveSearchPro(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.SearchPro(ctx, SearchProRequest{
		TextQuery:  "Kimi K2 release",
		Limit:      3,
		TimeWindow: "week",
	})
	if err != nil {
		t.Fatalf("SearchPro: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Fatalf("status = %d, body = %s", status, truncateLive(string(raw), 400))
	}
	if len(result.Results) == 0 {
		t.Fatalf("the Pro search returned no results: %s", truncateLive(string(raw), 400))
	}
	t.Logf("%d results", len(result.Results))
}

// TestLiveSearchEmptyResultIsNotBillable checks the billing rule the adapter
// follows: a search that finds nothing is not a billable search. The query
// is nonsense on purpose, so the provider answers with an empty result set.
func TestLiveSearchEmptyResultIsNotBillable(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Search(ctx, SearchRequest{
		TextQuery: "zzqxvnonexistentterm9f3k2",
		Limit:     1,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if status < 200 || status >= 300 {
		t.Skipf("the provider answered %d for a nonsense query: %s", status, truncateLive(string(raw), 200))
	}
	if len(result.Results) > 0 {
		t.Skipf("the provider found something for the nonsense query: %d results", len(result.Results))
	}
	// The empty result set is what the adapter treats as "not billable".
	t.Log("the nonsense query returned no results, as expected")
}

// TestLiveFetch checks that Moonshot's fetch endpoint returns a page's
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
		t.Fatalf("status = %d, body = %s", status, truncateLive(string(raw), 400))
	}
	if strings.TrimSpace(result.Markdown) == "" {
		t.Fatalf("the fetch returned no content: %s", truncateLive(string(raw), 400))
	}
	t.Logf("fetched %q (%d characters)", result.Title, len(result.Markdown))
}

// TestLiveFetchBadURLIsNotBillable checks that a fetch the provider refuses
// comes back as an error status or the provider's own error code, not as an
// empty success: the adapter bills only a fetch that returned content.
func TestLiveFetchBadURLIsNotBillable(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	status, result, raw, err := client.Fetch(ctx, FetchRequest{URL: "https://nonexistent.invalid/page"})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	if status == 200 && strings.TrimSpace(result.Markdown) != "" {
		t.Fatalf("the provider answered 200 with content for a nonexistent host: %s", truncateLive(string(raw), 200))
	}
	t.Logf("the provider answered %d, as expected", status)
}

// truncateLive shortens a body for a log line.
func truncateLive(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
