package moonshottools

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestTheRequestCarriesTheModelsOwnArguments checks that the arguments the
// model chose reach Moonshot as they are: the query, the limit, the timeout
// and the content flag.
func TestTheRequestCarriesTheModelsOwnArguments(t *testing.T) {
	var sent map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &sent)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"search_results":[]}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "k", Timeout: 5 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	args, err := ParseSearchArguments(`{"text_query":"kimi k3","limit":7,"timeout_seconds":45,"include_content":true}`)
	if err != nil {
		t.Fatalf("ParseSearchArguments: %v", err)
	}
	if _, _, _, err := client.Search(ctx, args.ToSearchRequest()); err != nil {
		t.Fatalf("Search: %v", err)
	}

	want := map[string]any{
		"text_query":      "kimi k3",
		"limit":           float64(7),
		"timeout_seconds": float64(45),
		"include_content": true,
	}
	for name, value := range want {
		if sent[name] != value {
			t.Errorf("the request's %q = %v, want %v", name, sent[name], value)
		}
	}
}

// TestTheProRequestCarriesItsOwnFields checks the Pro search's own fields:
// the sites to search in and the time window.
func TestTheProRequestCarriesItsOwnFields(t *testing.T) {
	var sent map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &sent)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"search_results":[]}`))
	}))
	defer server.Close()

	client := NewClient(ClientConfig{BaseURL: server.URL, APIKey: "k", Timeout: 5 * time.Second})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	args, err := ParseSearchProArguments(`{"text_query":"kimi k3","sites":["arxiv.org","github.com"],"time_window":"week","limit":3}`)
	if err != nil {
		t.Fatalf("ParseSearchProArguments: %v", err)
	}
	if _, _, _, err := client.SearchPro(ctx, args.ToSearchProRequest()); err != nil {
		t.Fatalf("SearchPro: %v", err)
	}

	if sent["time_window"] != "week" {
		t.Errorf("the request's time_window = %v, want week", sent["time_window"])
	}
	sites, ok := sent["sites"].([]any)
	if !ok || len(sites) != 2 {
		t.Fatalf("the request's sites = %v, want the two sites", sent["sites"])
	}
}
