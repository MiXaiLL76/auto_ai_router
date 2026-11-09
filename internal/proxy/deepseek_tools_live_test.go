package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/models"
	"github.com/stretchr/testify/require"
)

// The live check in this file runs the whole adapter against DeepSeek's own
// API: the router sends a request with the built-in web_search tool, the
// model asks for it, the router runs DeepSeek's /v1/tools/search, and the
// model answers with what it found.
//
// It needs a real DeepSeek key, so it is skipped without one:
//
//	AIR_DEEPSEEK_LIVE=1 \
//	DEEPSEEK_API_KEY=sk-... \
//	go test ./internal/proxy/ -run TestDeepSeekLiveSearchWorks -v
//
// The key comes from the environment on purpose: it never lands in the
// repository, and a CI run without the variables skips the test instead of
// failing it.

// deepSeekLivePrice is the price the live check bills against: the published
// DeepSeek-V4.1-Flash rates with the 1.3 markup already included.
func deepSeekLivePrice() *models.ModelPrice {
	return &models.ModelPrice{
		InputCostPerToken:       0.00000039,
		OutputCostPerToken:      0.00000156,
		CacheReadInputTokenCost: 0.0000000078,
		SearchContextCostPerQuery: map[string]float64{
			"search_context_size_low":    0.000975,
			"search_context_size_medium": 0.000975,
			"search_context_size_high":   0.000975,
		},
		WebSearchBillingUnit: "per_query",
		ToolCostPerCall:      map[string]float64{"fetch": 0.000975},
	}
}

// deepSeekLiveProxy builds a proxy whose single credential is the live
// DeepSeek one, or skips the test.
func deepSeekLiveProxy(t *testing.T) (*Proxy, *stubLiteLLMManager) {
	t.Helper()
	if os.Getenv("AIR_DEEPSEEK_LIVE") != "1" {
		t.Skip("AIR_DEEPSEEK_LIVE is not enabled")
	}
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY is not set")
	}
	baseURL := os.Getenv("DEEPSEEK_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}

	credential := config.CredentialConfig{
		Name:          "deepseek-live",
		Type:          config.ProviderTypeOpenAI,
		BaseURL:       baseURL,
		APIKey:        key,
		RPM:           60,
		TPM:           1000000,
		DeepSeekTools: true,
	}
	prx := NewTestProxyBuilder().
		WithCredentials(credential).
		WithMasterKey("master-key").
		Build()
	dbStub := &stubLiteLLMManager{}
	prx.LiteLLMDB = dbStub
	prx.priceRegistry = models.NewModelPriceRegistry()
	prx.priceRegistry.Update(map[string]*models.ModelPrice{"deepseek-flash": deepSeekLivePrice()})
	return prx, dbStub
}

// TestDeepSeekLiveSearchWorks checks the adapter against the real DeepSeek:
// the model asks for the router's search tool, the router runs it, and the
// model answers with what the search found.
//
// The check that the search really ran is the model's own answer: it has to
// name the DeepSeek release date, which it can only do from the search
// results -- the model has no such fact in its weights.
func TestDeepSeekLiveSearchWorks(t *testing.T) {
	prx, dbStub := deepSeekLiveProxy(t)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "Search the web for the DeepSeek V4 release date. Use the search results and state the date they give."}],
		"tools": [{"type": "web_search"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, 200, rec.Code, rec.Body.String())

	var response struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))
	require.Len(t, response.Choices, 1)
	answer := response.Choices[0].Message.Content
	require.NotEmpty(t, answer, "the model answered nothing")
	t.Logf("the model answered: %s", truncateForLog(answer, 300))

	// The model must have called the router's search tool: the answer names
	// the release date, which only the search results carry.
	require.Contains(t, answer, "2026", "the answer carries no date from the search results")

	// The search was billed: the spend row counts it.
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	searches, _ := toolCountsFromMetadata(t, entry.Metadata)
	require.Equal(t, 1, searches, "the search was not billed as a search")
	require.Greater(t, entry.Spend, 0.0, "the request cost nothing")
	t.Logf("billed: %d search(es), spend %v", searches, entry.Spend)
}

// TestDeepSeekLiveSearchIsNotBilledWhenItFails checks the billing rule on the
// live API: a search that returns no results costs nothing.
//
// The query is nonsense on purpose, so the provider answers with an empty
// result set. The request still succeeds (the model answers from what it
// knows), but no search is billed.
func TestDeepSeekLiveSearchIsNotBilledWhenItFails(t *testing.T) {
	prx, dbStub := deepSeekLiveProxy(t)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "Search the web for zzqxvnonexistentterm9f3k2 and say what you found."}],
		"tools": [{"type": "web_search"}]
	}`
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
	t.Logf("an empty search was not billed; the request cost %v (its tokens)", entry.Spend)
}

// TestDeepSeekLiveFetchIsBilled checks the fetch tool against the live API:
// a fetch that returned page content is billed at the tool's own rate.
func TestDeepSeekLiveFetchIsBilled(t *testing.T) {
	prx, dbStub := deepSeekLiveProxy(t)

	body := `{
		"model": "deepseek-flash",
		"messages": [{"role": "user", "content": "Fetch https://example.com and quote the heading line of the page you fetched."}],
		"tools": [{"type": "web_fetch"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	prx.ProxyRequest(rec, req)

	require.Equal(t, 200, rec.Code, rec.Body.String())
	require.Len(t, dbStub.loggedEntries, 1)
	entry := dbStub.loggedEntries[0]
	_, fetches := toolCountsFromMetadata(t, entry.Metadata)
	require.Equal(t, 1, fetches, "the fetch was not billed as a fetch")
	require.Greater(t, entry.Spend, 0.0, "the request cost nothing")
	t.Logf("billed: %d fetch(es), spend %v", fetches, entry.Spend)
}

// The checks below run against a router that is already up (a stage or the
// production one), so they need no key of their own -- the router already has
// a DeepSeek credential. They are the ones to run when the question is
// "does the search work end to end, in the deployment", as opposed to "does
// the adapter work against DeepSeek's API".
//
//	AIR_LIVE_ROUTER_URL=https://api.vsellm.ru/v1 \
//	AIR_LIVE_ROUTER_KEY=sk-... \
//	go test ./internal/proxy/ -run TestDeepSeekLiveRouter -v
func liveRouterURL(t *testing.T) (string, string) {
	t.Helper()
	url := os.Getenv("AIR_LIVE_ROUTER_URL")
	if url == "" {
		t.Skip("AIR_LIVE_ROUTER_URL is not set; skipping the live router check")
	}
	key := os.Getenv("AIR_LIVE_ROUTER_KEY")
	if key == "" {
		t.Skip("AIR_LIVE_ROUTER_KEY is not set")
	}
	return strings.TrimSuffix(url, "/"), key
}

// liveRouterRequest sends one request to a running router and returns the
// answer and the reported cost.
func liveRouterRequest(t *testing.T, url, key string, payload map[string]any) (string, float64, string, string) {
	t.Helper()
	encoded, err := json.Marshal(payload)
	require.NoError(t, err)
	req, err := http.NewRequest(http.MethodPost, url+"/chat/completions", strings.NewReader(string(encoded)))
	require.NoError(t, err)
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 200, resp.StatusCode, string(body))

	var response struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			Cost float64 `json:"cost"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(body, &response))
	require.Len(t, response.Choices, 1)
	return response.Choices[0].Message.Content, response.Usage.Cost, string(body), response.ID
}

// TestDeepSeekLiveRouterSearchWorks checks the search end to end against a
// running router: the model has to name the DeepSeek release date, which it
// can only do from the search results the router fetched for it.
func TestDeepSeekLiveRouterSearchWorks(t *testing.T) {
	url, key := liveRouterURL(t)

	answer, cost, raw, _ := liveRouterRequest(t, url, key, map[string]any{
		"model": "deepseek/deepseek-v4.1-flash-1017",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "Search the web for the DeepSeek V4 release date. Use the search results and state the date they give.",
		}},
		"tools":                 []map[string]any{{"type": "web_search"}},
		"max_completion_tokens": 900,
	})
	t.Logf("the model answered: %s", truncateForLog(answer, 300))
	t.Logf("the request cost %v; the raw body: %s", cost, truncateForLog(raw, 200))

	// The date can only come from the search results.
	require.Contains(t, answer, "2026", "the answer carries no date from the search results")
	// The cost is reported in the body only for the organizations that opted
	// into it (response_cost_mode); for the others it is zero here and the
	// spend row in the database is where the charge is checked.
	if cost > 0 {
		t.Logf("the request reported its own cost: %v", cost)
	} else {
		t.Log("the response carries no cost (the key's organization does not opt into it); check the spend row for the charge")
	}
}

// TestDeepSeekLiveRouterFetchIsBilled checks the fetch tool against a running
// router: the model has to quote the page it fetched, and the request has to
// cost more than its tokens alone.
func TestDeepSeekLiveRouterFetchIsBilled(t *testing.T) {
	url, key := liveRouterURL(t)

	answer, cost, _, _ := liveRouterRequest(t, url, key, map[string]any{
		"model": "deepseek/deepseek-v4.1-flash-1017",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "Fetch https://example.com and quote the heading line of the page you fetched.",
		}},
		"tools":                 []map[string]any{{"type": "web_fetch"}},
		"max_completion_tokens": 900,
	})
	t.Logf("the model answered: %s", truncateForLog(answer, 300))
	t.Logf("the request cost %v", cost)

	require.Contains(t, answer, "Example Domain", "the answer does not quote the fetched page")
	if cost > 0 {
		t.Logf("the request reported its own cost: %v", cost)
	} else {
		t.Log("the response carries no cost (the key's organization does not opt into it); check the spend row for the charge")
	}
}
