package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// liveBillingMu serializes the database checks below: they share the table
// and look for their own row by the tool counters they expect.
var liveBillingMu sync.Mutex

// The check below confirms the charge in the database: the router's response
// carries the cost only for the organizations that opted into it
// (response_cost_mode), so for the others the spend row is the only place
// where the search's price can be seen.
//
// It talks to ClickHouse over its HTTP interface, so it needs no client
// dependency:
//
//	AIR_CLICKHOUSE_URL=http://clickhouse:8123 \\
//	AIR_CLICKHOUSE_USER=air \\
//	AIR_CLICKHOUSE_PASSWORD=... \\
//	go test ./internal/proxy/ -run TestDeepSeekLiveRouterBilledInDatabase -v
//
// It is skipped without the variables, so a CI run without them skips it
// instead of failing.

// clickHouseQuery runs one query against ClickHouse over HTTP and returns the
// rows, each a map of column name to value.
//
// The values are the raw JSON values, so a number stays a number: the
// database returns them unquoted.
func clickHouseQuery(t *testing.T, query string) []map[string]any {
	t.Helper()
	base := os.Getenv("AIR_CLICKHOUSE_URL")
	if base == "" {
		t.Skip("AIR_CLICKHOUSE_URL is not set; skipping the database check")
	}
	user := os.Getenv("AIR_CLICKHOUSE_USER")
	password := os.Getenv("AIR_CLICKHOUSE_PASSWORD")
	if user == "" || password == "" {
		t.Skip("AIR_CLICKHOUSE_USER / AIR_CLICKHOUSE_PASSWORD are not set")
	}
	full := query + " FORMAT JSONEachRow"
	req, err := http.NewRequest(http.MethodPost, strings.TrimSuffix(base, "/")+"/", strings.NewReader(full))
	require.NoError(t, err)
	req.SetBasicAuth(user, password)
	req.Header.Set("Content-Type", "text/plain")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, 200, resp.StatusCode, string(body))

	var rows []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
		if line == "" {
			continue
		}
		var row map[string]any
		require.NoError(t, json.Unmarshal([]byte(line), &row), "decoding the row: %s", line)
		rows = append(rows, row)
	}
	return rows
}

// TestDeepSeekLiveRouterBilledInDatabase checks that a request with the
// built-in web_search tool is charged for the search: the spend row counts
// the search and prices it.
func TestDeepSeekLiveRouterBilledInDatabase(t *testing.T) {
	// The tests below look for their own spend row by the tool counters they
	// expect, and the runs share the database: a concurrent run's row would
	// match too. They run one after another.
	liveBillingMu.Lock()
	defer liveBillingMu.Unlock()

	url, key := liveRouterURL(t)

	answer, _, _, responseID := liveRouterRequest(t, url, key, map[string]any{
		"model": "deepseek/deepseek-v4.1-flash-1017",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "Search the web for the DeepSeek V4 release date and state the date.",
		}},
		"tools":                 []map[string]any{{"type": "web_search"}},
		"max_completion_tokens": 900,
	})
	require.Contains(t, answer, "2026", "the answer carries no date from the search results")

	// The row lands a moment after the response.
	var row map[string]any
	for attempt := 0; attempt < 12; attempt++ {
		time.Sleep(5 * time.Second)
		rows := clickHouseQuery(t, fmt.Sprintf(
			"SELECT request_id, web_search_requests, web_search_cost, web_fetch_requests, web_fetch_cost, total_cost "+
				"FROM %s.logs WHERE request_id = '%s'",
			clickHouseDB(), responseID))
		if len(rows) > 0 {
			row = rows[0]
			break
		}
	}
	require.NotNil(t, row, "no spend row with a search turned up")

	t.Logf("the spend row: %v", row)
	require.EqualValues(t, 1, row["web_search_requests"], "the search was not counted")
	// The price of one search on this model, with the 1.3 markup included.
	require.InDelta(t, 0.000975, numberForTest(t, row["web_search_cost"]), 1e-9,
		"the search was priced wrong")
	require.Greater(t, numberForTest(t, row["total_cost"]), 0.000975,
		"the request cost less than the search alone")
}

// clickHouseDB is the database the checks read, "air" unless the
// environment says otherwise.
func clickHouseDB() string {
	if db := os.Getenv("AIR_CLICKHOUSE_DB"); db != "" {
		return db
	}
	return "air"
}

// numberForTest reads a number the database returned, as a JSON number or as
// a string.
func numberForTest(t *testing.T, value any) float64 {
	t.Helper()
	switch typed := value.(type) {
	case float64:
		return typed
	case string:
		var number float64
		_, err := fmt.Sscanf(typed, "%g", &number)
		require.NoError(t, err, "parsing %q", typed)
		return number
	default:
		t.Fatalf("the value is not a number: %v (%T)", value, value)
		return 0
	}
}

// TestDeepSeekLiveRouterFetchBilledInDatabase checks the fetch tool's charge
// the same way: a fetch that returned page content is billed at the tool's
// own rate, in its own column.
func TestDeepSeekLiveRouterFetchBilledInDatabase(t *testing.T) {
	liveBillingMu.Lock()
	defer liveBillingMu.Unlock()

	url, key := liveRouterURL(t)

	answer, _, _, responseID := liveRouterRequest(t, url, key, map[string]any{
		"model": "deepseek/deepseek-v4.1-flash-1017",
		"messages": []map[string]any{{
			"role":    "user",
			"content": "Fetch https://example.com and quote the page title.",
		}},
		"tools":                 []map[string]any{{"type": "web_fetch"}},
		"max_completion_tokens": 900,
	})
	require.Contains(t, answer, "Example Domain", "the answer does not quote the fetched page")

	var row map[string]any
	for attempt := 0; attempt < 12; attempt++ {
		time.Sleep(5 * time.Second)
		rows := clickHouseQuery(t, fmt.Sprintf(
			"SELECT request_id, web_search_requests, web_fetch_requests, web_fetch_cost, total_cost "+
				"FROM %s.logs WHERE request_id = '%s'",
			clickHouseDB(), responseID))
		if len(rows) > 0 {
			row = rows[0]
			break
		}
	}
	require.NotNil(t, row, "no spend row with a fetch turned up")

	t.Logf("the spend row: %v", row)
	require.EqualValues(t, 1, row["web_fetch_requests"], "the fetch was not counted")
	require.InDelta(t, 0.000975, numberForTest(t, row["web_fetch_cost"]), 1e-9,
		"the fetch was priced wrong")
}
