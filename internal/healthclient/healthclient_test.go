// Tests for the health-check service client: the local alive/dead mirror.
package healthclient

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testConfig(url string, enabled bool) *config.HealthServiceConfig {
	return &config.HealthServiceConfig{
		Enabled:         enabled,
		URL:             url,
		AuthToken:       "",
		SyncInterval:    50 * time.Millisecond,
		CacheTTL:        200 * time.Millisecond,
		ReportQueueSize: 1024,
		HTTPTimeout:     2 * time.Second,
	}
}

func TestRetryAfterSeconds(t *testing.T) {
	assert.Equal(t, 60, RetryAfterSeconds("60"))
	assert.Equal(t, 0, RetryAfterSeconds(""))
	assert.Equal(t, 0, RetryAfterSeconds("Wed, 21 Oct 2015 07:28:00 GMT"))
	assert.Equal(t, 0, RetryAfterSeconds("-3"))
}

func TestDisabledClientIsFailOpen(t *testing.T) {
	c := New(testConfig("http://127.0.0.1:1", false), slog.Default())
	assert.False(t, c.IsDead("whatever"), "disabled client never reports dead")
	c.Report("cred", "model", 429, 60) // must not block or crash
}

func TestUnknownAndStaleAreFailOpen(t *testing.T) {
	srv := httptest.NewServer(mockStatusMux(map[string]statusView{
		"dead-cred": statusView{Alive: false, FailCount: 3, LastStatus: 429, UpdatedAt: 1},
	}))
	defer srv.Close()

	c := New(testConfig(srv.URL, true), slog.Default())
	c.pull(context.Background())

	assert.False(t, c.IsDead("never-seen"), "unknown accounts are not dead (fail-open)")
	assert.True(t, c.IsDead("dead-cred"), "cached dead account is dead")

	// Stale entries (older than cacheTTL) are treated as alive again.
	time.Sleep(300 * time.Millisecond)
	assert.False(t, c.IsDead("dead-cred"), "stale mirror entries fail open")
}

func TestPullRefreshesCache(t *testing.T) {
	state := &struct {
		Dead bool
	}{Dead: true}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/creds", func(w http.ResponseWriter, r *http.Request) {
		st := map[string]statusView{}
		if state.Dead {
			st["acc"] = statusView{Alive: false, FailCount: 1, LastStatus: 429, UpdatedAt: 2}
		} else {
			st["acc"] = statusView{Alive: true, FailCount: 0, LastStatus: 200, UpdatedAt: 3}
		}
		_ = json.NewEncoder(w).Encode(credsResponse{Credentials: st})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := New(testConfig(srv.URL, true), slog.Default())
	c.pull(context.Background())
	assert.True(t, c.IsDead("acc"), "pull sees the dead state")

	state.Dead = false
	c.pull(context.Background())
	assert.False(t, c.IsDead("acc"), "pull picks up the revived state")
}

func TestReportPostsToService(t *testing.T) {
	var received []string
	var mu sync.Mutex
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/report", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		defer mu.Unlock()
		received = append(received, string(body))
		w.WriteHeader(http.StatusOK)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := New(testConfig(srv.URL, true), slog.Default())
	c.Report("cred-1", "gpt-4o", 429, 90)
	c.Report("cred-2", "gpt-4o", 200, 0)

	// The background loop is what drains the queue; drive the same path
	// synchronously here.
	for i := 0; i < 2; i++ {
		select {
		case item := <-c.queue:
			c.postReport(context.Background(), item)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 2, len(received), "two reports reached the service")
	assert.Contains(t, received[0], `"credential":"cred-1"`)
	assert.Contains(t, received[0], `"status_code":429`)
	assert.Contains(t, received[0], `"retry_after_seconds":90`)
	assert.Contains(t, received[1], `"credential":"cred-2"`)
}

func TestReportQueueDropsWhenFull(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1", true)
	cfg.ReportQueueSize = 2
	c := New(cfg, slog.Default())

	for i := 0; i < 10; i++ {
		c.Report("cred", "model", 200, 0)
	}
	assert.Equal(t, 2, QueueDepth(c), "queue never grows past its bound; excess reports are dropped")
}

// QueueDepth counts queued reports (test-only introspection).
func QueueDepth(c *Client) int {
	depth := 0
	for {
		select {
		case <-c.queue:
			depth++
		default:
			return depth
		}
	}
}

// mockStatusMux serves /v1/creds with a fixed status map.
func mockStatusMux(statuses map[string]statusView) *MockStatusHandler {
	return &MockStatusHandler{statuses: statuses}
}

type MockStatusHandler struct {
	statuses map[string]statusView
}

// ServeHTTP implements http.Handler for httptest.NewServer.
func (m *MockStatusHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/creds" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(credsResponse{Credentials: m.statuses})
}
// recordingSink records the ban lists it received.
type recordingSink struct {
	received []BanInfo
}

func (r *recordingSink) SyncBans(bans []BanInfo) {
	r.received = append(r.received, bans...)
}

func TestPullSyncsBansToSink(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/creds", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(credsResponse{Credentials: map[string]statusView{}})
	})
	mux.HandleFunc("/v1/bans", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"bans": []map[string]interface{}{
				{"credential": "acc1", "model": "gpt-4o", "until": 1000000, "status_code": 429, "reason": "status 429"},
				{"credential": "acc2", "model": "*", "until": 0, "status_code": 0, "reason": "manual"},
			},
		})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cfg := testConfig(srv.URL, true)
	cfg.ApplyBans = true
	c := New(cfg, slog.Default())
	sink := &recordingSink{}
	c.SetBanSynchronizer(sink)

	c.pull(context.Background())

	require.Equal(t, 2, len(sink.received), "both bans mirrored to the sink")
	var gotAcc1, gotAcc2 *BanInfo
	for i := range sink.received {
		if sink.received[i].Credential == "acc1" {
			gotAcc1 = &sink.received[i]
		}
		if sink.received[i].Credential == "acc2" {
			gotAcc2 = &sink.received[i]
		}
	}
	require.NotNil(t, gotAcc1, "acc1 ban mirrored")
	assert.Equal(t, "gpt-4o", gotAcc1.Model)
	assert.Equal(t, 429, gotAcc1.StatusCode)
	assert.Equal(t, int64(1000000), gotAcc1.Until.Unix())

	require.NotNil(t, gotAcc2, "acc2 ban mirrored")
	assert.Equal(t, "*", gotAcc2.Model)
	assert.True(t, gotAcc2.Until.IsZero(), "until 0 = permanent (zero time)")
}

func TestApplyBansDisabledSkipsBans(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/creds", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(credsResponse{Credentials: map[string]statusView{}})
	})
	mux.HandleFunc("/v1/bans", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("bans must not be fetched when apply_bans is off")
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cfg := testConfig(srv.URL, true)
	cfg.ApplyBans = false
	c := New(cfg, slog.Default())
	sink := &recordingSink{}
	c.SetBanSynchronizer(sink)

	c.pull(context.Background())
	assert.Len(t, sink.received, 0, "no bans mirrored when apply_bans is off")
}
