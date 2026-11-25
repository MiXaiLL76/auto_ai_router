package router

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/mixaill76/auto_ai_router/internal/proxy"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cappedRouter serves /v1/videos with handler under a cap of maxInFlight.
func cappedRouter(t *testing.T, maxInFlight int, handler http.HandlerFunc) *Router {
	t.Helper()
	cfg := &config.Config{Server: config.ServerConfig{MaxInFlightRequests: maxInFlight}}
	r := New(createTestProxy(), nil, testhelpers.NewTestMonitoringConfig("/health", false, ""), testhelpers.NewTestLogger(), cfg)
	r.SetVideoHandler(handler)
	return r
}

// gate holds every admitted request until it is opened.
type gate struct {
	open     chan struct{}
	entered  atomic.Int64
	inside   atomic.Int64
	maxSeen  atomic.Int64
	openOnce sync.Once
}

func newGate() *gate { return &gate{open: make(chan struct{})} }

func (g *gate) release() { g.openOnce.Do(func() { close(g.open) }) }

func (g *gate) handler(w http.ResponseWriter, _ *http.Request) {
	now := g.inside.Add(1)
	for {
		seen := g.maxSeen.Load()
		if now <= seen || g.maxSeen.CompareAndSwap(seen, now) {
			break
		}
	}
	g.entered.Add(1)
	<-g.open
	g.inside.Add(-1)
	w.WriteHeader(http.StatusOK)
}

func TestInFlightCapHoldsUnderConcurrency(t *testing.T) {
	const maxInFlight, clients = 5, 60
	g := newGate()
	defer g.release()
	r := cappedRouter(t, maxInFlight, g.handler)
	server := httptest.NewServer(r)
	defer server.Close()
	shedBefore := testutil.ToFloat64(monitoring.RequestsShedTotal)

	var shed, ok atomic.Int64
	var wg sync.WaitGroup
	for range clients {
		wg.Go(func() {
			resp, err := http.Post(server.URL+"/v1/videos", "application/json", strings.NewReader(`{"model":"m"}`))
			if !assert.NoError(t, err) {
				return
			}
			defer func() { _ = resp.Body.Close() }()
			switch resp.StatusCode {
			case http.StatusOK:
				ok.Add(1)
			case http.StatusServiceUnavailable:
				assert.Equal(t, "1", resp.Header.Get("Retry-After"))
				assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))
				var body struct {
					Error struct {
						Message string `json:"message"`
						Type    string `json:"type"`
						Code    string `json:"code"`
					} `json:"error"`
				}
				assert.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
				assert.Equal(t, "server_error", body.Error.Type)
				assert.Equal(t, proxy.ErrorCodeOverloaded, body.Error.Code)
				assert.Contains(t, body.Error.Message, "Too many requests in flight")
				shed.Add(1)
			default:
				t.Errorf("unexpected status %d", resp.StatusCode)
			}
		})
	}

	// Every request beyond the cap is answered while the admitted ones are still held.
	require.Eventually(t, func() bool {
		return g.entered.Load() == maxInFlight && shed.Load() == clients-maxInFlight
	}, 10*time.Second, 5*time.Millisecond)
	assert.Equal(t, int64(maxInFlight), r.inFlight.current.Load())
	g.release()
	wg.Wait()

	assert.Equal(t, int64(maxInFlight), ok.Load())
	assert.Equal(t, int64(clients-maxInFlight), shed.Load())
	assert.LessOrEqual(t, g.maxSeen.Load(), int64(maxInFlight))
	assert.Zero(t, r.inFlight.current.Load(), "every slot is back")
	assert.Equal(t, shedBefore+clients-maxInFlight, testutil.ToFloat64(monitoring.RequestsShedTotal))
}

// readTracker reports whether the request body was read.
type readTracker struct{ read atomic.Bool }

func (b *readTracker) Read(p []byte) (int, error) {
	b.read.Store(true)
	return 0, io.EOF
}

func TestInFlightCapRefusesBeforeTheBodyIsRead(t *testing.T) {
	g := newGate()
	defer g.release()
	r := cappedRouter(t, 1, g.handler)
	go r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/videos", nil))
	require.Eventually(t, func() bool { return g.entered.Load() == 1 }, 5*time.Second, time.Millisecond)

	body := &readTracker{}
	response := httptest.NewRecorder()
	r.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body))

	assert.Equal(t, http.StatusServiceUnavailable, response.Code)
	assert.False(t, body.read.Load())
	g.release()
	require.Eventually(t, func() bool { return r.inFlight.current.Load() == 0 }, 5*time.Second, time.Millisecond)
}

func TestInFlightCapNeverRefusesHealthOrInternalEndpoints(t *testing.T) {
	g := newGate()
	defer g.release()
	r := cappedRouter(t, 1, g.handler)
	r.SetReady(true)
	go r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/videos", nil))
	require.Eventually(t, func() bool { return g.entered.Load() == 1 }, 5*time.Second, time.Millisecond)
	shedBefore := testutil.ToFloat64(monitoring.RequestsShedTotal)

	for _, path := range []string{"/health", "/health/readiness", "/trace", "/vhealth", "/api/bans", "/webui/style.css", "/litellm/.well-known/litellm-ui-config"} {
		response := httptest.NewRecorder()
		r.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		assert.Empty(t, response.Header().Get("Retry-After"), path)
		assert.NotContains(t, response.Body.String(), "Too many requests in flight", path)
	}
	assert.Equal(t, shedBefore, testutil.ToFloat64(monitoring.RequestsShedTotal))

	// The legacy unversioned API paths count as the /v1/ ones they are served as.
	response := httptest.NewRecorder()
	r.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/chat/completions", nil))
	assert.Equal(t, http.StatusServiceUnavailable, response.Code)
}

// streamHandler writes a chunk every 10 ms until it has written chunks or the client leaves.
func streamHandler(chunks int, done chan<- struct{}) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		defer func() { done <- struct{}{} }()
		flusher := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		for i := range chunks {
			if _, err := fmt.Fprintf(w, "data: %d\n\n", i); err != nil {
				return
			}
			flusher.Flush()
			select {
			case <-req.Context().Done():
				return
			case <-time.After(10 * time.Millisecond):
			}
		}
	}
}

func TestInFlightStreamsReleaseTheirSlotWhenTheyEnd(t *testing.T) {
	done := make(chan struct{}, 1)
	r := cappedRouter(t, 1, streamHandler(5, done))
	server := httptest.NewServer(r)
	defer server.Close()

	resp, err := http.Post(server.URL+"/v1/videos", "application/json", nil)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, int64(1), r.inFlight.current.Load(), "the stream holds its slot")
	data, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	_ = resp.Body.Close()
	assert.Contains(t, string(data), "data: 4")
	<-done

	require.Eventually(t, func() bool { return r.inFlight.current.Load() == 0 }, 5*time.Second, time.Millisecond)
	next, err := http.Post(server.URL+"/v1/videos", "application/json", nil)
	require.NoError(t, err)
	_, _ = io.Copy(io.Discard, next.Body)
	_ = next.Body.Close()
	assert.Equal(t, http.StatusOK, next.StatusCode, "the freed slot takes the next request")
	<-done
}

func TestInFlightStreamsReleaseTheirSlotWhenTheClientLeaves(t *testing.T) {
	done := make(chan struct{}, 1)
	r := cappedRouter(t, 1, streamHandler(100000, done))
	server := httptest.NewServer(r)
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/videos", nil)
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	require.NoError(t, err)
	assert.Equal(t, "data: 0\n", line)

	refused, err := http.Post(server.URL+"/v1/videos", "application/json", nil)
	require.NoError(t, err)
	_ = refused.Body.Close()
	assert.Equal(t, http.StatusServiceUnavailable, refused.StatusCode, "the stream still holds the only slot")

	cancel()
	_ = resp.Body.Close()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the handler did not see the client leave")
	}
	require.Eventually(t, func() bool { return r.inFlight.current.Load() == 0 }, 5*time.Second, time.Millisecond)
}

func TestInFlightCapIsOffByDefault(t *testing.T) {
	g := newGate()
	r := cappedRouter(t, 0, g.handler)
	server := httptest.NewServer(r)
	defer server.Close()

	var wg sync.WaitGroup
	statuses := make(chan int, 50)
	for range 50 {
		wg.Go(func() {
			resp, err := http.Post(server.URL+"/v1/videos", "application/json", nil)
			if assert.NoError(t, err) {
				statuses <- resp.StatusCode
				_ = resp.Body.Close()
			}
		})
	}
	require.Eventually(t, func() bool { return g.entered.Load() == 50 }, 10*time.Second, time.Millisecond)
	g.release()
	wg.Wait()
	close(statuses)
	for status := range statuses {
		assert.Equal(t, http.StatusOK, status)
	}
	assert.Zero(t, r.inFlight.current.Load())
}
