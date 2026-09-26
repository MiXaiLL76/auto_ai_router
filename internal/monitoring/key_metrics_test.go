package monitoring

import (
	"bufio"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestKeyMetrics(t *testing.T, opts KeyMetricsOptions) (*KeyMetrics, *prometheus.Registry) {
	t.Helper()
	reg := prometheus.NewRegistry()
	m, err := NewKeyMetrics(reg, opts)
	require.NoError(t, err)
	return m, reg
}

func keyRequests(m *KeyMetrics, key, status string) float64 {
	return testutil.ToFloat64(m.requests.WithLabelValues(key, status))
}

func TestKeyMetrics_ObserveCountsByKeyAndStatus(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	id := KeyIdentity{Key: "abc123def456", KeyAlias: "ci", TeamAlias: "ml"}

	m.Observe(id, 200)
	m.Observe(id, 200)
	m.Observe(id, 429)
	m.Observe(id, 0)
	m.ObserveStatus(id, 0)

	assert.Equal(t, 3.0, keyRequests(m, "abc123def456", "200"))
	assert.Equal(t, 1.0, keyRequests(m, "abc123def456", "429"))
	assert.Equal(t, 1.0, keyRequests(m, "abc123def456", "unknown"))
	assert.Equal(t, 1, testutil.CollectAndCount(m.info))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.info.WithLabelValues("abc123def456", "ci", "", "", "ml", "")))
}

func TestKeyMetrics_EmptyKeyAndNilAreNoops(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	m.Observe(KeyIdentity{KeyAlias: "no-key"}, 200)
	assert.Equal(t, 0, testutil.CollectAndCount(m.requests))

	var disabled *KeyMetrics
	disabled.Observe(KeyIdentity{Key: "k"}, 200)
	disabled.ObserveStatus(KeyIdentity{Key: "k"}, 0)
	disabled.evictIdle()
	h := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	assert.NotNil(t, disabled.Middleware(h))
}

func TestKeyMetrics_InfoReplacedOnOwnerChange(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{InfoLabels: []string{KeyInfoLabelKeyAlias}})

	m.Observe(KeyIdentity{Key: "k1", KeyAlias: "old"}, 200)
	m.Observe(KeyIdentity{Key: "k1", KeyAlias: "new"}, 200)

	// Exactly one info series per key, or PromQL group_left joins break.
	assert.Equal(t, 1, testutil.CollectAndCount(m.info))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.info.WithLabelValues("k1", "new")))
	assert.Equal(t, 2.0, keyRequests(m, "k1", "200"))
}

func TestKeyMetrics_CustomInfoLabels(t *testing.T) {
	_, reg := newTestKeyMetrics(t, KeyMetricsOptions{InfoLabels: []string{KeyInfoLabelTeamID, KeyInfoLabelUserEmail}})
	m2, err := NewKeyMetrics(prometheus.NewRegistry(), KeyMetricsOptions{InfoLabels: []string{}})
	require.NoError(t, err)
	m2.Observe(KeyIdentity{Key: "k", KeyAlias: "ignored"}, 200)
	assert.Equal(t, 1.0, testutil.ToFloat64(m2.info.WithLabelValues("k")))

	m, err := NewKeyMetrics(prometheus.NewRegistry(), KeyMetricsOptions{InfoLabels: []string{KeyInfoLabelTeamID, KeyInfoLabelUserEmail}})
	require.NoError(t, err)
	m.Observe(KeyIdentity{Key: "k", TeamID: "t1", UserEmail: "a@b.c", KeyAlias: "ignored"}, 200)
	assert.Equal(t, 1.0, testutil.ToFloat64(m.info.WithLabelValues("k", "t1", "a@b.c")))

	// Registering the same collectors twice on one registry must fail
	// instead of panicking.
	_, err = NewKeyMetrics(reg, KeyMetricsOptions{})
	assert.Error(t, err)
}

func TestKeyMetrics_MaxKeysOverflow(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{MaxKeys: 2})

	m.Observe(KeyIdentity{Key: "k1"}, 200)
	m.Observe(KeyIdentity{Key: "k2"}, 200)
	m.Observe(KeyIdentity{Key: "k3"}, 200)
	m.Observe(KeyIdentity{Key: "k4"}, 500)
	m.Observe(KeyIdentity{Key: "k1"}, 200) // already tracked: keeps its own label

	assert.Equal(t, 2.0, keyRequests(m, "k1", "200"))
	assert.Equal(t, 1.0, keyRequests(m, "k2", "200"))
	assert.Equal(t, 1.0, keyRequests(m, KeyLabelOverflow, "200"))
	assert.Equal(t, 1.0, keyRequests(m, KeyLabelOverflow, "500"))
	// No info series for the overflow bucket.
	assert.Equal(t, 2, testutil.CollectAndCount(m.info))
}

func TestKeyMetrics_EvictIdle(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{IdleTTL: time.Hour, MaxKeys: 1})
	now := time.Unix(1_700_000_000, 0)
	m.now = func() time.Time { return now }

	m.Observe(KeyIdentity{Key: "idle", KeyAlias: "a"}, 200)
	m.Observe(KeyIdentity{Key: "idle", KeyAlias: "a"}, 401)
	now = now.Add(30 * time.Minute)
	m.evictIdle()
	assert.Equal(t, 2, testutil.CollectAndCount(m.requests))

	now = now.Add(31 * time.Minute)
	m.evictIdle()
	assert.Equal(t, 0, testutil.CollectAndCount(m.requests))
	assert.Equal(t, 0, testutil.CollectAndCount(m.info))

	// Eviction frees the max_keys slot for a new key.
	m.Observe(KeyIdentity{Key: "fresh"}, 200)
	assert.Equal(t, 1.0, keyRequests(m, "fresh", "200"))
}

func TestKeyMetrics_EvictDisabled(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	m.now = func() time.Time { return time.Unix(0, 0) }
	m.Observe(KeyIdentity{Key: "k"}, 200)
	m.now = time.Now
	m.evictIdle()
	assert.Equal(t, 1, testutil.CollectAndCount(m.requests))
}

func TestKeyMetrics_MiddlewareRecordsFinalStatus(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	handler := m.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		SetRequestKeyIdentity(r.Context(), KeyIdentity{Key: r.Header.Get("X-Key")})
		switch r.URL.Path {
		case "/limited":
			w.WriteHeader(http.StatusTooManyRequests)
			w.WriteHeader(http.StatusOK) // superfluous; first status wins
		case "/implicit":
			_, _ = w.Write([]byte("ok"))
		case "/stream":
			w.(http.Flusher).Flush()
			_, _ = w.Write([]byte("data: x\n\n"))
		}
	}))

	for _, path := range []string{"/limited", "/implicit", "/stream", "/empty"} {
		req := httptest.NewRequest(http.MethodPost, path, nil)
		req.Header.Set("X-Key", "k1")
		handler.ServeHTTP(httptest.NewRecorder(), req)
	}
	// Unauthenticated request: no identity, nothing recorded.
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/limited", nil))

	assert.Equal(t, 1.0, keyRequests(m, "k1", "429"))
	assert.Equal(t, 3.0, keyRequests(m, "k1", "200"))
	assert.Equal(t, 2, testutil.CollectAndCount(m.requests))
}

type hijackableRecorder struct {
	*httptest.ResponseRecorder
}

func (h hijackableRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	server, client := net.Pipe()
	_ = client.Close()
	return server, bufio.NewReadWriter(bufio.NewReader(server), bufio.NewWriter(server)), nil
}

func TestKeyMetrics_MiddlewareSkipsHijacked(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	handler := m.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		SetRequestKeyIdentity(r.Context(), KeyIdentity{Key: "ws"})
		// gorilla/websocket asserts http.Hijacker directly.
		conn, _, err := w.(http.Hijacker).Hijack()
		require.NoError(t, err)
		_ = conn.Close()
	}))
	handler.ServeHTTP(hijackableRecorder{httptest.NewRecorder()}, httptest.NewRequest(http.MethodGet, "/v1/responses", nil))
	assert.Equal(t, 0, testutil.CollectAndCount(m.requests))
}

func TestKeyMetrics_MiddlewareHijackUnsupported(t *testing.T) {
	m, _ := newTestKeyMetrics(t, KeyMetricsOptions{})
	handler := m.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		SetRequestKeyIdentity(r.Context(), KeyIdentity{Key: "k"})
		_, _, err := w.(http.Hijacker).Hijack()
		assert.ErrorIs(t, err, http.ErrNotSupported)
		w.WriteHeader(http.StatusBadRequest)
	}))
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
	assert.Equal(t, 1.0, keyRequests(m, "k", "400"))
}

func TestWithKeyIdentitySlot_IsolatesTurns(t *testing.T) {
	outerCtx, outer := WithKeyIdentitySlot(httptest.NewRequest(http.MethodGet, "/", nil).Context())
	SetRequestKeyIdentity(outerCtx, KeyIdentity{Key: "session"})

	turnCtx, turn := WithKeyIdentitySlot(outerCtx)
	_, ok := turn()
	assert.False(t, ok)
	SetRequestKeyIdentity(turnCtx, KeyIdentity{Key: "turn"})

	id, ok := turn()
	require.True(t, ok)
	assert.Equal(t, "turn", id.Key)
	id, _ = outer()
	assert.Equal(t, "session", id.Key)

	// No slot, or empty key: silently ignored.
	SetRequestKeyIdentity(httptest.NewRequest(http.MethodGet, "/", nil).Context(), KeyIdentity{Key: "x"})
	SetRequestKeyIdentity(turnCtx, KeyIdentity{})
	id, _ = turn()
	assert.Equal(t, "turn", id.Key)
}

func TestKeyMetrics_ExposedNames(t *testing.T) {
	m, reg := newTestKeyMetrics(t, KeyMetricsOptions{InfoLabels: []string{KeyInfoLabelTeamAlias}})
	m.Observe(KeyIdentity{Key: "k", TeamAlias: "ml"}, 200)
	expected := `
# HELP auto_ai_router_key_info Owner metadata for each API key seen by the router (value is always 1)
# TYPE auto_ai_router_key_info gauge
auto_ai_router_key_info{key="k",team_alias="ml"} 1
`
	require.NoError(t, testutil.GatherAndCompare(reg, strings.NewReader(expected), "auto_ai_router_key_info"))
}
