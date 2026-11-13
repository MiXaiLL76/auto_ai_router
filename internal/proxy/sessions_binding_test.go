package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stickyRouteRequest(ctx context.Context, sessionID string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"route-a","messages":[{"role":"user","content":"hi"}],"user":"`+sessionID+`"}`,
	)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer master-key")
	req.Header.Set("Content-Type", "application/json")
	return req
}

func okProvider(t *testing.T, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(chatCompletionOKBody("id-ok", "ok")))
	}))
}

func failingProvider(t *testing.T, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	return newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"failed"}}`))
	}))
}

func stickyProxy(t *testing.T, creds ...config.CredentialConfig) *Proxy {
	t.Helper()
	prx := NewTestProxyBuilder().
		WithCredentials(creds...).
		WithMasterKey("master-key").
		WithMaxProviderRetries(1).
		WithSessionSticky(5 * time.Minute).
		Build()
	for _, c := range creds {
		registerTestModel(prx, c.Name, "route-a")
	}
	return prx
}

func boundTo(t *testing.T, prx *Proxy, sessionID string) string {
	t.Helper()
	cred, found := prx.sessionStore.Get(sessionID, "route-a")
	if !found {
		return ""
	}
	return cred
}

// A bound credential that fails hands the session to the credential that served
// the turn: errors move people along the cascade.
func TestStickyBinding_MovesWhenBoundCredentialFails(t *testing.T) {
	var failingCalls, okCalls atomic.Int32
	failing := failingProvider(t, &failingCalls)
	defer failing.Close()
	ok := okProvider(t, &okCalls)
	defer ok.Close()

	bound := directDenylistCredential("bound-provider", failing.URL)
	other := directDenylistCredential("other-provider", ok.URL)
	prx := stickyProxy(t, bound, other)
	prx.sessionStore.Set("sess-1", "route-a", bound.Name)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-1"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), failingCalls.Load())
	assert.Equal(t, int32(1), okCalls.Load())
	assert.Equal(t, other.Name, boundTo(t, prx, "sess-1"))
}

// A banned bound credential is skipped and the session moves.
func TestStickyBinding_MovesWhenBoundCredentialBanned(t *testing.T) {
	var boundCalls, otherCalls atomic.Int32
	boundSrv := okProvider(t, &boundCalls)
	defer boundSrv.Close()
	otherSrv := okProvider(t, &otherCalls)
	defer otherSrv.Close()

	bound := directDenylistCredential("bound-provider", boundSrv.URL)
	other := directDenylistCredential("other-provider", otherSrv.URL)
	prx := stickyProxy(t, bound, other)
	prx.sessionStore.Set("sess-2", "route-a", bound.Name)
	prx.balancer.BanUntil(bound.Name, "route-a", http.StatusTooManyRequests, time.Now().Add(time.Minute), "test")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-2"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, boundCalls.Load())
	assert.Equal(t, other.Name, boundTo(t, prx, "sess-2"))
}

// A bound credential that is only busy (its own RPM limit) keeps the session:
// this turn goes elsewhere, the next one comes back.
func TestStickyBinding_HeldWhileBoundCredentialRateLimited(t *testing.T) {
	var boundCalls, otherCalls atomic.Int32
	boundSrv := okProvider(t, &boundCalls)
	defer boundSrv.Close()
	otherSrv := okProvider(t, &otherCalls)
	defer otherSrv.Close()

	bound := directDenylistCredential("bound-provider", boundSrv.URL)
	other := directDenylistCredential("other-provider", otherSrv.URL)
	prx := stickyProxy(t, bound, other)
	prx.rateLimiter.AddModelWithTPM(bound.Name, "route-a", 1, -1)
	require.True(t, prx.rateLimiter.AllowModel(bound.Name, "route-a"), "consume the only RPM slot")
	prx.sessionStore.Set("sess-3", "route-a", bound.Name)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-3"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, boundCalls.Load())
	assert.Equal(t, int32(1), otherCalls.Load())
	assert.Equal(t, bound.Name, boundTo(t, prx, "sess-3"), "a busy credential must keep its session")
}

// A credential excluded only for this request (reasoning-only on a plain turn)
// keeps the session.
func TestStickyBinding_HeldWhenBoundCredentialExcludedForThisRequest(t *testing.T) {
	var boundCalls, otherCalls atomic.Int32
	boundSrv := okProvider(t, &boundCalls)
	defer boundSrv.Close()
	otherSrv := okProvider(t, &otherCalls)
	defer otherSrv.Close()

	bound := directDenylistCredential("bound-provider", boundSrv.URL)
	bound.ReasoningOnly = true
	other := directDenylistCredential("other-provider", otherSrv.URL)
	prx := stickyProxy(t, bound, other)
	prx.sessionStore.Set("sess-4", "route-a", bound.Name)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-4"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Zero(t, boundCalls.Load())
	assert.Equal(t, bound.Name, boundTo(t, prx, "sess-4"))
}

// A turn served by a lower priority tier does not pin the session there: the
// next turn tries the primary tier again.
func TestStickyBinding_NotBoundToLowerTier(t *testing.T) {
	var primaryCalls, reserveCalls atomic.Int32
	primarySrv := okProvider(t, &primaryCalls)
	defer primarySrv.Close()
	reserveSrv := okProvider(t, &reserveCalls)
	defer reserveSrv.Close()

	primary := directDenylistCredential("primary-provider", primarySrv.URL)
	primary.Priority = 1
	reserve := directDenylistCredential("reserve-provider", reserveSrv.URL)
	reserve.Priority = 2
	prx := stickyProxy(t, primary, reserve)
	prx.balancer.BanUntil(primary.Name, "route-a", http.StatusTooManyRequests, time.Now().Add(time.Minute), "test")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-5"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), reserveCalls.Load())
	assert.Empty(t, boundTo(t, prx, "sess-5"), "a lower-tier credential must not become the session's home")
}

// Regression: a session was bound to a primary-tier credential that genuinely
// fails this turn (not merely busy or excluded) and the cascade falls back to
// a reserve-tier credential that serves it. The stale binding to the failed
// primary must be cleared, not left in place.
//
// With a single credential in the primary tier (as here), the next turn's
// plain round-robin still has nowhere else to go and will try that same
// credential again regardless of this fix - that part of the picture only
// improves once the primary tier has another live member. What this fix does
// guarantee, and what this test checks, is that the store stops asserting a
// binding to a credential that just failed: nothing short-circuits the next
// turn's selection straight back to it via the sticky lookup, and once the
// primary tier does have other members, round-robin gets a fair shot at them
// instead of every turn being pinned to the one that failed.
func TestStickyBinding_ClearedWhenBoundCredentialFailsAndRescuerIsLowerTier(t *testing.T) {
	var primaryCalls, reserveCalls atomic.Int32
	primarySrv := failingProvider(t, &primaryCalls)
	defer primarySrv.Close()
	reserveSrv := okProvider(t, &reserveCalls)
	defer reserveSrv.Close()

	primary := directDenylistCredential("primary-provider", primarySrv.URL)
	primary.Priority = 1
	reserve := directDenylistCredential("reserve-provider", reserveSrv.URL)
	reserve.Priority = 2
	prx := stickyProxy(t, primary, reserve)
	prx.sessionStore.Set("sess-6b", "route-a", primary.Name)

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-6b"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, int32(1), primaryCalls.Load(), "the failed primary was actually tried")
	assert.Equal(t, int32(1), reserveCalls.Load())
	assert.Empty(t, boundTo(t, prx, "sess-6b"), "the stale binding to the failed primary must not survive")
}

// A binding to a credential that no longer serves the model moves to the one
// that served the turn.
func TestStickyBinding_MovesWhenBoundCredentialNoLongerServesModel(t *testing.T) {
	var calls atomic.Int32
	srv := okProvider(t, &calls)
	defer srv.Close()

	only := directDenylistCredential("only-provider", srv.URL)
	prx := stickyProxy(t, only)
	prx.sessionStore.Set("sess-6", "route-a", "removed-provider")

	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(context.Background(), "sess-6"))

	require.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, only.Name, boundTo(t, prx, "sess-6"))
}

// A client that hangs up says nothing about the credential: the binding stays.
func TestStickyBinding_KeptWhenClientCancels(t *testing.T) {
	var calls atomic.Int32
	srv := okProvider(t, &calls)
	defer srv.Close()

	bound := directDenylistCredential("bound-provider", srv.URL)
	prx := stickyProxy(t, bound)
	prx.sessionStore.Set("sess-7", "route-a", bound.Name)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w := httptest.NewRecorder()
	prx.ProxyRequest(w, stickyRouteRequest(ctx, "sess-7"))

	assert.Equal(t, bound.Name, boundTo(t, prx, "sess-7"), "client cancellation must not clear the session binding")
}

func TestShouldClearSessionBindingOnFailure(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()

	cases := []struct {
		name   string
		ctx    context.Context
		logCtx RequestLogContext
		clear  bool
	}{
		{"upstream error", context.Background(), RequestLogContext{HTTPStatus: http.StatusBadGateway}, true},
		{"client closed request", context.Background(), RequestLogContext{HTTPStatus: StatusClientClosedRequest}, false},
		{"client canceled origin", context.Background(), RequestLogContext{ErrorOrigin: ErrorOriginClientCanceled}, false},
		{"stream aborted by client mid-way", context.Background(), RequestLogContext{StreamOutcome: "client_aborted"}, false},
		{"request context canceled", canceled, RequestLogContext{}, false},
		{"deadline is not a client cancel", expired, RequestLogContext{HTTPStatus: http.StatusGatewayTimeout}, true},
		{"binding held while busy", context.Background(), RequestLogContext{StickyHeldCredential: "busy"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.clear, shouldClearSessionBindingOnFailure(tc.ctx, &tc.logCtx))
		})
	}
}

func TestStickySessionKey_ScopedByClientKey(t *testing.T) {
	prx := NewTestProxyBuilder().WithMasterKey("master-key").WithSessionSticky(time.Minute).Build()

	master := prx.stickySessionKey(&RequestLogContext{SessionID: "1", Token: "master-key"})
	clientA := prx.stickySessionKey(&RequestLogContext{SessionID: "1", Token: "sk-client-a"})
	clientB := prx.stickySessionKey(&RequestLogContext{SessionID: "1", Token: "sk-client-b"})

	assert.Equal(t, "1", master, "master-key hops keep the raw session ID so a router chain shares it")
	assert.NotEqual(t, clientA, clientB, "the same session ID from two client keys must not share a binding")
	assert.True(t, strings.HasSuffix(clientA, ":1"))
	assert.Empty(t, prx.stickySessionKey(&RequestLogContext{Token: "sk-client-a"}))
}

// ── Redis-backed store ──────────────────────────────────────────────────────

// fakeSessionKV is an in-memory sessionKV whose failures can be switched on.
type fakeSessionKV struct {
	mu    sync.Mutex
	data  map[string]string
	fail  bool
	calls int
}

func newFakeSessionKV() *fakeSessionKV { return &fakeSessionKV{data: map[string]string{}} }

var errFakeRedisDown = errors.New("redis down")

func (f *fakeSessionKV) Get(_ context.Context, key string) (string, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.fail {
		return "", false, errFakeRedisDown
	}
	v, ok := f.data[key]
	return v, ok, nil
}

func (f *fakeSessionKV) Set(_ context.Context, key, value string, _ time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.fail {
		return errFakeRedisDown
	}
	f.data[key] = value
	return nil
}

func (f *fakeSessionKV) Del(_ context.Context, key string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.fail {
		return errFakeRedisDown
	}
	delete(f.data, key)
	return nil
}

func (f *fakeSessionKV) setFail(v bool) {
	f.mu.Lock()
	f.fail = v
	f.mu.Unlock()
}

func (f *fakeSessionKV) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func TestRedisSessionStore_SharedBetweenReplicas(t *testing.T) {
	kv := newFakeSessionKV()
	replicaA := newRedisSessionStoreWithKV(kv, "ru02:", time.Minute, nil)
	replicaB := newRedisSessionStoreWithKV(kv, "ru02:", time.Minute, nil)

	replicaA.Set("sess", "model-a", "cred-1")
	cred, found := replicaB.Get("sess", "model-a")
	require.True(t, found, "a binding written by one replica must be visible to another")
	assert.Equal(t, "cred-1", cred)

	replicaB.Delete("sess", "model-a")
	_, found = replicaA.Get("sess", "model-a")
	assert.False(t, found)
}

func TestRedisSessionStore_FallsBackToLocalAndBacksOff(t *testing.T) {
	kv := newFakeSessionKV()
	store := newRedisSessionStoreWithKV(kv, "ru02:", time.Minute, nil)
	now := time.Date(2026, time.October, 20, 12, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return now }

	store.Set("sess", "model-a", "cred-1")
	kv.setFail(true)

	cred, found := store.Get("sess", "model-a")
	require.True(t, found, "with Redis down the replica-local copy must answer")
	assert.Equal(t, "cred-1", cred)

	callsAfterFailure := kv.callCount()
	store.Set("sess", "model-a", "cred-2")
	cred, _ = store.Get("sess", "model-a")
	assert.Equal(t, "cred-2", cred)
	assert.Equal(t, callsAfterFailure, kv.callCount(), "Redis must not be retried inside the back-off window")

	kv.setFail(false)
	now = now.Add(redisSessionRetryAfter + time.Second)
	store.Set("sess", "model-a", "cred-3")
	assert.Greater(t, kv.callCount(), callsAfterFailure, "Redis must be retried after the back-off window")
	cred, found = newRedisSessionStoreWithKV(kv, "ru02:", time.Minute, nil).Get("sess", "model-a")
	require.True(t, found)
	assert.Equal(t, "cred-3", cred)
}

func TestRedisSessionStore_KeyHashesSessionID(t *testing.T) {
	s := &redisSessionStore{keyPrefix: "ru02:"}
	key := s.key(strings.Repeat("x", 10_000), "model-a")
	assert.True(t, strings.HasPrefix(key, "ru02:session:"))
	assert.Len(t, key, len("ru02:session:")+32)
	assert.NotEqual(t, s.key("a", "model-a"), s.key("a", "model-b"))
}

// Integration against a real Valkey (VALKEY_ADDR), same pattern as responsestore.
func TestRedisSessionStore_ValkeyIntegration(t *testing.T) {
	addr := os.Getenv("VALKEY_ADDR")
	if addr == "" {
		t.Skip("VALKEY_ADDR not set, skipping Redis integration test")
	}
	client, err := ratelimit.NewValkeyClient(config.RedisConfig{InitAddresses: []string{addr}, ForceSingleClient: true})
	require.NoError(t, err)
	t.Cleanup(client.Close)
	prefix := "test-sticky-" + t.Name() + ":"
	a := newRedisSessionStore(client, prefix, 100*time.Millisecond, nil)
	b := newRedisSessionStore(client, prefix, 100*time.Millisecond, nil)

	a.Set("sess", "model-a", "cred-1")
	cred, found := b.Get("sess", "model-a")
	require.True(t, found)
	assert.Equal(t, "cred-1", cred)

	time.Sleep(250 * time.Millisecond)
	_, found = b.Get("sess", "model-a")
	assert.False(t, found, "PX TTL must expire the binding")
}
