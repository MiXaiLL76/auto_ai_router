package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/monitoring"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKeyIdentityFromTokenInfo(t *testing.T) {
	assert.Equal(t, monitoring.KeyIdentity{}, keyIdentityFromTokenInfo(nil))

	hash := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	id := keyIdentityFromTokenInfo(&dbmodels.TokenInfo{
		Token:          hash,
		KeyAlias:       "ci-bot",
		UserID:         "u1",
		UserEmail:      "u1@example.com",
		TeamID:         "t1",
		TeamAlias:      "ml",
		OrganizationID: "o1",
	})
	assert.Equal(t, monitoring.KeyIdentity{
		Key:            "0123456789ab",
		KeyAlias:       "ci-bot",
		UserID:         "u1",
		UserEmail:      "u1@example.com",
		TeamID:         "t1",
		TeamAlias:      "ml",
		OrganizationID: "o1",
	}, id)
	// The full hash is itself a bearer credential and must never be exposed.
	assert.NotContains(t, id.Key, hash)

	// Master key: HashToken of a non-sk- master key is the raw key, so no
	// part of it may leak into a label.
	master := keyIdentityFromTokenInfo(&dbmodels.TokenInfo{Token: "super-secret-master-key", IsMasterKey: true})
	assert.Equal(t, monitoring.KeyIdentity{Key: monitoring.KeyLabelMaster, KeyAlias: monitoring.KeyLabelMaster}, master)

	// Too short to be a stored hash: not attributable.
	assert.Equal(t, monitoring.KeyIdentity{}, keyIdentityFromTokenInfo(&dbmodels.TokenInfo{Token: "short"}))
}

func TestKeyMetricsCountAuthenticatedRequestsThroughMiddleware(t *testing.T) {
	hash := "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
	db := &clientAuthTestDB{
		tokens: map[string]*dbmodels.TokenInfo{"tenant-key": {Token: hash, KeyAlias: "ci", TeamAlias: "ml"}},
		errors: map[string]error{"blocked-key": litellmdb.ErrTokenBlocked},
	}
	prx := newClientAuthTestProxy(t, db, "http://example.invalid", config.ProviderTypeOpenAI, "provider-key")
	reg := prometheus.NewRegistry()
	km, err := monitoring.NewKeyMetrics(reg, monitoring.KeyMetricsOptions{InfoLabels: []string{monitoring.KeyInfoLabelKeyAlias}})
	require.NoError(t, err)
	prx.keyMetrics = km

	handler := km.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, ok := prx.AuthenticateClientRequest(w, r); !ok {
			return
		}
		if r.URL.Path == "/limited" {
			WriteErrorRateLimit(w, "Rate limit exceeded for key")
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	for _, tc := range []struct{ path, token string }{
		{"/ok", "tenant-key"},
		{"/ok", "tenant-key"},
		{"/limited", "tenant-key"},
		{"/ok", "master-key"},
		{"/ok", "blocked-key"},   // rejected at auth: no identity to attribute
		{"/ok", "unknown-token"}, // ditto
	} {
		req := httptest.NewRequest(http.MethodPost, tc.path, nil)
		req.Header.Set("Authorization", "Bearer "+tc.token)
		handler.ServeHTTP(httptest.NewRecorder(), req)
	}

	expected := `
# HELP auto_ai_router_key_info Owner metadata for each API key seen by the router (value is always 1)
# TYPE auto_ai_router_key_info gauge
auto_ai_router_key_info{key="fedcba987654",key_alias="ci"} 1
auto_ai_router_key_info{key="master",key_alias="master"} 1
# HELP auto_ai_router_key_requests_total Total client requests per API key (hash prefix) and final HTTP status; join with auto_ai_router_key_info for owner labels
# TYPE auto_ai_router_key_requests_total counter
auto_ai_router_key_requests_total{key="fedcba987654",status="200"} 2
auto_ai_router_key_requests_total{key="fedcba987654",status="429"} 1
auto_ai_router_key_requests_total{key="master",status="200"} 1
`
	require.NoError(t, testutil.GatherAndCompare(reg, strings.NewReader(expected)))
}

func newKeyTurnTestProxy(t *testing.T) (*Proxy, *prometheus.Registry) {
	t.Helper()
	reg := prometheus.NewRegistry()
	km, err := monitoring.NewKeyMetrics(reg, monitoring.KeyMetricsOptions{InfoLabels: []string{}})
	require.NoError(t, err)
	return &Proxy{keyMetrics: km}, reg
}

var keyTurnTestInfo = &dbmodels.TokenInfo{Token: "0123456789abcdef0123456789abcdef"}

func assertKeyTurnCounts(t *testing.T, reg *prometheus.Registry, counts ...string) {
	t.Helper()
	expected := `
# HELP auto_ai_router_key_requests_total Total client requests per API key (hash prefix) and final HTTP status; join with auto_ai_router_key_info for owner labels
# TYPE auto_ai_router_key_requests_total counter
`
	for i := 0; i < len(counts); i += 2 {
		expected += `auto_ai_router_key_requests_total{key="0123456789ab",status="` + counts[i] + `"} ` + counts[i+1] + "\n"
	}
	require.NoError(t, testutil.GatherAndCompare(reg, strings.NewReader(expected), "auto_ai_router_key_requests_total"))
}

func TestWithKeyTurn(t *testing.T) {
	var disabled Proxy
	ctx := context.Background()
	gotCtx, finish := disabled.withKeyTurn(ctx)
	assert.Equal(t, ctx, gotCtx)
	finish(http.StatusOK) // no-op without metrics

	prx, reg := newKeyTurnTestProxy(t)

	// The WS upgrade's own slot must not be touched by the turn.
	outerCtx, outer := monitoring.WithKeyIdentitySlot(context.Background())
	turnCtx, finish := prx.withKeyTurn(outerCtx)
	prx.noteRequestKey(turnCtx, keyTurnTestInfo)
	finish(0)
	_, ok := outer()
	assert.False(t, ok)

	_, finish = prx.withKeyTurn(context.Background())
	finish(http.StatusBadGateway) // turn never authenticated: not counted

	turnCtx, finish = prx.withKeyTurn(context.Background())
	prx.noteRequestKey(turnCtx, keyTurnTestInfo)
	finish(StatusClientClosedRequest)

	assertKeyTurnCounts(t, reg, "200", "1", "499", "1")
}

func TestNativeWSReleasePendingCountsSteer(t *testing.T) {
	prx, reg := newKeyTurnTestProxy(t)
	s := &nativeWSSession{proxy: prx}
	queue := func() {
		ctx, finish := prx.withKeyTurn(context.Background())
		prx.noteRequestKey(ctx, keyTurnTestInfo)
		s.pending = &nativeWSTurn{log: &RequestLogContext{}, finishKey: finish}
	}

	queue()
	s.releasePending(http.StatusConflict) // steer.failed
	queue()
	s.releasePending(StatusClientClosedRequest) // session closed after client abort
	s.releasePending(http.StatusOK)             // nothing pending: not counted

	assertKeyTurnCounts(t, reg, "409", "1", "499", "1")
}

func TestWSSSEWriterDetectsClientGone(t *testing.T) {
	serverConn := make(chan *websocket.Conn, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		require.NoError(t, err)
		serverConn <- conn
	}))
	defer srv.Close()
	client, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(srv.URL, "http"), nil)
	require.NoError(t, err)
	conn := <-serverConn
	defer func() { _ = conn.Close() }()

	w := newWSSSEWriter(conn)
	_, _ = w.Write([]byte("data: {\"type\":\"response.created\"}\n\n"))
	assert.False(t, w.clientGone)

	_ = client.Close()
	// The first writes after a disconnect may still land in socket buffers.
	require.Eventually(t, func() bool {
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\"}\n\n"))
		w.writeMu.Lock()
		defer w.writeMu.Unlock()
		return w.clientGone
	}, 5*time.Second, 10*time.Millisecond)
}
