package monitoring

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Per-key request metrics (monitoring.key_metrics). Deliberately minimal:
//
//	auto_ai_router_key_requests_total{key, status}      counter
//	auto_ai_router_key_info{key, <info_labels...>} = 1  gauge
//
// Owner dimensions (alias, user, team, org) live only on the info series and
// are joined in PromQL, e.g. requests per minute by team:
//
//	sum by (team_alias) (
//	  rate(auto_ai_router_key_requests_total[5m]) * 60
//	  * on (key, instance) group_left (team_alias) auto_ai_router_key_info
//	)
//
// This keeps the counter's cardinality at keys x statuses no matter how many
// owner labels are configured, and an alias rename only replaces one info
// series instead of forking every counter.

const (
	// KeyHashPrefixLen is the number of hex characters of the token's sha256
	// hash exposed as the key label. The full hash must never be exposed: the
	// LiteLLM auth path accepts an already-hashed token as a bearer credential
	// (see auth.HashToken), so the full hash on an unauthenticated /metrics
	// endpoint would leak working credentials. 12 hex chars (48 bits) are
	// collision-free for any realistic key count and still let an operator
	// find the key with `WHERE token LIKE '<prefix>%'`.
	KeyHashPrefixLen = 12

	// KeyLabelMaster labels requests authenticated with the router master key.
	KeyLabelMaster = "master"
	// KeyLabelOverflow collects keys first seen after max_keys was reached.
	KeyLabelOverflow = "__other__"
)

// Info label names accepted in monitoring.key_metrics.info_labels.
const (
	KeyInfoLabelKeyAlias       = "key_alias"
	KeyInfoLabelUserID         = "user_id"
	KeyInfoLabelUserEmail      = "user_email"
	KeyInfoLabelTeamID         = "team_id"
	KeyInfoLabelTeamAlias      = "team_alias"
	KeyInfoLabelOrganizationID = "organization_id"
)

// KeyInfoLabels lists every supported info label.
var KeyInfoLabels = []string{
	KeyInfoLabelKeyAlias,
	KeyInfoLabelUserID,
	KeyInfoLabelUserEmail,
	KeyInfoLabelTeamID,
	KeyInfoLabelTeamAlias,
	KeyInfoLabelOrganizationID,
}

// DefaultKeyInfoLabels is used when info_labels is not configured. user_email
// is opt-in: /metrics is usually unauthenticated inside the cluster.
var DefaultKeyInfoLabels = []string{
	KeyInfoLabelKeyAlias,
	KeyInfoLabelUserID,
	KeyInfoLabelTeamID,
	KeyInfoLabelTeamAlias,
	KeyInfoLabelOrganizationID,
}

// KeyIdentity describes the authenticated caller of one request.
type KeyIdentity struct {
	// Key is the exposed key label: a hash prefix, KeyLabelMaster, or empty
	// when the request is not attributable to a key (nothing is recorded).
	Key            string
	KeyAlias       string
	UserID         string
	UserEmail      string
	TeamID         string
	TeamAlias      string
	OrganizationID string
}

func (id KeyIdentity) infoValue(label string) string {
	switch label {
	case KeyInfoLabelKeyAlias:
		return id.KeyAlias
	case KeyInfoLabelUserID:
		return id.UserID
	case KeyInfoLabelUserEmail:
		return id.UserEmail
	case KeyInfoLabelTeamID:
		return id.TeamID
	case KeyInfoLabelTeamAlias:
		return id.TeamAlias
	case KeyInfoLabelOrganizationID:
		return id.OrganizationID
	}
	return ""
}

// KeyMetricsOptions configures NewKeyMetrics.
type KeyMetricsOptions struct {
	InfoLabels []string
	MaxKeys    int           // <= 0 means unlimited
	IdleTTL    time.Duration // <= 0 disables idle eviction
}

type keyState struct {
	info     []string // current info label values; nil for the overflow key
	statuses map[string]struct{}
	lastSeen time.Time
}

// KeyMetrics records per-key request counts. A nil *KeyMetrics is a valid,
// disabled recorder, so call sites never need their own enabled check.
type KeyMetrics struct {
	infoLabels []string
	maxKeys    int
	idleTTL    time.Duration

	requests *prometheus.CounterVec
	info     *prometheus.GaugeVec

	mu   sync.Mutex
	keys map[string]*keyState
	now  func() time.Time
}

// NewKeyMetrics registers the per-key collectors on reg.
func NewKeyMetrics(reg prometheus.Registerer, opts KeyMetricsOptions) (*KeyMetrics, error) {
	infoLabels := opts.InfoLabels
	if infoLabels == nil {
		infoLabels = DefaultKeyInfoLabels
	}
	m := &KeyMetrics{
		infoLabels: append([]string(nil), infoLabels...),
		maxKeys:    opts.MaxKeys,
		idleTTL:    opts.IdleTTL,
		requests: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "auto_ai_router_key_requests_total",
				Help: "Total client requests per API key (hash prefix) and final HTTP status; join with auto_ai_router_key_info for owner labels",
			},
			[]string{"key", "status"},
		),
		info: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "auto_ai_router_key_info",
				Help: "Owner metadata for each API key seen by the router (value is always 1)",
			},
			append([]string{"key"}, infoLabels...),
		),
		keys: make(map[string]*keyState),
		now:  time.Now,
	}
	if err := reg.Register(m.requests); err != nil {
		return nil, err
	}
	if err := reg.Register(m.info); err != nil {
		reg.Unregister(m.requests)
		return nil, err
	}
	return m, nil
}

// Observe counts one finished request for id with the HTTP status the client
// received.
func (m *KeyMetrics) Observe(id KeyIdentity, statusCode int) {
	if m == nil || id.Key == "" {
		return
	}
	status := "unknown"
	if statusCode >= 100 && statusCode <= 599 {
		status = strconv.Itoa(statusCode)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	key := id.Key
	state, ok := m.keys[key]
	if !ok && m.maxKeys > 0 && len(m.keys) >= m.maxKeys {
		key = KeyLabelOverflow
		state, ok = m.keys[key]
	}
	if !ok {
		state = &keyState{statuses: make(map[string]struct{})}
		m.keys[key] = state
	}
	state.lastSeen = m.now()
	if key != KeyLabelOverflow {
		m.setInfoLocked(key, state, id)
	}
	state.statuses[status] = struct{}{}
	m.requests.WithLabelValues(key, status).Inc()
}

// setInfoLocked keeps exactly one info series per key: a changed alias/owner
// replaces the old series, otherwise PromQL joins would hit many-to-many.
func (m *KeyMetrics) setInfoLocked(key string, state *keyState, id KeyIdentity) {
	values := make([]string, 0, len(m.infoLabels)+1)
	values = append(values, key)
	for _, label := range m.infoLabels {
		values = append(values, id.infoValue(label))
	}
	if state.info != nil {
		if equalStrings(state.info, values) {
			return
		}
		m.info.DeleteLabelValues(state.info...)
	}
	state.info = values
	m.info.WithLabelValues(values...).Set(1)
}

// evictIdle drops every series of keys without traffic for idleTTL, bounding
// memory and scrape size for keys that stopped being used. A returning key
// starts again from zero, which rate()/increase() treat as a counter reset.
func (m *KeyMetrics) evictIdle() {
	if m == nil || m.idleTTL <= 0 {
		return
	}
	cutoff := m.now().Add(-m.idleTTL)
	m.mu.Lock()
	defer m.mu.Unlock()
	for key, state := range m.keys {
		if state.lastSeen.After(cutoff) {
			continue
		}
		for status := range state.statuses {
			m.requests.DeleteLabelValues(key, status)
		}
		if state.info != nil {
			m.info.DeleteLabelValues(state.info...)
		}
		delete(m.keys, key)
	}
}

// Run evicts idle keys until ctx is done. It returns immediately when
// eviction is disabled.
func (m *KeyMetrics) Run(ctx context.Context) {
	if m == nil || m.idleTTL <= 0 {
		return
	}
	interval := min(max(m.idleTTL/10, 10*time.Second), 5*time.Minute)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.evictIdle()
		}
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ==================== Per-request identity slot ====================

type keySlotContextKey struct{}

type keySlot struct {
	identity atomic.Pointer[KeyIdentity]
}

// WithKeyIdentitySlot returns a context carrying a fresh identity slot and a
// function that reports the identity stored into it by
// SetRequestKeyIdentity. Each logical request (an HTTP request, or one
// WebSocket turn replayed through the proxy) gets its own slot.
func WithKeyIdentitySlot(ctx context.Context) (context.Context, func() (KeyIdentity, bool)) {
	slot := &keySlot{}
	return context.WithValue(ctx, keySlotContextKey{}, slot), func() (KeyIdentity, bool) {
		if id := slot.identity.Load(); id != nil {
			return *id, true
		}
		return KeyIdentity{}, false
	}
}

// SetRequestKeyIdentity records the authenticated key of the current request
// so its owner (Middleware or a WebSocket turn) can count it once the response
// is finished. It is a no-op without a slot in ctx.
func SetRequestKeyIdentity(ctx context.Context, id KeyIdentity) {
	if ctx == nil || id.Key == "" {
		return
	}
	if slot, ok := ctx.Value(keySlotContextKey{}).(*keySlot); ok {
		slot.identity.Store(&id)
	}
}

// ObserveStatus is Observe for a response writer's captured status, where 0
// means nothing set the status explicitly, i.e. an implicit 200.
func (m *KeyMetrics) ObserveStatus(id KeyIdentity, statusCode int) {
	if statusCode == 0 {
		statusCode = http.StatusOK
	}
	m.Observe(id, statusCode)
}

// Middleware counts every HTTP request that resolved to a key by the status
// actually written to the client. Requests without a recognizable key
// (missing/unknown token) are not attributable and are skipped. Hijacked
// connections (WebSocket upgrades) are skipped too: their turns are counted
// individually by the WebSocket handlers, not as one long-lived 101.
func (m *KeyMetrics) Middleware(next http.Handler) http.Handler {
	if m == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, identity := WithKeyIdentitySlot(r.Context())
		sw := &statusWriter{ResponseWriter: w}
		defer func() {
			if sw.hijacked {
				return
			}
			if id, ok := identity(); ok {
				m.ObserveStatus(id, sw.status)
			}
		}()
		next.ServeHTTP(sw, r.WithContext(ctx))
	})
}

// statusWriter captures the first status code written. It forwards Flush and
// Hijack explicitly because the proxy (streaming) and gorilla/websocket use
// direct interface assertions rather than http.ResponseController.
type statusWriter struct {
	http.ResponseWriter
	status   int
	hijacked bool
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 && code >= 200 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(p)
}

func (w *statusWriter) Flush() {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *statusWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	conn, rw, err := http.NewResponseController(w.ResponseWriter).Hijack()
	if err == nil {
		w.hijacked = true
	}
	return conn, rw, err
}

func (w *statusWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}
