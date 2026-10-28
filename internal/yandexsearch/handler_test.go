package yandexsearch

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

const validSearchBody = `{"query":{"queryText":"test query"}}`

type fakeAuth struct {
	tokenInfo *dbmodels.TokenInfo
	scopeCtx  scope.Context
	ok        bool
}

func (f fakeAuth) AuthenticateClientRequestScoped(w http.ResponseWriter, r *http.Request) (*dbmodels.TokenInfo, scope.Context, bool) {
	if !f.ok {
		w.WriteHeader(http.StatusUnauthorized)
	}
	return f.tokenInfo, f.scopeCtx, f.ok
}

func allowedAuth(key string) fakeAuth {
	return fakeAuth{
		tokenInfo: &dbmodels.TokenInfo{Token: key, UserID: "user-1", TeamID: "team-1", OrganizationID: "org-1"},
		scopeCtx:  scope.NewContext([]string{accessScope}, nil),
		ok:        true,
	}
}

type fakeSpend struct {
	calls []*dbmodels.SpendLogEntry
	err   error
}

func (f *fakeSpend) CommitSpend(_ context.Context, entry *dbmodels.SpendLogEntry) (litellmdb.SpendCommitResult, error) {
	f.calls = append(f.calls, entry)
	return litellmdb.SpendCommitResult{}, f.err
}

type fakeSearcher struct {
	status int
	body   []byte
	err    error
}

func (f fakeSearcher) Search(_ context.Context, _ []byte) (int, []byte, error) {
	return f.status, f.body, f.err
}

type fakeQuota struct {
	allowed    bool
	retryAfter time.Duration
	err        error
}

func (f fakeQuota) Allow(context.Context) (bool, time.Duration, error) {
	return f.allowed, f.retryAfter, f.err
}

func allowingQuota() fakeQuota { return fakeQuota{allowed: true} }

type fakePricing struct {
	usd    float64
	period string
}

func (f fakePricing) CostForTime(time.Time) (float64, string) { return f.usd, f.period }

func dayPricing(usd float64) fakePricing { return fakePricing{usd: usd, period: "day"} }

func baseHandlerConfig(auth Authenticator, searcher Searcher, spend SpendRecorder) HandlerConfig {
	return HandlerConfig{
		Client:         searcher,
		Auth:           auth,
		Spend:          spend,
		Quota:          allowingQuota(),
		Pricing:        dayPricing(0.005),
		Budget:         nil, // nil Reserver is nil-safe: always allow, no-op settle
		RPM:            60,
		MaxBodyBytes:   1 << 20,
		RequestTimeout: time.Second,
	}
}

func newTestHandler(auth Authenticator, searcher Searcher, spend SpendRecorder) *Handler {
	return NewHandler(baseHandlerConfig(auth, searcher, spend))
}

func newRequest(body string) *http.Request {
	return httptest.NewRequest(http.MethodPost, "/v1/search/yandex", bytes.NewReader([]byte(body)))
}

func TestHandlerDeniesKeyWithoutScope(t *testing.T) {
	auth := fakeAuth{
		tokenInfo: &dbmodels.TokenInfo{Token: "k1"},
		scopeCtx:  scope.NewContext(nil, nil), // no air_scopes set -> default-deny
		ok:        true,
	}
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusForbidden, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerStopsOnAuthFailure(t *testing.T) {
	auth := fakeAuth{ok: false}
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusUnauthorized, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerRejectsInvalidBodyBeforeAnyOtherCheck(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(`{"query":{"queryText":"hi"},"unknownField":true}`))

	assert.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerRejectsSmartSnippets(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)

	body := `{"query":{"queryText":"hi"},"metadata":{"fields":{"x-genesis-info-context":"on"}}}`
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(body))

	assert.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerQuotaExceededReturns429WithRetryAfter(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.Quota = fakeQuota{allowed: false, retryAfter: 3 * time.Second}
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusTooManyRequests, resp.Code)
	assert.NotEmpty(t, resp.Header().Get("Retry-After"))
	assert.Empty(t, spend.calls)
}

func TestHandlerQuotaUnavailableReturns503(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.Quota = fakeQuota{err: errors.New("redis down")}
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusServiceUnavailable, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerEnforcesPerKeyRPM(t *testing.T) {
	auth := allowedAuth("rpm-key")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.RPM = 1
	handler := NewHandler(cfg)

	for i := 0; i < 2; i++ {
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, newRequest(validSearchBody))
		if i == 0 {
			require.Equal(t, http.StatusOK, resp.Code)
		} else {
			require.Equal(t, http.StatusTooManyRequests, resp.Code)
			require.NotEmpty(t, resp.Header().Get("Retry-After"))
		}
	}
	assert.Len(t, spend.calls, 1, "only the allowed request should be billed")
}

func TestHandlerRejectsOversizedBody(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.MaxBodyBytes = 4
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(`{"query":{"queryText":"too long"}}`))

	assert.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Empty(t, spend.calls)
}

func TestHandlerBillsOnlyOn2xx(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{status: http.StatusBadRequest, body: []byte(`{"error":"bad query"}`)}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Equal(t, `{"error":"bad query"}`, resp.Body.String())
	assert.Empty(t, spend.calls, "a non-2xx upstream response must not be billed")
}

func TestHandlerCommitsSpendOnSuccessWithTokenIdentityAndPricing(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{"rawData":"abc"}`)}, spend)
	cfg.Pricing = fakePricing{usd: 0.0074, period: "day"}
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	require.Equal(t, http.StatusOK, resp.Code)
	require.Len(t, spend.calls, 1)
	entry := spend.calls[0]
	assert.Equal(t, "search/yandex", entry.CallType)
	assert.Equal(t, "yandex_search", entry.CustomLLMProvider)
	assert.Equal(t, 0.0074, entry.Spend)
	assert.Equal(t, "k1", entry.APIKey)
	assert.Equal(t, "user-1", entry.UserID)
	assert.Equal(t, "team-1", entry.TeamID)
	assert.Equal(t, "org-1", entry.OrganizationID)
	assert.Equal(t, "success", entry.Status)
	assert.Contains(t, entry.Metadata, `"pricing_period":"day"`)
	assert.Contains(t, entry.Metadata, `"upstream_status":200`)
	assert.Contains(t, entry.Metadata, `"operation":"web_sync"`, "the operation type is fixed by this adapter, never client-supplied")
}

func TestHandlerNightPricingReflectedInSpend(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.Pricing = fakePricing{usd: 0.0056, period: "night"}
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	require.Equal(t, http.StatusOK, resp.Code)
	require.Len(t, spend.calls, 1)
	assert.Equal(t, 0.0056, spend.calls[0].Spend)
	assert.Contains(t, spend.calls[0].Metadata, `"pricing_period":"night"`)
}

func TestHandlerMapsUpstreamErrors(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
	}{
		{"invalid body from client.Search", ErrInvalidBody, http.StatusBadRequest},
		{"timeout", context.DeadlineExceeded, http.StatusGatewayTimeout},
		{"generic failure", errors.New("boom"), http.StatusBadGateway},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			auth := allowedAuth("k1")
			spend := &fakeSpend{}
			handler := newTestHandler(auth, fakeSearcher{err: tc.err}, spend)

			resp := httptest.NewRecorder()
			handler.ServeHTTP(resp, newRequest(validSearchBody))

			assert.Equal(t, tc.wantStatus, resp.Code)
		})
	}
}

func TestHandlerTimeoutRecordsIndeterminateSpendRow(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{err: context.DeadlineExceeded}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusGatewayTimeout, resp.Code)
	require.Len(t, spend.calls, 1, "a timeout must still leave an audit row for reconciliation")
	entry := spend.calls[0]
	assert.Equal(t, 0.0, entry.Spend, "the client must not be billed for an indeterminate outcome")
	assert.Equal(t, "failure", entry.Status)
	assert.Contains(t, entry.Metadata, `"outcome":"indeterminate_timeout"`)
}

func TestHandlerNonTimeoutErrorDoesNotRecordIndeterminateRow(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	handler := newTestHandler(auth, fakeSearcher{err: errors.New("connection reset")}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusBadGateway, resp.Code)
	assert.Empty(t, spend.calls, "a plain network error (not a timeout) has no indeterminate-billing concern")
}

// fakeKafka records the analytics events the handler publishes, mirroring
// kafkalog.Manager.LogSpend's error contract.
type fakeKafka struct {
	events []*kafkalog.SpendEvent
	err    error
}

func (f *fakeKafka) LogSpend(event *kafkalog.SpendEvent) error {
	f.events = append(f.events, event)
	return f.err
}

// fakeIdentity is the RouterIdentity double.
type fakeIdentity struct {
	routerID string
	version  string
	commit   string
}

func (f fakeIdentity) GetRouterID() string { return f.routerID }
func (f fakeIdentity) GetVersion() string  { return f.version }
func (f fakeIdentity) GetCommit() string   { return f.commit }

func TestHandlerPublishesKafkaSpendEventOnSuccess(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	kafka := &fakeKafka{}
	cfg := baseHandlerConfig(auth, fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.Pricing = fakePricing{usd: 0.0074, period: "day"}
	cfg.Kafka = kafka
	cfg.Identity = fakeIdentity{routerID: "av-api-1", version: "1.2.3", commit: "abc123"}
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	require.Equal(t, http.StatusOK, resp.Code)
	require.Len(t, kafka.events, 1, "the analytics projection must be published alongside the Postgres row")
	event := kafka.events[0]
	assert.Equal(t, "search/yandex", event.CallType)
	assert.Equal(t, "/v1/search/yandex", event.APIBase)
	assert.Equal(t, "success", event.Status)
	assert.Equal(t, http.StatusOK, event.HTTPStatus)
	assert.Equal(t, 0.0074, event.TotalCost)
	assert.Equal(t, "k1", event.APIKeyHash)
	assert.Equal(t, "user-1", event.UserID)
	assert.Equal(t, "team-1", event.TeamID)
	assert.Equal(t, "org-1", event.OrganizationID)
	assert.Equal(t, spendModel, event.Model, "a non-empty model keeps the row attributable in ClickHouse analytics")
	assert.NotEmpty(t, event.RequestID, "the event must carry a request id to key the ClickHouse row")
	assert.Equal(t, "av-api-1", event.ServerRouterID)
	assert.Equal(t, "1.2.3", event.ServerVersion)
	assert.Equal(t, "abc123", event.ServerCommit)
}

func TestHandlerPublishesKafkaEventForIndeterminateTimeout(t *testing.T) {
	auth := allowedAuth("k1")
	spend := &fakeSpend{}
	kafka := &fakeKafka{}
	cfg := baseHandlerConfig(auth, fakeSearcher{err: context.DeadlineExceeded}, spend)
	cfg.Kafka = kafka
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusGatewayTimeout, resp.Code)
	require.Len(t, kafka.events, 1, "a timeout's $0 audit row must reach ClickHouse too")
	assert.Equal(t, "failure", kafka.events[0].Status)
	assert.Equal(t, 0.0, kafka.events[0].TotalCost)
	assert.Equal(t, http.StatusGatewayTimeout, kafka.events[0].HTTPStatus)
}

func TestHandlerNilKafkaPublisherIsNoop(t *testing.T) {
	spend := &fakeSpend{}
	handler := newTestHandler(allowedAuth("k1"), fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusOK, resp.Code, "an unwired Kafka (NoopManager) must not affect the request")
	require.Len(t, spend.calls, 1)
}

func TestHandlerKafkaPublishFailureDoesNotAffectBillingOrResponse(t *testing.T) {
	spend := &fakeSpend{}
	kafka := &fakeKafka{err: kafkalog.ErrQueueFull}
	cfg := baseHandlerConfig(allowedAuth("k1"), fakeSearcher{status: http.StatusOK, body: []byte(`{}`)}, spend)
	cfg.Kafka = kafka
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusOK, resp.Code, "a Kafka queue-full must never fail an already-billed request")
	require.Len(t, spend.calls, 1, "Postgres billing is independent of the analytics write-path")
}

func TestHandlerDoesNotPublishKafkaEventForNonBilledStatus(t *testing.T) {
	spend := &fakeSpend{}
	kafka := &fakeKafka{}
	cfg := baseHandlerConfig(allowedAuth("k1"), fakeSearcher{status: http.StatusBadGateway, body: []byte(`{}`)}, spend)
	cfg.Kafka = kafka
	handler := NewHandler(cfg)

	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, newRequest(validSearchBody))

	assert.Equal(t, http.StatusBadGateway, resp.Code)
	assert.Empty(t, spend.calls, "a non-2xx is not billed")
	assert.Empty(t, kafka.events, "and has no analytics spend event either")
}

// TestHandlerMintsDistinctAirEventIDPerCall covers two calls that carry the
// same request id, as they do when a trusted inbound traceparent is reused:
// each Kafka event must still get its own air_event_id, for both the billed
// and the indeterminate-timeout path, while the Postgres row keeps the
// request id as its AirEventID (its replay handling is out of this change).
func TestHandlerMintsDistinctAirEventIDPerCall(t *testing.T) {
	for name, searcher := range map[string]fakeSearcher{
		"success":               {status: http.StatusOK, body: []byte(`{}`)},
		"indeterminate timeout": {err: context.DeadlineExceeded},
	} {
		t.Run(name, func(t *testing.T) {
			spend := &fakeSpend{}
			kafka := &fakeKafka{}
			cfg := baseHandlerConfig(allowedAuth("k1"), searcher, spend)
			cfg.Kafka = kafka
			handler := NewHandler(cfg)

			const sharedRequestID = "4bf92f3577b34da6a3ce929d0e0e4736"
			for range 2 {
				req := newRequest(validSearchBody)
				handler.ServeHTTP(httptest.NewRecorder(), req.WithContext(requestid.WithID(req.Context(), sharedRequestID)))
			}

			require.Len(t, spend.calls, 2)
			require.Len(t, kafka.events, 2)
			for i, event := range kafka.events {
				assert.Equal(t, sharedRequestID, event.RequestID)
				assert.NotEmpty(t, event.AirEventID)
				assert.NotEqual(t, sharedRequestID, event.AirEventID, "air_event_id must not come from the client-influenced request id")
				assert.Equal(t, sharedRequestID, spend.calls[i].AirEventID, "the Postgres row must keep its AirEventID")
			}
			assert.NotEqual(t, kafka.events[0].AirEventID, kafka.events[1].AirEventID)
		})
	}
}
