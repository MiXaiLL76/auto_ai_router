package yandexsearch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/kafkalog"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/budget"
	dbmodels "github.com/mixaill76/auto_ai_router/internal/litellmdb/models"
	"github.com/mixaill76/auto_ai_router/internal/ratelimit"
	"github.com/mixaill76/auto_ai_router/internal/requestid"
	"github.com/mixaill76/auto_ai_router/internal/scope"
)

// accessScope gates the endpoint per client key/team/organization: only a
// token whose own, team's, or organization's LiteLLM metadata.air_scopes
// includes "yandex_search" (or "*") may call it (see
// internal/proxy/scope.go's scopeContextFromTokenInfo). A token with no
// air_scopes set anywhere is denied by default — enabling yandex_search in
// config.yaml alone never exposes the endpoint to every client; an admin
// must additionally opt a key/team/org in.
const accessScope = "yandex_search"

const callType = "search/yandex"
const customLLMProvider = "yandex_search"
const apiBase = "/v1/search/yandex"

// spendModel is the `model` dimension this adapter puts on its spend rows.
// The operation has no LLM model, but leaving the column empty makes the rows
// hard to use in analytics: air.logs (unlike LiteLLM_SpendLogs) has no
// separate provider column, so an empty `model` leaves a search row
// identifiable only by its call_type. A stable synthetic name keeps the rows
// attributable and filterable (it is not a vendor id, so the admin panel's
// provider breakdown does not fold it into a real LLM provider).
const spendModel = "yandex-search"

// operation identifies this adapter's own Yandex Search API operation type
// for billing/audit (config.OperationWebSync's value) — a fixed constant,
// never derived from anything in the client's request: this handler is the
// web_sync adapter and nothing else, so there is nothing to "select".
const operation = "web_sync"

// perKeyRPMRetryAfterSeconds is a fixed Retry-After for the per-key RPM
// limiter's 429s. internal/ratelimit.RPMLimiter's API doesn't expose
// time-to-next-free-slot, unlike FolderQuota's precise sliding-window
// calculation, so this is a simple, conservative constant rather than an
// exact figure.
const perKeyRPMRetryAfterSeconds = 2

// Authenticator resolves the calling client's identity and scope from the
// inbound request, writing the error response itself on failure.
// *proxy.Proxy satisfies this structurally — callers pass it directly, no
// adapter needed.
type Authenticator interface {
	AuthenticateClientRequestScoped(w http.ResponseWriter, r *http.Request) (*dbmodels.TokenInfo, scope.Context, bool)
}

// SpendRecorder commits one billing event. litellmdb's SpendCommitter
// satisfies this structurally.
type SpendRecorder interface {
	CommitSpend(ctx context.Context, entry *dbmodels.SpendLogEntry) (litellmdb.SpendCommitResult, error)
}

// SpendEventPublisher publishes the expanded analytics copy of a spend entry
// to Kafka (internal/kafkalog), the write-path ClickHouse's air.logs reads.
// Independent of SpendRecorder: Postgres is the billing source of truth, Kafka
// is the analytics projection, and a Kafka outage must never affect billing or
// the request itself (kafkalog.Manager.LogSpend only reports queue-full, the
// same best-effort contract the LLM path uses in proxy_log.go). nil is
// accepted and makes publishing a no-op, so a deployment with Kafka disabled
// runs exactly as before.
type SpendEventPublisher interface {
	LogSpend(event *kafkalog.SpendEvent) error
}

// RouterIdentity supplies the server identity fields kafkalog.SpendEvent
// carries for the same reason the LLM path does: ClickHouse's air.logs keys a
// row by (request_id, server_router_id), and the admin panel's per-router
// breakdowns group by it. *proxy.Proxy satisfies this structurally.
type RouterIdentity interface {
	GetRouterID() string
	GetVersion() string
	GetCommit() string
}

// Searcher performs the upstream Yandex call. *Client satisfies this.
type Searcher interface {
	Search(ctx context.Context, body []byte) (status int, respBody []byte, err error)
}

// QuotaChecker admits or rejects one request against a shared quota.
// *FolderQuota satisfies this structurally.
type QuotaChecker interface {
	Allow(ctx context.Context) (allowed bool, retryAfter time.Duration, err error)
}

// PricingTable selects the marked-up USD price for a query sent at t, and
// which rate period applied. config.YandexSearchPricing satisfies this
// structurally — cmd/server passes cfg.YandexSearch.Pricing directly, no
// adapter needed (same pattern as Authenticator/SpendRecorder).
type PricingTable interface {
	CostForTime(t time.Time) (usd float64, period string)
}

// HandlerConfig configures a Handler.
type HandlerConfig struct {
	Client  Searcher
	Auth    Authenticator
	Spend   SpendRecorder
	Quota   QuotaChecker
	Pricing PricingTable
	// Kafka publishes the expanded analytics copy of each spend entry to the
	// air.spend_logs topic ClickHouse ingests; nil disables it (no-op). Unlike
	// Postgres billing it is best-effort by construction — a publish failure is
	// logged and swallowed, never failing the request or the human-facing spend
	// row.
	Kafka SpendEventPublisher
	// Identity fills ServerRouterID/ServerVersion/ServerCommit on the Kafka
	// event; nil leaves them empty.
	Identity RouterIdentity
	// Budget is nil-safe (internal/litellmdb/budget.Reserver's own
	// nil-receiver no-op) — pass nil when budget reservation is disabled
	// system-wide or Redis is unavailable; the handler degrades to
	// always-allow/no-op settle with no special-casing.
	Budget         *budget.Reserver
	RPM            int
	MaxBodyBytes   int64
	RequestTimeout time.Duration
	Logger         *slog.Logger
}

// Handler serves POST /v1/search/yandex.
type Handler struct {
	client         Searcher
	auth           Authenticator
	spend          SpendRecorder
	kafka          SpendEventPublisher
	identity       RouterIdentity
	quota          QuotaChecker
	pricing        PricingTable
	budget         *budget.Reserver
	rpm            int
	maxBodyBytes   int64
	requestTimeout time.Duration
	logger         *slog.Logger

	// limiter is a dedicated, in-process, per-key RPM bucket, independent
	// of the model-RPM machinery (which has no concept of a non-model
	// feature) AND independent of quota (the shared, Redis-backed,
	// cross-replica Yandex folder quota) — this one is per-client fairness
	// only.
	limiter *ratelimit.RPMLimiter
}

// NewHandler builds a Handler. Panics if any required dependency is nil —
// this is a startup-time wiring error, not a runtime condition.
func NewHandler(cfg HandlerConfig) *Handler {
	if cfg.Client == nil || cfg.Auth == nil || cfg.Spend == nil || cfg.Quota == nil || cfg.Pricing == nil {
		panic("yandexsearch: Client, Auth, Spend, Quota and Pricing are required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Handler{
		client:         cfg.Client,
		auth:           cfg.Auth,
		spend:          cfg.Spend,
		kafka:          cfg.Kafka,
		identity:       cfg.Identity,
		quota:          cfg.Quota,
		pricing:        cfg.Pricing,
		budget:         cfg.Budget,
		rpm:            cfg.RPM,
		maxBodyBytes:   cfg.MaxBodyBytes,
		requestTimeout: cfg.RequestTimeout,
		logger:         logger,
		limiter:        ratelimit.New(),
	}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	tokenInfo, scopeCtx, ok := h.auth.AuthenticateClientRequestScoped(w, r)
	if !ok {
		return // the authenticator already wrote the error response (401)
	}

	if !scopeCtx.Allows([]string{accessScope}, nil) {
		writeError(w, http.StatusForbidden, "Yandex search is not enabled for this key")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "request body too large or unreadable")
		return
	}

	if err := validateSearchRequest(body); err != nil {
		h.writeValidationError(w, err)
		return
	}

	// The shared, cross-replica Yandex quota (1 RPS + hourly, scoped to the
	// whole folder_id) is checked before the per-key limiter: it's the
	// scarcer resource and protecting it matters more than per-key
	// fairness, which only matters once the shared budget has room at all.
	admitted, retryAfter, err := h.quota.Allow(r.Context())
	if err != nil {
		h.logger.WarnContext(r.Context(), "Yandex search: shared folder quota unavailable", "error", err)
		writeError(w, http.StatusServiceUnavailable, "Yandex search is temporarily unavailable")
		return
	}
	if !admitted {
		w.Header().Set("Retry-After", strconv.FormatInt(int64(retryAfter.Seconds()+1), 10))
		writeError(w, http.StatusTooManyRequests, "Yandex search quota exceeded")
		return
	}

	// Seeding on every request is cheap (a lock-protected map write) and
	// keeps this correct if the configured RPM ever changes without a
	// process restart; TryAllowAllCtx below does the actual accounting.
	h.limiter.AddCredential(tokenInfo.Token, h.rpm)
	if !h.limiter.TryAllowAllCtx(r.Context(), tokenInfo.Token, "") {
		w.Header().Set("Retry-After", strconv.Itoa(perKeyRPMRetryAfterSeconds))
		writeError(w, http.StatusTooManyRequests, "Yandex search rate limit exceeded for this key")
		return
	}

	sendTime := time.Now()
	estimateUSD, period := h.pricing.CostForTime(sendTime)

	res, err := budgetReserve(r.Context(), h.budget, tokenInfo, estimateUSD, h.logger)
	if err != nil {
		writeError(w, http.StatusPaymentRequired, budgetExceededMessage(err))
		return
	}
	// Safety net: whatever happens below (early return, panic recovered
	// upstream in the router), this settles the reservation to "not
	// billed" unless a code path below explicitly settles it first.
	// reservation.settle is idempotent, so an explicit settle elsewhere in
	// this function makes this a no-op.
	defer res.settle(context.WithoutCancel(r.Context()), h.budget, 0, h.logger)

	ctx, cancel := context.WithTimeout(r.Context(), h.requestTimeout)
	defer cancel()

	status, respBody, err := h.client.Search(ctx, body)
	duration := time.Since(start)
	if err != nil {
		h.handleUpstreamError(w, r, tokenInfo, res, err, start, sendTime, duration)
		return
	}

	h.logger.InfoContext(r.Context(), "Yandex search request completed",
		"status", status, "duration_ms", duration.Milliseconds(), "pricing_period", period)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(respBody)

	if status < 200 || status >= 300 {
		// Local failures, Yandex auth errors, and Yandex's own internal
		// errors are not billed — the deferred settle(0) above handles it.
		return
	}
	res.settle(r.Context(), h.budget, estimateUSD, h.logger)
	h.commitSpend(r.Context(), tokenInfo, start, duration, estimateUSD, period, status, "success")
}

// handleUpstreamError classifies a failed Search call and, for a timeout
// specifically, writes an indeterminate-outcome spend row: Yandex may have
// processed (and will bill us for) a request whose response we never saw,
// so the reservation is still released (never billed to the client) but a
// $0 row is kept for manual reconciliation against Yandex's own invoice.
func (h *Handler) handleUpstreamError(w http.ResponseWriter, r *http.Request, tokenInfo *dbmodels.TokenInfo, _ *reservation, err error, start, sendTime time.Time, duration time.Duration) {
	h.logger.WarnContext(r.Context(), "Yandex search request failed",
		"error", err, "duration_ms", duration.Milliseconds())

	switch {
	case errors.Is(err, ErrInvalidBody):
		writeError(w, http.StatusBadRequest, "request body must be a JSON object")
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusGatewayTimeout, "Yandex search request timed out")
		h.commitIndeterminateSpend(r.Context(), tokenInfo, start, sendTime, duration)
	default:
		writeError(w, http.StatusBadGateway, "Yandex search request failed")
	}
	// res.settle(0) runs via the caller's defer either way — this request
	// is never billed, timeout included.
}

func (h *Handler) commitSpend(ctx context.Context, tokenInfo *dbmodels.TokenInfo, start time.Time, duration time.Duration, spendUSD float64, period string, upstreamStatus int, status string) {
	requestID := requestIDFor(ctx)
	metadata, err := json.Marshal(map[string]any{
		"air_event_id":    requestID,
		"operation":       operation,
		"pricing_period":  period,
		"upstream_status": upstreamStatus,
		"request_count":   1,
	})
	if err != nil {
		h.logger.ErrorContext(ctx, "Yandex search: failed to encode spend metadata", "error", err)
		metadata = []byte("{}")
	}
	endTime := start.Add(duration)
	entry := &dbmodels.SpendLogEntry{
		RequestID:         requestID,
		AirEventID:        requestID,
		StartTime:         start,
		EndTime:           endTime,
		RequestDurationMS: int(duration.Milliseconds()),
		CallType:          callType,
		APIBase:           apiBase,
		Model:             spendModel,
		CustomLLMProvider: customLLMProvider,
		Metadata:          string(metadata),
		Spend:             spendUSD,
		APIKey:            tokenInfo.Token,
		UserID:            tokenInfo.UserID,
		TeamID:            tokenInfo.TeamID,
		BillingTeamID:     tokenInfo.TeamID,
		OrganizationID:    tokenInfo.OrganizationID,
		Status:            status,
	}
	if _, err := h.spend.CommitSpend(ctx, entry); err != nil {
		h.logger.ErrorContext(ctx, "Yandex search: failed to commit spend", "error", err)
	}
	h.publishSpendEvent(entry, upstreamStatus)
}

// publishSpendEvent mirrors the LLM path's dual write (proxy_log.go): the
// Postgres row above is the billing source of truth, while this Kafka event is
// the analytics projection ClickHouse's air.logs ingests. Best-effort by
// construction — a publish failure is logged and swallowed, exactly like
// proxy.logSpendToKafka, so neither billing nor the already-sent client
// response is affected. No-op when Kafka is unwired (h.kafka == nil).
func (h *Handler) publishSpendEvent(entry *dbmodels.SpendLogEntry, upstreamStatus int) {
	if h.kafka == nil {
		return
	}
	event := &kafkalog.SpendEvent{
		RequestID:      entry.RequestID,
		StartTime:      entry.StartTime,
		EndTime:        entry.EndTime,
		DurationMs:     int64(entry.RequestDurationMS),
		CallType:       entry.CallType,
		APIBase:        entry.APIBase,
		Status:         entry.Status,
		HTTPStatus:     upstreamStatus,
		Model:          entry.Model,
		ModelGroup:     entry.ModelGroup,
		APIKeyHash:     entry.APIKey,
		UserID:         entry.UserID,
		TeamID:         entry.TeamID,
		OrganizationID: entry.OrganizationID,
		TotalCost:      entry.Spend,
	}
	if h.identity != nil {
		event.ServerRouterID = h.identity.GetRouterID()
		event.ServerVersion = h.identity.GetVersion()
		event.ServerCommit = h.identity.GetCommit()
	}
	if err := h.kafka.LogSpend(event); err != nil {
		h.logger.Warn("Yandex search: failed to queue Kafka spend event",
			"error", err, "request_id", entry.RequestID)
	}
}

// commitIndeterminateSpend writes a $0 audit row for a request whose
// outcome we don't know (upstream call timed out): Yandex may still charge
// us even though the client isn't billed and never saw a result. Not a
// hard requirement for the handler's own correctness — purely a
// reconciliation aid — so a failure to write it is logged and otherwise
// ignored.
func (h *Handler) commitIndeterminateSpend(ctx context.Context, tokenInfo *dbmodels.TokenInfo, start, sendTime time.Time, duration time.Duration) {
	requestID := requestIDFor(ctx)
	metadata, err := json.Marshal(map[string]any{
		"air_event_id": requestID,
		"operation":    operation,
		"outcome":      "indeterminate_timeout",
		"sent_at":      sendTime.UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		metadata = []byte(`{"outcome":"indeterminate_timeout"}`)
	}
	entry := &dbmodels.SpendLogEntry{
		RequestID:         requestID,
		AirEventID:        requestID,
		StartTime:         start,
		EndTime:           start.Add(duration),
		RequestDurationMS: int(duration.Milliseconds()),
		CallType:          callType,
		APIBase:           apiBase,
		Model:             spendModel,
		CustomLLMProvider: customLLMProvider,
		Metadata:          string(metadata),
		Spend:             0,
		APIKey:            tokenInfo.Token,
		UserID:            tokenInfo.UserID,
		TeamID:            tokenInfo.TeamID,
		BillingTeamID:     tokenInfo.TeamID,
		OrganizationID:    tokenInfo.OrganizationID,
		Status:            "failure",
	}
	if _, err := h.spend.CommitSpend(ctx, entry); err != nil {
		h.logger.ErrorContext(ctx, "Yandex search: failed to record indeterminate-timeout spend row", "error", err)
	}
	h.publishSpendEvent(entry, http.StatusGatewayTimeout)
}

func requestIDFor(ctx context.Context) string {
	if id := requestid.FromContext(ctx); id != "" {
		return id
	}
	return requestid.New()
}

func (h *Handler) writeValidationError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrQueryTooLong):
		writeError(w, http.StatusBadRequest, "query.queryText exceeds the 400-character/40-word limit")
	case errors.Is(err, ErrSmartSnippetsNotSupported):
		writeError(w, http.StatusBadRequest, "smart snippets are not supported on this endpoint")
	case errors.Is(err, ErrInvalidBody):
		writeError(w, http.StatusBadRequest, "request body must be a JSON object matching WebSearch.Search's schema")
	default:
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request: %v", err))
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"message": message,
		},
	})
}
