// Package healthclient lets the router report upstream outcomes to the
// health-check service (internal/healthservice deployment) and keeps a
// locally cached view of which accounts the service considers dead.
//
// The service is the shared source of truth (state lives in Redis), so every
// router replica sees the same alive/dead picture. This client is the local
// mirror:
//
//   Report(...)    fire-and-forget queue of outcome observations; never blocks
//                  the request path, drops (and counts) when the queue is full.
//   Start(ctx)     background loop: periodically pulls the full status list
//                  (GET /v1/creds) into the cache, and drains the report
//                  queue (POST /v1/report) as reports arrive.
//   IsDead(name)   consults the cache. Fail-open: unknown or stale entries
//                  are not dead, so a downed health-check service never
//                  hard-blocks traffic.
package healthclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ReportsTotal counts outcome reports actually POSTed to the service.
	ReportsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_health_reports_total",
			Help: "Outcome reports sent to the health-check service",
		},
	)

	// ReportsDropped counts reports dropped because the local queue was full.
	ReportsDropped = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_health_reports_dropped_total",
			Help: "Outcome reports dropped (queue full); the service stays the source of truth via probes",
		},
	)

	// PullsFailed counts failed status-list synchronizations.
	PullsFailed = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "auto_ai_router_health_pulls_failed_total",
			Help: "Failed status list synchronizations from the health-check service",
		},
	)
)

// NoopChecker reports every account alive; used as the default checker so
// the balancer/proxy never deal with nil when the feature is disabled.
type NoopChecker struct {
}

func NewNoopChecker() *NoopChecker {
	return &NoopChecker{}
}

// IsDead (HealthChecker): feature disabled — nothing is dead.
func (_ *NoopChecker) IsDead(_ string) bool {
	return false
}

// HealthChecker is the interface consumed by the balancer and the proxy:
// a cached answer to "is this account dead?". Unknown accounts are not dead
// (fail-open), matching the fail2ban convention.
type HealthChecker interface {
	IsDead(name string) bool
}

// BanInfo is one active ban mirrored from the health-check service. Model
// "*" means the whole account. Until zero means permanent.
type BanInfo struct {
	Credential string
	Model      string
	Until      time.Time
	StatusCode int
	Reason     string
}

// BanSynchronizer materializes the service's bans into the router's local
// fail2ban so the balancer keeps working unchanged (the router's own
// fail2ban becomes a read-only cache when its error_codes are disabled).
type BanSynchronizer interface {
	SyncBans(bans []BanInfo)
}

// reportItem is one queued outcome observation.
type reportItem struct {
	credential         string
	model              string
	statusCode         int64
	retryAfterSeconds  int64
}

// cachedEntry is one account's cached status.
type cachedEntry struct {
	alive     bool
	fetchedAt time.Time
}

// Client mirrors the health-check service state locally.
type Client struct {
	baseURL      string
	authToken    string
	http         *http.Client
	queue        chan reportItem
	cacheMu      sync.Mutex
	cache        map[string]cachedEntry
	enabled      bool
	applyBans    bool
	banSink      BanSynchronizer // optional fail2ban mirror
	syncInterval time.Duration
	cacheTTL     time.Duration
	logger       *slog.Logger
}

// New creates a disabled client. Call SetEnabled(true) before Start.
func New(cfg *config.HealthServiceConfig, logger *slog.Logger) *Client {
	base := cfg.URL
	if len(base) > 0 && base[len(base)-1] == '/' {
		base = base[:len(base)-1]
	}
	queueSize := cfg.ReportQueueSize
	if queueSize < 1 {
		queueSize = 1024
	}
	return &Client{
		baseURL:      base,
		authToken:    cfg.AuthToken,
		http:         &http.Client{Timeout: cfg.HTTPTimeout},
		queue:        make(chan reportItem, queueSize),
		cacheMu:      sync.Mutex{},
		cache:        map[string]cachedEntry{},
		enabled:      cfg.Enabled,
		applyBans:    cfg.ApplyBans,
		syncInterval: cfg.SyncInterval,
		cacheTTL:     cfg.CacheTTL,
		logger:       logger,
	}
}

// Report queues one observed upstream outcome. Never blocks: when the queue
// is full the report is dropped and counted (the service's own probes keep
// the state moving anyway). HTTP status 0 means "transport error, no status".
func (c *Client) Report(credential, model string, statusCode int, retryAfterSeconds int) {
	if c == nil || !c.enabled {
		return
	}
	item := reportItem{
		credential:        credential,
		model:             model,
		statusCode:        int64(statusCode),
		retryAfterSeconds: int64(retryAfterSeconds),
	}
	select {
	case c.queue <- item:
		// queued for the background flusher
	default:
		ReportsDropped.Inc()
	}
}

// IsDead returns true when the cached state says the account is dead.
// Unknown or stale entries are treated as alive (fail-open).
func (c *Client) IsDead(name string) bool {
	if c == nil || !c.enabled {
		return false
	}
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()

	entry, ok := c.cache[name]
	if !ok {
		return false
	}
	if time.Since(entry.fetchedAt) > c.cacheTTL {
		return false // stale snapshot: assume alive until the next pull
	}
	return !entry.alive
}

// SetBanSynchronizer installs the local fail2ban mirror. When set (and
// apply_bans enabled), every status pull also fetches /v1/bans and syncs
// them through the sink.
func (c *Client) SetBanSynchronizer(sink BanSynchronizer) {
	c.banSink = sink
}

// Start runs the background loop: a warm pull, then periodic pulls plus
// report flushing, until ctx is cancelled.
func (c *Client) Start(ctx context.Context) {
	if c == nil || !c.enabled {
		return
	}
	c.pull(ctx)
	ticker := time.NewTicker(c.syncInterval)
	defer ticker.Stop()

	c.logger.Info("Health-check service client started",
		"url", c.baseURL,
		"sync_interval", c.syncInterval,
		"cache_ttl", c.cacheTTL,
	)

	for {
		select {
		case <-ctx.Done():
			c.logger.Info("Health-check service client stopped")
			return
		case <-ticker.C:
			c.pull(ctx)
		case item := <-c.queue:
			c.postReport(ctx, item)
		}
	}
}

// statusView mirrors the service's per-account status JSON.
type statusView struct {
	Alive      bool   `json:"alive"`
	FailCount  int64  `json:"fail_count"`
	LastStatus int64  `json:"last_status,omitempty"`
	UpdatedAt  int64  `json:"updated_at"`
}

// credsResponse mirrors the service's GET /v1/creds JSON.
type credsResponse struct {
	Credentials map[string]statusView `json:"credentials"`
}

// reportBody is the POST /v1/report payload.
type reportBody struct {
	Credential        string `json:"credential"`
	Model             string `json:"model,omitempty"`
	StatusCode        int64  `json:"status_code"`
	RetryAfterSeconds int64  `json:"retry_after_seconds,omitempty"`
}

// pull synchronizes the local cache from GET /v1/creds. Failures keep the
// previous cache (entries age out via cacheTTL).
func (c *Client) pull(ctx context.Context) {
	url := c.baseURL + "/v1/creds"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: failed to build status request", "error", err)
		return
	}
	c.setAuth(req)

	resp, err := c.http.Do(req)
	if err != nil {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: status sync failed", "url", url, "error", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: status sync failed",
			"url", url,
			"status_code", resp.StatusCode,
		)
		return
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if readErr != nil {
		PullsFailed.Inc()
		return
	}
	var parsed credsResponse
	if json.Unmarshal(body, &parsed) != nil {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: status response unparseable", "url", url)
		return
	}

	now := time.Now().UTC()
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()
	snapshot := make(map[string]cachedEntry, len(parsed.Credentials))
	for name, st := range parsed.Credentials {
		snapshot[name] = cachedEntry{alive: st.Alive, fetchedAt: now}
	}
	c.cache = snapshot

	if c.applyBans && c.banSink != nil {
		c.syncBans(ctx)
	}
}

// bansResponse mirrors GET /v1/bans.
type bansResponse struct {
	Bans []struct {
		Credential string `json:"credential"`
		Model      string `json:"model"`
		Until      int64  `json:"until"`
		StatusCode int64  `json:"status_code,omitempty"`
		Reason     string `json:"reason,omitempty"`
	} `json:"bans"`
}

// syncBans mirrors the service's active bans into the local fail2ban: bans
// that appeared are applied, bans that disappeared are unbaned. The diff is
// computed against the last mirrored set the service itself reported.
func (c *Client) syncBans(ctx context.Context) {
	url := c.baseURL + "/v1/bans"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		PullsFailed.Inc()
		return
	}
	c.setAuth(req)

	resp, err := c.http.Do(req)
	if err != nil {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: bans sync failed", "url", url, "error", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		PullsFailed.Inc()
		c.logger.Warn("Health-check service: bans sync failed", "url", url, "status_code", resp.StatusCode)
		return
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if readErr != nil {
		PullsFailed.Inc()
		return
	}
	var parsed bansResponse
	if json.Unmarshal(body, &parsed) != nil {
		PullsFailed.Inc()
		return
	}

	bans := make([]BanInfo, 0, len(parsed.Bans))
	for _, b := range parsed.Bans {
		until := time.Time{}
		if b.Until > 0 {
			until = time.Unix(b.Until, 0).UTC()
		}
		bans = append(bans, BanInfo{
			Credential: b.Credential,
			Model:      b.Model,
			Until:      until,
			StatusCode: int(b.StatusCode),
			Reason:     b.Reason,
		})
	}
	// The sink (main's fail2ban adapter) diffs against its own active bans:
	// unknown -> BanUntil, missing -> Unban.
	c.banSink.SyncBans(bans)
}

// postReport POSTs one queued outcome to the service. Failures are logged
// and counted; the state remains consistent because the service's active
// probes and the other router replicas keep feeding it.
func (c *Client) postReport(ctx context.Context, item reportItem) {
	payload := reportBody{
		Credential:        item.credential,
		Model:             item.model,
		StatusCode:        item.statusCode,
		RetryAfterSeconds: item.retryAfterSeconds,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		ReportsDropped.Inc()
		return
	}

	url := c.baseURL + "/v1/report"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		ReportsDropped.Inc()
		return
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	resp, err := c.http.Do(req)
	if err != nil {
		c.logger.Warn("Health-check service: report failed", "credential", item.credential, "error", err)
		return
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		c.logger.Warn("Health-check service: report rejected",
			"credential", item.credential,
			"status_code", resp.StatusCode,
		)
		return
	}
	ReportsTotal.Inc()
}

// setAuth attaches the shared token when configured.
func (c *Client) setAuth(req *http.Request) {
	if c.authToken != "" {
		req.Header.Set("X-API-Key", c.authToken)
	}
}

// RetryAfterSeconds parses an upstream Retry-After header (integer seconds).
// Returns 0 for HTTP dates and garbage; the service falls back to its own
// cooldown rules then.
func RetryAfterSeconds(v string) int {
	if v == "" {
		return 0
	}
	secs, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	if secs < 0 {
		return 0
	}
	return secs
}