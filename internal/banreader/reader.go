// Package banreader mirrors the health worker's bans from Redis into the
// router's local fail2ban, so the balancer keeps working unchanged while the
// ban decisions live in the worker. It also feeds the account-level liveness
// check (whole-account wildcard bans) used by the balancer's IsDead path.
//
// Key layout (mirror of the healthcheck-service store):
//   {prefix}{provider}:bans       ZSET member=credential|model (* = whole
//                                 account), score=ban_until (unix secs).
//   {prefix}{provider}:ban:{key}  HASH details: until/code/reason/origin.
//   {prefix}providers             SET of provider namespaces.
package banreader

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/fail2ban"
	"github.com/valkey-io/valkey-go"
)

const wildcardModel = "*"

// Reader refreshes the local fail2ban from the worker's Redis bans.
type Reader struct {
	client       valkey.Client
	keyPrefix    string
	providersKey string
	f2b          *fail2ban.Fail2Ban
	localTypes   []string // credential types from config (fallback providers)
	interval     time.Duration
	logger       *slog.Logger

	mu      sync.RWMutex
	dead    map[string]int64  // credential -> wildcard ban until (0 = none)
	applied map[string]bool   // ban keys currently materialized in fail2ban
}

// New creates the reader over the shared valkey client. localTypes are the
// credential provider types known from the static config; the worker's
// providers set is merged over them. interval 0 uses 2s.
//
// The reader keys live under cfg.HealthKeyPrefix (default "hc:"), a
// namespace dedicated to the health worker — deliberately NOT cfg.KeyPrefix,
// which other subsystems (budget, auth, response store, rate limits, hybrid
// sync) share for their own keys. Likewise, the poll interval comes from the
// caller (account_events.reader_interval), not from redis.sync_interval,
// which the budget/rate-limit hybrid backends share.
func New(client valkey.Client, cfg config.RedisConfig, f2b *fail2ban.Fail2Ban, localTypes []string, interval time.Duration, logger *slog.Logger) *Reader {
	if interval <= 0 {
		interval = 2 * time.Second
	}
	healthPrefix := cfg.HealthKeyPrefix
	if healthPrefix == "" {
		healthPrefix = "hc:"
	}
	return &Reader{
		client:       client,
		keyPrefix:    healthPrefix,
		providersKey: healthPrefix + "providers",
		f2b:          f2b,
		localTypes:   localTypes,
		interval:     interval,
		logger:       logger,
		dead:         map[string]int64{},
		applied:      map[string]bool{},
	}
}

// Start runs the sync loop until ctx is cancelled.
func (r *Reader) Start(ctx context.Context) {
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	r.sync(ctx)
	r.logger.Info("Health worker ban reader started", "interval", r.interval)
	for {
		select {
		case <-ctx.Done():
			r.logger.Info("Health worker ban reader stopped")
			return
		case <-ticker.C:
			r.sync(ctx)
		}
	}
}

// IsDead reports whether the account is banned as a whole (wildcard ban
// active in the latest snapshot). Unknown accounts are alive (fail-open).
func (r *Reader) IsDead(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	until, ok := r.dead[name]
	return ok && until > time.Now().UTC().Unix()
}

type banEntry struct {
	cred   string
	model  string
	until  int64
	code   int64
	reason string
}

// providers returns the set of provider namespaces to scan: the router's
// own credential types unioned with the worker's providers registry.
func (r *Reader) providers(ctx context.Context) []string {
	set := map[string]bool{}
	for _, t := range r.localTypes {
		if t != "" {
			set[t] = true
		}
	}
	if members, err := r.client.Do(ctx,
		r.client.B().Smembers().Key(r.providersKey).Build()).AsStrSlice(); err == nil {
		for _, m := range members {
			set[m] = true
		}
	}
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// sync pulls the active bans from Redis, materializes them in fail2ban and
// updates the wildcard liveness map.
//
// The snapshot is atomic: any read failure aborts the whole sync BEFORE the
// diff step. A partial snapshot must never be diffed against the applied
// bans — records missing from it are indistinguishable from lifted bans, so
// a transient Redis blip would silently unban live bans and let traffic hit
// accounts the worker still considers dead. On failure the applied bans
// simply stay until their own timer expires or the next successful sync.
func (r *Reader) sync(ctx context.Context) {
	now := time.Now().UTC().Unix()

	var keyProv []string // detail keys in provider order for the pipelined read
	provOf := map[string]string{}
	want := map[string]banEntry{}
	for _, provider := range r.providers(ctx) {
		entries, err := r.client.Do(ctx,
			r.client.B().Zrangebyscore().
				Key(r.keyPrefix + provider + ":bans").
				Min(fmt.Sprintf("%d", now)).
				Max("+inf").
				Withscores().
				Build(),
		).AsZScores()
		if err != nil {
			r.logger.Warn("Ban reader: failed to scan bans, aborting sync (applied bans stay intact)",
				"provider", provider, "error", err)
			return
		}
		for _, ze := range entries {
			cred, model := splitBanKey(ze.Member)
			key := banKey(cred, model)
			want[key] = banEntry{
				cred:   cred,
				model:  model,
				until:  int64(ze.Score),
				code:   429,
				reason: "banned by health worker",
			}
			keyProv = append(keyProv, key)
			provOf[key] = provider
		}
	}

	r.fillDetails(ctx, keyProv, provOf, want)

	r.mu.Lock()
	defer r.mu.Unlock()

	// Apply bans the local cache does not have yet (never shortens).
	for key, e := range want {
		if !r.applied[key] {
			r.f2b.BanUntil(e.cred, e.model, int(e.code), time.Unix(e.until, 0).UTC(), e.reason)
		}
	}
	// Lift bans that disappeared from the worker's snapshot.
	for key := range r.applied {
		if _, keep := want[key]; !keep {
			cred, model := splitBanKey(key)
			r.f2b.Unban(cred, model)
		}
	}
	r.applied = make(map[string]bool, len(want))
	for key := range want {
		r.applied[key] = true
	}

	// Wildcard liveness map for the balancer's account-level check.
	dead := map[string]int64{}
	for _, e := range want {
		if e.model == wildcardModel {
			dead[e.cred] = e.until
		}
	}
	r.dead = dead
}

// fillDetails reads the ban detail hashes in one pipelined round trip,
// using the provider namespace each key was scanned from.
func (r *Reader) fillDetails(ctx context.Context, keys []string, provOf map[string]string, want map[string]banEntry) {
	if len(keys) == 0 {
		return
	}
	cmds := make([]valkey.Completed, 0, len(keys))
	for _, key := range keys {
		p := provOf[key]
		if p == "" {
			p = "unknown"
		}
		cmds = append(cmds, r.client.B().Hgetall().Key(r.keyPrefix+p+":ban:"+key).Build())
	}
	results := r.client.DoMulti(ctx, cmds...)
	for i, key := range keys {
		fields, err := results[i].AsMap()
		if err != nil {
			r.logger.Debug("Ban details read failed, using defaults", "ban", key, "error", err)
			continue
		}
		e := want[key]
		if v, ok := fields["code"]; ok {
			e.code = atoiSafe(stringFromMsg(v))
		}
		if v, ok := fields["reason"]; ok {
			e.reason = stringFromMsg(v)
		}
		want[key] = e
	}
}

func banKey(credential, model string) string {
	return credential + "|" + model
}

func splitBanKey(key string) (credential, model string) {
	idx := strings.IndexByte(key, '|')
	if idx < 0 {
		return key, ""
	}
	return key[:idx], key[idx+1:]
}

func atoiSafe(s string) int64 {
	var v int64
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0
		}
		v = v*10 + int64(c-'0')
	}
	return v
}

func stringFromMsg(m valkey.ValkeyMessage) string {
	bs, _ := m.AsBytes()
	return string(bs)
}
