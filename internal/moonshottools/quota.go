package moonshottools

import (
	"context"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/toolquota"
	"github.com/valkey-io/valkey-go"
)

// Moonshot's account-level QPS limits for the three tools: the limits are
// per account (not per key), and the pools are separate -- traffic to one
// tool does not consume another's quota.
const (
	// DefaultSearchQPS is the account-level Basic search limit.
	DefaultSearchQPS = 100
	// DefaultSearchProQPS is the account-level Pro search limit.
	DefaultSearchProQPS = 100
	// DefaultFetchQPS is the account-level Fetch limit.
	DefaultFetchQPS = 25
)

// The names of the three pools, in the Redis keys and in the quota's own
// lookup.
const (
	poolSearch    = "search"
	poolSearchPro = "search_pro"
	poolFetch     = "fetch"
)

// Quota admits Search, Search Pro and Fetch calls against Moonshot's
// per-account QPS limits.
//
// It is the shared limiter (internal/toolquota) over three named pools: the
// limits are counted per Moonshot ACCOUNT (not per key), because Moonshot
// counts them per account regardless of which key made the call, and the
// router must not exceed it by spreading traffic over several keys of the
// same account.
type Quota struct {
	quota *toolquota.Quota
}

// NewQuota builds a Quota. A nil client selects the in-process fallback.
func NewQuota(client valkey.Client, keyPrefix string, searchQPS, searchProQPS, fetchQPS int) *Quota {
	if searchQPS <= 0 {
		searchQPS = DefaultSearchQPS
	}
	if searchProQPS <= 0 {
		searchProQPS = DefaultSearchProQPS
	}
	if fetchQPS <= 0 {
		fetchQPS = DefaultFetchQPS
	}
	return &Quota{quota: toolquota.NewQuota(client, keyPrefix, map[string]int{
		poolSearch:    searchQPS,
		poolSearchPro: searchProQPS,
		poolFetch:     fetchQPS,
	})}
}

// AllowSearch admits one Basic search call for the given account.
func (q *Quota) AllowSearch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.quota.Allow(ctx, poolSearch, account)
}

// AllowSearchPro admits one Pro search call for the given account.
func (q *Quota) AllowSearchPro(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.quota.Allow(ctx, poolSearchPro, account)
}

// AllowFetch admits one Fetch call for the given account.
func (q *Quota) AllowFetch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.quota.Allow(ctx, poolFetch, account)
}
