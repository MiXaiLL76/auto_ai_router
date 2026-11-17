package deepseektools

import (
	"context"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/toolquota"
	"github.com/valkey-io/valkey-go"
)

// DeepSeek's account-level QPS limits for the two tools, per the vendor's
// guide: the limits are per account (not per key), the two are independent
// (traffic to one does not consume the other's quota), and Fetch is
// additionally capped per domain.
const (
	// DefaultSearchQPS is the account-level Search limit.
	DefaultSearchQPS = 100
	// DefaultFetchQPS is the account-level Fetch limit.
	DefaultFetchQPS = 25
)

// The names of the two pools, in the Redis keys and in the quota's own
// lookup.
const (
	poolSearch = "search"
	poolFetch  = "fetch"
)

// Quota admits Search and Fetch calls against DeepSeek's per-account QPS
// limits.
//
// It is the shared limiter (internal/toolquota) over two named pools: the
// limits are counted per DeepSeek ACCOUNT (not per key), because DeepSeek
// counts them per account regardless of which key made the call, and the
// router must not exceed it by spreading traffic over several keys of the
// same account.
type Quota struct {
	quota *toolquota.Quota
}

// NewQuota builds a Quota. A nil client selects the in-process fallback.
func NewQuota(client valkey.Client, keyPrefix string, searchQPS, fetchQPS int) *Quota {
	if searchQPS <= 0 {
		searchQPS = DefaultSearchQPS
	}
	if fetchQPS <= 0 {
		fetchQPS = DefaultFetchQPS
	}
	return &Quota{quota: toolquota.NewQuota(client, keyPrefix, map[string]int{
		poolSearch: searchQPS,
		poolFetch:  fetchQPS,
	})}
}

// AllowSearch admits one Search call for the given account.
func (q *Quota) AllowSearch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.quota.Allow(ctx, poolSearch, account)
}

// AllowFetch admits one Fetch call for the given account.
func (q *Quota) AllowFetch(ctx context.Context, account string) (bool, time.Duration, error) {
	return q.quota.Allow(ctx, poolFetch, account)
}
