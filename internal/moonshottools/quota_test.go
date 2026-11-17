package moonshottools

import (
	"context"
	"testing"
)

// TestQuotaKeepsThePoolsApart checks that the three tools have separate
// pools: traffic to one tool does not consume another's quota.
func TestQuotaKeepsThePoolsApart(t *testing.T) {
	quota := NewQuota(nil, "test:", 1, 1, 1)
	ctx := context.Background()

	// The Basic search pool is exhausted.
	admitted, _, err := quota.AllowSearch(ctx, "account")
	if err != nil || !admitted {
		t.Fatalf("the first search was not admitted: %v", err)
	}
	admitted, _, err = quota.AllowSearch(ctx, "account")
	if err != nil {
		t.Fatalf("AllowSearch: %v", err)
	}
	if admitted {
		t.Error("the second search must not be admitted (the limit is 1)")
	}

	// The Pro search and the fetch have their own budgets.
	admitted, _, err = quota.AllowSearchPro(ctx, "account")
	if err != nil || !admitted {
		t.Errorf("the Pro search must have its own budget: %v", err)
	}
	admitted, _, err = quota.AllowFetch(ctx, "account")
	if err != nil || !admitted {
		t.Errorf("the fetch must have its own budget: %v", err)
	}
}

// TestQuotaIsPerAccount checks that the limit is counted per account: one
// account's traffic does not block another's.
func TestQuotaIsPerAccount(t *testing.T) {
	quota := NewQuota(nil, "test:", 1, 1, 1)
	ctx := context.Background()

	if admitted, _, _ := quota.AllowSearch(ctx, "account-a"); !admitted {
		t.Fatal("the first search of account-a was not admitted")
	}
	if admitted, _, _ := quota.AllowSearch(ctx, "account-a"); admitted {
		t.Error("the second search of account-a must not be admitted")
	}
	if admitted, _, _ := quota.AllowSearch(ctx, "account-b"); !admitted {
		t.Error("account-b's first search must be admitted")
	}
}

// TestQuotaDefaultsAreTheProviderLimits checks the fallback limits: an
// unconfigured quota uses the provider's own account-level limits.
func TestQuotaDefaultsAreTheProviderLimits(t *testing.T) {
	quota := NewQuota(nil, "test:", 0, 0, 0)
	if quota.searchQPS != DefaultSearchQPS {
		t.Errorf("the search limit = %d, want %d", quota.searchQPS, DefaultSearchQPS)
	}
	if quota.searchProQPS != DefaultSearchProQPS {
		t.Errorf("the search_pro limit = %d, want %d", quota.searchProQPS, DefaultSearchProQPS)
	}
	if quota.fetchQPS != DefaultFetchQPS {
		t.Errorf("the fetch limit = %d, want %d", quota.fetchQPS, DefaultFetchQPS)
	}
}
