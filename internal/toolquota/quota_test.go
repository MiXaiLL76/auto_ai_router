package toolquota

import (
	"context"
	"testing"
)

// TestPoolsAreSeparate checks that the named pools do not share a budget:
// traffic to one tool does not consume another's quota.
func TestPoolsAreSeparate(t *testing.T) {
	quota := NewQuota(nil, "test:", map[string]int{"search": 1, "search_pro": 1, "fetch": 1})
	ctx := context.Background()

	if admitted, _, err := quota.Allow(ctx, "search", "account"); err != nil || !admitted {
		t.Fatalf("the first search was not admitted: %v", err)
	}
	if admitted, _, err := quota.Allow(ctx, "search", "account"); err != nil {
		t.Fatalf("Allow: %v", err)
	} else if admitted {
		t.Error("the second search must not be admitted (the limit is 1)")
	}
	if admitted, _, err := quota.Allow(ctx, "search_pro", "account"); err != nil || !admitted {
		t.Errorf("the Pro search must have its own budget: %v", err)
	}
	if admitted, _, err := quota.Allow(ctx, "fetch", "account"); err != nil || !admitted {
		t.Errorf("the fetch must have its own budget: %v", err)
	}
}

// TestTheLimitIsPerAccount checks that the limit is counted per account: one
// account's traffic does not block another's.
func TestTheLimitIsPerAccount(t *testing.T) {
	quota := NewQuota(nil, "test:", map[string]int{"search": 1})
	ctx := context.Background()

	if admitted, _, _ := quota.Allow(ctx, "search", "account-a"); !admitted {
		t.Fatal("the first search of account-a was not admitted")
	}
	if admitted, _, _ := quota.Allow(ctx, "search", "account-a"); admitted {
		t.Error("the second search of account-a must not be admitted")
	}
	if admitted, _, _ := quota.Allow(ctx, "search", "account-b"); !admitted {
		t.Error("account-b's first search must be admitted")
	}
}

// TestAnUnlimitedPoolAdmitsEverything checks that a limit of zero (or less)
// means unlimited.
func TestAnUnlimitedPoolAdmitsEverything(t *testing.T) {
	quota := NewQuota(nil, "test:", map[string]int{"search": 0, "fetch": -1})
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		if admitted, _, err := quota.Allow(ctx, "search", "account"); err != nil || !admitted {
			t.Fatalf("call %d was not admitted: %v", i, err)
		}
		if admitted, _, err := quota.Allow(ctx, "fetch", "account"); err != nil || !admitted {
			t.Fatalf("call %d was not admitted: %v", i, err)
		}
	}
}

// TestAnUnknownPoolIsUnlimited checks that a pool the limiter does not know
// about is not limited: a tool without a configured limit must not be
// blocked by mistake.
func TestAnUnknownPoolIsUnlimited(t *testing.T) {
	quota := NewQuota(nil, "test:", map[string]int{"search": 1})
	ctx := context.Background()
	for i := 0; i < 10; i++ {
		if admitted, _, err := quota.Allow(ctx, "unknown", "account"); err != nil || !admitted {
			t.Fatalf("call %d was not admitted: %v", i, err)
		}
	}
}
