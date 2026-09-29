package budget

import "context"

// reserveBackend is the pluggable storage layer behind Reserver. All
// implementations must be safe for concurrent use.
type reserveBackend interface {
	// tryReserve seeds the counter from dbSpend (only if not already seeded),
	// adds estimatedCost, and checks the result against maxBudget. maxBudget < 0
	// means unlimited (always allowed, spend still tracked). Returns
	// allowed=false without applying estimatedCost when the budget is exceeded.
	tryReserve(ctx context.Context, entity string, dbSpend, estimatedCost, maxBudget float64) (bool, error)

	// reconcile adjusts a previously reserved amount by delta.
	reconcile(ctx context.Context, entity string, delta float64) error
}
