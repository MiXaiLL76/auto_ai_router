package converterutil

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestImageFetchBudgetCapsImages checks the image cap: a request carrying
// more attachments than the router is willing to download is refused at the
// cap rather than after downloading all of them.
func TestImageFetchBudgetCapsImages(t *testing.T) {
	budget := NewImageFetchBudget(ImageFetchLimits{MaxImages: 2, Timeout: time.Minute})

	assert.True(t, budget.Reserve(), "the first image fits the budget")
	assert.True(t, budget.Reserve(), "the second image fits the budget")
	assert.False(t, budget.Reserve(), "the third image exceeds the cap")
	assert.Equal(t, 2, budget.Fetched())
}

// TestImageFetchBudgetWithoutCap checks that a non-positive cap means "no
// limit".
func TestImageFetchBudgetWithoutCap(t *testing.T) {
	budget := NewImageFetchBudget(ImageFetchLimits{Timeout: time.Minute})
	for range 100 {
		assert.True(t, budget.Reserve())
	}
	assert.Equal(t, 100, budget.Fetched())
}

// TestNilImageFetchBudgetIsPermissive checks that the nil budget (no budget
// configured) allows every fetch — the budget must never be the reason a
// request fails when it was not configured.
func TestNilImageFetchBudgetIsPermissive(t *testing.T) {
	var budget *ImageFetchBudget
	assert.True(t, budget.Reserve())
	assert.Equal(t, 0, budget.Fetched())
	assert.Equal(t, time.Duration(0), budget.Remaining())

	ctx, cancel := budget.WithImageFetchDeadline(context.Background())
	defer cancel()
	assert.NotNil(t, ctx)
}

// TestImageFetchBudgetDeadline checks the request-wide deadline: the
// remaining time shrinks and a fetch started after the deadline is
// cancelled immediately.
func TestImageFetchBudgetDeadline(t *testing.T) {
	budget := NewImageFetchBudget(ImageFetchLimits{MaxImages: 10, Timeout: 200 * time.Millisecond})

	remaining := budget.Remaining()
	assert.Positive(t, remaining, "a budget with a timeout has time left")
	assert.LessOrEqual(t, remaining, 200*time.Millisecond)

	ctx, cancel := budget.WithImageFetchDeadline(context.Background())
	defer cancel()

	deadline, ok := ctx.Deadline()
	require.True(t, ok, "the fetch context must carry the budget's deadline")
	assert.WithinDuration(t, time.Now().Add(200*time.Millisecond), deadline, 100*time.Millisecond)

	// Once the deadline passes, the context is done.
	time.Sleep(250 * time.Millisecond)
	select {
	case <-ctx.Done():
	default:
		t.Fatal("the context must be cancelled once the budget's deadline passes")
	}
}

// TestImageFetchBudgetWithoutTimeoutHasNoDeadline checks that a budget
// without a timeout yields the parent context unchanged.
func TestImageFetchBudgetWithoutTimeoutHasNoDeadline(t *testing.T) {
	budget := NewImageFetchBudget(ImageFetchLimits{MaxImages: 10})

	assert.Equal(t, time.Duration(0), budget.Remaining())

	ctx, cancel := budget.WithImageFetchDeadline(context.Background())
	defer cancel()
	_, ok := ctx.Deadline()
	assert.False(t, ok, "a budget without a timeout must not impose a deadline")
}

// TestImageFetchBudgetIsConcurrencySafe checks the budget under parallel
// use: the cap must hold no matter how many goroutines race for it.
func TestImageFetchBudgetIsConcurrencySafe(t *testing.T) {
	const cap = 10
	budget := NewImageFetchBudget(ImageFetchLimits{MaxImages: cap, Timeout: time.Minute})

	granted := make(chan bool, 100)
	for range 100 {
		go func() { granted <- budget.Reserve() }()
	}

	count := 0
	for range 100 {
		if <-granted {
			count++
		}
	}
	assert.Equal(t, cap, count, "exactly the cap may be granted")
}
