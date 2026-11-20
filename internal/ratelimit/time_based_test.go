package ratelimit

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestTimeBasedRateLimiter_WaitZeroInterval(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()

	// Zero interval should return immediately
	err := limiter.Wait(ctx, "key", 0)
	assert.NoError(t, err)
}

func TestTimeBasedRateLimiter_WaitNegativeInterval(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()

	// Negative interval should return immediately
	err := limiter.Wait(ctx, "key", -1*time.Second)
	assert.NoError(t, err)
}

func TestTimeBasedRateLimiter_FirstCall(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()

	// First call with any interval should return immediately
	start := time.Now()
	err := limiter.Wait(ctx, "key", 100*time.Millisecond)
	assert.NoError(t, err)
	elapsed := time.Since(start)
	assert.Less(t, elapsed, 50*time.Millisecond)
}

func TestTimeBasedRateLimiter_EnforcesInterval(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()
	interval := 80 * time.Millisecond

	// First call returns immediately
	err := limiter.Wait(ctx, "key", interval)
	assert.NoError(t, err)

	// Small delay to ensure we're within the wait window
	time.Sleep(10 * time.Millisecond)

	// Second call should wait for remaining time
	start := time.Now()
	err = limiter.Wait(ctx, "key", interval)
	assert.NoError(t, err)
	elapsed := time.Since(start)

	// Should have waited at least 40ms (80ms - 10ms elapsed - 30ms tolerance for system jitter)
	assert.GreaterOrEqual(t, elapsed, 40*time.Millisecond)
}

func TestTimeBasedRateLimiter_MultipleKeys(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()
	interval := 60 * time.Millisecond

	// Test that key1 timer doesn't interfere with key2 timer
	// Call Wait for key1 - returns immediately
	err := limiter.Wait(ctx, "key1", interval)
	assert.NoError(t, err)

	// Call Wait for key2 - returns immediately (independent of key1)
	err = limiter.Wait(ctx, "key2", interval)
	assert.NoError(t, err)

	// Sleep 25ms. The sleep is measured rather than assumed: a loaded machine
	// (a CI runner under parallel packages) can overshoot it by tens of
	// milliseconds, and the waits below are computed from the interval, not
	// from the sleep's nominal length.
	sleepStart := time.Now()
	time.Sleep(25 * time.Millisecond)
	slept := time.Since(sleepStart)

	// Call Wait for key1 again - should wait the rest of the interval
	// (60 - slept). The floor is what the interval still owes, minus a small
	// tolerance for the clock's own resolution; asserting the nominal 30ms
	// instead failed whenever the sleep above overshot.
	remaining := interval - slept
	start := time.Now()
	err = limiter.Wait(ctx, "key1", interval)
	assert.NoError(t, err)
	elapsed := time.Since(start)
	assert.GreaterOrEqual(t, elapsed, remaining-10*time.Millisecond,
		"key1 should wait out the rest of its interval")
	assert.Less(t, elapsed, interval,
		"key1 must not wait longer than its interval")

	// Immediately after (before key2's interval expires), call Wait for key2
	// It should return immediately since total time is ~60ms from its first call
	start = time.Now()
	err = limiter.Wait(ctx, "key2", interval)
	assert.NoError(t, err)
	elapsed = time.Since(start)
	// key2 had the sleep + key1's wait = the full interval, so it returns
	// immediately. The bound is the interval itself: a loaded machine may
	// have spent longer in the two waits above than planned, in which case
	// key2's interval has already expired and it returns immediately too --
	// either way it must not block.
	assert.Less(t, elapsed, interval,
		"key2's own interval is independent of key1's and must have expired by now")
}

func TestTimeBasedRateLimiter_ContextCancellation(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	interval := 200 * time.Millisecond

	// First call
	err := limiter.Wait(context.Background(), "key", interval)
	assert.NoError(t, err)

	// Second call with cancelled context should return error
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = limiter.Wait(ctx, "key", interval)
	assert.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestTimeBasedRateLimiter_ContextDeadline(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	interval := 200 * time.Millisecond

	// First call
	err := limiter.Wait(context.Background(), "key", interval)
	assert.NoError(t, err)

	// Second call with deadline that expires before interval
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err = limiter.Wait(ctx, "key", interval)
	assert.Error(t, err)
	assert.Equal(t, context.DeadlineExceeded, err)
}

func TestTimeBasedRateLimiter_Reset(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()
	interval := 50 * time.Millisecond

	// First call
	err := limiter.Wait(ctx, "key", interval)
	assert.NoError(t, err)

	// Reset the limiter
	limiter.Reset("key")

	// Now second call should return immediately (as if first call never happened)
	start := time.Now()
	err = limiter.Wait(ctx, "key", interval)
	assert.NoError(t, err)
	elapsed := time.Since(start)
	assert.Less(t, elapsed, 10*time.Millisecond)
}

func TestTimeBasedRateLimiter_ResetAll(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()
	interval := 50 * time.Millisecond

	// First calls with multiple keys
	for _, key := range []string{"key1", "key2", "key3"} {
		err := limiter.Wait(ctx, key, interval)
		assert.NoError(t, err)
	}

	// Reset all
	limiter.ResetAll()

	// Now all keys should return immediately
	for _, key := range []string{"key1", "key2", "key3"} {
		start := time.Now()
		err := limiter.Wait(ctx, key, interval)
		assert.NoError(t, err)
		elapsed := time.Since(start)
		assert.Less(t, elapsed, 10*time.Millisecond)
	}
}

func TestTimeBasedRateLimiter_ThreadSafety(t *testing.T) {
	limiter := NewTimeBasedRateLimiter()
	ctx := context.Background()
	interval := 10 * time.Millisecond

	// Make concurrent calls - should not panic or race
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(index int) {
			key := "key" + string(rune(index%3))
			_ = limiter.Wait(ctx, key, interval)
			done <- true
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}
