package ratelimit

import (
	"context"
	"sync"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/utils"
)

// TimeBasedRateLimiter enforces a minimum time interval between operations per key.
// Use this for simple interval-based rate limiting scenarios where you need to enforce
// a fixed minimum wait time between consecutive operations.
//
// Different from RPMLimiter:
// - TimeBasedRateLimiter: enforces fixed minimum interval (e.g., wait 100ms between requests)
// - RPMLimiter: tracks usage against configurable RPM/TPM limits (e.g., allow max 100 requests per minute)
//
// Thread-safe via internal mutex.
type TimeBasedRateLimiter struct {
	mu   sync.Mutex
	last map[string]time.Time
}

// NewTimeBasedRateLimiter creates a new interval-based rate limiter
func NewTimeBasedRateLimiter() *TimeBasedRateLimiter {
	return &TimeBasedRateLimiter{
		last: make(map[string]time.Time),
	}
}

// Wait blocks until the minimum interval has passed since the last operation for the key.
// If minInterval <= 0, returns immediately (no rate limiting).
// Returns error if context is cancelled while waiting.
func (l *TimeBasedRateLimiter) Wait(ctx context.Context, key string, minInterval time.Duration) error {
	if minInterval <= 0 {
		return nil
	}

	l.mu.Lock()
	now := utils.NowUTC()
	last := l.last[key]
	waitFor := minInterval - now.Sub(last)
	if waitFor <= 0 {
		l.last[key] = now
		l.mu.Unlock()
		return nil
	}
	l.mu.Unlock()

	timer := time.NewTimer(waitFor)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		l.mu.Lock()
		l.last[key] = utils.NowUTC()
		l.mu.Unlock()
		return nil
	}
}

// Reset clears the tracking for a specific key
func (l *TimeBasedRateLimiter) Reset(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.last, key)
}

// ResetAll clears all tracking
func (l *TimeBasedRateLimiter) ResetAll() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.last = make(map[string]time.Time)
}
