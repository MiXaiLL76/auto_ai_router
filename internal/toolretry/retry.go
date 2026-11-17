// Package toolretry retries the router's own tool calls when the provider
// answers transiently.
//
// The providers ask for a retry after a 429 (their QPS limit) and after a
// 5xx: those are transient, and the router's own QPS limiter makes traffic
// near the limit ordinary. Other 4xx answers are the request's own fault
// and are not retried -- retrying them would only waste the account's quota.
package toolretry

import (
	"context"
	"net/http"
	"time"
)

// IsRetryableStatus reports whether a status is worth retrying: the
// providers ask for a retry after a 429 (their QPS limit) and after a 5xx.
func IsRetryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// Delay is how long to wait before the next attempt: a growing delay, longer
// after a 429 than after a 5xx (a 429 means the account is at its limit, so
// the wait has to outlast the sliding window).
func Delay(status int, attempt int) time.Duration {
	if status == http.StatusTooManyRequests {
		return time.Duration(attempt+1) * 500 * time.Millisecond
	}
	return time.Duration(attempt+1) * time.Second
}

// Sleep waits for the given duration, or until the context is done. It
// returns the context's error when the wait was cut short.
func Sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Do runs the call until it succeeds, until the answer is not worth
// retrying, or until the retries run out.
//
// The call reports the provider's status and its raw body; a non-2xx answer
// is a valid answer the caller has to interpret, not an error. Only a
// transport failure is an error, and it is not retried: the provider was
// never reached.
func Do(
	ctx context.Context,
	retries int,
	call func() (int, []byte, error),
) (int, []byte, error) {
	var status int
	var body []byte
	var err error
	for attempt := 0; ; attempt++ {
		status, body, err = call()
		if err != nil {
			return status, body, err
		}
		if !IsRetryableStatus(status) || attempt >= retries {
			return status, body, nil
		}
		if waitErr := Sleep(ctx, Delay(status, attempt)); waitErr != nil {
			return status, body, waitErr
		}
	}
}
