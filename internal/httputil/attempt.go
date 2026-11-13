package httputil

import (
	"context"
	"sync"
	"time"
)

// The kinds of upstream attempt (aar.attempt.kind).
const (
	// AttemptFirst is the first send of a request.
	AttemptFirst = "first"
	// AttemptRetry is a send to the next credential after one failed.
	AttemptRetry = "retry"
	// AttemptFallback is a send to a fallback credential.
	AttemptFallback = "fallback"
)

type Attempts struct {
	mu         sync.Mutex
	nextKind   string
	nextReason string
	attempts   []attempt
}

type attempt struct {
	credential string
	kind       string
	reason     string
	began      time.Time
	tunnel     tunnel // the way of the connection it was sent on
	reused     *bool  // that connection was a pooled one
	connWait   *time.Duration
	ttfb       *time.Duration
}

// Stages is how a request's time divided around one of its attempts.
type Stages struct {
	// Retry is the first attempt's send to this one's: what the ones before it cost.
	Retry time.Duration
	// ConnWait is asking for a connection to having one.
	ConnWait *time.Duration
	// Reused is whether that connection was a pooled one.
	Reused *bool
	// TTFB is the request written to the first byte of the response.
	TTFB *time.Duration
	// ExitRTT and ExitConnect are the round trip to the exit and the exit's
	// connect to the provider, measured when the connection was opened.
	ExitRTT     *time.Duration
	ExitConnect *time.Duration
}

type attemptsKey struct{}

// WithAttempts returns ctx carrying a new, empty Attempts.
func WithAttempts(ctx context.Context) context.Context {
	return context.WithValue(ctx, attemptsKey{}, &Attempts{})
}

// AttemptsFrom returns the Attempts ctx carries, nil without one.
func AttemptsFrom(ctx context.Context) *Attempts {
	a, _ := ctx.Value(attemptsKey{}).(*Attempts)
	return a
}

// Next says why the attempt that follows happens: its kind (AttemptRetry or
// AttemptFallback) and the reason the one before it failed.
func (a *Attempts) Next(kind, reason string) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.nextKind, a.nextReason = kind, reason
}

// Begin records that an attempt is being sent to credential. Without a Next
// before it, the first is AttemptFirst and a later one AttemptRetry.
func (a *Attempts) Begin(credential string) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	kind := a.nextKind
	if kind == "" {
		kind = AttemptFirst
		if len(a.attempts) > 0 {
			kind = AttemptRetry
		}
	}
	a.attempts = append(a.attempts, attempt{credential: credential, kind: kind, reason: a.nextReason, began: time.Now()})
	a.nextKind, a.nextReason = "", ""
}

// Count is the number of attempts sent.
func (a *Attempts) Count() int {
	if a == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.attempts)
}

// Exit is the exit of the last attempt sent to credential, "" without one.
func (a *Attempts) Exit(credential string) string {
	if a == nil {
		return ""
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := len(a.attempts) - 1; i >= 0; i-- {
		if a.attempts[i].credential == credential {
			return a.attempts[i].tunnel.exit
		}
	}
	return ""
}

// Stages is the stages of the last attempt sent to credential, false without one.
func (a *Attempts) Stages(credential string) (Stages, bool) {
	if a == nil {
		return Stages{}, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := len(a.attempts) - 1; i >= 0; i-- {
		if at := a.attempts[i]; at.credential == credential {
			return Stages{
				Retry:       at.began.Sub(a.attempts[0].began),
				ConnWait:    at.connWait,
				Reused:      at.reused,
				TTFB:        at.ttfb,
				ExitRTT:     at.tunnel.exitRTT,
				ExitConnect: at.tunnel.exitConnect,
			}, true
		}
	}
	return Stages{}, false
}

// current is the attempt in flight and its number from 1, 0 before the first Begin.
func (a *Attempts) current() (int, attempt) {
	if a == nil {
		return 0, attempt{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.attempts) == 0 {
		return 0, attempt{}
	}
	return len(a.attempts), a.attempts[len(a.attempts)-1]
}

// observed adds what a round trip of attempt number learned. Of several
// round trips (an embeddings fan-out) the first measured values stay.
func (a *Attempts) observed(number int, way tunnel, reused *bool, connWait, ttfb *time.Duration) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if number < 1 || number > len(a.attempts) {
		return
	}
	at := &a.attempts[number-1]
	if way != (tunnel{}) {
		at.tunnel = way
	}
	if at.reused == nil {
		at.reused = reused
	}
	if at.connWait == nil {
		at.connWait = connWait
	}
	if at.ttfb == nil {
		at.ttfb = ttfb
	}
}
