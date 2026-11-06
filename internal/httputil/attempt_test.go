package httputil

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// said is what an attempt says of itself, without when it was sent.
func said(a attempt) attempt {
	return attempt{credential: a.credential, kind: a.kind, reason: a.reason, tunnel: tunnel{exit: a.tunnel.exit}}
}

func TestAttempts_KindsReasonsAndExits(t *testing.T) {
	ctx := WithAttempts(context.Background())
	attempts := AttemptsFrom(ctx)
	number, _ := attempts.current()
	assert.Zero(t, number, "nothing sent yet")
	assert.Zero(t, attempts.Count())

	attempts.Begin("cred-a")
	number, current := attempts.current()
	assert.Equal(t, 1, number)
	assert.Equal(t, attempt{credential: "cred-a", kind: AttemptFirst}, said(current))
	attempts.observed(number, tunnel{exit: "exit-a"}, nil, nil, nil)

	attempts.Next(AttemptRetry, "rate_limit")
	attempts.Begin("cred-b")
	number, current = attempts.current()
	assert.Equal(t, 2, number)
	assert.Equal(t, attempt{credential: "cred-b", kind: AttemptRetry, reason: "rate_limit"}, said(current))

	attempts.Begin("cred-c")
	_, current = attempts.current()
	assert.Equal(t, attempt{credential: "cred-c", kind: AttemptRetry}, said(current), "a later attempt without a Next is a retry, and the reason is not carried over")

	attempts.Next(AttemptFallback, "server_error")
	attempts.Begin("cred-a")
	number, current = attempts.current()
	assert.Equal(t, attempt{credential: "cred-a", kind: AttemptFallback, reason: "server_error"}, said(current))
	attempts.observed(number, tunnel{exit: "exit-z"}, nil, nil, nil)
	attempts.observed(number, tunnel{}, nil, nil, nil)
	attempts.observed(99, tunnel{exit: "nowhere"}, nil, nil, nil)

	assert.Equal(t, 4, attempts.Count())
	assert.Equal(t, "exit-z", attempts.Exit("cred-a"), "the last attempt on the credential")
	assert.Empty(t, attempts.Exit("cred-b"), "it named no exit")
	assert.Empty(t, attempts.Exit("never-tried"))
}

func TestAttempts_Stages(t *testing.T) {
	attempts := AttemptsFrom(WithAttempts(context.Background()))
	_, ok := attempts.Stages("cred-a")
	assert.False(t, ok, "nothing sent yet")

	attempts.Begin("cred-a")
	time.Sleep(15 * time.Millisecond)
	attempts.Next(AttemptRetry, "server_error")
	attempts.Begin("cred-b")

	first, ok := attempts.Stages("cred-a")
	require.True(t, ok)
	assert.Zero(t, first.Retry, "nothing before the first attempt")
	assert.Nil(t, first.ConnWait)
	assert.Nil(t, first.Reused)
	assert.Nil(t, first.TTFB)
	assert.Nil(t, first.ExitRTT)
	assert.Nil(t, first.ExitConnect)

	wait, firstByte, later := 3*time.Millisecond, 40*time.Millisecond, time.Second
	pooled, dialed, rtt, connect := true, false, 50*time.Millisecond, 9*time.Millisecond
	attempts.observed(2, tunnel{exit: "exit-a", exitRTT: &rtt, exitConnect: &connect}, &pooled, &wait, &firstByte)
	attempts.observed(2, tunnel{}, &dialed, &later, &later)
	second, ok := attempts.Stages("cred-b")
	require.True(t, ok)
	assert.GreaterOrEqual(t, second.Retry, 15*time.Millisecond, "what the first attempt cost")
	assert.Equal(t, &wait, second.ConnWait, "an attempt of several round trips keeps the first's times")
	assert.Equal(t, &firstByte, second.TTFB)
	assert.Equal(t, &pooled, second.Reused)
	assert.Equal(t, &rtt, second.ExitRTT, "a round trip that says nothing of the way leaves it")
	assert.Equal(t, &connect, second.ExitConnect)

	_, ok = attempts.Stages("never-tried")
	assert.False(t, ok)
}

func TestAttempts_NilRecordsNothing(t *testing.T) {
	attempts := AttemptsFrom(context.Background())
	assert.Nil(t, attempts)
	attempts.Next(AttemptRetry, "rate_limit")
	attempts.Begin("cred-a")
	attempts.observed(1, tunnel{exit: "exit-a"}, nil, nil, nil)
	number, _ := attempts.current()
	assert.Zero(t, number)
	assert.Zero(t, attempts.Count())
	assert.Empty(t, attempts.Exit("cred-a"))
	_, ok := attempts.Stages("cred-a")
	assert.False(t, ok)
}
