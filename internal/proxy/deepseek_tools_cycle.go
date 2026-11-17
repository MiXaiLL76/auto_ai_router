package proxy

import (
	"errors"
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/toolloop"
)

// errDeepSeekToolCycleAborted ends the cycle when the client went away.
//
// It is the shared sentinel (internal/toolloop.ErrAborted), so the router's
// dispatch checks it once for every provider's loop: a disconnect is not a
// failure of the loop.
var errDeepSeekToolCycleAborted = toolloop.ErrAborted

// upstreamStatusError is an upstream answer the router relays as it is.
type upstreamStatusError struct {
	status int
	body   []byte
}

func (e *upstreamStatusError) Error() string {
	return fmt.Sprintf("deepseektools: upstream returned HTTP %d", e.status)
}

// errToolLoopAborted checks whether an error is the cycle's abort sentinel.
func errToolLoopAborted(err error) bool {
	return errors.Is(err, toolloop.ErrAborted)
}
