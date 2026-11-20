package proxy

import (
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

// toolLoopRun is what a router-run tool loop produced, whichever provider's
// tools it ran.
//
// The two adapters answer with the same shape, so the router's dispatch is
// one block: the provider's own loop fills the fields it knows and leaves
// the rest at zero.
type toolLoopRun interface {
	// body is the model's final answer, in the upstream's own Chat shape.
	body() []byte
	// usage is the summed usage of every step of the loop.
	usage() *converter.TokenUsage
	// searchCalls is the billable Basic search count.
	searchCalls() int
	// searchProCalls is the billable Pro search count.
	searchProCalls() int
	// fetchCalls is the billable fetch count.
	fetchCalls() int
}
