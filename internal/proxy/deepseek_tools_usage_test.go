package proxy

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
)

// TestApplyDeepSeekToolUsageLeavesUnreportedFigures covers the loop's empty sum:
// a loop whose steps never reported usage has zero counts, which is "not
// reported", not "nothing generated". The returned answer's locally estimated
// figures must survive it; a positive loop count still replaces them.
func TestApplyDeepSeekToolUsageLeavesUnreportedFigures(t *testing.T) {
	usage := &converter.TokenUsage{
		PromptTokens:      308,
		CompletionTokens:  42,
		CachedInputTokens: 10,
		ReasoningTokens:   7,
		WebSearchRequests: 2,
	}
	loop := &converter.TokenUsage{WebSearchRequests: 1}

	applyDeepSeekToolUsageTo(usage, loop)

	assert.Equal(t, 308, usage.PromptTokens, "an unreported loop prompt must not zero the local estimate")
	assert.Equal(t, 42, usage.CompletionTokens, "an unreported loop completion must not zero the streamed answer")
	assert.Equal(t, 10, usage.CachedInputTokens)
	assert.Equal(t, 7, usage.ReasoningTokens)
	// The router's own tool counter is still folded in.
	assert.Equal(t, 3, usage.WebSearchRequests)

	reported := &converter.TokenUsage{
		PromptTokens:      1000,
		CompletionTokens:  50,
		CachedInputTokens: 100,
		ReasoningTokens:   20,
	}
	applyDeepSeekToolUsageTo(usage, reported)

	assert.Equal(t, 1000, usage.PromptTokens)
	assert.Equal(t, 50, usage.CompletionTokens)
	assert.Equal(t, 100, usage.CachedInputTokens)
	assert.Equal(t, 20, usage.ReasoningTokens)
}
