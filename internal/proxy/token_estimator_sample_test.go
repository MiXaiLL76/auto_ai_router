package proxy

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func sampleTestBody(t *testing.T, messages int) map[string]interface{} {
	t.Helper()
	parts := []string{
		"Refactor the handler so that the request body is decoded once and passed down.",
		"Посмотри, почему падает сборка после последнего мержа в основную ветку.",
		"func (p *Proxy) estimate(body []byte) int { return len(body) / 4 }",
		`{"level":"error","msg":"upstream timeout","ms":30000}`,
	}
	msgs := make([]map[string]interface{}, 0, messages)
	for i := range messages {
		var b strings.Builder
		for j := range 8 {
			fmt.Fprintf(&b, "%s %d ", parts[(i+j)%len(parts)], i*8+j)
		}
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs = append(msgs, map[string]interface{}{"role": role, "content": b.String()})
	}
	data, err := json.Marshal(map[string]interface{}{"model": "gpt-6-luna", "messages": msgs})
	require.NoError(t, err)
	raw, ok := decodeRequestBody(data)
	require.True(t, ok)
	return raw
}

func TestEstimatePromptTokensBySampleCountsShortPromptExactly(t *testing.T) {
	raw := sampleTestBody(t, 10)
	assert.Equal(t, estimatePromptTokensFromRaw(raw, "gpt-6-luna"), estimatePromptTokensBySample(raw, "gpt-6-luna"))
}

func TestEstimatePromptTokensBySampleKeepsShortTextBoundaries(t *testing.T) {
	parts := make([]map[string]interface{}, 4000)
	for i := range parts {
		parts[i] = map[string]interface{}{"type": "text", "text": "hello"}
	}
	data, err := json.Marshal(map[string]interface{}{
		"model":    "gpt-6-luna",
		"messages": []map[string]interface{}{{"role": "user", "content": parts}},
	})
	require.NoError(t, err)
	raw, ok := decodeRequestBody(data)
	require.True(t, ok)
	full := estimatePromptTokensFromRaw(raw, "gpt-6-luna")
	sample := estimatePromptTokensBySample(raw, "gpt-6-luna")
	assert.InEpsilon(t, full, sample, 0.02)
}

func TestEstimatePromptTokensBySampleStaysCloseOnLongPrompt(t *testing.T) {
	raw := sampleTestBody(t, 400)
	full := estimatePromptTokensFromRaw(raw, "gpt-6-luna")
	sample := estimatePromptTokensBySample(raw, "gpt-6-luna")
	assert.InEpsilon(t, full, sample, 0.1)
}
