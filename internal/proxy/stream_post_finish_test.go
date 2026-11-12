package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatPostFinishDetectorPerChoice(t *testing.T) {
	var detector chatPostFinishDetector
	observe := func(data string) []chatPostFinishViolation {
		return detector.observePayload([]byte(data))
	}

	assert.Empty(t, observe(`{"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":"stop"},{"index":1,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}`))
	assert.Empty(t, observe(`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}`))
	assert.Empty(t, observe(`{"choices":[{"index":1,"delta":{"content":"still running"},"finish_reason":null}]}`))
	assert.Empty(t, observe(`{"choices":[{"index":0,"delta":{"content":"","reasoning_content":null},"finish_reason":null}]}`))
	violations := observe(`{"choices":[{"index":0,"delta":{"reasoning_content":"late"},"finish_reason":null}]}`)
	require.Equal(t, []chatPostFinishViolation{{index: 0, finishReason: "stop", field: "reasoning_content"}}, violations)
	assert.Empty(t, observe(`{"choices":[{"index":0,"delta":{"content":"another late delta"}}]}`), "report once per choice")
	assert.Empty(t, observe(`{"choices":[{"index":1,"delta":{},"finish_reason":"length"}]}`))
	require.Equal(t, []chatPostFinishViolation{{index: 1, finishReason: "length", field: "content"}},
		observe(`{"choices":[{"index":1,"delta":{"content":"late answer"}}]}`))
}

func TestChatPostFinishDetectorReassemblesSplitSSELine(t *testing.T) {
	var detector chatPostFinishDetector
	var lines streamUsageLines
	var violations []chatPostFinishViolation
	consume := func(line []byte) {
		for _, payload := range splitSSEPayloads(line, nil) {
			violations = append(violations, detector.observePayload(payload)...)
		}
	}
	lines.Observe([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"), consume)
	lines.Observe([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_"), consume)
	assert.Empty(t, violations)
	lines.Observe([]byte("content\":\"late\"}}]}\n\n"), consume)
	require.Equal(t, []chatPostFinishViolation{{index: 0, finishReason: "stop", field: "reasoning_content"}}, violations)
}
