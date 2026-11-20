package litellm

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatStreamKeepsRefusalOnlyDelta(t *testing.T) {
	for _, tc := range []struct {
		name, finish string
	}{
		{"separate finish", `null`},
		{"refusal with finish", `"content_filter"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Anthropic conversion emits a role first, then a refusal-only delta.
			// After the role has been sent an unrecognized delta would be dropped.
			source := `data: {"id":"refusal-fixture","choices":[{"index":0,"delta":{"role":"assistant"}}]}` + "\n\n" +
				`data: {"id":"refusal-fixture","choices":[{"index":0,"delta":{"refusal":"Request declined. (category: reasoning_extraction)"},"finish_reason":` + tc.finish + `}]}` + "\n\n"
			if tc.finish == "null" {
				source += `data: {"id":"refusal-fixture","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}` + "\n\n"
			}
			source += `data: {"id":"refusal-fixture","choices":[],"usage":{"prompt_tokens":17,"completion_tokens":0,"total_tokens":17}}` + "\n\ndata: [DONE]\n\n"
			output, err := io.ReadAll(New().Stream(Context{Endpoint: "/v1/chat/completions", IncludeUsage: true}, strings.NewReader(source)))
			require.NoError(t, err)
			frames := splitDataFrames(string(output))
			require.NotEmpty(t, frames)
			assert.Equal(t, "[DONE]", frames[len(frames)-1])
			refusals, finishes, usages := 0, 0, 0
			for _, frame := range frames[:len(frames)-1] {
				var body map[string]interface{}
				require.NoError(t, json.Unmarshal([]byte(frame), &body))
				if usage, ok := body["usage"].(map[string]interface{}); ok {
					usages++
					assert.Equal(t, float64(17), usage["prompt_tokens"])
					assert.Equal(t, float64(0), usage["completion_tokens"])
				}
				for _, raw := range body["choices"].([]interface{}) {
					choice := raw.(map[string]interface{})
					delta := choice["delta"].(map[string]interface{})
					if refusal, ok := delta["refusal"].(string); ok {
						refusals++
						assert.Equal(t, "Request declined. (category: reasoning_extraction)", refusal)
						assert.Nil(t, choice["finish_reason"], "content must precede its terminal chunk")
					}
					if choice["finish_reason"] == "content_filter" {
						finishes++
						assert.Equal(t, 1, refusals, "refusal must reach the client before finish")
					}
				}
			}
			assert.Equal(t, 1, refusals)
			assert.Equal(t, 1, finishes)
			assert.Equal(t, 1, usages)
		})
	}
}
