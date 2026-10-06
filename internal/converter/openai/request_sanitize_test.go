package openai

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDropEmptyTools(t *testing.T) {
	out := decodeBody(t, DropEmptyTools([]byte(`{"model":"m","tools":[],"tool_choice":"auto","parallel_tool_calls":false,"messages":[]}`)))
	assert.NotContains(t, out, "tools")
	assert.NotContains(t, out, "tool_choice")
	assert.NotContains(t, out, "parallel_tool_calls")
	assert.Contains(t, out, "messages")

	for _, body := range []string{
		`{"model":"m","tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":"auto"}`,
		`{"model":"m","tools":null,"tool_choice":"none"}`,
		`{"model":"m","tool_choice":"none"}`,
		`{"model":"m","tools":{}}`,
		`not json "tools"`,
	} {
		assert.Equal(t, body, string(DropEmptyTools([]byte(body))), body)
	}
}

func TestMapReasoningEffort(t *testing.T) {
	resolve := func(v string) (string, bool) {
		if v == "minimal" {
			return "low", true
		}
		return v, false
	}

	out := decodeBody(t, MapReasoningEffort([]byte(
		`{"reasoning_effort":"minimal","chat_template_kwargs":{"reasoning_effort":"minimal","enable_thinking":true},"reasoning":{"effort":"minimal","summary":"auto"}}`,
	), resolve))
	assert.Equal(t, "low", out["reasoning_effort"])
	assert.Equal(t, map[string]any{"reasoning_effort": "low", "enable_thinking": true}, out["chat_template_kwargs"])
	assert.Equal(t, map[string]any{"effort": "low", "summary": "auto"}, out["reasoning"])

	for _, body := range []string{
		`{"reasoning_effort":"medium"}`,
		`{"reasoning_effort":null}`,
		`{"reasoning":{"effort":3}}`,
		`{"messages":[{"content":"minimal effort"}]}`,
		`{"model":"m"}`,
	} {
		assert.Equal(t, body, string(MapReasoningEffort([]byte(body), resolve)), body)
	}
	assert.True(t, strings.Contains(string(MapReasoningEffort([]byte(`{"reasoning_effort":"minimal"}`), nil)), "minimal"))
}
