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

func TestDropNullParams(t *testing.T) {
	keys := []string{"max_tokens", "max_completion_tokens", "max_output_tokens"}

	t.Run("null limits are removed, the rest stays", func(t *testing.T) {
		out := decodeBody(t, DropNullParams([]byte(`{"model":"m","max_tokens":null,"max_completion_tokens":null,"max_output_tokens":null,"temperature":null}`), keys...))
		assert.NotContains(t, out, "max_tokens")
		assert.NotContains(t, out, "max_completion_tokens")
		assert.NotContains(t, out, "max_output_tokens")
		assert.Contains(t, out, "temperature", "only the listed keys are touched")
		assert.Equal(t, "m", out["model"])
	})

	t.Run("a numeric limit is kept", func(t *testing.T) {
		out := decodeBody(t, DropNullParams([]byte(`{"model":"m","max_tokens":128,"max_completion_tokens":null}`), keys...))
		assert.EqualValues(t, 128, out["max_tokens"])
		assert.NotContains(t, out, "max_completion_tokens")
	})

	t.Run("body returned unchanged when nothing is null", func(t *testing.T) {
		for _, body := range []string{
			`{"model":"m","max_tokens":128}`,
			`{"model":"m","note":"null"}`,
			`{"model":"m"}`,
			`not json null`,
		} {
			assert.Equal(t, body, string(DropNullParams([]byte(body), keys...)), body)
		}
	})
}
