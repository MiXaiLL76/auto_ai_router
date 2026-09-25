package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStripServerToolCalls(t *testing.T) {
	t.Run("body without server tool calls is returned as is", func(t *testing.T) {
		body := []byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,` +
			`"tool_calls":[{"id":"toolu_01A","type":"function","function":{"name":"web_search","arguments":"{}"}}]}}]}`)
		assert.Equal(t, body, StripServerToolCalls(body))
	})

	t.Run("id prefix inside text alone changes nothing", func(t *testing.T) {
		body := []byte(`{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"see \"srvtoolu_x\""}}]}`)
		assert.Equal(t, body, StripServerToolCalls(body))
	})

	t.Run("provider counter wins over the stripped calls", func(t *testing.T) {
		body := []byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"ok",` +
			`"tool_calls":[{"id":"srvtoolu_01A","type":"function","function":{"name":"web_search","arguments":"{}"}}]}}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":2,"server_tool_use":{"web_search_requests":3}}}`)
		var out map[string]any
		require.NoError(t, json.Unmarshal(StripServerToolCalls(body), &out))
		usage := out["usage"].(map[string]any)
		assert.Equal(t, float64(3), usage["server_tool_use"].(map[string]any)["web_search_requests"])
		choice := out["choices"].([]any)[0].(map[string]any)
		assert.Equal(t, "stop", choice["finish_reason"])
	})

	t.Run("other server tools are stripped but not counted as searches", func(t *testing.T) {
		body := []byte(`{"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"ok",` +
			`"tool_calls":[{"id":"srvtoolu_01F","type":"function","function":{"name":"web_fetch","arguments":"{}"}}]}}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
		var out map[string]any
		require.NoError(t, json.Unmarshal(StripServerToolCalls(body), &out))
		message := out["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
		assert.NotContains(t, message, "tool_calls")
		assert.NotContains(t, out["usage"].(map[string]any), "server_tool_use")
	})

	t.Run("numbers and text survive the re-encode exactly", func(t *testing.T) {
		body := []byte(`{"created":1790265307,"choices":[{"index":0,"finish_reason":"tool_calls","logprobs":{"content":[{"logprob":-0.000012345678901234}]},` +
			`"message":{"role":"assistant","content":"a <b> & c","tool_calls":[{"id":"srvtoolu_01A","type":"function","function":{"name":"web_search","arguments":"{}"}}]}}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
		out := string(StripServerToolCalls(body))
		assert.Contains(t, out, `"created":1790265307`)
		assert.Contains(t, out, `"logprob":-0.000012345678901234`)
		assert.Contains(t, out, `"content":"a <b> & c"`)
		assert.Contains(t, out, `"server_tool_use":{"web_search_requests":1}`)
	})
}
