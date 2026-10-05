package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Edge cases of which assistant turns keep reasoning_content when a Responses
// input is converted to Chat messages: only those after the last user message.
func TestRequestToChat_ReasoningScopeEdgeCases(t *testing.T) {
	t.Run("no user message keeps reasoning on every turn", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R1"}]},
			{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "c1", "output": "1"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R2"}]},
			{"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "c2", "output": "2"}
		]}`)
		require.Len(t, messages, 4)
		assert.Equal(t, "R1", messages[0]["reasoning_content"])
		assert.Equal(t, "R2", messages[2]["reasoning_content"])
	})

	t.Run("system and developer messages are not a turn boundary", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R1"}]},
			{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
			{"type": "function_call_output", "call_id": "c1", "output": "1"},
			{"role": "developer", "content": "be brief"},
			{"role": "system", "content": "sys"}
		]}`)
		require.Len(t, messages, 5)
		assert.Equal(t, "R1", messages[1]["reasoning_content"])
	})

	t.Run("reasoning before an unconverted item leaves no empty assistant", func(t *testing.T) {
		// computer_call has no Chat equivalent and is skipped; the reasoning
		// that led to it must not survive as a reasoning-only assistant message.
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R1"}]},
			{"type": "computer_call", "call_id": "cc1", "action": {"type": "screenshot"}},
			{"type": "computer_call_output", "call_id": "cc1", "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"}}
		]}`)
		for _, m := range messages {
			assert.NotEqual(t, "assistant", m["role"])
			assert.NotContains(t, m, "reasoning_content")
		}
	})

	t.Run("several past turns all lose reasoning", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q1"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R1"}]},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A1"}]},
			{"role": "user", "content": "q2"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "R2"}]},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A2"}]},
			{"role": "user", "content": "q3"}
		]}`)
		require.Len(t, messages, 5)
		raw, err := json.Marshal(messages)
		require.NoError(t, err)
		assert.NotContains(t, string(raw), "reasoning_content")
		assert.NotContains(t, string(raw), "R1")
		assert.NotContains(t, string(raw), "R2")
	})

	t.Run("trailing reasoning-only turn after last user is dropped", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "cut off"}]}
		]}`)
		require.Len(t, messages, 1)
		assert.Equal(t, "user", messages[0]["role"])
	})

	t.Run("reasoning-only turn between messages leaves no empty assistant", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q1"},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "cut off"}]},
			{"role": "user", "content": "q2"}
		]}`)
		require.Len(t, messages, 2)
		for _, m := range messages {
			assert.Equal(t, "user", m["role"])
		}
	})

	t.Run("reasoning after an assistant message stays with that turn", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "A"}]},
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "after"}]}
		]}`)
		require.Len(t, messages, 2)
		assert.Equal(t, "after", messages[1]["reasoning_content"])
	})

	t.Run("reasoning with only encrypted_content adds nothing", func(t *testing.T) {
		messages := chatMessages(t, `{"model": "m", "input": [
			{"role": "user", "content": "q"},
			{"type": "reasoning", "encrypted_content": "enc"},
			{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
		]}`)
		require.Len(t, messages, 2)
		assert.NotContains(t, messages[1], "reasoning_content")
		assert.Len(t, messages[1]["tool_calls"], 1)
	})
}
