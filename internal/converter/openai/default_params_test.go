package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func decodeBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var out map[string]any
	require.NoError(t, json.Unmarshal(body, &out))
	return out
}

func TestApplyDefaultParams(t *testing.T) {
	defaults := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
		"temperature":          0.7,
		"top_k":                20.0,
	}

	t.Run("fills only what the client left out", func(t *testing.T) {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.1}`)
		got := decodeBody(t, ApplyDefaultParams(body, defaults))
		assert.EqualValues(t, 0.1, got["temperature"], "the client's value wins")
		assert.EqualValues(t, 20, got["top_k"])
		assert.Equal(t, map[string]any{"enable_thinking": false}, got["chat_template_kwargs"])
		assert.Equal(t, "m", got["model"])
		assert.Len(t, got["messages"], 1)
	})

	t.Run("an explicit null from the client is respected", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","temperature":null}`), defaults))
		assert.Nil(t, got["temperature"])
		assert.Contains(t, got, "temperature")
	})

	t.Run("client chat_template_kwargs is not merged with the default", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","chat_template_kwargs":{"enable_thinking":true}}`), defaults))
		assert.Equal(t, map[string]any{"enable_thinking": true}, got["chat_template_kwargs"])
	})

	t.Run("client values keep their exact bytes", func(t *testing.T) {
		out := string(ApplyDefaultParams([]byte(`{"model":"m","seed":12345678901234567890}`), defaults))
		assert.Contains(t, out, `"seed":12345678901234567890`)
	})

	t.Run("body is returned untouched when there is nothing to add", func(t *testing.T) {
		body := []byte(`{"model":"m", "temperature": 1, "top_k": 5, "chat_template_kwargs": {}}`)
		assert.Equal(t, string(body), string(ApplyDefaultParams(body, defaults)))
	})

	t.Run("no defaults", func(t *testing.T) {
		body := []byte(`{"model":"m"}`)
		assert.Equal(t, string(body), string(ApplyDefaultParams(body, nil)))
		assert.Equal(t, string(body), string(ApplyDefaultParams(body, map[string]any{})))
	})

	t.Run("non-object bodies are left alone", func(t *testing.T) {
		for _, body := range []string{``, `not json`, `[1,2]`, `null`, `"text"`} {
			assert.Equal(t, body, string(ApplyDefaultParams([]byte(body), defaults)), body)
		}
	})
}
