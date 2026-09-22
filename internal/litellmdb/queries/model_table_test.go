package queries

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultRequestParams(t *testing.T) {
	decode := func(t *testing.T, raw string) *GenericLiteLLMParams {
		t.Helper()
		var p GenericLiteLLMParams
		require.NoError(t, json.Unmarshal([]byte(raw), &p))
		return &p
	}

	t.Run("collects only the keys that are set", func(t *testing.T) {
		p := decode(t, `{"model":"m","temperature":0.7,"top_k":20,"frequency_penalty":0.5,"chat_template_kwargs":{"enable_thinking":false}}`)
		assert.Equal(t, map[string]any{
			"temperature":          0.7,
			"top_k":                20.0,
			"frequency_penalty":    0.5,
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		}, p.DefaultRequestParams())
	})

	t.Run("integer params keep their literal", func(t *testing.T) {
		p := decode(t, `{"max_tokens":4096,"seed":9007199254740993}`)
		got := p.DefaultRequestParams()
		encoded, err := json.Marshal(got)
		require.NoError(t, err)
		assert.JSONEq(t, `{"max_tokens":4096,"seed":9007199254740993}`, string(encoded))
		assert.Contains(t, string(encoded), "9007199254740993", "a float64 round trip would print 9007199254740992")
	})

	t.Run("a quoted integer is sent as a number", func(t *testing.T) {
		encoded, err := json.Marshal(decode(t, `{"max_tokens":"512"}`).DefaultRequestParams())
		require.NoError(t, err)
		assert.JSONEq(t, `{"max_tokens":512}`, string(encoded))
	})

	t.Run("nothing set", func(t *testing.T) {
		assert.Nil(t, decode(t, `{"model":"m"}`).DefaultRequestParams())
		var nilParams *GenericLiteLLMParams
		assert.Nil(t, nilParams.DefaultRequestParams())
	})
}
