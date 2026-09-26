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

	t.Run("client chat_template_kwargs wins on the keys it sent", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","chat_template_kwargs":{"enable_thinking":true}}`), defaults))
		assert.Equal(t, map[string]any{"enable_thinking": true}, got["chat_template_kwargs"])
	})

	t.Run("client values keep their exact bytes", func(t *testing.T) {
		out := string(ApplyDefaultParams([]byte(`{"model":"m","seed":12345678901234567890}`), defaults))
		assert.Contains(t, out, `"seed":12345678901234567890`)
	})

	t.Run("body is returned untouched when there is nothing to add", func(t *testing.T) {
		body := []byte(`{"model":"m", "temperature": 1, "top_k": 5, "chat_template_kwargs": {"enable_thinking": true}}`)
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

func TestApplyDefaultParams_MaxTokensSynonym(t *testing.T) {
	defaults := map[string]any{"max_tokens": json.Number("1024")}

	t.Run("filled when the client set neither spelling", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m"}`), defaults))
		assert.EqualValues(t, 1024, got["max_tokens"])
	})

	t.Run("not added next to the client's max_completion_tokens", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","max_completion_tokens":256}`), defaults))
		assert.EqualValues(t, 256, got["max_completion_tokens"])
		assert.NotContains(t, got, "max_tokens", "two spellings of one limit are ambiguous")
	})

	t.Run("the client's max_tokens wins", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","max_tokens":64}`), defaults))
		assert.EqualValues(t, 64, got["max_tokens"])
	})

	t.Run("a large seed default keeps every digit", func(t *testing.T) {
		out := string(ApplyDefaultParams([]byte(`{"model":"m"}`), map[string]any{"seed": json.Number("9007199254740993")}))
		assert.Contains(t, out, `"seed":9007199254740993`)
	})

	t.Run("a default keyed max_completion_tokens is not added next to the client's max_tokens", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams(
			[]byte(`{"model":"m","max_tokens":64}`),
			map[string]any{"max_completion_tokens": json.Number("1024")},
		))
		assert.EqualValues(t, 64, got["max_tokens"])
		assert.NotContains(t, got, "max_completion_tokens", "two spellings of one limit are ambiguous")
	})
}

func TestApplyDefaultParams_MergesObjects(t *testing.T) {
	defaults := map[string]any{
		"skip_special_tokens": false,
		"vllm_xargs":          map[string]any{"ngram_size": 35, "window_size": 128},
		"chat_template_kwargs": map[string]any{
			"reasoning_effort": "low",
			"nested":           map[string]any{"a": 1, "b": 2},
		},
	}

	t.Run("missing keys of a client object are filled from the default", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","vllm_xargs":{"window_size":1024}}`), defaults))
		assert.Equal(t, map[string]any{"ngram_size": 35.0, "window_size": 1024.0}, got["vllm_xargs"])
		assert.Equal(t, false, got["skip_special_tokens"])
	})

	t.Run("merge is recursive and the client wins at every depth", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams(
			[]byte(`{"model":"m","chat_template_kwargs":{"enable_thinking":true,"nested":{"b":20}}}`), defaults))
		assert.Equal(t, map[string]any{
			"enable_thinking":  true,
			"reasoning_effort": "low",
			"nested":           map[string]any{"a": 1.0, "b": 20.0},
		}, got["chat_template_kwargs"])
	})

	t.Run("an explicit null object opts out of the default", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","vllm_xargs":null}`), defaults))
		assert.Contains(t, got, "vllm_xargs")
		assert.Nil(t, got["vllm_xargs"])
	})

	t.Run("a non-object client value replaces the default object", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams([]byte(`{"model":"m","vllm_xargs":[1,2]}`), defaults))
		assert.Equal(t, []any{1.0, 2.0}, got["vllm_xargs"])
	})

	t.Run("a client object that already has every key is left byte-identical", func(t *testing.T) {
		body := []byte(`{"model":"m", "skip_special_tokens": true, "vllm_xargs": {"window_size": 1, "ngram_size": 2},` +
			` "chat_template_kwargs": {"reasoning_effort": "max", "nested": {"a": 0, "b": 0}}}`)
		assert.Equal(t, string(body), string(ApplyDefaultParams(body, defaults)))
	})

	t.Run("unrelated client values keep their exact bytes when an object is merged", func(t *testing.T) {
		out := string(ApplyDefaultParams([]byte(`{"model":"m","seed":12345678901234567890,"vllm_xargs":{}}`), defaults))
		assert.Contains(t, out, `"seed":12345678901234567890`)
	})

	t.Run("a max_tokens key inside an object is not treated as a request-level synonym", func(t *testing.T) {
		got := decodeBody(t, ApplyDefaultParams(
			[]byte(`{"model":"m","opts":{"max_completion_tokens":1}}`),
			map[string]any{"opts": map[string]any{"max_tokens": 5}},
		))
		assert.Equal(t, map[string]any{"max_completion_tokens": 1.0, "max_tokens": 5.0}, got["opts"])
	})
}

func TestUnwrapExtraBody(t *testing.T) {
	t.Run("keys are lifted and extra_body is dropped", func(t *testing.T) {
		got := decodeBody(t, UnwrapExtraBody([]byte(
			`{"model":"m","extra_body":{"skip_special_tokens":false,"vllm_xargs":{"ngram_size":35}}}`)))
		assert.Equal(t, false, got["skip_special_tokens"])
		assert.Equal(t, map[string]any{"ngram_size": 35.0}, got["vllm_xargs"])
		assert.NotContains(t, got, "extra_body")
	})

	t.Run("a top-level key wins over extra_body", func(t *testing.T) {
		got := decodeBody(t, UnwrapExtraBody([]byte(`{"model":"m","temperature":0.1,"extra_body":{"temperature":0.9,"top_k":5}}`)))
		assert.EqualValues(t, 0.1, got["temperature"])
		assert.EqualValues(t, 5, got["top_k"])
	})

	t.Run("bodies without an extra_body object are returned as is", func(t *testing.T) {
		for _, body := range []string{``, `not json`, `[1]`, `{"model":"m"}`, `{"model":"m", "extra_body": null}`, `{"extra_body":"x"}`} {
			assert.Equal(t, body, string(UnwrapExtraBody([]byte(body))), body)
		}
	})

	t.Run("lifted values keep their exact bytes", func(t *testing.T) {
		out := string(UnwrapExtraBody([]byte(`{"model":"m","extra_body":{"seed":12345678901234567890}}`)))
		assert.Contains(t, out, `"seed":12345678901234567890`)
	})
}
