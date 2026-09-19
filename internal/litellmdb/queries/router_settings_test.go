package queries

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseRouterSettings(t *testing.T) {
	t.Run("alias and fallbacks as exported from the work database", func(t *testing.T) {
		raw := []byte(`{
			"timeout": 600.0,
			"fallbacks": [{"coder-ultra": ["qwen-ultra"]}, {"qwen-ultra": ["coder-ultra"]}, {"qwen-36-35b": ["coder-ultra"]}],
			"model_group_alias": {"gpt-oss": "gpt-oss-120b", "qwen-flash": "qwen-36-35b-fast", "qwen-ultra": "qwen3.5-397b-a17b-fp8"},
			"routing_strategy": "simple-shuffle"
		}`)
		got, err := ParseRouterSettings(raw)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{
			"gpt-oss":    "gpt-oss-120b",
			"qwen-flash": "qwen-36-35b-fast",
			"qwen-ultra": "qwen3.5-397b-a17b-fp8",
		}, got.ModelGroupAlias)
		assert.Equal(t, map[string][]string{
			"coder-ultra": {"qwen-ultra"},
			"qwen-ultra":  {"coder-ultra"},
			"qwen-36-35b": {"coder-ultra"},
		}, got.Fallbacks)
	})

	t.Run("alias may be an object", func(t *testing.T) {
		got, err := ParseRouterSettings([]byte(`{"model_group_alias": {"a": {"model": "b", "hidden": true}, "c": {"hidden": true}}}`))
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"a": "b"}, got.ModelGroupAlias)
	})

	t.Run("unusable entries are skipped instead of failing the sync", func(t *testing.T) {
		got, err := ParseRouterSettings([]byte(`{"model_group_alias": {"": "x", "n": 5, "ok": " target "}, "fallbacks": [{"g": [1, "f", ""]}]}`))
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"ok": "target"}, got.ModelGroupAlias)
		assert.Equal(t, map[string][]string{"g": {"f"}}, got.Fallbacks)
	})

	t.Run("empty and null", func(t *testing.T) {
		for _, raw := range [][]byte{nil, []byte(""), []byte("null"), []byte("{}")} {
			got, err := ParseRouterSettings(raw)
			require.NoError(t, err)
			assert.Empty(t, got.ModelGroupAlias)
			assert.Empty(t, got.Fallbacks)
		}
	})

	t.Run("malformed json is an error", func(t *testing.T) {
		_, err := ParseRouterSettings([]byte(`{"model_group_alias": `))
		require.Error(t, err)
	})
}

func TestGenericLiteLLMParams_DefaultRequestParams(t *testing.T) {
	var params GenericLiteLLMParams
	// The exact litellm_params of the work deployment "qwen-36-35b-fast".
	require.NoError(t, json.Unmarshal([]byte(`{
		"chat_template_kwargs": {"enable_thinking": false},
		"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
		"presence_penalty": 1.5, "repetition_penalty": 1,
		"input_cost_per_token": 2e-7, "use_in_pass_through": false
	}`), &params))

	got := params.DefaultRequestParams()
	assert.Equal(t, map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
		"temperature":          0.7,
		"top_p":                0.8,
		"top_k":                20.0,
		"min_p":                0.0, // an explicit zero is a real default, not "unset"
		"presence_penalty":     1.5,
		"repetition_penalty":   1.0,
	}, got)

	body, err := json.Marshal(got)
	require.NoError(t, err)
	assert.JSONEq(t, `{"chat_template_kwargs":{"enable_thinking":false},"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0,"presence_penalty":1.5,"repetition_penalty":1}`, string(body))
}

func TestGenericLiteLLMParams_DefaultRequestParams_None(t *testing.T) {
	var params GenericLiteLLMParams
	require.NoError(t, json.Unmarshal([]byte(`{"input_cost_per_token": 1e-8}`), &params))
	assert.Nil(t, params.DefaultRequestParams())
	assert.Nil(t, (*GenericLiteLLMParams)(nil).DefaultRequestParams())
}

func TestGenericLiteLLMParams_EffectiveCredentialName(t *testing.T) {
	str := func(s string) *string { return &s }

	assert.Equal(t, "", (*GenericLiteLLMParams)(nil).EffectiveCredentialName())
	assert.Equal(t, "", (&GenericLiteLLMParams{}).EffectiveCredentialName())
	// LiteLLM's real key wins over the legacy spelling.
	assert.Equal(t, "real", (&GenericLiteLLMParams{LiteLLMCredentialName: str("real"), CredentialName: str("legacy")}).EffectiveCredentialName())
	assert.Equal(t, "legacy", (&GenericLiteLLMParams{CredentialName: str("legacy")}).EffectiveCredentialName())

	var parsed GenericLiteLLMParams
	require.NoError(t, json.Unmarshal([]byte(`{"litellm_credential_name": "ray-service-prod"}`), &parsed))
	assert.Equal(t, "ray-service-prod", parsed.EffectiveCredentialName())
}
