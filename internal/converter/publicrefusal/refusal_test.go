package publicrefusal

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPublicRefusalDoesNotExposeProviderText(t *testing.T) {
	for _, category := range []string{"reasoning_extraction", "frontier_llm", "cyber", "bio", "general_harms", "private-key-in-category", ""} {
		t.Run(category, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"stop_reason": "refusal", "content": []any{}, "usage": map[string]any{"input_tokens": 17},
				"stop_details": map[string]any{"category": category, "explanation": "private-key-in-explanation",
					"fallback_credit_token": "private-credit", "nested": map[string]any{"api_key": "private-key"}},
			})
			require.NoError(t, err)
			clean, changed, err := SanitizeJSON(body)
			require.NoError(t, err)
			assert.True(t, changed)
			assert.NotContains(t, string(clean), "private-")
			var out map[string]any
			require.NoError(t, json.Unmarshal(clean, &out))
			assert.Equal(t, "refusal", out["stop_reason"])
			details := out["stop_details"].(map[string]any)
			assert.Len(t, details, 3)
			assert.NotEmpty(t, details["explanation"])
			if category == "reasoning_extraction" {
				assert.Equal(t, "The safety classifier identified a request to reproduce internal reasoning.", details["explanation"])
			}
			if category == "" || category == "private-key-in-category" {
				assert.Nil(t, details["category"])
				assert.Equal(t, Unknown, details["explanation"])
			}
			second, changed, err := SanitizeJSON(clean)
			require.NoError(t, err)
			assert.False(t, changed, "safe refusal details must not trigger another rewrite")
			assert.Equal(t, clean, second)
		})
	}
}

func TestPublicRefusalLeavesModelContentUntouched(t *testing.T) {
	body := []byte(`{"stop_reason":"end_turn","content":[{"type":"tool_use","input":{"stop_reason":"refusal","stop_details":{"explanation":"user data"}}},{"type":"thinking","signature":"signed"}],"usage":{"input_tokens":9007199254740993}}`)
	clean, changed, err := SanitizeJSON(body)
	require.NoError(t, err)
	assert.False(t, changed)
	assert.Equal(t, body, clean)
	for _, malformed := range []string{`{"stop_reason":"refusal"} secret`, `{"stop_details":`, `{}` + `{}`, `{} null`} {
		_, _, err := SanitizeJSON([]byte(malformed))
		assert.Error(t, err)
	}
}

func TestPublicRefusalRejectsUnexpectedDetailTypes(t *testing.T) {
	for _, details := range []string{
		`{"type":{},"category":[],"explanation":[]}`,
		`{"type":"refusal","category":"cyber","explanation":{}}`,
		`{"type":"refusal","category":null,"explanation":null}`,
	} {
		clean, changed, err := SanitizeJSON([]byte(`{"stop_reason":"refusal","stop_details":` + details + `}`))
		require.NoError(t, err)
		assert.True(t, changed)
		assert.NotContains(t, string(clean), `"explanation":{}`)
		var obj map[string]any
		require.NoError(t, json.Unmarshal(clean, &obj))
		assert.NotEmpty(t, obj["stop_details"].(map[string]any)["explanation"])
	}
}
