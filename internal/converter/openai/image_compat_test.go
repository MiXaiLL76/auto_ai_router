package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageMiniJSONPreservesInvalidInput(t *testing.T) {
	for _, body := range []string{`{"quality":`, `null`, `{"quality":123}`, `{"quality":"custom"}`} {
		require.Equal(t, body, string(RewriteImageMiniJSON([]byte(body), "gpt-image-1-mini", false)))
	}
}

func TestImageMiniJSONModelScope(t *testing.T) {
	for _, model := range []string{"gpt-image-1-mini", "openai/gpt-image-1-mini", "gpt-image-1", "gpt-image-2", "gemini-3.1-flash-image"} {
		body := RewriteImageMiniJSON([]byte(`{"quality":"hd","input_fidelity":"high","extra":{"id":9007199254740993}}`), model, true)
		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(body, &fields))
		require.JSONEq(t, `{"id":9007199254740993}`, string(fields["extra"]))
		if model == "gpt-image-1-mini" || model == "openai/gpt-image-1-mini" {
			require.Equal(t, `"high"`, string(fields["quality"]))
			require.NotContains(t, fields, "input_fidelity")
		} else {
			require.Equal(t, `"hd"`, string(fields["quality"]))
			require.Contains(t, fields, "input_fidelity")
		}
	}
}
