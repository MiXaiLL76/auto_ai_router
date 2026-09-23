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

func TestAddsImageWatermarkModelScope(t *testing.T) {
	for model, want := range map[string]bool{
		"seedream-4-5-251128":          true,
		"seedream-5-0-260128":          true,
		"dola-seedream-5-0-pro-260628": true,
		"bytedance/seedream-4.5":       true,
		"seededit-3-0-i2i-250628":      true,
		"Seedream-4-0-250828":          true,
		"gpt-image-1":                  false,
		"grok-imagine-image":           false,
		"gemini-3-pro-image":           false,
		"my-seedream-clone":            false,
		"seedreamer":                   false,
	} {
		require.Equal(t, want, AddsImageWatermark(model), model)
	}
}

func TestDisableImageWatermark(t *testing.T) {
	for _, tt := range []struct {
		name, body, want string
	}{
		{"adds opt-out when absent", `{"model":"m","prompt":"p"}`, `{"model":"m","prompt":"p","watermark":false}`},
		{"overrides explicit true", `{"model":"m","watermark":true}`, `{"model":"m","watermark":false}`},
		{"overrides non-boolean value", `{"model":"m","watermark":"yes"}`, `{"model":"m","watermark":false}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			require.JSONEq(t, tt.want, string(DisableImageWatermark([]byte(tt.body))))
		})
	}

	t.Run("already disabled is returned byte-identical", func(t *testing.T) {
		body := `{"model":"m", "watermark": false}`
		require.Equal(t, body, string(DisableImageWatermark([]byte(body))))
	})

	t.Run("other fields are preserved exactly", func(t *testing.T) {
		body := DisableImageWatermark([]byte(`{"seed":9007199254740993,"image":["https://img.example/a.png"],"sequential_image_generation_options":{"max_images":3}}`))
		var fields map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(body, &fields))
		require.Equal(t, `9007199254740993`, string(fields["seed"]))
		require.JSONEq(t, `["https://img.example/a.png"]`, string(fields["image"]))
		require.JSONEq(t, `{"max_images":3}`, string(fields["sequential_image_generation_options"]))
		require.Equal(t, `false`, string(fields["watermark"]))
	})

	t.Run("non-object bodies are untouched", func(t *testing.T) {
		for _, body := range []string{``, `null`, `[1,2]`, `{"watermark":`, "--boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\np\r\n--boundary--"} {
			require.Equal(t, body, string(DisableImageWatermark([]byte(body))))
		}
	})
}
