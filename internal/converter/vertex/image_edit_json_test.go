package vertex

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/require"
)

func TestImageEditJSONValidation(t *testing.T) {
	for _, tt := range []struct {
		name, body, contentType string
		status                  int
	}{
		{"malformed JSON", `{"prompt":`, "application/json", 400},
		{"missing image", `{"prompt":"test"}`, "application/json", 400},
		{"invalid image type", `{"prompt":"test","image":12}`, "application/json", 400},
		{"empty image array", `{"prompt":"test","images":[]}`, "application/json", 400},
		{"invalid data URL", `{"prompt":"test","image":"data:image/png;base64,broken!"}`, "application/json", 400},
		{"unsupported file ID", `{"prompt":"test","images":[{"file_id":"file-test"}]}`, "application/json", 400},
		{"invalid mask", `{"prompt":"test","image":"data:image/png;base64,aW1n","mask":false}`, "application/json", 400},
		{"unsupported content type", `test`, "text/plain", 415},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := imageEditRequestToOpenAIChatRequest([]byte(tt.body), tt.contentType, "gemini-3.1-flash-image")
			var validation *converterutil.RequestValidationError
			require.ErrorAs(t, err, &validation)
			status := validation.StatusCode
			if status == 0 {
				status = 400
			}
			require.Equal(t, tt.status, status)
		})
	}
}

func TestImageEditJSONReusesGenerationOptions(t *testing.T) {
	body, err := imageEditRequestToOpenAIChatRequest([]byte(`{"model":"alias","prompt":"test","image":"https://example.com/image.png","size":"1792x2400","n":2,"seed":7,"temperature":0.5,"top_p":0.9}`), "application/json", "gemini-3.1-flash-image")
	require.NoError(t, err)
	var request openai.OpenAIRequest
	require.NoError(t, json.Unmarshal(body, &request))
	require.Equal(t, 2, *request.N)
	require.EqualValues(t, 7, *request.Seed)
	require.Equal(t, 0.5, *request.Temperature)
	require.Equal(t, 0.9, *request.TopP)
	generation := request.ExtraBody["generation_config"].(map[string]any)
	image := generation["image_config"].(map[string]any)
	require.Equal(t, "3:4", image["aspectRatio"])
	require.Equal(t, "2K", image["imageSize"])
}
