package upstreamerror

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImageErrorClassification(t *testing.T) {
	for _, tt := range []struct {
		name, body, message, code, param string
	}{
		{"missing prompt", `{"error":{"message":"Missing required parameter 'prompt'.","param":"prompt"}}`, "Missing required parameter", "missing_required_parameter", "prompt"},
		{"prompt type", `{"error":{"message":"Invalid type for 'prompt'. Expected a string.","code":"invalid_type","param":"prompt"}}`, "Invalid parameter type", "invalid_type", "prompt"},
		{"size value", `{"error":{"message":"Invalid value for 'size'. See http://provider.internal","code":"invalid_value","param":"size"}}`, "Invalid parameter value", "invalid_value", "size"},
		{"image data", `{"error":{"message":"Invalid image data from credential private-key","code":"invalid_image"}}`, "Invalid image data", "invalid_image", "image"},
		{"Gemini image data", `{"error":{"code":400,"message":"Unable to process input image. Please retry.","status":"INVALID_ARGUMENT"}}`, "Invalid image data", "invalid_image", "image"},
		{"mask data", `{"error":{"message":"Could not decode image","param":"mask"}}`, "Invalid image data", "invalid_image", "mask"},
		{"image size", `{"error":{"message":"Invalid image size","param":"size","code":"invalid_image_size"}}`, "Invalid image size", "invalid_image_size", "size"},
		{"JSON", `{"error":{"message":"Invalid JSON","param":"image_config","code":"invalid_json"}}`, "Invalid JSON", "invalid_json", "image_config"},
		{"multipart", `{"error":{"message":"Invalid multipart form data","param":"image","code":"invalid_multipart"}}`, "Invalid multipart form data", "invalid_multipart", "image"},
		{"unknown detail", `{"error":{"message":"Unrecognized provider failure at http://provider.internal"}}`, "Invalid request", "invalid_request", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(tt.body)
			for range 3 {
				got := ClassifyBadRequest(body)
				assert.Equal(t, tt.message, got.Message)
				assert.Equal(t, tt.code, got.Code)
				if tt.param == "" {
					assert.Nil(t, got.Param)
				} else {
					require.NotNil(t, got.Param)
					assert.Equal(t, tt.param, *got.Param)
				}
				var err error
				body, err = json.Marshal(map[string]any{"error": map[string]any{"message": got.Message, "code": got.Code, "param": got.Param}})
				require.NoError(t, err)
				assert.NotContains(t, string(body), "provider.internal")
				assert.NotContains(t, string(body), "private-key")
			}
		})
	}
}
