package upstreamerror

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClassifyBadRequest_ImageFetchErrors guards the round-trip that
// litellm compatibility mode depends on: the compat writer re-runs
// ClassifyBadRequest over the router's own already-classified body, so
// every image failure code must classify back into itself. Without this,
// a client on a litellm-shaped endpoint would see the generic "Invalid
// messages.content.image_url.url" instead of the actual reason.
func TestClassifyBadRequest_ImageFetchErrors(t *testing.T) {
	for _, tt := range []struct {
		code    string
		message string
	}{
		{"image_fetch_failed", "Failed to fetch image URL"},
		{"image_url_not_allowed", "Image URL is not allowed"},
		{"unsupported_image_media_type", "Unsupported image media type"},
		{"image_too_large", "Image is too large"},
	} {
		t.Run(tt.code, func(t *testing.T) {
			raw := []byte(`{"error":{"message":"` + tt.message +
				`","type":"invalid_request_error","param":"messages.content.image_url.url","code":"` + tt.code + `"}}`)

			classified := ClassifyBadRequest(raw)
			assert.Equal(t, tt.message, classified.Message)
			assert.Equal(t, tt.code, classified.Code)
			require.NotNil(t, classified.Param)
			assert.Equal(t, "messages.content.image_url.url", *classified.Param)

			// Idempotency: the router body must survive a second pass.
			routerBody, err := json.Marshal(map[string]any{
				"error": map[string]any{
					"message": classified.Message,
					"type":    "invalid_request_error",
					"param":   classified.Param,
					"code":    classified.Code,
				},
			})
			require.NoError(t, err)

			second := ClassifyBadRequest(routerBody)
			assert.Equal(t, classified.Message, second.Message,
				"message must survive a second classification pass")
			assert.Equal(t, classified.Code, second.Code,
				"code must survive a second classification pass")
			require.NotNil(t, second.Param)
			assert.Equal(t, *classified.Param, *second.Param,
				"param must survive a second classification pass")
		})
	}
}

// TestClassifyBadRequest_ImageErrorsCarryNoURL keeps the classifier from
// becoming a channel for the failed URL: whatever a provider says about an
// image, the classified output stays a fixed label.
//
// A provider body that merely borrows one of the router's codes but carries
// its own message is NOT accepted as a router body — the classifier falls
// back to the generic label rather than adopting the code. That is the
// property asserted here: the URL never reaches the output, whichever way
// the body is classified.
func TestClassifyBadRequest_ImageErrorsCarryNoURL(t *testing.T) {
	for _, raw := range [][]byte{
		[]byte(`{"error":{"message":"failed to fetch https://cdn.example.com/p.jpg?sig=SECRET","code":"image_fetch_failed"}}`),
		[]byte(`{"error":{"message":"could not load https://cdn.example.com/p.jpg?sig=SECRET","code":"image_fetch_failed"}}`),
		[]byte(`{"error":{"message":"Invalid image data at https://cdn.example.com/p.jpg?sig=SECRET","code":"invalid_image"}}`),
	} {
		classified := ClassifyBadRequest(raw)
		assert.NotContains(t, classified.Message, "SECRET")
		assert.NotContains(t, classified.Message, "cdn.example.com")
		assert.NotContains(t, classified.Message, "http")
	}
}
