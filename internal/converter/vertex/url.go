package vertex

import (
	"fmt"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

// determineVertexPublisher determines the Vertex AI publisher based on the model ID
func determineVertexPublisher(modelID string) string {
	modelLower := strings.ToLower(modelID)
	if strings.Contains(modelLower, "claude") {
		return "anthropic"
	}
	// Default to Google for Gemini and other models
	return "google"
}

// BuildGeminiURL constructs the Google AI Studio (Gemini) URL.
// Format: {base_url}/v1beta/models/{model}:generateContent (or streamGenerateContent?alt=sse)
func BuildGeminiURL(cred *config.CredentialConfig, modelID string, streaming bool) string {
	baseURL := strings.TrimSuffix(cred.BaseURL, "/")

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}

	return fmt.Sprintf("%s/v1beta/models/%s:%s", baseURL, modelID, endpoint)
}

// BuildVertexURL constructs the Vertex AI URL dynamically
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{endpoint}
func BuildVertexURL(cred *config.CredentialConfig, modelID string, streaming bool) string {
	publisher := determineVertexPublisher(modelID)

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}

	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/%s/models/%s:%s",
			cred.ProjectID, publisher, modelID, endpoint,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/%s/models/%s:%s",
		cred.Location, cred.ProjectID, cred.Location, publisher, modelID, endpoint,
	)
}
