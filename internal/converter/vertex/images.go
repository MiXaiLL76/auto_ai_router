package vertex

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/mixaill76/auto_ai_router/internal/config"
	converterutil "github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"

	"google.golang.org/genai"
)

// VertexImageRequest represents Vertex AI Imagen request
type VertexImageRequest struct {
	Instances  []VertexImageInstance `json:"instances"`
	Parameters VertexImageParameters `json:"parameters"`
}

type VertexImageInstance struct {
	Prompt string `json:"prompt"`
}

type VertexImageParameters struct {
	SampleCount       int    `json:"sampleCount,omitempty"`
	AspectRatio       string `json:"aspectRatio,omitempty"`
	SafetyFilterLevel string `json:"safetyFilterLevel,omitempty"`
	PersonGeneration  string `json:"personGeneration,omitempty"`
}

// VertexImageResponse represents Vertex AI Imagen response
type VertexImageResponse struct {
	Predictions []VertexImagePrediction `json:"predictions"`
}

type VertexImagePrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}

// BuildVertexImageURL constructs the Vertex AI URL for image generation
// Format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
func BuildVertexImageURL(cred *config.CredentialConfig, modelID string) string {
	// For global location (no regional prefix)
	if cred.Location == "global" {
		return fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/global/publishers/google/models/%s:predict",
			cred.ProjectID, modelID,
		)
	}

	// For regional locations
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		cred.Location, cred.ProjectID, cred.Location, modelID,
	)
}

// OpenAIImageToVertex converts OpenAI image request to Vertex AI Imagen format
func OpenAIImageToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq openai.OpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &openAIReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI image request: %w", err)
	}

	// Convert size to aspect ratio
	aspectRatio := "1:1" // default
	switch openAIReq.Size {
	case "1024x1024", "512x512", "256x256":
		aspectRatio = "1:1"
	case "1792x1024":
		aspectRatio = "16:9"
	case "1024x1792":
		aspectRatio = "9:16"
	}

	// Set sample count (max 10 for image generation)
	sampleCount := 1
	if openAIReq.N != nil && *openAIReq.N > 0 {
		sampleCount = *openAIReq.N
		if sampleCount > 10 {
			sampleCount = 10
		}
	}

	// Handle quality and style (basic mapping)
	safetyLevel := "block_some"
	if openAIReq.Quality == "hd" {
		// For HD quality, we might want stricter safety
		safetyLevel = "block_few"
	}

	vertexReq := VertexImageRequest{
		Instances: []VertexImageInstance{
			{Prompt: openAIReq.Prompt},
		},
		Parameters: VertexImageParameters{
			SampleCount:       sampleCount,
			AspectRatio:       aspectRatio,
			SafetyFilterLevel: safetyLevel,
			PersonGeneration:  "allow_adult",
		},
	}

	return json.Marshal(vertexReq)
}

// VertexImageToOpenAI converts Vertex AI Imagen response to OpenAI format
func VertexImageToOpenAI(vertexBody []byte) ([]byte, error) {
	var vertexResp VertexImageResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex image response: %w", err)
	}

	openAIResp := openai.OpenAIImageResponse{
		Created: converterutil.GetCurrentTimestamp(),
		Data:    make([]openai.OpenAIImageData, 0),
	}

	// Convert predictions to OpenAI format
	for _, prediction := range vertexResp.Predictions {
		data := openai.OpenAIImageData{
			B64JSON: prediction.BytesBase64Encoded,
		}
		openAIResp.Data = append(openAIResp.Data, data)
	}

	return json.Marshal(openAIResp)
}

// ImageRequestToOpenAIChatRequest converts OpenAI image generation request to OpenAI chat request format
// This allows Gemini models to generate images through chat API with response_modalities: ["IMAGE"]
func ImageRequestToOpenAIChatRequest(openAIBody []byte) ([]byte, error) {
	var imageReq openai.OpenAIImageRequest
	if err := json.Unmarshal(openAIBody, &imageReq); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI image request: %w", err)
	}

	genConfig := map[string]interface{}{
		"response_modalities": []string{"IMAGE"},
	}

	// Add image config if size is provided
	if imageReq.Size != "" {
		imageConfig := map[string]interface{}{}

		// Convert OpenAI size format to Gemini aspect ratio
		// Supported by Gemini: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
		aspectRatio := sizeToAspectRatio(imageReq.Size)
		if aspectRatio != "" {
			imageConfig["aspectRatio"] = aspectRatio
		}

		// Convert size to Gemini image size (1K, 2K, 4K)
		imageSize := sizeToImageSize(imageReq.Size)
		if imageSize != "" {
			imageConfig["imageSize"] = imageSize
		}

		if len(imageConfig) > 0 {
			genConfig["image_config"] = imageConfig
		}
	}

	// Convert to OpenAI chat request format
	chatReq := openai.OpenAIRequest{
		Model: imageReq.Model,
		Messages: []openai.OpenAIMessage{
			{
				Role:    "user",
				Content: imageReq.Prompt,
			},
		},
		ExtraBody: map[string]interface{}{
			"generation_config": genConfig,
		},
	}

	return json.Marshal(chatReq)
}

// sizeToAspectRatio converts OpenAI size format (e.g., "1792x1024") to Gemini aspect ratio
// Supported by Gemini: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
func sizeToAspectRatio(size string) string {
	switch size {
	case "1024x1024", "512x512", "256x256":
		return "1:1"
	case "1792x1024":
		return "16:9"
	case "1024x1792":
		return "9:16"
	case "1536x1024":
		return "3:2"
	case "1024x1536":
		return "2:3"
	case "768x1024":
		return "3:4"
	case "1024x768":
		return "4:3"
	case "819x1024":
		return "4:5"
	case "1024x819":
		return "5:4"
	case "576x1024":
		return "9:16"
	case "2016x1008":
		return "21:9"
	default:
		// Default to 1:1 if size not recognized
		return "1:1"
	}
}

// sizeToImageSize converts OpenAI size to Gemini image size (1K, 2K, 4K)
// 1K ≈ 1024x1024, 2K ≈ 2048x2048, 4K ≈ 4096x4096
func sizeToImageSize(size string) string {
	switch size {
	// 1K sizes
	case "1024x1024", "512x512", "256x256", "1792x1024", "1024x1792", "1536x1024", "1024x1536", "768x1024", "1024x768", "819x1024", "1024x819", "576x1024":
		return "1K"
	// 2K sizes (larger variations)
	case "2048x2048", "3584x2048", "2048x3584":
		return "2K"
	// 4K sizes
	case "4096x4096", "7168x4096", "4096x7168":
		return "4K"
	default:
		return "1K" // Default to 1K
	}
}

// VertexChatResponseToOpenAIImage converts Vertex AI chat response with image to OpenAI image format
// Extracts inline image data from chat response and returns it in OpenAI image generation format
func VertexChatResponseToOpenAIImage(vertexBody []byte) ([]byte, error) {
	var vertexResp genai.GenerateContentResponse
	if err := json.Unmarshal(vertexBody, &vertexResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vertex chat response: %w", err)
	}

	openAIResp := openai.OpenAIImageResponse{
		Created: converterutil.GetCurrentTimestamp(),
		Data:    make([]openai.OpenAIImageData, 0),
	}

	// Extract images from candidates
	for _, candidate := range vertexResp.Candidates {
		if candidate.Content != nil && candidate.Content.Parts != nil {
			for _, part := range candidate.Content.Parts {
				// Extract inline data (image) from part
				if part.InlineData != nil {
					// Encode binary image data to base64
					b64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
					imageData := openai.OpenAIImageData{
						B64JSON: b64Data,
					}
					openAIResp.Data = append(openAIResp.Data, imageData)
				}
			}
		}
	}

	return json.Marshal(openAIResp)
}
