package vertex_transform

import (
	"encoding/json"
	"fmt"
)

// OpenAIImageRequest represents OpenAI image generation request
type OpenAIImageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              *int   `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Quality        string `json:"quality,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	Style          string `json:"style,omitempty"`
	User           string `json:"user,omitempty"`
}

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

// OpenAIImageResponse represents OpenAI image response
type OpenAIImageResponse struct {
	Created int64             `json:"created"`
	Data    []OpenAIImageData `json:"data"`
}

type OpenAIImageData struct {
	B64JSON       string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// VertexImageResponse represents Vertex AI Imagen response
type VertexImageResponse struct {
	Predictions []VertexImagePrediction `json:"predictions"`
}

type VertexImagePrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}

// OpenAIImageToVertex converts OpenAI image request to Vertex AI Imagen format
func OpenAIImageToVertex(openAIBody []byte) ([]byte, error) {
	var openAIReq OpenAIImageRequest
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

	// Set sample count
	sampleCount := 1
	if openAIReq.N != nil && *openAIReq.N > 0 {
		sampleCount = *openAIReq.N
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

	openAIResp := OpenAIImageResponse{
		Created: getCurrentTimestamp(),
		Data:    make([]OpenAIImageData, 0),
	}

	// Convert predictions to OpenAI format
	for _, prediction := range vertexResp.Predictions {
		data := OpenAIImageData{
			B64JSON: prediction.BytesBase64Encoded,
		}
		openAIResp.Data = append(openAIResp.Data, data)
	}

	return json.Marshal(openAIResp)
}
