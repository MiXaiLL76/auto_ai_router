package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
)

type openAIUsageResponse struct {
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func extractOpenAITotalTokens(payload []byte) int {
	var openAIResp openAIUsageResponse
	if err := json.Unmarshal(payload, &openAIResp); err != nil {
		return 0
	}

	return openAIResp.Usage.TotalTokens
}

func extractTokensFromStreamingChunk(chunk string) int {
	// Look for usage information in streaming chunks
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}

			tokens := extractOpenAITotalTokens([]byte(jsonData))
			if tokens > 0 {
				return tokens
			}
		}
	}
	return 0
}

// extractMetadataFromBody extracts the model ID from the request body
// and ensures stream_options.include_usage is true for streaming requests
func extractMetadataFromBody(body []byte) (string, bool, []byte) {
	// Check for empty body
	if len(body) == 0 {
		return "", false, body
	}

	// Parse JSON body
	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return "", false, body // Return original if parsing fails
	}

	model, ok := reqBody["model"].(string)
	if !ok {
		return "", false, body // Return original if model is missing or not streaming
	}

	// Check if this is a streaming request
	stream, ok := reqBody["stream"].(bool)
	if !ok || !stream {
		return model, false, body // Not a streaming request, return as-is
	}

	// Ensure stream_options exists and include_usage is true
	streamOptions, exists := reqBody["stream_options"]
	if !exists {
		// Create stream_options if it doesn't exist
		reqBody["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	} else if streamOptionsMap, ok := streamOptions.(map[string]interface{}); ok {
		// Update existing stream_options to ensure include_usage is true
		streamOptionsMap["include_usage"] = true
	} else {
		// stream_options exists but is not a map, replace it
		reqBody["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	}

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(reqBody)
	if err != nil {
		return model, stream, body // Return original if marshaling fails
	}

	return model, stream, modifiedBody
}

// decodeResponseBody decodes the response body based on Content-Encoding
func decodeResponseBody(body []byte, encoding string) string {
	// Check if response is gzip-encoded
	if strings.Contains(strings.ToLower(encoding), "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return string(body) // Return as-is if can't decode
		}
		defer func() {
			_ = reader.Close()
		}()

		decoded, err := io.ReadAll(reader)
		if err != nil {
			return string(body) // Return as-is if can't read
		}
		return string(decoded)
	}

	// Return as plain text
	return string(body)
}

// extractTokensFromResponse extracts total_tokens from the response body
// Supports both OpenAI format (usage.total_tokens) and Vertex AI format (usageMetadata.totalTokenCount)
func extractTokensFromResponse(body string, credType config.ProviderType) int {
	if body == "" {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == config.ProviderTypeVertexAI {
		var vertexResp struct {
			UsageMetadata struct {
				TotalTokenCount int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}

		if err := json.Unmarshal([]byte(body), &vertexResp); err != nil {
			return 0
		}
		return vertexResp.UsageMetadata.TotalTokenCount
	}

	// For OpenAI and other providers, use standard format
	return extractOpenAITotalTokens([]byte(body))
}

func maskKey(key string) string {
	if len(key) <= 7 {
		return "***"
	}
	return key[:7] + "..."
}
