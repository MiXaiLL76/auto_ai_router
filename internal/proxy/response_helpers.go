package proxy

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"strconv"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/converter"
)

type usageTotalTokens struct {
	TotalTokens int `json:"total_tokens"`
}

type openAIUsageResponse struct {
	// Chat Completions: usage at top level
	Usage usageTotalTokens `json:"usage"`
	// Responses API: usage nested inside response object (response.completed event)
	Response struct {
		Usage usageTotalTokens `json:"usage"`
	} `json:"response"`
}

func extractOpenAITotalTokens(payload []byte) int {
	var openAIResp openAIUsageResponse
	if err := json.Unmarshal(payload, &openAIResp); err != nil {
		return 0
	}

	if openAIResp.Usage.TotalTokens > 0 {
		return openAIResp.Usage.TotalTokens
	}
	return openAIResp.Response.Usage.TotalTokens
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

// extractTokenUsageFromStreamingChunk parses full TokenUsage (prompt+completion+details)
// from an SSE chunk. Returns nil if no usage data is found.
func extractTokenUsageFromStreamingChunk(chunk string) *converter.TokenUsage {
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}
			if usage := converter.ExtractTokenUsage([]byte(jsonData)); usage != nil {
				return usage
			}
		}
	}
	return nil
}

// extractMetadataFromBody extracts the model ID and session ID from the request body
// and ensures stream_options.include_usage is true for streaming requests
// Returns: model, streaming, sessionID, body
func extractMetadataFromBody(body []byte, contentType string) (string, bool, string, []byte) {
	// Check for empty body
	if len(body) == 0 {
		return "", false, "", body
	}

	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		model, sessionID := extractMetadataFromMultipartBody(body, contentType)
		return model, false, sessionID, body
	}

	// Parse JSON body
	var reqBody map[string]interface{}
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return "", false, "", body // Return original if parsing fails
	}

	model, ok := reqBody["model"].(string)
	if !ok {
		return "", false, "", body // Return original if model is missing
	}

	// Extract session ID (check extra_body first, then root level)
	// Priority: litellm_session_id > chat_id > session_id > user > safety_identifier > prompt_cache_key
	sessionID := ""
	if extraBody, ok := reqBody["extra_body"].(map[string]interface{}); ok {
		// Check litellm_session_id
		if sid, ok := extraBody["litellm_session_id"].(string); ok && sid != "" {
			sessionID = sid
		} else if cid, ok := extraBody["chat_id"].(string); ok && cid != "" {
			sessionID = cid
		} else if sid, ok := extraBody["session_id"].(string); ok && sid != "" {
			sessionID = sid
		}
	}
	// Check at root level if not found in extra_body
	if sessionID == "" {
		if sid, ok := reqBody["session_id"].(string); ok && sid != "" {
			sessionID = sid
		} else if uid, ok := reqBody["user"].(string); ok && uid != "" {
			sessionID = uid
		} else if sid, ok := reqBody["safety_identifier"].(string); ok && sid != "" {
			sessionID = sid
		} else if pck, ok := reqBody["prompt_cache_key"].(string); ok && pck != "" {
			sessionID = pck
		}
	}

	// Check if this is a streaming request
	stream, ok := reqBody["stream"].(bool)
	if !ok || !stream {
		return model, false, sessionID, body // Not a streaming request, return as-is
	}

	// Responses API (/v1/responses) uses "input" instead of "messages" and does NOT
	// support stream_options — it always returns usage in streaming.
	// Only inject stream_options for Chat Completions API requests.
	_, hasInput := reqBody["input"]
	_, hasMessages := reqBody["messages"]
	isResponsesAPI := hasInput && !hasMessages

	if !isResponsesAPI {
		// Ensure stream_options exists and include_usage is true (Chat Completions only)
		streamOptions, exists := reqBody["stream_options"]
		if !exists {
			reqBody["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}
		} else if streamOptionsMap, ok := streamOptions.(map[string]interface{}); ok {
			streamOptionsMap["include_usage"] = true
		} else {
			reqBody["stream_options"] = map[string]interface{}{
				"include_usage": true,
			}
		}
	}

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(reqBody)
	if err != nil {
		return model, stream, sessionID, body // Return original if marshaling fails
	}

	return model, stream, sessionID, modifiedBody
}

func extractMetadataFromMultipartBody(body []byte, contentType string) (string, string) {
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return "", ""
	}
	boundary := params["boundary"]
	if boundary == "" {
		return "", ""
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var model, sessionID string
	for {
		part, err := reader.NextPart()
		if err != nil {
			break
		}
		if part.FileName() != "" {
			continue
		}
		data, err := io.ReadAll(io.LimitReader(part, 1024*1024))
		if err != nil {
			continue
		}
		value := strings.TrimSpace(string(data))
		if value == "" {
			continue
		}
		switch part.FormName() {
		case "model":
			model = value
		case "session_id", "user":
			if sessionID == "" {
				sessionID = value
			}
		}
	}
	return model, sessionID
}

func extractImageCountFromBody(body []byte, contentType string) int {
	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		_, params, err := mime.ParseMediaType(contentType)
		if err != nil {
			return 1
		}
		boundary := params["boundary"]
		if boundary == "" {
			return 1
		}
		reader := multipart.NewReader(bytes.NewReader(body), boundary)
		for {
			part, err := reader.NextPart()
			if err != nil {
				break
			}
			if part.FileName() != "" || part.FormName() != "n" {
				continue
			}
			data, err := io.ReadAll(io.LimitReader(part, 64))
			if err != nil {
				break
			}
			n, err := strconv.Atoi(strings.TrimSpace(string(data)))
			if err == nil && n > 0 {
				return n
			}
			break
		}
		return 1
	}

	var imgReq struct {
		N *int `json:"n"`
	}
	if err := json.Unmarshal(body, &imgReq); err == nil && imgReq.N != nil && *imgReq.N > 0 {
		return *imgReq.N
	}
	return 1
}

// decodeResponseBody decodes the response body based on Content-Encoding
func decodeResponseBody(body []byte, encoding string) string {
	lowerEncoding := strings.ToLower(encoding)

	// Check if response is gzip-encoded
	if strings.Contains(lowerEncoding, "gzip") {
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

	// Check if response is deflate-encoded
	if strings.Contains(lowerEncoding, "deflate") {
		reader := flate.NewReader(bytes.NewReader(body))
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

func decodeResponseBodyPrefix(body []byte, encoding string, limit int64) []byte {
	if limit <= 0 {
		return nil
	}

	lowerEncoding := strings.ToLower(encoding)
	if strings.Contains(lowerEncoding, "gzip") {
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return truncateBytes(body, limit)
		}
		defer func() {
			_ = reader.Close()
		}()
		decoded, err := io.ReadAll(io.LimitReader(reader, limit))
		if err != nil {
			return truncateBytes(body, limit)
		}
		return decoded
	}

	if strings.Contains(lowerEncoding, "deflate") {
		reader := flate.NewReader(bytes.NewReader(body))
		defer func() {
			_ = reader.Close()
		}()
		decoded, err := io.ReadAll(io.LimitReader(reader, limit))
		if err != nil {
			return truncateBytes(body, limit)
		}
		return decoded
	}

	return truncateBytes(body, limit)
}

func truncateBytes(body []byte, limit int64) []byte {
	if int64(len(body)) <= limit {
		return body
	}
	return body[:limit]
}

// extractTokensFromResponse extracts total_tokens from the response body
// Supports both OpenAI format (usage.total_tokens) and Vertex AI format (usageMetadata.totalTokenCount)
func extractTokensFromResponse(body string, credType config.ProviderType) int {
	if body == "" {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == config.ProviderTypeVertexAI || credType == config.ProviderTypeGemini {
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

// injectStreamOptions ensures stream_options.include_usage is set in a Chat Completions request body.
// Used after Responses API conversion where extractMetadataFromBody skipped injection.
func injectStreamOptions(body []byte) []byte {
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}

	streamOptions, exists := raw["stream_options"]
	if !exists {
		raw["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	} else if soMap, ok := streamOptions.(map[string]interface{}); ok {
		soMap["include_usage"] = true
	} else {
		raw["stream_options"] = map[string]interface{}{
			"include_usage": true,
		}
	}

	modified, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return modified
}
