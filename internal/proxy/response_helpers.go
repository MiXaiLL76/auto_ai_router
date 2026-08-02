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

	// Also used by request ingress sanitization below. RawMessage keeps large
	// numbers and provider-specific fields byte-for-byte while the request map
	// is inspected and selectively rebuilt.
	goccyjson "github.com/goccy/go-json"

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
	if err := goccyjson.Unmarshal(payload, &openAIResp); err != nil {
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
	return extractTokenUsageFromStreamingChunkWithOptions(chunk, converter.TokenUsageExtractionOptions{AudioInputIncludesCachedAudio: true})
}

func extractTokenUsageFromStreamingChunkWithOptions(chunk string, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if jsonData == "[DONE]" {
				continue
			}
			if usage := converter.ExtractTokenUsageWithOptions([]byte(jsonData), opts); usage != nil {
				return usage
			}
		}
	}
	return nil
}

// stripClientControlledServiceTier removes the two request locations through
// which clients may select a more expensive upstream service tier. It is
// intentionally not recursive: service_tier in metadata, messages, or tool
// schemas is user data and must be preserved.
func stripClientControlledServiceTier(reqBody map[string]goccyjson.RawMessage) (bool, error) {
	changed := false
	if _, exists := reqBody["service_tier"]; exists {
		delete(reqBody, "service_tier")
		changed = true
	}

	extraBodyRaw, exists := reqBody["extra_body"]
	if !exists {
		return changed, nil
	}
	trimmed := bytes.TrimSpace(extraBodyRaw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return changed, nil
	}

	var extraBody map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(extraBodyRaw, &extraBody); err != nil {
		return changed, err
	}
	if _, exists := extraBody["service_tier"]; !exists {
		return changed, nil
	}

	delete(extraBody, "service_tier")
	sanitizedExtraBody, err := goccyjson.Marshal(extraBody)
	if err != nil {
		return true, err
	}
	reqBody["extra_body"] = sanitizedExtraBody
	return true, nil
}

func rawString(raw goccyjson.RawMessage) string {
	var value string
	if err := goccyjson.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return value
}

func rawBool(raw goccyjson.RawMessage) bool {
	var value bool
	return goccyjson.Unmarshal(raw, &value) == nil && value
}

func extractSessionID(reqBody map[string]goccyjson.RawMessage) string {
	// Priority: litellm_session_id > chat_id > session_id > user >
	// safety_identifier > prompt_cache_key.
	if extraBodyRaw, exists := reqBody["extra_body"]; exists {
		var extraBody map[string]goccyjson.RawMessage
		if goccyjson.Unmarshal(extraBodyRaw, &extraBody) == nil {
			for _, key := range []string{"litellm_session_id", "chat_id", "session_id"} {
				if value := rawString(extraBody[key]); value != "" {
					return value
				}
			}
		}
	}
	for _, key := range []string{"session_id", "user", "safety_identifier", "prompt_cache_key"} {
		if value := rawString(reqBody[key]); value != "" {
			return value
		}
	}
	return ""
}

// extractMetadataFromBody extracts the model ID and session ID, removes
// client-controlled service_tier, and ensures stream_options.include_usage is
// true for streaming Chat Completions requests.
// Returns: model, streaming, sessionID, body, error.
func extractMetadataFromBody(body []byte, contentType string) (string, bool, string, []byte, error) {
	// Check for empty body
	if len(body) == 0 {
		return "", false, "", body, nil
	}

	if strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		model, sessionID := extractMetadataFromMultipartBody(body, contentType)
		return model, false, sessionID, body, nil
	}

	// Parse JSON body
	var reqBody map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(body, &reqBody); err != nil {
		return "", false, "", body, nil // Existing invalid-body handling reports missing model.
	}

	changed, err := stripClientControlledServiceTier(reqBody)
	if err != nil {
		return "", false, "", nil, err
	}

	model := rawString(reqBody["model"])
	if model == "" {
		return "", false, "", body, nil
	}
	sessionID := extractSessionID(reqBody)

	// Check if this is a streaming request
	stream := rawBool(reqBody["stream"])
	if !stream && !changed {
		return model, false, sessionID, body, nil
	}

	// Responses API (/v1/responses) uses "input" instead of "messages" and does NOT
	// support stream_options — it always returns usage in streaming.
	// Only inject stream_options for Chat Completions API requests.
	_, hasInput := reqBody["input"]
	_, hasMessages := reqBody["messages"]
	isResponsesAPI := hasInput && !hasMessages

	if stream && !isResponsesAPI {
		// Ensure stream_options exists and include_usage is true (Chat Completions only)
		streamOptions := make(map[string]goccyjson.RawMessage)
		streamOptionsRaw, exists := reqBody["stream_options"]
		if !exists || goccyjson.Unmarshal(streamOptionsRaw, &streamOptions) != nil || streamOptions == nil {
			streamOptions = make(map[string]goccyjson.RawMessage)
		}
		streamOptions["include_usage"] = goccyjson.RawMessage("true")
		marshaledStreamOptions, marshalErr := goccyjson.Marshal(streamOptions)
		if marshalErr != nil {
			return model, stream, sessionID, nil, marshalErr
		}
		reqBody["stream_options"] = marshaledStreamOptions
		changed = true
	}

	if !changed {
		return model, stream, sessionID, body, nil
	}
	modifiedBody, err := goccyjson.Marshal(reqBody)
	if err != nil {
		return model, stream, sessionID, nil, err
	}

	return model, stream, sessionID, modifiedBody, nil
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

func extractWebSearchRequestUsage(body []byte, contentType string) (bool, string) {
	if len(body) == 0 || strings.HasPrefix(strings.ToLower(contentType), "multipart/form-data") {
		return false, ""
	}

	var req struct {
		WebSearchOptions map[string]interface{}   `json:"web_search_options,omitempty"`
		Tools            []map[string]interface{} `json:"tools,omitempty"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return false, ""
	}

	if req.WebSearchOptions != nil {
		return true, webSearchContextSizeFromMap(req.WebSearchOptions)
	}
	for _, tool := range req.Tools {
		if !isWebSearchTool(tool) {
			continue
		}
		return true, webSearchContextSizeFromMap(tool)
	}
	return false, ""
}

func isWebSearchTool(tool map[string]interface{}) bool {
	toolType, _ := tool["type"].(string)
	return toolType == "web_search" || strings.HasPrefix(toolType, "web_search_")
}

func webSearchContextSizeFromMap(values map[string]interface{}) string {
	if values == nil {
		return converter.NormalizeWebSearchContextSize("")
	}
	size, _ := values["search_context_size"].(string)
	return converter.NormalizeWebSearchContextSize(size)
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
// Takes []byte (not string) because every caller already holds the response
// body as []byte — a string param would force a full copy in and back out
// for no reason, doubling the cost of scanning large (e.g. embedding-vector)
// bodies just to pull out a usage count.
func extractTokensFromResponse(body []byte, credType config.ProviderType) int {
	if len(body) == 0 {
		return 0
	}

	// For Vertex AI, use usageMetadata format
	if credType == config.ProviderTypeVertexAI || credType == config.ProviderTypeGemini {
		var vertexResp struct {
			UsageMetadata struct {
				TotalTokenCount int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}

		if err := goccyjson.Unmarshal(body, &vertexResp); err != nil {
			return 0
		}
		return vertexResp.UsageMetadata.TotalTokenCount
	}

	// For OpenAI and other providers, use standard format
	return extractOpenAITotalTokens(body)
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
