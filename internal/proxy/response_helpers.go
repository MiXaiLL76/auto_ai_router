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

// extractMessagesTotalTokens reads Anthropic Messages API streaming usage
// (message_delta event) for the rate limiter's crude per-chunk token count.
// Kept as a fallback behind extractOpenAITotalTokens in extractTokensFromPayloads
// so the extra unmarshal only runs for the minority Anthropic-shaped traffic,
// not on every chunk of the majority Chat Completions/Responses/Vertex/Bedrock
// hot path.
func extractMessagesTotalTokens(payload []byte) int {
	var event struct {
		Type  string `json:"type"`
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &event); err != nil || event.Type != "message_delta" {
		return 0
	}
	return event.Usage.InputTokens +
		event.Usage.OutputTokens +
		event.Usage.CacheReadInputTokens +
		event.Usage.CacheCreationInputTokens
}

// extractOpenAITokensAndUsage runs both non-streaming token-accounting
// consumers off a single shared decode of body (converter.ExtractTotalTokensAndUsageWithOptions):
// the plain "total_tokens" count consumed by the rate limiter (RPM/TPM
// enforcement) and the full converter.TokenUsage consumed by spend
// logging/metrics. Replaces two independent full-body decodes
// (extractTokensFromResponse + converter.ExtractTokenUsageWithOptions) at
// each non-streaming response call site (proxy.go's proxy-credential and
// direct-provider branches, retry.go's fallback branch) with one — see
// ExtractTotalTokensAndUsageWithOptions's doc comment for why the two
// returned numbers are computed independently (not guaranteed equal) even
// though the decode is shared. Only wired in for the OpenAI-shaped
// (config.ProviderTypeOpenAI) path, matching all three call sites this
// replaces — none of them ever passed a Vertex/Gemini credType to
// extractTokensFromResponse.
func extractOpenAITokensAndUsage(body []byte, opts converter.TokenUsageExtractionOptions) (int, *converter.TokenUsage) {
	return converter.ExtractTotalTokensAndUsageWithOptions(body, opts)
}

// extractTokensFromStreamingChunk splits chunk via the shared zero-copy
// splitSSEPayloads (plan item C — one parse, no string(chunk) copy) and looks
// for usage information in the resulting payloads. Not on any hot per-chunk
// path itself (callers that already split the chunk once use
// extractTokensFromPayloads directly with the shared sub-slices instead), but
// kept as a convenience wrapper for callers that only have a raw chunk.
func extractTokensFromStreamingChunk(chunk []byte) int {
	return extractTokensFromPayloads(splitSSEPayloads(chunk, nil))
}

// extractTokensFromPayloads is the split-once building block behind
// extractTokensFromStreamingChunk — callers that already hold
// splitSSEPayloads' result (e.g. tokenCapturingWriter.Write) call this
// directly to avoid re-splitting the same chunk.
func extractTokensFromPayloads(payloads [][]byte) int {
	for _, payload := range payloads {
		if tokens := extractOpenAITotalTokens(payload); tokens > 0 {
			return tokens
		}
		if tokens := extractMessagesTotalTokens(payload); tokens > 0 {
			return tokens
		}
	}
	return 0
}

func extractTokenUsageFromStreamingChunkWithOptions(chunk []byte, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	return extractTokenUsageFromPayloads(splitSSEPayloads(chunk, nil), opts)
}

// extractTokenUsageFromPayloads is the split-once building block behind
// extractTokenUsageFromStreamingChunkWithOptions — callers that already hold
// splitSSEPayloads' result call this directly instead of re-splitting the
// same chunk (plan item C).
//
// Merges usage across every payload in the batch (via MergeNonZero) rather
// than returning the first non-nil hit: a single Read can surface multiple
// SSE frames (e.g. a web-search-only annotation frame followed by the final
// usage frame), and returning early on the first would silently drop the
// real prompt/completion tokens carried by a later frame in the same batch.
func extractTokenUsageFromPayloads(payloads [][]byte, opts converter.TokenUsageExtractionOptions) *converter.TokenUsage {
	var merged *converter.TokenUsage
	for _, payload := range payloads {
		if usage := converter.ExtractTokenUsageWithOptions(payload, opts); usage != nil {
			if merged == nil {
				merged = &converter.TokenUsage{}
			}
			merged.MergeNonZero(usage)
		}
	}
	return merged
}

// rawJSONString mirrors a `.(string)` type assertion on a decoded
// interface{}, but on a json.RawMessage sub-slice: it only succeeds if the
// raw value is actually a JSON string, matching the exists+ok semantics the
// old map[string]interface{} code relied on (e.g. a JSON null or number
// under the same key must NOT be treated as a present string).
func rawJSONString(raw goccyjson.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '"' {
		return "", false
	}
	var s string
	if err := goccyjson.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

// rawJSONBool is the bool counterpart of rawJSONString.
func rawJSONBool(raw goccyjson.RawMessage) (bool, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || (trimmed[0] != 't' && trimmed[0] != 'f') {
		return false, false
	}
	var v bool
	if err := goccyjson.Unmarshal(raw, &v); err != nil {
		return false, false
	}
	return v, true
}

// extractSessionIDFromRawBody mirrors the extra_body/root-level session ID
// lookup that used to run against a fully-decoded map[string]interface{}.
// extra_body is small, so decoding it as its own map[string]RawMessage is
// cheap; the large fields of reqBody (messages/input) are never touched.
func extractSessionIDFromRawBody(reqBody map[string]goccyjson.RawMessage) string {
	if extraRaw, ok := reqBody["extra_body"]; ok {
		var extraBody map[string]goccyjson.RawMessage
		if err := goccyjson.Unmarshal(extraRaw, &extraBody); err == nil {
			if sid, ok := rawJSONString(extraBody["litellm_session_id"]); ok && sid != "" {
				return sid
			}
			if cid, ok := rawJSONString(extraBody["chat_id"]); ok && cid != "" {
				return cid
			}
			if sid, ok := rawJSONString(extraBody["session_id"]); ok && sid != "" {
				return sid
			}
		}
	}
	if sid, ok := rawJSONString(reqBody["session_id"]); ok && sid != "" {
		return sid
	}
	if uid, ok := rawJSONString(reqBody["user"]); ok && uid != "" {
		return uid
	}
	if sid, ok := rawJSONString(reqBody["safety_identifier"]); ok && sid != "" {
		return sid
	}
	if pck, ok := rawJSONString(reqBody["prompt_cache_key"]); ok && pck != "" {
		return pck
	}
	return ""
}

var rawIncludeUsageTrue = goccyjson.RawMessage("true")

// injectIncludeUsageRaw ensures reqBody["stream_options"]["include_usage"] is
// true, operating entirely on RawMessage sub-slices so the rest of reqBody
// (in particular messages/input) is never boxed into interface{}.
func injectIncludeUsageRaw(reqBody map[string]goccyjson.RawMessage) error {
	streamOptionsRaw, exists := reqBody["stream_options"]
	var streamOptions map[string]goccyjson.RawMessage
	if exists {
		_ = goccyjson.Unmarshal(streamOptionsRaw, &streamOptions)
	}
	if streamOptions == nil {
		streamOptions = map[string]goccyjson.RawMessage{"include_usage": rawIncludeUsageTrue}
	} else {
		streamOptions["include_usage"] = rawIncludeUsageTrue
	}

	marshaled, err := goccyjson.Marshal(streamOptions)
	if err != nil {
		return err
	}
	reqBody["stream_options"] = marshaled
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

	// Parse JSON body — RawMessage sub-slices instead of interface{} boxing:
	// only six scalar top-level fields are ever read, so the large
	// messages/input field never needs to be decoded (see doc comment above).
	var reqBody map[string]goccyjson.RawMessage
	if err := goccyjson.Unmarshal(body, &reqBody); err != nil {
		return "", false, "", body, nil // Existing invalid-body handling reports missing model.
	}

	changed, err := stripClientControlledServiceTier(reqBody)
	if err != nil {
		return "", false, "", nil, err
	}

	model, ok := rawJSONString(reqBody["model"])
	if !ok {
		return "", false, "", body, nil
	}

	// Extract session ID (check extra_body first, then root level)
	// Priority: litellm_session_id > chat_id > session_id > user > safety_identifier > prompt_cache_key
	sessionID := extractSessionIDFromRawBody(reqBody)

	// Check if this is a streaming request
	stream, _ := rawJSONBool(reqBody["stream"])
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
		if err := injectIncludeUsageRaw(reqBody); err != nil {
			return model, stream, sessionID, nil, err
		}
		changed = true
	}

	if !changed {
		return model, stream, sessionID, body, nil
	}
	// Marshal back to JSON — untouched fields (messages/input, etc.) pass
	// through as raw bytes, no re-encoding cost.
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
