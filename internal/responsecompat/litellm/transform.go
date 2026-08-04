package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

var rateLimitHeaders = map[string]struct{}{
	"x-ratelimit-limit-requests":     {},
	"x-ratelimit-remaining-requests": {},
	"x-ratelimit-limit-tokens":       {},
	"x-ratelimit-remaining-tokens":   {},
}

func (t *Transformer) Transform(ctx Context, response Response) Response {
	response.Headers = t.TransformHeaders(ctx, response.Headers)
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		response.Body = normalizeError(response.StatusCode, response.Body)
		response.Headers.Set("Content-Type", "application/json")
		return response
	}

	var body map[string]any
	if err := json.Unmarshal(response.Body, &body); err != nil {
		return internalError(response.Headers, "Unable to get json response")
	}

	if meaningfulError, ok := body["error"]; ok && hasMeaningfulError(meaningfulError) {
		status := errorStatus(meaningfulError)
		return Response{
			StatusCode: status,
			Headers:    response.Headers,
			Body:       normalizeError(status, response.Body),
		}
	}

	var err error
	switch ctx.Endpoint {
	case "/v1/models":
		normalizeModels(body)
	case "/v1/embeddings":
		normalizeEmbedding(ctx, body)
	case "/v1/chat/completions":
		err = normalizeCompletion(ctx, body)
	case "/v1/completions":
		err = normalizeTextCompletion(ctx, body)
	default:
		overrideModel(ctx.RequestedModel, body)
	}
	if err != nil {
		return internalError(response.Headers, err.Error())
	}
	stripRoutingMetadata(body)

	response.Body, err = json.Marshal(stripNulls(body, false))
	if err != nil {
		return internalError(response.Headers, "Unable to serialize response")
	}
	response.Headers.Set("Content-Type", "application/json")
	return response
}

func normalizeEmbedding(ctx Context, body map[string]any) {
	overrideModel(ctx.RequestedModel, body)
	delete(body, "id")
	delete(body, "provider")
	if usage, ok := body["usage"].(map[string]any); ok {
		delete(usage, "cost")
		delete(usage, "cost_details")
		delete(usage, "is_byok")
		normalizeUsage(usage)
	}
}

func normalizeTextCompletion(ctx Context, body map[string]any) error {
	rawChoices, ok := body["choices"].([]any)
	if !ok || len(rawChoices) == 0 {
		return fmt.Errorf("LiteLLM: provider returned a response with no 'choices'")
	}

	body["object"] = "text_completion"
	overrideModel(ctx.RequestedModel, body)
	if id, _ := body["id"].(string); id == "" {
		body["id"] = "cmpl-" + uuid.NewString()
	}
	if body["created"] == nil {
		body["created"] = time.Now().Unix()
	}

	for index, rawChoice := range rawChoices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			return fmt.Errorf("LiteLLM: provider returned an invalid choice")
		}
		choice["index"] = index
		finishReason, _ := choice["finish_reason"].(string)
		if finishReason == "" {
			finishReason = "stop"
		}
		choice["finish_reason"] = mapFinishReason(finishReason)
	}
	if usage, ok := body["usage"].(map[string]any); ok {
		normalizeUsage(usage)
	}
	return nil
}

func (t *Transformer) TransformHeaders(ctx Context, source http.Header) http.Header {
	headers := make(http.Header)
	for key, values := range source {
		lowerKey := strings.ToLower(key)
		switch lowerKey {
		case "content-type":
			for _, value := range values {
				headers.Add(key, value)
			}
		case "retry-after":
			for _, value := range values {
				headers.Add(key, value)
			}
		}
		if _, ok := rateLimitHeaders[lowerKey]; ok {
			for _, value := range values {
				headers.Add(lowerKey, value)
			}
		}
	}
	return headers
}

func normalizeCompletion(ctx Context, body map[string]any) error {
	rawChoices, ok := body["choices"].([]any)
	if !ok || len(rawChoices) == 0 {
		return fmt.Errorf("LiteLLM: provider returned a response with no 'choices'")
	}

	body["object"] = "chat.completion"
	overrideModel(ctx.RequestedModel, body)
	if id, _ := body["id"].(string); id == "" {
		body["id"] = "chatcmpl-" + uuid.NewString()
	}
	if body["created"] == nil {
		body["created"] = time.Now().Unix()
	}

	for index, rawChoice := range rawChoices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			return fmt.Errorf("LiteLLM: provider returned an invalid choice")
		}
		choice["index"] = index

		message, ok := choice["message"].(map[string]any)
		if !ok {
			message = map[string]any{"content": nil}
			choice["message"] = message
		}
		if role, _ := message["role"].(string); role == "" {
			message["role"] = "assistant"
		}
		if _, ok := message["content"]; !ok {
			message["content"] = nil
		}
		moveProviderSpecificField(message, "refusal")
		moveProviderSpecificField(choice, "content_filter_results")
		if reasoning, ok := message["reasoning"]; ok {
			if _, exists := message["reasoning_content"]; !exists {
				message["reasoning_content"] = reasoning
			}
			delete(message, "reasoning")
		}

		finishReason, _ := choice["finish_reason"].(string)
		if finishReason == "" {
			finishReason, _ = choice["finish_details"].(string)
		}
		if finishReason == "" {
			finishReason = "stop"
		}
		if finishReason == "stop" && hasToolCalls(message) {
			finishReason = "tool_calls"
		}
		choice["finish_reason"] = mapFinishReason(finishReason)
	}

	if usage, ok := body["usage"].(map[string]any); ok {
		normalizeUsage(usage)
	}
	return nil
}

func normalizeModels(body map[string]any) {
	models, _ := body["data"].([]any)
	for _, rawModel := range models {
		if model, ok := rawModel.(map[string]any); ok {
			model["owned_by"] = "openai"
		}
	}
}

func moveProviderSpecificField(target map[string]any, field string) {
	value, exists := target[field]
	if !exists {
		return
	}
	providerFields, _ := target["provider_specific_fields"].(map[string]any)
	if providerFields == nil {
		providerFields = make(map[string]any)
		target["provider_specific_fields"] = providerFields
	}
	providerFields[field] = value
	delete(target, field)
}

func overrideModel(requestedModel string, body map[string]any) {
	if requestedModel != "" {
		if _, exists := body["model"]; exists {
			body["model"] = requestedModel
		}
	}
}

func normalizeUsage(usage map[string]any) {
	for _, field := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if usage[field] == nil {
			usage[field] = float64(0)
		}
	}
}

func hasToolCalls(message map[string]any) bool {
	toolCalls, ok := message["tool_calls"].([]any)
	return ok && len(toolCalls) > 0
}

func mapFinishReason(reason string) string {
	switch reason {
	case "length", "max_tokens", "max_output_tokens":
		return "length"
	case "tool_use", "function_call":
		return "tool_calls"
	case "content_filter", "content_filtered":
		return "content_filter"
	default:
		return reason
	}
}

func stripNulls(value any, preserveContent bool) any {
	switch typed := value.(type) {
	case map[string]any:
		cleaned := make(map[string]any, len(typed))
		for key, child := range typed {
			keepNull := (preserveContent && key == "content") || key == "refusal"
			if child == nil && !keepNull {
				continue
			}
			childPreservesContent := key == "message" || key == "delta"
			cleaned[key] = stripNulls(child, childPreservesContent)
		}
		return cleaned
	case []any:
		cleaned := make([]any, 0, len(typed))
		for _, child := range typed {
			cleaned = append(cleaned, stripNulls(child, preserveContent))
		}
		return cleaned
	default:
		return value
	}
}

func hasMeaningfulError(value any) bool {
	switch typed := value.(type) {
	case string:
		return typed != ""
	case map[string]any:
		message, _ := typed["message"].(string)
		return message != "" || typed["code"] != nil
	default:
		return value != nil
	}
}

func errorStatus(value any) int {
	if body, ok := value.(map[string]any); ok {
		if code, ok := body["code"].(float64); ok && code >= 400 && code <= 599 {
			return int(code)
		}
	}
	return http.StatusUnprocessableEntity
}

func normalizeError(status int, _ []byte) []byte {
	message := "Request failed"
	switch status {
	case http.StatusTooManyRequests:
		message = "Rate limit exceeded"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		message = "Request timed out"
	}

	result, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errorTypeForStatus(status),
			"param":   nil,
			"code":    fmt.Sprintf("%d", status),
		},
	})
	return result
}

func stripRoutingMetadata(value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if isRoutingMetadataField(key) {
				delete(typed, key)
				continue
			}
			stripRoutingMetadata(child)
		}
	case []any:
		for _, child := range typed {
			stripRoutingMetadata(child)
		}
	}
}

func isRoutingMetadataField(key string) bool {
	lower := strings.ToLower(key)
	if lower == "provider_specific_fields" {
		return false
	}
	switch lower {
	case "provider", "provider_id", "provider_name",
		"credential", "credential_id", "credential_name",
		"api_base", "base_url",
		"router", "router_id", "route", "route_id",
		"deployment", "deployment_id", "deployment_name",
		"fallback", "fallbacks", "fallback_route",
		"selected_provider", "selected_credential",
		"upstream", "upstream_url",
		"litellm_metadata", "litellm_params", "litellm_call_id",
		"llm_provider", "llm_provider_id":
		return true
	}
	return strings.HasPrefix(lower, "litellm_") ||
		strings.HasPrefix(lower, "provider_") ||
		strings.HasPrefix(lower, "credential_") ||
		strings.HasPrefix(lower, "router_") ||
		strings.HasPrefix(lower, "route_") ||
		strings.HasPrefix(lower, "deployment_") ||
		strings.HasPrefix(lower, "fallback_") ||
		strings.HasPrefix(lower, "llm_provider")
}

func errorTypeForStatus(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_denied"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return "timeout_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	default:
		return "api_error"
	}
}

func internalError(headers http.Header, message string) Response {
	headers.Set("Content-Type", "application/json")
	return Response{
		StatusCode: http.StatusInternalServerError,
		Headers:    headers,
		Body:       normalizeError(http.StatusInternalServerError, []byte(message)),
	}
}
