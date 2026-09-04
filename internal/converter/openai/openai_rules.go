package openai

import (
	"bytes"
	"encoding/json"
	"strings"
)

// ModelParamsMapping defines parameter transformations for a model family.
type ModelParamsMapping struct {
	// KeysToReplace maps old parameter names to new ones (e.g., "max_tokens" → "max_completion_tokens").
	// Replacement is skipped if the new key already exists in the request body.
	KeysToReplace map[string]string
	// KeysToRemove lists parameters to strip from the request body.
	KeysToRemove []string
}

// UpdateJSONField applies parameter transformations (rename + remove) to a JSON body.
func UpdateJSONField(body []byte, mapping ModelParamsMapping) []byte {
	var data map[string]any

	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}

	// 1. Replace keys (rename parameters)
	for oldKey, newKey := range mapping.KeysToReplace {
		if val, ok := data[oldKey]; ok {
			// Only replace if the target key is NOT already present.
			// This prevents overwriting an explicitly set max_completion_tokens
			// when max_tokens is also provided.
			if _, exists := data[newKey]; !exists {
				data[newKey] = val
			}
			delete(data, oldKey)
		}
	}

	// 2. Remove unsupported keys
	for _, key := range mapping.KeysToRemove {
		delete(data, key)
	}

	// 3. Marshal back
	updatedBody, err := json.Marshal(data)
	if err != nil {
		return body
	}

	return updatedBody
}

// ReplaceModelInBody replaces the "model" field value in a JSON body.
// Uses byte-level replacement of `"model":"oldValue"` to avoid full re-serialization.
func ReplaceModelInBody(body []byte, oldModel, newModel string) []byte {
	oldToken, _ := json.Marshal(oldModel) //nolint:errcheck // json.Marshal on a plain string never fails //
	newToken, _ := json.Marshal(newModel) //nolint:errcheck // json.Marshal on a plain string never fails //

	// Replace "model":"oldModel" → "model":"newModel"
	// Handles both with and without spaces after colon
	patterns := [][]byte{
		append([]byte(`"model":`), oldToken...),
		append([]byte(`"model": `), oldToken...),
	}
	replacements := [][]byte{
		append([]byte(`"model":`), newToken...),
		append([]byte(`"model": `), newToken...),
	}

	for i, pattern := range patterns {
		if bytes.Contains(body, pattern) {
			return bytes.Replace(body, pattern, replacements[i], 1)
		}
	}

	return body
}

// --- Model family parameter mappings ---

// o1Mapping: o1, o1-mini, o1-preview, o1-pro
// These reasoning models reject temperature, top_p, penalties, and logprobs.
var o1Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
		"frequency_penalty",
		"presence_penalty",
		"logprobs",
		"top_logprobs",
	},
}

// o3Mapping: o3, o3-mini, o3-pro
// Reasoning models that support reasoning_effort but reject temperature/top_p/penalties/logprobs.
var o3Mapping = ModelParamsMapping{ //  — added frequency_penalty, presence_penalty, logprobs, top_logprobs
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
		"frequency_penalty",
		"presence_penalty",
		"logprobs",
		"top_logprobs",
	},
}

// o4Mapping: o4-mini and future o4 models.
// Reasoning models that reject sampling parameters
var o4Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
		"frequency_penalty",
		"presence_penalty",
		"logprobs",
		"top_logprobs",
	},
}

// gpt5Mapping: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1, gpt-5.2, etc.
// Reasoning models that reject sampling parameters. //
var gpt5Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
		"frequency_penalty",
		"presence_penalty",
		"logprobs",
		"top_logprobs",
	},
}

// modelMappings maps model family prefixes to their parameter transformations.
// Order matters: longer prefixes are checked first via matchModelFamily.
var modelMappings = []struct {
	prefix  string
	mapping ModelParamsMapping
}{
	{"o1", o1Mapping},
	{"o3", o3Mapping},
	{"o4", o4Mapping},
	{"gpt-5", gpt5Mapping},
}

// extractBaseModelName strips provider prefixes and known suffixes from a model ID.
// Examples:
//
//	"openai/gpt-5"      → "gpt-5"
//	"openai:gpt-5"      → "gpt-5"
//	"gpt-5_chat"        → "gpt-5"
//	"gpt-5-chat"        → "gpt-5"
//	"provider/o3-mini"   → "o3-mini"
//	"gpt-4o"            → "gpt-4o"
func extractBaseModelName(modelID string) string {
	// Strip provider prefix: "openai/gpt-5" → "gpt-5", "vertex/o3" → "o3"
	if idx := strings.LastIndex(modelID, "/"); idx >= 0 {
		modelID = modelID[idx+1:]
	}

	// Strip provider prefix with colon: "openai:gpt-5" → "gpt-5"
	if idx := strings.LastIndex(modelID, ":"); idx >= 0 {
		modelID = modelID[idx+1:]
	}

	// Strip known suffixes: "_chat", "-chat"
	modelID = strings.TrimSuffix(modelID, "_chat")
	modelID = strings.TrimSuffix(modelID, "-chat")

	return strings.ToLower(modelID)
}

// matchModelFamily checks if modelID belongs to a given model family.
// Strips provider prefixes and suffixes before matching.
// Matches: exact name ("o1"), or name followed by "-" or "." ("o1-mini", "gpt-5.1").
func matchModelFamily(modelID, family string) bool {
	base := extractBaseModelName(modelID)
	if base == family {
		return true
	}
	return strings.HasPrefix(base, family+"-") || strings.HasPrefix(base, family+".")
}

// ReplaceBodyParam applies model-specific parameter transformations to the request body.
// This ensures unsupported parameters are removed and renamed before sending to the provider.
func ReplaceBodyParam(modelID string, body []byte) []byte {
	for _, m := range modelMappings {
		if matchModelFamily(modelID, m.prefix) {
			return UpdateJSONField(body, m.mapping)
		}
	}
	return body
}

// NormalizeDeveloperRole downgrades a "developer"-role Chat Completions message
// to "system" — the same rename the Responses→Chat converter already applies
// (see converter/responses.convertMessage), but for requests that arrive
// already Chat-Completions-shaped and never go through that converter (a
// client calling /v1/chat/completions directly, or one of the OpenAI SDKs
// that emits "developer" for reasoning models). "developer" is OpenAI's own
// rename of "system"; most Chat Completions providers reached through this
// generic path (DeepSeek and other OpenAI-compatible backends) only
// recognize the classic role set and reject "developer" outright. Returns
// body unchanged if there's no "messages" array or nothing to rename.
func NormalizeDeveloperRole(body []byte) []byte {
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}

	messagesRaw, ok := data["messages"]
	if !ok {
		return body
	}
	messages, ok := messagesRaw.([]any)
	if !ok {
		return body
	}

	changed := false
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if role, ok := msg["role"].(string); ok && role == "developer" {
			msg["role"] = "system"
			changed = true
		}
	}
	if !changed {
		return body
	}

	result, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return result
}

// ConvertWebSearchTools normalises non-function tools in an OpenAI Chat
// Completions request body.
//
//   - For OpenAI search-preview models, web_search / web_search_preview are
//     converted to the legacy top-level web_search_options parameter
//     (supported options copied from the built-in tool).
//   - For every other model, web_search / web_search_preview tools and any
//     tool_choice referencing them are passed through unchanged. The router
//     must not silently remove a requested capability; the selected upstream
//     owns validation of its Chat Completions tool contract.
//   - All other non-function tools (computer_use, google_search_retrieval,
//     code_execution, etc.) are dropped for every vendor; they have no
//     Chat Completions equivalent and would cause a 400 upstream.
//   - If a non-function tool_choice remains after tools are filtered, it is
//     also removed so the provider defaults to "auto" (unless it references
//     a preserved web_search tool, see above).
func ConvertWebSearchTools(body []byte) []byte {
	var data map[string]any
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}

	modelID, _ := data["model"].(string)
	preserveWebSearch := !isOpenAIModel(modelID) || !isWebSearchModel(modelID)

	toolsRaw, ok := data["tools"]
	if !ok {
		// No tools array; still clean up a stray non-function tool_choice.
		if dropNonFunctionToolChoice(data, preserveWebSearch) {
			result, err := json.Marshal(data)
			if err != nil {
				return body
			}
			return result
		}
		return body
	}
	toolsArr, ok := toolsRaw.([]any)
	if !ok {
		return body
	}

	var retainedTools []any
	var webSearchOptions map[string]any
	hasWebSearch := false
	nonFunctionDropped := false

	for _, t := range toolsArr {
		toolMap, ok := t.(map[string]any)
		if !ok {
			retainedTools = append(retainedTools, t)
			continue
		}
		toolType, _ := toolMap["type"].(string)
		switch toolType {
		case "web_search", "web_search_preview":
			if preserveWebSearch {
				// Non-legacy model: leave the built-in tool as-is and let the
				// selected upstream validate its own tool contract.
				retainedTools = append(retainedTools, t)
				continue
			}
			hasWebSearch = true
			nonFunctionDropped = true
			if webSearchOptions == nil {
				webSearchOptions = make(map[string]any)
			}
			for _, key := range []string{"search_context_size", "user_location"} {
				if value, exists := toolMap[key]; exists {
					if _, alreadySet := webSearchOptions[key]; !alreadySet {
						webSearchOptions[key] = value
					}
				}
			}
		case "function":
			retainedTools = append(retainedTools, t)
		default:
			// computer_use, text_editor, bash, google_search_retrieval,
			// code_execution, etc. — not supported by OpenAI Chat Completions.
			nonFunctionDropped = true
		}
	}

	if !hasWebSearch && !nonFunctionDropped {
		// Nothing changed.
		return body
	}

	if hasWebSearch && isWebSearchModel(modelID) {
		if existing, ok := data["web_search_options"].(map[string]any); ok {
			for key, value := range webSearchOptions {
				if _, exists := existing[key]; !exists {
					existing[key] = value
				}
			}
		} else if _, exists := data["web_search_options"]; !exists {
			data["web_search_options"] = webSearchOptions
		}
	}

	if len(retainedTools) > 0 {
		data["tools"] = retainedTools
	} else {
		delete(data, "tools")
		delete(data, "tool_choice")
	}

	// If tool_choice still references a non-function built-in, remove it.
	dropNonFunctionToolChoice(data, preserveWebSearch)

	result, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return result
}

func isWebSearchModel(modelID string) bool {
	return strings.Contains(strings.ToLower(modelID), "search-preview")
}

// isOpenAIModel reports whether modelID belongs to a real OpenAI model
// family (gpt-*, chatgpt-*, o1/o3/o4). OpenAI-specific request quirks (the
// web_search "-search-preview" naming convention, web_search_options
// rewrite) only apply to these — every other model reaching this converter
// is an OpenAI-compatible but otherwise unrelated vendor (xAI/Grok, etc.)
// plugged into the same provider slot.
func isOpenAIModel(modelID string) bool {
	base := extractBaseModelName(modelID)
	// "gpt-oss" is OpenAI's open-weight model family, commonly served by
	// non-OpenAI inference providers (Groq, Together, etc.) through this same
	// OpenAI-compatible converter slot — it must not be treated as a real
	// OpenAI-hosted model just because its name starts with "gpt".
	if strings.HasPrefix(base, "gpt-oss") {
		return false
	}
	if strings.HasPrefix(base, "gpt") || strings.HasPrefix(base, "chatgpt") {
		return true
	}
	for _, family := range []string{"o1", "o3", "o4"} {
		if matchModelFamily(modelID, family) {
			return true
		}
	}
	return false
}

// dropNonFunctionToolChoice removes tool_choice from data if it is a
// map-style object whose type is not "function". When preserveWebSearch is
// true, a tool_choice referencing web_search/web_search_preview is left in
// place instead. Returns true if removed.
func dropNonFunctionToolChoice(data map[string]any, preserveWebSearch bool) bool {
	tc, ok := data["tool_choice"].(map[string]any)
	if !ok {
		return false
	}
	tcType, _ := tc["type"].(string)
	if tcType == "function" {
		return false
	}
	if preserveWebSearch && (tcType == "web_search" || tcType == "web_search_preview") {
		return false
	}
	delete(data, "tool_choice")
	return true
}

// IsGptImage1Model reports whether the given model ID belongs to the gpt-image-1 family.
// This family does not support the response_format parameter in /v1/images/generations.
func IsGptImage1Model(modelID string) bool {
	return matchModelFamily(modelID, "gpt-image-1")
}

// StripResponseFormat removes the response_format field from a JSON request body.
// Used for model families (e.g. gpt-image-1) that reject this parameter.
func StripResponseFormat(body []byte) []byte {
	return UpdateJSONField(body, ModelParamsMapping{
		KeysToRemove: []string{"response_format"},
	})
}
