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
	oldToken, _ := json.Marshal(oldModel)
	newToken, _ := json.Marshal(newModel)

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
// Reasoning models that support reasoning_effort but reject temperature/top_p.
var o3Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
	},
}

// gpt5Mapping: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1, gpt-5.2, etc.
var gpt5Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{
		"max_tokens": "max_completion_tokens",
	},
	KeysToRemove: []string{
		"temperature",
		"top_p",
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
