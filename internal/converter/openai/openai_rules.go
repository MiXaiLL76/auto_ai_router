package openai

import (
	"bytes"
	"encoding/json"
	"net/url"
	"slices"
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

// defaultParamSynonymGroups lists sets of request keys that set the same value. A
// client that sent any member of a group has already chosen it, so a default keyed
// on another member must not be added too: max_tokens next to max_completion_tokens
// is ambiguous for the server. Grouped (rather than keyed one-directionally) so the
// check works regardless of which spelling the default itself happens to use.
var defaultParamSynonymGroups = [][]string{
	{"max_tokens", "max_completion_tokens"},
}

// ApplyDefaultParams sets each key of defaults that is absent from the top level of a
// JSON request body, leaving every key the client sent untouched. It mirrors LiteLLM,
// where a deployment's litellm_params are merged under the request kwargs. Values
// already in the body keep their exact bytes; the body is returned as is when there is
// nothing to add or it is not a JSON object.
func ApplyDefaultParams(body []byte, defaults map[string]any) []byte {
	if len(defaults) == 0 || len(body) == 0 {
		return body
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return body
	}
	changed := false
	for key, value := range defaults {
		if clientSetParam(top, key) {
			continue
		}
		raw, err := json.Marshal(value)
		if err != nil {
			continue
		}
		top[key] = raw
		changed = true
	}
	if !changed {
		return body
	}
	out, err := json.Marshal(top)
	if err != nil {
		return body
	}
	return out
}

// clientSetParam reports whether the request already carries key or a synonym of it.
func clientSetParam(top map[string]json.RawMessage, key string) bool {
	if _, present := top[key]; present {
		return true
	}
	for _, group := range defaultParamSynonymGroups {
		if !slices.Contains(group, key) {
			continue
		}
		for _, synonym := range group {
			if synonym == key {
				continue
			}
			if _, present := top[synonym]; present {
				return true
			}
		}
	}
	return false
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

var gpt6Mapping = ModelParamsMapping{
	KeysToReplace: map[string]string{"max_tokens": "max_completion_tokens"},
	KeysToRemove:  []string{"temperature", "top_p", "logprobs", "top_logprobs"},
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
	{"gpt-6", gpt6Mapping},
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

// StripCacheSalt removes the cache_salt field from a JSON request body.
// cache_salt is a real OpenAI Chat Completions parameter (partitions prompt
// caching), but it's recent enough that most other OpenAI-compatible server
// implementations -- vLLM-based deployments, aggregators, anything using a
// strict Pydantic/JSON-Schema request model -- don't recognize it yet and
// reject the whole request with a 400 ("cache_salt: Extra inputs are not
// permitted") rather than ignoring an unknown field. See IsRealOpenAIHost:
// only genuine api.openai.com should ever see this field forwarded.
func StripCacheSalt(body []byte) []byte {
	// This runs on every request through the default (OpenAI-compatible)
	// branch, so skip the unmarshal/marshal round trip in the common case
	// where the field isn't present at all.
	if !bytes.Contains(body, []byte(`"cache_salt"`)) {
		return body
	}
	return UpdateJSONField(body, ModelParamsMapping{
		KeysToRemove: []string{"cache_salt"},
	})
}

var streamOptionsIncludeUsageOnly = json.RawMessage(`{"include_usage":true}`)

// RebuildStreamOptionsIncludeUsageOnly replaces the stream_options object
// with exactly {"include_usage": true} when present, discarding any other
// keys. The ingress sanitizer guarantees stream_options exists with
// include_usage=true for every streaming Chat Completions request but
// otherwise preserves whatever the client sent (e.g. vLLM's
// continuous_usage_stats extension) -- that's fine for a genuine self-hosted
// vLLM destination, which understands the key, but api.openai.com and other
// strict OpenAI-compatible servers reject an unrecognized key outright with
// a 400 ("stream_options: Extra inputs are not permitted"). Callers decide
// when to call this based on the resolved provider (see
// ProviderConverter.shouldStripStreamOptionsExtras); it's a no-op if
// stream_options isn't present.
func RebuildStreamOptionsIncludeUsageOnly(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"stream_options"`)) {
		return body
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(body, &data); err != nil {
		return body
	}
	if _, exists := data["stream_options"]; !exists {
		return body
	}
	data["stream_options"] = streamOptionsIncludeUsageOnly
	marshaled, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return marshaled
}

// StripStreamOptions removes the stream_options field from a JSON request
// body entirely, unlike RebuildStreamOptionsIncludeUsageOnly. Used for the
// native Anthropic Messages API (/v1/messages), which -- unlike the OpenAI
// wire protocol bucket -- doesn't merely reject unrecognized keys inside
// stream_options, it has no stream_options concept at all and rejects the
// whole field outright with a 400 ("stream_options: Extra inputs are not
// permitted"). Native Anthropic streaming always includes usage regardless,
// so there's no include_usage equivalent to preserve here.
func StripStreamOptions(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"stream_options"`)) {
		return body
	}
	return UpdateJSONField(body, ModelParamsMapping{
		KeysToRemove: []string{"stream_options"},
	})
}

// IsRealOpenAIHost reports whether baseURL points at OpenAI's own API
// (api.openai.com or a subdomain), as opposed to a third-party server that
// merely speaks the OpenAI-compatible wire protocol (OpenRouter, a
// self-hosted vLLM deployment, most aggregators) -- credentials of type
// "openai" cover both cases here, since AIR's provider Type field only
// records the wire protocol, not who actually operates the endpoint.
func IsRealOpenAIHost(baseURL string) bool {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" {
		return false
	}
	u, err := url.Parse(trimmed)
	if err != nil || u.Hostname() == "" {
		u, err = url.Parse("https://" + trimmed)
		if err != nil {
			return false
		}
	}
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	return host == "api.openai.com" || strings.HasSuffix(host, ".api.openai.com")
}

func ReplaceResponsesBodyParam(modelID string, body []byte) []byte {
	if !matchModelFamily(modelID, "gpt-6") {
		return body
	}
	var data map[string]json.RawMessage
	if json.Unmarshal(body, &data) != nil {
		return body
	}
	changed := false
	for _, key := range []string{"temperature", "top_p", "top_logprobs"} {
		if _, exists := data[key]; exists {
			delete(data, key)
			changed = true
		}
	}
	var include []string
	if json.Unmarshal(data["include"], &include) == nil {
		filtered := make([]string, 0, len(include))
		for _, item := range include {
			if item != "message.output_text.logprobs" {
				filtered = append(filtered, item)
			}
		}
		if len(filtered) != len(include) {
			changed = true
			if len(filtered) == 0 {
				delete(data, "include")
			} else {
				encoded, err := json.Marshal(filtered)
				if err != nil {
					return body
				}
				data["include"] = encoded
			}
		}
	}
	if !changed {
		return body
	}
	updated, err := json.Marshal(data)
	if err != nil {
		return body
	}
	return updated
}
