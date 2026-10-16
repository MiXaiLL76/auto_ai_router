package anthropic

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/mixaill76/auto_ai_router/internal/converter/converterutil"
	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
)

const maxOpenAIToolNameLength = 64

var invalidToolUseIDChar = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

type MessagesAdapterMetadata struct {
	ToolNames      map[string]string
	AnthropicBetas []string
}

func MessagesToChat(body []byte) ([]byte, MessagesAdapterMetadata, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		// The body isn't even valid JSON at this point (request is a generic map, so a
		// present-but-wrong-typed field would unmarshal fine here and get caught by the
		// specific checks below instead) -- still the client's mistake, not ours.
		return nil, MessagesAdapterMetadata{}, converterutil.RequestJSONValidationError(err)
	}

	// Every field below follows the same three-step order: present? -- right type? --
	// acceptable value? A wrong-typed field's zero value can accidentally satisfy an
	// earlier value check -- "model" used to check "== ''" before checking the type
	// assertion's own ok, so a non-string model (e.g. 123, zero-valuing to "") matched
	// the "missing" branch first and the invalid_type branch below it was unreachable.
	modelField, hasModel := request["model"]
	if !hasModel {
		return nil, MessagesAdapterMetadata{}, converterutil.NewRequestValidationError("model", "Missing required parameter")
	}
	model, modelIsString := modelField.(string)
	if !modelIsString {
		return nil, MessagesAdapterMetadata{}, converterutil.NewInvalidTypeError("model")
	}
	if model == "" {
		return nil, MessagesAdapterMetadata{}, converterutil.NewRequestValidationError("model", "Missing required parameter")
	}
	maxTokensField, hasMaxTokens := request["max_tokens"]
	if !hasMaxTokens {
		return nil, MessagesAdapterMetadata{}, converterutil.NewRequestValidationError("max_tokens", "Missing required parameter")
	}
	maxTokens, maxTokensIsNumber := maxTokensField.(float64)
	if !maxTokensIsNumber {
		return nil, MessagesAdapterMetadata{}, converterutil.NewInvalidTypeError("max_tokens")
	}
	if maxTokens <= 0 {
		return nil, MessagesAdapterMetadata{}, converterutil.NewInvalidValueError("max_tokens")
	}
	messagesField, hasMessages := request["messages"]
	if !hasMessages {
		return nil, MessagesAdapterMetadata{}, converterutil.NewRequestValidationError("messages", "Missing required parameter")
	}
	rawMessages, messagesIsArray := messagesField.([]interface{})
	if !messagesIsArray {
		return nil, MessagesAdapterMetadata{}, converterutil.NewInvalidTypeError("messages")
	}
	if len(rawMessages) == 0 {
		return nil, MessagesAdapterMetadata{}, converterutil.NewInvalidValueError("messages")
	}

	messages, err := messagesToChatMessages(rawMessages)
	if err != nil {
		return nil, MessagesAdapterMetadata{}, err
	}
	if system, ok := request["system"]; ok {
		if converted := systemToChatContent(system); converted != nil {
			messages = append([]interface{}{map[string]interface{}{"role": "system", "content": converted}}, messages...)
		}
	}

	chat := make(map[string]interface{}, len(request)+1)
	for key, value := range request {
		switch key {
		case "messages", "system", "metadata", "stop_sequences", "tools", "tool_choice", "thinking", "output_format", "output_config", "anthropic_beta":
		default:
			chat[key] = value
		}
	}
	chat["model"] = model
	chat["messages"] = messages
	if metadata, ok := request["metadata"].(map[string]interface{}); ok {
		if userID, ok := metadata["user_id"].(string); ok && userID != "" {
			chat["user"] = userID
		}
	}
	if stop, ok := request["stop_sequences"]; ok {
		chat["stop"] = stop
	}

	toolNames := map[string]string{}
	if tools, ok := request["tools"].([]interface{}); ok && len(tools) > 0 {
		chat["tools"], toolNames = toolsToChat(tools)
	}
	if choice, ok := request["tool_choice"].(map[string]interface{}); ok {
		chat["tool_choice"] = toolChoiceToChat(choice)
	}
	outputConfig, _ := request["output_config"].(map[string]interface{})
	configuredEffort, _ := outputConfig["effort"].(string)
	thinking, hasThinking := request["thinking"].(map[string]interface{})
	if !hasThinking && configuredEffort != "" && isClaudeModel(model) && thinksByDefault(model) {
		// Omitting thinking runs adaptive on this model: say so, so the effort can go along.
		thinking, hasThinking = map[string]interface{}{"type": "adaptive"}, true
	}
	if hasThinking {
		if isClaudeModel(model) {
			chat["thinking"] = thinking
			// The Chat form has no output_config: OpenAIToAnthropic reads the effort back from
			// reasoning_effort. Only next to a thinking config that thinks: beside "disabled",
			// an OpenAI-compatible provider would read it as a request to reason. Effort with
			// thinking off, or without thinking on a model where that means off, is not carried.
			if configuredEffort != "" && (thinking["type"] == "adaptive" || thinking["type"] == "enabled") {
				chat["reasoning_effort"] = configuredEffort
			}
		} else if effort := thinkingToReasoningEffort(thinking); effort != "" {
			if configuredEffort != "" && thinking["type"] == "adaptive" {
				effort = configuredEffort
			}
			chat["reasoning_effort"] = effort
		}
	}
	if outputFormat := extractOutputFormat(request); outputFormat != nil {
		chat["response_format"] = outputFormat
	}
	if stream, _ := request["stream"].(bool); stream {
		chat["stream_options"] = map[string]interface{}{"include_usage": true}
	}

	converted, err := json.Marshal(chat)
	if err != nil {
		return nil, MessagesAdapterMetadata{}, fmt.Errorf("failed to encode Chat Completions request: %w", err)
	}
	return converted, MessagesAdapterMetadata{
		ToolNames:      toolNames,
		AnthropicBetas: extractBetaStrings(request["anthropic_beta"]),
	}, nil
}

// NormalizeMessagesForPassthrough applies the same thinking/output_config/anthropic_beta
// normalization OpenAIToAnthropic performs during the Messages->Chat->Messages round trip,
// directly to a raw native Anthropic Messages API request body — for the /v1/messages
// passthrough path, which forwards the client's body to an Anthropic-wire-compatible
// provider without going through Chat Completions shape at all. Shares OpenAIToAnthropic's
// thinking mapping (mapThinkingConfig, applyThinkingSideEffects), so real API requirements
// (the effort-2025-11-24 beta for adaptive thinking, temperature=1.0 while thinking is
// enabled, upgrading legacy enabled+budget_tokens to adaptive+effort on adaptive-only
// models, and disabling autonomous thinking by default on non-Anthropic backends behind
// CometAPI/ProMan) keep working without paying for the full round trip. The client's
// output_config is kept as sent; an effort derived from budget_tokens fills it only when
// it has none.
func NormalizeMessagesForPassthrough(body []byte, model string, isRealAnthropicBackend bool) ([]byte, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, converterutil.RequestJSONValidationError(err)
	}

	// Models that no longer accept sampling params (Claude Opus 4.7+ — see
	// SamplingRemoved) reject temperature/top_p/top_k. Drop any client-supplied
	// values so the native /v1/messages passthrough body does not 400.
	dropSampling := SamplingRemoved(model)
	if dropSampling {
		delete(request, "temperature")
		delete(request, "top_p")
		delete(request, "top_k")
	}

	tc, oc := mapThinkingConfig(request["thinking"], "", model)
	outputConfig, _ := request["output_config"].(map[string]interface{})
	effort, _ := outputConfig["effort"].(string)
	if err := validateBetweenTools(tc, effort); err != nil {
		return nil, err
	}
	if tc != nil {
		request["thinking"] = tc
		existingMaxTokens, hasMaxTokens := request["max_tokens"].(float64)
		maxTokens, betas, temp := applyThinkingSideEffects(tc, oc, int(existingMaxTokens), extractBetaStrings(request["anthropic_beta"]))
		if hasMaxTokens {
			request["max_tokens"] = maxTokens
		}
		if len(betas) > 0 {
			request["anthropic_beta"] = betas
		}
		// An effort the client set explicitly wins over the one derived from budget_tokens.
		if oc != nil && effort == "" {
			if outputConfig != nil {
				outputConfig["effort"] = oc.Effort
			} else {
				request["output_config"] = oc
			}
		}
		if !dropSampling {
			request["temperature"] = temp
		}
	} else if thinking, _ := request["thinking"].(map[string]interface{}); !isRealAnthropicBackend || thinking["type"] == "between_tools" {
		// between_tools exists on Sonnet 5.5 only (mapped above); elsewhere it means off.
		// A model that can't turn thinking off (at this effort) gets no thinking field and
		// runs adaptive, rather than a 400.
		if off := thinkingOff(model, effort); off != nil {
			request["thinking"] = off
		} else {
			delete(request, "thinking")
		}
	}

	return json.Marshal(request)
}

func extractBetaStrings(raw interface{}) []string {
	var betas []string
	switch v := raw.(type) {
	case string:
		if s := strings.TrimSpace(v); s != "" {
			betas = append(betas, s)
		}
	case []interface{}:
		for _, b := range v {
			if s, ok := b.(string); ok {
				if s = strings.TrimSpace(s); s != "" {
					betas = append(betas, s)
				}
			}
		}
	case []string:
		for _, s := range v {
			if s = strings.TrimSpace(s); s != "" {
				betas = append(betas, s)
			}
		}
	}
	return betas
}

func messagesToChatMessages(rawMessages []interface{}) ([]interface{}, error) {
	converted := make([]interface{}, 0, len(rawMessages))
	for _, raw := range rawMessages {
		message, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid message")
		}
		role, _ := message["role"].(string)
		switch role {
		case "user":
			messages, err := userMessageToChat(message)
			if err != nil {
				return nil, err
			}
			converted = append(converted, messages...)
		case "assistant":
			if assistant := assistantMessageToChat(message); assistant != nil {
				converted = append(converted, assistant)
			}
		case "system", "developer":
			// The Messages API spec puts the system prompt in the top-level "system"
			// field, never in "messages" — but some OpenAI-oriented clients pointed at
			// this Anthropic-compatible endpoint send it as a role:"system"/"developer"
			// message instead. Accept it the same way the reverse converter
			// (convertOpenAIMessagesToAnthropic) already does, rather than rejecting an
			// otherwise well-formed request.
			if content := systemToChatContent(message["content"]); content != nil {
				converted = append(converted, map[string]interface{}{"role": "system", "content": content})
			}
		default:
			return nil, fmt.Errorf("unsupported message role %q", role)
		}
	}
	return converted, nil
}

func userMessageToChat(message map[string]interface{}) ([]interface{}, error) {
	content := message["content"]
	if text, ok := content.(string); ok {
		return []interface{}{map[string]interface{}{"role": "user", "content": text}}, nil
	}
	blocks, _ := content.([]interface{})
	var userContent []interface{}
	var toolMessages []interface{}
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			part := map[string]interface{}{"type": "text", "text": stringValue(block["text"])}
			copyCacheControl(block, part)
			userContent = append(userContent, part)
		case "image":
			if imageURL := SourceToImageURL(block["source"]); imageURL != "" {
				part := map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL}}
				copyCacheControl(block, part)
				userContent = append(userContent, part)
			}
		case "video":
			// Video is not part of Anthropic's own content-block set, but some
			// Anthropic-compatible providers accept it. The block is kept verbatim so the
			// request-side conversion can hand it back to the provider untouched: a client
			// that deliberately sent a video block is targeting a provider that reads one,
			// and silently dropping it here leaves the model answering about media it never
			// received. The source shape is validated against the same url/base64 rules the
			// later Anthropic-shape conversion enforces, so a block accepted here always
			// survives the round trip instead of being dropped downstream.
			if source := sourceToVideoSource(block["source"]); source != nil {
				part := map[string]interface{}{"type": "video", "source": source}
				copyCacheControl(block, part)
				userContent = append(userContent, part)
			}
		case "document":
			part, err := sourceToDocumentPart(block["source"])
			if err != nil {
				return nil, err
			}
			if part != nil {
				copyCacheControl(block, part)
				userContent = append(userContent, part)
			}
		case "tool_result":
			content, err := toolResultContentToChat(block["content"])
			if err != nil {
				return nil, err
			}
			tool := map[string]interface{}{
				"role":         "tool",
				"tool_call_id": normalizeToolUseID(stringValue(block["tool_use_id"])),
				"content":      content,
			}
			copyCacheControl(block, tool)
			toolMessages = append(toolMessages, tool)
		}
	}
	if len(userContent) > 0 {
		toolMessages = append(toolMessages, map[string]interface{}{"role": "user", "content": userContent})
	}
	return toolMessages, nil
}

func assistantMessageToChat(message map[string]interface{}) interface{} {
	content := message["content"]
	if text, ok := content.(string); ok {
		return map[string]interface{}{"role": "assistant", "content": text}
	}
	blocks, _ := content.([]interface{})
	var text strings.Builder
	var textBlocks []interface{}
	var toolCalls []interface{}
	var thinkingBlocks []interface{}
	hasTextCacheControl := false
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			text.WriteString(stringValue(block["text"]))
			part := map[string]interface{}{"type": "text", "text": stringValue(block["text"])}
			copyCacheControl(block, part)
			if _, ok := part["cache_control"]; ok {
				hasTextCacheControl = true
			}
			textBlocks = append(textBlocks, part)
		case "tool_use":
			name := truncateToolName(stringValue(block["name"]))
			arguments, _ := json.Marshal(block["input"])
			call := map[string]interface{}{
				"id":   normalizeToolUseID(stringValue(block["id"])),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": string(arguments),
				},
			}
			copyCacheControl(block, call)
			toolCalls = append(toolCalls, call)
		case "thinking", "redacted_thinking":
			thinkingBlocks = append(thinkingBlocks, block)
		}
	}
	if len(textBlocks) == 0 && len(toolCalls) == 0 && len(thinkingBlocks) == 0 {
		return nil
	}
	var chatContent interface{} = text.String()
	if len(textBlocks) == 0 {
		chatContent = nil
	} else if hasTextCacheControl {
		chatContent = textBlocks
	}
	result := map[string]interface{}{"role": "assistant", "content": chatContent}
	if len(toolCalls) > 0 {
		result["tool_calls"] = toolCalls
	}
	if len(thinkingBlocks) > 0 {
		result["thinking_blocks"] = thinkingBlocks
	}
	return result
}

func systemToChatContent(system interface{}) interface{} {
	if text, ok := system.(string); ok {
		if text == "" {
			return nil
		}
		return text
	}
	blocks, _ := system.([]interface{})
	content := make([]interface{}, 0, len(blocks))
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok || block["type"] != "text" {
			continue
		}
		part := map[string]interface{}{"type": "text", "text": stringValue(block["text"])}
		copyCacheControl(block, part)
		content = append(content, part)
	}
	if len(content) == 0 {
		return nil
	}
	return content
}

func toolsToChat(tools []interface{}) ([]interface{}, map[string]string) {
	converted := make([]interface{}, 0, len(tools))
	names := map[string]string{}
	for index, raw := range tools {
		tool, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		toolType, _ := tool["type"].(string)
		if strings.HasPrefix(toolType, "computer_") || strings.HasPrefix(toolType, "bash_") ||
			strings.HasPrefix(toolType, "text_editor_") || strings.HasPrefix(toolType, "web_search_") {
			converted = append(converted, tool)
			continue
		}
		name := strings.TrimSpace(stringValue(tool["name"]))
		if name == "" {
			name = fmt.Sprintf("litellm_unnamed_tool_%d", index)
		}
		truncated := truncateToolName(name)
		if truncated != name {
			names[truncated] = name
		}
		function := map[string]interface{}{"name": truncated}
		if schema, ok := tool["input_schema"]; ok {
			function["parameters"] = schema
		}
		if description, ok := tool["description"]; ok {
			function["description"] = description
		}
		convertedTool := map[string]interface{}{"type": "function", "function": function}
		copyCacheControl(tool, convertedTool)
		converted = append(converted, convertedTool)
	}
	return converted, names
}

func toolChoiceToChat(choice map[string]interface{}) interface{} {
	switch choice["type"] {
	case "any":
		return "required"
	case "auto":
		return "auto"
	case "none":
		return "none"
	case "tool":
		return map[string]interface{}{
			"type":     "function",
			"function": map[string]interface{}{"name": truncateToolName(stringValue(choice["name"]))},
		}
	default:
		return choice
	}
}

// SourceToImageURL turns an Anthropic image block source (base64 or url) into an
// OpenAI-style image URL; a base64 source without media_type is taken as JPEG.
// Returns "" for sources that have no URL form (file, missing data).
func SourceToImageURL(raw interface{}) string {
	source, _ := raw.(map[string]interface{})
	switch source["type"] {
	case "base64":
		data := stringValue(source["data"])
		if data == "" {
			return ""
		}
		mediaType := stringValue(source["media_type"])
		if mediaType == "" {
			mediaType = "image/jpeg"
		}
		return "data:" + mediaType + ";base64," + data
	case "url":
		return stringValue(source["url"])
	default:
		return ""
	}
}

// sourceToVideoSource validates a video block's source the same way mediaSourceFromMap
// validates it on the outbound path, so a block accepted here is guaranteed to survive the
// later conversion instead of being silently dropped after round-tripping through chat.
func sourceToVideoSource(raw interface{}) map[string]interface{} {
	source, ok := raw.(map[string]interface{})
	if !ok {
		return nil
	}
	switch source["type"] {
	case "base64":
		if stringValue(source["data"]) == "" {
			return nil
		}
	case "url":
		if stringValue(source["url"]) == "" {
			return nil
		}
	default:
		return nil
	}
	return source
}

func sourceToDocumentPart(raw interface{}) (map[string]interface{}, error) {
	source, ok := raw.(map[string]interface{})
	if !ok {
		return nil, converterutil.NewRequestValidationError("messages.content.document.source", "missing document source")
	}
	switch source["type"] {
	case "base64":
		data := stringValue(source["data"])
		mediaType := stringValue(source["media_type"])
		if mediaType == "" {
			return nil, converterutil.NewRequestValidationError("messages.content.document.source.media_type", "missing document media_type")
		}
		if data == "" {
			return nil, converterutil.NewRequestValidationError("messages.content.document.source.data", "missing document base64 data")
		}
		return map[string]interface{}{
			"type": "document",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": mediaType,
				"data":       data,
			},
		}, nil
	case "url":
		url := stringValue(source["url"])
		if url == "" {
			return nil, converterutil.NewRequestValidationError("messages.content.document.source.url", "missing document URL")
		}
		return map[string]interface{}{
			"type": "document",
			"source": map[string]interface{}{
				"type": "url",
				"url":  url,
			},
		}, nil
	case "file", "file_id", "provider_file":
		return nil, converterutil.NewRequestValidationError("messages.content.document.source.file_id", "file_id is not supported for this route")
	default:
		return nil, converterutil.NewRequestValidationError("messages.content.document.source.type", "unsupported document source type")
	}
}

func toolResultContentToChat(raw interface{}) (interface{}, error) {
	if raw == nil {
		return "", nil
	}
	if text, ok := raw.(string); ok {
		return text, nil
	}
	blocks, _ := raw.([]interface{})
	converted := make([]interface{}, 0, len(blocks))
	for _, rawBlock := range blocks {
		if text, ok := rawBlock.(string); ok {
			converted = append(converted, map[string]interface{}{"type": "text", "text": text})
			continue
		}
		block, ok := rawBlock.(map[string]interface{})
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			converted = append(converted, map[string]interface{}{"type": "text", "text": stringValue(block["text"])})
		case "image":
			if imageURL := SourceToImageURL(block["source"]); imageURL != "" {
				converted = append(converted, map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL}})
			}
		case "document":
			part, err := sourceToDocumentPart(block["source"])
			if err != nil {
				return nil, err
			}
			if part != nil {
				converted = append(converted, part)
			}
		}
	}
	if len(converted) == 1 {
		if part, ok := converted[0].(map[string]interface{}); ok && part["type"] == "text" {
			return part["text"], nil
		}
	}
	return converted, nil
}

func copyCacheControl(source, target map[string]interface{}) {
	if value, ok := source["cache_control"]; ok {
		target["cache_control"] = value
	}
}

func truncateToolName(name string) string {
	if len(name) <= maxOpenAIToolNameLength {
		return name
	}
	truncated := name[:maxOpenAIToolNameLength]
	for len(truncated) > 0 && !utf8.ValidString(truncated) {
		truncated = truncated[:len(truncated)-1]
	}
	return truncated
}

func isClaudeModel(model string) bool {
	lower := strings.ToLower(model)
	return strings.Contains(lower, "anthropic") || strings.Contains(lower, "claude")
}

func thinkingToReasoningEffort(thinking map[string]interface{}) string {
	switch thinking["type"] {
	case "adaptive":
		return "medium"
	case "enabled":
		budget, _ := thinking["budget_tokens"].(float64)
		switch {
		case budget >= 4096:
			return "high"
		case budget >= 2048:
			return "medium"
		case budget >= 1024:
			return "low"
		default:
			return "minimal"
		}
	default:
		return ""
	}
}

func extractOutputFormat(request map[string]interface{}) interface{} {
	output := request["output_format"]
	if output == nil {
		if config, ok := request["output_config"].(map[string]interface{}); ok {
			output = config["format"]
		}
	}
	format, ok := output.(map[string]interface{})
	if !ok || format["type"] != "json_schema" || format["schema"] == nil {
		return nil
	}
	schema, ok := format["schema"].(map[string]interface{})
	if !ok {
		return nil
	}
	makeOpenAISchemaStrict(schema)
	return map[string]interface{}{
		"type": "json_schema",
		"json_schema": map[string]interface{}{
			"name":   "structured_output",
			"schema": schema,
			"strict": true,
		},
	}
}

func makeOpenAISchemaStrict(schema map[string]interface{}) {
	if schema["type"] == "object" {
		if properties, ok := schema["properties"].(map[string]interface{}); ok {
			schema["additionalProperties"] = false
			required := make([]string, 0, len(properties))
			for name, raw := range properties {
				required = append(required, name)
				if child, ok := raw.(map[string]interface{}); ok {
					makeOpenAISchemaStrict(child)
				}
			}
			sort.Strings(required)
			schema["required"] = required
		}
	}
	if item, ok := schema["items"].(map[string]interface{}); ok {
		makeOpenAISchemaStrict(item)
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		alternatives, _ := schema[key].([]interface{})
		for _, raw := range alternatives {
			if child, ok := raw.(map[string]interface{}); ok {
				makeOpenAISchemaStrict(child)
			}
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		definitions, _ := schema[key].(map[string]interface{})
		for _, raw := range definitions {
			if child, ok := raw.(map[string]interface{}); ok {
				makeOpenAISchemaStrict(child)
			}
		}
	}
}

// ChatToMessages converts a Chat Completions response body to the Anthropic
// Messages API format. cacheWriteTTLFallback5mTokens/1hTokens carry the
// cache-write TTL split sourced from upstream response headers (used by
// Kimi/Moonshot, whose body never reports a 5m/1h breakdown); they are
// applied only when the body itself has none. Pass 0, 0 when no such
// fallback is available.
func ChatToMessages(body []byte, metadata MessagesAdapterMetadata, cacheWriteTTLFallback5mTokens, cacheWriteTTLFallback1hTokens int) ([]byte, error) {
	var response struct {
		ID      string              `json:"id"`
		Model   string              `json:"model"`
		Usage   *openai.OpenAIUsage `json:"usage"`
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content          *string                  `json:"content"`
				ReasoningContent string                   `json:"reasoning_content"`
				ThinkingBlocks   []map[string]interface{} `json:"thinking_blocks"`
				ToolCalls        []openai.OpenAIToolCall  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to parse Chat Completions response: %w", err)
	}
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("chat completions response has no choices")
	}

	choice := response.Choices[0]
	content := make([]interface{}, 0, 1+len(choice.Message.ToolCalls))
	// Only the block's own fields: a Chat entry also carries its list "index" (see
	// chatThinkingBlock), which Anthropic rejects when the client echoes the block back.
	for _, block := range choice.Message.ThinkingBlocks {
		switch block["type"] {
		case "thinking":
			content = append(content, map[string]interface{}{
				"type": "thinking", "thinking": stringValue(block["thinking"]), "signature": block["signature"],
			})
		case "redacted_thinking":
			content = append(content, map[string]interface{}{"type": "redacted_thinking", "data": block["data"]})
		}
	}
	if len(choice.Message.ThinkingBlocks) == 0 && choice.Message.ReasoningContent != "" {
		content = append(content, map[string]interface{}{
			"type": "thinking", "thinking": choice.Message.ReasoningContent, "signature": nil,
		})
	}
	// A tool_use-only turn has content "" in Chat form; echoed back as an empty
	// text block it gets a 400 from Anthropic.
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		content = append(content, map[string]interface{}{"type": "text", "text": *choice.Message.Content})
	}
	for _, toolCall := range choice.Message.ToolCalls {
		var input interface{}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
			input = map[string]interface{}{}
		}
		name := toolCall.Function.Name
		if original := metadata.ToolNames[name]; original != "" {
			name = original
		}
		content = append(content, map[string]interface{}{
			"type":  "tool_use",
			"id":    normalizeToolUseID(toolCall.ID),
			"name":  name,
			"input": input,
		})
	}

	converted := map[string]interface{}{
		"id":            response.ID,
		"type":          "message",
		"role":          "assistant",
		"content":       content,
		"model":         response.Model,
		"stop_reason":   chatFinishReasonToMessages(choice.FinishReason),
		"stop_sequence": nil,
		"usage":         chatUsageToMessages(response.Usage, cacheWriteTTLFallback5mTokens, cacheWriteTTLFallback1hTokens),
	}
	return json.Marshal(converted)
}

func chatUsageToMessages(usage *openai.OpenAIUsage, cacheWriteTTLFallback5mTokens, cacheWriteTTLFallback1hTokens int) *AnthropicUsage {
	if usage == nil {
		return &AnthropicUsage{}
	}
	cacheRead := 0
	cacheType := ""
	if usage.PromptTokensDetails != nil {
		cacheRead = usage.PromptTokensDetails.CachedTokens
		cacheType = usage.PromptTokensDetails.CacheType
	}
	cacheCreation, cacheCreation5m, cacheCreation1h := usage.PromptTokensDetails.CacheWrite()
	// Kimi/Moonshot never reports a TTL split in the body — fall back to the
	// header-sourced split the caller supplied.
	if cacheCreation5m == 0 && cacheCreation1h == 0 {
		cacheCreation5m = cacheWriteTTLFallback5mTokens
		cacheCreation1h = cacheWriteTTLFallback1hTokens
	}
	result := &AnthropicUsage{
		InputTokens:              max(usage.PromptTokens-cacheRead-cacheCreation, 0),
		OutputTokens:             usage.CompletionTokens,
		CacheReadInputTokens:     cacheRead,
		CacheCreationInputTokens: cacheCreation,
		CacheType:                cacheType,
	}
	// Billing downstream (stream reader, next AIR hop) sees only this usage:
	// without the split a 1h write bills at the 5m price, without
	// server_tool_use searches do not bill at all.
	if cacheCreation5m > 0 || cacheCreation1h > 0 {
		result.CacheCreation = &CacheCreationDetails{
			Ephemeral5mInputTokens: cacheCreation5m,
			Ephemeral1hInputTokens: cacheCreation1h,
		}
	}
	if usage.ServerToolUse != nil && usage.ServerToolUse.WebSearchRequests > 0 {
		result.ServerToolUse = &ServerToolUsageDetails{
			WebSearchRequests: usage.ServerToolUse.WebSearchRequests,
		}
	}
	return result
}

func chatFinishReasonToMessages(reason string) string {
	switch reason {
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func normalizeToolUseID(id string) string {
	if separator := strings.Index(id, "__thought__"); separator >= 0 {
		id = id[:separator]
	}
	normalized := invalidToolUseIDChar.ReplaceAllString(id, "_")
	if normalized == "" {
		return "tool_use_id"
	}
	return normalized
}

type messagesStreamState struct {
	writer                        io.Writer
	model                         string
	metadata                      MessagesAdapterMetadata
	messageID                     string
	blockIndex                    int
	blockType                     string
	thinkingStreamed              bool // the open thinking block got thinking_delta text
	toolIndex                     int
	finish                        string
	usage                         *openai.OpenAIUsage
	started                       bool
	cacheWriteTTLFallback5mTokens int
	cacheWriteTTLFallback1hTokens int
}

// TransformChatStreamToMessages converts a Chat Completions SSE stream to
// Anthropic Messages API SSE. cacheWriteTTLFallback5mTokens/1hTokens carry
// the cache-write TTL split sourced from upstream response headers (used by
// Kimi/Moonshot, whose chunks never report a 5m/1h breakdown); applied only
// when a chunk's own usage has none. Pass 0, 0 when no such fallback is
// available.
func TransformChatStreamToMessages(reader io.Reader, writer io.Writer, model string, metadata MessagesAdapterMetadata, cacheWriteTTLFallback5mTokens, cacheWriteTTLFallback1hTokens int) error {
	state := messagesStreamState{
		writer:                        writer,
		model:                         model,
		metadata:                      metadata,
		messageID:                     "msg_" + uuid.NewString(),
		cacheWriteTTLFallback5mTokens: cacheWriteTTLFallback5mTokens,
		cacheWriteTTLFallback1hTokens: cacheWriteTTLFallback1hTokens,
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			return state.finishStream()
		}
		var chunk openai.OpenAIStreamingChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if chunk.ID != "" {
			state.messageID = chunk.ID
		}
		if chunk.Model != "" {
			state.model = chunk.Model
		}
		if err := state.startMessage(); err != nil {
			return err
		}
		if chunk.Usage != nil {
			state.usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.ReasoningContent != "" {
				if err := state.writeDelta("thinking", -1, "", "", choice.Delta.ReasoningContent); err != nil {
					return err
				}
			}
			blocks, _ := choice.Delta.ThinkingBlocks.([]interface{})
			for _, raw := range blocks {
				block, _ := raw.(map[string]interface{})
				if err := state.writeThinkingBlock(block); err != nil {
					return err
				}
			}
			if choice.Delta.Content != "" {
				if err := state.writeDelta("text", -1, "", "", choice.Delta.Content); err != nil {
					return err
				}
			}
			for _, call := range choice.Delta.ToolCalls {
				id := normalizeToolUseID(call.ID)
				name := ""
				arguments := ""
				if call.Function != nil {
					name = call.Function.Name
					arguments = call.Function.Arguments
				}
				if original := metadata.ToolNames[name]; original != "" {
					name = original
				}
				if err := state.writeDelta("tool_use", call.Index, id, name, arguments); err != nil {
					return err
				}
			}
			if choice.FinishReason != nil {
				state.finish = chatFinishReasonToMessages(*choice.FinishReason)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("chat completions stream scanner: %w", err)
	}
	return state.finishStream()
}

// PassthroughMessagesStream relays a provider stream that already is in the Messages
// API format, one SSE event per write, so thinking signatures, block order and
// input_transformations reach the client untouched. Two things change: message_start's
// message.model becomes model, and message_delta's usage gets the fields message_start
// reported and it leaves out — Anthropic may report input and cache tokens in
// message_start only, while billing reads the final message_delta. Provider fields are
// copied as sent; only two counters are filled in when missing: web searches, from the
// server_tool_use blocks, and the cache-write total, from its TTL split.
func PassthroughMessagesStream(reader io.Reader, writer io.Writer, model string) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var state passthroughStreamState
	var event []byte
	flush := func() error {
		if len(event) == 0 {
			return nil
		}
		_, err := writer.Write(append(event, '\n'))
		event = event[:0]
		return err
	}
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if payload, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			if rewritten, ok := rewritePassthroughEvent(bytes.TrimSpace(payload), model, &state); ok {
				line = append([]byte("data: "), rewritten...)
			}
		}
		event = append(append(event, line...), '\n')
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("messages stream scanner: %w", err)
	}
	return flush()
}

// passthroughStreamState is what PassthroughMessagesStream carries between events.
type passthroughStreamState struct {
	startUsage      map[string]json.RawMessage // message_start's message.usage, field by field as sent
	webSearchBlocks int                        // server_tool_use web_search blocks opened so far
}

func rewritePassthroughEvent(payload []byte, model string, state *passthroughStreamState) ([]byte, bool) {
	if !bytes.Contains(payload, []byte(`"message_`)) && !bytes.Contains(payload, []byte(`"server_tool_use"`)) {
		return nil, false
	}
	var event map[string]json.RawMessage
	if json.Unmarshal(payload, &event) != nil || event == nil {
		return nil, false
	}
	switch string(bytes.TrimSpace(event["type"])) {
	case `"content_block_start"`:
		var block ContentBlock
		if json.Unmarshal(event["content_block"], &block) == nil && block.Type == "server_tool_use" && isWebSearchToolName(block.Name) {
			state.webSearchBlocks++
		}
		return nil, false
	case `"message_start"`:
		var message map[string]json.RawMessage
		if json.Unmarshal(event["message"], &message) != nil || message == nil {
			return nil, false
		}
		state.startUsage = decodeJSONObject(message["usage"])
		message["model"], _ = json.Marshal(model)
		event["message"], _ = json.Marshal(message)
	case `"message_delta"`:
		usage := decodeJSONObject(event["usage"])
		for key, value := range state.startUsage {
			if _, ok := usage[key]; !ok {
				usage[key] = value
			}
		}
		completeWebSearchCount(usage, state.webSearchBlocks)
		completeCacheCreationTotal(usage)
		if len(usage) == 0 {
			return nil, false
		}
		event["usage"], _ = json.Marshal(usage)
	default:
		return nil, false
	}
	rewritten, err := json.Marshal(event)
	return rewritten, err == nil
}

// decodeJSONObject decodes raw as a JSON object, field by field. Anything else (absent,
// null, malformed) gives an empty, writable map.
func decodeJSONObject(raw json.RawMessage) map[string]json.RawMessage {
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil || object == nil {
		return map[string]json.RawMessage{}
	}
	return object
}

// completeWebSearchCount sets server_tool_use.web_search_requests from the web_search
// blocks seen when usage reports none: a provider that ran the searches still bills
// them. The object's other counters (web_fetch_requests, ...) are kept as sent.
func completeWebSearchCount(usage map[string]json.RawMessage, blocks int) {
	if blocks == 0 {
		return
	}
	serverToolUse := decodeJSONObject(usage["server_tool_use"])
	var reported int
	if json.Unmarshal(serverToolUse["web_search_requests"], &reported) == nil && reported > 0 {
		return
	}
	serverToolUse["web_search_requests"], _ = json.Marshal(blocks)
	usage["server_tool_use"], _ = json.Marshal(serverToolUse)
}

// completeCacheCreationTotal fills cache_creation_input_tokens from cache_creation's TTL
// split when a provider reports only the split (see NormalizeCacheCreationUsage): the
// rate limiter counts the flat fields only.
func completeCacheCreationTotal(usage map[string]json.RawMessage) {
	var total int
	if json.Unmarshal(usage["cache_creation_input_tokens"], &total) == nil && total > 0 {
		return
	}
	var details CacheCreationDetails
	if json.Unmarshal(usage["cache_creation"], &details) != nil {
		return
	}
	if total, _, _ = NormalizeCacheCreationUsage(0, &details); total > 0 {
		usage["cache_creation_input_tokens"], _ = json.Marshal(total)
	}
}

func (s *messagesStreamState) startMessage() error {
	if s.started {
		return nil
	}
	s.started = true
	return writeMessagesSSE(s.writer, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":            s.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []interface{}{},
			"model":         s.model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]interface{}{
				"input_tokens":                0,
				"output_tokens":               0,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens":     0,
			},
		},
	})
}

func (s *messagesStreamState) writeDelta(blockType string, toolIndex int, id, name, delta string) error {
	if s.blockType != blockType || blockType == "tool_use" && s.toolIndex != toolIndex {
		if err := s.stopBlock(); err != nil {
			return err
		}
		s.blockType = blockType
		s.toolIndex = toolIndex
		contentBlock := map[string]interface{}{"type": blockType}
		switch blockType {
		case "text":
			contentBlock["text"] = ""
		case "thinking":
			contentBlock["thinking"] = ""
			contentBlock["signature"] = ""
		case "tool_use":
			contentBlock["id"] = id
			contentBlock["name"] = name
			contentBlock["input"] = map[string]interface{}{}
		}
		if err := writeMessagesSSE(s.writer, "content_block_start", map[string]interface{}{
			"type": "content_block_start", "index": s.blockIndex, "content_block": contentBlock,
		}); err != nil {
			return err
		}
	}
	deltaBody := map[string]interface{}{"type": blockType + "_delta"}
	switch blockType {
	case "text":
		deltaBody["type"] = "text_delta"
		deltaBody["text"] = delta
	case "thinking":
		deltaBody["type"] = "thinking_delta"
		deltaBody["thinking"] = delta
	case "tool_use":
		deltaBody["type"] = "input_json_delta"
		deltaBody["partial_json"] = delta
	}
	if delta == "" {
		return nil
	}
	if blockType == "thinking" {
		s.thinkingStreamed = true
	}
	return writeMessagesSSE(s.writer, "content_block_delta", map[string]interface{}{
		"type": "content_block_delta", "index": s.blockIndex, "delta": deltaBody,
	})
}

// writeThinkingBlock finishes a thinking block from a Chat thinking_blocks entry: it
// signs and closes the block the reasoning deltas streamed, or emits a redacted_thinking
// block. When no reasoning streamed into it (display "omitted", or a producer that sends
// the text only in thinking_blocks), the block is opened here with the entry's own text,
// which is what the signature covers. Entries without a signature or data only repeat
// streamed text and are skipped.
func (s *messagesStreamState) writeThinkingBlock(block map[string]interface{}) error {
	switch signature, data := stringValue(block["signature"]), stringValue(block["data"]); {
	case block["type"] == "thinking" && signature != "":
		text := ""
		if s.blockType != "thinking" || !s.thinkingStreamed {
			text = stringValue(block["thinking"])
		}
		if err := s.writeDelta("thinking", -1, "", "", text); err != nil {
			return err
		}
		if err := writeMessagesSSE(s.writer, "content_block_delta", map[string]interface{}{
			"type": "content_block_delta", "index": s.blockIndex,
			"delta": map[string]interface{}{"type": "signature_delta", "signature": signature},
		}); err != nil {
			return err
		}
	case block["type"] == "redacted_thinking" && data != "":
		if err := s.stopBlock(); err != nil {
			return err
		}
		s.blockType = "redacted_thinking"
		if err := writeMessagesSSE(s.writer, "content_block_start", map[string]interface{}{
			"type": "content_block_start", "index": s.blockIndex,
			"content_block": map[string]interface{}{"type": "redacted_thinking", "data": data},
		}); err != nil {
			return err
		}
	default:
		return nil
	}
	return s.stopBlock()
}

func (s *messagesStreamState) stopBlock() error {
	if s.blockType == "" {
		return nil
	}
	if err := writeMessagesSSE(s.writer, "content_block_stop", map[string]interface{}{
		"type": "content_block_stop", "index": s.blockIndex,
	}); err != nil {
		return err
	}
	s.blockIndex++
	s.blockType = ""
	s.thinkingStreamed = false
	return nil
}

func (s *messagesStreamState) finishStream() error {
	if err := s.startMessage(); err != nil {
		return err
	}
	if s.blockType == "" && s.blockIndex == 0 {
		if err := s.writeDelta("text", -1, "", "", ""); err != nil {
			return err
		}
	}
	if err := s.stopBlock(); err != nil {
		return err
	}
	reason := s.finish
	if reason == "" {
		reason = "end_turn"
	}
	usage := chatUsageToMessages(s.usage, s.cacheWriteTTLFallback5mTokens, s.cacheWriteTTLFallback1hTokens)
	if err := writeMessagesSSE(s.writer, "message_delta", map[string]interface{}{
		"type":  "message_delta",
		"delta": map[string]interface{}{"stop_reason": reason, "stop_sequence": nil},
		"usage": usage,
	}); err != nil {
		return err
	}
	return writeMessagesSSE(s.writer, "message_stop", map[string]interface{}{"type": "message_stop"})
}

func writeMessagesSSE(writer io.Writer, event string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", event, data)
	return err
}

func stringValue(value interface{}) string {
	text, _ := value.(string)
	return text
}
