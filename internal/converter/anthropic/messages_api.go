package anthropic

import (
	"bufio"
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
		return nil, MessagesAdapterMetadata{}, fmt.Errorf("failed to parse Messages request: %w", err)
	}

	model, _ := request["model"].(string)
	if model == "" {
		return nil, MessagesAdapterMetadata{}, fmt.Errorf("model is required")
	}
	maxTokens, ok := request["max_tokens"].(float64)
	if !ok || maxTokens <= 0 {
		return nil, MessagesAdapterMetadata{}, fmt.Errorf("max_tokens is required")
	}
	rawMessages, ok := request["messages"].([]interface{})
	if !ok || len(rawMessages) == 0 {
		return nil, MessagesAdapterMetadata{}, fmt.Errorf("messages is required")
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
	if thinking, ok := request["thinking"].(map[string]interface{}); ok {
		if isClaudeModel(model) {
			chat["thinking"] = thinking
		} else if effort := thinkingToReasoningEffort(thinking); effort != "" {
			if outputConfig, ok := request["output_config"].(map[string]interface{}); ok && thinking["type"] == "adaptive" {
				if configured, ok := outputConfig["effort"].(string); ok && configured != "" {
					effort = configured
				}
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
			if imageURL := sourceToImageURL(block["source"]); imageURL != "" {
				part := map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL}}
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

func sourceToImageURL(raw interface{}) string {
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
			if imageURL := sourceToImageURL(block["source"]); imageURL != "" {
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

func ChatToMessages(body []byte, metadata MessagesAdapterMetadata) ([]byte, error) {
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
	for _, block := range choice.Message.ThinkingBlocks {
		switch block["type"] {
		case "thinking", "redacted_thinking":
			content = append(content, block)
		}
	}
	if len(choice.Message.ThinkingBlocks) == 0 && choice.Message.ReasoningContent != "" {
		content = append(content, map[string]interface{}{
			"type": "thinking", "thinking": choice.Message.ReasoningContent, "signature": nil,
		})
	}
	if choice.Message.Content != nil {
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
		"usage":         chatUsageToMessages(response.Usage),
	}
	return json.Marshal(converted)
}

func chatUsageToMessages(usage *openai.OpenAIUsage) *AnthropicUsage {
	if usage == nil {
		return &AnthropicUsage{}
	}
	cacheRead := 0
	cacheCreation := 0
	if usage.PromptTokensDetails != nil {
		cacheRead = usage.PromptTokensDetails.CachedTokens
		cacheCreation = usage.PromptTokensDetails.CacheCreationTokens
		if cacheCreation == 0 {
			cacheCreation = usage.PromptTokensDetails.CacheWriteTokens
		}
	}
	return &AnthropicUsage{
		InputTokens:              max(usage.PromptTokens-cacheRead-cacheCreation, 0),
		OutputTokens:             usage.CompletionTokens,
		CacheReadInputTokens:     cacheRead,
		CacheCreationInputTokens: cacheCreation,
	}
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
	writer     io.Writer
	model      string
	metadata   MessagesAdapterMetadata
	messageID  string
	blockIndex int
	blockType  string
	toolIndex  int
	finish     string
	usage      *openai.OpenAIUsage
	started    bool
}

func TransformChatStreamToMessages(reader io.Reader, writer io.Writer, model string, metadata MessagesAdapterMetadata) error {
	state := messagesStreamState{writer: writer, model: model, metadata: metadata, messageID: "msg_" + uuid.NewString()}
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
	return writeMessagesSSE(s.writer, "content_block_delta", map[string]interface{}{
		"type": "content_block_delta", "index": s.blockIndex, "delta": deltaBody,
	})
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
	return nil
}

func (s *messagesStreamState) finishStream() error {
	if err := s.startMessage(); err != nil {
		return err
	}
	if s.blockType == "" {
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
	usage := chatUsageToMessages(s.usage)
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
